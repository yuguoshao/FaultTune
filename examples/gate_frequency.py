"""The worked example: gate frequencies of an 8x8 lattice, with the XEB algorithm subgraphs.

The NumPy `GateFrequencyModel` below stands in for the physics (relaxation defects, crosstalk,
detuning). It is the Python twin of the toy model of the Rust example (`gate_frequency.rs`), with
the same physics but its own random device. The script walks through everything FaultTune offers
around a calibration:

1. **Calibrate** the whole goal with the XEB setting of the paper (d_P = 1, d_R = 2, d_T = 2),
   printing progress through `on_step`.
2. **Compare** the calibrated database with a naive baseline and with other settings:
   d_P = 0 / 1 / 2 and the three traversal heuristics.
3. **Re-calibrate locally** after a qubit drifted: only the region around it is forgotten and
   re-learned; the script shows which entries changed.
4. **Save and load** the database as JSON.
5. **Checkpoint and resume**: a journal records every committed step, a run is cancelled
   half-way (`Snake.cancel` from `on_step`), and a fresh database is rebuilt from the journal
   and continues.
6. **Stitch** the four quadrants of the lattice: calibrate them in parallel, then calibrate the
   gap between them.

Run it with `python examples/gate_frequency.py` (after `maturin develop --release`, or
`pip install faulttune`). It takes a few seconds. The output is the same on every run and on every
number of threads, because the calibration is deterministic; wall-clock times are therefore not
printed (see `Report.wall_time` if you want them). The script ends with self-checks, so a
broken installation fails loudly.
"""

from __future__ import annotations

import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import faulttune as ft

# The size of the lattice: 64 qubits, 112 couplers, 98 parasitic diagonals.
ROWS = 8
COLS = 8


# ---------------------------------------------------------------------------------------------
# The error model
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StepPlan:
    """Everything one step needs to evaluate many candidates, resolved once in `prepare`.

    Looking up neighbours, couplings and endpoint frequencies is cheap but not free; doing it
    once per step instead of once per `error` call is the typical use of `prepare` and
    `step.data`.
    """

    #: Per parameter: defect centres, squared half widths and strengths (arrays of one size).
    defects: tuple[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]], ...]
    #: Per two-qubit gate (a column): the sources of its endpoint qubits' frequencies, each
    #: either ``("fixed", value)`` (a constraint) or ``("column", j)`` (another parameter).
    detuning: tuple[tuple[int, tuple[tuple[str, float], ...]], ...]
    #: ``pairs[j, l]`` is the crosstalk coupling of parameters j < l that interact, else 0.
    pairs: NDArray[np.float64]
    #: ``with_constraints[j, c]`` is the coupling of parameter j and constraint c, else 0.
    with_constraints: NDArray[np.float64]
    #: The fixed frequencies of the constraints, in the order of the columns above.
    constraint_values: NDArray[np.float64]


class GateFrequencyModel(ft.ErrorModel):
    """A toy model of gate frequencies on a qubit lattice.

    Nodes are single-qubit gate frequencies, engineered edges are two-qubit gate frequencies.
    Each element chooses among 100 frequencies in [5, 7] GHz. The error of an element is a
    smooth sum of

    * **relaxation**: every qubit has three random defects, each a Lorentzian
      ``s w^2 / ((f - c)^2 + w^2)`` with centre ``c`` in [5, 7] GHz, half width ``w`` of
      5 to 20 MHz and strength ``s`` in [0.5, 2]. A single-qubit gate pays for the defects of its
      qubit, a two-qubit gate for those of both of its qubits, at the gate frequency;
    * **detuning**: a two-qubit gate prefers a frequency near the mean of its two qubits'
      frequencies (``(f - f_ref)^2``, weight 1 per GHz^2), where known;
    * **crosstalk**: two elements within distance 2 of each other (in the sense of the paper) pay
      ``g w_x^2 / ((f_a - f_b)^2 + w_x^2)`` with ``w_x`` = 40 MHz and a random coupling ``g`` in
      [0.2, 1].

    The error of a candidate is the own terms of the step's parameters, plus the crosstalk of
    every interacting pair among the parameters and of every interacting (parameter, constraint)
    pair. It is a pure function of the step that only reads the model's data, so it is safe with
    any cache policy, and calls from the several subgoals of a run may overlap.
    """

    NUM_OPTIONS = 100
    FREQUENCY_MIN = 5.0
    FREQUENCY_MAX = 7.0
    DEFECTS_PER_QUBIT = 3
    INTERACTION_DISTANCE = 2
    CROSSTALK_WIDTH = 0.04
    DETUNING_WEIGHT = 1.0
    BASELINE_FREQUENCY = 6.0

    def __init__(self, graph: ft.ProcessorGraph, seed: int = 53) -> None:
        self.graph = graph
        elements = graph.all_elements()
        n = len(elements)
        rng = np.random.default_rng(seed)

        # Random defects of every qubit, kept per element: a node owns its own, an edge owns
        # those of both endpoints.
        qubit_defects: dict[int, tuple[Any, Any, Any]] = {}
        for node in graph.nodes():
            centre = rng.uniform(self.FREQUENCY_MIN, self.FREQUENCY_MAX, self.DEFECTS_PER_QUBIT)
            width = rng.uniform(0.005, 0.020, self.DEFECTS_PER_QUBIT)
            strength = rng.uniform(0.5, 2.0, self.DEFECTS_PER_QUBIT)
            qubit_defects[node.index] = (centre, width * width, strength)
        self.defects: list[
            tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        ] = []
        for element in elements:
            owners = [element] if element.is_node else list(element.endpoints or ())
            parts = [qubit_defects[o.index] for o in owners]
            self.defects.append(
                (
                    np.concatenate([p[0] for p in parts]),
                    np.concatenate([p[1] for p in parts]),
                    np.concatenate([p[2] for p in parts]),
                )
            )

        # Symmetric random couplings of all pairs, and which pairs interact at all.
        coupling = np.triu(rng.uniform(0.2, 1.0, (n, n)), k=1)
        self.coupling = coupling + coupling.T
        self.interacts = self._interaction_matrix(graph, self.INTERACTION_DISTANCE)

        self.options_ = np.linspace(self.FREQUENCY_MIN, self.FREQUENCY_MAX, self.NUM_OPTIONS)

    @staticmethod
    def _interaction_matrix(graph: ft.ProcessorGraph, distance: int) -> NDArray[np.bool_]:
        """Which pairs of elements are within `distance` of each other (paper sense).

        Distances are counted on the incidence graph (each node linked to its incident edges):
        an element is within `d` of another if the other is within incidence distance
        ``2d`` (for an edge) or ``2d - 1`` (for a node) of it. The relation is symmetric.
        """
        elements = graph.all_elements()
        neighbours: list[list[int]] = [[] for _ in elements]
        for edge in graph.edges():
            for node in edge.endpoints or ():
                neighbours[edge.index].append(node.index)
                neighbours[node.index].append(edge.index)
        interacts = np.zeros((len(elements), len(elements)), dtype=bool)
        for element in elements:
            radius = 2 * distance if element.is_edge else max(2 * distance - 1, 0)
            seen = {element.index}
            frontier = [element.index]
            for _ in range(radius):
                frontier = [j for i in frontier for j in neighbours[i] if j not in seen]
                seen.update(frontier)
            interacts[element.index, sorted(seen)] = True
            interacts[element.index, element.index] = False
        assert (interacts == interacts.T).all(), "interaction must be symmetric"
        return interacts

    # -- the ErrorModel interface -------------------------------------------------------------

    def options(self, element: ft.Element) -> NDArray[np.float64]:
        """The 100 candidate frequencies of a node or an engineered edge."""
        if element.kind not in (ft.ElementKind.NODE, ft.ElementKind.ENGINEERED_EDGE):
            raise ValueError(f"{element.label} has no gate frequency; exclude it from the goal")
        return self.options_

    def prepare(self, step: ft.Step) -> StepPlan:
        """Resolves the step once; the result is `step.data` in every `error` call."""
        return self.plan(step.parameters, step.constraints)

    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        """The errors of all candidates at once: shape (n_candidates,)."""
        return self.evaluate(step.data, candidates)

    # -- the physics --------------------------------------------------------------------------

    def plan(self, parameters: list[ft.Element], constraints: dict[ft.Element, float]) -> StepPlan:
        """Resolves which defects, couplings and endpoint frequencies a step involves."""
        ids = [p.index for p in parameters]
        column = {i: j for j, i in enumerate(ids)}
        fixed = {c.index: float(f) for c, f in constraints.items()}

        detuning = []
        for j, parameter in enumerate(parameters):
            if not parameter.is_edge:
                continue
            sources: list[tuple[str, float]] = []
            for end in parameter.endpoints or ():
                if end.index in fixed:
                    sources.append(("fixed", fixed[end.index]))
                elif end.index in column:
                    sources.append(("column", float(column[end.index])))
            detuning.append((j, tuple(sources)))

        constraint_ids = list(fixed)
        pairs = np.triu(self.coupling[np.ix_(ids, ids)] * self.interacts[np.ix_(ids, ids)], k=1)
        with_constraints = (
            self.coupling[np.ix_(ids, constraint_ids)] * self.interacts[np.ix_(ids, constraint_ids)]
        )
        return StepPlan(
            defects=tuple(self.defects[i] for i in ids),
            detuning=tuple(detuning),
            pairs=pairs,
            with_constraints=with_constraints,
            constraint_values=np.array([fixed[c] for c in constraint_ids], dtype=np.float64),
        )

    def evaluate(self, plan: StepPlan, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        """The error of every row of `candidates` (float64, shape (n_candidates, n_parameters))."""
        total = np.zeros(candidates.shape[0])
        w2 = self.CROSSTALK_WIDTH**2
        for j, (centre, width_sq, strength) in enumerate(plan.defects):
            d = candidates[:, j, None] - centre
            total += (strength * width_sq / (d * d + width_sq)).sum(axis=1)
        for j, sources in plan.detuning:
            known = [f if kind == "fixed" else candidates[:, int(f)] for kind, f in sources]
            if known:
                reference = 0.5 * (known[0] + known[1]) if len(known) == 2 else known[0]
                total += self.DETUNING_WEIGHT * (candidates[:, j] - reference) ** 2
        if plan.pairs.any():
            d = candidates[:, :, None] - candidates[:, None, :]
            total += (plan.pairs * w2 / (d * d + w2)).sum(axis=(1, 2))
        if plan.with_constraints.size:
            d = candidates[:, :, None] - plan.constraint_values
            total += (plan.with_constraints * w2 / (d * d + w2)).sum(axis=(1, 2))
        return total

    # -- measuring a result -------------------------------------------------------------------

    def assignment_error(self, values: dict[ft.Element, float]) -> float:
        """The full-model error of an assignment: every own term and every interacting pair."""
        parameters = list(values)
        plan = self.plan(parameters, {})
        row = np.array([[values[p] for p in parameters]])
        return float(self.evaluate(plan, row)[0])

    def mean_error(self, db: ft.ParameterDatabase) -> float:
        """Mean error per calibrated gate."""
        values = db.to_dict()
        return self.assignment_error(values) / len(values) if values else 0.0

    def baseline_mean_error(self, db: ft.ParameterDatabase) -> float:
        """The mean error if every gate of `db` ran at 6 GHz, the middle of the band."""
        values = dict.fromkeys(db.to_dict(), self.BASELINE_FREQUENCY)
        return self.assignment_error(values) / len(values) if values else 0.0


class DriftedModel(ft.ErrorModel):
    """The model plus one new defect (a two-level system) near a chosen qubit.

    The relaxation error of that qubit, and of the gates it takes part in, gets one more
    Lorentzian bump. It shows how a model is wrapped to describe a device that changed: the
    wrapper delegates to the base model and adds the new term.
    """

    WIDTH = 0.015  #: half width of the new defect (GHz)
    STRENGTH = 3.0  #: height of the bump

    def __init__(self, base: GateFrequencyModel, qubit: ft.Element, centre: float) -> None:
        self.base = base
        self.qubit = qubit
        self.centre = centre

    def options(self, element: ft.Element) -> NDArray[np.float64]:
        return self.base.options(element)

    def prepare(self, step: ft.Step) -> StepPlan:
        return self.base.prepare(step)

    def penalty(self, element: ft.Element, frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
        """The extra error of `element` at `frequencies` (zero unless it touches the qubit)."""
        touches = element == self.qubit or self.qubit in (element.endpoints or ())
        if not touches:
            return np.zeros_like(frequencies)
        d = frequencies - self.centre
        return self.STRENGTH * self.WIDTH**2 / (d * d + self.WIDTH**2)

    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        total = self.base.error(step, candidates)
        for j, parameter in enumerate(step.parameters):
            total = total + self.penalty(parameter, candidates[:, j])
        return total

    def total_penalty(self, db: ft.ParameterDatabase) -> float:
        """The summed extra error of every calibrated element of `db`."""
        return float(sum(self.penalty(e, np.array([v]))[0] for e, v in db.to_dict().items()))


# ---------------------------------------------------------------------------------------------
# The device: graph, goal, algorithm subgraphs and model
# ---------------------------------------------------------------------------------------------


class Device:
    """Everything that describes the device and the calibration problem."""

    def __init__(self) -> None:
        self.graph = ft.ProcessorGraph.square_lattice(ROWS, COLS)
        #: Nodes and engineered edges: the elements that have a frequency.
        self.goal = self.graph.goal(exclude=[ft.ElementKind.PARASITIC_EDGE])
        #: The five XEB subgraphs (Fig. 3 of the paper).
        self.xeb = ft.AlgorithmSubgraphs.xeb_square_lattice(self.graph, ROWS, COLS)
        self.model = GateFrequencyModel(self.graph, seed=53)

    def snake(self, parameter_distance: int = 1, **settings: Any) -> ft.Snake:
        """A Snake for this device: the XEB setting with a chosen d_P (other keywords as given)."""
        settings.setdefault("constraint_distance", 2)  # d_R
        settings.setdefault("traversal_distance", 2)  # d_T
        return ft.Snake(
            self.graph,
            self.goal,
            parameter_distance=parameter_distance,  # d_P
            algorithm=self.xeb,
            **settings,
        )


def section(title: str) -> None:
    """Prints a section heading."""
    print(f"\n== {title} ==")


# ---------------------------------------------------------------------------------------------
# 1. Calibrate, with progress printed by on_step
# ---------------------------------------------------------------------------------------------


def calibrate_with_progress(device: Device) -> ft.ParameterDatabase:
    """Calibrates with the default (XEB) setting; `on_step` prints every eighth step."""
    section("1. Calibrate (d_P = 1, d_R = 2, d_T = 2, breadth-first), with progress")

    # on_step is called on the calling thread, in order, for every committed step; this is
    # where a progress bar (tqdm, a notebook widget) would be updated. `record.index` is the
    # ordinal of the step within its subgoal and pass, so it is only an ordering hint here.
    def progress(record: ft.StepRecord) -> None:
        if record.index % 8 == 0:
            print(
                f"    step {record.index:>3}: centre {record.centre.label:<10} "
                f"{len(record.parameters)} parameters, {len(record.constraints):>2} constraints, "
                f"error {record.error:.3f}"
            )

    snake = device.snake()
    db = snake.calibrate(device.model, on_step=progress)

    report = snake.last_report
    assert report is not None
    subgoals = {(s.pass_, s.subgoal) for s in db.steps()}
    print(
        f"Done: {report.steps} steps in {report.passes} pass(es) and {len(subgoals)} subgoal(s), "
        f"{report.evaluations} evaluations; calibrated {len(db)} of {len(device.goal)} elements."
    )
    print("Worst steps (largest error), the natural candidates for re-optimization:")
    for step in report.worst_steps[:3]:
        print(f"  centre {step.centre.label:<10} error {step.error:.3f}")
    return db


# ---------------------------------------------------------------------------------------------
# 2. Compare with a baseline and with other settings
# ---------------------------------------------------------------------------------------------


def compare_settings(device: Device, reference: ft.ParameterDatabase) -> dict[str, float]:
    """Calibrates with several settings and prints a comparison table."""
    section("2. Comparison of settings (mean error per calibrated gate, lower is better)")
    print(f"  {'setting':<34} {'steps':>7} {'evaluations':>12} {'mean error':>11}")
    baseline = device.model.baseline_mean_error(reference)
    print(f"  {'naive: every gate at 6 GHz':<34} {'-':>7} {'-':>12} {baseline:>11.4f}")

    settings = [
        ("d_P = 0, breadth-first", 0, "breadth_first"),
        ("d_P = 1, breadth-first (default)", 1, "breadth_first"),
        ("d_P = 2, breadth-first", 2, "breadth_first"),
        ("d_P = 1, depth-first", 1, "depth_first"),
        ("d_P = 1, random", 1, "random"),
    ]
    means: dict[str, float] = {}
    for name, parameter_distance, heuristic in settings:
        snake = device.snake(parameter_distance, heuristic=heuristic)
        db = snake.calibrate(device.model)
        report = snake.last_report
        assert report is not None
        means[name] = device.model.mean_error(db)
        print(f"  {name:<34} {report.steps:>7} {report.evaluations:>12} {means[name]:>11.4f}")
    worst = max(means.values())
    first, last = means["d_P = 0, breadth-first"], means["d_P = 2, breadth-first"]
    print(
        "A larger d_P optimizes more parameters together: fewer, bigger steps and a lower "
        "error, at the price of more evaluations.\n"
        f"  Here d_P = 2 has {100 * (1 - last / first):.0f}% lower error than d_P = 0, and every "
        f"setting beats the naive baseline by a factor of at least {baseline / worst:.0f}."
    )
    means["baseline"] = baseline
    return means


# ---------------------------------------------------------------------------------------------
# 3. Local re-calibration of a drifted qubit
# ---------------------------------------------------------------------------------------------


def recalibrate_after_drift(device: Device, db: ft.ParameterDatabase) -> None:
    """Qubit q3_4 develops a defect right at its operating frequency.

    `recalibrate` forgets the qubit and its couplers (radius `expiry_radius = 1`) and re-learns
    them subject to all the surrounding gates, which stay untouched.
    """
    section("3. Local re-calibration of a drifted qubit")
    db = db.copy()
    qubit = device.graph.node(3, 4)
    stale_frequency = db[qubit]
    drifted = DriftedModel(device.model, qubit, centre=stale_frequency)
    print(
        f"{qubit.label} runs at {stale_frequency:.3f} GHz, and a new defect appears exactly there."
    )

    snake = device.snake()
    n = len(db)
    before = db.copy()
    stale_penalty = drifted.total_penalty(db)
    history_before = len(db.history)

    # Call it with the database, the expired element(s) and the (now different) model.
    report = snake.recalibrate(db, qubit, drifted)

    # What was forgotten: the qubit and its engineered couplers (expiry_radius = 1).
    radius = snake.config["expiry_radius"]
    assert radius == 1
    forgotten = {qubit} | {e for e in device.goal if e.is_edge and qubit in (e.endpoints or ())}
    old, new = before.to_dict(), db.to_dict()
    changed = [e for e in device.graph.all_elements() if old.get(e) != new.get(e)]
    outside = [e for e in changed if e not in forgotten]
    assert not outside, f"entries outside the neighbourhood changed: {outside}"
    assert device.goal <= db.calibrated, "the goal is not fully calibrated any more"

    print(
        f"Forgot {len(forgotten)} elements (radius {radius}) and re-learned them in "
        f"{report.steps} step(s), {report.evaluations} evaluations."
    )
    print("Entries that changed value:")
    for e in changed:
        print(f"  {e.label:<12} {old.get(e, float('nan')):.3f} GHz -> {new[e]:.3f} GHz")
    print(
        f"{n - len(changed)} of the {n} calibrated entries are untouched; the history grew from "
        f"{history_before} to {len(db.history)} records (one Invalidation, then the new steps)."
    )
    fresh_penalty = drifted.total_penalty(db)
    print(
        f"Extra error caused by the defect, summed over all gates: {stale_penalty:.3f} with the "
        f"stale calibration, {fresh_penalty:.3f} after re-calibration "
        f"(mean error per gate {(device.model.mean_error(before) * n + stale_penalty) / n:.4f} -> "
        f"{(device.model.mean_error(db) * n + fresh_penalty) / n:.4f})."
    )
    assert qubit in changed, "the drifted qubit must move away from its defect"
    assert fresh_penalty < stale_penalty
    assert any(isinstance(r, ft.Invalidation) for r in db.history[history_before:])


# ---------------------------------------------------------------------------------------------
# 4. Save to and load from JSON
# ---------------------------------------------------------------------------------------------


def save_and_load(device: Device, db: ft.ParameterDatabase, scratch: Path) -> None:
    """Saves the calibration, loads it back and shows that loading it onto another device fails."""
    section("4. Save and load (JSON)")
    path = scratch / "calibration.json"
    db.save(path)
    text = path.read_text()
    print(f"Saved {len(db)} entries and {len(db.history)} history records. The start of the file:")
    for line in text.splitlines()[:12]:
        print(f"  {line}")
    print("  ...")

    loaded = ft.ParameterDatabase.load(path, device.graph)
    same = loaded.same_outcome(db)
    print(f"Loaded it back: same values and history (bit for bit): {same}")
    assert same, "the loaded database differs from the saved one"

    # The file records a fingerprint of the graph: another device is refused.
    other = ft.ProcessorGraph.square_lattice(4, 4)
    try:
        ft.ParameterDatabase.load(path, other)
    except ft.GraphMismatchError as err:
        print(f"Loading it onto a 4x4 lattice is refused: {err}")
    else:
        raise AssertionError("a calibration was loaded onto the wrong graph")


# ---------------------------------------------------------------------------------------------
# 5. Checkpoint journal and resume
# ---------------------------------------------------------------------------------------------


class HoldingModel(ft.ErrorModel):
    """The model, but at one step its `error` announces itself and waits to be released.

    A real calibration measures hardware: each step takes seconds, so `on_step` (which runs on
    the main thread as soon as a step is committed) is always ahead of the engine. The toy model
    is so fast that the engine could finish the whole run before the callback gets to cancel
    it, and a cancelled run would stop after a different number of steps every time. This
    wrapper makes the example deterministic: at step `hold_step` it sets `reached` and then
    waits for `release`, so the callback can cancel while that step is known to be running.
    """

    def __init__(self, base: GateFrequencyModel, hold_step: int) -> None:
        self.base = base
        self.hold_step = hold_step
        self.reached = threading.Event()
        self.release = threading.Event()

    def options(self, element: ft.Element) -> NDArray[np.float64]:
        return self.base.options(element)

    def prepare(self, step: ft.Step) -> StepPlan:
        return self.base.prepare(step)

    def error(self, step: ft.Step, candidates: NDArray[np.float64]) -> NDArray[np.float64]:
        if step.index == self.hold_step:
            self.reached.set()
            if not self.release.wait(60):
                raise TimeoutError("on_step never released the step")
        return self.base.error(step, candidates)


def checkpoint_and_resume(device: Device, scratch: Path) -> None:
    """Cancels a run after 25 steps (standing in for a crash or Ctrl-C), then resumes it."""
    section("5. Checkpoint journal and resume after an interruption")
    stop_after = 25  # the run is interrupted during step 25: 26 steps are committed
    journal = scratch / "run.journal"
    snake = device.snake()

    # First run: `checkpoint=` appends every committed step to the journal as one JSON line,
    # flushed immediately. on_step cancels the run while step number `stop_after` is running
    # (the model holds that step until the callback has cancelled). A step in progress always
    # finishes and is committed; no further step starts.
    model = HoldingModel(device.model, hold_step=stop_after)

    def stop_early(record: ft.StepRecord) -> None:
        if record.index + 1 == stop_after:
            model.reached.wait(60)
            snake.cancel()  # Ctrl-C does the same
            model.release.set()

    interrupted = ft.ParameterDatabase(device.graph)
    try:
        snake.calibrate(model, database=interrupted, on_step=stop_early, checkpoint=journal)
    except ft.CancelledError as err:
        # The exception carries the report; the database passed in holds every committed step.
        assert err.partial is interrupted
        print(
            f"Run cancelled after {interrupted.step_count} steps; the database kept all of them "
            f"({len(interrupted)} of {len(device.goal)} elements)."
        )
    else:
        raise AssertionError("the run finished before it was cancelled")
    assert interrupted.step_count == stop_after + 1

    # A crash in the middle of writing a line leaves a torn last line. Simulate one.
    with journal.open("a") as f:
        f.write('{"type":"step","pass":0,"subgoal":0,"centre":"q4')

    # "New process": nothing is in memory but the journal. read_journal gives the step records
    # (with option indices, not values); replay_journal applies them to a database. The model
    # is only asked for its options, to turn the indices back into values.
    steps = ft.read_journal(journal, device.graph)
    restored = ft.ParameterDatabase(device.graph)
    applied = restored.replay_journal(journal, device.model)
    identical = restored.same_outcome(interrupted)
    print(
        f"Journal read (the torn last line is ignored): {len(steps)} steps, {applied} replayed; "
        f"the replayed database equals the interrupted one: {identical}"
    )
    assert identical, "replaying the journal did not reproduce the interrupted database"
    assert len(steps) == applied == interrupted.step_count

    # Resume: the same call as a fresh calibration, on the partly filled database. The journal
    # is appended to (and the torn line repaired), so it stays complete.
    resumed_snake = device.snake()
    resumed = resumed_snake.calibrate(device.model, database=restored, checkpoint=journal)
    assert resumed is restored and device.goal <= restored.calibrated
    resumed_report = resumed_snake.last_report
    assert resumed_report is not None
    full = device.snake()
    full.calibrate(device.model)
    full_report = full.last_report
    assert full_report is not None
    print(
        f"Resumed: {resumed_report.steps} more steps ({full_report.steps} in an uninterrupted "
        f"run), all {len(device.goal)} goal elements calibrated."
    )
    print(
        f"Mean error per gate after resuming: {device.model.mean_error(restored):.4f}. A resumed "
        "run starts a fresh traversal from the remaining\n  elements, so it need not equal the "
        "uninterrupted run, only complete it."
    )

    # The journal alone rebuilds the final state: a third party can do it.
    rebuilt = ft.ParameterDatabase(device.graph)
    rebuilt.replay_journal(journal, device.model)
    assert rebuilt.same_outcome(restored)
    assert len(ft.read_journal(journal, device.graph)) == restored.step_count


# ---------------------------------------------------------------------------------------------
# 6. Stitched parallel calibration
# ---------------------------------------------------------------------------------------------


def quadrants(device: Device) -> list[ft.ElementSet]:
    """The goal elements of each quadrant of the lattice.

    Uses the layout coordinates of the graph (edges sit at the midpoint of their endpoints).
    """
    coords = device.graph.coords
    assert coords is not None, "the graph has no layout"
    mid_x, mid_y = (COLS - 1) / 2, (ROWS - 1) / 2
    groups: list[list[ft.Element]] = [[], [], [], []]
    for element in device.goal:
        x, y = coords[element.index]
        groups[int(x > mid_x) + 2 * int(y > mid_y)].append(element)
    return [device.graph.element_set(g) for g in groups]


def stitched_calibration(device: Device, reference: ft.ParameterDatabase) -> None:
    """Splits the lattice into quadrants, calibrates them in parallel and then the gap."""
    section("6. Stitched calibration: four quadrants in parallel, then the gap")
    regions = quadrants(device)
    sizes = [len(r) for r in regions]

    snake = device.snake()
    plan = snake.stitch_plan(regions)
    print(
        f"Quadrants hold {sizes} elements. Stage one calibrates {len(plan.stage_one)} of them "
        f"(the cores, more than max(d_P, d_R) = 2 apart),\n  stage two the {len(plan.gap)} "
        "elements of the gap."
    )

    stitched = ft.ParameterDatabase(device.graph)
    report = snake.stitch(stitched, plan, device.model)
    # The subgoals of both stages, as the steps of the history show them.
    subgoals = sorted({(s.pass_, s.subgoal) for s in stitched.steps()})
    print(
        f"Stitched calibration: {report.steps} steps in {report.passes} passes and "
        f"{len(subgoals)} subgoals, mean error per gate "
        f"{device.model.mean_error(stitched):.4f} "
        f"(unstitched: {device.model.mean_error(reference):.4f})."
    )
    assert device.goal <= stitched.calibrated

    # The result must not depend on the thread count: compare one thread with four.
    print("Same stitched calibration with different thread counts:")
    databases = []
    for threads in (1, 4):
        db = ft.ParameterDatabase(device.graph)
        device.snake(threads=threads).stitch(db, plan, device.model)
        databases.append(db)
    identical = databases[0].same_outcome(databases[1]) and databases[0].same_outcome(stitched)
    print(f"  identical databases, history included: {identical}")
    assert identical, "the stitched calibration depends on the thread count"
    print(
        "(Python error models take the GIL, so the subgoals of a small NumPy model hardly\n"
        "  overlap. With a model that waits for hardware or does large array operations, they\n"
        "  run side by side.)"
    )


# ---------------------------------------------------------------------------------------------
# Self-checks and main
# ---------------------------------------------------------------------------------------------


def check_batch_contract(device: Device, db: ft.ParameterDatabase) -> None:
    """`error` must give each row its own answer, however the rows are grouped."""
    steps = db.steps()
    rng = np.random.default_rng(0)
    for record in (steps[0], steps[len(steps) // 2], steps[-1]):
        constraints = {c: db[c] for c in record.constraints}
        plan = device.model.plan(record.parameters, constraints)
        rows = rng.choice(device.model.options_, size=(7, len(record.parameters)))
        batch = device.model.evaluate(plan, rows)
        single = np.array([device.model.evaluate(plan, r[None, :])[0] for r in rows])
        assert batch.shape == (7,) and np.allclose(batch, single, rtol=1e-12, atol=1e-12)


def main() -> None:
    device = Device()
    print(
        f"Device: {ROWS}x{COLS} lattice, {len(device.graph.all_elements())} elements, "
        f"{len(device.goal)} of them to calibrate, {len(device.xeb.subgraphs)} algorithm subgraphs"
    )

    calibrated = calibrate_with_progress(device)
    means = compare_settings(device, calibrated)
    recalibrate_after_drift(device, calibrated)
    with tempfile.TemporaryDirectory(prefix="faulttune-example-") as directory:
        scratch = Path(directory)
        save_and_load(device, calibrated, scratch)
        checkpoint_and_resume(device, scratch)
    stitched_calibration(device, calibrated)

    # Self-checks: a broken installation or a regression fails loudly.
    section("Self-checks")
    assert device.goal <= calibrated.calibrated and len(calibrated) == len(device.goal)
    assert all(
        device.model.FREQUENCY_MIN <= f <= device.model.FREQUENCY_MAX
        for f in calibrated.to_dict().values()
    )
    assert all(v < means["baseline"] / 5 for k, v in means.items() if k != "baseline")
    assert means["d_P = 2, breadth-first"] < means["d_P = 0, breadth-first"]
    check_batch_contract(device, calibrated)
    again = device.snake().calibrate(device.model)
    assert again.same_outcome(calibrated), "calibration must be deterministic"
    print("All self-checks passed.")


if __name__ == "__main__":
    main()

"""Saving, loading, checkpoint journals and the database API."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import threading
from typing import Any

import numpy as np
import pytest
from test_native_common import GatedModel, TargetModel, assert_same_database, lattice

import faulttune as ft


@pytest.fixture
def calibrated() -> tuple[ft.ProcessorGraph, ft.ElementSet, ft.ParameterDatabase]:
    graph, goal = lattice(4, 4)
    db = ft.Snake(graph, goal).calibrate(TargetModel(k=9, repel=0.2))
    return graph, goal, db


def test_database_api(calibrated: Any) -> None:
    graph, goal, db = calibrated
    q = graph.node(1, 1)
    value = db[q]
    assert isinstance(value, float) and db["q1_1"] == value and db.get(q) == value
    assert q in db and "q1_1" in db and graph.node(0, 0) in db
    assert graph.edges()[-1] not in db, "parasitic edges are not in the goal"
    assert graph.edges()[-1].kind == ft.ElementKind.PARASITIC_EDGE
    with pytest.raises(KeyError, match="not calibrated"):
        db[graph.edges()[-1]]
    assert db.get(graph.edges()[-1]) is None and db.get(graph.edges()[-1], "x") == "x"
    assert 5 not in db and "nope" not in db and None not in db
    with pytest.raises(KeyError):
        db["nope"]
    with pytest.raises(TypeError, match="element"):
        db[3.5]
    with pytest.raises(ValueError, match="different graph"):
        db[ft.ProcessorGraph.square_lattice(4, 4).node(0, 0)]
    assert len(db) == len(goal) == len(list(db))
    assert list(db) == goal.to_list()
    assert db.calibrated == goal and db.graph is graph
    assert db.fingerprint == graph.fingerprint
    assert db.to_dict(labels=True) == {e.label: v for e, v in db.to_dict().items()}
    assert list(db.to_dict()) == goal.to_list()
    assert db.step_of(q) >= 0 and 0 <= db.option_index(q) < 9
    with pytest.raises(KeyError):
        db.option_index(graph.edges()[-1])
    with pytest.raises(KeyError):
        db.step_of(graph.edges()[-1])
    assert "ParameterDatabase" in repr(db)
    assert db.step_count == len(db.steps()) == len(db.history)


def test_new_databases_and_copies(calibrated: Any) -> None:
    graph, goal, db = calibrated
    empty = ft.ParameterDatabase(graph)
    assert len(empty) == 0 and empty.fingerprint == graph.fingerprint
    assert not empty.calibrated and empty.history == [] and empty.steps() == []
    clone = db.copy()
    assert clone is not db and clone.same_outcome(db) and db.same_outcome(db)
    assert not empty.same_outcome(db)
    with pytest.raises(TypeError):
        db.same_outcome(5)
    with pytest.raises(TypeError, match="ProcessorGraph|graph"):
        ft.ParameterDatabase(5)  # type: ignore[arg-type]


def test_save_and_load(calibrated: Any, tmp_path: pathlib.Path) -> None:
    graph, goal, db = calibrated
    path = tmp_path / "calibration.json"
    db.save(path)
    loaded = ft.ParameterDatabase.load(path, graph)
    assert_same_database(db, loaded)
    assert loaded.graph is graph
    # Values survive bit for bit, and a str path works too.
    db.save(str(tmp_path / "again.json"))
    again = ft.ParameterDatabase.load(str(tmp_path / "again.json"), graph)
    assert all(again[e] == v for e, v in db.to_dict().items())
    text = path.read_text()
    document = json.loads(text)
    assert document["format"] == "faulttune.parameters/1"
    assert document["graph"]["fingerprint"] == graph.fingerprint
    assert set(document["values"]) == {e.label for e in goal}
    assert [t.name for t in tmp_path.iterdir() if t.name.startswith(".")] == [], "no temp files"

    # A database loaded onto another graph object of the same shape works (same fingerprint)...
    twin = ft.ProcessorGraph.from_json(graph.to_json())
    assert ft.ParameterDatabase.load(path, twin).to_dict(labels=True) == db.to_dict(labels=True)
    # ... but the wrong device fails loudly.
    other = ft.ProcessorGraph.square_lattice(4, 5)
    with pytest.raises(ft.GraphMismatchError):
        ft.ParameterDatabase.load(path, other)
    with pytest.raises(ft.GraphMismatchError):
        ft.Snake(other, other.goal()).calibrate(TargetModel(), database=db)


def test_load_errors(calibrated: Any, tmp_path: pathlib.Path) -> None:
    graph, goal, db = calibrated
    with pytest.raises(FileNotFoundError):
        ft.ParameterDatabase.load(tmp_path / "missing.json", graph)
    garbage = tmp_path / "garbage.json"
    garbage.write_text("not json at all")
    with pytest.raises(ft.FormatError):
        ft.ParameterDatabase.load(garbage, graph)
    wrong = tmp_path / "wrong.json"
    wrong.write_text(json.dumps({"format": "something/1"}))
    with pytest.raises(ft.FormatError):
        ft.ParameterDatabase.load(wrong, graph)
    with pytest.raises(OSError):
        db.save(tmp_path / "no" / "such" / "dir" / "x.json")
    with pytest.raises(TypeError):
        ft.ParameterDatabase.load(5, graph)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="graph"):
        ft.ParameterDatabase.load(garbage, "graph")  # type: ignore[arg-type]
    assert issubclass(ft.FormatError, ValueError) and issubclass(ft.FormatError, ft.FaultTuneError)


def test_json_round_trip(calibrated: Any) -> None:
    graph, goal, db = calibrated
    text = db.to_json()
    assert json.loads(text)["format"] == "faulttune.parameters/1"
    assert_same_database(db, ft.ParameterDatabase.from_json(graph, text))
    with pytest.raises(ft.FormatError):
        ft.ParameterDatabase.from_json(graph, "{")
    with pytest.raises(ft.GraphMismatchError):
        ft.ParameterDatabase.from_json(ft.ProcessorGraph.square_lattice(2, 2), text)


def test_history_and_invalidations(calibrated: Any) -> None:
    graph, goal, db = calibrated
    snake = ft.Snake(graph, goal)
    snake.recalibrate(db, graph.node(2, 2), TargetModel(k=9, repel=0.2))
    invalidations = [h for h in db.history if isinstance(h, ft.Invalidation)]
    assert len(invalidations) == 1
    removed = invalidations[0].elements
    assert removed == sorted(removed) and graph.node(2, 2) in removed
    assert all(e in goal for e in removed)
    assert "Invalidation" in repr(invalidations[0])
    assert len(db.steps()) == db.step_count < len(db.history) + 1
    # Save and load keep the invalidation record.
    again = ft.ParameterDatabase.from_json(graph, db.to_json())
    assert_same_database(db, again)


def test_checkpoint_journal_and_replay(tmp_path: pathlib.Path) -> None:
    graph, goal = lattice(4, 4)
    journal = tmp_path / "run.journal"
    model = TargetModel(k=9, repel=0.2)
    db = ft.Snake(graph, goal).calibrate(model, checkpoint=journal)

    entries = [json.loads(line) for line in journal.read_text().splitlines()]
    assert {e["type"] for e in entries} <= {"step", "invalidated"}
    steps = [e for e in entries if e["type"] == "step"]
    assert len(steps) == db.step_count
    first = steps[0]
    assert "parameters" in first and "options" in first

    records = ft.read_journal(journal, graph)
    assert len(records) == db.step_count
    assert [r.centre for r in records] == [s.centre for s in db.steps()]
    assert [r.option_indices for r in records] == [s.option_indices for s in db.steps()]
    assert [r.index for r in records] == list(range(len(records)))

    # Replaying onto an empty database reproduces the run.
    replayed = ft.ParameterDatabase(graph)
    applied = replayed.replay_journal(journal, TargetModel(k=9))
    assert applied == db.step_count
    assert_same_database(db, replayed)
    # Replaying twice applies nothing the second time.
    assert replayed.replay_journal(str(journal), TargetModel(k=9)) == 0
    assert_same_database(db, replayed)


def test_resume_from_a_journal_after_an_interrupted_run(tmp_path: pathlib.Path) -> None:
    graph, goal = lattice(4, 4)
    journal = tmp_path / "run.journal"
    snake = ft.Snake(graph, goal)
    lost = ft.ParameterDatabase(graph)  # a database that did not survive the "crash"

    gate = threading.Event()

    def crash(record: ft.StepRecord) -> None:
        if record.index == 4:
            snake.cancel()
            gate.set()

    with pytest.raises(ft.CancelledError):
        snake.calibrate(GatedModel(gate, 6, k=9), database=lost, on_step=crash, checkpoint=journal)
    committed = lost.step_count
    assert committed >= 5

    # Simulate a torn final line, as after a real crash while writing.
    with journal.open("a") as f:
        f.write('{"type":"step","pass":0,"subgoal":0,"centre":"q')
    assert len(ft.read_journal(journal, graph)) == committed, "a torn last line is ignored"

    restored = ft.ParameterDatabase(graph)
    assert restored.replay_journal(journal, TargetModel(k=9)) == committed
    assert_same_database(restored, lost)

    # Continue the run, appending to the same (repaired) journal.
    final = snake.calibrate(TargetModel(k=9), database=restored, checkpoint=journal)
    assert final is restored and final.calibrated == goal
    assert len(ft.read_journal(journal, graph)) == restored.step_count
    # A third party can rebuild the final state from the journal alone.
    rebuilt = ft.ParameterDatabase(graph)
    rebuilt.replay_journal(journal, TargetModel(k=9))
    assert_same_database(rebuilt, restored)


def test_replay_journal_errors(tmp_path: pathlib.Path) -> None:
    graph, goal = lattice(3, 3)
    journal = tmp_path / "run.journal"
    ft.Snake(graph, goal).calibrate(TargetModel(k=9), checkpoint=journal)
    db = ft.ParameterDatabase(graph)

    with pytest.raises(FileNotFoundError):
        db.replay_journal(tmp_path / "missing", TargetModel())
    with pytest.raises(TypeError, match="options"):
        db.replay_journal(journal, object())
    # Fewer options than the journal's option indices: a format error.
    with pytest.raises(ft.FormatError, match="out of range"):
        db.replay_journal(journal, TargetModel(k=2))
    # A failing options() surfaces as a ModelError chained to the original exception.

    class Failing(TargetModel):
        def options(self, element: ft.Element) -> Any:
            raise LookupError("no options for this element")

    fresh = ft.ParameterDatabase(graph)
    with pytest.raises(ft.ModelError, match="options") as info:
        fresh.replay_journal(journal, Failing())
    assert isinstance(info.value.__cause__, LookupError)
    assert len(fresh) == 0
    # The journal of another graph is refused.
    other = ft.ProcessorGraph.square_lattice(2, 2)
    with pytest.raises(ft.FormatError, match="unknown element label"):
        ft.read_journal(journal, other)
    with pytest.raises(ft.FormatError):
        ft.ParameterDatabase(other).replay_journal(journal, TargetModel())
    garbled = tmp_path / "garbled"
    garbled.write_text("{}\n{\n")
    with pytest.raises(ft.FormatError):
        ft.read_journal(garbled, graph)


def test_checkpoint_path_errors(tmp_path: pathlib.Path) -> None:
    graph, goal = lattice(3, 3)
    snake = ft.Snake(graph, goal)
    with pytest.raises(OSError):
        snake.calibrate(TargetModel(), checkpoint=tmp_path / "no" / "dir" / "j")
    with pytest.raises(TypeError):
        snake.calibrate(TargetModel(), checkpoint=5)  # type: ignore[arg-type]
    # An unrelated file is not appended to.
    other = tmp_path / "other.txt"
    other.write_text("hello")
    with pytest.raises(ft.FormatError):
        snake.calibrate(TargetModel(), checkpoint=other)
    assert other.read_text() == "hello"
    db = ft.ParameterDatabase(graph)
    snake.calibrate(TargetModel(), database=db)
    assert len(db) == len(goal), "the snake still works after the failed attempts"


def test_values_are_exact_floats(tmp_path: pathlib.Path) -> None:
    graph, goal = lattice(3, 3)

    class Awkward(TargetModel):
        def options(self, element: ft.Element) -> Any:
            return np.array([0.1, 1 / 3, np.pi, 1e-300, 123456789.123456789])

    db = ft.Snake(graph, goal).calibrate(Awkward())
    db.save(tmp_path / "x.json")
    loaded = ft.ParameterDatabase.load(tmp_path / "x.json", graph)
    assert all(loaded[e] == v for e, v in db.to_dict().items())


class Dies(TargetModel):
    """A model that fails on every ``error`` call (the device went away)."""

    def error(self, step: ft.Step, candidates: Any) -> Any:
        raise RuntimeError("device went away")


def test_journal_records_recalibrate_invalidations(tmp_path: pathlib.Path) -> None:
    graph, goal = lattice(5, 5)
    journal = tmp_path / "run.journal"
    snake = ft.Snake(graph, goal)
    db = snake.calibrate(TargetModel(k=9), checkpoint=journal)
    base = tmp_path / "base.json"
    db.save(base)
    q = graph.node(2, 2)
    # A recalibrate that dies after the elements were forgotten: the removal must be journaled.
    with pytest.raises(ft.ModelError):
        snake.recalibrate(db, q, Dies(k=9), checkpoint=journal)
    assert q not in db
    kinds = [json.loads(line)["type"] for line in journal.read_text().splitlines()]
    assert "invalidated" in kinds
    stale = ft.ParameterDatabase.load(base, graph)
    assert q in stale
    stale.replay_journal(journal, TargetModel(k=9))
    assert q not in stale and len(stale) == len(db)
    assert_same_database(stale, db)


def test_replay_reproduces_calibrate_recalibrate_and_stitch(tmp_path: pathlib.Path) -> None:
    graph, goal = lattice(6, 6)
    journal = tmp_path / "run.journal"
    snake = ft.Snake(graph, goal)
    model = TargetModel(k=9, repel=0.2)
    db = snake.calibrate(model, checkpoint=journal)
    mid = tmp_path / "mid.json"
    db.save(mid)
    snake.recalibrate(db, graph.node(2, 2), TargetModel(k=9), checkpoint=journal)
    snake.recalibrate(db, [graph.node(0, 0), graph.node(5, 5)], model, checkpoint=journal)
    assert sum(isinstance(h, ft.Invalidation) for h in db.history) == 2

    # Onto an empty database, onto a save taken in the middle of the run, and again.
    empty = ft.ParameterDatabase(graph)
    empty.replay_journal(journal, model)
    assert_same_database(empty, db)
    assert empty.replay_journal(journal, model) == 0
    from_mid = ft.ParameterDatabase.load(mid, graph)
    from_mid.replay_journal(journal, model)
    assert_same_database(from_mid, db)
    # A replay onto the finished database changes nothing.
    done = db.copy()
    assert done.replay_journal(journal, model) == 0
    assert_same_database(done, db)


def test_replay_reproduces_stitch(tmp_path: pathlib.Path) -> None:
    graph, goal = lattice(8, 8)
    journal = tmp_path / "stitch.journal"
    snake = ft.Snake(graph, goal)
    assert graph.coords is not None
    coords = graph.coords
    cols = [
        graph.element_set(e for e in goal if coords[e.index][0] <= 2),
        graph.element_set(e for e in goal if coords[e.index][0] >= 5),
    ]
    db = ft.ParameterDatabase(graph)
    snake.stitch(db, cols, TargetModel(k=9), checkpoint=journal)
    assert db.calibrated == goal
    replayed = ft.ParameterDatabase(graph)
    replayed.replay_journal(journal, TargetModel(k=9))
    assert_same_database(replayed, db)


def test_resumed_run_with_its_own_journal_replays_idempotently(tmp_path: pathlib.Path) -> None:
    graph, goal = lattice(4, 4)
    first, second = tmp_path / "first.journal", tmp_path / "second.journal"
    snake = ft.Snake(graph, goal)
    db = ft.ParameterDatabase(graph)
    calls = 0

    class Stop(TargetModel):
        def error(self, step: ft.Step, candidates: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls > 20:
                raise RuntimeError("crash")
            return super().error(step, candidates)

    with pytest.raises(ft.ModelError):
        snake.calibrate(Stop(k=9), database=db, checkpoint=first)
    partial = db.step_count
    assert 0 < partial
    snake.calibrate(TargetModel(k=9), database=db, checkpoint=second)
    assert db.step_count > partial
    # Replaying the journal of the resumed part onto the finished database applies nothing.
    again = db.copy()
    assert again.replay_journal(second, TargetModel(k=9)) == 0
    assert again.step_count == db.step_count
    # Both journals, one after the other, rebuild the run from nothing.
    rebuilt = ft.ParameterDatabase(graph)
    rebuilt.replay_journal(first, TargetModel(k=9))
    rebuilt.replay_journal(second, TargetModel(k=9))
    assert_same_database(rebuilt, db)


def test_fresh_run_into_a_used_journal_warns(tmp_path: pathlib.Path) -> None:
    graph, goal = lattice(3, 3)
    journal = tmp_path / "run.journal"
    snake = ft.Snake(graph, goal)
    snake.calibrate(TargetModel(k=9), checkpoint=journal)
    with pytest.warns(RuntimeWarning, match="already holds"):
        snake.calibrate(TargetModel(k=9), checkpoint=journal)


JOURNAL_FAILURE_SCRIPT = """
import resource, signal, sys, warnings
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
sys.path.insert(0, sys.argv[1])
import faulttune as ft
from test_native_common import TargetModel, lattice
graph, goal = lattice(5, 5)
resource.setrlimit(resource.RLIMIT_FSIZE, (2000, 2000))
visible = []
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    ft.Snake(graph, goal, threads=1).calibrate(
        TargetModel(k=6, delay=0.01),
        checkpoint=sys.argv[2],
        on_step=lambda record: visible.append(len(caught)),
    )
journal_warnings = [w for w in caught if "checkpoint journal" in str(w.message)]
first = next(i for i, n in enumerate(visible) if n)
print(len(journal_warnings), first, len(visible))
"""


@pytest.mark.skipif(sys.platform == "win32", reason="needs RLIMIT_FSIZE")
def test_journal_write_failure_is_reported_while_the_run_goes_on(tmp_path: pathlib.Path) -> None:
    tests = str(pathlib.Path(__file__).parent)
    result = subprocess.run(
        [sys.executable, "-c", JOURNAL_FAILURE_SCRIPT, tests, str(tmp_path / "j")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    count, first, total = (int(x) for x in result.stdout.split())
    assert count == 1, "reported once"
    assert first < total - 5, "reported while the run was still going, not only at its end"


def test_foreign_elements_are_not_in_a_database(calibrated: Any) -> None:
    graph, goal, db = calibrated
    twin = ft.ProcessorGraph.square_lattice(4, 4)
    assert twin.node(0, 0) not in db, "like `in graph` and `in ElementSet`, not an error"
    assert twin.node(0, 0) not in graph
    assert graph.node(0, 0) in db


def test_a_database_of_another_graph_object_is_refused() -> None:
    graph, goal = lattice(4, 4)
    twin = ft.ProcessorGraph.square_lattice(4, 4)  # structurally identical, another object
    assert twin.fingerprint == graph.fingerprint
    snake = ft.Snake(graph, goal)
    for call in (
        lambda db: snake.calibrate(TargetModel(), database=db),
        lambda db: snake.recalibrate(db, graph.node(0, 0), TargetModel()),
    ):
        with pytest.raises(ValueError, match="different ProcessorGraph object"):
            call(ft.ParameterDatabase(twin))
    # A database loaded with the Snake's own graph works.
    assert len(snake.calibrate(TargetModel(), database=ft.ParameterDatabase(graph))) == len(goal)

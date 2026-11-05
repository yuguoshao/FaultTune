"""A three-qubit, physically consistent context-blindness benchmark.

This is an exactly specified toy Hamiltonian/channel model, not a device pulse
model. Set hbar=1 and use dimensionless time. During a control window,

    H(u) = (omega + u) X_0 / 2 + gamma |11><11|_(1,2).

The second term is fixed throughout an experiment. It equals
gamma * (I - Z_1 - Z_2 + Z_1 Z_2) / 4: a ZZ interaction with fixed local Z
terms, rather than a context-dependent probability assignment. The old control
is (u=0, duration=1); the fixed benchmark candidate is (u=-omega, duration=2).
This analytic candidate is fixed in advance, not selected using hidden Delta.

Both circuits start in |000>. They prepare H on q1. The spectator_0 circuit
leaves q2 in |0>; spectator_plus additionally prepares H on q2. After the same
control window they undo their preparation and measure all three qubits in Z.
Failure means ANY measured bit is one. Tuple entry b0+2*b1+4*b2 corresponds to
outcome |b2 b1 b0>. The only ideal successful output is 000 in both contexts.

Optional dephasing on q1/q2 has fixed rate kappa: off-diagonal coherence decays
as exp(-kappa*duration). Its Kraus operators commute with the control-window
Hamiltonian, so applying the channel at the window end is exact. Independent
bit-flip readout channels act on all three qubits after preparation is undone.
All channels are computed from explicit Kraus operators, not sampled here.

The full-probability and truth functions belong to the private simulator/scorer.
A validator must receive only an opaque old/new label, a context label, and a
paid measurement interface supplied by the runner. This module intentionally
has no measurement API or uncharged sampling path for a validator.
"""

import cmath
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence, Tuple


MODEL_VERSION = "faulttune.three_qubit_context_model.v1"
N_QUBITS = 3
DIMENSION = 1 << N_QUBITS
VARIANTS = ("old", "new")
CONTEXTS = ("spectator_0", "spectator_plus")
CALIBRATION_CONTEXT = "spectator_0"
WORKLOAD_CONTEXT = "spectator_plus"
State = Tuple[complex, ...]
Matrix2 = Tuple[Tuple[complex, complex], Tuple[complex, complex]]

I2 = ((1.0 + 0j, 0j), (0j, 1.0 + 0j))
X2 = ((0j, 1.0 + 0j), (1.0 + 0j, 0j))
Z2 = ((1.0 + 0j, 0j), (0j, -1.0 + 0j))
_ROOT_HALF = math.sqrt(0.5)
H2 = ((_ROOT_HALF + 0j, _ROOT_HALF + 0j),
      (_ROOT_HALF + 0j, -_ROOT_HALF + 0j))


@dataclass(frozen=True)
class QuantumModel:
    omega: float = math.pi / 3.0
    gamma: float = 0.0
    readout_flip: float = 0.01
    dephasing_rate: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.omega) or not math.isfinite(self.gamma):
            raise ValueError("omega and gamma must be finite")
        if not 0.0 <= self.readout_flip <= 1.0:
            raise ValueError("readout_flip must be a probability in [0, 1]")
        if not math.isfinite(self.dephasing_rate) or self.dephasing_rate < 0.0:
            raise ValueError("dephasing_rate must be finite and nonnegative")


@dataclass(frozen=True)
class PulseControl:
    amplitude: float
    duration: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.amplitude):
            raise ValueError("control amplitude must be finite")
        if not math.isfinite(self.duration) or self.duration <= 0.0:
            raise ValueError("control duration must be finite and positive")


def control_for(model: QuantumModel, variant: str) -> PulseControl:
    if variant == "old":
        return PulseControl(0.0, 1.0)
    if variant == "new":
        return PulseControl(-model.omega, 2.0)
    raise ValueError("variant must be one of {}".format(VARIANTS))


def _check_context(context: str) -> None:
    if context not in CONTEXTS:
        raise ValueError("context must be one of {}".format(CONTEXTS))


def _check_state(state: Sequence[complex]) -> None:
    if len(state) != DIMENSION:
        raise ValueError("this model requires a length-eight three-qubit state")


def _check_qubit(qubit: int) -> None:
    if isinstance(qubit, bool) or not isinstance(qubit, int) or not 0 <= qubit < N_QUBITS:
        raise ValueError("qubit must be 0, 1 or 2")


def state_norm_squared(state: Sequence[complex]) -> float:
    return math.fsum(abs(value) ** 2 for value in state)


def rx_matrix(angle: float) -> Matrix2:
    if not math.isfinite(angle):
        raise ValueError("rotation angle must be finite")
    cosine = math.cos(angle / 2.0)
    off_diagonal = -1j * math.sin(angle / 2.0)
    return ((complex(cosine), off_diagonal), (off_diagonal, complex(cosine)))


def apply_single_qubit(state: Sequence[complex], operator: Matrix2, qubit: int) -> State:
    """Apply an arbitrary 2x2 operator; do not renormalize Kraus branches."""
    _check_state(state)
    _check_qubit(qubit)
    mask = 1 << qubit
    output = list(state)
    for zero in range(DIMENSION):
        if zero & mask:
            continue
        one = zero | mask
        left, right = state[zero], state[one]
        output[zero] = operator[0][0] * left + operator[0][1] * right
        output[one] = operator[1][0] * left + operator[1][1] * right
    return tuple(output)


def apply_controlled_phase(state: Sequence[complex], angle: float,
                           first: int = 1, second: int = 2) -> State:
    """Apply exp(-i angle |11><11|) on the two specified qubits."""
    _check_state(state)
    _check_qubit(first)
    _check_qubit(second)
    if first == second:
        raise ValueError("controlled-phase qubits must differ")
    if not math.isfinite(angle):
        raise ValueError("controlled-phase angle must be finite")
    mask = (1 << first) | (1 << second)
    phase = cmath.exp(-1j * angle)
    return tuple(value * phase if index & mask == mask else value
                 for index, value in enumerate(state))


def _scale_operator(operator: Matrix2, scale: float) -> Matrix2:
    return tuple(tuple(scale * value for value in row) for row in operator)


def bit_flip_kraus(probability: float) -> Tuple[Matrix2, Matrix2]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Kraus probability must be in [0, 1]")
    return (_scale_operator(I2, math.sqrt(1.0 - probability)),
            _scale_operator(X2, math.sqrt(probability)))


def dephasing_kraus(probability: float) -> Tuple[Matrix2, Matrix2]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Kraus probability must be in [0, 1]")
    return (_scale_operator(I2, math.sqrt(1.0 - probability)),
            _scale_operator(Z2, math.sqrt(probability)))


def dephasing_probability(model: QuantumModel, variant: str) -> float:
    duration = control_for(model, variant).duration
    return -math.expm1(-model.dephasing_rate * duration) / 2.0


def apply_kraus_channel(branches: Iterable[State], operators: Sequence[Matrix2],
                        qubit: int) -> Tuple[State, ...]:
    """Exact pure-state ensemble for a channel; branch norms carry their weights."""
    result = []
    for state in branches:
        for operator in operators:
            branch = apply_single_qubit(state, operator, qubit)
            if any(value != 0 for value in branch):
                result.append(branch)
    return tuple(result)


def _prepare(context: str) -> State:
    _check_context(context)
    state = (1.0 + 0j,) + (0j,) * (DIMENSION - 1)
    state = apply_single_qubit(state, H2, 1)
    if context == WORKLOAD_CONTEXT:
        state = apply_single_qubit(state, H2, 2)
    return state


def _window(state: State, model: QuantumModel, variant: str) -> State:
    control = control_for(model, variant)
    angle = (model.omega + control.amplitude) * control.duration
    # X0 and P11_(1,2) commute: these operations are simultaneous Hamiltonian
    # terms over ONE duration, not two separately timed physical gates.
    state = apply_single_qubit(state, rx_matrix(angle), 0)
    return apply_controlled_phase(state, model.gamma * control.duration)


def _unprepare(state: State, context: str) -> State:
    state = apply_single_qubit(state, H2, 1)
    if context == WORKLOAD_CONTEXT:
        state = apply_single_qubit(state, H2, 2)
    return state


def unitary_statevector(model: QuantumModel, variant: str, context: str) -> State:
    """Unitary portion only, without either of the optional noise channels."""
    return _unprepare(_window(_prepare(context), model, variant), context)


def circuit_probabilities(model: QuantumModel, variant: str, context: str) -> Tuple[float, ...]:
    """Private scorer: exact eight-outcome distribution from gates and Kraus maps."""
    state = _window(_prepare(context), model, variant)
    branches = (state,)
    phase_channel = dephasing_kraus(dephasing_probability(model, variant))
    for qubit in (1, 2):
        branches = apply_kraus_channel(branches, phase_channel, qubit)
    branches = tuple(_unprepare(branch, context) for branch in branches)
    readout_channel = bit_flip_kraus(model.readout_flip)
    for qubit in range(N_QUBITS):
        branches = apply_kraus_channel(branches, readout_channel, qubit)
    probabilities = tuple(math.fsum(abs(branch[index]) ** 2 for branch in branches)
                          for index in range(DIMENSION))
    total = math.fsum(probabilities)
    if abs(total - 1.0) > 1e-10 or any(p < 0.0 for p in probabilities):
        raise ArithmeticError("unitary/Kraus evolution did not preserve probability")
    # Remove only final floating-point trace roundoff, after checking it.
    return tuple(probability / total for probability in probabilities)


def is_failure(outcome: int) -> bool:
    if isinstance(outcome, bool) or not isinstance(outcome, int) or not 0 <= outcome < DIMENSION:
        raise ValueError("outcome must be an integer in [0, 8)")
    return outcome != 0


def failure_probability(model: QuantumModel, variant: str, context: str) -> float:
    probabilities = circuit_probabilities(model, variant, context)
    return math.fsum(probabilities[index] for index in range(DIMENSION) if is_failure(index))


def analytic_probabilities(model: QuantumModel, variant: str, context: str) -> Tuple[float, ...]:
    """Independent closed-form full distribution; no statevector/Kraus helpers.

    Before noise, q0 has flip probability sin^2(epsilon/2). In the driven
    context, the q1/q2 probabilities are (1-3s/4,s/4,s/4,s/4), where
    s=sin^2(gamma*duration/2). Dephasing before each undo-H is equivalent to
    a bit flip afterward. Compose that flip with readout flips classically.
    """
    _check_context(context)
    control = control_for(model, variant)
    error_angle = (model.omega + control.amplitude) * control.duration
    local_flip = math.sin(error_angle / 2.0) ** 2
    r = model.readout_flip
    p0_one = r + (1.0 - 2.0 * r) * local_flip
    phase_flip = (1.0 - math.exp(-model.dephasing_rate * control.duration)) / 2.0
    effective = r + phase_flip - 2.0 * r * phase_flip
    if context == CALIBRATION_CONTEXT:
        pair_ideal = (1.0, 0.0, 0.0, 0.0)
        pair_flips = (effective, r)
    else:
        s = math.sin(model.gamma * control.duration / 2.0) ** 2
        pair_ideal = (1.0 - 0.75 * s, 0.25 * s, 0.25 * s, 0.25 * s)
        pair_flips = (effective, effective)
    pair_final = []
    for observed in range(4):
        terms = []
        for original, probability in enumerate(pair_ideal):
            weight = probability
            for bit, flip in enumerate(pair_flips):
                weight *= flip if (observed ^ original) & (1 << bit) else 1.0 - flip
            terms.append(weight)
        pair_final.append(math.fsum(terms))
    return tuple((p0_one if index & 1 else 1.0 - p0_one) * pair_final[index >> 1]
                 for index in range(DIMENSION))


def analytic_failure_probability(model: QuantumModel, variant: str, context: str) -> float:
    """Closed scalar formula independent of both full-distribution implementations."""
    _check_context(context)
    control = control_for(model, variant)
    a = math.sin((model.omega + control.amplitude) * control.duration / 2.0) ** 2
    r = model.readout_flip
    phase_flip = (1.0 - math.exp(-model.dephasing_rate * control.duration)) / 2.0
    effective = r + phase_flip - 2.0 * r * phase_flip
    local_success = (1.0 - r) - (1.0 - 2.0 * r) * a
    if context == CALIBRATION_CONTEXT:
        pair_success = (1.0 - effective) * (1.0 - r)
    else:
        q = 1.0 - effective
        s = math.sin(model.gamma * control.duration / 2.0) ** 2
        pair_success = q * q - (q * q - 0.25) * s
    return 1.0 - local_success * pair_success


def make_blind_pair(omega: float = math.pi / 3.0, readout_flip: float = 0.01,
                    dephasing_rate: float = 0.0) -> Tuple[QuantumModel, QuantumModel]:
    """Fixed benchmark worlds; no search or filtering by a hidden effect sign.

    Couplings are 0 and omega, with all other parameters shared. Their full
    calibration-context distributions coincide for every valid noise setting.
    Opposite workload signs hold, for example, at the defaults. Arbitrarily
    large dephasing or other parameters need not preserve the sign separation.
    """
    return (QuantumModel(omega, 0.0, readout_flip, dephasing_rate),
            QuantumModel(omega, omega, readout_flip, dephasing_rate))


def control_snapshot(model: QuantumModel, variant: str) -> Dict[str, Any]:
    control = control_for(model, variant)
    return {"variant": variant, "amplitude": control.amplitude,
            "window_duration": control.duration,
            "construction": "fixed analytic benchmark control; not selected by hidden Delta"}


def model_snapshot(model: QuantumModel) -> Dict[str, Any]:
    """Private provenance record; do not expose hidden parameters to policies."""
    return {
        "model_version": MODEL_VERSION,
        "n_qubits": N_QUBITS,
        "basis_index": "q0 + 2*q1 + 4*q2",
        "omega": model.omega,
        "gamma": model.gamma,
        "readout_flip": model.readout_flip,
        "dephasing_rate": model.dephasing_rate,
        "hamiltonian": "(omega+u)*X0/2 + gamma*(I-Z1-Z2+Z1*Z2)/4",
        "dephasing_generator": "rate/2 * (Z*rho*Z-rho), independently on q1 and q2",
        "readout_channel": "(1-r)*rho + r*X*rho*X, independently on q0,q1,q2 after undo-H",
        "controls": {variant: control_snapshot(model, variant) for variant in VARIANTS},
        "failure_predicate": "all-qubit Z outcome index is nonzero",
        "scope": "stationary three-qubit toy Hamiltonian and channels; ideal instantaneous H gates and ideal reset",
    }


def _hash_snapshot(snapshot: Dict[str, Any]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def model_hash(model: QuantumModel) -> str:
    return _hash_snapshot(model_snapshot(model))


def circuit_snapshot(model: QuantumModel, variant: str, context: str) -> Dict[str, Any]:
    """Executed pulse program/cost descriptors, excluding unknown model coefficients.

    H gates are ideal instantaneous gates in the physical model. A benchmark
    may price them explicitly, but must distinguish that cost abstraction from
    continuous Hamiltonian evolution during real finite-duration pulses.
    """
    _check_context(context)
    control = control_for(model, variant)
    hadamards = [1] if context == CALIBRATION_CONTEXT else [1, 2]
    return {
        "program_version": MODEL_VERSION,
        "variant": variant,
        "context": context,
        "reset_state": "000",
        "n_qubits": N_QUBITS,
        "prepare_h_qubits": hadamards,
        "window_drive_qubit": 0,
        "window_drive_amplitude": control.amplitude,
        "window_duration": control.duration,
        "undo_h_qubits": hadamards,
        "measured_z_qubits": [0, 1, 2],
        "reset_objects": 3,
        "readout_objects": 3,
        "failure_predicate": "outcome != 000",
        "timing": "H gates instantaneous; one simultaneous Hamiltonian window; old=1,new=2",
    }


def circuit_hash(model: QuantumModel, variant: str, context: str) -> str:
    return _hash_snapshot(circuit_snapshot(model, variant, context))


def truth_summary(model: QuantumModel, context: str = WORKLOAD_CONTEXT) -> Dict[str, Any]:
    """Private scorer output for a fixed old/new comparison."""
    old = failure_probability(model, "old", context)
    new = failure_probability(model, "new", context)
    return {
        "model_hash": model_hash(model),
        "context": context,
        "p_fail_old": old,
        "p_fail_new": new,
        "delta": new - old,
        "analytic_p_fail_old": analytic_failure_probability(model, "old", context),
        "analytic_p_fail_new": analytic_failure_probability(model, "new", context),
        "old_window_duration": 1.0,
        "new_window_duration": 2.0,
        "failure_predicate": "outcome != 000; all three qubits are read",
    }

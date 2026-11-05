"""Phase-sensitive measurements of the original stationary three-qubit model.

This is a private physical forward model, not an acceptance policy.  The new
control amplitude is PUBLIC and fixed independently of the unknown parameters.
The four unknowns are (omega, gamma, readout_flip, dephasing_rate).  Preparation,
analysis gates and reset remain ideal, as in the original circuit model.
Every menu item is one separately paid circuit execution with three readouts.
No reference observation is supplied for free.
"""

from dataclasses import dataclass
import math

import numpy as np


PARAMETER_NAMES = ("omega", "gamma", "readout_flip", "dephasing_rate")
MODEL_VERSION = "faulttune.phase_menu.v1"
I = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex)/math.sqrt(2)
S_DAGGER = np.diag([1, -1j])
ANALYSIS = {"Z": I, "X": H, "Y": H.dot(S_DAGGER)}
BITS = np.array([[(j >> q) & 1 for q in range(3)] for j in range(8)])
DEPHASING_DISTANCE = sum((BITS[:, q, None] != BITS[None, :, q]).astype(float)
                         for q in (1, 2))


@dataclass(frozen=True)
class PhaseParameters:
    omega: float
    gamma: float
    readout_flip: float
    dephasing_rate: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in self.as_array()):
            raise ValueError("all physical parameters must be finite")
        if not 0 <= self.readout_flip <= 1 or self.dephasing_rate < 0:
            raise ValueError("invalid readout probability or dephasing rate")

    def as_array(self):
        return np.array([self.omega, self.gamma, self.readout_flip, self.dephasing_rate])


@dataclass(frozen=True)
class PublicControls:
    candidate_amplitude: float
    old_duration: float = 1.0
    new_duration: float = 2.0

    def __post_init__(self):
        if not math.isfinite(self.candidate_amplitude):
            raise ValueError("candidate amplitude must be finite")
        if not all(math.isfinite(v) and v > 0 for v in (self.old_duration, self.new_duration)):
            raise ValueError("durations must be positive")

    def for_variant(self, variant):
        if variant == "old":
            return 0.0, self.old_duration
        if variant == "new":
            return -self.candidate_amplitude, self.new_duration
        raise ValueError("variant must be old or new")


@dataclass(frozen=True)
class Experiment:
    variant: str
    q2_preparation: str
    q0_basis: str = "Z"
    q1_basis: str = "X"
    q2_basis: str = "Z"

    def __post_init__(self):
        if self.variant not in ("old", "new") or self.q2_preparation not in ("0", "+", "1"):
            raise ValueError("invalid control or spectator preparation")
        if any(axis not in ANALYSIS for axis in (self.q0_basis, self.q1_basis, self.q2_basis)):
            raise ValueError("analysis axes must be X, Y, or Z")

    @property
    def name(self):
        return "{}_q2{}_{}{}{}".format(self.variant, self.q2_preparation,
                                       self.q0_basis, self.q1_basis, self.q2_basis)

    def description(self, controls):
        amplitude, duration = controls.for_variant(self.variant)
        return {"name": self.name, "variant": self.variant,
                "prepare": {"q0": "0", "q1": "+", "q2": self.q2_preparation},
                "analysis_axes": [self.q0_basis, self.q1_basis, self.q2_basis],
                "analysis_gates": {"X": "H then Z measurement", "Y": "S_dagger then H then Z measurement",
                                   "Z": "Z measurement"},
                "window_amplitude": amplitude, "window_duration": duration,
                "shots": 1, "readout_objects": 3,
                "reference_information": "q2 Z supplies a readout reference; q1 X with q2=0 supplies a dephasing reference"
                    if self.q2_preparation in ("0", "1") and self.q2_basis == "Z" else None}


def task_menu():
    """Four original task experiments, in old0, old+, new0, new+ order."""
    return tuple(Experiment(variant, context, "Z", "X", "X" if context == "+" else "Z")
                 for variant in ("old", "new") for context in ("0", "+"))


def phase_menu():
    """24 fixed actions; no axis, preparation, or amplitude is chosen from truth."""
    return tuple(Experiment(variant, context, q0, q1, "X" if context == "+" else "Z")
                 for variant in ("old", "new") for context in ("0", "+", "1")
                 for q0 in ("Z", "Y") for q1 in ("X", "Y"))


def reference_phase_menu():
    """Eight fixed actions: q0 Y, q1 X/Y, q2=0/1 measured in Z, both durations."""
    return tuple(Experiment(variant, context, "Y", q1, "Z")
                 for variant in ("old", "new") for context in ("0", "1") for q1 in ("X", "Y"))


def _readout_matrix(r):
    channel = np.array([[1-r, r], [r, 1-r]])
    derivative = np.array([[-1., 1.], [1., -1.]])
    transition = np.kron(channel, np.kron(channel, channel))
    jacobian = (np.kron(derivative, np.kron(channel, channel))
                + np.kron(channel, np.kron(derivative, channel))
                + np.kron(channel, np.kron(channel, derivative)))
    return transition, jacobian


def probabilities_and_jacobian(parameters, controls, experiment):
    """Return P[8] and analytic dP[8,4], with bit index q0+2*q1+4*q2."""
    u, duration = controls.for_variant(experiment.variant)
    angle = (parameters.omega+u)*duration
    q0 = np.array([math.cos(angle/2), -1j*math.sin(angle/2)])
    dq0 = duration/2*np.array([-math.sin(angle/2), -1j*math.cos(angle/2)])
    q2 = {"0": np.array([1.,0.]), "1": np.array([0.,1.]),
          "+": np.array([1.,1.])/math.sqrt(2)}[experiment.q2_preparation]
    occupied = BITS[:,1]*BITS[:,2]
    phase = np.exp(-1j*parameters.gamma*duration*occupied)
    prefactor = q2[BITS[:,2]]*phase/math.sqrt(2)
    state = q0[BITS[:,0]]*prefactor
    dstate_omega = dq0[BITS[:,0]]*prefactor
    dstate_gamma = -1j*duration*occupied*state
    rho = np.outer(state, state.conj())
    drho = np.zeros((4,8,8), dtype=complex)
    for index, dstate in enumerate((dstate_omega,dstate_gamma)):
        drho[index] = np.outer(dstate,state.conj())+np.outer(state,dstate.conj())
    attenuation = np.exp(-parameters.dephasing_rate*duration*DEPHASING_DISTANCE)
    rho *= attenuation
    drho *= attenuation
    drho[3] = -duration*DEPHASING_DISTANCE*rho
    analysis = np.kron(ANALYSIS[experiment.q2_basis],
                       np.kron(ANALYSIS[experiment.q1_basis],ANALYSIS[experiment.q0_basis]))
    ideal = np.diag(analysis.dot(rho).dot(analysis.conj().T)).real
    dideal = np.array([np.diag(analysis.dot(d).dot(analysis.conj().T)).real for d in drho]).T
    transition, readout_derivative = _readout_matrix(parameters.readout_flip)
    probabilities = transition.dot(ideal)
    jacobian = transition.dot(dideal)
    jacobian[:,2] += readout_derivative.dot(ideal)
    if np.min(probabilities) < -2e-14 or abs(np.sum(probabilities)-1) > 2e-13:
        raise ArithmeticError("physical probability conservation failed")
    if np.max(np.abs(np.sum(jacobian,axis=0))) > 2e-12:
        raise ArithmeticError("derivative probability conservation failed")
    return probabilities, jacobian


def probabilities(parameters, controls, experiment):
    return probabilities_and_jacobian(parameters,controls,experiment)[0]


def target_delta_and_gradient(parameters, controls, weight_plus=.8):
    if not 0 <= weight_plus <= 1:
        raise ValueError("workload weight must be a probability")
    weights = np.array([-(1-weight_plus), -weight_plus, 1-weight_plus, weight_plus])
    rows = [probabilities_and_jacobian(parameters,controls,e) for e in task_menu()]
    loss = np.array([1-p[0] for p,j in rows])
    loss_jacobian = np.array([-j[0] for p,j in rows])
    return float(weights.dot(loss)), weights.dot(loss_jacobian)


def fisher_information(parameters, controls, experiment, observation="full"):
    probabilities, jacobian = probabilities_and_jacobian(parameters,controls,experiment)
    if observation == "loss":
        probabilities = np.array([probabilities[0], np.sum(probabilities[1:])])
        jacobian = np.array([jacobian[0],np.sum(jacobian[1:],axis=0)])
    elif observation != "full":
        raise ValueError("observation must be full or loss")
    if np.any(probabilities <= 0):
        raise ValueError("regular Fisher diagnostic requires strictly positive outcome probabilities")
    return jacobian.T.dot(jacobian/probabilities[:,None])


def self_check():
    """Finite numerical cross-checks, not a physical or statistical theorem."""
    checks, worst_moment_error, worst_derivative_error = 0, 0., 0.
    controls = PublicControls(.23)
    for parameters in (PhaseParameters(.17,.31,.01,.003),PhaseParameters(.27,-.13,.07,.021)):
        for experiment in phase_menu():
            p,jacobian = probabilities_and_jacobian(parameters,controls,experiment)
            u,t = controls.for_variant(experiment.variant)
            q0_mean = float(np.dot(p,1-2*BITS[:,0]))
            q0_expected = (1-2*parameters.readout_flip)*(math.cos((parameters.omega+u)*t)
                           if experiment.q0_basis == "Z" else -math.sin((parameters.omega+u)*t))
            worst_moment_error = max(worst_moment_error,abs(q0_mean-q0_expected))
            if experiment.q2_preparation in ("0","1"):
                phase = parameters.gamma*t if experiment.q2_preparation == "1" else 0.
                expected = (1-2*parameters.readout_flip)*math.exp(-parameters.dephasing_rate*t)*(math.cos(phase)
                            if experiment.q1_basis == "X" else -math.sin(phase))
                worst_moment_error = max(worst_moment_error,abs(float(np.dot(p,1-2*BITS[:,1]))-expected))
            for k in range(4):
                step=1e-6
                left,right=parameters.as_array(),parameters.as_array()
                left[k]-=step
                right[k]+=step
                finite=(probabilities(PhaseParameters(*right),controls,experiment)
                        -probabilities(PhaseParameters(*left),controls,experiment))/(2*step)
                worst_derivative_error=max(worst_derivative_error,float(np.max(np.abs(finite-jacobian[:,k]))))
                checks+=1
    passed = worst_moment_error < 1e-12 and worst_derivative_error < 5e-8
    return {"passed":passed,"analytic_derivatives_checked":checks,
            "maximum_moment_error":worst_moment_error,"maximum_derivative_error":worst_derivative_error,
            "scope":"two fixed interior worlds, all 24 fixed menu actions; closed-form marginal moments and independent symmetric finite differences"}

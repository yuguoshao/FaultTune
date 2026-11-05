"""Predeclared physical benchmark cases for the PRIVATE simulator/scorer.

The 36 main cases are the full Cartesian product of three practical gaps,
two workload weights, three noise mechanisms, and two effect signs. No case
is selected using an algorithm's observations, decisions, cost, or accuracy.
The factory deterministically sets Hamiltonian parameters using the existing
physical closed form and independently evaluates the final probabilities by
the statevector/Kraus implementation. Constructing a benchmark at prescribed
effects is not an uncharged estimation capability for the validator.

At gamma=0, write A_old/A_new for the weighted spectator-pair success
probabilities and q=1-r. Then
    Delta = q*(A_old-A_new) - (1-2*r)*A_old*sin(omega/2)**2.
For this registered grid r<1/2 and A_old>0, so Delta is strictly decreasing in
omega on (0,pi/3). First set it to -2*epsilon, then bracket gamma in [0,pi/3]
to attain the registered signed effect. Root brackets and residuals are
floating-point numerical diagnostics, not interval-certified proof objects.

Development main cases target +/-1.6*epsilon; confirmation main cases target
+/-1.25*epsilon. This creates different main physical parameters before any
algorithm data. The gray and blind diagnostic controls are shared across
phases and are not held-out efficiency evidence. This remains one three-qubit
model family, not three system sizes or independent real-device evidence.

Returned dictionaries contain hidden truth and MUST NOT be given to a policy.
The runner may expose only a PublicExperiment and a paid observation API.
An infeasible prescribed cell stays in the returned matrix, with an explicit
failure status; downstream reports must not silently discard it.
"""

import math

from quantum_model import (CONTEXTS, QuantumModel, analytic_failure_probability,
                           circuit_hash, failure_probability, model_hash,
                           model_snapshot)


FACTORY_VERSION = "faulttune.circuit_case_matrix.v1"
EPSILONS = (0.01, 0.003, 0.001)
PLUS_WEIGHTS = (0.2, 0.8)
NOISE_SETTINGS = (
    ("low_readout", 0.0001, 0.0),
    ("standard_readout", 0.01, 0.0),
    ("dephasing", 0.003, 0.003),
)
PHASE_MULTIPLIERS = {"development": 1.6, "confirmation": 1.25}
PARAMETER_UPPER = math.pi / 3.0


def _phase(value):
    aliases = {"formal": "confirmation", "confirmatory": "confirmation"}
    value = aliases.get(value, value)
    if value not in PHASE_MULTIPLIERS:
        raise ValueError("phase must be development or confirmation")
    return value


def _tag(value):
    return format(value, ".6g").replace(".", "p").replace("-", "m")


def _arm_probabilities(model, analytic=False):
    function = analytic_failure_probability if analytic else failure_probability
    return tuple(function(model, variant, context)
                 for variant in ("old", "new") for context in CONTEXTS)


def _mixed_delta(model, weights, analytic=True):
    probabilities = _arm_probabilities(model, analytic=analytic)
    old = math.fsum(w * p for w, p in zip(weights, probabilities[:2]))
    new = math.fsum(w * p for w, p in zip(weights, probabilities[2:]))
    return new - old


def _bracketed_root(function, target, lower=0.0, upper=PARAMETER_UPPER):
    """Numerical continuous-function bracketing; keep failed endpoint evidence."""
    left_value, right_value = function(lower), function(upper)
    result = {
        "target": target, "initial_bracket": [lower, upper],
        "endpoint_values": [left_value, right_value],
        "numeric_status": "binary64 bisection; brackets and residuals are not machine-certified",
    }
    left_residual, right_residual = left_value - target, right_value - target
    if left_residual == 0 or right_residual == 0:
        root = lower if left_residual == 0 else upper
        result.update(feasible=True, root=root, final_bracket=[root, root], residual=0.0, iterations=0)
        return result
    if (left_residual > 0) == (right_residual > 0):
        root, residual = (lower, left_residual) if abs(left_residual) < abs(right_residual) else (upper, right_residual)
        result.update(feasible=False, root=root, final_bracket=[lower, upper], residual=residual,
                      iterations=0, reason="requested target is not bracketed in the registered parameter interval")
        return result
    iterations = 0
    for iterations in range(1, 81):
        middle = lower + (upper - lower) / 2
        if middle == lower or middle == upper:
            break
        value = function(middle) - target
        if value == 0:
            lower = upper = middle
            break
        if (value > 0) == (left_residual > 0):
            lower, left_residual = middle, value
        else:
            upper, right_residual = middle, value
        if upper - lower <= 2e-14:
            break
    candidates = (lower, lower + (upper - lower) / 2, upper)
    root = min(candidates, key=lambda value: abs(function(value) - target))
    residual = function(root) - target
    result.update(feasible=abs(residual) <= 2e-12, root=root, final_bracket=[lower, upper],
                  residual=residual, iterations=iterations)
    if not result["feasible"]:
        result["reason"] = "bisection residual exceeds the registered numerical tolerance"
    return result


def _base_omega(epsilon, weights, readout_flip, dephasing_rate):
    return _bracketed_root(
        lambda omega: _mixed_delta(QuantumModel(omega, 0.0, readout_flip, dephasing_rate), weights),
        -2 * epsilon,
    )


def _gamma_derivative_factor(model):
    """Positive value suffices for monotonic Delta(gamma) on [0,pi/3]."""
    q = 1 - model.readout_flip
    old_phase = -math.expm1(-model.dephasing_rate) / 2
    new_phase = -math.expm1(-2 * model.dephasing_rate) / 2
    q_old = q - (1 - 2 * model.readout_flip) * old_phase
    q_new = q - (1 - 2 * model.readout_flip) * new_phase
    old_local = q - (1 - 2 * model.readout_flip) * math.sin(model.omega / 2) ** 2
    # dDelta/dgamma = w*sin(gamma) *
    # [2*q*(q_new**2-.25)*cos(gamma) - .5*old_local*(q_old**2-.25)].
    return 2 * q * (q_new ** 2 - .25) * math.cos(PARAMETER_UPPER) - .5 * old_local * (q_old ** 2 - .25)


def _finish_case(case_id, phase, model, weights, epsilon, category, role,
                 noise_name, requested_delta, construction, feasible=True):
    probabilities = _arm_probabilities(model)
    analytic = _arm_probabilities(model, analytic=True)
    physics_error = max(abs(left - right) for left, right in zip(probabilities, analytic))
    if physics_error > 1e-11:
        raise ArithmeticError("statevector/Kraus and physical closed-form probabilities disagree")
    old = math.fsum(w * p for w, p in zip(weights, probabilities[:2]))
    new = math.fsum(w * p for w, p in zip(weights, probabilities[2:]))
    true_delta = new - old
    target_error = None if requested_delta is None else true_delta - requested_delta
    if target_error is not None and abs(target_error) > 2e-12:
        feasible = False
    reason = None
    if not feasible:
        failed = [name for name, value in construction.items() if isinstance(value, dict) and value.get("feasible") is False]
        reason = "unmet prescribed construction: " + ", ".join(failed or ["final physical target residual"])
    return {
        "case_id": case_id, "phase": phase, "factory_version": FACTORY_VERSION,
        "model": model_snapshot(model), "model_hash": model_hash(model),
        "candidate_id": circuit_hash(model, "new", "spectator_plus"),
        "contexts": list(CONTEXTS), "weights": list(weights), "epsilon": epsilon,
        "probabilities": list(probabilities), "old_failure": old, "new_failure": new,
        "true_delta": true_delta, "requested_delta": requested_delta,
        "actual_gap_multiple": true_delta / epsilon,
        "category": category, "benchmark_role": role,
        "efficiency_eligible": role == "main" and feasible and abs(true_delta) >= epsilon,
        "intended_efficiency_eligible": role == "main",
        "stratum": noise_name,
        "strata": {"noise": noise_name, "epsilon": epsilon, "weight_plus": weights[1],
                   "intended_sign": -1 if category in ("beneficial", "blind_beneficial") else
                   1 if category in ("harmful", "near_boundary", "blind_harmful") else 0},
        "feasible": feasible, "status": "ready" if feasible else "infeasible",
        "infeasible_reason": reason, "construction": construction,
        "target_residual": target_error, "physics_crosscheck_max_error": physics_error,
        "truth_access": "private scorer only; do not pass this dictionary to a validator",
        "selection_uses_algorithm_data": False,
        "scope": "one stationary three-qubit physical model family; not hardware or three system sizes",
    }


def _target_case(case_id, phase, epsilon, weight_plus, noise, requested_delta,
                 category, role="main", base=None):
    noise_name, readout_flip, dephasing_rate = noise
    weights = (1 - weight_plus, weight_plus)
    if base is None:
        base = _base_omega(epsilon, weights, readout_flip, dephasing_rate)
    gamma = _bracketed_root(
        lambda value: _mixed_delta(QuantumModel(base["root"], value, readout_flip, dephasing_rate), weights),
        requested_delta,
    )
    model = QuantumModel(base["root"], gamma["root"], readout_flip, dephasing_rate)
    construction = {
        "kind": "registered physical-parameter roots before algorithm observations",
        "omega_root": dict(base), "gamma_root": gamma,
        "gamma_derivative_factor_lower_numeric": _gamma_derivative_factor(model),
        "candidate_rule": "old=(u=0,tau=1), new=(u=-omega,tau=2), frozen before validation",
        "case_truth_used_only_before_validation": True,
    }
    return _finish_case(case_id, phase, model, weights, epsilon, category, role,
                        noise_name, requested_delta, construction,
                        feasible=base["feasible"] and gamma["feasible"])


def build_cases(phase="development"):
    """Return all 44 registered rows, including any explicitly infeasible rows.

    Main: 3 epsilons x 2 weights x 3 noise settings x 2 signs = 36.
    Gray diagnostics: three zero effects plus three +epsilon/10 effects = 6.
    Blind diagnostics: the two classical omega=pi/3 worlds at plus weight .9.
    """
    phase = _phase(phase)
    multiplier = PHASE_MULTIPLIERS[phase]
    prefix = "dev" if phase == "development" else "confirm"
    rows = []
    for epsilon in EPSILONS:
        for weight_plus in PLUS_WEIGHTS:
            for noise in NOISE_SETTINGS:
                base = _base_omega(epsilon, (1 - weight_plus, weight_plus), noise[1], noise[2])
                for category, sign in (("beneficial", -1), ("harmful", 1)):
                    case_id = "{}_main_e{}_w{}_{}_{}".format(prefix, _tag(epsilon), _tag(weight_plus), noise[0], category)
                    rows.append(_target_case(case_id, phase, epsilon, weight_plus, noise,
                                             sign * multiplier * epsilon, category, base=base))
    for epsilon in EPSILONS:
        rows.append(_finish_case(
            "{}_zero_e{}".format(prefix, _tag(epsilon)), phase,
            QuantumModel(0.0, 0.0, .01, 0.0), (.2, .8), epsilon,
            "zero", "gray_diagnostic", "standard_readout", 0.0,
            {"kind": "fixed zero control Hamiltonian; equal old/new distributions", "shared_across_phases": True},
        ))
        near = _target_case("{}_near_boundary_e{}".format(prefix, _tag(epsilon)), phase,
                            epsilon, .8, NOISE_SETTINGS[1], epsilon / 10,
                            "near_boundary", role="gray_diagnostic")
        near["construction"]["shared_across_phases"] = True
        rows.append(near)
    for world, gamma, category in (("m0", 0.0, "blind_beneficial"),
                                   ("m1", PARAMETER_UPPER, "blind_harmful")):
        rows.append(_finish_case(
            "{}_blind_{}".format(prefix, world), phase,
            QuantumModel(PARAMETER_UPPER, gamma, .01, 0.0), (.1, .9), .01,
            category, "blind_diagnostic", "standard_readout", None,
            {"kind": "fixed analytic context-blindness worlds", "shared_across_phases": True},
        ))
    return rows

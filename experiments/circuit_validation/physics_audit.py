"""Deterministic physical-model audit; Python 3.8 standard library only.

Run ``python -B physics_audit.py`` from this directory. The JSON is a private
scorer artifact, not input available to any validation policy. Rational checks
are exact under the stated trigonometric identities; finite floating-point
checks are reported separately and are not a proof of physical-device fidelity.
"""

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import quantum_model as qm


AUDIT_VERSION = "faulttune.three_qubit_physics_audit.v1"
TOLERANCE = 1e-12
HERE = Path(__file__).resolve().parent


def _require(condition, description):
    if not condition:
        raise AssertionError(description)


def _fraction_pmf(local_flip, phase_sine_squared):
    """Independent exact reference from the tensor-factor probabilities."""
    pair = (1 - 3 * phase_sine_squared / 4,) + (phase_sine_squared / 4,) * 3
    return tuple((local_flip if outcome & 1 else 1 - local_flip) * pair[outcome >> 1]
                 for outcome in range(8))


def _fraction_readout(pmf, readout_flip):
    result = []
    for observed in range(8):
        probability = Fraction(0)
        for original, original_probability in enumerate(pmf):
            distance = bin(observed ^ original).count("1")
            probability += (original_probability * readout_flip ** distance
                            * (1 - readout_flip) ** (3 - distance))
        result.append(probability)
    return tuple(result)


def exact_reference(readout_flip=Fraction(0)):
    """omega=pi/3: sin^2(pi/6)=1/4 and sin^2(pi/3)=3/4."""
    result = {}
    for world in ("M0", "M1"):
        result[world] = {}
        for context in qm.CONTEXTS:
            rows = {}
            for variant in qm.VARIANTS:
                local = Fraction(1, 4) if variant == "old" else Fraction(0)
                phase = Fraction(0)
                if world == "M1" and context == qm.WORKLOAD_CONTEXT:
                    phase = Fraction(1, 4) if variant == "old" else Fraction(3, 4)
                pmf = _fraction_readout(_fraction_pmf(local, phase), readout_flip)
                _require(sum(pmf) == 1, "exact reference must be normalized")
                _require(all(p >= 0 for p in pmf), "exact reference must be nonnegative")
                rows[variant] = {"pmf": pmf, "p_fail": sum(pmf[1:])}
            rows["delta"] = rows["new"]["p_fail"] - rows["old"]["p_fail"]
            result[world][context] = rows
    return result


def _serialize_fractions(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_fractions(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_fractions(item) for item in value]
    return value


def audit_exact_reference():
    noiseless = exact_reference()
    reference_values = (("M0", Fraction(1, 4), Fraction(0), Fraction(-1, 4)),
                        ("M1", Fraction(25, 64), Fraction(9, 16), Fraction(11, 64)))
    for world, old, new, delta in reference_values:
        target = noiseless[world][qm.WORKLOAD_CONTEXT]
        _require((target["old"]["p_fail"], target["new"]["p_fail"], target["delta"])
                 == (old, new, delta), "rational reference values disagree")
    _require(noiseless["M0"][qm.CALIBRATION_CONTEXT]
             == noiseless["M1"][qm.CALIBRATION_CONTEXT], "exact calibration equality")
    expected_old = tuple(Fraction(x, 64) for x in (39, 13, 3, 1, 3, 1, 3, 1))
    expected_new = tuple(Fraction(x, 16) for x in (7, 0, 3, 0, 3, 0, 3, 0))
    _require(noiseless["M1"][qm.WORKLOAD_CONTEXT]["old"]["pmf"] == expected_old,
             "exact old complete PMF")
    _require(noiseless["M1"][qm.WORKLOAD_CONTEXT]["new"]["pmf"] == expected_new,
             "exact new complete PMF")
    max_residual = 0.0
    for world, model in zip(("M0", "M1"), qm.make_blind_pair(readout_flip=0.0)):
        for context in qm.CONTEXTS:
            for variant in qm.VARIANTS:
                actual = qm.circuit_probabilities(model, variant, context)
                expected = noiseless[world][context][variant]["pmf"]
                max_residual = max(max_residual,
                                   max(abs(a - float(b)) for a, b in zip(actual, expected)))
    _require(max_residual <= TOLERANCE, "numeric model versus exact rational reference")
    noisy = exact_reference(Fraction(1, 100))
    negative_calibration_delta = -noisy["M1"][qm.CALIBRATION_CONTEXT]["delta"]
    plus_delta = noisy["M1"][qm.WORKLOAD_CONTEXT]["delta"]
    threshold = negative_calibration_delta / (negative_calibration_delta + plus_delta)
    _require(threshold == Fraction(39204, 66305), "exact noisy mixture threshold")
    return {
        "status": "exact rational arithmetic under the two stated trigonometric identities",
        "trigonometric_identities": ["sin(pi/6)^2 = 1/4", "sin(pi/3)^2 = 3/4"],
        "omega": "pi/3", "gamma_M0": "0", "gamma_M1": "pi/3",
        "dephasing_rate": "0",
        "noiseless": _serialize_fractions(noiseless),
        "readout_flip_1_over_100": _serialize_fractions(noisy),
        "max_statevector_vs_noiseless_fraction_pmf_error": max_residual,
        "mixture_caveat": {
            "definition": "w is spectator_plus weight; spectator_0 weight is 1-w",
            "noiseless_strict_positive_M1_threshold": "16/27",
            "default_readout_strict_positive_M1_threshold": str(threshold),
            "default_readout_threshold_decimal": float(threshold),
            "equal_weight_default_M1_delta": str((plus_delta - negative_calibration_delta) / 2),
            "equal_weight_default_M1_delta_decimal": float((plus_delta - negative_calibration_delta) / 2),
        },
    }


def _kdagger_k_error(operators):
    return max(abs(sum(operator[k][row].conjugate() * operator[k][column]
                       for operator in operators for k in range(2)) - (row == column))
               for row in range(2) for column in range(2))


def audit_channels_and_gates():
    channel_errors = []
    for probability in (0.0, 0.01, 0.17, 0.5, 1.0):
        for name, constructor in (("readout_bit_flip", qm.bit_flip_kraus),
                                  ("window_dephasing", qm.dephasing_kraus)):
            residual = _kdagger_k_error(constructor(probability))
            _require(residual <= TOLERANCE, "Kraus completeness")
            channel_errors.append({"channel": name, "probability": probability,
                                   "max_completeness_error": residual})
    gate_errors = {"H": _kdagger_k_error((qm.H2,))}
    for angle in (0.0, 0.37, math.pi, -2.0):
        gate_errors["Rx({})".format(angle)] = _kdagger_k_error((qm.rx_matrix(angle),))
    _require(max(gate_errors.values()) <= TOLERANCE, "single-qubit unitarity")
    cp_error = 0.0
    idle_is_identity = True
    for angle in (0.0, 0.713, -1.3):
        for basis in range(8):
            state = tuple(complex(int(index == basis)) for index in range(8))
            actual = qm.apply_controlled_phase(state, angle)
            z1, z2 = 1 - 2 * ((basis >> 1) & 1), 1 - 2 * ((basis >> 2) & 1)
            eigenvalue = (1 - z1 - z2 + z1 * z2) / 4.0
            expected_phase = complex(math.cos(angle * eigenvalue), -math.sin(angle * eigenvalue))
            expected = tuple(value * expected_phase for value in state)
            cp_error = max(cp_error, max(abs(a - b) for a, b in zip(actual, expected)))
            if basis < 4:
                idle_is_identity = idle_is_identity and actual == state
    _require(cp_error <= TOLERANCE and idle_is_identity, "fixed Hamiltonian CP/idle checks")
    return {"kraus_completeness": channel_errors, "single_qubit_unitarity_errors": gate_errors,
            "fixed_ZZ_plus_local_Z_phase_max_error": cp_error,
            "CP_identity_on_entire_q2_zero_subspace": idle_is_identity,
            "analytic_CPTP_argument": "Kraus form gives complete positivity; sum K^dagger K=(1-p)I+pI=I for every 0<=p<=1",
            "numeric_scope": "listed gates/channel parameters only; not a physical-device certification"}


def audit_forward_model():
    max_full_pmf_error = max_scalar_error = max_trace_error = 0.0
    calibration_equalities = 0
    program_checks = 0
    for omega in (0.063, math.pi / 3.0, 1.7):
        for flip in (0.0, 0.01, 0.2):
            for rate in (0.0, 0.03, 0.4):
                first, second = qm.make_blind_pair(omega, flip, rate)
                for variant in qm.VARIANTS:
                    a = qm.circuit_probabilities(first, variant, qm.CALIBRATION_CONTEXT)
                    b = qm.circuit_probabilities(second, variant, qm.CALIBRATION_CONTEXT)
                    _require(a == b, "full calibration PMFs must coincide exactly in floating-point execution")
                    calibration_equalities += 1
                for model in (first, second):
                    for variant in qm.VARIANTS:
                        for context in qm.CONTEXTS:
                            actual = qm.circuit_probabilities(model, variant, context)
                            analytic = qm.analytic_probabilities(model, variant, context)
                            max_full_pmf_error = max(max_full_pmf_error,
                                                     max(abs(a - b) for a, b in zip(actual, analytic)))
                            scalar = qm.analytic_failure_probability(model, variant, context)
                            max_scalar_error = max(max_scalar_error, abs(math.fsum(actual[1:]) - scalar))
                            max_trace_error = max(max_trace_error, abs(math.fsum(actual) - 1.0))
                            _require(all(0.0 <= p <= 1.0 for p in actual), "valid complete PMF")
                            program_checks += 1
    _require(max(max_full_pmf_error, max_scalar_error, max_trace_error) <= TOLERANCE,
             "independent forward-model checks")
    return {"fixed_grid": {"omega": [0.063, math.pi / 3.0, 1.7],
                           "readout_flip": [0.0, 0.01, 0.2], "dephasing_rate": [0.0, 0.03, 0.4],
                           "gamma": "0 and omega for every grid point"},
            "full_program_checks": program_checks, "bitwise_equal_calibration_pmf_pairs": calibration_equalities,
            "max_full_pmf_analytic_vs_statevector_kraus_error": max_full_pmf_error,
            "max_scalar_analytic_vs_statevector_kraus_error": max_scalar_error,
            "max_normalized_pmf_sum_error": max_trace_error,
            "noise_warning": "the grid audits distribution agreement and blindness; opposite signs are not required or selected on this grid"}


def audit_default_pair():
    result = {}
    first, second = qm.make_blind_pair()
    for world, model in zip(("M0", "M1"), (first, second)):
        result[world] = {"model": qm.model_snapshot(model), "model_hash": qm.model_hash(model),
                         "contexts": {}}
        for context in qm.CONTEXTS:
            rows = {"truth": qm.truth_summary(model, context)}
            for variant in qm.VARIANTS:
                rows[variant] = {"pmf": qm.circuit_probabilities(model, variant, context),
                                 "analytic_pmf": qm.analytic_probabilities(model, variant, context),
                                 "program": qm.circuit_snapshot(model, variant, context),
                                 "program_hash": qm.circuit_hash(model, variant, context)}
            result[world]["contexts"][context] = rows
    for variant in qm.VARIANTS:
        first_rows = result["M0"]["contexts"][qm.CALIBRATION_CONTEXT][variant]
        second_rows = result["M1"]["contexts"][qm.CALIBRATION_CONTEXT][variant]
        _require(first_rows["pmf"] == second_rows["pmf"], "default complete calibration equality")
        for context in qm.CONTEXTS:
            _require(qm.circuit_hash(first, variant, context) == qm.circuit_hash(second, variant, context),
                     "same public program in both worlds")
    _require(qm.truth_summary(first)["delta"] < 0 < qm.truth_summary(second)["delta"],
             "default spectator-plus target effects have opposite signs")
    _require(qm.model_hash(first) != qm.model_hash(second), "hidden models differ")
    return result


def build_audit():
    source_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                     for name in ("quantum_model.py", "physics_audit.py")}
    return {"schema_version": AUDIT_VERSION, "passed": True,
            "scope": "specified stationary three-qubit toy Hamiltonian/channel model; private scorer artifact",
            "basis_index": "q0 + 2*q1 + 4*q2; tuples are ordered as |q2 q1 q0>",
            "failure_predicate": "any nonzero all-three-qubit Z readout",
            "blind_menu": ["(old,spectator_0)", "(new,spectator_0)"],
            "opposite_sign_target": "spectator_plus alone; see mixture caveat before changing target weights",
            "candidate": "fixed analytic amplitude and duration controls, not selected by hidden effect sign",
            "window_durations": {"old": 1.0, "new": 2.0},
            "timing_caveat": "H gates instantaneous and reset ideal in physics; additional preparation/readout/reset/switch costs are an external accounting model",
            "tensor_factor_caveat": "q0 does not entangle with q1q2; the changed common duration alters the q1q2 interaction phase",
            "observations": "the benchmark oracle exposes the paid failure bit, a deterministic pushforward of the full PMF, rather than full bitstrings",
            "floating_tolerance": TOLERANCE, "source_sha256": source_hashes,
            "exact_reference": audit_exact_reference(),
            "channels_and_gates": audit_channels_and_gates(),
            "forward_model_grid": audit_forward_model(), "default_pair": audit_default_pair()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "physics_audit.json")
    args = parser.parse_args()
    artifact = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    print("Physics audit passed: {} programs, {} identical calibration PMF pairs; {}".format(
        artifact["forward_model_grid"]["full_program_checks"],
        artifact["forward_model_grid"]["bitwise_equal_calibration_pmf_pairs"], args.output.resolve()))


if __name__ == "__main__":
    main()

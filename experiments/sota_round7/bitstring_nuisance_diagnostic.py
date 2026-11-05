"""Deterministic Round 7 audit: same-duration updates retain useful bitstrings.

No Monte Carlo, fitted validator, or SOTA result.  The new control family uses
the existing Hamiltonian and Kraus maps without changing frozen model code.
Run from the repository root with Python 3.8 or newer.  Output is append-only.
"""

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.circuit_validation import quantum_model as qm
from experiments.circuit_validation.direct_tests import _Bounds


def gate_probabilities(model, control, context=qm.WORKLOAD_CONTEXT):
    """Scorer-only pipeline with explicit controls; never patch control_for."""
    state = qm._prepare(context)
    state = qm.apply_single_qubit(
        state, qm.rx_matrix((model.omega + control.amplitude) * control.duration), 0)
    state = qm.apply_controlled_phase(state, model.gamma * control.duration)
    branches = (state,)
    phase_flip = -math.expm1(-model.dephasing_rate * control.duration) / 2.0
    for qubit in (1, 2):
        branches = qm.apply_kraus_channel(branches, qm.dephasing_kraus(phase_flip), qubit)
    branches = tuple(qm._unprepare(branch, context) for branch in branches)
    for qubit in (0, 1, 2):
        branches = qm.apply_kraus_channel(
            branches, qm.bit_flip_kraus(model.readout_flip), qubit)
    pmf = tuple(math.fsum(abs(branch[i]) ** 2 for branch in branches) for i in range(8))
    assert abs(math.fsum(pmf) - 1.0) < 2e-13
    return pmf


def convolve_spectators(ideal, first_flip, second_flip):
    # Works in either exact Fraction or floating arithmetic; no gate helpers.
    result = []
    for observed in range(4):
        terms = []
        for original, probability in enumerate(ideal):
            weight = probability
            for bit, flip in enumerate((first_flip, second_flip)):
                weight *= flip if (observed ^ original) & (1 << bit) else 1 - flip
            terms.append(weight)
        result.append(sum(terms))
    return tuple(result)


def independent_product(model, control, context=qm.WORKLOAD_CONTEXT):
    local = math.sin((model.omega + control.amplitude) * control.duration / 2.0) ** 2
    r = model.readout_flip
    x = r + (1 - 2 * r) * local
    phase_flip = (1 - math.exp(-model.dephasing_rate * control.duration)) / 2.0
    effective = r + phase_flip - 2 * r * phase_flip
    if context == qm.WORKLOAD_CONTEXT:
        t = math.sin(model.gamma * control.duration / 2.0) ** 2
        ideal = (1 - 3 * t / 4, t / 4, t / 4, t / 4)
        q = convolve_spectators(ideal, effective, effective)
    else:
        q = convolve_spectators((1.0, 0.0, 0.0, 0.0), effective, r)
    pmf = tuple((x if i & 1 else 1 - x) * q[i >> 1] for i in range(8))
    return x, q, pmf


def bound(value):
    if isinstance(value, Fraction):
        return _Bounds(value.numerator) / value.denominator
    return _Bounds(value)


def kl(p, q):
    p, q = bound(p), bound(q)
    return p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()


def js(a, b):
    m = (a + b) / 2
    return (kl(a, m) + kl(b, m)) / 2


def encode_interval(interval):
    return {"lower_decimal": str(interval.lo), "upper_decimal": str(interval.hi),
            "display_midpoint": float((interval.lo + interval.hi) / 2)}


def exact_example():
    eta, xo, xn, s = Fraction(1, 200), Fraction(3, 100), Fraction(1, 100), Fraction(1, 2)
    ao, an = (xo - eta) / (1 - 2 * eta), (xn - eta) / (1 - 2 * eta)
    t = ((1 - eta) ** 2 - s) / ((1 - eta) ** 2 - Fraction(1, 4))
    controls = (qm.PulseControl(2 * math.asin(math.sqrt(float(ao))), 1.0),
                qm.PulseControl(2 * math.asin(math.sqrt(float(an))), 1.0))
    model = qm.QuantumModel(0.0, 2 * math.asin(math.sqrt(float(t))), float(eta), 0.0)
    ideal = (1 - 3 * t / 4, t / 4, t / 4, t / 4)
    q = convolve_spectators(ideal, eta, eta)
    assert q[0] == s and sum(q) == 1
    exact_pmfs = [tuple((x if i & 1 else 1 - x) * q[i >> 1] for i in range(8))
                  for x in (xo, xn)]
    pmf_errors = [max(abs(a - float(b)) for a, b in zip(gate_probabilities(model, c), p))
                  for c, p in zip(controls, exact_pmfs)]
    assert max(pmf_errors) < 2e-13
    yo, yn = 1 - s * (1 - xo), 1 - s * (1 - xn)
    assert yn - yo == s * (xn - xo) == -Fraction(1, 100)

    # A same-control physical world reverses the update sign while leaving Q
    # unchanged, because sin^2 is even.  This is a legal binary lower-bound
    # comparator even when the full control response is public knowledge.
    reflected = qm.QuantumModel(-controls[0].amplitude - controls[1].amplitude,
                                model.gamma, model.readout_flip, model.dephasing_rate)
    reflection_errors = [max(abs(a - float(b)) for a, b in zip(
        gate_probabilities(reflected, c), p))
        for c, p in zip(controls, reversed(exact_pmfs))]
    assert max(reflection_errors) < 2e-13

    jx, jy = js(xo, xn), js(yo, yn)
    dx = (kl(xo, xn) + kl(xn, xo)) / 2
    dy = (kl(yo, yn) + kl(yn, yo)) / 2
    event_kl = kl(Fraction(95, 100), Fraction(1, 100))
    midpoint = (xo + xn) / 2
    fisher_ratio = (1 - s + s * midpoint) / (s * midpoint)
    variance = lambda x: x * (1 - x)
    v_full = 2 * s * s * (variance(xo) + variance(xn)) + (xn - xo) ** 2 * s * (1 - s)
    v_or = 2 * (variance(yo) + variance(yn))

    # A concrete warning: the original unequal-duration family need not have
    # common Q. Even a smaller q0 error can then worsen all-qubit failure.
    unequal_new = qm.PulseControl(controls[1].amplitude / 2.0, 2.0)
    unequal_x, unequal_q, unequal_p = independent_product(model, unequal_new)
    unequal_delta = 1 - unequal_p[0] - float(yo)
    assert abs(unequal_x - float(xn)) < 1e-14 and unequal_delta > 0.0

    return {
        "private_scorer_truth": {
            "omega": model.omega, "gamma": model.gamma, "readout_flip": str(eta),
            "dephasing_rate": model.dephasing_rate, "ideal_phase_sin_squared": str(t),
            "old_local_ideal_flip": str(ao), "new_local_ideal_flip": str(an),
            "observed_q0_old": str(xo), "observed_q0_new": str(xn),
            "common_spectator_pmf_exact": [str(v) for v in q],
            "old_full_pmf_exact": [str(v) for v in exact_pmfs[0]],
            "new_full_pmf_exact": [str(v) for v in exact_pmfs[1]],
            "old_failure": str(yo), "new_failure": str(yn), "Delta": str(yn - yo),
        },
        "public_fixed_controls": {
            "old": {"amplitude": controls[0].amplitude, "duration": 1.0},
            "new": {"amplitude": controls[1].amplitude, "duration": 1.0},
            "formula": "u_old=2 asin(sqrt(5/198)); u_new=2 asin(sqrt(1/198))",
            "status": "Static exploratory design; never generated from a held-out scorer truth.",
        },
        "gate_kraus_vs_exact_product_max_error": max(pmf_errors),
        "physical_reflected_world": {
            "omega": reflected.omega, "formula": "omega'=-u_old-u_new-omega (duration=1)",
            "Delta": str(yo - yn), "shared_Q": True,
            "swapped_full_pmfs_max_error": max(reflection_errors),
        },
        "enlarged_factorized_model_information": {
            "model": "Independent Bern(x_j) times arbitrary common Q; x_old,x_new,Q unrestricted.",
            "balanced_JS_full_per_shot": encode_interval(jx),
            "balanced_JS_OR_per_shot": encode_interval(jy),
            "balanced_JS_ratio_full_over_OR": encode_interval(jx / jy),
            "local_fisher_ratio_at_x_midpoint": str(fisher_ratio),
            "delta_variance_full_coefficient_exact": str(v_full),
            "delta_variance_OR_coefficient_exact": str(v_or),
            "delta_variance_ratio_exact": str(v_or / v_full),
            "delta_variance_ratio_display": float(v_or / v_full),
            "event_kl_95_percent_power_1_percent_risk": encode_interval(event_kl),
            "necessary_balanced_total_shots_full": encode_interval(event_kl / jx),
            "necessary_balanced_total_shots_OR": encode_interval(event_kl / jy),
            "scope": "Necessary expected-shot bounds for paired/balanced tests valid over the enlarged class; not attainable stopping costs, not physical-model universal bounds, not a ratio of optimal costs.",
        },
        "physical_binary_reflection_information": {
            "balanced_KL_full_per_shot": encode_interval(dx),
            "balanced_KL_OR_per_shot": encode_interval(dy),
            "balanced_KL_ratio_full_over_OR": encode_interval(dx / dy),
            "necessary_balanced_total_shots_full": encode_interval(event_kl / dx),
            "necessary_balanced_total_shots_OR": encode_interval(event_kl / dy),
            "scope": "A genuine physical alternative with opposite sign and identical public controls/context. Change-of-measure necessary expected-shot bounds apply to tests with P_accept>=.95 here and <=.01 in the reflected world, stopping only after paid independent old/new pairs. These lower bounds alone do not demonstrate achievable speedup.",
        },
        "unequal_duration_obstruction": {
            "new_duration": 2.0, "new_amplitude": unequal_new.amplitude,
            "new_q0_failure": unequal_x, "new_spectator_success": unequal_q[0],
            "new_all_qubit_failure": 1 - unequal_p[0], "Delta": unequal_delta,
            "meaning": "Local q0 improves, but the changed spectator distribution makes the target worse. Common duration/noise is an essential structural assumption.",
        },
    }


def deterministic_physics_audit():
    errors, identity_errors, shared_errors = [], [], []
    # Fixed Cartesian points, not random or chosen by effect signs.
    for eta in (0.0, 0.005, 0.07):
        for kappa in (0.0, 0.1):
            for gamma in (0.0, 0.7, 2.3):
                for context in qm.CONTEXTS:
                    model = qm.QuantumModel(0.41, gamma, eta, kappa)
                    controls = (qm.PulseControl(0.23, 1.3), qm.PulseControl(-0.31, 1.3))
                    outputs = [independent_product(model, c, context) for c in controls]
                    gates = [gate_probabilities(model, c, context) for c in controls]
                    for output, gate in zip(outputs, gates):
                        errors.append(max(abs(a - b) for a, b in zip(output[2], gate)))
                    xo, qo, po = outputs[0]
                    xn, qn, pn = outputs[1]
                    shared_errors.append(max(abs(a - b) for a, b in zip(qo, qn)))
                    identity_errors.append(abs((po[0] - pn[0]) - qo[0] * (xn - xo)))
    assert max(errors) < 2e-13
    assert max(identity_errors) < 2e-13
    assert max(shared_errors) == 0.0
    return {"fixed_models_contexts": len(shared_errors), "gate_pmf_checks": len(errors),
            "gate_vs_product_max_error": max(errors),
            "Delta_factorization_max_error": max(identity_errors),
            "shared_Q_max_error": max(shared_errors),
            "status": "Floating gate verification plus exact rational designated instance; analytic factorization proof is stated separately."}


def build_report():
    report = {
        "schema": "faulttune.round7.bitstring_nuisance_diagnostic.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "deterministic_exploration_not_SOTA_claim",
        "mechanism": "Same-duration q0 amplitude update, shared spectator noise, complete three-bit readout.",
        "analytic_derivation": [
            "The q0 Hamiltonian acts on a distinct tensor factor from the q1,q2 Hamiltonian. Initial preparation and all noise channels also factor across this partition.",
            "Changing only q0 amplitude at a common duration leaves the spectator channel Q identical: P_j(X,S)=Bern(x_j) times Q(S).",
            "For all-bit-zero success and s=Q(00), y_j=1-s(1-x_j); hence Delta=s(x_new-x_old). For s>0 the signs agree exactly.",
            "Complete readout supplies X from each paid shot. OR compression discards X whenever S is nonzero. No physical postselection or extra shot is required to retain X.",
            "In the enlarged arbitrary-common-Q model, the likelihood factor involving Q cancels in the sign test, so X counts suffice. This need not be a minimal sufficient statistic for the more restricted physical model with shared nuisance parameters.",
            "A balanced equal-mean null has m=(x_old+x_new)/2, and per-shot KL equals Bernoulli JS. Fisher information for x from OR divided by that from X is s*x/(1-s+s*x).",
            "Delta magnitude can use pooled spectator success: var coefficient 2*s^2*(v(x_old)+v(x_new))+(x_new-x_old)^2*s*(1-s). This is asymptotic, not a finite-sample certificate.",
        ],
        "example": exact_example(),
        "deterministic_physics_audit": deterministic_physics_audit(),
        "required_strong_comparators": [
            {"name": "Safe two-sample Bernoulli e-process on q0", "reference": "Turner, Ly and Gruenwald; existing safe full-count/KT evidence implementation", "requirement": "Same complete bitstrings, known common-Q factorization, target context and paid costs; extract X directly."},
            {"name": "Frozen exact paired-sign mixture on q0", "reference": "Existing strongest qualified paired-sign mixtures", "requirement": "Fresh independent old/new shots, charge concordant pairs too, use identical risk and power targets."},
            {"name": "Bernoulli GLR with valid mixture threshold", "reference": "Kaufmann and Koolen 2021; Track-and-Stop when unequal allocation is allowed", "requirement": "Prove implemented threshold applies to the composite sign null and permitted allocation; count all learning shots."},
            {"name": "Physics-aware known-control likelihood/e-process", "reference": "Restricted shared-omega/readout Hamiltonian family", "requirement": "May exploit exactly the same public controls, tensor factorization and response relations as FaultTune; unknown physical parameters stay hidden and are inferred from paid outcomes."},
        ],
        "fairness_and_limits": [
            "The roughly fifty-fold effect is loss from compressing observed bitstrings to OR, not superiority over SOTA methods given the same information.",
            "Any comparator can use X; no claim is made that FaultTune outperforms those comparators.",
            "No free inverse, controlled-U, amplitude amplification, coherent repeat queries, simulator truth, or shared old/new random noise is available.",
            "Every old/new target shot costs one complete circuit run. Both configurations have equal duration, readout/reset cost, context and three-bit measurement.",
            "A full-output theorem does not repair the previously proved wrong-context nonidentifiability: this construction requires target-context shots.",
            "Equal duration is a new frozen control family; prior duration-changing benchmark results remain untouched.",
            "Common-Q factorization is supplied by a declared exact model, not inferred for free. Arm-dependent spectator noise or q0-spectator entangling terms require a new robust certificate.",
            "The designated high-gain instance is exploratory and cannot be repurposed as a held-out SOTA result.",
            "JS and variance ratios are deterministic information diagnostics; they are not measured sample-cost speedups or finite-cap power guarantees.",
        ],
        "recommended_next_gate": "Only benchmark a new validator after including structure-aware q0 safe/paired/GLR baselines. A result against OR-only validation is an instrumentation benefit. Seek extra measurement or across-update structure if the goal remains >=3x over same-information SOTA.",
        "source_files": {},
    }
    for relative in ("experiments/sota_round7/bitstring_nuisance_diagnostic.py",
                     "experiments/circuit_validation/quantum_model.py",
                     "experiments/circuit_validation/direct_tests.py"):
        report["source_files"][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/sota_round7/mechanisms/bitstring_nuisance_diagnostic.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    report = build_report()
    if not args.check_only:
        output = ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    ex = report["example"]
    print(json.dumps({"status": report["status"], "output": args.output if not args.check_only else None,
        "JS_ratio": ex["enlarged_factorized_model_information"]["balanced_JS_ratio_full_over_OR"],
        "physical_reflection_KL_ratio": ex["physical_binary_reflection_information"]["balanced_KL_ratio_full_over_OR"],
        "physics_audit": report["deterministic_physics_audit"]}, indent=2))


if __name__ == "__main__":
    main()

"""Independent algebra checks for two phase-validation claims.

No frozen source is imported, no observations are sampled, and no formal
experimental results are changed. The finite checks below accompany the
following analytic arguments; they do not replace those arguments.

1. The original unequal-duration model has a nuisance obstruction.
   Write a=1-2r, q=1-r, s=sin(omega/2)^2, z=exp(-kappa), and gamma=0.
   The exact target change is

       Delta = (q-a*s)*((1+a*z)/2)^2 - q*((1+a*z*z)/2)^2.

   For r=1/100, s=2*epsilon, and 0<epsilon<=1/100, the world z=1 has
   Delta=-1.920996*epsilon. The world z=1-4*epsilon has exactly

       1.920996*epsilon - 19.171152*epsilon^2
       + 53.321408*epsilon^3 - 60.850944*epsilon^4,

   which is greater than 1.729223629056*epsilon. Both worlds use the SAME
   old/new controls: (0,1), (-omega,2). Only the unknown kappa changes.

   Any fixed-duration analysis circuit has output L1 distance at most
   2*tau*abs(kappa_1-kappa_0): each of the two dephasing channels contributes
   at most tau*abs(dkappa), and later CPTP maps contract trace distance.
   Independent readout flips after analysis give every 3-bit outcome
   probability at least r^3, hence KL <= L1^2/r^3 = O(epsilon^2).
   The stopped-transcript change-of-measure bound therefore forces
   Omega(epsilon^-2 * kl(0.95,delta)) paid independent circuit executions.
   This covers adaptive choices from bounded-duration analysis menus, with
   shared classical randomness. It excludes growing-duration free probes.

2. A separate equal-duration, single-Pauli-rotation model has an exact route.
   A single R_P(x)=exp(-i*x*P/2), P^2=I, between x-independent CPTP maps and
   followed by a fixed POVM gives f(x)=A+B*cos(x)+C*sin(x), even with unknown
   coefficients. Put t=sin(h/2) and m=theta+h/2. Then exactly

       f(theta+h)-f(theta) = t*(f(m+pi/2)-f(m-pi/2)).

   For a fixed endpoint, put L=sqrt(log(1/delta))+sqrt(log(1/beta)).
   Hoeffding with n >= 2*t^2*epsilon^-2*L^2 independent observations per
   shifted angle and acceptance threshold
       -abs(t)*sqrt(2*log(1/delta)/n)
   on t*(mean_plus-mean_minus) gives false acceptance <=delta and acceptance
   >=1-beta when Delta<=-epsilon. Set beta=0.05. Thus the total cost is
   O((1+t^2/epsilon^2)*log(1/delta)) for fixed beta. The two
   observations are independent and BOTH are paid. No r/kappa estimation
   is needed under the exact common-channel model. Equal duration alone
   does not imply this model if simultaneous drive and noise fail to commute.

   For a matching lower-bound family, set
       f_+(x)=1/2+c*sin(x-m), f_-(x)=1/2-c*sin(x-m), c=epsilon/(2*t).
   Direct old/new probabilities differ across worlds by exactly epsilon.
   Here direct full-output means the original fixed task POVM, including
   all its bit strings; it does NOT grant new measurement bases or a
   quantum output state. Such measurements belong to the broader probe class.
   An arbitrary angle changes them by at most epsilon/abs(t). These laws
   are physical: use antipodal mixed-qubit Bloch vectors with length
   b=epsilon/(abs(t)*(1-2r)) <=1/2, a fixed Rx gate and Z measurement, and
   the same positive readout rate r. Independent ancillas add no information.
   In fact ANY single-copy POVM after common channels has information at
   most D(rho_+||rho_-)=b*log((1+b)/(1-b)) <= (8/3)*b^2.
   Thus direct full-output queries require Omega(epsilon^-2), while even
   arbitrary independent-reset probes require Omega(t^2*epsilon^-2), up to
   the common binary kl(power,false_accept) factor. Shift queries attain the
   latter order. With t^2=Theta(epsilon), this is epsilon^-2 vs epsilon^-1.

The trigonometric identity, data processing, Hoeffding, and transcript KL
argument are standard. This is a conditional observation-information
separation, not evidence of beating a model-aware SOTA algorithm supplied
with the SAME shifted probes. Parameter-dependent noise can destroy the
identity; a global error bound |f-f_harmonic|<=eta leaves identity residual
at most R=2*eta*(1+abs(t)). Robust acceptance subtracts R from its threshold
and requires an effective margin epsilon-2*R, not epsilon-R.

3. That mismatch condition is sharp for shifted-only information.
   For t>0 and C0=-eta*(1+t)/t, take harmonic functions
       g_w(x)=1/2+(C0+w*eta)*sin(x-m), w in {-1,+1}.
   Let e_w have values -w*eta, +w*eta at old,new, respectively, and
   -w*eta,+w*eta at shift_plus,shift_minus, respectively. Bounded smooth
   interpolation is possible when the four angles are distinct.
   Both worlds give identical shifted probabilities, but their task
   changes are -4*eta*(1+t) and zero. If eta>=epsilon/[4*(1+t)], shifted
   observations cannot guarantee the stated power and safety even with
   infinite samples. At the equality eta=Theta(epsilon), every angle has
   inter-world probability difference at most 4*eta=O(epsilon). With a
   positive probability floor, arbitrary-angle independent-reset probes
   therefore again require Omega(epsilon^-2) observations. Unknown
   angle-dependent replacement channels realize this SPECIFIC family:
   with fixed readout r, put u_w(x)=(f_w(x)-r)/(1-2*r) and use
       Lambda_w,x(rho)=Tr(rho)*diag(1-u_w(x),u_w(x)).
   The four Kraus operators sqrt(p_i)*|i><j|, i,j in {0,1}, satisfy
   sum K^dagger K=I whenever 0<=u<=1. On our family u stays in [1/4,3/4].
   Consequently any single-copy POVM has KL <=16*epsilon^2/(3*(1-2*r)^2)
   at eta=epsilon/[4*(1+t)], by quantum data processing applied to diagonal
   output states. Independent ancillas do not add information. This is a
   minimax obstruction for a model class admitting those angle-dependent
   channels, not a theorem that arbitrary response errors are automatically
   compatible with a particular microscopic or weak-noise hardware model.
   No coherent query access across executions is assumed.
"""

import json
import math
import random
from fractions import Fraction


def original_zero_phase_delta(s, r, z):
    """Exact for Fraction arguments; z=exp(-kappa), gamma=0."""
    a, q = 1 - 2 * r, 1 - r
    q1, q2 = (1 + a * z) / 2, (1 + a * z * z) / 2
    return (q - a * s) * q1 * q1 - q * q2 * q2


def original_small_parameter_coefficients(r):
    """Delta=a*q^2*(kappa-s)+(3*q/4)*(q^2-1/4)*gamma^2+remainder.

    The remainder is O(kappa^2+gamma^4+s*kappa+s*gamma^2+kappa*gamma^2),
    uniformly on a compact readout interval strictly below 1/2.
    """
    a, q = 1 - 2 * r, 1 - r
    return {"kappa": a * q * q, "s": -a * q * q,
            "gamma_squared": Fraction(3, 4) * q * (q * q - Fraction(1, 4))}


def fourier_response(x, coefficients):
    a, b, c = coefficients
    return a + b * math.cos(x) + c * math.sin(x)


def finite_parameter_shift(theta, update, coefficients):
    mid = theta + update / 2
    return math.sin(update / 2) * (
        fourier_response(mid + math.pi / 2, coefficients)
        - fourier_response(mid - math.pi / 2, coefficients))


def lower_bound_response(angle, theta, update, epsilon, sign):
    t = math.sin(update / 2)
    if not t:
        raise ValueError("A zero sine update has identically zero task change")
    return 0.5 + sign * epsilon / (2 * t) * math.sin(angle - theta - update / 2)


def bernoulli_kl(p, q):
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def fixed_accept_design(epsilon, update, delta=0.01, beta=0.05, eta=0.0):
    """An explicit FIXED-ENDPOINT acceptance test, with all 2*n shots paid.

    Let D_j be independent plus-minus Bernoulli differences in [-1,1].
    Reject the update's nonbenefit hypothesis only if
        sin(update/2)*mean(D_j) < returned acceptance_threshold.
    The paired differences are an analysis device, not coupled measurement
    noise. This function does not authorize optional peeking at the same
    fixed-endpoint threshold.
    """
    if not (epsilon > 0 and 0 < delta < 1 and 0 < beta < 1 and eta >= 0):
        raise ValueError("Invalid design parameters")
    t = abs(math.sin(update / 2))
    residual = 2 * eta * (1 + t)
    effective_margin = epsilon - 2 * residual
    if not t or effective_margin <= 0:
        raise ValueError("No positive robust shifted-observation margin")
    total_root_logs = math.sqrt(math.log(1 / delta)) + math.sqrt(math.log(1 / beta))
    per_angle = max(1, math.ceil(2 * t * t * total_root_logs ** 2 / effective_margin ** 2))
    safety_radius = t * math.sqrt(2 * math.log(1 / delta) / per_angle)
    power_radius = t * math.sqrt(2 * math.log(1 / beta) / per_angle)
    assert safety_radius + power_radius <= effective_margin * (1 + 1e-14)
    return {"observations_per_angle": per_angle, "total_paid_shots": 2 * per_angle,
            "false_accept_bound": delta, "beneficial_accept_lower": 1 - beta,
            "acceptance_threshold": -residual - safety_radius,
            "identity_residual_bound": residual, "effective_margin": effective_margin}


def exact_obstruction_check():
    r = Fraction(1, 100)
    a, q = 1 - 2 * r, 1 - r
    coefficients = [2 * a * q * q,
                    -4 * q * a * a - 16 * q * q * a,
                    64 * q * a * a - 8 * a * a * a,
                    -64 * q * a * a]
    endpoint = Fraction(1, 100)
    lower_coefficient = (coefficients[0] + coefficients[1] * endpoint
                         + coefficients[3] * endpoint ** 3)
    assert coefficients[2] > 0 and lower_coefficient > 1
    cases = []
    for denominator in (100, 300, 1000, 3000, 10000, 30000):
        epsilon = Fraction(1, denominator)
        good = original_zero_phase_delta(2 * epsilon, r, Fraction(1))
        bad = original_zero_phase_delta(2 * epsilon, r, 1 - 4 * epsilon)
        polynomial = sum(c * epsilon ** (j + 1) for j, c in enumerate(coefficients))
        assert bad == polynomial
        assert good <= -epsilon and bad >= lower_coefficient * epsilon
        cases.append({"epsilon": str(epsilon), "good_delta": str(good),
                      "bad_delta": str(bad),
                      "bad_kappa": -math.log1p(-4 * float(epsilon))})
    return {"polynomial_coefficients_exact": [str(c) for c in coefficients],
            "polynomial_coefficients_decimal": [float(c) for c in coefficients],
            "uniform_bad_delta_over_epsilon_lower_exact": str(lower_coefficient),
            "uniform_bad_delta_over_epsilon_lower": float(lower_coefficient),
            "small_parameter_coefficients": {
                k: str(v) for k, v in original_small_parameter_coefficients(r).items()},
            "checked_exact_instances": cases}


def conditional_shift_check():
    rng = random.Random(614229)
    maximum_residual = 0.0
    for _ in range(200):
        theta, update = rng.uniform(-3, 3), rng.uniform(-1, 1)
        radius, phase = rng.uniform(0, 0.4), rng.uniform(-3, 3)
        coefficients = (0.5, radius * math.cos(phase), radius * math.sin(phase))
        direct = (fourier_response(theta + update, coefficients)
                  - fourier_response(theta, coefficients))
        residual = abs(direct - finite_parameter_shift(theta, update, coefficients))
        maximum_residual = max(maximum_residual, residual)
    assert maximum_residual < 2e-15
    cases = []
    theta, readout, delta = 0.2, 0.01, 0.01
    for epsilon in (0.01, 0.003, 0.001, 0.0003, 0.0001):
        update = 2 * math.asin(math.sqrt(epsilon))
        t = math.sin(update / 2)
        mid = theta + update / 2
        b = epsilon / (abs(t) * (1 - 2 * readout))
        assert b <= 0.5
        task_gaps = []
        for sign in (-1, 1):
            old = lower_bound_response(theta, theta, update, epsilon, sign)
            new = lower_bound_response(theta + update, theta, update, epsilon, sign)
            task_gaps.append(new - old)
            assert abs(new - old - sign * epsilon) < 1e-15
        p_old = [lower_bound_response(theta, theta, update, epsilon, sign)
                 for sign in (-1, 1)]
        p_shift = [lower_bound_response(mid + math.pi / 2, theta, update, epsilon, sign)
                   for sign in (-1, 1)]
        quantum_kl = b * math.log((1 + b) / (1 - b))
        assert quantum_kl <= (8 / 3) * b * b
        cases.append({"epsilon": epsilon, "update": update,
                      "task_gaps": task_gaps,
                      "direct_bernoulli_kl": bernoulli_kl(*p_old),
                      "shifted_bernoulli_kl": bernoulli_kl(*p_shift),
                      "any_single_copy_probe_quantum_kl_upper": quantum_kl,
                      "quantum_kl_analytic_bound": (8 / 3) * b * b,
                      "fixed_endpoint_acceptance_design":
                          fixed_accept_design(epsilon, update, delta=delta, beta=0.05)})
    return {"finite_identity_checks": 200, "maximum_float_residual": maximum_residual,
            "rate_family": "sin(update/2)^2=epsilon; fixed positive readout=0.01",
            "checked_instances": cases}


def mismatch_obstruction_check():
    """Exact four-angle indistinguishability at the sharp residual threshold."""
    cases = []
    for denominator in (10, 30, 100):
        t = Fraction(1, denominator)
        epsilon = t * t
        eta = epsilon / (4 * (1 + t))
        c0 = -eta * (1 + t) / t
        sine = {"old": -t, "new": t, "shift_plus": Fraction(1),
                "shift_minus": Fraction(-1)}
        error_sign = {"old": -1, "new": 1, "shift_plus": -1, "shift_minus": 1}
        worlds = {}
        for sign in (-1, 1):
            probabilities = {
                angle: Fraction(1, 2) + (c0 + sign * eta) * value
                + sign * error_sign[angle] * eta for angle, value in sine.items()}
            assert all(0 < p < 1 for p in probabilities.values())
            worlds[sign] = probabilities
        assert worlds[-1]["new"] - worlds[-1]["old"] == -epsilon
        assert worlds[1]["new"] - worlds[1]["old"] == 0
        assert all(worlds[-1][angle] == worlds[1][angle]
                   for angle in ("shift_plus", "shift_minus"))
        global_probability_floor = Fraction(1, 2) - abs(c0) - 2 * eta
        assert global_probability_floor > 0
        r = Fraction(1, 100)
        readout_contraction = 1 - 2 * r
        quantum_output_floor = (global_probability_floor - r) / readout_contraction
        assert quantum_output_floor >= Fraction(1, 4)
        quantum_output_populations = {}
        for world, probabilities in worlds.items():
            quantum_output_populations[str(world)] = {}
            for angle, probability in probabilities.items():
                u = (probability - r) / readout_contraction
                assert 0 <= u <= 1 and (1 - u) + u == 1
                assert r + readout_contraction * u == probability
                quantum_output_populations[str(world)][angle] = str(u)
        cases.append({"sin_half_update": str(t), "epsilon": str(epsilon),
                      "eta": str(eta), "global_probability_floor": str(global_probability_floor),
                      "global_interworld_probability_difference_upper": str(4 * eta),
                      "replacement_channel_output_population_floor": str(quantum_output_floor),
                      "replacement_channel_populations": quantum_output_populations,
                      "any_single_copy_povm_kl_upper":
                          str(Fraction(16, 3) * epsilon * epsilon / readout_contraction ** 2),
                      "worlds": {str(w): {k: str(v) for k, v in p.items()}
                                 for w, p in worlds.items()}})
    return {"claim": "Identical shifted laws; good Delta=-epsilon; bad Delta=0",
            "eta_threshold": "epsilon / (4 * (1 + abs(sin(update/2))))",
            "checked_exact_instances": cases}


def main():
    print(json.dumps({"schema": "faulttune.phase_theory_check.v1",
                      "status": "algebra_checks_passed",
                      "scope": "Deterministic algebra only; no observations or frozen code imported",
                      "original_model_nuisance_obstruction": exact_obstruction_check(),
                      "conditional_common_channel_shift": conditional_shift_check(),
                      "gate_dependent_mismatch_obstruction": mismatch_obstruction_check()}, indent=2))


if __name__ == "__main__":
    main()

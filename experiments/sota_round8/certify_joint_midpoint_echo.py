"""Directed deterministic constants for the joint midpoint/echo policy.

This file is a development-instance certificate, not a benchmark. No sampled
observations are generated. Public constants are separate from scoring truth.
"""
import hashlib
import json
import math
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from fractions import Fraction as F
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.circuit_validation.direct_tests import _Bounds as I


def iv(x):
    x = F(x)
    return I(x.numerator)/x.denominator


def enclose(x):
    return [str(x.lo), str(x.hi)]


def sin_rational(x):
    """Alternating sine bounds on [0, 1.2], exact rational terms."""
    x = F(x)
    assert 0 <= x <= F(6, 5)
    partial = F(0)
    for k in range(26):
        partial += (-1)**k*x**(2*k+1)/math.factorial(2*k+1)
    # The 26-term (last term negative) partial sum is below sin(x).
    upper = partial+x**53/math.factorial(53)
    return I(iv(partial).lo, iv(upper).hi)


def binary_kl(p, q):
    return p*(p/q).log()+(1-p)*((1-p)/(1-q)).log()


def compute():
    sine = (I(985).sqrt()-I(193).sqrt())/198
    assert sine.lo > iv(F(883, 10000)).hi
    # h lies in (0,pi/2), hence this also proves 2h<.177.
    assert sine.hi < sin_rational(F(177, 2000)).lo
    amin = iv(F(9, 10))*(-iv(F(9, 200))).exp()
    cone = iv(F(3)*F(177, 1000)/(2*F(883, 10000)))*iv(F(1, 5))/(amin*sin_rational(F(6, 5)))
    assert cone.hi < iv(F(3, 4)).lo
    beta_e = F(29, 250)
    echo_boundary = amin*sin_rational(F(8, 59))
    assert iv(beta_e).hi < echo_boundary.lo

    # Scoring truth, used ONLY here: d=.02, chi=0, not given to policy.
    mu = iv(F(1, 50))/sine
    p, q, beta = iv(F(4, 7)), iv(F(3, 7)), F(3, 25)
    a = iv(1-beta*beta).log()/2
    b = (iv(1+beta).log()-iv(1-beta).log())/2
    drift = a+b*p*mu
    assert drift.lo > 0
    overshoot = iv(1+beta).log()
    level = I(200).log()
    linear = b*q.sqrt()
    y = (linear+(linear*linear+4*drift*(level+overshoot)).sqrt())/(2*drift)
    expected_joint = y*y
    cap = 65536
    power_joint = 1-iv(F(1, 200))-expected_joint/cap

    cutoff, alternative = F(60, 883), F(1, 5)
    target_up = iv((1+alternative)/(1+cutoff)).log()
    target_down = iv((1-alternative)/(1-cutoff)).log()
    target_drift = (1+mu)/2*target_up+(1-mu)/2*target_down
    target_upper = (level+target_up)/target_drift
    ea = -iv(1-beta_e*beta_e).log()/2
    eb = (iv(1+beta_e).log()-iv(1-beta_e).log())/2
    ey = (eb+(eb*eb+4*ea*(I(100).log()+ea+eb)).sqrt())/(2*ea)
    echo_upper = ey*ey
    separate_upper = target_upper+echo_upper
    separate_power = 1-iv(F(1, 200))-separate_upper/cap
    # Lower bound only for the named two-stage rule without fallback.
    target_lower = binary_kl(iv(F(19, 20)), iv(F(1, 200)))/binary_kl((1+mu)/2, iv((1+cutoff)/2))
    echo_floor_i = I(100).log()/ea
    echo_floor_lo = int(echo_floor_i.lo.to_integral_value(rounding=ROUND_CEILING))
    echo_floor_hi = int(echo_floor_i.hi.to_integral_value(rounding=ROUND_CEILING))
    assert echo_floor_lo == echo_floor_hi
    separate_lower = target_lower+iv(F(19, 20))*echo_floor_lo
    ratio = separate_lower/expected_joint
    assert ratio.lo > 1
    assert power_joint.lo > Decimal('.95') and separate_power.lo > Decimal('.95')

    return {
        'schema': 'faulttune.round8.joint_midpoint_echo_certificate.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'certified_reference_advantage_over_specified_two_stage_control_only',
        'sota_goal_complete': False,
        'new_monte_carlo_observations': 0,
        'public_contract': {
            'model': 'H(u)=((omega+u)/2)X0+(gamma+chi*u)P11, disjoint tensor factors; no extra q0 noise; symmetric readout r<=.05; stationary spectator Z dephasing kappa<=.005.',
            'old_amplitude': '2 asin sqrt(5/198)',
            'new_amplitude': '2 asin sqrt(1/198)',
            'echo': 'Three low=-1/high=+1 echo blocks, each sensing window duration1, gain6; |chi|<=.2; ideal known preparation/analysis/X gates.',
            'echo_noise_duration_upper': '9',
            'minimum_echo_contrast': '.9 exp(-.045)',
            'price_scope': 'Full scheduled echo duration<=9 under the explicit simulation cost schedule. This conservative contrast envelope also covers sensing-only dephasing.',
            'power_failure_scope': 'Cost and power assume delivered physical outcomes and no exogenous execution errors or interruption. Interrupted/failed executions receive full-cap score but need their own failure-rate model for probabilistic performance.',
        },
        'directed_constants': {
            'sin_half_update_gap': enclose(sine),
            'half_gap_upper_test_sin_00885': enclose(sin_rational(F(177, 2000))),
            'minimum_echo_contrast': enclose(amin),
            'physical_cone_coefficient': enclose(cone),
            'rational_cone_coefficient': '3/4',
            'echo_mean_at_chi_4_over_177': enclose(echo_boundary),
            'safe_separate_echo_boundary': str(beta_e),
        },
        'physical_sign_proof': [
            'For equal-duration targets, s(theta)=a-(a-1/4)sin^2(theta/2), a in[1/4,1]; s>=1/4 and |s_prime|<=3/8. Thus d>(3/2)|chi|*du guarantees Delta<0, with the mirrored sufficient condition for Delta>0.',
            'Midpoint Rx(S*pi/2) demodulation gives d=sin(h)*mu_d exactly. Echo Rx(-S*pi/2) on q1 gives mu_e=A sin(6chi). Since sinc decreases on[0,1.2], |chi|<=.2|mu_e|/(Amin*sin1.2).',
            'The certified cone coefficient is strictly below3/4. Therefore mu_d>(3/4)|mu_e| implies benefit, and mu_d<-(3/4)|mu_e| implies harm. These are sufficient conditions only; many nonzero target effects need a richer fallback.',
        ],
        'joint_rule': {
            'action_probabilities': {'midpoint': '4/7', 'echo': '3/7'},
            'bet': '3/25', 'threshold_each': 200, 'cap': cap,
            'rule': 'Four products M[t,s]; midpoint multiplies1+beta*t*Z_d, echo1+beta*s*Z_e. Report t only when both CURRENT s products exceed200. Stop at first report or cap.',
            'risk_proof': 'If direction t is wrong then t*mu_d<=3|mu_e|/4; at least one s has (4/7)*t*mu_d+(3/7)*s*mu_e<=0. That nonnegative product is a supermartingale with unit start. Its Ville bound controls a wrong direction by.005. At Delta=0 the union of two wrong directions is<=.01.',
        },
        'reference_results': {
            'truth_only_for_scoring': 'x_old=.03, x_new=.01, chi=0, original reference r=.005, s=.5, Delta=-.01. Sign-reflected midpoint signal has the same cost and power bounds.',
            'midpoint_mean': enclose(mu), 'joint_log_drift': enclose(drift),
            'joint_expected_shots_upper': str(expected_joint.hi),
            'joint_full_cap_penalized_expected_shots_upper': str(expected_joint.hi),
            'joint_correct_stop_lower': str(power_joint.lo),
            'target_stage_expected_shots_upper': str(target_upper.hi),
            'echo_stage_expected_shots_upper': str(echo_upper.hi),
            'separate_expected_shots_upper': str(separate_upper.hi),
            'separate_correct_stop_lower': str(separate_power.lo),
            'separate_target_expected_shots_lower': str(target_lower.lo),
            'separate_echo_pathwise_shot_floor': echo_floor_lo,
            'separate_expected_shots_lower': str(separate_lower.lo),
            'separate_over_joint_shot_cost_ratio_lower': str(ratio.lo),
            'joint_sensing_time_upper': str(((p+6*q)*expected_joint).hi),
            'joint_device_time_upper': str(((iv(F(33,10))*p+9*q)*expected_joint).hi),
            'scope': 'A single development reference and its mirror. The comparator lower bound is specific to the named two-stage rule; no lower bound for all same-action methods is claimed. Device-time figures are upper bounds only, not comparative speedups.',
        },
        'expected_cost_proof': [
            'At chi=0, log min_s M[+,s]=n*a+b(S_mid-|S_echo|). Each separate log product is a sum of iid bounded increments with mean D>0. The inclusion {T>n} subset of union_s{log M[+,s]_n<L} and Hoeffding give an exponential tail, proving integrability before optional stopping.',
            'At the first simultaneous crossing T, min log wealth<=L+log(1+beta). Wald gives E S_mid=p*mu_d*ET. The echo martingale isometry gives E S_echo^2=q*ET. Hence D*ET<=L+log(1+beta)+b*sqrt(q*ET); solve the quadratic in sqrt(ET).',
            'The executable rule stops at tau=min(T_any,cap), so its actual shots and full-cap-penalized shots agree pathwise and are<=T. No premature abandonment is allowed by this theorem. Correct-stop probability is at least1-.005-ET/cap.',
            'The separate rule first waits for a target LR direction, then echo equivalence. If its final correct power>=.95, the target positive crossing event has probability>=.95, versus<=.005 under midpoint mean60/883. Stopped relative-entropy data processing gives target expected cost>=kl(.95,.005)/kl(Bern((1+mu_d)/2),Bern((1+60/883)/2)).',
            'Echo log min wealth=n*ea-eb|S|<=n*ea, so no success can occur before ceil(log100/ea) observations. The echo stage is entered with probability>=.95. Add.95 times this deterministic floor to the target lower bound. The argument concerns this specified stage architecture with no direct-target fallback.',
            'Separate upper: the target log overshoot is bounded bylog(f_plus); for echo use stopped Wald/isometry with positive drift ea and overshoot<=ea+eb. Sum the two infinite-stage upper bounds, then apply the same cap and wrong-direction bound.',
        ],
        'remaining_gates': [
            'Independent theorem-to-code review and executable same-action controls.',
            'Stronger same-action alternatives, optimized stage thresholds/allocation, mixtures, direct target fallback, and literature applicability.',
            'Freeze a fresh candidate and independent physical test cases; certify original1percent/95percent/3x and simultaneous cost lower bound>=2.',
            'No transfer of this development example into the frozen48-case operational result.',
        ],
    }


def main():
    result = compute()
    paths = [Path(__file__).resolve(), ROOT/'experiments/sota_round8/joint_midpoint_echo_control.py']
    result['source_sha256'] = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out = ROOT/'results/sota_round8/joint_midpoint_echo_certificate.json'
    out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['reference_results'], indent=2))


if __name__ == '__main__':
    main()

"""Paid fixed-stage target/echo guard, with analytic risk and power certificates.

No Monte Carlo. Planning probabilities occur only in the deterministic planner;
the runnable certificate accepts observed counts and a frozen public design.
"""
import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR, localcontext
from fractions import Fraction as F
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.circuit_validation.direct_tests import _Bounds as I


def iv(x):
    x = F(x)
    return I(x.numerator)/x.denominator


def enc(x):
    return {'lower': str(x.lo), 'upper': str(x.hi)}


def asin_positive(x):
    """Positive arcsin series with a geometric bound on the remaining tail."""
    square = x*x
    terms = []
    power = x
    for n in range(21):
        terms.append(iv(F(math.comb(2*n, n), 4**n*(2*n+1))) * power)
        power = power*square
    n = 21
    next_term = iv(F(math.comb(2*n, n), 4**n*(2*n+1))) * power
    total = sum(terms, I(0))
    # Consecutive positive-series term ratio is strictly smaller than x^2.
    tail = next_term/(1-square)
    return I(total.lo, (total+tail).hi)


def public_design(d0=F(3, 500), stake=F(7, 20)):
    update_du = 2*(asin_positive(iv(F(5,198)).sqrt())-asin_positive(iv(F(1,198)).sqrt()))
    assert Decimal('.1769') < update_du.lo < update_du.hi < Decimal('.177')
    du_upper = F(177,1000)
    h = 2*d0/(3*du_upper)
    gain, total_duration = F(6), F(6)
    chi_max, kappa_max, visibility_min = F(1,5), F(1,200), F(9,10)
    assert gain*chi_max == F(6,5)  # strictly less than pi/2
    minimum_contrast = iv(visibility_min)*(-iv(kappa_max*total_duration)).exp()
    phase = gain*h
    sine_lower = phase-phase**3/6
    threshold = minimum_contrast*iv(sine_lower)
    # A rational threshold strictly below the physical null mean makes every
    # implemented likelihood multiplier exact apart from directed log checks.
    with localcontext() as ctx:
        ctx.rounding = ROUND_FLOOR
        threshold_decimal = threshold.lo.quantize(Decimal('.000001'))
    mu0 = F(threshold_decimal)
    assert 0 < mu0 < sine_lower
    return {
        'd0': str(d0), 'stake': str(stake), 'echo_null_mean': str(mu0),
        'update_amplitude_gap_interval': enc(update_du),
        'update_gap_upper': str(du_upper), 'chi_critical': str(h),
        'echo_gain': str(gain), 'echo_duration': str(total_duration),
        'echo_control_amplitude_gap': '2', 'echo_blocks': 3,
        'chi_prior_abs_upper': str(chi_max), 'kappa_upper': str(kappa_max),
        'visibility_lower': str(visibility_min),
        'physical_null_mean_lower_interval': enc(threshold),
        'target_e_threshold': 200, 'echo_e_threshold': 100,
    }


def coefficients(design):
    d0, stake, mu0 = [F(design[k]) for k in ('d0','stake','echo_null_mean')]
    assert 0 < stake < 1/(1+d0)
    factors = {d: 1+stake*(d-d0) for d in (-1,0,1)}
    log_factors = {d: iv(value).log() for d, value in factors.items()}
    a = -iv(1-mu0*mu0).log()/2
    b = (iv(1+mu0).log()-iv(1-mu0).log())/2
    return factors, log_factors, a, b


def certificate(pair_counts, echo_plus, echo_count, design=None):
    """Count order [old0new0,old0new1,old1new0,old1new1]. No truth inputs."""
    if design is None:
        design = public_design()
    values = list(pair_counts)+[echo_plus,echo_count]
    if len(pair_counts) != 4 or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in values):
        raise ValueError('Nonnegative integer counts required')
    if echo_plus > echo_count:
        raise ValueError('Echo plus count exceeds trials')
    _, logs, a, b = coefficients(design)
    zero, minus, plus = pair_counts[0]+pair_counts[3], pair_counts[1], pair_counts[2]
    log_accept = zero*logs[0]+minus*logs[-1]+plus*logs[1]
    log_reject = zero*logs[0]+plus*logs[-1]+minus*logs[1]
    echo_sum = 2*echo_plus-echo_count
    log_echo = echo_count*a-abs(echo_sum)*b
    target_threshold, echo_threshold = iv(200).log(), iv(100).log()
    echo_pass = log_echo.lo >= echo_threshold.hi
    accept = echo_pass and log_accept.lo >= target_threshold.hi
    reject = echo_pass and log_reject.lo >= target_threshold.hi
    assert not (accept and reject)
    return {'decision': 'accept' if accept else 'reject' if reject else 'abstain',
            'log_target_accept': enc(log_accept), 'log_target_reject': enc(log_reject),
            'log_echo_equivalence': enc(log_echo), 'echo_pass': echo_pass,
            'physical_shots': 2*sum(pair_counts)+echo_count,
            'sensing_window_time': 2*sum(pair_counts)+6*echo_count,
            'scope': 'Declared ideal-echo physical contract, frozen fixed stages; every supplied outcome must be paid.'}


def fixed_plan(design, xo=F(3,100), xn=F(1,100), echo_abs_mean=F(0)):
    factors, logs, a, b = coefficients(design)
    # A fixed t=1/2 Chernoff bound on log target evidence. This is a planner,
    # never a validation certificate or access path to hidden rates.
    probabilities = {-1:(1-xo)*xn, 1:xo*(1-xn), 0:xo*xn+(1-xo)*(1-xn)}
    mgf = sum((iv(probabilities[d])/iv(factors[d]).sqrt() for d in (-1,0,1)), I(0))
    log_mgf = mgf.log()
    assert log_mgf.hi < 0
    target_log_threshold, echo_log_threshold = iv(200).log(), iv(100).log()
    failure_per_stage = F(1,40)
    needed_target = (target_log_threshold/2-iv(failure_per_stage).log())/(-log_mgf)
    pairs = int(needed_target.hi)+1
    target_failure = (target_log_threshold/2+pairs*log_mgf).exp()
    assert target_failure.hi <= Decimal('.025')

    # If |mu|<=m, P(na-b|S_n|<L) <=
    # 2 exp(-[n(a-b*m)-L]^2/(2*n*b^2)), provided the bracket is positive.
    drift = a-b*iv(echo_abs_mean)
    if drift.lo <= 0:
        return {'status': 'no_finite_power_bound_from_this_fixed_alternative',
                'echo_abs_mean_upper': str(echo_abs_mean)}
    c = b*(-2*iv(failure_per_stage/2).log()).sqrt()
    y = (c+(c*c+4*drift*echo_log_threshold).sqrt())/(2*drift)
    n_echo = int((y*y).hi)+1
    margin = n_echo*drift-echo_log_threshold
    assert margin.lo > 0
    failure_echo = 2*(-margin*margin/(2*n_echo*b*b)).exp()
    assert failure_echo.hi <= Decimal('.025')
    power = 1-target_failure-failure_echo
    assert power.lo > Decimal('.95')
    return {'status': 'finite_fixed_stage_power_bound',
            'planning_x_old': str(xo), 'planning_x_new': str(xn),
            'echo_abs_mean_upper': str(echo_abs_mean),
            'target_pairs': pairs, 'echo_shots': n_echo,
            'total_physical_shots': 2*pairs+n_echo,
            'total_sensing_window_time': 2*pairs+6*n_echo,
            'target_failure_upper': str(target_failure.hi),
            'echo_failure_upper': str(failure_echo.hi),
            'correct_accept_probability_lower': str(power.lo),
            'target_chernoff_mgf_interval': enc(mgf),
            'scope': 'Planning truth is not passed to certificate. Power is proved only for the displayed target rates and echo mean range. Reversing old/new gives the symmetric rejection case.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/sota_round8/fixed_stage_guard.json')
    args = parser.parse_args()
    design = public_design()
    plans = [fixed_plan(design, echo_abs_mean=m) for m in (F(0), F(1,50), F(1,25), F(2,25))]
    result = {
        'schema': 'faulttune.round8.fixed_stage_guard.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'finite_safe_fixed_stage_control_no_SOTA_claim',
        'public_frozen_design': design,
        'dephasing_extension': {
            'proof': 'At common duration tau, e=(1-(1-2r)exp(-kappa*tau))/2 is in[r,1/2] for 0<=r<=1/2 and kappa>=0. Write a=(1-e)^2. Then s(theta)=a-(a-1/4)sin^2(theta/2), so s>=1/4 and |ds/dtheta|<=(a-1/4)/2<=3/8. Shared kappa and equal durations therefore preserve the stated spectator bound for every kappa>=0.',
            'sign_envelope': 'For d=x_old-x_new, Delta=-s_old*d+(s_old-s_new)(1-x_new). If d>=d0 and |chi|<h=2*d0/(3*update_du_upper), Delta<0. If d<=-d0 the symmetric bound implies Delta>0.',
            'limitations': 'Different durations, arm-dependent dephasing/readout, nonfactorizing couplings or imperfect control primitives need separate envelope terms.'},
        'echo_contract': {
            'law': 'A paid phase-cycle sign measurement gives Z in{-1,+1}, E[Z]=A sin(6 chi), with A=v exp(-6 kappa), .9<=v<=1, 0<=kappa<=.005, and |chi|<=.2. The gain6 uses three two-window echoes at amplitude separation2. Public hardware controls and target update gap are distinct.',
            'null_implication': '|chi|>=h implies |E Z|>=mu0 because |6chi|<=1.2<pi/2 and sin(6h)>=6h-(6h)^3/6. mu0 is rounded downward from this certified physical bound.',
            'cost': 'Each target observation is one duration1 shot. Each echo observation is one shot containing6 sensing windows and6 ideal X1 pulses, plus preparation/analysis/readout. Time above counts sensing windows only; pulse and readout overhead are not free and must be added from the registered device-cost model.'},
        'risk_proof': [
            'Let D=X_old-X_new. Mplus=product[1+lambda(D-d0)] is a nonnegative supermartingale if d<=d0; Mminus uses -D and is valid if d>=-d0. Each crossing threshold200 has probability<=.005 on its null.',
            'For echo Z, Lplus=product[1/(1+mu0 Z)] is a supermartingale if mu>=mu0; Lminus=product[1/(1-mu0 Z)] is valid if mu<=-mu0. Echo certification requires both to reach the threshold100 at the same fixed stage, equivalently exp(n*a-b*abs(S))>=100. On the union null |mu|>=mu0 one corresponding supermartingale controls false equivalence by.01; no extra factor2 is required.',
            'Accept only if Mplus and the echo gate pass. At any fixed parameter with Delta>=0, either d<=d0 and wrong acceptance is controlled by Mplus at.005, or d>d0 and the physical envelope forces |chi|>=h, controlled by the echo gate at.01. This is a fixed-parameter partition, not a union of data-selected claims.',
            'Wrong rejection is symmetric. At Delta=0, if |d|<=d0 the union of the two target crossings is<=.01; otherwise the echo null is true and every report requires a false echo gate, also<=.01. Thus either wrong direction and any report at equality have risk<=1percent.',
            'The partition proof needs no independence between the two certificates. The registered implementation nevertheless uses separate paid observations; no truth, uncharged shared noise or initialization data are provided.'
        ],
        'finite_plans': plans,
        'power_scope': 'Fixed Chernoff/Hoeffding bounds, no Monte Carlo. The simple fixed alternative does not certify all useful updates: if echo mean approaches its equivalence boundary, this particular alternative may have nonpositive growth. A general confidence sequence or direct-target fallback is required for whole-benchmark power.',
        'same_action_comparator': 'This classical fixed-stage combination, and stronger optimized or sequential variants using exactly the same echo actions, must be included as baselines. Improvement relative to the previous restricted action menu is not same-action SOTA superiority.',
        'implementation': 'certificate(pair_counts, echo_plus, echo_count, design) has no private probability inputs and uses outward Decimal intervals for every decision.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_monte_carlo_observations': 0,
    }
    output = ROOT/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'design': design,
                      'finite_plans': plans}, indent=2))


if __name__ == '__main__':
    main()

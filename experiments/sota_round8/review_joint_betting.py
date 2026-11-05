"""Independent directed certificate for joint betting versus a named two-stage rule."""
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sota_round8.fixed_stage_guard import iv, enc
from experiments.circuit_validation.direct_tests import _Bounds as I


def sin_interval(value):
    x = iv(value)
    term = x
    total = x
    lower = None
    for n in range(1,31):
        term = -term*x*x/((2*n)*(2*n+1))
        total = total+term
        if n == 29:
            lower = total.lo
    # The alternating terms decrease on [0,1.2]. Odd partial sums are lower.
    return I(lower,total.hi)


def kl(p,q):
    return p*(p/q).log()+(1-p)*((1-p)/(1-q)).log()


def main():
    sin_h = (iv(985).sqrt()-iv(193).sqrt())/198
    mu_good = iv(F(1,50))/sin_h
    k_upper = F(3)*F(177,1000)/(2*F(883,10000))
    sin_limit = sin_interval(F(6,5))
    beta, p, q = F(3,25), F(4,7), F(3,7)
    a = iv(1-beta*beta).log()/2
    b = (iv(1+beta).log()-iv(1-beta).log())/2
    drift = a+b*iv(p)*mu_good
    assert drift.lo > 0
    max_step = iv(1+beta).log()
    l = iv(200).log()
    c = b*iv(q).sqrt()
    y = (c+(c*c+4*drift*(l+max_step)).sqrt())/(2*drift)
    joint_upper = y*y
    cap = 65536
    power = 1-iv(F(1,200))-joint_upper/cap
    assert power.lo > Decimal('.95')

    target_cut = F(60,883)
    target_information = kl((1+mu_good)/2,iv((1+target_cut)/2))
    event = kl(iv(F(95,100)),iv(F(1,200)))
    target_lower = event/target_information
    cases = []
    for duration, echo_boundary in ((F(6),F(47,400)),(F(9),F(29,250))):
        amin = iv(F(9,10))*(-iv(F(1,200)*duration)).exp()
        cone_constant = iv(k_upper*F(1,5))/(amin*sin_limit)
        assert cone_constant.hi < Decimal('.75')
        physical_echo_null = amin*sin_interval(6*F(4,177))
        assert iv(echo_boundary).hi <= physical_echo_null.lo
        ae = -iv(1-echo_boundary*echo_boundary).log()/2
        floor_interval = iv(100).log()/ae
        assert int(floor_interval.lo) == int(floor_interval.hi)
        echo_minimum = int(floor_interval.hi)+1
        baseline_lower = target_lower+iv(F(95,100)*echo_minimum)
        ratio = baseline_lower/joint_upper
        assert ratio.lo > 1
        cases.append({
            'name': 'sensing_only_T6' if duration==6 else 'scheduled_envelope_T9',
            'echo_dephasing_exposure_upper':str(duration),
            'echo_visibility_lower':enc(amin),
            'physical_sufficient_cone_constant_upper':str(cone_constant.hi),
            'margin_below_three_quarters_lower':str((iv(F(3,4))-cone_constant).lo),
            'separate_echo_boundary_mean':str(echo_boundary),
            'physical_echo_mean_lower_at_chi_cut':enc(physical_echo_null),
            'separate_echo_pathwise_minimum_count':echo_minimum,
            'separate_baseline_expected_shots_lower':str(baseline_lower.lo),
            'separate_over_joint_expected_cost_ratio_lower':str(ratio.lo),
        })
    result = {
        'schema':'faulttune.round8.joint_betting_independent_review.v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'analytic_candidate_passes_named_comparator_separation_not_SOTA',
        'joint_design':{'midpoint_probability':str(p),'echo_probability':str(q),
                        'bet_size':str(beta),'each_process_threshold':200,'shot_cap':cap},
        'good_world':{'x_old':'3/100','x_new':'1/100','chi':'0',
                      'midpoint_mean_interval':enc(mu_good),'echo_mean':'0'},
        'physical_cone_proof':[
            'For d=sin(h)*mu_d, the target envelope implies mu_d>[3*Delta_u/(2*sin(h))]*abs(chi) is sufficient for benefit; the negative mirror is sufficient for harm.',
            'On abs(chi)<=.2 and echo gain6, sin(6abs(chi))/abs(chi)>=sin(1.2)/.2 because sinc decreases on[0,1.2]. With contrast A>=Amin, abs(chi)<=.2*abs(mu_echo)/(Amin*sin1.2).',
            'Both explicitly separated noise contracts below give [3*.177/(2*.0883)]*.2/(Amin*sin1.2)<.75. Thus mu_d>.75*abs(mu_echo) implies benefit, and -mu_d>.75*abs(mu_echo) implies harm.',
        ],
        'risk_proof':[
            'At each fresh shot choose midpoint with probability4/7 or echo with probability3/7, independently of future outcomes. For direction t and echo sign e in{-1,+1}, multiply M_(t,e) by1+beta*t*Z_d on a midpoint shot, or1+beta*e*Z_echo on an echo shot.',
            'Under the wrong-direction null, t*mu_d<=.75*abs(mu_echo); choose the fixed true-parameter sign e opposite to mu_echo. Then the conditional mean multiplier is1+beta[(4/7)*t*mu_d-(3/7)*abs(mu_echo)]<=1. One process for that direction is a nonnegative supermartingale.',
            'A direction may report only when its two current processes both exceed200. Its wrong-report probability is therefore<=.005. At Delta=0 both directions are wrong, and their union is<=.01. This proof does not union over all four processes.',
        ],
        'expected_cost_proof':[
            'For the good direction at chi=0, min_e log M_(+,e)(n)=n*a+b[S_mid(n)-abs(S_echo(n))], where a=.5log(1-beta^2)<0 and b=atanh(beta)>0.',
            'Each fixed-e log process has iid bounded increments with positive mean D=a+b*(4/7)*mu_d. The first simultaneous-crossing time T has P(T>n)<=sum_e P(log M_(+,e)(n)<log200); Hoeffding gives an exponentially summable tail once nD>log200. This establishes E[T]<infinity before optional-stopping identities are applied.',
            'The stopped midpoint sum has expectation(4/7)*mu_d*E[T]. The centered echo sum is a martingale with predictable second moment(3/7)*T, hence E[S_echo(T)^2]=(3/7)*E[T] and E|S_echo(T)|<=sqrt((3/7)*E[T]).',
            'The minimum log evidence can overshoot its threshold by at most log(1+beta). Consequently D*E[T]<=log200+log(1+beta)+b*sqrt((3/7)*E[T]). The positive quadratic root gives the stated upper bound.',
            'The executed algorithm stops at the first report in either direction or at the total shot cap. Its actual cost is pathwise<=T. Wrong-direction probability<=.005 and Markov give P(correct report)>=1-.005-E[T]/cap.',
        ],
        'joint_log_drift_interval':enc(drift),
        'joint_expected_shots_upper':str(joint_upper.hi),
        'joint_correct_report_probability_lower':str(power.lo),
        'named_separate_baseline_lower_bound':{
            'contract':'The named separate rule must obtain a midpoint stage crossing200 with boundary mean60/883 and an echo equivalence stage crossing100 before any correct acceptance. It has no alternative acceptance or direct-target fallback. Its inference uses midpoint Z_d and echo Z_echo; other recorded bits do not adapt the stage rules.',
            'midpoint_stage_KL_interval':enc(target_information),
            'midpoint_stage_expected_shots_lower':str(target_lower.lo),
            'proof':'If complete good-world acceptance is>=.95, its midpoint-positive-stage event also has probability>=.95. At the midpoint mean boundary60/883 the implemented stage crossing has probability<=.005. Change of measure on that stage gives E[N_mid]>=kl(.95,.005)/KL(Bern((1+mu_good)/2)||Bern((1+cut)/2)). Echo information is unchanged in this stage-only comparison. Independently, echo min log evidence is n*a_echo-b_echo*abs(S)<=n*a_echo, so every full acceptance requires at least ceil(log100/a_echo) echo observations. Thus E[N_echo]>=.95*that floor. Add the two costs.',
            'scope':'This is an implementation-specific lower bound for the stated two-stage baseline. Its mean-cut comparison is not a target-sign-null lower bound for arbitrary physical validators. Other same-action methods, optimized separate cuts or fallback policies are not bounded by this argument.',
        },
        'separate_noise_contracts':cases,
        'scored_cost_and_cap':{
            'joint':'There is no early abstention. Without a decision, execution continues to the total cap. Consequently scored cost equals actual cost on every path, including wrong decisions, and its mean retains the joint upper bound.',
            'separate':'The proved separate lower bound is on actual observations. Charging an abstention at full budget can only increase its scored mean, so its lower bound and the comparison ratio survive.',
            'not_covered':'An implementation that stops early without a decision but is scored at cap needs a separate scored-cost calculation; the current joint rule disallows that behavior.'},
        'limits':[
            'This certificate is evaluated at the explicit good world chi=0. No claim is made of uniform95percent finite-cap power across all useful physical instances.',
            'The conservative scheduled T9 cone margin is narrow; do not silently change the noise, phase-gain, prior, control gap or analysis-primitive assumptions.',
            'The midpoint identity requires the previously reviewed absence of extra q0 noise, or an independently proved matching quadrature transfer. An arbitrary common q0 Markov channel is insufficient.',
            'Echo q2 preparation, registered ideal primitives, contrast bounds and paid cost accounting retain the physical-model limitations in echo_and_midpoint_mechanisms.json.',
            'This proves a modest improvement against the named classical separate-stage control in the declared instance, not a3x improvement, not novelty and not superiority over the strongest same-action SOTA.'
        ],
        'new_monte_carlo_observations':0,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output=ROOT/'results/sota_round8/joint_betting_independent_review.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':result['status'],'joint_expected_shots_upper':str(joint_upper.hi),
                      'correct_report_lower':str(power.lo),'cases':cases},indent=2))


if __name__=='__main__':
    main()

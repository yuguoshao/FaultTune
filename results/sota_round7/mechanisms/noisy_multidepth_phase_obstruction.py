"""Finite-risk obstruction for a concrete noisy multi-depth Ramsey family.

No Monte Carlo. Constants use rational Taylor bounds and directed Decimal
arithmetic. The old benchmark and its acceptance criteria are not modified.
"""
from decimal import Decimal
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.circuit_validation.direct_tests import _Bounds as I, _PI_LOW, _PI_HIGH


def enclosure(value):
    value = F(value)
    return I(value.numerator)/value.denominator


def interval(value):
    return [str(value.lo), str(value.hi)]


def trig(value, cosine=False):
    value = F(value)
    if value < 0:
        result = trig(-value, cosine)
        return result if cosine else -result
    if value > 1:
        raise ValueError('This alternating-series helper requires |x|<=1.')
    term = F(1) if cosine else value
    total = term
    for n in range(1, 26):
        denominator = (2*n-1)*(2*n) if cosine else (2*n)*(2*n+1)
        term *= -value*value/F(denominator)
        total += term
    # The retained final term has odd index25 and is negative, hence lower.
    n = 26
    denominator = (2*n-1)*(2*n) if cosine else (2*n)*(2*n+1)
    next_term = term*(-value*value/F(denominator))
    return I(enclosure(total).lo, enclosure(total+next_term).hi)


def main():
    phi, gamma, visibility = F(1, 384), F(1, 16), F(19, 20)
    epsilon, phase_max, gamma_max = F(1, 1000), F(1, 50), F(13, 200)
    cap = 2**20
    h2 = enclosure(phi*phi)
    g = enclosure(gamma)
    e = I(1).exp()
    log2 = I(2).log()
    pi = I(_PI_LOW, _PI_HIGH)
    series_sum = 4*log2-pi*pi/6
    info_per_gate = h2*log2/(2*e*g)
    info_per_shot = h2*series_sum/(2*e*e*g*g)
    kappa = enclosure(F(19, 20))*I(95).log()+enclosure(F(1, 20))*enclosure(F(5, 99)).log()
    lower_shots = kappa/info_per_shot
    lower_gates = kappa/info_per_gate
    lower_device = kappa*(enclosure(F(11, 5))/info_per_shot+I(1)/info_per_gate)
    delta = enclosure(visibility)*(-g).exp()*trig(phi)/2
    assert delta.lo > enclosure(epsilon).hi
    r1 = (-g).exp()
    atanh_r1 = ((1+r1)/(1-r1)).log()/2
    depth1_information = r1*atanh_r1*(1-trig(phi, True))
    depth1_lower_shots = kappa/depth1_information
    baseline_rows = []
    baseline_values = {}
    for depth, bet in ((8, F(3, 250)), (16, F(7, 500))):
        b = enclosure(bet)
        atanh_b = ((1+b)/(1-b)).log()/2
        intercept = (1-b*b).log()/2
        mu = enclosure(visibility)*(-g*depth).exp()*trig(depth*phi)
        mu_min = (enclosure(2*epsilon*depth)*
                  (-enclosure(gamma_max*(depth-1))).exp()*trig(depth*phase_max, True))
        minimum_drift = intercept+I(mu_min.lo)*atanh_b
        actual_drift = intercept+mu*atanh_b
        assert minimum_drift.lo > 0 and actual_drift.lo > 0
        expected_shots_upper = (I(200).log()+(1+b).log())/actual_drift
        deviation = minimum_drift*cap-I(200).log()
        assert deviation.lo > 0
        tail = (-deviation*deviation/(2*cap*atanh_b*atanh_b)).exp()
        correct_probability_lower = 1-enclosure(F(1, 200))-tail
        assert correct_probability_lower.lo > enclosure(F(19, 20)).hi
        gate_cost_upper = depth*expected_shots_upper
        device_cost_upper = enclosure(F(depth)+F(23, 10))*expected_shots_upper
        ratios = {
            'primary_shot_speedup_upper': interval(expected_shots_upper/lower_shots),
            'gate_use_speedup_upper': interval(gate_cost_upper/lower_gates),
            'device_time_speedup_upper': interval(device_cost_upper/lower_device)
        }
        baseline_values[depth] = expected_shots_upper
        baseline_rows.append({
            'name': 'randomized_quadrature_two_sided_SPRT_depth_%s' % depth,
            'depth': depth, 'bet_exact': str(bet), 'each_direction_threshold': 200,
            'shot_cap': cap,
            'minimum_demodulated_mean_for_all_eligible_models': interval(mu_min),
            'minimum_log_evidence_drift': interval(minimum_drift),
            'cap_non_crossing_probability_upper': interval(tail),
            'eligible_correct_report_probability_lower': interval(correct_probability_lower),
            'score_point_demodulated_mean': interval(mu),
            'score_point_expected_shots_upper': interval(expected_shots_upper),
            'score_point_expected_gate_uses_upper': interval(gate_cost_upper),
            'score_point_expected_device_time_upper': interval(device_cost_upper),
            'universal_new_method_speedup_upper_using_this_feasible_control': ratios
        })
    best_shot_ratio = baseline_values[16]/lower_shots
    best_gate_ratio = 8*baseline_values[8]/lower_gates
    best_device_ratio = enclosure(F(103, 10))*baseline_values[8]/lower_device
    assert best_shot_ratio.hi < Decimal('1.593')
    assert best_gate_ratio.hi < Decimal('1.954')
    assert best_device_ratio.hi < Decimal('2.045')
    depth1_shot_advantage = depth1_lower_shots/baseline_values[16]
    depth1_device_advantage = enclosure(F(16, 5))*depth1_lower_shots/(enclosure(F(103, 10))*baseline_values[8])
    assert depth1_shot_advantage.lo > 15
    assert depth1_device_advantage.lo > 3
    sources = (
        Path(__file__).resolve(),
        ROOT/'docs/VALIDATION_ACCEPTANCE_CRITERIA.md',
        ROOT/'experiments/acceptance_benchmark/criteria_v0.json',
        ROOT/'experiments/circuit_validation/quantum_model.py',
        ROOT/'experiments/circuit_validation/observation.py',
        ROOT/'experiments/circuit_validation/direct_tests.py'
    )
    result = {
        'status': 'analytic_and_directed_numeric_obstruction_pending_independent_review',
        'new_simulation_observations': 0,
        'relation_to_FaultTune': {
            'task': 'Accept or reject a fixed pulse update according to the actual target-circuit failure change Delta=p_fail,new-p_fail,old. This is a new explicit Ramsey/pulse-error model family, not a relabeling of the frozen Round4/5 test cases.',
            'existing_physical_connection': 'The existing circuit simulator already uses an unknown coherent Rx phase from H=(omega+u)X/2, explicit commuting phase evolution and Markov dephasing Kraus maps. A known basis change gives the single-qubit Z-phase model here. Repeating a sensing pulse changes both phase and accumulated dephasing; no Bernoulli action probabilities are assigned independently by hand.',
            'criteria_retained': 'Primary cost remains all circuit executions; risks remain1% and eligible correct-decision power95%, with epsilon=.001 from the existing grid. Gate uses and modeled device time are additional paid resource costs. The benchmark files are read only and unchanged.',
            'applicability_limit': 'The contract assumes a known phase axis, a stationary commuting dephasing model, calibrated controllable measurement phases, and a public bounded phase domain. It does not validate the original three-qubit spectator-crosstalk task or arbitrary pulse errors.'
        },
        'physical_model_and_unknown_parameters': {
            'primitive': 'U_phi=exp(-i phi Z/2) is one sensing pulse. Each pulse has the commuting dephasing channel with off-diagonal attenuation exp(-gamma). A fresh Ramsey circuit prepares |+>, appliesk sensing pulses, then measures an arbitrary known equatorial phase.',
            'unknown_domain': {'phi': '[-1/50,1/50]', 'gamma': '[3/50,13/200]',
                               'readout_alpha_0_alpha_1': 'alpha_0>=0, alpha_1>=0, alpha_0+alpha_1<=1/20',
                               'visibility': 'v=1-alpha_0-alpha_1 in[19/20,1]',
                               'bias': 'a=alpha_0+v/2; it is unknown and is never supplied to a validator'},
            'frozen_update': 'old has phase phi-pi/6, new has phase phi+pi/6 for one target pulse. The known update is delta=pi/3, fixed before validation and independent ofphi. The midpoint pulse U_phi is accessible by the public midpoint control. Known control and analysis phases are the same resource for all methods.',
            'target_probability': 'p_fail(theta)=a-(v exp(-gamma)/2)cos(theta); hence Delta=(v exp(-gamma)/2)sin(phi). Bias cancels exactly but contrast and dephasing are unknown.',
            'query_probability': 'At depthk and a chosen quadrature signS in{-1,+1}, Y|S is Bernoulli(a+S*v exp(-gamma*k)sin(k*phi)/2). This follows from the same unitary, dephasing and classical readout channel.',
            'paid_observations': 'Every delivered Y is a fresh, separately charged circuit outcome. No shared hidden noise, exact simulator probability, truephi/gamma/v, or uncharged calibration data is available.',
            'domain_prior_boundary': 'The phase interval is a declared parameter-domain assumption, not a data-derived confidence interval. An application lacking this prior must pay to localize phase and handle aliases, with the same information given to every comparator.'
        },
        'resource_contract': {
            'actions': 'All methods may choose any positive integer depthk and any equatorial analysis phase, adaptively between fresh circuits; known commuting phase corrections are allowed. The lower bound even grants arbitrary final single-copy POVMs before the same readout channel.',
            'excluded_new_resources': 'No cross-circuit quantum memory, entangled multi-probe inputs, coherent noncommuting error-correction controls, or access to a purification of the noise channel. Those would require a different model and new equal-resource baselines.',
            'primary_cost': 'N=sum1 over all executed circuits, including learning, incorrect decisions and cap abstentions.',
            'gate_cost': 'G=sumk over those circuits. One unit is one physical sensing-pulse/dephasing period; coherent amplification is never a free query.',
            'device_cost': 'T=2.2*N+G+0.1*S_switch, following the existing preparation/readout2.2 and configuration-switch0.1 convention. A randomized-quadrature baseline hasT<=sum(k+2.3), whereas every method hasT>=2.2*N+G.',
            'budget': 'Both explicit baselines use the common finite capN=2^20, already an existing default shot-budget scale. They execute at most16*N sensing pulses. A future comparison must also freeze the allowed depth/menu and any device-time cap; the present information lower bound is stronger in scope because it allows unrestricted depths.',
            'abstention_penalty': 'The policies only abstain upon exhausting the full shot cap, so actual and full-cap-penalized primary costs agree. No unresolved run is dropped.'
        },
        'strong_comparator_policy': {
            'family': 'Include robust multi-depth Ramsey/phase-estimation and frequency-learning methods with identical depths, nuisance model and risk guarantee. For the impossibility result it suffices to exhibit the two valid fixed-depth controls below: the strongest qualifying comparator is never more expensive than either one.',
            'no_straw_man': 'A plain depth1 A/B control is not the strongest same-resource comparator after powersU^k are allowed. Multi-depth controls also receive midpoint pulses, randomized quadrature debiasing, and sign-specific valid stopping.',
            'literature_anchor_pending_detail': 'Kimmel, Low and Yoder, Robust calibration of a universal single-qubit gate set via robust phase estimation, Phys.Rev.A92,062315(2015), arXiv1502.02677. Its noiseless/robust phase-estimation rates must not be applied unmodified to depth-dependent dephasing. This file does not claim to reproduce that algorithm.',
            'executable_mathematical_policy': 'At fixedk draw independent fairS, observeY, setZ=S(2Y-1). UpdateM_plus*=1+bZ andM_minus*=1-bZ. Starting from1, stop at the first value>=200: plus rejects the update, minus accepts it. If neither crosses by2^20 observations, abstain. Use(k,b)=(8,3/250) or(16,7/500). All constants are public and independent of the unknown model.'
        },
        'proofs': [
            {'item': 'Exact nuisance-bias removal and sign safety',
             'argument': 'P[Z=+1]=[P(Y=1|S=+1)+P(Y=0|S=-1)]/2=(1+mu_k)/2, mu_k=v exp(-gamma*k)sin(k*phi), independent ofa. Sincek<=16 and|phi|<=.02, sin(k*phi) has the sign ofphi and henceDelta. UnderDelta<=0, M_plus is a nonnegative supermartingale; underDelta>=0, M_minus is one. Ville gives each wrong-direction probability<=1/200. AtDelta=0 the union of both directions is<=1/100, satisfying the original1% requirement.'},
            {'item': 'Uniform power with a finite cap and unknown gamma/visibility',
             'argument': 'For |Delta|>=epsilon, |mu_k|=2|Delta|exp(-gamma(k-1))*sin(k|phi|)/sin(|phi|). On0<|phi|<=H withkH<pi/2, sin(k|phi|)>=k|phi|cos(kH) and sin(|phi|)<=|phi|. Hence|mu_k|>=mu_min=2epsilon*k exp(-gamma_max(k-1))cos(kH). The correct log process is a sum of iid increments a_b+atanh(b)Z, a_b=log(1-b^2)/2, with drift at leastd_min=a_b+mu_min*atanh(b)>0. Fornd_min>log200, Hoeffding gives P(no correct crossing byn)<=exp(-(nd_min-log200)^2/[2n atanh(b)^2]). Adding the wrong-direction bound1/200 gives the reported uniform correct-decision lower bound. No Monte Carlo or posterior credible interval is used.'},
            {'item': 'Finite expected baseline cost including cap outcomes',
             'argument': 'Consider the one-sided correct crossing time tau_plus without a cap. Positive bounded-increment drift and the preceding exponential tail make its expectation finite. Wald identity and overshoot at mostlog(1+b) yieldE[tau_plus]<=(log200+log(1+b))/d_actual. The actual two-sided capped time is no larger than tau_plus pathwise, so this upper bound includes all outcomes. Multiplying byk or(k+2.3) gives paid gate/device upper bounds. Mirrored negativephi cases have exactly the same demodulated stopping-time distribution.'},
            {'item': 'All-depth information bound from the same physical noisy state',
             'argument': 'Before readout, rho_phi,k=(I+r_k[cos(kphi)X+sin(kphi)Y])/2, r_k=exp(-gamma*k). Use the single legal nullphi=0 with the same unknown gamma and readout parameters. Direct2x2 spectral algebra gives D(rho_phi,k||rho_0,k)=r_k atanh(r_k)(1-cos(kphi)). Any allowed measurement and the same noisy readout have classical KL no larger by data processing. This is one common null across all depths; no depth-dependent nuisance null is silently substituted.'},
            {'item': 'Uniform information per paid gate',
             'argument': 'r atanh(r)=sum_{n>=0}r^(2n+2)/(2n+1). Withx=gamma*k and1-cos(kphi)<=(kphi)^2/2, D_k/k<=[phi^2/(2gamma)]x exp(-x)atanh(exp(-x)). Eachx exp(-(2n+2)x) has maximum1/[e(2n+2)]. Summing the resulting positive terms gives sum1/[(2n+1)(2n+2)]=log2. ThereforeD_k/k<=phi^2 log2/(2e gamma) for EVERY positive depth, with no small-phi approximation.'},
            {'item': 'Uniform information per circuit execution',
             'argument': 'SimilarlyD_k<=[phi^2/(2gamma^2)]x^2 exp(-x)atanh(exp(-x)). The maximum ofx^2 exp(-(2n+2)x) is1/[e^2(n+1)^2]. The resulting sum is sum_{m>=1}1/[m^2(2m-1)]=4log2-pi^2/6, from1/[m^2(2m-1)]=4/(2m-1)-2/m-1/m^2. ThusD_k<=phi^2(4log2-pi^2/6)/(2e^2 gamma^2) at every depth.'},
            {'item': 'Sequential finite-risk lower bounds, including abstention',
             'argument': 'For any predictable sequence of allowed actions and any finite-budget decision, the chain rule bounds transcript KL byI_shot*E[N] and also byI_gate*E[G]. On the event of a correct beneficial acceptance or harmful rejection, its probability is>=.95 at the scored model and<=.01 at the common nullphi=0. Binary data processing therefore gives transcript KL>=kappa=kl(.95||.01), even with cap abstention. HenceE[N]>=kappa/I_shot, E[G]>=kappa/I_gate, andE[T]>=kappa(2.2/I_shot+1/I_gate). These bounds hold with arbitrary adaptive depth choices and all accessible nuisance-learning observations charged.'},
            {'item': 'Exclusion relative to the strongest qualifying control',
             'argument': 'For a proposed new verifier A meeting the same error and power obligations, C_A is at least the appropriate information lower bound. The strongest qualifying comparator cost C_best is at most the displayed feasible fixed-depth upper bound. ThusC_best/C_A<=C_control_upper/C_information_lower. The certified ratios below are all<3 at both mirrored score points. Since the mean-ratio geometric mean over these two points is also<3, this declared two-point model experiment cannot meet the original3x criterion, even if a stronger comparator is subsequently added.'}
        ],
        'score_points': {
            'exact_parameters': {'phi': ['-1/384', '1/384'], 'gamma': '1/16',
                                 'alpha_0': '3/100', 'alpha_1': '1/50', 'v': '19/20'},
            'absolute_target_failure_change_interval': interval(delta),
            'same_cost_by_mirror': True,
            'kappa_interval': interval(kappa),
            'all_depth_information_per_shot_upper': interval(info_per_shot),
            'all_depth_information_per_gate_upper': interval(info_per_gate),
            'any_valid_verifier_expected_shots_lower': interval(lower_shots),
            'any_valid_verifier_expected_gate_uses_lower': interval(lower_gates),
            'any_valid_verifier_expected_device_time_lower': interval(lower_device)
        },
        'qualified_baseline_bounds': baseline_rows,
        'maximum_possible_new_method_speedup_vs_strongest_qualified_comparator': {
            'primary_shots': interval(best_shot_ratio),
            'paid_gate_uses': interval(best_gate_ratio),
            'modeled_device_time': interval(best_device_ratio),
            'rounded_strict_upper_bounds': {'primary_shots': '1.593', 'paid_gate_uses': '1.954', 'modeled_device_time': '2.045'},
            'threefold_target': 'Excluded for this declared family score pair by a finite-error information bound and a feasible same-resource comparator.'
        },
        'amplification_really_changes_the_depth1_resource_limit': {
            'depth1_quantum_information_interval': interval(depth1_information),
            'every_valid_depth1_method_expected_shots_lower': interval(depth1_lower_shots),
            'depth16_feasible_shot_saving_vs_every_depth1_method_lower': interval(depth1_shot_advantage),
            'depth8_feasible_device_time_saving_vs_every_depth1_method_lower': interval(depth1_device_advantage),
            'interpretation': 'Coherent repetition is a genuine resource improvement here: it supports>15x fewer shots and>3x modeled device-time saving against the restricted depth1 class. Both benefits are shared by existing multi-depth/Ramsey methods and cannot be advertised as FaultTune beating the strongest same-resource comparator.'
        },
        'decision': 'Do not launch a Monte Carlo campaign seeking3x vs equal-resource multi-depth controls on this narrow noisy-phase family. The phase-amplification mechanism is useful relative to depth1 but its gain is already accessible to simple qualified controls. To pursue3x, identify a task structure or resource that supplies information unavailable to all such equally equipped controls, and rederive the null/cost model.',
        'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'claim_boundary': 'This is a concrete proposed model-family feasibility obstruction, pending independent review. It does not claim that all noisy phase-estimation tasks lack3x opportunities, does not modify the frozen acceptance benchmark, and is not a SOTA comparison or full RPE reproduction.'
    }
    output = Path(__file__).with_suffix('.json')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                      'status': result['status'],
                      'speedup_upper_bounds': result['maximum_possible_new_method_speedup_vs_strongest_qualified_comparator'],
                      'depth1_comparison': result['amplification_really_changes_the_depth1_resource_limit']}, indent=2))


if __name__ == '__main__':
    main()

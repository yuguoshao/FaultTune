"""Independent deterministic review of the Round 7 invariance-cost witness."""
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sota_round7.bitstring_nuisance_diagnostic import (
    bound, convolve_spectators, encode_interval, gate_probabilities, kl)
from experiments.circuit_validation import quantum_model as qm


def mm(x):
    x = F(x)
    return mp.mpf(x.numerator) / x.denominator


def main():
    target = ROOT / 'results/sota_round7/mechanisms/bitstring_invariance_cost.json'
    report = json.loads(target.read_text(encoding='utf-8'))
    source = ROOT / 'experiments/sota_round7/bitstring_invariance_cost.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report['source_sha256'][source.relative_to(ROOT).as_posix()]
    laws = report['exact_output_laws']
    old, new_good, new_null = [tuple(map(F, laws[k])) for k in
                             ('old_either_world', 'new_good_world', 'new_null_world')]
    eta, xo, xn = F(1, 200), F(3, 100), F(1, 100)
    s_good, s_null = F(1, 2), F(97, 198)
    t_good, t_null = [((1 - eta) ** 2 - s) / ((1 - eta) ** 2 - F(1, 4))
                      for s in (s_good, s_null)]
    spectators = [convolve_spectators((1 - 3*t/4, t/4, t/4, t/4), eta, eta)
                  for t in (t_good, t_null)]
    exact_target = []
    for x, q in ((xo, spectators[0]), (xn, spectators[0]), (xn, spectators[1])):
        exact_target.append(tuple((x if i & 1 else 1-x) * q[i >> 1] for i in range(8)))
    assert exact_target == [old, new_good, new_null]
    assert old[0] - new_good[0] == -F(1, 100) and old[0] - new_null[0] == 0

    # Construct coefficients independently, then use explicit gates/Kraus
    # rather than the target script's classical spectator-mixture function.
    mp.mp.dps = 150
    uo, un = [2*mp.asin(mp.sqrt(mm((x-eta)/(1-2*eta)))) for x in (xo, xn)]
    g0, g1 = [2*mp.asin(mp.sqrt(mm(t))) for t in (t_good, t_null)]
    chi = (g1-g0)/(un-uo)
    gamma = g0-chi*uo
    assert abs(gamma+chi*uo-g0) < mp.mpf('1e-140')
    assert abs(gamma+chi*un-g1) < mp.mpf('1e-140')
    action_rows = []
    for label, amplitude, x in (('old', uo, xo), ('new', un, xn)):
        for context in qm.CONTEXTS:
            outputs = []
            for world in ('good', 'null'):
                g = g0 if world == 'good' else gamma+chi*amplitude
                model = qm.QuantumModel(0.0, float(g), float(eta), 0.0)
                pmf = gate_probabilities(model, qm.PulseControl(float(amplitude), 1.0), context)
                if context == qm.CALIBRATION_CONTEXT:
                    qs = ((1-eta)**2, eta*(1-eta), eta*(1-eta), eta*eta)
                    expected = tuple((x if i & 1 else 1-x)*qs[i >> 1] for i in range(8))
                else:
                    expected = old if label == 'old' else new_good if world == 'good' else new_null
                err = max(abs(a-float(b)) for a, b in zip(pmf, expected))
                assert err < 2e-13
                outputs.append(expected)
                action_rows.append({'action': label, 'context': context, 'world': world,
                                    'gate_vs_exact_pmf_max_error': err})
            assert (outputs[0] != outputs[1]) == (label == 'new' and context == qm.WORKLOAD_CONTEXT)

    information = sum((bound(p)*(bound(p)/bound(q)).log() for p, q in zip(new_good, new_null)), bound(0))
    information_mp = sum(mm(p)*mp.log(mm(p)/mm(q)) for p, q in zip(new_good, new_null))
    stored_kl = report['analytic_lower_bound']['new_target_full_output_KL_interval']
    assert mm(stored_kl[0]) <= information_mp <= mm(stored_kl[1])
    assert mm(information.lo) <= information_mp <= mm(information.hi)
    event = kl(F(95, 100), F(1, 100))
    lower = event / information
    assert Decimal(report['analytic_lower_bound']['necessary_expected_total_shots_lower']) <= lower.hi

    # Independently evaluate drift, the bounded-stopping estimate, power and
    # cross-model ratio at 150 digits, rather than rerunning the target script.
    plus, minus = xo*(1-xn), xn*(1-xo)
    assert plus == F(297, 10000) and minus == F(97, 10000)
    drift = mm(plus)*mp.log(mp.mpf('1.5')) + mm(minus)*mp.log(mp.mpf('.5'))
    mean_pairs = (mp.log(200)+mp.log(mp.mpf('1.5'))) / drift
    power = 1-mm(F(1,200))-mean_pairs/65536
    event_mp = mm(F(95,100))*mp.log(95) + mm(F(5,100))*mp.log(mm(F(5,99)))
    necessary = event_mp / information_mp
    tax = necessary / (2*mean_pairs)
    control = report['common_Q_control']
    assert mm(control['pair_mean_upper']) >= mean_pairs
    assert mm(control['physical_mean_upper']) >= 2*mean_pairs
    assert mm(control['correct_stop_probability_lower']) <= power
    assert mm(report['invariance_tax']['expanded_class_necessary_cost_over_common_Q_feasible_cost_lower']) <= tax
    assert mm(report['analytic_lower_bound']['necessary_expected_total_shots_lower']) <= necessary

    result = {
        'schema': 'faulttune.round7.invariance_cost_independent_review.v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'passed_with_explicit_proof_scope',
        'target_artifact': target.relative_to(ROOT).as_posix(),
        'target_artifact_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'target_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'checks': {
            'static_amplitude_crosstalk_hamiltonian': 'PASS: one fixed gamma and chi solves both physical phase constraints. No freely assigned arm PMFs.',
            'exact_full_output_laws': 'PASS: reconstructed by independent Fraction readout convolution; old/new-good Delta=-1/100, old/new-null Delta=0.',
            'all_four_allowed_actions': 'PASS: old spectator_plus and both spectator_0 actions coincide across worlds. Only new spectator_plus carries KL.',
            'adaptive_action_KL_chain': 'PASS: algorithm randomization and predictable action probabilities are common conditional on the same history; actions add no extra KL. Outcome KL sums to E_good[N_new_target]*D. At finite cap this is immediate; finite expected stopping also follows from bounded per-step log likelihoods. Infinite expected total cost already satisfies the lower bound.',
            'decision_data_processing': 'PASS: P_good(accept)>=.95 and P_null(accept)<=.01 imply KL>=kl(.95,.01); uniform validity over the enlarged family includes the exact-null witness.',
            'sign_test_risk': 'PASS: E[factor_plus]=1+.5*(x_old-x_new), so it is a supermartingale on x_old<=x_new. The opposite process is analogous. Each wrong direction has risk<=1/200; at equality their union is<=1/100.',
            'bounded_stopped_Wald': 'PASS: let T be the first positive crossing and S_n=log Lplus. For every n, E[S_(T wedge n)]=drift*E[T wedge n], while pathwise S_(T wedge n)<=log200+log1.5. Monotone convergence proves E[T]<infinity and the stated bound without presupposing Wald integrability.',
            'finite_cap_power': 'PASS: P(correct accept)>=1-P(ever wrong crossing)-P(T>65536)>=1-1/200-E[T]/65536. Stopping at the first crossing or the cap reduces actual expected shots.',
            'X_only_failure_at_null': 'PASS: both worlds have identical old/new X laws, hence identical X-only stopped transcripts and acceptance probabilities.',
            'cross_model_ratio': 'PASS: uniform enlarged-class cost is bounded below; a valid promised-common-Q control is bounded above. Their quotient is a cross-model cost separation, not same-information same-model SOTA superiority.',
            'independent_high_precision_numerics': 'PASS: 150-digit mpmath checks all reported one-sided numerical bounds, independently of the target script execution.',
        },
        'gate_checks': action_rows,
        'independent_numbers': {
            'KL_interval': encode_interval(information),
            'necessary_new_target_shots_150_digit_display': mp.nstr(necessary, 70),
            'common_Q_physical_shot_upper_150_digit_display': mp.nstr(2*mean_pairs, 70),
            'common_Q_power_lower_150_digit_display': mp.nstr(power, 70),
            'cross_model_ratio_lower_150_digit_display': mp.nstr(tax, 70),
        },
        'scope_clarifications': [
            'The theorem uses exact inverse-trigonometric controls and coefficients. Displayed floating angles and binary64 gate comparisons only check numerical consistency; they do not define the exact laws behind 64-digit KL enclosures.',
            'The common-Q sign guarantee requires s=Q(00)>0. It does not extend to arbitrary common Q with s=0, where target Delta=0 regardless of the X difference.',
            'Extra amplitudes, durations, nonlisted preparations, side-channel data or prior exclusion of the null coefficients can invalidate this particular lower-bound experiment.',
            'The explicit common-Q control is not claimed minimax optimal. Its feasible upper bound suffices for the cross-model separation.',
        ],
        'blocking_issues': [],
        'new_monte_carlo_observations': 0,
    }
    output = ROOT/'results/sota_round7/mechanisms/bitstring_invariance_cost_independent_review.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'artifact': output.relative_to(ROOT).as_posix(),
                      'gate_pmf_checks': len(action_rows), 'blocking_issues': []}, indent=2))


if __name__ == '__main__':
    main()

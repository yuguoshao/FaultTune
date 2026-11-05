"""Physical crosstalk witness for the cost of certifying spectator invariance.

No sampling. Exact rational output laws, analytic Hermitian circuit models,
and directed KL enclosures separate a useful common-nuisance shortcut from
a validator that must also tolerate amplitude-dependent spectator crosstalk.
"""
from decimal import Decimal
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

import mpmath as mp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.circuit_validation.direct_tests import _Bounds as I


def iv(x):
    x = F(x)
    return I(x.numerator)/x.denominator


def bounds(x):
    return [str(x.lo), str(x.hi)]


def relative_entropy(p, q):
    assert len(p) == len(q) and sum(p) == sum(q) == 1
    assert all(v > 0 for v in p+q)
    return sum((iv(a)*(iv(a)/iv(b)).log() for a, b in zip(p, q)), I(0))


def spectator(s, readout):
    """Full two-bit PMF after a controlled phase and independent readout flips."""
    q = 1-readout
    base = [q*q, q*readout, q*readout, readout*readout]
    t = (q*q-s)/(q*q-F(1, 4))
    assert 0 < t < 1
    result = [b+t*(F(1, 4)-b) for b in base]
    assert result[0] == s and sum(result) == 1
    return t, result


def full_pmf(x, spec):
    return [probability*(x if bit else 1-x) for probability in spec for bit in (0, 1)]


def mm(x):
    x = F(x)
    return mp.mpf(x.numerator)/x.denominator


def main():
    output = ROOT/'results/sota_round7/mechanisms/bitstring_invariance_cost.json'
    if output.exists():
        raise FileExistsError('Preserve the completed witness')
    r, xo, xn, s = F(1, 200), F(3, 100), F(1, 100), F(1, 2)
    s_null = s*(1-xo)/(1-xn)
    t_good, q_good = spectator(s, r)
    t_null, q_null = spectator(s_null, r)
    old = full_pmf(xo, q_good)
    good_new = full_pmf(xn, q_good)
    null_new = full_pmf(xn, q_null)
    delta_good = old[0]-good_new[0]
    delta_null = old[0]-null_new[0]
    assert s_null == F(97, 198) and delta_good == F(-1, 100) and delta_null == 0
    information = relative_entropy(good_new, null_new)
    spec_information = relative_entropy(q_good, q_null)
    assert max(information.lo, spec_information.lo) <= min(information.hi, spec_information.hi)
    kappa = relative_entropy([F(95, 100), F(5, 100)], [F(1, 100), F(99, 100)])
    necessary = I(kappa.lo)/I(information.hi)

    # Explicit common-Q control: fixed alternative q=.75 conditional McNemar
    # LR, one chance to stop after every independent old/new physical pair.
    positive, negative = xo*(1-xn), xn*(1-xo)
    drift = iv(positive)*iv(F(3, 2)).log()+iv(negative)*iv(F(1, 2)).log()
    assert drift.lo > 0
    pair_mean_upper = (iv(200).log()+iv(F(3, 2)).log())/I(drift.lo)
    physical_mean_upper = 2*pair_mean_upper
    cap_pairs = 65536
    power_lower = 1-iv(F(1, 200))-pair_mean_upper/cap_pairs
    assert power_lower.lo > Decimal('.95')
    cost_separation_lower = I(necessary.lo)/I(physical_mean_upper.hi)

    mp.mp.dps = 120
    u_old = 2*mp.asin(mp.sqrt(mm((xo-r)/(1-2*r))))
    u_new = 2*mp.asin(mp.sqrt(mm((xn-r)/(1-2*r))))
    gamma_good = 2*mp.asin(mp.sqrt(mm(t_good)))
    gamma_new = 2*mp.asin(mp.sqrt(mm(t_null)))
    chi_null = (gamma_new-gamma_good)/(u_new-u_old)
    gamma_null = gamma_good-chi_null*u_old
    assert abs(gamma_null+chi_null*u_old-gamma_good) < mp.mpf('1e-100')
    assert abs(gamma_null+chi_null*u_new-gamma_new) < mp.mpf('1e-100')
    for pmf, reference in ((q_good, gamma_good), (q_null, gamma_new)):
        t = mp.sin(reference/2)**2
        base = [mm((1-r)**2), mm(r*(1-r)), mm(r*(1-r)), mm(r*r)]
        predicted = [b+t*(mp.mpf('.25')-b) for b in base]
        assert max(abs(a-mm(b)) for a, b in zip(predicted, pmf)) < mp.mpf('1e-100')
    numerical_kl = sum(mm(a)*mp.log(mm(a)/mm(b)) for a, b in zip(good_new, null_new))
    assert mm(information.lo) <= numerical_kl <= mm(information.hi)
    result = {
        'status': 'complete_physical_null_witness_and_finite_cost_obstruction',
        'new_simulation_observations': 0,
        'resource_contract': {
            'actions': 'Only the two frozen amplitudes, old and new, with common duration1, and the existing spectator_plus or spectator_0 preparations. Each execution returns the complete three-bit outcome and costs one. Fresh shot noise is independent; no hidden-noise coupling.',
            'expanded_hamiltonian': 'H(u)=((omega+u)/2)X_0+(gamma+chi*u)P11_(1,2). Unknown fixed real omega,gamma,chi; fixed Markov/readout nuisance may also be unknown. The witness uses omega=0,kappa=0,readout=1/200 in both worlds.',
            'good_world': 'chi=0, gamma=2 asin(sqrt(t_good)); spectators are exactly invariant under the two amplitudes.',
            'null_world': 'chi=(2 asin(sqrt(t_null))-2 asin(sqrt(t_good)))/(u_new-u_old), gamma=2 asin(sqrt(t_good))-chi*u_old. All coefficients are fixed before observations, so this is a single amplitude-dependent Hamiltonian, not freely assigned arm probabilities.',
            'shared_information': 'Controls, target task, complete bitstrings and the declared enlarged Hamiltonian family are identical for both worlds and all algorithms. Hidden omega/gamma/chi are never supplied to the validator.',
            'not_covered': 'Additional amplitudes, durations, spectator probes, known chi=0, or a sufficiently restrictive prior bound on chi change the information experiment; this lower bound does not cover them.'
        },
        'exact_output_laws': {
            'readout_flip': str(r), 'x_old': str(xo), 'x_new': str(xn),
            's_good_both': str(s), 's_null_new': str(s_null),
            't_good': str(t_good), 't_null_new': str(t_null),
            'old_either_world': list(map(str, old)),
            'new_good_world': list(map(str, good_new)),
            'new_null_world': list(map(str, null_new)),
            'delta_good': str(delta_good), 'delta_null': str(delta_null),
            'index_order': 'x+2*spectator_index; success is outcome000',
            'factorization': 'q0 is independent of the spectator pair because the two Hamiltonian terms commute on disjoint subsystems; independent readout preserves factorization. Spectator_0 is blind to P11 in either world.'
        },
        'analytic_lower_bound': {
            'new_target_full_output_KL_interval': bounds(information),
            'spectator_KL_interval_independent_factorization_check': bounds(spec_information),
            'kappa_kl_095_001_interval': bounds(kappa),
            'necessary_expected_new_target_shots_lower': str(necessary.lo),
            'necessary_expected_total_shots_lower': str(necessary.lo),
            'proof': 'Old and spectator_0 laws coincide between good and null worlds. Only new spectator_plus shots contribute KL=D(Q_good||Q_null). For any predictable randomized allocation and finite stopping time, transcript KL=E_good[N_new_target]*D. Data processing of accept implies transcript KL>=kl(.95,.01). Therefore E_good[N_total]>=E_good[N_new_target]>=kappa/D.',
            'scope': 'This is a necessary cost at the good world for a validator uniformly1percent-risk-valid over the expanded physical crosstalk class, with95percent correct acceptance at that good world. It is not a lower bound for a validator allowed to assume chi=0.'
        },
        'common_Q_control': {
            'rule': 'Observe old/new X0 outcomes with independent shot noise; multiply L+ by1.5 on(old1,new0),by.5 on(old0,new1),and by1 otherwise. L- swaps those factors. Accept/reject on the respective first crossing of200; stop at cap if neither crosses.',
            'risk': 'When x_old<=x_new, L+ is a nonnegative supermartingale; when x_old>=x_new, L- is. Under equality their union crossing probability is<=.01. With common spectator success s>0 these are exactly the target sign nulls.',
            'pair_mean_upper': str(pair_mean_upper.hi),
            'physical_mean_upper': str(physical_mean_upper.hi),
            'physical_cap': 2*cap_pairs,
            'correct_stop_probability_lower': str(power_lower.lo),
            'cost_and_power_proof': 'In the good world, log L+ has the displayed positive mean increment and positive overshoot<=log1.5. Stopped Wald gives E[T+] <=(log200+log1.5)/drift. Stopping at either sign and capping reduce expected cost. Wrong-sign probability<=.005 and Markov cap abstention<=E[T+]/cap_pairs give the correct-stop lower bound.',
            'scope': 'Classical fixed-alternative conditional McNemar/SPRT control, not a new FaultTune method or full-field SOTA winner.'
        },
        'invariance_tax': {
            'expanded_class_necessary_cost_over_common_Q_feasible_cost_lower': str(cost_separation_lower.lo),
            'exact_acceptance_failure_when_ignoring_spectators': 'The entire X0-only history has the identical law in both worlds. Thus the common-Q control accepts the null world with probability at least its good-world correct-stop lower bound, exceeding95percent. It cannot be promoted as robust to unknown amplitude crosstalk.',
            'interpretation': 'The large bitstring gain depends materially on the model promise. Uniform validity over a larger physically plausible family can force an order-of-magnitude larger cost even at a truly invariant point. This is a cross-model cost separation, not a same-model algorithm speedup.'
        },
        'high_precision_physical_parameters_diagnostic': {
            'digits': 120, 'u_old': mp.nstr(u_old, 80), 'u_new': mp.nstr(u_new, 80),
            'gamma_good': mp.nstr(gamma_good, 80),
            'gamma_null': mp.nstr(gamma_null, 80), 'chi_null': mp.nstr(chi_null, 80),
            'scope': 'Angle decimals only verify the analytic exact inverse-trigonometric construction. Rational PMFs and directed KL bounds carry the cost statement.'
        },
        'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).resolve(), ROOT/'experiments/circuit_validation/direct_tests.py')},
        'claim_boundary': 'No Monte Carlo, no SOTA gain, no reclassification of an old failed candidate, and no claim that all full-bitstring mechanisms fail. The useful invariant-family shortcut and the expanded-family obstruction are distinct statements.'
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'status': result['status'],
                      'expanded_class_shot_lower': str(necessary.lo),
                      'common_Q_shot_upper': str(physical_mean_upper.hi),
                      'common_Q_power_lower': str(power_lower.lo),
                      'invariance_cost_ratio_lower': str(cost_separation_lower.lo)}, indent=2))


if __name__ == '__main__':
    main()

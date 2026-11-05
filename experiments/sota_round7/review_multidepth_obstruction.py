"""Independent root proof and numerical audit of the noisy-depth screen."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

import mpmath as mp
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]


def scalar(x):
    x = F(x)
    return mp.mpf(x.numerator)/x.denominator


def contains(interval, x):
    return scalar(interval[0]) <= x <= scalar(interval[1])


def main():
    output = ROOT/'results/sota_round7/mechanisms/noisy_multidepth_root_review.json'
    if output.exists():
        raise FileExistsError('Preserve completed review')
    source = ROOT/'results/sota_round7/mechanisms/noisy_multidepth_phase_obstruction.json'
    closure_path = ROOT/'results/sota_round7/mechanisms/noisy_multidepth_phase_budget_closure.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    closure = json.loads(closure_path.read_text(encoding='utf-8'))
    checked_hashes = 0
    for payload in (data, closure):
        for rel, expected in payload['source_sha256'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == expected
            checked_hashes += 1
    assert closure['main_source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    mp.mp.dps = 130
    phi, gamma, v = scalar(F(1, 384)), scalar(F(1, 16)), scalar(F(19, 20))
    kappa = scalar(F(19, 20))*mp.log(95)+scalar(F(1, 20))*mp.log(scalar(F(5, 99)))
    sum_two = 4*mp.log(2)-mp.pi**2/6
    igate = phi**2*mp.log(2)/(2*mp.e*gamma)
    ishot = phi**2*sum_two/(2*mp.e**2*gamma**2)
    n_lower, g_lower = kappa/ishot, kappa/igate
    t_lower = scalar(F(11, 5))*n_lower+g_lower
    expected = {
        'absolute_target_failure_change_interval': v*mp.exp(-gamma)*mp.sin(phi)/2,
        'kappa_interval': kappa,
        'all_depth_information_per_shot_upper': ishot,
        'all_depth_information_per_gate_upper': igate,
        'any_valid_verifier_expected_shots_lower': n_lower,
        'any_valid_verifier_expected_gate_uses_lower': g_lower,
        'any_valid_verifier_expected_device_time_lower': t_lower
    }
    for field, value in expected.items():
        assert contains(data['score_points'][field], value), field
    m = sp.symbols('m', positive=True)
    assert sp.simplify(1/(m*m*(2*m-1))-(4/(2*m-1)-2/m-1/(m*m))) == 0
    x, rate = sp.symbols('x rate', positive=True)
    assert sp.simplify(sp.diff(x*sp.exp(-rate*x), x).subs(x, 1/rate)) == 0
    assert sp.simplify(sp.diff(x*x*sp.exp(-rate*x), x).subs(x, 2/rate)) == 0
    matrix_rows = []
    for depth in (1, 8, 16, 97):
        r, angle = mp.exp(-gamma*depth), phi*depth
        rho = mp.matrix([[(1+r*mp.cos(angle))/2, r*mp.sin(angle)/2],
                         [r*mp.sin(angle)/2, (1-r*mp.cos(angle))/2]])
        eigenvalues, vectors = mp.eigsy(rho)
        log_rho = vectors*mp.diag([mp.log(x) for x in eigenvalues])*vectors.T
        log_null = mp.diag([mp.log((1+r)/2), mp.log((1-r)/2)])
        product = rho*(log_rho-log_null)
        matrix_kl = product[0, 0]+product[1, 1]
        formula = r*mp.atanh(r)*(1-mp.cos(angle))
        assert abs(matrix_kl-formula) < mp.mpf('1e-120')
        assert matrix_kl <= ishot and matrix_kl/depth <= igate
        matrix_rows.append({'depth': depth, 'KL_formula_error': mp.nstr(abs(matrix_kl-formula), 15)})
    baseline_rows = []
    means = {}
    cap = 2**20
    for row in data['qualified_baseline_bounds']:
        depth, b = row['depth'], scalar(row['bet_exact'])
        mu = v*mp.exp(-gamma*depth)*mp.sin(depth*phi)
        mu_min = 2*scalar(F(1, 1000))*depth*mp.exp(-scalar(F(13, 200))*(depth-1))*mp.cos(scalar(F(depth, 50)))
        drift = mp.log(1-b*b)/2+mu*mp.atanh(b)
        drift_min = mp.log(1-b*b)/2+mu_min*mp.atanh(b)
        mean = (mp.log(200)+mp.log(1+b))/drift
        tail = mp.exp(-(cap*drift_min-mp.log(200))**2/(2*cap*mp.atanh(b)**2))
        assert drift_min > 0
        assert contains(row['score_point_demodulated_mean'], mu)
        assert contains(row['score_point_expected_shots_upper'], mean)
        assert scalar(row['cap_non_crossing_probability_upper'][1]) >= tail
        assert scalar(row['eligible_correct_report_probability_lower'][0]) <= 1-scalar(F(1, 200))-tail
        assert scalar(row['eligible_correct_report_probability_lower'][0]) > scalar(F(19, 20))
        assert depth*cap <= closure['common_budgets']['sensing_pulse_uses']
        assert F(10*depth+23, 10)*cap <= F(closure['common_budgets']['modeled_device_time_exact'])
        means[depth] = mean
        baseline_rows.append({'depth': depth, 'independent_mean_cost_bound': mp.nstr(mean, 80),
                              'independent_uniform_power_lower': mp.nstr(1-scalar(F(1, 200))-tail, 80)})
    ratios = {'primary_shots': means[16]/n_lower,
              'paid_gate_uses': 8*means[8]/g_lower,
              'modeled_device_time': scalar(F(103, 10))*means[8]/t_lower}
    for name, value in ratios.items():
        assert contains(data['maximum_possible_new_method_speedup_vs_strongest_qualified_comparator'][name], value)
        assert value < 3
    for row in closure['optional_penalty_results']:
        assert F(row['optional_gate_penalty_speedup_upper'][1]) < 3
        assert F(row['optional_time_penalty_speedup_upper'][1]) < 3
    result = {
        'status': 'independent_root_review_passed',
        'blocking_findings': [],
        'new_simulation_observations': 0,
        'analytic_review': [
            'The target difference and randomized quadrature arise from one physical repeated-phase/dephasing/readout model. Multiplying the observed sign by an independently randomized phase sign removes the unknown readout bias without giving the true contrast or dephasing to the policy.',
            'The bounded phase domain prevents aliases for both fixed controls. The wrong-direction e-process is a supermartingale over the entire declared domain; equality uses a union bound on two thresholds200. The confidence/risk and power levels are kept distinct.',
            'The sin(kh)/sin(h) lower bound and the largest declared gamma give a uniform positive drift for every eligible case. Hoeffding at the finite cap bounds failure of the correct process to cross; adding the wrong-direction Ville event proves correct-stop power, including both signs.',
            'For finite expected cost, bounded stopped-Wald may be applied to the uncapped correct crossing at T wedge n. The stopped log value is at most log200+log(1+b), yielding E[T wedge n]<=that value/drift. Monotone convergence establishes integrability and the claimed upper cost without circularly assuming Wald at an integrable T.',
            'The common physical null phi=0 retains exactly the same dephasing and readout nuisance for all depths. Quantum relative entropy bounds every allowed final measurement. It is not optimized separately under each history.',
            'The positive series for r atanh(r), the inequality1-cos(theta)<=theta^2/2, and termwise upper maxima are global. Their sums equal log2 and4log2-pi^2/6. No weak-phase asymptotic or unrestricted-noiseless Heisenberg claim is used.',
            'Both E[N] and E[G] lower bounds hold simultaneously for every allowed transcript, so their positive linear combination bounds modeled device time. Ignoring switches in a lower bound is conservative.',
            'The explicit shared shot/gate/time budgets contain both baseline paths for all outcomes. Penalized shot and actual costs coincide; optional larger gate/time penalties also retain the claimed exclusions.',
            'A valid baseline upper cost divided by a universal candidate lower cost correctly bounds the speedup over the strongest qualified comparator, whose cost can only be smaller. The exclusion applies to the declared score pair, not every phase-estimation problem.'
        ],
        'numerical_precision_digits': 130,
        'independent_matrix_relative_entropy_checks': matrix_rows,
        'symbolic_checks': 3,
        'independent_baselines': baseline_rows,
        'independent_speedup_upper_constants': {key: mp.nstr(value, 80) for key, value in ratios.items()},
        'source_links_checked': checked_hashes+1,
        'scope_limits': [
            'The known small phase domain, single phase axis and stationary commuting noise are substantive model promises. A broader application must pay for phase localization and test a new physical model.',
            'Reference-assisted, cross-circuit coherent memory, noncommuting dynamical control and error correction are outside this resource contract.',
            'The >15x advantage over restricted depth1 methods is not superiority over equally equipped multi-depth SOTA. No new operational Monte Carlo or original-benchmark success is claimed.',
            'The RPE literature is used only as a comparator pointer. Its corrected or uncorrected constants are not inputs to this proof.'
        ],
        'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).resolve(), source, closure_path)}
    }
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'status': result['status'], 'symbolic_checks': 3,
                      'matrix_checks': len(matrix_rows), 'baselines': len(baseline_rows)}))


if __name__ == '__main__':
    main()

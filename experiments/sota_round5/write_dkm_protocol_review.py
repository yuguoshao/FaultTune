"""Freeze a targeted primary-source DKM/FWSP implementation review."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'results/sota_round2/literature'
NEW = ROOT/'results/sota_round5/literature'
OUT = NEW/'dkm_fwsp_protocol_review.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path, url=None, locator=None):
    result = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}
    if url:
        result['url'] = url
    if locator:
        result['locator'] = locator
    return result


def main():
    current = json.loads((NEW/'current_source_downloads.json').read_text())
    assert all(row['status'] == 'downloaded' for row in current['downloads'])
    branch = json.loads((NEW/'tidnabbil_master_current.json').read_text())
    commit = branch['target']['hash']
    assert commit == 'fdb307afcee834cee17fa849f3f6ca7e4d518a64'
    assert sha(NEW/'fw_current.html') == sha(OLD/'fw_latest.html')
    assert sha(NEW/'fw_arxiv_current.html') == sha(OLD/'fw_abs.html')
    baseline = ROOT/'experiments/sota_round2/literature/dkm_baseline.py'
    component = ROOT/'experiments/sota_round5/dkm_oracle_certificate.py'
    checks = json.loads((NEW/'dkm_oracle_component_checks.json').read_text())
    assert checks['status'] == 'passed' and checks['source_sha256'][component.relative_to(ROOT).as_posix()] == sha(component)
    old_review = json.loads((ROOT/'results/sota_round2/theory/dkm_review.json').read_text())
    assert old_review['status'] == 'passed'
    pdf_url = 'https://proceedings.neurips.cc/paper_files/paper/2019/file/8d1de7457fa769ece8d93a13a59c8552-Paper.pdf'
    pinned = 'https://api.bitbucket.org/2.0/repositories/wmkoolen/tidnabbil/src/'+commit+'/'
    result = {
        'status': 'targeted_current_source_review_and_one_component_implemented',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'DKM2019 and the directly relevant FWSP2025 follow-up; not an exhaustive2026 SOTA survey or a performance claim.',
        'current_verification': {
            'web_search': {'attempted': True, 'status': 'transport_error',
                           'queries': ['Degenne Koolen Menard2019 official code tidnabbil', 'Pure Exploration via Frank-Wolfe Self-Play2509.19901 Bernoulli code'],
                           'fallback': 'Targeted read-only current NeurIPS/arXiv/Bitbucket official endpoints; no search results invented.'},
            'download_ledger': binding(NEW/'current_source_downloads.json'),
            'dkm_current_publication': binding(NEW/'dkm_neurips_current.html', pdf_url.replace('/file/', '/hash/').replace('-Paper.pdf', '-Abstract.html')),
            'dkm_arxiv_current': binding(NEW/'dkm_arxiv_current.html', 'https://arxiv.org/abs/1906.10431'),
            'fw_current_abstract': binding(NEW/'fw_arxiv_current.html', 'https://arxiv.org/abs/2509.19901'),
            'fw_current_fulltext': binding(NEW/'fw_current.html', 'https://arxiv.org/html/2509.19901'),
            'fw_version': 'Official current abstract/fulltext still identify v1 submitted2025-09-24; refreshed content hashes equal the previously inspected version.',
            'official_repository': binding(NEW/'tidnabbil_repo_current.json', 'https://bitbucket.org/wmkoolen/tidnabbil'),
            'official_current_branch': binding(NEW/'tidnabbil_master_current.json', 'https://api.bitbucket.org/2.0/repositories/wmkoolen/tidnabbil/refs/branches/master'),
            'official_current_commit': commit, 'official_commit_date': branch['target']['date'],
            'code_inspection_scope': 'Read cached exact-commit source files after independently refreshing and matching the author-linked repository head. Official Julia code was not executed.'},
        'primary_sources': {
            'DKM2019': {'title': 'Non-Asymptotic Pure Exploration by Solving Games',
                        'authors': ['Remy Degenne', 'Wouter M. Koolen', 'Pierre Menard'], 'publication': 'NeurIPS2019',
                        'paper': binding(OLD/'dkm_neurips.txt', pdf_url, 'Section2.1; Algorithm1; Section2.2 Theorem1; Section2.3; Section3.1; Section4'),
                        'appendices': binding(OLD/'dkm_arxiv.txt', 'https://arxiv.org/abs/1906.10431', 'AppendicesB,C,D,F'),
                        'applicability': 'Finite-arm structured one-parameter exponential-family identification explicitly includes Bernoulli and general finite answer maps. A registered finite rational-angle single-frequency mean model fits this structure. Compact-interior and exact-oracle assumptions must still be respected for the efficiency theorem.'},
            'KK2021': {'title': 'Mixture Martingales Revisited with Applications to Sequential Tests and Confidence Intervals',
                       'publication': 'JMLR22(2021)', 'paper': binding(OLD/'kk_arxiv.txt', 'https://jmlr.org/papers/v22/18-798.html', 'Theorem7 equations9-10; Proposition15 equation20'),
                       'applicability': 'General exponential-family/Bernoulli time-uniform all-arm GLR calibration; the smaller Gaussian-specific formulas are not substituted.'},
            'FWSP2025': {'title': 'Pure Exploration via Frank-Wolfe Self-Play', 'authors': ['Xinyu Liu', 'Chao Qin', 'Wei You'],
                         'paper': binding(NEW/'fw_current.html', 'https://arxiv.org/html/2509.19901', 'Assumptions1-3; Theorem1; Section6 Algorithm1'),
                         'theorem1_exact_scope': 'BAI in unstructured single-parameter exponential-family bandits and BAI in Gaussian linear bandits; convergence of the static information-allocation game.',
                         'two_psr_arm_case': 'For fixed twoPSR arms, sign(C) equals which Bernoulli mean is larger, so the static unstructured BernoulliBAI game falls inside the stated theorem.',
                         'six_arm_single_frequency_case': 'A Bernoulli mean-linear physical-cone model with answer sign(C) is not automatically covered by the Gaussian-linearBAI result. Merely listing Bernoulli in the general model discussion is insufficient.',
                         'learning_and_stopping_gap': 'Section6 assumes a structure-respecting posterior update/sample oracle and explicitly leaves theoretical analysis of the proposed learner for future work. No complete delta-correct Bernoulli-single-frequency sign stopping protocol is supplied by that result.',
                         'single_opposite_component': 'For a fixed sign, the closed opposite physical cone is one convex component. The skeptic simplex over alternative components is a singleton; this does not prove the missing learning/stopping guarantee.',
                         'code_status': 'No author-linked FWSP executable repository was identified in the inspected current paper/abstract. The web-search connector failed; no global absence-of-code claim is made.'}},
        'official_code_components': {
            'paper_identity': binding(OLD/'tidnabbil_purex_README.txt', pinned+'purex_games_paper/README', 'Names DKM2019; Julia1.1.0; experiment entrypoints'),
            'sampling': binding(OLD/'tidnabbil_samplingrules.jl', pinned+'purex_games_paper/samplingrules.jl', 'DaBombState: one AdaHedge per answer; optimistic_gradient; best response'),
            'tracking': binding(OLD/'tidnabbil_tracking.jl', pinned+'tracking.jl', 'CTracking.track uses argmin(N-sumw), initialized at paid counts'),
            'threshold': binding(OLD/'tidnabbil_thresholds.jl', pinned+'thresholds.jl', 'GK16(t)=log((log(t)+1)/delta)'),
            'runner': binding(OLD/'tidnabbil_runit.jl', pinned+'purex_games_paper/runit.jl', 'Bernoulli initialization repeats each arm until both outcomes appear; GLRT before each new draw'),
            'task_examples': binding(OLD/'tidnabbil_peps.jl', pinned+'peps.jl', 'SameSign tests all arm means relative to a known threshold; it is not this unknown-A C-sign model'),
            'paper_vs_code_differences': [
                'Printed Algorithm1 initializes each arm once and uses cumulative RATIO tracking; official CTracking uses count-minus-cumulative-target tracking.',
                'Official Bernoulli runner may spend multiple initialization pulls per arm until observing both0 and1. At a physical endpoint it need not finish; this is not the one-pull paper initialization.',
                'Section4 explicitly calls beta=log((1+log(t))/delta) and f=log(t) unlicensed by theory. The official plot protocol is therefore not used as a formally1% certified baseline without an additional proof.',
                'Official SameSign is a different answer model. Its glrt subtracts feasible-model fit from opposite-model fit, whereas the printed generic GLR uses the saturated exponential-family numerator. For structured data these need not agree.'],
            'oracle_truth_warning': 'Official FixedWeights/oracle allocations in plots consume the true model and are labelled oracle benchmarks. They must not be used as an executable paid-data strategy.'},
        'concrete_implementable_protocol': {
            'name': 'DKM2019 Algorithm1 Section3.1 finite-rational-grid Bernoulli C-sign adaptation with KK21 stopping',
            'grid': 'Freeze before observations: six arms(cos,sin)=(-3/5,+/-4/5),(0,+/-1),(3/5,+/-4/5), or a separately preregistered matched rational grid.',
            'physical_model': 'p_a=A+B*cos_a+C*sin_a; Omega={0<=A<=1,sqrt(B^2+C^2)<=min(A,1-A)}; answers sign(C), opposite closure sign*C<=0.',
            'input_contract': 'Only fixed arms, paid per-arm integer(N,S), declareddelta/margin/tolerances and internal state; no trueA/B/C or case-dependent prior/allocation.',
            'initialization': 'Exactly one paid pull per arm for the paper protocol. Any official-code-style initialization is a separately named and fully charged variant.',
            'design_domain': 'Declare eta=1e-6 and the contracted physical model eta+(1-2eta)*X*u, u inOmega. Clip empirical means for design to[eta,1-eta]. Do not claim the original compact-interior efficiency theorem for true boundary cases outside this design assumption.',
            'projection': 'Build per-arm KL intervals with f(n)=x+log(x)+1/2, x=(1+a)(1+b)log(n), a=b=.1. Minimize sumN_a KL(mu_hat_a,mu_a) over model/interval intersection; select the sign of the fittedC. Distinguish proven empty intersection from a numerical projection failure.',
            'stop_order': 'Select the sign and test it on the SAME current paid counts before planning or drawing another observation.',
            'best_response': 'Per selected answer, query AdaHedge for weightsw; minimize sumw_a KL(mu_hat_a,p_a(lambda)) on the opposite contracted cone. Require a logged certified gap tolerance or explicit unresolved status, not only an optimizer success flag.',
            'optimism': 'U_a=max(f(n)/N_a,KL(interval_lower_a,p_a(lambda)),KL(interval_upper_a,p_a(lambda))). Update that answer-specific AdaHedge with loss-U.',
            'tracking': 'Paper protocol: initialize accumulated target to paid initial counts; addw; choose argminN_a/target_a. A count-minus-target official-code variant must be separately labelled and fixed before comparison.',
            'schedule': 'One physical shot per update and current-count stopping after every paid observation. Batch/checkpoint coarsening is a separately named adaptation.',
            'cap_and_abstention': 'Charge all observations including initialization, errors, and cap-abstention paths. At finite cap output abstain unless the common valid certificate passes. C=0 has error protection but no finite-termination or power guarantee.',
            'comparison_fairness': 'Use matched grid, cap, delta, monitoring and stopping-calibration opportunities for new methods and literature controls. Report per-case correct-stop power and unconditional cost; do not compare90%-power costs as95%-power baselines.'},
        'stopping_safety_requirements': {
            'KK21_primary': {'statistic': 'Z_sign=sum_a sup_q logL_a(q)-sup_theta_in_full_closed_opposite_Omega logL(theta)',
                             'threshold': 'sum_a3*log(1+logN_a)+K*C_exp(log(1/delta)/K)',
                             'C_exp': '2*h_tilde_(3/2)((h_inverse(1+x)+log(2*zeta(2)))/2), with h(u)=u-logu and the KK21 piecewise h_tilde definition.',
                             'numerical_acceptance': 'Outward lower bound onZ strictly greater than outward upper threshold. The optimizer is merely a proposal; the null likelihood needs a global concavity/support upper bound.',
                             'common_error_event': 'Whenever a selected sign is false, the true physical mean vector lies in that opposite cone. ItsZ is no larger than the all-arm saturated-to-true KL statistic controlled by KK21. Both signs, including any report atC=0, share one event of probability<=delta; no unlicensed double delta budget.',
                             'boundary_scope': 'Bernoulli support-edge extension needs a finite-history continuity/contraction argument; it extends validity, not exact-oracle efficiency.'},
            'KT_secondary_adaptation': {'label': 'DKM sampling plus product-Jeffreys universal-inference stopping; explicitly not the original plot threshold',
                                        'Q': 'product_a B(S_a+1/2,N_a-S_a+1/2)/B(1/2,1/2)',
                                        'test': 'logQ-sup_full_opposite logL>log(1/delta), with outward bounds',
                                        'proof_requirement': 'Predictable fixed-arm selection makesQ a normalized sequential law; Q/L_true is a common nonnegativeLR martingale(supermartingale on support edges). Every false-sign rejection implies this same process crosses1/delta.'},
            'forbidden_shortcuts': ['Do not use the stylised official plot threshold as a proven1% boundary.',
                                   'Do not use empirical input truth or oracle allocations to help the policy.',
                                   'Do not use local optimized null likelihood as an upper bound without certification.',
                                   'Do not infer sample-complexity optimality from an absolute1e-5 floating best-response gap.']},
        'current_implementation_gap': {
            'current_baseline': binding(baseline), 'initial_issue': binding(OLD/'dkm_checks_initial_oracle_issue.json'),
            'corrected_checks': binding(OLD/'dkm_checks.json'), 'prior_independent_review': binding(ROOT/'results/sota_round2/theory/dkm_review.json'),
            'already_corrected': 'The initial huge-oracle-gap and stale-label probe predates the current frozenbaseline. Current code added pooled restarts, convex linear extension, cone repair, line corrections and current-count stopping order; the old issue file is retained, not treated as current performance.',
            'remaining_gaps': ['Absolute floating1e-5 gap can be large relative to a weak-signal information game; certified accuracy and accumulated approximate-oracle regret are still needed.',
                               'Projection may fail numerically; this is not evidence the model-confidence intersection is empty.',
                               'No step-by-step run of the pinned official Julia implementation has been completed.',
                               'Six-arm full-rank physical-cone and continuous-angle controllers have different action sets; same-grid controls are necessary for attribution.',
                               'A strict valid KK21 threshold differs from the paper plotted threshold; report both formula and scope rather than claiming complete original reproduction.']},
        'completed_component': {
            'module': binding(component), 'checks': binding(NEW/'dkm_oracle_component_checks.json'),
            'APIs': ['certificate_from_paid_counts(arms,trials,successes,weights,sign,point,...)',
                     'exact_psr_best_response_from_paid_counts(trials,successes,weights,sign,...)',
                     'numerical_proposal_with_certificate(arms,trials,successes,weights,sign,...)'],
            'mathematics': 'Exact rational feasible proposal and gradient; F(u)-minF<=gradient.u+support_opposite(-gradient), outward enclosed. Accept only if gap_upper<=atol+rtol*certified optimum_lower. The point is an optimizer proposal, not case truth.',
            'exact_psr_component': 'For two fixedPSR arms the opposite-answer Bernoulli KL best response is exact weighted isotonic pooling, or unchanged empirical means when already in the opposite ordering; no optimizer or true parameters are required.',
            'finite_checks': 'Six deterministic analytic PSR count/weight/sign cases, mirrored/support-edge/zero-weight cases, a six-arm observed-count proposal, exact infeasibility rejection, and an explicit weak-signal absolute-tolerance counterexample. No RNG or new samples.',
            'counterexample_scope': 'A constructed feasible weak-signal proposal has gap about1.11e-7, below the old1e-5 gate but rejected by1e-12+0.1% relative accuracy. This demonstrates insufficiency of the tolerance guarantee, not an allegation that the currentold runner generated this point.',
            'not_yet_done': 'The component has not been integrated into a new full sampled DKM run. It does not establish an exact-oracle efficiency theorem, a runtime advantage, a95%-power result, or SOTA superiority.'},
        'new_simulation_observations': 0, 'downloaded_official_code_executed': False,
        'final_claim_boundary': 'A concrete auditable literature-control protocol and a repaired accuracy-certification component are available. The full finite-grid Bernoulli-single-frequency controller is an adaptation of DKM ideas with stated valid stopping, not the original paper experimental protocol or a complete FWSP reproduction.'}
    result['producer'] = binding(Path(__file__).resolve())
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'path': str(OUT), 'sha256': sha(OUT),
                      'new_simulation_observations': 0}), flush=True)


if __name__ == '__main__':
    main()

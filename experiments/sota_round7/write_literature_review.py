"""Produce the bounded round7 original-source and resource-comparator review."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/sota_round7/literature'


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def main():
    report = {
        'status': 'targeted_original_source_review_complete_with_explicit_coverage_limits',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'stream_sota',
        'new_simulation_observations': 0,
        'sota_claim_supported': False,
        'field_wide_novelty_established': False,
        'headline': 'Neither inspected Tendick Appendix I nor Mitra distance-resource construction supplies the fixed-basis binary-qubit KL reflected-pair theorem. The quantity itself remains prior art from Baek2020. New observation resources require equally resourced physics-aware controls; resource gain alone is not a SOTA algorithm gain.',
        'direct_priority_findings': [
            {
                'authors': 'Lucas Tendick, Martin Kliesch, Hermann Kampermann, Dagmar Bruss',
                'title': 'Distance-based resource quantification for sets of quantum measurements',
                'doi': '10.22331/q-2023-05-15-1003',
                'version': 'arXiv2205.08546v2, 2023-05-02',
                'url': 'https://arxiv.org/html/2205.08546v2#A9.E108',
                'inspected_locations': 'Definitions of normalized diamond distance and assemblage resource, Eqs.(13)-(15), and full Appendix I (Dichotomic measurements), Eq.(108).',
                'result': 'For dichotomic assemblages, the diamond-distance, unassisted total-variation-distance and operator-norm resource quantities agree: R_diamond=R_l1=R_infinity. For a binary channel difference, the two outcome differences are negatives of each other, so the same input maximizes their absolute value; entanglement does not improve diamond-norm discrimination.',
                'limitation': 'This is not a relative-entropy identity. The proof and its no-entanglement conclusion cannot be transferred to KL. No fixed-basis Bernoulli-KL optimizer, entropy-secant center, or exact reflected-pair structural theorem appears in the inspected material.',
            },
            {
                'authors': 'Arindam Mitra, S. Mukherjee, C. Lee',
                'title': 'Distance-based measures and Epsilon-measures for measurement-based quantum resources',
                'doi': '10.1103/wwn7-j8df',
                'version': 'arXiv2505.11331v5, 2026-02-24',
                'url': 'https://arxiv.org/html/2505.11331v5#S3.E55',
                'inspected_locations': 'Section III.1 conditions D1-D5; Section III.2 Eq.(55) and ensuing resource properties.',
                'result': 'The work studies general distance resource monotones and epsilon smoothing. Eq.(55) minimizes the distance to free measurements after infimizing over ancillary system dimensions and trivially extending the measured system by M tensor T_B.',
                'metric_caveat': 'Its concrete diamond distance is unnormalized (maximum 2), unlike Tendick. D1-D5 state convexity, tensor and free-transformation properties, swap invariance and marginalization; do not falsely attribute an explicit symmetry requirement to that list.',
                'limitation': 'The extension-based resource definition is not automatically Baek2020 fixed-system unassisted min-diagonal-Q sup-rho KL. The inspected original does not supply the fixed-basis binary-qubit relative-entropy optimizer or reflected-pair theorem. This targeted absence is not a field-wide novelty conclusion.',
            },
        ],
        'retained_prior_art': {
            'source': 'results/sota_round6/literature/reflected_pair_priority_review.json',
            'decisive_prior': 'Baek, Sohbi, Lee, Kim, Nha (2020), Quantifying coherence of quantum measurements, New Journal of Physics22,093019, Definition2/Eq.(7).',
            'url': 'https://arxiv.org/html/2008.03999v1#S3.E7',
            'mapping': 'For effect H=A I+B Z+C X and fixed incoherent Z basis, their unassisted measurement-coherence quantity is min_diagonal_Q sup_rho KL(p_H(rho)||p_Q(rho)). The labeled pure-circle design maxmin equals it by the regularization/Sion argument retained in round6.',
            'potential_contribution_only': 'An exact globally valid reflected support and tangent-null structural solution in the established fixed-interior weak-C regime; no new resource quantity, general KL criterion or all-C theorem is claimed.',
        },
        'resource_comparators': {
            'full_bitstring_fixed_duration': {
                'model_as_supplied_by_stats_branch': 'H=(omega+u)X_0/2+gamma|11><11|_12; same duration tau for old/new amplitude controls. Prepare/unprepare spectators with Hadamards and observe all three Z bits. Independent readout flips and q1/q2 dephasing. No free inverse, controlled-U or extra coherent amplification.',
                'mechanism': 'Writing target outcome X and spectator outcome S, the OR-failure difference is P(S=00)[P_new(X=1)-P_old(X=1)]. Full readout exposes X, avoiding the OR attenuation.',
                'sufficiency_boundary': 'X is sufficient for the sign in the enlarged Bern(x_j) times arbitrary shared-Q model. Under the fully specified physical model, spectator data may constrain shared readout nuisance and feasible rate pairs. Do not assert X is universally sufficient for physics-aware algorithms.',
                'minimum_executable_controls': ['Current qualified fixed exact-sign mixture and beta/full-count safe rules applied to X with identical information.', 'Sequential two-sample e-variable/control based on Turner-Ly-Grunwald, with direction-specific validity established.', 'A full-multinomial physical likelihood/GLR or mixture control exploiting the same shared nuisance and amplitude constraints; finite-risk validity and computation must be supplied.'],
                'source': {
                    'title': 'Generic E-Variables for Exact Sequential k-Sample Tests that allow for Optional Stopping',
                    'authors': 'Rosanne Turner, Alexander Ly, Peter Grunwald',
                    'version': 'arXiv2106.02693v3, 2022-06-22',
                    'url': 'https://arxiv.org/html/2106.02693v3',
                    'locations': 'Section2.1 Eq.(2.2), Theorem1; Eq.(2.3) block product; Section2.2 Eqs.(2.5)-(2.7) predictive version; Section3 Bernoulli; Section7.2 conditional alternative.',
                    'mapping': 'A generic two-stream e-variable supports exact optional stopping and has simple-alternative growth optimality when the single-observation family is convex, including Bernoulli.',
                    'implementation_constraints': 'Use complete blocks, compute predictors only from earlier blocks, and respect arm-allocation assumptions. Equality-null rejection alone does not justify a subsequently selected sign; one-sided composite-null risk and two-direction accounting need an explicit proof.',
                },
                'claim_boundary': 'Threefold gain against an OR-only control can demonstrate information loss in the scalar summary. It does not demonstrate algorithmic improvement over controls receiving the same full bitstrings.',
            },
            'multiple_depth_ramsey': {
                'model_as_supplied_by_design_branch': 'Fresh single-qubit Ramsey U_phi^k with unknown Markov dephasing gamma in [.06,.065], readout contrast v>=.95, phi in [-.02,.02]. Target is the original-depth old/new contrast Delta=(v exp(-gamma)/2)sin(phi), epsilon=.001. Charge shots and k+2.2 gate/setup/readout cost and switching<=.1.',
                'mechanism': 'Fairly randomizing opposite quadratures yields Z=S(2Y-1), E[Z]=v exp(-gamma k)sin(k phi), canceling unknown offset. All controls may use the same randomization and depths.',
                'minimum_executable_controls': ['Fixed depth8/16 two-sided sequential test with a frozen non-oracle betting parameter and a valid signed stopping certificate.', 'Robust multi-depth Ramsey/RPE or adaptive frequency-learning design, using the same available phases, depth costs, noise assumptions, and valid stopping certificate.'],
                'source': {
                    'authors': 'Shelby Kimmel, Guang Hao Low, Theodore J. Yoder',
                    'title': 'Robust calibration of a universal single-qubit gate set via robust phase estimation',
                    'version': 'arXiv1502.02677v3, 2021-10-21; original publication Physical Review A92,062315(2015)',
                    'url': 'https://arxiv.org/html/1502.02677v3#S1.Thmthm1',
                    'locations': 'Opening Errata paragraphs p1.1/p2.1; Theorem I.1; Section IV noise discussion; Appendix B known principal-range discussion.',
                    'theorem_assumptions': 'Sin and cos experiment families indexed by repeated-gate count k, execution time proportional to k, and uniform additive probability errors below 1/sqrt(8). If the bound holds only below k*, the arbitrary-depth scaling conclusion does not apply.',
                    'erratum': 'The v3 opening explicitly corrects the phase-unwrapping error criterion: pi/(2 k_j) was insufficient, while pi/(3 k_j) is the relevant bound. It says the Section V.1 analysis is incorrect and the Sections V.1/V.2 scaling constants must be replaced by corrected arguments in cited work. The robust Heisenberg-scaling approach survives with larger constants.',
                    'repair_reference_lead_only': 'The erratum points to Russo, Kirby, Rudinger, Baczewski, Kimmel (2021), Consistency testing for robust phase estimation, Physical Review A103,042609, Appendix B. That separate original was not fetched in this targeted pass.',
                    'comparison_implication': 'Do not implement old quoted constants as certified 1% stopping thresholds. Finite Markov dephasing limits usable depths. With a known narrow phase range, a fixed well-chosen deep circuit can be stronger than a generic unwrapping schedule; the RPE Appendix B explicitly recognizes the known-range case.',
                },
                'claim_boundary': 'Fewer shots alone does not establish lower paid quantum cost, and an asymptotic variance or Bayesian credible interval is not an anytime-valid signed-decision guarantee.',
            },
        },
        'acceptance_and_status': {
            'risk': 'At most1% uniform report risk; unchanged.',
            'power': 'At least95% correct stop separately in every evaluation case; unchanged.',
            'cost': 'Threefold geometric improvement in unconditional paid cost remains the research goal, with identical resources and charged learning for controls.',
            'latest_frozen_result': 'Candidate geometric cost is1.04081670022 times strongest qualified implemented fixed mixture mix_0.75_0.99_w0.5; original1% risk passes but stricter internal.00999 fails. No round7 literature result changes this.',
            'baseline_completion': 'Faithful full algorithm-level SOTA implementations and qualification remain incomplete.',
            'old_obstruction_scope': 'The <2.593x adaptive-angle ceiling concerns the old eight positive cases and mirrors in the single-harmonic model. It does not automatically apply to new bitstring or multiple-depth resource models.',
        },
        'coverage_limits': ['Targeted examination of the two requested measurement-resource originals and one directly applicable RPE original; no claim of exhaustive literature search.', 'No simulation or new operational performance measurement was performed in this branch.', 'The two new resource specifications are branch proposals; this review does not independently certify their physics or information bounds.'],
    }
    sources = [
        'experiments/sota_round7/write_literature_review.py',
        'experiments/sota_round7/fetch_literature.py',
        'experiments/sota_round7/inspect_and_fetch_rpe.py',
        'results/sota_round7/literature/public_source_ledger.json',
        'results/sota_round7/literature/rpe_source_ledger.json',
        'results/sota_round6/literature/reflected_pair_priority_review.json',
        'results/sota_round6/research_state.json',
        'results/sota_round6/distance_to_sota.json',
        'results/structured_validation/literature_cache/continuous_fisher_turner_ly_grunwald_v3.html',
        'results/structured_validation/literature_cache/continuous_fisher_turner_ly_grunwald_v3.txt',
    ]
    for name in ['tendick2023_abs', 'tendick2023_full', 'mitra2025_abs', 'mitra2025_full', 'kimmel2015_abs', 'kimmel2015_html']:
        sources.extend('results/sota_round7/literature/' + name + suffix for suffix in ['.html', '.txt'])
    report['source_sha256'] = {path: sha(path) for path in sources}
    path = OUT / 'direct_priority_and_resource_comparators.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path.relative_to(ROOT)), 'source_files': len(sources), 'sota_claim_supported': False}))


if __name__ == '__main__':
    main()

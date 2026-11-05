"""Close this deterministic research round without closing the SOTA goal."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT/'results/sota_round6/research_state.json'
    if output.exists():
        raise FileExistsError('Preserve completed research state')
    paths = [
        'results/sota_round6/distance_to_sota.json',
        'results/sota_round6/theory/continuum_null_certificate.json',
        'results/sota_round6/theory/continuum_certificate_peer_review.json',
        'results/sota_round6/theory/reflected_pair_small_background_theorem.json',
        'results/sota_round6/theory/reflected_pair_near_symmetric_theorem.json',
        'results/sota_round6/theory/local_theorems_root_review.json',
        'results/sota_round6/theory/inverse_kl_curvature_reduction.json',
        'results/sota_round6/reflection_counterexamples/search_plan.json',
        'results/sota_round6/reflection_counterexamples/summary.json',
        'results/sota_round6/reflection_counterexamples/three_support_technical_retry.json',
        'results/sota_round6/reflection_counterexamples/interval_certificate_handoff.json'
    ]
    # The separately delegated literature report is supplied as a relative path.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--literature-report', required=True)
    args = parser.parse_args()
    paths.append(args.literature_report)
    checked = []
    for rel in paths:
        path = ROOT/rel
        data = json.loads(path.read_text(encoding='utf-8'))
        for field in ('source_sha256', 'input_sha256', 'frozen_sources_sha256'):
            for source, expected in data.get(field, {}).items():
                if isinstance(expected, str) and len(expected) == 64:
                    assert digest(ROOT/source) == expected, (rel, source)
                    checked.append({'artifact': rel, 'source': source})
    certificate = json.loads((ROOT/paths[1]).read_text(encoding='utf-8'))
    review = json.loads((ROOT/paths[2]).read_text(encoding='utf-8'))
    gap = json.loads((ROOT/paths[0]).read_text(encoding='utf-8'))
    assert certificate['certified_models'] == 48
    assert certificate['unresolved_models'] == 0
    assert review['blocking_findings'] == []
    state = {
        'status': 'round6_deterministic_theory_certificates_and_reviews_complete_goal_active',
        'updated_utc': datetime.now(timezone.utc).isoformat(),
        'objective': 'Continue pursuing superiority over applicable SOTA without weakening the original1percent risk,95percent per-case correct-stop,threefold geometric cost and broader task requirements.',
        'sota_goal_complete': False,
        'previous_goal_turn_classification': 'progress',
        'current_goal_turn_classification': 'progress',
        'new_simulation_observations': 0,
        'operational_performance_changed': False,
        'latest_cost_comparison': gap['latest_operational_comparison'],
        'completed_work': {
            'fixed_model_continuum_certificates': {
                'models': 48,
                'complete_circle_interval_leaves': sum(len(r['accepted_leaves']) for r in certificate['rows']),
                'maximum_certified_oracle_information_upper_over_reflected_pair_lower': max((r['upper_over_pair_lower'] for r in certificate['rows']), key=lambda s: __import__('decimal').Decimal(s)),
                'meaning': 'For each listed exact model, arbitrary angle distributions have oracle information at most1.000001times that of the listed reflected pair. Directed all-angle proof and independent full leaf replay replace the former finite-mesh-only evidence.',
                'tests': 'Four targeted tests passed: interval squares, independent derivative checks including asymmetric/negative-C cases, exact symmetric capacity and complete leaf coverage, invalid inputs/budget exhaustion.',
                'independent_review': 'All5537 leaves replayed exactly; coverage, parameter semantics and source bindings checked.150digit capacity checks,288 independent scalar-chart points with derivatives, and15 symbolic identities passed.'
            },
            'local_analytic_theorems': 'Two independently reviewed global-design existence theorems extend the earlier weak-C region: sufficiently small B at any fixed strictly interior(A,C), and sufficiently small A-1/2 at any fixed strictly physical(B,C). Neither supplies an explicit epsilon or full-parameter theorem.',
            'proof_obstruction': 'Inverse-KL curvature has a simplified exact square formula. A directed-interval example shows it is not monotone in radius even below log2; a larger-radius check cannot automatically certify the desired R_star curvature.',
            'boundary_screen': '353 preregistered deterministic parameter points,209 high-precision rechecks, no retained curvature/tangent counterexample.20 finite-grid technical retries returned two-support optima and no certified three-support improvement; original Python3.8 packaging failures preserved. These scans are not continuum proofs.',
            'literature': 'Direct overlap identified with the relative-entropy coherence of measurements introduced in Baek et al.(2020). The mathematical quantity is established prior art; priority of the particular local design theorem remains unverified. Full applicable stopping-algorithm SOTA comparisons remain incomplete.'
        },
        'claim_limits': [
            'No new stopping candidate or operational cost improvement was produced in this round.',
            'The latest independent operational grid still gives candidate/best-fixed-mixture geometric cost1.04081670022approximately; tying it requires saving3.921603percent of current geometric cost.',
            'The older eight-case and mirror model certificate excludes3x even with adaptive angles and allocation: speedup versus qualified beta is strictly less than2.593. This does not prove impossibility in every future task or on the newer independent grid.',
            'Original user risk/power requirements pass for the frozen candidate, but its stricter internal.00999 risk preregistration fails; failure has not been erased.',
            'Oracle information optimality does not provide paid-learning, finite-cost superiority, or literature-first status.'
        ],
        'next_required': [
            'Prioritize a concrete observation/task structure that can change the certified information limit and quantify its potential before additional stopping-rule tuning or Monte Carlo. Give strong comparators identical resources.',
            'If pursuing the all-parameter reflected-pair theorem, use the exact remaining curvature inequality or another genuine global dual proof; finite meshes and unjustified radius monotonicity cannot close it.',
            'The five exact/algebraic extreme-model seeds are preserved for future certified interval analysis; no all-parameter conclusion follows from the completed48-model certificates.',
            'Complete faithful strongest applicable literature baselines, then freeze any new operational candidate and independent evaluation under unchanged acceptance criteria.'
        ],
        'source_binding_audit': {'checked_links': len(checked), 'failures': [], 'links': checked},
        'artifact_sha256': {rel: digest(ROOT/rel) for rel in paths},
        'state_writer_sha256': digest(Path(__file__).resolve())
    }
    with output.open('x', encoding='utf-8') as handle:
        json.dump(state, handle, indent=2)
    print(json.dumps({'status': state['status'], 'source_links_checked': len(checked),
                      'models': certificate['certified_models'],
                      'leaf_count': state['completed_work']['fixed_model_continuum_certificates']['complete_circle_interval_leaves']}))


if __name__ == '__main__':
    main()

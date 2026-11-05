"""Translate frozen comparison ratios into the remaining cost reductions."""
from decimal import Decimal
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.sota_round6.continuum_null_certificate import enclosure


def endpoint_interval(low, high):
    return [str(enclosure(low).lo), str(enclosure(high).hi)]


def main():
    output = ROOT/'results/sota_round6/distance_to_sota.json'
    if output.exists():
        raise FileExistsError('Preserve gap report')
    source = ROOT/'results/sota_round5/hybrid_minimax/independent_comparison.json'
    adaptive_path = ROOT/'results/sota_round5/theory/certify_adaptive_angle_speedup_limit.json'
    adaptive_review = ROOT/'results/sota_round5/theory/adaptive_angle_speedup_limit_independent_review.json'
    comparison = json.loads(source.read_text(encoding='utf-8'))
    adaptive = json.loads(adaptive_path.read_text(encoding='utf-8'))
    controls = comparison['qualified_fixed_controls']
    strongest = max(controls, key=lambda name: F(comparison['all_fixed_comparisons'][name]['geometric_cost_ratio'][0]))
    summaries = {}
    for name in ('beta_exact', strongest):
        row = comparison['all_fixed_comparisons'][name]
        low, high = map(F, row['geometric_cost_ratio'])
        assert low > 1
        summaries[name] = {
            'candidate_over_comparator_geometric_cost_ratio': row['geometric_cost_ratio'],
            'candidate_extra_cost_percent': endpoint_interval(100*(low-1), 100*(high-1)),
            'fraction_of_current_geometric_cost_to_save_for_tie': endpoint_interval(1-1/low, 1-1/high),
            'fraction_of_current_geometric_cost_to_save_for_threefold_speedup': endpoint_interval(1-1/(3*low), 1-1/(3*high)),
            'candidate_cheaper_cases': row['candidate_proved_cheaper_cases'],
            'candidate_more_expensive_cases': row['candidate_proved_more_expensive_cases']
        }
    assert adaptive['exact_geometric_threefold_excluded'] is True
    result = {
        'status': 'complete_source_bound_quantitative_gap_report',
        'sota_goal_complete': False,
        'new_simulation_observations': 0,
        'latest_operational_comparison': {
            'main_cases': comparison['main_cases'],
            'distinct_positive_parameter_cases': comparison['main_cases']//2,
            'sign_mirrors_included': True,
            'strongest_qualified_implemented_fixed_control': strongest,
            'comparisons': summaries,
            'minimum_correct_stop_probability_interval': comparison['candidate_minimum_correct_probability_interval'],
            'user_one_percent_risk_pass': comparison['candidate_user_one_percent_risk_pass'],
            'stricter_internal_risk_target_pass': comparison['candidate_stricter_internal_00999_pass'],
            'qualification_caveat': 'Passes original user risk/power requirements but fails its stricter preregistered internal .00999 risk target; no replacement or repair has been accepted.'
        },
        'original_threefold_obstruction': {
            'geometric_speedup_strict_upper': adaptive['upward_thousandth_geometric_speedup_bound'],
            'baseline': 'qualified beta_exact, cap8192 pairs /16384 shots',
            'scope': 'Old eight positive single-harmonic Bernoulli cases and their exact sign mirrors; arbitrary predictable angles, allocation and single-shot stopping, with every paid observation counted and .01 uniform null-report risk/.95 per-case correct stop.',
            'interpretation': 'Within this fixed model and case set, tuning the stopping rule or adaptively choosing more angles cannot meet3x. A different observation/task structure must first demonstrate greater accessible information.',
            'nontransfer': 'This certificate is not a theorem about every future FaultTune task and was not proved for the newer48-case independent grid.'
        },
        'remaining_research_gates': [
            'Find an observation/task mechanism with sufficient information gain, retaining the original risk, power, and3x ambition and giving applicable baselines the same resources.',
            'Implement a paid-learning and stopping rule with a finite-cost advantage; local oracle optimality alone supplies no operational gain.',
            'Finish complete, faithful implementations and qualification of the strongest applicable literature baselines; the best currently implemented mixture is not established to equal full-field SOTA.',
            'Freeze a new candidate and a genuinely independent evaluation, and pass the unchanged acceptance requirements.'
        ],
        'time_estimate': 'No defensible completion time for a missing research mechanism. The3.92percent tie gap to an implemented comparator is not a percentage completion estimate for the SOTA objective.',
        'source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).resolve(), source, adaptive_path, adaptive_review)}
    }
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({'strongest': strongest, 'comparisons': summaries,
                      'threefold_excluded_on_old_grid': True}, indent=2))


if __name__ == '__main__':
    main()

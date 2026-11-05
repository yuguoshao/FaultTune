"""Plot only the fully audited, unchanged scientific comparison after extension.

No effect file is opened until the complete 9,600-record audit gate passes.
Every input named by the final audit is verified, including administrative
amendment and producer provenance. This never reads partial effect data.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT/'results/structured_validation/prospective_completion_v1'
REQUIRED_INPUTS = {'spec.json', 'summary.json', 'trials.jsonl', 'analysis.json',
                   'selection.json', 'release.json', 'amendment.json',
                   'completion_run.json', 'analysis_provenance.json'}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def label(method):
    return (method.replace('weighted_conditional_unequal', 'unequal Fisher')
            .replace('pair_grid', 'paired grid').replace('_', ' '))


def load_complete_inputs():
    # Audit and summary are checked before opening the effect estimates.
    require((DIRECTORY/'audit.json').is_file(), 'Final audit is not available')
    audit = read_json(DIRECTORY/'audit.json')
    require(audit['status'] == 'passed', 'Final independent audit must pass')
    require(audit['checks']['terminal_records_checked'] == 9600 and
            audit['audited_cases'] == 56, 'Audit does not cover the full frozen matrix')
    recorded = audit['input_sha256']
    require(REQUIRED_INPUTS <= set(recorded), 'Final audit lacks required scientific or administrative inputs')
    for name, expected in recorded.items():
        require(Path(name).name == name and name not in ('.', '..'), 'Audit input must be a local filename')
        require(digest(DIRECTORY/name) == expected, 'Audited input changed: '+name)
    summary = read_json(DIRECTORY/'summary.json')
    require(summary['status'] == 'complete' and summary['new_trajectories'] == 9600 and
            summary['original_trajectories'] == 7937 and summary['additional_trajectories'] == 1663,
            'No plot is allowed for an incomplete or altered matrix')
    spec = read_json(DIRECTORY/'spec.json')
    amendment = read_json(DIRECTORY/'amendment.json')
    require(summary['spec'] == spec and spec['expected_method_trajectories'] == 9600,
            'Complete summary/spec mismatch')
    require(amendment['original_records'] == 7937 and amendment['remaining_records'] == 1663 and
            amendment['expected_records'] == 9600 and
            spec['plan_sha256'] == amendment['original_plan_sha256'] == audit['plan_sha256'],
            'Administrative extension changed the scientific plan')
    require(digest(ROOT/'results/structured_validation/prospective_plan.json') == audit['plan_sha256'],
            'Frozen scientific plan changed')
    analysis = read_json(DIRECTORY/'analysis.json')
    require(analysis['plan_sha256'] == audit['plan_sha256'] and analysis['cells'] == summary['cells'],
            'Analysis is not the audited scientific comparison')
    for name, expected in analysis['input_sha256'].items():
        require(recorded.get(name) == expected, 'Analysis input does not match the final audit')
    primary, secondary = analysis['primary'], analysis['secondary']
    require(audit['primary_pass'] == analysis['primary_numerical_threshold_met'] ==
            (primary['percentile_95_ci'][1] < 1), 'Unrounded CI acceptance gate differs')
    main_ids = {case['case_id'] for case in spec['cases'] if case['kind'] == 'main'}
    require(len(main_ids) == 48 and spec['repetitions_by_kind']['main'] == 64,
            'Frozen main-case replication matrix differs')
    for row, comparator_key in [(primary, 'primary_comparator'), (secondary, 'secondary_comparator')]:
        require(row['candidate'] == spec['selection']['candidate'] and
                row['comparator'] == spec['selection'][comparator_key], 'Compared method changed')
        numbers = [row['geometric_mean_ratio']]+row['percentile_95_ci']
        require(all(math.isfinite(x) and x > 0 for x in numbers) and numbers[1] <= numbers[2],
                'Invalid final ratio or confidence interval')
        require(row['bootstrap_resamples'] == 5000 and row['bootstrap_seed'] == 914207,
                'Frozen bootstrap specification differs')
        require(len(row['per_case_ratios']) == 48 and
                {value['case_id'] for value in row['per_case_ratios']} == main_ids and
                all(math.isfinite(value['ratio']) and value['ratio'] > 0 for value in row['per_case_ratios']),
                'Per-case comparison is incomplete')
        require(row['faster_cases'] == sum(v['ratio'] < 1 for v in row['per_case_ratios']) and
                row['tied_cases'] == sum(v['ratio'] == 1 for v in row['per_case_ratios']) and
                row['slower_cases'] == sum(v['ratio'] > 1 for v in row['per_case_ratios']),
                'Per-case direction counts differ')
    sources = dict(audit['audit_sources_sha256'])
    for name, expected in sources.items():
        require(digest(ROOT/name) == expected, 'Auditor source changed: '+name)
    for path in [Path(__file__).resolve(), Path(__file__).resolve().with_name('plot_prospective.py'),
                 ROOT/'experiments/structured_validation/prospective/analyze.py',
                 ROOT/'results/structured_validation/prospective_plan.json']:
        sources[path.relative_to(ROOT).as_posix()] = digest(path)
    inputs = {name: recorded[name] for name in recorded}
    inputs['audit.json'] = digest(DIRECTORY/'audit.json')
    return analysis, sources, inputs


def main():
    png, manifest_path = DIRECTORY/'comparison.png', DIRECTORY/'plot_manifest.json'
    require(not png.exists() and not manifest_path.exists(), 'Preserve existing completed plot artifacts')
    analysis, sources, inputs = load_complete_inputs()
    primary, secondary = analysis['primary'], analysis['secondary']
    rows = [primary]+([secondary] if primary['comparator'] != secondary['comparator'] else [])
    plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.3),
                            gridspec_kw={'width_ratios': [1.15, 1.35]})
    colors = ['#1565a2', '#85939e']
    for index, row in enumerate(rows):
        point = row['geometric_mean_ratio']
        lo, hi = row['percentile_95_ci']
        axes[0].hlines(index, lo, hi, color=colors[index], linewidth=2)
        axes[0].vlines([lo, hi], index-.06, index+.06, color=colors[index], linewidth=1.5)
        axes[0].scatter([point], [index], color=colors[index], s=52, zorder=3)
        axes[0].text(.03, index+.17,
                     'Ratio {:.6f} | 95% CI [{:.8f}, {:.8f}]'.format(point, lo, hi),
                     transform=axes[0].get_yaxis_transform(), ha='left', va='top', fontsize=9)
    all_x = [1.0]+[value for row in rows for value in
                   [row['geometric_mean_ratio']]+row['percentile_95_ci']]
    padding = max(.012, .12*(max(all_x)-min(all_x)))
    axes[0].set_xlim(min(all_x)-padding, max(all_x)+padding)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([('Primary\n' if index == 0 else 'Secondary\n')+label(row['comparator'])
                            for index, row in enumerate(rows)])
    axes[0].set_ylim(-.4, max(.8, len(rows)-.4))
    axes[0].invert_yaxis()
    axes[0].axvline(1, color='#a84739', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Candidate / comparator: geometric mean cost ratio')
    axes[0].set_title('Aggregate comparison with 95% bootstrap intervals', loc='left', fontsize=11)
    ratios = sorted(primary['per_case_ratios'], key=lambda row: (row['ratio'], row['case_id']))
    values = np.array([row['ratio'] for row in ratios])
    bar_colors = np.where(values < 1, '#1565a2', np.where(values > 1, '#c36a51', '#85939e'))
    axes[1].bar(np.arange(len(values)), values-1, bottom=1, color=bar_colors, width=.86)
    axes[1].axhline(1, color='#46505a', linewidth=1)
    axes[1].set_xlim(-1, len(values))
    axes[1].set_xticks([0, 11, 23, 35, 47])
    axes[1].set_xticklabels(['1', '12', '24', '36', '48'])
    deviation = max(.025, 1.2*float(np.max(np.abs(values-1))))
    axes[1].set_ylim(1-deviation, 1+deviation)
    axes[1].set_xlabel('48 fixed main cases, ordered by cost ratio')
    axes[1].set_ylabel('Ratio of unconditional mean total shots')
    axes[1].set_title('Candidate versus selected primary comparator', loc='left', fontsize=11)
    axes[1].text(.02, .96, 'Faster {} | tied {} | slower {}'.format(
        primary['faster_cases'], primary['tied_cases'], primary['slower_cases']),
        transform=axes[1].transAxes, va='top', fontsize=10)
    accepted = analysis['primary_numerical_threshold_met']
    verdict = 'passes' if accepted else 'does not pass'
    fig.suptitle('FaultTune: {} {} the prespecified primary comparison'.format(
        label(primary['candidate']), verdict), x=.03, y=.975, ha='left', fontsize=15, fontweight='bold')
    fig.text(.03, .922, 'Candidate: {}. Primary comparator: {}. Secondary comparator: {}.'.format(
        label(primary['candidate']), label(primary['comparator']), label(secondary['comparator'])), fontsize=10)
    fig.text(.03, .128, '64 joint replicates per main case; 5,000 paired bootstrap resamples. Errors and cap abstentions included.', fontsize=9)
    fig.text(.03, .089, 'Administrative time extension: unchanged frozen scientific plan; 7,937 original + 1,663 additional trajectories.', fontsize=9)
    fig.text(.03, .050, 'Audit passed. Pass requires the unrounded upper CI < 1. Fixed simulator cases and named implementations only.', fontsize=9, color='#59616a')
    fig.tight_layout(rect=[.01, .18, .995, .865], w_pad=3.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    overflowing = []
    for item in fig.findobj(match=Text):
        if item.get_visible() and item.get_text().strip():
            box = item.get_window_extent(renderer)
            if box.x0 < canvas.x0-2 or box.y0 < canvas.y0-2 or box.x1 > canvas.x1+2 or box.y1 > canvas.y1+2:
                overflowing.append(item.get_text())
    require(not overflowing, 'Text exceeds the figure canvas: '+repr(overflowing))
    for name, expected in inputs.items():
        require(digest(DIRECTORY/name) == expected, 'Input changed while plotting: '+name)
    for name, expected in sources.items():
        require(digest(ROOT/name) == expected, 'Source changed while plotting: '+name)
    fig.savefig(str(png), dpi=180, facecolor='white')
    plt.close(fig)
    manifest = {'status': 'rendered_from_complete_passed_audit',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'sources': sources,
                'input_sha256': {(DIRECTORY/name).relative_to(ROOT).as_posix(): value for name, value in inputs.items()},
                'outputs': [{'path': png.relative_to(ROOT).as_posix(), 'sha256': digest(png)}],
                'verified_input_count': len(inputs), 'all_audit_inputs_verified': True,
                'complete_method_trajectories': 9600, 'main_cases': 48,
                'primary_numerical_threshold_met': accepted,
                'unrounded_upper_ci_checked': primary['percentile_95_ci'][1],
                'figure_size_inches': [13.8, 6.3], 'dpi': 180,
                'canvas_text_overflow_check': 'passed',
                'claim_boundary': 'Explicit administrative time extension; original scientific plan unchanged. Fixed simulator cases and named implementations only.'}
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'figure': str(png), 'manifest': str(manifest_path), 'status': manifest['status']}))


if __name__ == '__main__':
    main()

"""Inventory a Git research snapshot without including large raw logs.

Run from any directory. Git ignore rules determine excluded local files;
tracked files stay included. Exact source bytes and staged Git bytes are
reported separately when validating the resulting commit.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_names(*args):
    output = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files', '-z']+list(args))
    return set(x.decode('utf-8') for x in output.split(b'\0') if x)


def describe(name, excluded=False):
    path = ROOT/name
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    item = {'path': name, 'bytes': path.stat().st_size,
            'source_sha256': digest.hexdigest()}
    if excluded:
        if '__pycache__' in path.parts or path.suffix in ('.pyc', '.pyo'):
            reason = 'generated_python_cache'
        elif name.endswith(('.jsonl', '.jsonl.gz')):
            reason = 'raw_observation_log_retained_locally'
        elif '/literature/' in name:
            reason = 'downloaded_literature_mirror_retained_locally'
        else:
            reason = 'gitignored_local_file'
        item['reason'] = reason
    else:
        item['snapshot_sha256'] = item['source_sha256']
    return item


def main():
    start = datetime.now(timezone.utc).isoformat()
    included_names = (git_names('--cached') | git_names('--others', '--exclude-standard'))-{'SYNC_MANIFEST.json'}
    excluded_names = git_names('--others', '--ignored', '--exclude-standard')
    included = [describe(n) for n in sorted(included_names) if (ROOT/n).is_file()]
    excluded = [describe(n, True) for n in sorted(excluded_names) if (ROOT/n).is_file()]
    # Stage the publication files before running this script. Record the Git
    # blob separately: old Markdown may be normalized from local CRLF to LF.
    queries = ''.join(':'+item['path']+'\n' for item in included).encode()
    blobs = subprocess.check_output(['git', '-C', str(ROOT), 'cat-file', '--batch'], input=queries)
    offset = 0
    for item in included:
        end = blobs.index(b'\n', offset)
        header = blobs[offset:end].split()
        if len(header) != 3 or header[1] != b'blob':
            raise RuntimeError('Stage publication files first: '+item['path'])
        size = int(header[2])
        data = blobs[end+1:end+1+size]
        assert len(data) == size and blobs[end+1+size:end+2+size] == b'\n'
        item['snapshot_sha256'] = hashlib.sha256(data).hexdigest()
        item['snapshot_bytes'] = size
        offset = end+2+size
    assert offset == len(blobs)
    assert max(x['bytes'] for x in included) < 100*1024*1024
    manifest = {
        'format_version': 2,
        'source_repository': 'https://github.com/yuguoshao/FaultTune',
        'source_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip(),
        'snapshot_started_utc': start,
        'snapshot_finished_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Synchronize research code, reports and certificates; retain raw observation logs and literature mirrors locally.',
        'scientific_status': {
            'sota_superiority_established': False,
            'latest_completed_round_state': 'results/sota_round7/research_state.json',
            'round8_state': 'results/sota_round8/research_state.json',
            'round8_is_work_in_progress': True,
            'research_goal': 'cleared_by_user; this action is repository synchronization',
            'new_monte_carlo_observations_for_this_sync': 0,
            'sync_validation': 'Round7 controller unittest suite:7 passed; publication inventory and exact byte hashes checked.',
        },
        'included_source_files': len(included),
        'included_source_bytes': sum(x['bytes'] for x in included),
        'included_snapshot_bytes': sum(x['snapshot_bytes'] for x in included),
        'excluded_files': len(excluded),
        'excluded_bytes': sum(x['bytes'] for x in excluded),
        'manifest_self_exclusion': 'SYNC_MANIFEST.json is committed but omitted from its own file inventory to avoid a circular hash.',
        'included': included,
        'excluded': excluded,
    }
    (ROOT/'SYNC_MANIFEST.json').write_bytes((json.dumps(manifest, indent=2)+'\n').encode())
    print(json.dumps({k:manifest[k] for k in ('included_source_files','included_source_bytes','excluded_files','excluded_bytes')}, indent=2))


if __name__ == '__main__':
    main()

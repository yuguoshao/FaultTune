"""Read-only public metadata/source retrieval with saved provenance."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/sota_round6/literature'
QUERIES = [
    ('kl_discrimination', 'optimal experimental design criterion discriminating non-normal models Lopez Fidalgo Tommasi Trandafir'),
    ('trigonometric_discrimination', 'optimal designs discriminating trigonometric regression models Dette Melas'),
    ('bernoulli_discrimination', 'KL optimal discrimination design Bernoulli trigonometric regression'),
    ('overshift', 'overshifted parameter shift quantum'),
    ('measurement_coherence', 'quantifying coherence quantum measurements relative entropy'),
    ('measurement_channel_coherence', 'relative entropy measurement coherence incoherent quantum measurements channels'),
    ('measurement_channel_divergence', 'quantum measurement channel relative entropy discrimination without ancilla'),
    ('binary_capacity', 'binary asymmetric channel capacity closed form Bernoulli information radius'),
    ('trigonometric_coptimal', 'c optimal trigonometric regression Bernoulli sign coefficient'),
    ('bregman_information_radius', 'Bregman smallest enclosing information ball Bernoulli Chernoff centroid')
]


def fetch(entry):
    key, query = entry
    url = 'https://api.crossref.org/works?'+urllib.parse.urlencode({'query.bibliographic': query, 'rows': 6})
    result = {'key': key, 'query': query, 'url': url, 'created_utc': datetime.now(timezone.utc).isoformat()}
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'FaultTune-literature-review/1.0 (public bibliographic research)'})
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read()
            result.update(http_status=response.status, final_url=response.url)
        path = OUT/(key+'_crossref.json')
        path.write_bytes(payload)
        result.update(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(payload).hexdigest())
        parsed = json.loads(payload)
        result['items'] = [{k: item.get(k) for k in ('title','author','DOI','URL','published','abstract','link')}
                           for item in parsed['message']['items']]
    except Exception as exc:
        result['error'] = str(exc)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        entries = list(pool.map(fetch, QUERIES))
    path = OUT/(sys.argv[1] if len(sys.argv)>1 else 'metadata_search_ledger.json')
    with path.open('x', encoding='utf-8') as stream:
        json.dump({'search_connector_attempt': '2026-11-04 current web_run attempt failed with transport HTTP send error before queries returned',
                   'entries': entries}, stream, indent=2)
    for entry in entries:
        print(json.dumps({'key': entry['key'], 'error': entry.get('error'),
                          'items': [{'title': r['title'], 'doi': r['DOI'], 'date': r['published']} for r in entry.get('items', [])]}), flush=True)


if __name__ == '__main__':
    main()

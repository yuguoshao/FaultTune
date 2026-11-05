"""Extract exact source anchors and fetch one specifically applicable comparator."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import urllib.request
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/sota_round7/literature'


def inspect_sources():
    for name, number in [('tendick2023', '108'), ('mitra2025', '55')]:
        abstract = BeautifulSoup((OUT / (name + '_abs.html')).read_bytes(), 'html.parser')
        print(name, 'version', abstract.select_one('.submission-history').get_text(' ', strip=True), flush=True)
        soup = BeautifulSoup((OUT / (name + '_full.html')).read_bytes(), 'html.parser')
        for equation in soup.select('.ltx_equation'):
            tags = equation.select('.ltx_tag_equation')
            if any(t.get_text(strip=True) == '(' + number + ')' for t in tags):
                print(name, 'equation', equation.get('id'), equation.get_text(' ', strip=True), flush=True)
        for heading in soup.select('h1,h2,h3,h4'):
            if any(word in heading.get_text(' ', strip=True).lower() for word in ['dichotomic', 'general distance']):
                print(name, 'heading', heading.parent.get('id'), heading.get_text(' ', strip=True), flush=True)


def main():
    inspect_sources()
    records = []
    for kind in ['abs', 'html']:
        url = 'https://arxiv.org/' + kind + '/1502.02677'
        record = {'url': url, 'created_utc': datetime.now(timezone.utc).isoformat()}
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (public research source verification)'})
            with urllib.request.urlopen(req, timeout=25) as response:
                data = response.read()
                record.update(http_status=response.status, final_url=response.url)
            path = OUT / ('kimmel2015_' + kind + '.html')
            with path.open('xb') as stream:
                stream.write(data)
            soup = BeautifulSoup(data, 'html.parser')
            extracted = '\n'.join(n.get_text(' ', strip=True) for n in soup.select('h1,h2,h3,h4,p,.ltx_equation,.ltx_bibitem'))
            with path.with_suffix('.txt').open('x', encoding='utf-8') as stream:
                stream.write(extracted)
            record.update(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(data).hexdigest(), text_sha256=hashlib.sha256(path.with_suffix('.txt').read_bytes()).hexdigest(), bytes=len(data))
        except Exception as exc:
            record['error'] = str(exc)
        records.append(record)
        print(json.dumps(record), flush=True)
    with (OUT / 'rpe_source_ledger.json').open('x', encoding='utf-8') as stream:
        json.dump({'records': records}, stream, indent=2)


if __name__ == '__main__':
    main()

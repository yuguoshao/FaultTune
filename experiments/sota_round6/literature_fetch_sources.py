"""Fetch a fixed list of public publisher/arXiv URLs; save every outcome."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/sota_round6/literature'


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    records = []
    for entry in jobs['jobs']:
        row = dict(entry, created_utc=datetime.now(timezone.utc).isoformat())
        try:
            request = urllib.request.Request(entry['url'], headers={'User-Agent':'Mozilla/5.0 (academic literature review; public sources only)'})
            with urllib.request.urlopen(request, timeout=35) as response:
                payload = response.read()
                row.update(http_status=response.status, final_url=response.url, content_type=response.headers.get('Content-Type'))
            destination = OUT/entry['filename']
            with destination.open('xb') as stream:
                stream.write(payload)
            row.update(path=destination.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        except Exception as exc:
            row['error'] = str(exc)
        records.append(row)
        print(json.dumps({k: row[k] for k in ('key','http_status','bytes','error') if k in row}), flush=True)
        time.sleep(2)
    with (OUT/jobs['ledger']).open('x',encoding='utf-8') as stream:
        json.dump({'records':records},stream,indent=2)


if __name__ == '__main__':
    main()

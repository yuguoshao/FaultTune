"""Obtain complete original when arXiv HTML stops before relevant sections."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/sota_round8/literature'
url='https://arxiv.org/pdf/1611.02427v2'
record={'url':url,'created_utc':datetime.now(timezone.utc).isoformat(),'reason':'HTML conversion ends at Section III and does not contain echo sensing sections.'}
try:
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (targeted public original verification)'})
    with urllib.request.urlopen(req,timeout=30) as response:
        data=response.read();record.update(http_status=response.status,final_url=response.url)
    path=OUT/'degen2017_v2.pdf'
    with path.open('xb') as stream:stream.write(data)
    record.update(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
except Exception as exc:record['error']=str(exc)
with (OUT/'sensing_pdf_source_ledger.json').open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
print(json.dumps(record))

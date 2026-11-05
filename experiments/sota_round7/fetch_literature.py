"""Read-only targeted public original-text retrieval with provenance."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import urllib.request
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/sota_round7/literature'
JOBS=[('tendick2023_abs','https://arxiv.org/abs/2205.08546'),
      ('tendick2023_full','https://arxiv.org/html/2205.08546'),
      ('mitra2025_abs','https://arxiv.org/abs/2505.11331'),
      ('mitra2025_full','https://arxiv.org/html/2505.11331')]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for key,url in JOBS:
        record={'key':key,'url':url,'created_utc':datetime.now(timezone.utc).isoformat()}
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (public original literature review)'})
            with urllib.request.urlopen(req,timeout=35) as response:
                data=response.read()
                record.update(http_status=response.status,final_url=response.url,content_type=response.headers.get('Content-Type'))
            path=OUT/(key+'.html')
            with path.open('xb') as stream:stream.write(data)
            record.update(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
            soup=BeautifulSoup(data,'html.parser')
            text='\n'.join(n.get_text(' ',strip=True) for n in soup.select('h1,h2,h3,h4,p,.ltx_equation,.ltx_bibitem'))
            with path.with_suffix('.txt').open('x',encoding='utf-8') as stream:stream.write(text)
            record['text_sha256']=hashlib.sha256(path.with_suffix('.txt').read_bytes()).hexdigest()
        except Exception as exc:
            record['error']=str(exc)
        records.append(record)
        print(json.dumps({k:record[k] for k in ('key','http_status','bytes','error') if k in record}),flush=True)
    with (OUT/'public_source_ledger.json').open('x',encoding='utf-8') as stream:
        json.dump({'records':records},stream,indent=2)


if __name__=='__main__':main()

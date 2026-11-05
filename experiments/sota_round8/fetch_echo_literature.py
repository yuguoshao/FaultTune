"""Targeted public-original retrieval for the echoed crosstalk route."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import urllib.request
from bs4 import BeautifulSoup

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/sota_round8/literature'
JOBS=[
    ('sheldon2016_abs','https://arxiv.org/abs/1603.04821'),
    ('sheldon2016_full','https://arxiv.org/html/1603.04821'),
    ('dai2021_abs','https://arxiv.org/abs/2105.14360'),
    ('dai2021_full','https://arxiv.org/html/2105.14360'),
    ('degen2017_abs','https://arxiv.org/abs/1611.02427'),
    ('degen2017_full','https://arxiv.org/html/1611.02427'),
]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for name,url in JOBS:
        record={'name':name,'url':url,'created_utc':datetime.now(timezone.utc).isoformat()}
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (public original research verification)'})
            with urllib.request.urlopen(req,timeout=25) as response:
                data=response.read()
                record.update(http_status=response.status,final_url=response.url)
            path=OUT/(name+'.html')
            with path.open('xb') as stream:stream.write(data)
            soup=BeautifulSoup(data,'html.parser')
            extracted='\n'.join(node.get_text(' ',strip=True) for node in soup.select('h1,h2,h3,h4,p,.ltx_equation,.ltx_bibitem'))
            with path.with_suffix('.txt').open('x',encoding='utf-8') as stream:stream.write(extracted)
            title=soup.select_one('h1')
            record.update(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(data).hexdigest(),text_sha256=hashlib.sha256(path.with_suffix('.txt').read_bytes()).hexdigest(),title=title.get_text(' ',strip=True) if title else None,bytes=len(data))
        except Exception as exc:
            record['error']=str(exc)
        records.append(record)
        print(json.dumps(record),flush=True)
    with (OUT/'public_source_ledger.json').open('x',encoding='utf-8') as stream:json.dump({'records':records,'search_connector_failure':'2026-11-05T06:58:50Z transport failed for chatgpt.com/backend-api/ps/mcp; targeted public originals used instead.'},stream,indent=2)


if __name__=='__main__':main()

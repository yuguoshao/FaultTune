"""Download public literature metadata/documents only, never execute sources."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/sota_round3/literature'
SOURCES = {
 'sequential_index.html':'https://cran.r-project.org/web/packages/Sequential/index.html',
 'sequential_manual.pdf':'https://cran.r-project.org/web/packages/Sequential/Sequential.pdf',
 'simctest_index.html':'https://cran.r-project.org/web/packages/simctest/index.html',
 'simctest_manual.pdf':'https://cran.r-project.org/web/packages/simctest/simctest.pdf',
 'gandy_crossref.json':'https://api.crossref.org/works?query.title=Sequential%20implementation%20of%20Monte%20Carlo%20tests%20with%20uniformly%20bounded%20resampling%20risk&rows=3',
 'howard_abs.html':'https://arxiv.org/abs/1810.08240',
 'howard_full.html':'https://arxiv.org/html/1810.08240',
 'howard_full.pdf':'https://arxiv.org/pdf/1810.08240',
 'lademets_crossref.json':'https://api.crossref.org/works/10.1093/biomet/70.3.659',
 'maxsprt_crossref.json':'https://api.crossref.org/works/10.1080/07474946.2011.539924',
 'johari_abs.html':'https://arxiv.org/abs/1512.04922',
 'johari_full.html':'https://arxiv.org/html/1512.04922',
 'gandy_home.html':'https://www.ma.imperial.ac.uk/~agandy/',
 'gandy_literature.html':'https://www.ma.imperial.ac.uk/~agandy/lit.html',
 'sequential_4.6.3.tar.gz':'https://cran.r-project.org/src/contrib/Sequential_4.6.3.tar.gz',
 'simctest_intro.pdf':'https://cran.r-project.org/web/packages/simctest/vignettes/simctest-intro.pdf',
 'optimal_spending_crossref.json':'https://api.crossref.org/works?query.title=Optimal%20alpha%20spending%20for%20sequential%20analysis%20with%20binomial%20data&rows=2',
 'exact_sequential_crossref.json':'https://api.crossref.org/works/10.1002/sim.9094',
}

class Extract(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text, self.links, self.hide = [], [], 0
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ('script', 'style'): self.hide += 1
        if tag in ('p','div','br','li','h1','h2','h3','section','tr'): self.text.append('\n')
        if tag == 'math' and attrs.get('alttext'): self.text.append(' [LATEX: '+attrs['alttext']+'] ')
        if tag == 'a' and attrs.get('href'): self.links.append(attrs['href'])
    def handle_endtag(self, tag):
        if tag in ('script','style'): self.hide = max(0,self.hide-1)
    def handle_data(self, data):
        if not self.hide: self.text.append(data)

def fetch(item):
    name,url = item
    record = {'file':name,'url':url}
    try:
        path = OUT/name
        if path.exists():
            raw = path.read_bytes()
            record['status'] = 'already_cached'
        else:
            request = urllib.request.Request(url,headers={'User-Agent':'FaultTune-public-literature-check/1.0'})
            with urllib.request.urlopen(request,timeout=40) as response:
                raw = response.read()
                record.update(final_url=response.geturl(),content_type=response.headers.get('Content-Type'))
            path.write_bytes(raw)
            record['status'] = 'downloaded'
            if name.endswith('.html'):
                parser=Extract();parser.feed(raw.decode('utf-8','replace'))
                path.with_suffix('.txt').write_text('\n'.join(line.strip() for line in ''.join(parser.text).splitlines() if line.strip()),encoding='utf-8')
                path.with_suffix('.links.json').write_text(json.dumps(sorted(set(parser.links)),indent=2)+'\n',encoding='utf-8')
        record.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    except Exception as error:
        record.update(status='failed',error=repr(error))
    return record

if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(fetch,SOURCES.items()))
    manifest={'checked_utc':datetime.now(timezone.utc).isoformat(),'records':rows,
              'scope':'Public primary documents and metadata; no downloaded code executed and no observations generated.'}
    (OUT/'downloads.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))

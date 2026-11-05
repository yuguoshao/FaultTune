"""Cache public primary sources; never changes prior rounds or executes downloads."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
DESTINATION = ROOT / "results/sota_round2/literature"
SOURCES = {
    "dkm_abs.html": "https://arxiv.org/abs/1906.10431",
    "dkm_ar5iv.html": "https://ar5iv.labs.arxiv.org/html/1906.10431",
    "dkm_neurips.pdf": "https://proceedings.neurips.cc/paper_files/paper/2019/file/8d1de7457fa769ece8d93a13a59c8552-Paper.pdf",
    "fw_abs.html": "https://arxiv.org/abs/2509.19901",
    "fw_latest.html": "https://arxiv.org/html/2509.19901",
    "dkm_github_search.json": "https://api.github.com/search/repositories?q=Non-Asymptotic+Pure+Exploration+Games",
    "fw_github_search.json": "https://api.github.com/search/repositories?q=Frank-Wolfe+Self-Play",
    "dkm_arxiv.pdf": "https://arxiv.org/pdf/1906.10431v1",
    "kk_abs.html": "https://arxiv.org/abs/1811.11419",
    "kk_arxiv.pdf": "https://arxiv.org/pdf/1811.11419",
    "degenne_github_repos.json": "https://api.github.com/users/RemyDegenne/repos?per_page=100",
    "koolen_github_repos.json": "https://api.github.com/users/wouterkoolen/repos?per_page=100",
    "degenne_home.html": "https://remydegenne.github.io/",
    "koolen_home.html": "https://wouterkoolen.info/",
    "dkm_code_readme_search.json": "https://api.github.com/search/repositories?q=1906.10431+in:readme&per_page=20",
    "fw_code_readme_search.json": "https://api.github.com/search/repositories?q=2509.19901+in:readme&per_page=20",
    "koolen_software.html": "https://wouterkoolen.info/software.html",
    "koolen_publications.html": "https://wouterkoolen.info/publications.html",
    "tidnabbil_repo.json": "https://api.bitbucket.org/2.0/repositories/wmkoolen/tidnabbil",
    "tidnabbil_root.json": "https://api.bitbucket.org/2.0/repositories/wmkoolen/tidnabbil/src/master/?pagelen=100",
    "purex_games_repo.json": "https://api.bitbucket.org/2.0/repositories/wmkoolen/purex_games",
    "adahedge_jmlr.pdf": "https://jmlr.org/papers/volume15/rooij14a/rooij14a.pdf",
    "tidnabbil_README.md": "https://api.bitbucket.org/2.0/repositories/wmkoolen/tidnabbil/src/fdb307afcee834cee17fa849f3f6ca7e4d518a64/README.md",
    "tidnabbil_peps.jl": "https://api.bitbucket.org/2.0/repositories/wmkoolen/tidnabbil/src/fdb307afcee834cee17fa849f3f6ca7e4d518a64/peps.jl",
    "tidnabbil_tracking.jl": "https://api.bitbucket.org/2.0/repositories/wmkoolen/tidnabbil/src/fdb307afcee834cee17fa849f3f6ca7e4d518a64/tracking.jl",
    "tidnabbil_thresholds.jl": "https://api.bitbucket.org/2.0/repositories/wmkoolen/tidnabbil/src/fdb307afcee834cee17fa849f3f6ca7e4d518a64/thresholds.jl",
    "tidnabbil_go.jl": "https://api.bitbucket.org/2.0/repositories/wmkoolen/tidnabbil/src/fdb307afcee834cee17fa849f3f6ca7e4d518a64/go.jl",
    "tidnabbil_purex_directory.json": "https://api.bitbucket.org/2.0/repositories/wmkoolen/tidnabbil/src/fdb307afcee834cee17fa849f3f6ca7e4d518a64/purex_games_paper/?pagelen=100",
    "tidnabbil_samplingrules.jl": "https://api.bitbucket.org/2.0/repositories/wmkoolen/tidnabbil/src/fdb307afcee834cee17fa849f3f6ca7e4d518a64/purex_games_paper/samplingrules.jl",
    "tidnabbil_runit.jl": "https://api.bitbucket.org/2.0/repositories/wmkoolen/tidnabbil/src/fdb307afcee834cee17fa849f3f6ca7e4d518a64/purex_games_paper/runit.jl",
    "tidnabbil_purex_README.txt": "https://api.bitbucket.org/2.0/repositories/wmkoolen/tidnabbil/src/fdb307afcee834cee17fa849f3f6ca7e4d518a64/purex_games_paper/README",
}


class Extract(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.links, self.hide = [], [], 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("script", "style"):
            self.hide += 1
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "section", "tr"):
            self.parts.append("\n")
        if tag == "math" and attrs.get("alttext"):
            self.parts.append(" [LATEX: " + attrs["alttext"] + "] ")
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.hide = max(0, self.hide - 1)

    def handle_data(self, data):
        if not self.hide:
            self.parts.append(data)


def fetch(item):
    name, url = item
    result = {"requested_url": url, "file": name}
    try:
        path = DESTINATION / name
        if path.exists():
            raw = path.read_bytes()
            result.update(status="already_cached", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            return result
        request = urllib.request.Request(url, headers={"User-Agent": "FaultTune-public-source-verification/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
            result.update(final_url=response.geturl(), content_type=response.headers.get("Content-Type"))
        path.write_bytes(raw)
        result.update(status="downloaded", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if name.endswith(".html"):
            parser = Extract()
            parser.feed(raw.decode("utf-8", "replace"))
            path.with_suffix(".txt").write_text("\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip()), encoding="utf-8")
            path.with_suffix(".links.json").write_text(json.dumps(sorted(set(parser.links)), indent=2) + "\n", encoding="utf-8")
    except Exception as error:
        result.update(status="failed", error=repr(error))
    return result


if __name__ == "__main__":
    DESTINATION.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, SOURCES.items()))
    manifest = {"checked_utc": datetime.now(timezone.utc).isoformat(), "sources": records,
                "scope": "Primary-source downloads and metadata only; no external code executed."}
    (DESTINATION / "source_downloads.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))

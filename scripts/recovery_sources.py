"""Fetch public MODMA recovery metadata; never submits forms or sends messages."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'recovery' / 'sources'
TARGETS = {
    'kaggle_modma_search.json': 'https://www.kaggle.com/api/v1/datasets/list?search=modma&page=1',
    'hf_modma_search.json': 'https://huggingface.co/api/datasets?search=modma&limit=100',
    'modma_issues.json': 'https://api.github.com/repos/UAIS-LANZHOU/MODMA-Dataset/issues?state=all&per_page=100',
    'tri_dep_issues.json': 'https://api.github.com/repos/annisaafitrinn/tri-dep1/issues?state=all&per_page=100',
    'modma_readme.md': 'https://raw.githubusercontent.com/UAIS-LANZHOU/MODMA-Dataset/main/README.md',
    'reshare.html': 'https://reshare.ukdataservice.ac.uk/854301/',
    'official_application.html': 'https://modma.lzu.edu.cn/data/application/',
    'known_defect_2021.xml': 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8296267/fullTextXML',
}

def fetch(item):
    name, url = item
    record = {'url': url, 'checked_utc': datetime.now(timezone.utc).isoformat()}
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'MODMA-S00-recovery-audit/1.0'})
        with urllib.request.urlopen(req, timeout=35) as response:
            data = response.read(12_000_001)
            if len(data) > 12_000_000:
                raise ValueError('metadata response exceeds 12 MB limit')
            resolved=urllib.parse.urlsplit(response.url)
            # Temporary public download signatures are unnecessary in provenance.
            final_url=urllib.parse.urlunsplit((resolved.scheme,resolved.netloc,resolved.path,'','')) if resolved.hostname=='storage.googleapis.com' else response.url
            record.update(status=response.status, final_url=final_url,
                          content_type=response.headers.get('Content-Type'))
        (OUT / name).write_bytes(data)
        record.update(path=str((OUT / name).relative_to(ROOT)), bytes=len(data),
                      sha256=hashlib.sha256(data).hexdigest())
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    return name, record

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        result = dict(pool.map(fetch, TARGETS.items()))
    (OUT / 'index.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))

if __name__ == '__main__':
    main()

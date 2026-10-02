from __future__ import annotations
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

WORK = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = WORK / 'tridep_ccml/experiments/EEG-E2-37-fulltext'
REPORT = WORK / 'output/analysis/eeg_e2_20260929'
PLAN = WORK / 'output/plans/EEG先行_整体改进双主线计划_v1.md'
R1 = WORK / 'tridep_ccml/experiments/EEG-R1-37-fulltext'
P3 = WORK / 'tridep_ccml/experiments/EEG-P3-37-fulltext'
P2 = WORK / 'output/analysis/eeg_quality_scale_20260928'
REVISION = WORK / 'tridep_ccml/reports/ccml_v2_audit/eeg_channel_revision.json'

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()

def desc(path):
    path = Path(path)
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': digest(path)}

def save(path, data, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    if immutable and path.exists():
        assert read(path) == data, 'Immutable artifact differs: ' + str(path)
        return
    path.write_text(value, encoding='utf-8')

def freeze_history():
    roots = [R1, WORK/'tridep_ccml/experiments/EEG-P1-37-fulltext', P2, P3,
             WORK/'output/analysis/eeg_generalization_20260929']
    result = {}
    for root in roots:
        manifest = root/'deliverable_hashes.json'
        if not manifest.exists():
            continue
        entries = read(manifest)
        for name, sha in entries.items():
            assert digest(root/name) == sha, 'Old frozen artifact changed: ' + str(root/name)
        result[root.relative_to(WORK).as_posix()] = {'files': len(entries), 'manifest': desc(manifest)}
    return result

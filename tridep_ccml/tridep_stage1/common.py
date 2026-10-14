from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import numpy as np
import torch

WORK = Path(__file__).resolve().parents[2]
CODE = WORK / 'tridep_ccml/tridep_stage1'
OUT = WORK / 'tridep_ccml/experiments/TriDep-Stage1-EEG-37-fulltext'
REPORT = WORK / 'output/analysis/tridep_stage1_20261005'
STAGE0 = WORK / 'output/analysis/tridep_stage0_20261004'
ANNEX = WORK / 'output/plans/TriDep_阶段1_EEG结构与骨干执行附表_v1_20261005.md'
CONDITIONS = ['CB-mean', 'CB-spatial', 'CB-temporal', 'CB-spatiotemporal', 'CB-originalCNN', 'LaBraM-originalCNN']
SEEDS = [11, 22, 33]
MODALITIES = ['eeg', 'speech', 'text']


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def desc(path):
    return {'path': Path(path).resolve().relative_to(WORK).as_posix(), 'sha256': digest(path)}


def checked(record):
    p = (WORK / record['path']).resolve()
    if not p.is_relative_to(WORK) or digest(p) != record['sha256']:
        raise ValueError('Source changed: ' + str(p))
    return p


def save(path, data, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read(path) != data:
            raise ValueError('Existing frozen record differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def save_rows(path, rr):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rr), encoding='utf-8')


def runtime(seed):
    if not torch.cuda.is_available():
        raise ValueError('This registered paired run requires CUDA')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True


def state_digest(state):
    h = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        x = tensor.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def context():
    sys.path.insert(0, str(WORK / 'S00/scripts'))
    from cohort import validate_active_cohort
    lock = validate_active_cohort(require_data=True)
    fold_data = read(WORK / 'S00' / lock['fold_file'])
    labels = {r['subject_id']: r['label'] for r in read(WORK / 'S00/manifests/subjects.json')}
    return lock, fold_data, labels

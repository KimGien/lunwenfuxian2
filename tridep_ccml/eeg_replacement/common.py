from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from tridep_ccml.records import RecordContext, digest, canonical_digest, read_json, require
from tridep_ccml.v2.common import descriptor, verify_descriptor, set_seed, write_new, save_array, LOSS
from tridep_ccml.train import state_digest

WORK = Path(__file__).resolve().parents[2]
TC = WORK / 'tridep_ccml'
OUT = TC / 'experiments/EEG-R1-37-fulltext'
CODE = Path(__file__).resolve().parent
PLAN = WORK / 'output/plans/EEG分支替换执行计划_R1.md'
REVISION = TC / 'reports/ccml_v2_audit/eeg_channel_revision.json'
SEEDS = [11, 22, 33]
FOLDS = [1, 2, 3, 4, 5]
CONDITIONS = ['U0', 'U1', 'C0', 'C1', 'U2', 'C2', 'C3', 'C4']
STAGES = [['U0', 'U1', 'C0', 'C1'], ['U2', 'C2'], ['C3', 'C4']]
MODALITIES = ['eeg', 'speech', 'text']


def now():
    return datetime.now(timezone.utc).isoformat()


def replace_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    # Windows readers may briefly hold a handle without delete-sharing.
    # Keep the complete temporary file and retry only this transient replacement.
    for attempt in range(101):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 100:
                raise
            time.sleep(.05)


def run_dir(condition, seed, fold):
    return OUT / f'runs/{condition}/seed_{seed}/fold_{fold}'


def source_hashes():
    files = list(CODE.glob('*.py')) + [PLAN, REVISION, TC/'ccml.py', TC/'metrics.py',
        TC/'records.py', TC/'train.py', TC/'v2/common.py', TC/'v2/extract_eeg.py',
        TC/'v2/scale.py', WORK/'S00/upstream-dev/lib/models/models.py']
    return {p.relative_to(WORK).as_posix(): digest(p) for p in sorted(files)}


def verify_lock():
    cfg = read_json(OUT/'configuration.json')
    require(cfg['sources'] == source_hashes(), 'Locked R1 implementation changed')
    require(cfg['augmentation'] is False and cfg['epochs'] == 60 and cfg['conditions'] == CONDITIONS,
            'R1 fixed protocol changed')
    for key in ['data_manifest', 'structure_manifest', 'preflight']:
        verify_descriptor(cfg[key])
    return cfg

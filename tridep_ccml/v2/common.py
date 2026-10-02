from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch
from tridep_ccml.records import RecordContext, digest, read_json, require

WORK = Path(__file__).resolve().parents[2]
TC = WORK / 'tridep_ccml'
OUT = TC / 'experiments/CCML-v2-E-37-fulltext'
REVISION = TC / 'reports/ccml_v2_audit/eeg_channel_revision.json'
LOSS = {'rho': 1.25, 'delta': 1., 'gamma': 1., 'anneal_epochs': 50}
ADAPT = 'CCML-v2-E-adapt-37-fulltext'
CONTROL = 'CCML-v2-E-control-37-fulltext'


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(read_json(path) == value, f'Refusing different existing artifact: {path}')
        return
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def descriptor(path):
    path = Path(path)
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': digest(path)}


def verify_descriptor(item):
    path = WORK / item['path']
    require(digest(path) == item['sha256'], f'Artifact hash changed: {path}')
    return path


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def label_indices(labels, required):
    require(len(set(required)) == len(required), 'Duplicate requested electrode')
    require(all(labels.count(name) == 1 for name in required), 'Missing/duplicate source electrode label')
    return [labels.index(name) for name in required]


def configuration(context=None):
    context = context or RecordContext()
    return {'experiment_family': 'CCML-v2-E-37-fulltext', 'version': 1,
            'cohort_id': context.lock['cohort_id'], 'cohort_sha256': context.cohort_sha256,
            'folds_sha256': context.folds_sha256, 'subject_ids': context.ids,
            'seeds': [11, 22, 33], 'folds': [1, 2, 3, 4, 5],
            'revision': descriptor(REVISION), 'epochs': 60, 'batch_size': 8,
            'optimizer': 'Adamax', 'eeg_lr': .0001, 'head_lr': .001,
            'weight_decay': 0., 'gradient_clip': 1., 'ccml': LOSS,
            'threshold': .5, 'tie_class': 0, 'checkpoint_selection': 'fixed_last_epoch',
            'parent_adaptation': 'same-fold EEG CNN+projection warm start; all retained EEG aggregator parameters updated with CCML; speech/text cached representations frozen',
            'historical_supervised_initialization': 'S00 CE-trained EEG weights are reused, not retrained with CE; speech/text also retain historical supervised ancestry',
            'dropout_mode': 'eval during adaptation and export, gradients enabled for retained EEG parameters',
            'backbone': 'frozen Mumtaz CBraMod', 'eeg_unit_policy': 'native numerical values unchanged',
            'head_control': 'freeze revised 128D representations; fresh shared seed-specific 774-parameter heads',
            'adaptation_runs': 15, 'control_runs': 15, 'total_updates': 7200,
            'validation_or_test_selection': False, 'nr_started': False,
            'causal_scope': 'channel correction plus CCML EEG adaptation; historical TC04 comparison does not isolate the channel-only effect'}


def protect_existing():
    p = OUT / 'historical_files_before.json'
    if p.exists():
        return read_json(p)
    files = set()
    for folder in ['configs', 'manifests', 'features', 'runs', 'reports', 'references']:
        files.update(x for x in (TC/folder).rglob('*') if x.is_file() and '__pycache__' not in x.parts)
    files.update(TC.glob('*.py'))
    for folder in ['configs', 'manifests', 'scripts']:
        files.update(x for x in (WORK/'S00'/folder).rglob('*') if x.is_file() and '__pycache__' not in x.parts)
    files.add(WORK/'S00/upstream-dev/lib/models/models.py')
    files.add(WORK/'output/plans/CCML_v2_性能优化与后续计划.md')
    result = {x.relative_to(WORK).as_posix(): digest(x) for x in sorted(files)}
    write_new(p, result)
    return result


def check_existing():
    files = read_json(OUT/'historical_files_before.json')
    require(all(digest(WORK/p) == h for p,h in files.items()), 'A protected historical artifact changed')
    return len(files)


def setup():
    context = RecordContext()
    OUT.mkdir(parents=True, exist_ok=True)
    protect_existing()
    write_new(OUT/'configuration.json', configuration(context))
    return context, read_json(OUT/'configuration.json')


def save_array(path, array):
    require(np.isfinite(array).all(), 'Nonfinite array')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(np.array_equal(np.load(path, allow_pickle=False), array), f'Existing array differs: {path}')
        return
    temporary = path.with_suffix('.npy.part')
    with temporary.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)

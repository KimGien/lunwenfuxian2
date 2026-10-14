from __future__ import annotations

from pathlib import Path
from tridep_ccml.tridep_stage1.common import (
    WORK, STAGE0, now, read, rows, digest, desc, checked, save, save_rows,
    runtime, state_digest, context, np, torch,
)

CODE = WORK / 'tridep_ccml/tridep_eegpt'
OUT = WORK / 'tridep_ccml/experiments/TriDep-Stage4-EEGPT-Matched4s-37-fulltext-20261006'
REPORT = WORK / 'output/analysis/tridep_eegpt_20261006'
ANNEX = WORK / 'output/plans/TriDep_EEGPT匹配4秒备选执行附表_v1_20261006.md'
CONDITIONS = ['LaBraM-native4s-originalCNN', 'EEGPT-mean-projection4s-originalCNN']
SEEDS = [11, 22, 33]
KEYS = ['macro_f1', 'balanced_accuracy', 'mdd_recall', 'auc', 'nll', 'brier']
VIEWS = {'eeg': 'p_eeg', 'fixed_EST': 'p_fixed_EST'}
TRAINING = {'epochs': 60, 'batch_size': 16, 'optimizer': 'Adamax',
            'learning_rate': .005, 'weight_decay': 0., 'gradient_clip': None,
            'parameters': 498434, 'updates_per_model': 120,
            'checkpoint_selection': 'fixed_final_epoch'}
FEATURE_BUDGET = {
    'eegpt_first_windows': 2220, 'eegpt_repeat_windows': 296,
    'eegpt_total_windows': 2516, 'eegpt_batches': 333,
    'labram_first_windows': 2220, 'labram_repeat_windows': 296,
    'labram_total_windows': 2516, 'labram_batches': 333,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def registration():
    path = checked(read(OUT / 'registration_anchor.json')['registration'])
    reg = read(path)
    require(reg['conditions'] == CONDITIONS and reg['seeds'] == SEEDS and
            reg['training'] == {c: TRAINING for c in CONDITIONS} and
            reg['formal_fits'] == 30 and reg['formal_updates'] == 3600 and
            reg['budget']['features'] == FEATURE_BUDGET and
            reg['inner_fits'] == reg['calibrator_fits'] == reg['threshold_searches'] == 0 and
            reg['fusion_selections'] == 0 and reg['model_execution_allowed'],
            'Registered matched design or fitting budget changed')
    for name, expected in reg['sources'].items():
        checked({'path': name, 'sha256': expected})
    return reg


def begin(action):
    registration()
    path = OUT / 'actions' / (action + '.json')
    require(not path.exists(), 'Action already started; preserve evidence and investigate: ' + action)
    save(path, {'action': action, 'status': 'started', 'started_at': now(),
                'registration': desc(OUT / 'registration.json')}, immutable=True)


def finish(action):
    path = OUT / 'actions' / (action + '.json')
    result = read(path)
    require(result['status'] == 'started', 'Action completion requires its started record')
    result.update(status='complete', finished_at=now())
    save(path, result)


def failed(action, error):
    path = OUT / 'actions' / (action + '.json')
    result = read(path)
    result.update(status='failed', stopped_at=now(), error_type=type(error).__name__, error=str(error))
    save(path, result)

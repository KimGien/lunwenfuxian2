from __future__ import annotations
from tridep_ccml.eeg_replacement.common import (WORK, TC, RecordContext, read_json,
    digest, canonical_digest, require, descriptor, verify_descriptor, set_seed,
    write_new, replace_json, state_digest, now, SEEDS, FOLDS, MODALITIES, LOSS, np, torch, Path)
from tridep_ccml.eeg_replacement.common import OUT as PARENT
from tridep_ccml.eeg_replacement.common import verify_lock as verify_parent_lock

OUT = TC/'experiments/EEG-P1-37-fulltext'
CODE = Path(__file__).resolve().parent
PLAN = WORK/'output/plans/EEG_P1有效学习验收与SparNet修复固定计划.md'
CONDITIONS = ['U0','U2','C0','C2']
GATES = {'gradient_l2_min':1e-12, 'representation_std_rms_min':1e-6,
         'representation_relative_std_min':1e-4, 'zero_input_response_min':1e-4,
         'region_response_min':1e-6, 'small_fit_steps':500, 'small_fit_ce_max':.1,
         'small_fit_accuracy_min':1., 'small_fit_lr':1e-3, 'small_fit_weight_decay':0.}


def run_dir(condition,seed,fold):
    return OUT/f'runs/{condition}/seed_{seed}/fold_{fold}'


def sources():
    return {p.relative_to(WORK).as_posix():digest(p) for p in sorted(list(CODE.glob('*.py'))+[PLAN])}


def check_history():
    verify_parent_lock()
    hashes = read_json(PARENT/'deliverable_hashes.json')
    for name,sha in hashes.items():
        require(digest(PARENT/name)==sha, 'Frozen R1 artifact changed: '+name)
    return len(hashes)


def verify_lock():
    cfg = read_json(OUT/'configuration.json')
    require(cfg['sources']==sources(),'P1 locked implementation changed')
    require(cfg['gates']==GATES and cfg['conditions']==CONDITIONS and not cfg['augmentation'], 'P1 protocol changed')
    verify_parent_lock()
    for key in ['parent_configuration','data_manifest','preflight','structure_manifest']:
        verify_descriptor(cfg[key])
    return cfg

from __future__ import annotations
from tridep_ccml.eeg_replacement.common import *
from tridep_ccml.eeg_replacement.common import OUT as R1
from tridep_ccml.eeg_p2.common import history_check as previous_history_check

CODE = Path(__file__).resolve().parent
OUT = TC/'experiments/EEG-P3-37-fulltext'
REPORT = WORK/'output/analysis/eeg_generalization_20260929'
PLAN = WORK/'output/plans/EEG_P3泛化BN与校准固定计划.md'
P2 = WORK/'output/analysis/eeg_quality_scale_20260928'
CORE = ['__init__.py','common.py','methods.py','engine.py','test_contracts.py','run.py']

def source_hashes():
    return {p.relative_to(WORK).as_posix():digest(p) for p in [CODE/n for n in CORE]+[PLAN]}

def history_check():
    result = previous_history_check()
    hashes = read_json(P2/'deliverable_hashes.json')
    for name, sha in hashes.items():
        require(digest(P2/name)==sha,'P2 frozen artifact changed: '+name)
    result['P2'] = len(hashes)
    return result

def lock_check():
    cfg=read_json(OUT/'configuration.json')
    require(cfg['sources']==source_hashes(),'P3 registered code changed')
    return cfg

def folder(c,arm,seed,fold,inner=None):
    base=OUT/f'runs/{c}_{arm}/seed_{seed}/fold_{fold}'
    return base if inner is None else base/f'inner_{inner}'

def named_parameters_digest(model):
    return state_digest(dict(model.named_parameters()))

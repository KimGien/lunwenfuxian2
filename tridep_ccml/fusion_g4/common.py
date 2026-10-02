from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import numpy as np
import torch
from tridep_ccml.fusion_g1.common import WORK, PLAN, read, save, digest, desc, now, verify, state_hash, set_seed, orders, Data, RecordContext, SEEDS, FOLDS, METRICS
from tridep_ccml.fusion_g1.model import Model, gradient_norm
from tridep_ccml.eeg_p3.methods import inner_splits, fit_threshold, fit_sigmoid, apply_sigmoid, validate_oof

CODE=Path(__file__).resolve().parent
OUT=WORK/'tridep_ccml/experiments/FUSION-G4-37-fulltext'
REPORT=WORK/'output/analysis/fusion_g4_20260930'
APPENDIX=WORK/'output/plans/FUSION_G4后处理开发实验执行附表.md'
G1=WORK/'tridep_ccml/experiments/FUSION-G1-37-fulltext'
CONDITIONS=['G-CCML-ST','G-CCML-EST']
VARIANTS=['raw','threshold','sigmoid']

def history():
    from tridep_ccml.fusion_g1.common import history as prior_history
    result=prior_history();entries=read(G1/'deliverable_hashes.json')
    for p,sha in entries.items():assert digest(G1/p)==sha,p
    result[G1.relative_to(WORK).as_posix()]={'files':len(entries),'manifest':desc(G1/'deliverable_hashes.json')}
    return result

def config_check():
    cfg=read(OUT/'registration.json')
    for d in cfg['frozen_files']:verify(d)
    return cfg

def folder(c,seed,fold,inner):
    return OUT/f'runs/{c}/seed_{seed}/fold_{fold}/inner_{inner}'

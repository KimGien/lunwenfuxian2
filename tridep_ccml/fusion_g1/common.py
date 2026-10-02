from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import numpy as np
import torch
from tridep_ccml.eeg_e2.common import WORK, PLAN, read, save, digest, desc, now
from tridep_ccml.records import RecordContext, canonical_digest
from tridep_ccml.v2.common import set_seed

CODE = Path(__file__).resolve().parent
OUT = WORK/'tridep_ccml/experiments/FUSION-G1-37-fulltext'
REPORT = WORK/'output/analysis/fusion_g1_20260929'
APPENDIX = WORK/'output/plans/FUSION_G1六条件执行附表.md'
E2 = WORK/'tridep_ccml/experiments/EEG-E2v2-37-fulltext'
R1 = WORK/'tridep_ccml/experiments/EEG-R1-37-fulltext'
SEEDS = [11, 22, 33]
FOLDS = [1, 2, 3, 4, 5]
CONDITIONS = [f'G-{f}-{m}' for f in ['CE','EDL','CCML'] for m in ['ST','EST']]
METRICS = ['macro_f1','balanced_accuracy','mdd_recall','auc','nll','brier']

def verify(d):
    p=WORK/d['path']
    assert digest(p)==d['sha256'], str(p)
    return p

def history():
    from tridep_ccml.eeg_e2v2.common import history as old_history
    result=old_history()
    entries=read(E2/'deliverable_hashes.json')
    for p, sha in entries.items(): assert digest(E2/p)==sha, p
    result[E2.relative_to(WORK).as_posix()]={'files':len(entries),'manifest':desc(E2/'deliverable_hashes.json')}
    return result

def lock_check():
    lock=read(OUT/'registration.json')
    for d in lock['frozen_files']: verify(d)
    return lock

def state_hash(state):
    import hashlib
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class Data:
    def __init__(self,device='cuda'):
        self.ctx=RecordContext(); self.ids=self.ctx.ids
        manifest=read(OUT/'input_manifest.json')
        assert manifest['subject_ids']==self.ids
        self.inputs={m:torch.from_numpy(np.load(verify(d))).to(device) for m,d in manifest['arrays'].items()}
        self.y=torch.tensor([self.ctx.labels[s] for s in self.ids],device=device)
        self.index={s:i for i,s in enumerate(self.ids)}
    def batch(self,ids):
        i=[self.index[s] for s in ids]
        return {m:x[i] for m,x in self.inputs.items()}, self.y[i]

def orders(ids,seed):
    g=torch.Generator().manual_seed(seed)
    return [[ids[i] for i in torch.randperm(len(ids),generator=g).tolist()] for _ in range(60)]

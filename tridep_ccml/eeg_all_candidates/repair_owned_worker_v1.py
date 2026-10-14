"""Owned Q6 repair worker; no tensor import or GPU work without exact admission."""
from __future__ import annotations
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
from . import storage as S
from . import repair_runtime_owner_v1 as O

SOURCE=Path(__file__).resolve()

def require(ok,message):
    if not ok:raise ValueError(message)

def run(*,condition,repair_registration,admission,token,resume=False):
    value,identity=O.enter(admission,token,condition=condition,repair_registration=repair_registration)
    if condition=='EEGMamba-lastblock-finetune':
        require(sys.platform=='linux' and Path(sys.prefix).resolve()==(S.OUT/'assets/eegmamba_linux_env').resolve(),
                'Original separately qualified Linux Mamba environment required')
    else:require(sys.platform=='win32','Original Windows CodeBrain environment required')
    module=importlib.import_module('.finetune_v4',__package__)
    require(S.desc(Path(module.__file__))==value['formal_source'],'Admitted actual formal source changed')
    path=S.checked(admission).parent/'worker.json'
    record={'kind':'owned_formal_driver','status':'running','started_at':S.now(),'source':S.desc(SOURCE),
            'formal_source':value['formal_source'],'condition':condition,'repair_registration':repair_registration,
            'admission':admission,'worker_identity':identity,'completed_calls':[],
            'timer_is_nested_supplemental_not_F4_primary_cost':True,'GPU_probes_added':0}
    S.save(path,record,immutable=True);started=time.monotonic()
    original=module.fit
    def owned_fit(condition,seed,fold,**kwargs):
        owner=O.require_live(value,identity)
        require(condition==value['condition'] and kwargs.get('repair_registration')==repair_registration,
                'Unexpected actual worker fit registration')
        result=original(condition,seed,fold,**kwargs)
        record['completed_calls'].append({'condition':condition,'seed':seed,'fold':fold,'actual_queue_owner':owner})
        S.save(path,record)
        return result
    module.fit=owned_fit
    try:
        module.run(condition,repair_registration=repair_registration,resume=resume)
        O.require_live(value,identity)
        record['status']='complete'
    except BaseException as error:
        record.update(status='failed',failure_type=type(error).__name__,failure=str(error))
        raise
    finally:
        module.fit=original
        record.update(finished_at=S.now(),inclusive_worker_driver_seconds=time.monotonic()-started)
        S.save(path,record)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--condition')
    parser.add_argument('--repair-registration');parser.add_argument('--repair-registration-sha256')
    parser.add_argument('--worker-admission');parser.add_argument('--worker-admission-sha256')
    parser.add_argument('--token');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    if not args.run:
        require(not args.resume,'Actual resume requires bound --run')
        print(json.dumps({'status':'preparation_only','model_calls':0,'GPU_queries':0,'fit_or_registration_writes':0}));return
    require(all((args.condition,args.repair_registration,args.repair_registration_sha256,
                 args.worker_admission,args.worker_admission_sha256,args.token)),
            'Exact actual queue child admission and repair registration required')
    return run(condition=args.condition,repair_registration={'path':args.repair_registration,'sha256':args.repair_registration_sha256},
               admission={'path':args.worker_admission,'sha256':args.worker_admission_sha256},token=args.token,resume=args.resume)

if __name__=='__main__':main()

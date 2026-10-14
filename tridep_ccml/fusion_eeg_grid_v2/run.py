"""Single owned pipeline. Validated latest checkpoints resume automatically."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import time
import traceback
import uuid
from . import common as C

def protocol():
    sources=[C.desc(p) for p in sorted(C.CODE.rglob('*.py')) if '__pycache__' not in p.parts]
    sources.extend([C.desc(C.PLAN),C.desc(C.CODE/'upstream_sources.json'),C.desc(C.ROOT/'tridep_ccml/ccml.py'),
                    C.desc(C.ROOT/'tridep_ccml/references/ccml/source.json')])
    return {'schema':'Fusion-EEG15-Methods5-v2.1','sources':sources,'EEG':C.EEG,'excluded':'E14',
      'methods':list(C.METHODS),'seeds':list(C.SEEDS),'folds':[1,2,3,4,5],
      'formal_fits':1200,'EST':1125,'ST':75,'epochs_max':120,'batch':8,'last_batch_kept':True,
      'warmup':10,'lr':.001,'weight_decay':.001,'gradient_clip':1.,'dropout':.2,
      'selection':'lowest validation final-prediction NLL; earliest tie',
      'early_stop':{'minimum_epoch':60,'patience':60,'minimum_delta':1e-4},
      'QMF':{'author_path':'RGBD','rank_lambda':.1,'confidence':'.1 logsumexp true logits',
             'quality_grad':'retained in dynamic fusion','history_update':'after rank objective',
             'equal_history_guard':'all-zero normalized ranks and margin; no divide by zero'},
      'ECML':{'author_path':'pinned RCML','gamma':1.,'anneal':50,'aggregation':'sequential evidence average E->S->T or S->T',
              'consistency':'get_dc_loss confidence-weighted probability distance; full author EDL mean'},
      'CCML':{'rho':1.25,'delta':1.,'gamma':1.,'anneal':50,'view_reduction':'sum'},
      'LMF':{'rank':2,'output':16,'constant':1,'classification_head':'ReLU Dropout.2 Linear16->2'},
      'MMDynamics':{'hidden':[32],'original_full_loss':True},
      'threshold':'pMDD>0.5; tie HC','baseline_macro_f1':.7255026455026455,
      'base_model_training':0,'test_evaluation_gate':'all1200 complete','created_at':C.now()}

def register():
    p=C.OUT/'registration.json'
    if p.exists():
        v=C.read(p)
        for s in v['sources']:C.checked(s)
        return C.desc(p)
    C.save(p,protocol(),immutable=True)
    return C.desc(p)

def status():
    p=C.OUT/'status.json';v=C.read(p) if p.exists() else {'phase':'not_started'}
    owner=C.OUT/'control/owner.json'
    v['paused']=C.paused()
    if owner.exists():v['owner']=C.read(owner)
    return v

def acquire():
    C.check_pause()
    control=C.OUT/'control';control.mkdir(parents=True,exist_ok=True)
    lock=control/'queue.lock';ownerpath=control/'owner.json'
    if lock.exists():
        C.require(ownerpath.exists(),'Queue lock without owner: inspect transaction, do not blindly restart')
        owner=C.read(ownerpath);actual=C.inventory()
        if any(r in actual for r in owner['native_tree']):
            raise RuntimeError('Healthy or still-existing queue owner; no second launch')
        C.save(control/f"lost_owner_{owner['token']}.json",dict(owner,detected_at=C.now(),unknown_tail_seconds=None))
        # Both exact targets remain inside the new experiment control directory.
        lock.rename(control/f"lost_queue_{owner['token']}.lock")
        ownerpath.rename(control/f"lost_owner_record_{owner['token']}.json")
    token=uuid.uuid4().hex
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,token.encode());os.close(fd)
    tree=C.own_tree()
    owner={'token':token,'native_tree':tree,'started_at':C.now(),'source':C.desc(__file__),
           'module':'tridep_ccml.fusion_eeg_grid_v2.run','child_queues':0}
    C.save(ownerpath,owner)
    # Refuse simultaneous related queue owners. Do not suspend other programs.
    ownpids={r['pid'] for r in tree}
    other=[r for r in C.inventory() if r['pid'] not in ownpids]
    C.require(not other,'Another workspace experiment process exists; wait for resource release: '+json.dumps(other))
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
                       capture_output=True,text=True,check=True,timeout=20).stdout
    C.require(not any('python' in s.lower() or 'wsl' in s.lower() for s in gpu.splitlines()),'GPU occupied by another model process')
    return owner

def release(owner):
    control=C.OUT/'control';p=control/'owner.json';lock=control/'queue.lock'
    if p.exists() and C.read(p)['token']==owner['token']:
        C.save(control/f"released_{owner['token']}.json",dict(C.read(p),released_at=C.now()))
        p.unlink()
        if lock.exists() and lock.read_text()==owner['token']:
            lock.unlink()

def run():
    if not (C.OUT/'control/user_pause.json').exists():C.save(C.OUT/'control/user_pause.json',{'paused':False})
    owner=acquire()
    try:
        registration=register()
        C.save(C.OUT/'status.json',{'phase':'engineering','started_at':owner['started_at'],'at':C.now(),'formal_fits_completed':0})
        from . import engineering,features,training,evaluation
        proof=engineering.run()
        C.save(C.OUT/'status.json',{'phase':'source_registration','at':C.now(),'formal_fits_completed':0})
        features.export_all()
        completed=0
        for identity in C.identities():
            C.check_pause()
            C.save(C.OUT/'status.json',{'phase':'formal_training','active_fit':identity,'formal_fits_completed':completed,
                        'formal_fits_total':1200,'at':C.now(),'backend':proof['methods'][identity['method']]['device']})
            done=training.fit(identity,proof['methods'][identity['method']]['device'],registration)
            completed+=1
            print(f"Formal fit {completed}/1200 {identity} epochs={done['epochs']} updates={done['updates']}",flush=True)
        C.save(C.OUT/'status.json',{'phase':'unified_evaluation','formal_fits_completed':1200,'at':C.now()})
        evaluation.run(registration,proof)
        C.save(C.OUT/'status.json',{'phase':'awaiting_independent_final_audit','formal_fits_completed':1200,
                                  'all_plan_complete':False,'at':C.now()})
    except InterruptedError as error:
        current=status();current.update(phase='user_paused',error=str(error),at=C.now());C.save(C.OUT/'status.json',current)
    except BaseException as error:
        current=status();current.update(phase='failure_requires_review',error=repr(error),at=C.now())
        C.save(C.OUT/'status.json',current)
        C.save(C.OUT/'control'/f"failure_{time.time_ns()}.json",{'owner':owner,'error':repr(error),'traceback':traceback.format_exc(),
            'unknown_active_tail_seconds':None,'at':C.now()})
        raise
    finally:
        release(owner)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--status',action='store_true')
    a=p.parse_args()
    if a.run:
        run()
    else:
        print(json.dumps(status(),ensure_ascii=False))

if __name__=='__main__':main()

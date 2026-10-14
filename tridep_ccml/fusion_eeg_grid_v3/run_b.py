"""Single owned stage-B queue. New tests remain sealed until stages B-D pass."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import uuid
from . import common as C

def identities():
    for stage,norm,variant in (('B_main','N1','Q1'),('B_N1Q0','N1','Q0'),('B_N0Q1','N0','Q1')):
        for base in C.identities():
            if stage!='B_main' and base['method']!='QMF': continue
            yield dict(base,stage=stage,normalization=norm,qmf_variant=variant,base_lr=.001,budget=120)

def prereview():
    review_path=C.OUT/'implementation_review_b.json'
    review=C.read(review_path)
    C.require(review['passed'] and review['formal_training_authorized'],'Stage B lacks independent accepted review')
    for record in review['reviewed_sources']: C.checked(record)
    for record in review['evidence'].values(): C.checked(record)
    C.require(C.read(C.OUT/'source_gate.json')['passed'],'Source gate incomplete')
    C.require(C.read(C.OUT/'backend_gate.json')['passed'],'Backend gate incomplete')
    return review_path,review

def register():
    path=C.OUT/'registration_b.json'
    if path.exists():
        value=C.read(path)
        for record in value['sources']: C.checked(record)
        C.checked(value['review'])
        C.require(value['identities']==list(identities()),'Registered B queue changed')
        return C.desc(path)
    C.check_pause()
    review_path,review=prereview()
    sources=review['reviewed_sources']+[C.desc(C.PLAN)]
    value={'schema':'fusion_v3_B_registration_1','created_at':C.now(),
           'sources':sources,'review':C.desc(review_path),'identities':list(identities()),
           'new_initializations_max':1680,'B_main':1200,'B_N1Q0':240,'B_N0Q1':240,
           'methods':list(C.METHODS),'formal_test_forwards_allowed':False,
           'backend_gate':C.desc(C.OUT/'backend_gate.json'),'source_gate':C.desc(C.OUT/'source_gate.json'),
           'plan_total_new_initializations_max':3180,'D_budget_identities':480,'D_new_initializations':0,
           'no_automatic_v2_fit_reuse':'No full transform+training equivalence has been certified for B',
           'post_B_next':'C panel and fold-specific selection, then D; B completion is NOT experiment completion'}
    snapshots=[]
    for i,record in enumerate(sources):
        original=C.checked(record)
        target=C.OUT/'frozen_sources_b'/f'{i:03d}_{original.name}'
        C.atomic(target,original.read_bytes(),immutable=True)
        snapshots.append({'source':record,'frozen_copy':C.desc(target)})
    value['frozen_sources']=snapshots
    C.save(path,value,immutable=True)
    return C.desc(path)

def snapshot():
    path=C.OUT/'status.json'
    value=C.read(path) if path.exists() else {'phase':'implementation_and_review'}
    value['paused']=C.paused()
    owner=C.OUT/'control/owner.json'
    if owner.exists(): value['owner']=C.read(owner)
    return value

def acquire(module='tridep_ccml.fusion_eeg_grid_v3.run_b', formal=True):
    C.check_pause()
    control=C.OUT/'control';control.mkdir(parents=True,exist_ok=True)
    lock=control/'queue.lock';ownerpath=control/'owner.json'
    actual=C.inventory()
    if lock.exists() or ownerpath.exists():
        C.require(lock.exists() and ownerpath.exists(),'Incomplete owner transaction; inspect instead of blind restart')
        old=C.read(ownerpath)
        C.require(lock.read_text()==old['token'],'Owner token differs')
        C.require(not any(entry in actual for entry in old['native_tree']),'Existing exact queue owner; do not duplicate')
        # This is a verified dead process, not permission to replay an old intent.
        # All code and latest state must still pass the frozen registration below.
        C.save(control/f"lost_owner_{old['token']}_{time.time_ns()}.json",
               dict(old,detected_at=C.now(),unknown_tail_updates=None,unknown_tail_seconds=None))
        lock.rename(control/f"lost_queue_{old['token']}.lock")
        ownerpath.rename(control/f"lost_owner_record_{old['token']}.json")
    tree=C.own_tree(); own={r['pid'] for r in tree}
    others=[r for r in actual if r['pid'] not in own]
    C.require(not others,'Other workspace process is active; finish nonconflicting preparations or wait: '+json.dumps(others))
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
                       text=True,capture_output=True,check=True,timeout=20).stdout
    other_gpu=[]
    for row in gpu.splitlines():
        pid,_,name=row.partition(',')
        if pid.strip().isdigit() and int(pid.strip()) not in own and Path(name.strip()).name.lower()!='chatgpt.exe':
            other_gpu.append(row)
    C.require(not other_gpu,'GPU occupied by another workload: '+repr(other_gpu))
    C.check_pause()
    token=uuid.uuid4().hex
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as stream:
        stream.write(token);stream.flush();os.fsync(stream.fileno())
    owner={'token':token,'native_tree':tree,'started_at':C.now(),'source':C.desc(__file__),
           'module':module,'formal_queue_count':int(formal),
           'boundary_protocol':'pause before next epoch/fit; atomic latest at each completed epoch'}
    C.save(ownerpath,owner)
    return owner

def release(owner):
    control=C.OUT/'control';path=control/'owner.json';lock=control/'queue.lock'
    if path.exists() and C.read(path)['token']==owner['token']:
        C.save(control/f"released_{owner['token']}.json",dict(C.read(path),released_at=C.now()))
        path.unlink()
        if lock.exists() and lock.read_text()==owner['token']:lock.unlink()

def finish_existing(registration):
    path=C.OUT/'phase_b_complete.json'
    if not path.exists():return False
    done=C.read(path)
    expected=list(identities())
    C.require(done['passed'] and done['registration']==registration and done['count']==len(expected)==1680
              and done['test_forwards']==0 and not done['all_plan_complete'],'B terminal certificate differs')
    C.require(len(done['completed'])==len(expected),'B terminal identity count differs')
    for identity,descriptor in zip(expected,done['completed']):
        p=C.checked(descriptor)
        C.require(p==C.fitdir(identity)/'complete.json','B completion order/path differs')
        fit=C.read(p)
        C.require(fit['passed'] and fit['identity']==identity and fit['test_forwards']==0,'B terminal fit differs')
    C.save(C.OUT/'status.json',{'phase':'B_complete_next_C','B_completed':1680,'B_total':1680,
           'all_plan_complete':False,'phase_b_complete':C.desc(path),'test_forwards':0,
           'next_action':'C panel/selection and D remain; new tests stay sealed','at':C.now()})
    return True

def run():
    C.check_pause()
    blocked=C.OUT/'control/failure_requires_review.json'
    C.require(not blocked.exists(),'An actual failure requires recorded independent repair/recovery review')
    prereview()
    registration=register()
    owner=acquire()
    completed=[]
    try:
        if finish_existing(registration):
            print('B certificate already verified; proceed to C without new fits',flush=True)
            return
        from . import training as T
        backend=C.read(C.OUT/'backend_gate.json')
        for index,identity in enumerate(identities()):
            C.check_pause()
            C.save(C.OUT/'status.json',{'phase':'B_formal_training','active_fit':identity,
                  'B_completed':len(completed),'B_total':1680,'all_plan_complete':False,
                  'test_forwards':0,'owner_token':owner['token'],'at':C.now(),
                  'backend':backend['methods'][identity['method']]['device']})
            done=T.fit(identity,backend['methods'][identity['method']]['device'],registration)
            completed.append(C.desc(C.fitdir(identity)/'complete.json'))
            print(json.dumps({'B_completed':len(completed),'B_total':1680,'identity':identity,
                              'epochs':done['epochs'],'updates':done['updates'],'at':C.now()},ensure_ascii=False),flush=True)
        C.save(C.OUT/'phase_b_complete.json',{'passed':True,'scope':'B formal fits only; C/D/E and final independent audits remain',
               'registration':registration,'completed':completed,'count':1680,'test_forwards':0,
               'all_plan_complete':False,'at':C.now()},immutable=True)
        C.save(C.OUT/'status.json',{'phase':'B_complete_next_C','B_completed':1680,'B_total':1680,
              'all_plan_complete':False,'next_action':'Implement/review or resume C panel from v3 plan; do not evaluate new tests',
              'test_forwards':0,'at':C.now()})
    except InterruptedError as error:
        value=snapshot();value.update(phase='user_paused',error=str(error),at=C.now(),all_plan_complete=False)
        value.pop('owner',None);C.save(C.OUT/'status.json',value)
    except BaseException as error:
        report={'passed':False,'owner':owner,'registration':registration,'identity':identity if 'identity' in locals() else None,
                'error':repr(error),'traceback':traceback.format_exc(),'at':C.now(),
                'unknown_tail_seconds':None,'unknown_tail_updates':None,
                'no_automatic_retry_before_review':True}
        failure= C.OUT/'control'/f'failure_{time.time_ns()}.json'
        C.save(failure,report,immutable=True)
        C.save(blocked,{'failure':C.desc(failure),'at':C.now()})
        value=snapshot();value.update(phase='failure_requires_review',error=repr(error),at=C.now(),all_plan_complete=False)
        value.pop('owner',None);C.save(C.OUT/'status.json',value)
        raise
    finally:release(owner)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--register',action='store_true')
    args=parser.parse_args()
    if args.run:run()
    elif args.register:print(json.dumps(register()))
    else:print(json.dumps(snapshot(),ensure_ascii=False))
if __name__=='__main__':main()


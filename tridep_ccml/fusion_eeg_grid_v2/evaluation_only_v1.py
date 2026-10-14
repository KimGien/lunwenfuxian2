"""Reviewed evaluation-only recovery: no fit, encoder export or engineering rerun."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import time
import traceback
import uuid
from . import common as C

SOURCE=Path(__file__).resolve()
CONTROL=C.OUT/'evaluation_recovery_v1'

def prepare():
    diagnosis=C.read(C.OUT/'control/LMF_validation_reload_diagnosis_v2.json')
    observations=diagnosis['observations']
    C.require(diagnosis['identity']=={'eeg':'ST','method':'LMF','seed':11,'fold':3} and
              observations[0]['parameter_requires_grad'] and observations[0]['bitwise_equal'] and
              not observations[1]['parameter_requires_grad'] and not observations[1]['bitwise_equal'] and
              all(o['grad_none'] for o in observations),'Exact failed validation/kernel diagnosis missing')
    sources=[C.desc(SOURCE),C.desc(C.CODE/'evaluation_v2.py'),C.desc(C.CODE/'evaluation.py'),
             C.desc(C.OUT/'control/LMF_validation_reload_diagnosis_v1.json'),
             C.desc(C.OUT/'control/LMF_validation_reload_diagnosis_v2.json'),C.desc(C.OUT/'registration.json')]
    previous=[C.desc(C.fitdir(i)/'evaluation.json') for i in C.identities() if (C.fitdir(i)/'evaluation.json').exists()]
    value={'passed':True,'sources':sources,'cached_successful_evaluations':previous,
      'repair':'retain original Parameter flags to preserve CPU matmul kernel; all forwards inference_mode and no optimizer',
      'same_backend_bitwise_requirement_preserved':True,'formal_training_sources_unchanged':True,
      'validation_diagnosis_forwards':4,'test_diagnosis_forwards':0,'diagnostic_cost_seconds':None,
      'original_failed_validation_attempt_seconds':None,'new_optimizer_updates':0,
      'original_first_candidate_probe':'fold1 was checked before precise missing unit identification; preserved as extra diagnostic cost'}
    C.save(CONTROL/'review.json',value,immutable=True)
    return value

def run():
    C.check_pause();review=prepare()
    C.require(not (C.OUT/'control/owner.json').exists() and not (C.OUT/'control/queue.lock').exists(),
              'Another queue owner exists')
    active=[x for x in C.inventory() if 'fusion_eeg_grid_v2.run' in x['command'] and '--run' in x['command']]
    C.require(not active,'Original native training queue is still alive')
    C.runtime(11,'cpu')
    from . import evaluation_v2 as E
    E.gate()
    registration=C.desc(C.OUT/'registration.json');reg=C.read(C.OUT/'registration.json')
    for source in reg['sources']:C.checked(source)
    for source in review['sources']:C.checked(source)
    engineering=C.read(C.OUT/'engineering/complete.json')
    token=uuid.uuid4().hex;control=C.OUT/'control';lock=control/'queue.lock';ownerpath=control/'owner.json'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,token.encode());os.close(fd)
    owner={'token':token,'native_tree':C.own_tree(),'started_at':C.now(),'source':C.desc(SOURCE),
           'module':'tridep_ccml.fusion_eeg_grid_v2.evaluation_only_v1','role':'evaluation_only',
           'new_optimizer_updates':0,'formal_fits_completed':1200}
    C.save(ownerpath,owner)
    try:
        C.save(C.OUT/'status.json',{'phase':'unified_evaluation_recovery','formal_fits_completed':1200,
                                  'owner_token':token,'at':C.now(),'no_new_training':True})
        E.run(registration,engineering)
        C.save(CONTROL/'complete.json',{'passed':True,'review':C.desc(CONTROL/'review.json'),
          'evaluation_complete':C.desc(C.OUT/'evaluation_complete.json'),'new_optimizer_updates':0,
          'cached_successes_reused':len(review['cached_successful_evaluations']),'owner':owner,'at':C.now()},immutable=True)
        C.save(C.OUT/'status.json',{'phase':'awaiting_independent_final_audit','formal_fits_completed':1200,
                                  'all_plan_complete':False,'at':C.now()})
    except BaseException as error:
        C.save(CONTROL/f'failure_{time.time_ns()}.json',{'owner':owner,'error':repr(error),
             'traceback':traceback.format_exc(),'unknown_uncommitted_tail_seconds':None,'at':C.now()})
        C.save(C.OUT/'status.json',{'phase':'evaluation_recovery_failure','formal_fits_completed':1200,
                                  'error':repr(error),'at':C.now()})
        raise
    finally:
        C.save(control/f'released_{token}.json',dict(owner,released_at=C.now()),immutable=True)
        if ownerpath.exists() and C.read(ownerpath)['token']==token:ownerpath.unlink()
        if lock.exists() and lock.read_text()==token:lock.unlink()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--prepare',action='store_true')
    a=p.parse_args()
    if a.run:run()
    else:print(prepare())

if __name__=='__main__':main()

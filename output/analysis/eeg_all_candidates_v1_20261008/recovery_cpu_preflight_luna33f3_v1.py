"""Snapshot and CPU-only full-state validation of the lost LUNA33/f3 fit.

Two original CPU tensor loads, no model/GPU/raw/test/process operations. All
original bytes are retained; an interrupted buffered trace remains running.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

SOURCE=Path(__file__).resolve(); WORK=SOURCE.parents[3]
sys.path.insert(0,str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S, registry as R
HELPER=SOURCE.with_name('safe_stop_acceleration_v1.py')
HELPER_SHA='6c0c6d084b5463ed3cfa2326cc17bb105fa3a9369fd75527d4912d198489071c'
IDENTITY=('LUNA-lastblock-finetune',33,3)
FOLDER=S.OUT/'runs/LUNA-lastblock-finetune/seed_33/fold_3'
ATTEMPT=S.OUT/'finetune_v2_attempts/LUNA-lastblock-finetune/f_1791637178936133600_69c184b0/attempt.json'
OLD_QUEUE=S.OUT/'continuation_queue_v5/attempts/1791625233145752000-LUNA-lastblock-finetune-fit/status.json'
def require(ok,msg):
    if not ok: raise ValueError(msg)
def write_new(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
def copy_exact(path,target):
    before=S.desc(path);shutil.copy2(path,target);after=S.desc(path);snap=S.desc(target)
    require(before==after and before['sha256']==snap['sha256'],'Snapshot bytes changed')
    return {'original':before,'snapshot':snap}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--absence',type=Path,required=True)
    args=parser.parse_args();absence_binding=S.desc(args.absence);absence=S.read(args.absence)
    require(absence.get('schema')=='q5_process_absence_observation_v2' and absence.get('query_succeeded') is True
            and absence.get('all_expected_processes_absent') is True and absence.get('observed_rows')==[],'True original process absence required')
    require(S.digest(HELPER)==HELPER_SHA,'Sealed checkpoint validator changed')
    require(R.fit_status(*IDENTITY)['status']=='training' and not (FOLDER/'complete.json').exists(),'Only actual incomplete fit may resume')
    stamp=str(time.time_ns());snaproot=S.OUT/'process_loss_recovery_v2'/('snapshot_'+stamp);snaproot.mkdir(parents=True)
    retained={}
    for name in ('initial.pt','latest.pt','best.pt','train_history.json','validation_history.json','progress.json','started.json','status.json'):
        path=FOLDER/name
        if path.exists():retained[name]=copy_exact(path,snaproot/name)
    for key,path in (('attempt',ATTEMPT),('owner',S.OUT/'continuation_queue_v5/owner.json'),('queue_attempt',OLD_QUEUE)):
        retained[key]=copy_exact(path,snaproot/(key+'.json'))
    snap_binding=S.desc(snaproot/'snapshot_manifest.json') if (snaproot/'snapshot_manifest.json').exists() else None
    write_new(snaproot/'snapshot_manifest.json',{'schema':'q5_loss_byte_snapshot_v2','source':S.desc(SOURCE),
              'created_at':S.now(),'identity':list(IDENTITY),'absence':absence_binding,'retained':retained,'tensor_reads':0})
    snap_binding=S.desc(snaproot/'snapshot_manifest.json')
    report_path=S.REPORT/('recovery_cpu_preflight_luna33f3_'+stamp+'.json')
    report={'schema':'actual_cpu_recovery_preflight_luna33f3_v1','source':S.desc(SOURCE),'passed':False,
      'condition':IDENTITY[0],'seed':33,'fold':3,'snapshot_manifest':snap_binding,'process_absence_evidence':absence_binding,
      'checkpoint_loads':0,'model_constructions':0,'GPU_operations':0,'raw_arrays_loaded':0,'test_probability_reads':0,
      'process_operations':0,'original_artifacts_modified':False,'training_complete_promoted':False}
    try:
        spec=importlib.util.spec_from_file_location('sealed_resume_payload_validator',HELPER)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        import torch
        require(not torch.cuda.is_initialized(),'CUDA must remain uninitialized')
        allowed={FOLDER/'initial.pt',FOLDER/'latest.pt'};loaded={};old_load=torch.load;old_lazy=torch.cuda._lazy_init
        def cpu_load(file,*args,**kwargs):
            target=Path(file).resolve();require(target in allowed and target not in loaded and kwargs.get('map_location')=='cpu','Two distinct original CPU checkpoints only')
            value=old_load(file,*args,**kwargs);loaded[target]=value;report['checkpoint_loads']+=1;return value
        def forbidden(*args,**kwargs):raise AssertionError('GPU initialization forbidden')
        try:
            torch.load=cpu_load;torch.cuda._lazy_init=forbidden
            checked=helper.validate_checkpoint(FOLDER);initial=loaded[FOLDER/'initial.pt'];latest=loaded[FOLDER/'latest.pt']
            require(checked['cpu_checkpoint_loads']==report['checkpoint_loads']==2,'Two CPU loads required')
            require(latest['history']==S.read(FOLDER/'train_history.json')==S.read(snaproot/'train_history.json'),'Embedded durable history must match disk/snapshot')
            require(S.read(FOLDER/'validation_history.json')==S.read(snaproot/'validation_history.json'),'Validation history snapshot differs')
            decision=helper.P.training_decision('finetune',[r['val_eval_ce'] for r in latest['history']],extended=latest['scheduler_state_dict']['extended'])
            require(decision['stop'] is False,'Checkpoint already reaches registered stop; explicit repair required')
            contract=S.read(S.checked(R._condition_lock(IDENTITY[0])))['contract'];names=contract['trainable_parameter_names']
            enc=[n for n in names if n.startswith('encoder.')];head=[n for n in names if n.startswith('head.')]
            require(len(enc)==17 and len(head)==6 and not any(n.endswith('rotary_emb.freqs') for n in enc),'Registered author LUNA parameter selection differs')
            changes=lambda state,selected:sum(not torch.equal(initial['state_dict'][n],state[n]) for n in selected)
            state={'selected_encoder_changed':changes(latest['best_selected_state_dict'],enc),'selected_head_changed':changes(latest['best_selected_state_dict'],head),
                   'latest_encoder_changed':changes(latest['state_dict'],enc),'latest_head_changed':changes(latest['state_dict'],head),
                   'selected_epoch':decision['best_epoch'],'selected_phase':'head_only_warmup' if decision['best_epoch']<=10 else 'lastblock_finetune'}
            require(not torch.cuda.is_initialized(),'CUDA unexpectedly initialized')
        finally:torch.load=old_load;torch.cuda._lazy_init=old_lazy
        trace=S.read(ATTEMPT);progress=S.read(FOLDER/'progress.json')
        require(tuple(trace.get(k) for k in ('condition','seed','fold'))==IDENTITY and trace.get('status')=='running'
                and trace.get('implementation',{}).get('version')==3 and trace.get('graceful_terminal_flush') is False
                and trace.get('unflushed_tail_unknown') is True and trace.get('unflushed_tail_counters') is None,'Original lost trace must retain running/unknown semantics')
        durable={'epoch':checked['epoch'],'optimizer_updates':checked['optimizer_updates'],'backward_calls':checked['backward_calls']}
        obs=trace['latest_durable_checkpoint'];require(all(obs[k]==v for k,v in durable.items()),'Observer/checkpoint durable boundary differs')
        require(trace['optimizer_step_returned']>=durable['optimizer_updates'] and trace['backward_returned']>=durable['backward_calls'],'Persisted observations precede durable state')
        require(progress['optimizer_updates_completed']==trace['optimizer_step_returned'] and progress['backward_calls_completed']==trace['backward_returned'],'Progress and persisted returned counters differ')
        for pair in retained.values():require(S.desc(S.checked(pair['original']))==pair['original'] and S.digest(S.checked(pair['snapshot']))==pair['original']['sha256'],'Original/snapshot byte bindings changed')
        report.update(passed=True,helper_source=S.desc(HELPER),checkpoint_validation=checked,durable_history_matches_current_and_snapshot=True,
          durable=durable,selected_terminal_changes=state,resume_next_epoch=durable['epoch']+1,
          buffered_trace_pair=retained['attempt'],resume_checkpoint=retained['latest.pt']['original'],preserved_checkpoint=retained['latest.pt']['snapshot'],
          known_extra_lower_bounds={'optimizer_updates':trace['optimizer_step_returned']-durable['optimizer_updates'],
                                   'backward_calls':trace['backward_returned']-durable['backward_calls']},
          observed_progress=progress,unflushed_tail_counters=None,final_elapsed_seconds=None,exit_code=None,actual_exit_time=None,reason=None,
          known_extra_not_added_again_to_original_returned_counters=True,full_plan_completed_claimed=False)
    except BaseException as error:report.update(error_type=type(error).__name__,error=str(error));raise
    finally:
        write_new(report_path,report)
        print(json.dumps({'artifact':S.desc(report_path),'passed':report['passed'],'checkpoint_loads':report['checkpoint_loads'],
                         'snapshot_manifest':snap_binding,'durable':report.get('durable'),'known_extra_lower_bounds':report.get('known_extra_lower_bounds')},ensure_ascii=True),flush=True)
if __name__=='__main__':main()

"""Capture exact original platform learning-rate references; stdlib/JSON only.
No tolerance, training mutation, Tensor/model/GPU or outer probability access.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
SOURCE=Path(__file__).resolve()
WORK=SOURCE.parents[3]
sys.path.insert(0,str(WORK))
from tridep_ccml.eeg_all_candidates import protocol as P
def desc(path):
 path=Path(path).resolve()
 return {'path':path.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def require(ok,message):
 if not ok: raise ValueError(message)
def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--run',action='store_true')
 args=parser.parse_args()
 if not args.run:
  print(json.dumps({'status':'preparation_only','model':0,'Tensor':0,'GPU':0,'test':0,'writes':0})); return
 started=time.monotonic()
 out=P.OUT
 q=out/'continuation_queue_v14'
 failed=out.parent.parent.parent.parent/'unused'
 terminal=json.loads((q/'status.json').read_text(encoding='utf-8'))
 require(terminal['status']=='failed' and terminal['error_type']=='CalledProcessError','Actual Q14 CPU failure required')
 require(not(q/'owner.json').exists(),'Actual Q14 owner must be released')
 cpu=WORK/'output/analysis/eeg_all_candidates_v1_20261008/completed_finetune_checkpoint_audit_v16_1791831771185269400.json'
 old=json.loads(cpu.read_text(encoding='utf-8'))
 require(old['schema_version']==16 and old['checkpoint_loads']==360 and old['verified_completed_finetune_fits']==120 and len(old['errors'])==15,'Actual failed CPU16 scope differs')
 source_binding=desc(SOURCE); protocol_binding=desc(Path(P.__file__))
 queue_binding=desc(q/'registration.json'); terminal_binding=desc(q/'status.json')
 require(queue_binding['sha256']=='93b19228fb8d3f032f87be4072f4ff2fb3a0bbe2d89adce93117896b5bc59742','Frozen Q14 differs')
 registered=json.loads((q/'registration.json').read_text(encoding='utf-8'))
 require(protocol_binding in registered['sources'],'Original registered protocol required')
 table=[]
 for epoch in range(1,P.POLICIES['finetune'].final_epochs+1):
  row={'epoch':epoch}
  for group in ('head','backbone'):
   m=P.learning_rate_multiplier('finetune',epoch,parameter_group=group)
   value=getattr(P.POLICIES['finetune'],group+'_lr')*m
   row[group+'_multiplier']=m;row[group+'_multiplier_hex']=m.hex()
   row[group+'_lr']=value;row[group+'_lr_hex']=value.hex()
  table.append(row)
 fits=[]; bindings=[]
 for seed in (11,22,33):
  for fold in range(1,6):
   folder=out/'runs/EEGMamba-lastblock-finetune'/f'seed_{seed}'/f'fold_{fold}'
   status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
   require(status['status']=='training_complete','Only actual completed Mamba')
   hpath=folder/'train_history.json'; before=desc(hpath)
   history=json.loads(hpath.read_text(encoding='utf-8'))
   differences=[]
   for h in history:
    expected=table[h['epoch']-1]
    for group in ('head','backbone'):
     actual=h[group+'_lr']; wanted=expected[group+'_lr']
     require(type(actual) is float and math.isfinite(actual),'Finite original LR required')
     if actual!=wanted:
      differences.append({'epoch':h['epoch'],'group':group,'recorded':actual,'recorded_hex':actual.hex(),'expected':wanted,'expected_hex':wanted.hex(),'absolute_difference':abs(actual-wanted),'ulps_at_expected':abs(actual-wanted)/math.ulp(wanted)})
   require(desc(hpath)==before,'History changed during capture')
   bindings.append(before)
   fits.append({'fit_id':f'EEGMamba-lastblock-finetune/seed_{seed}/fold_{fold}','status':desc(folder/'status.json'),'history':before,'epochs_checked':len(history),'scalar_LRs_checked':2*len(history),'strict_equal':not differences,'differences':differences})
 require(desc(SOURCE)==source_binding and desc(Path(P.__file__))==protocol_binding and desc(q/'status.json')==terminal_binding,'Source/terminal changed')
 value={'schema':'actual_platform_learning_rate_reference_v1','status':'captured_exact_original_platform_reference','source':source_binding,'protocol':protocol_binding,'Q14_registration':queue_binding,'Q14_failed_terminal':terminal_binding,'failed_CPU16':desc(cpu),'captured_at':datetime.now(timezone.utc).isoformat(),'runtime':{'platform':sys.platform,'python':sys.version,'executable':sys.executable,'base_prefix':sys.base_prefix,'prefix':sys.prefix,'libc':platform.libc_ver()},'reference_table':table,'fits':fits,'all_15_histories_strict_equal_to_this_platform':all(x['strict_equal'] for x in fits),'differences_total':sum(len(x['differences']) for x in fits),'zero_tolerance':True,'all_actual_training_and_history_bytes_unchanged':True,'model_calls':0,'Tensor_loads':0,'GPU_calls':0,'outer_test_reads':0,'formal_fit_or_optimizer_updates_added':0,'primary_stdlib_capture_seconds':time.monotonic()-started}
 target=WORK/'output/analysis/eeg_all_candidates_v1_20261008'/('actual_platform_learning_rate_reference_'+sys.platform+'_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as f:
  json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
 print(json.dumps({'artifact':desc(target),'platform':sys.platform,'differences_total':value['differences_total'],'all_15_strict_equal':value['all_15_histories_strict_equal_to_this_platform'],'source':source_binding}))
if __name__=='__main__':main()

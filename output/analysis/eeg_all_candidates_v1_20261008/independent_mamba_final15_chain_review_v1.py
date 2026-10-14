"""Independent Q14 final-Mamba15 source/evidence review; stdlib, no runtime training."""
import ast,copy,hashlib,importlib.util,json,os,sys,time,types
from pathlib import Path
from unittest.mock import patch
SOURCE=Path(__file__).resolve();A=SOURCE.parent;W=SOURCE.parents[3];M=W/'tridep_ccml/eeg_all_candidates';OUT=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008';sys.path.insert(0,str(W));checks={}
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);v=importlib.util.module_from_spec(s);s.loader.exec_module(v);return v
H=load(A/'independent_core_prefix_repair_chain_review_v1.py','_review_helpers');desc,read,node,tree,dump,require=H.desc,H.read,H.node,H.tree,H.dump,H.require

def check(k,v):require(v,k);checks[k]=True
def reject(k,f):
 try:f()
 except (ValueError,KeyError,AssertionError):checks[k]=True
 else:raise AssertionError('Unsafe acceptance: '+k)
def main():
 contract=A/'mamba_final15_chain_stdlib_contract_1791825597211897600.json';c=read(contract)
 check('author_55_contract_exact',desc(contract)['sha256']=='dc59fdf0d8c016b37d1e780465808449438e2faabbd1e8c14a2724749d2b93dd' and c['passed'] is True and len(c['checks'])==55 and all(c['checks'].values()) and desc(W/c['source']['path'])==c['source'])
 from tridep_ccml.eeg_all_candidates import continuation_queue_v14 as Q,deterministic_repair_execution_v9 as D
 sources=Q.sources();prior=read(OUT/'continuation_queue_v13/registration.json')
 check('exact194_frozen_source_chain_all175_retained',sources==c['sources'] and len(sources)==len({x['path'] for x in sources})==194 and len(prior['sources'])==175 and all(d in sources and desc(W/d['path'])==d for d in prior['sources']))
 for d in c['new_sources']:require(desc(W/d['path'])==d,'New source changed')
 check('all_new_runtime_audit_eval_cost_sources_exact',all(d in sources for d in c['new_sources']))
 q=M/'continuation_queue_v14.py';d=M/'deterministic_repair_execution_v9.py';f=M/'finetune_v12.py';e=M/'evaluate_v11.py'
 reg=ast.unparse(node(q,'register'));acquire=ast.unparse(node(q,'acquire'));child=ast.unparse(node(q,'child'));run=ast.unparse(node(q,'run'))
 check('only_Mamba15_original_ids_and_current345',Q.CONDITIONS==('EEGMamba-lastblock-finetune',) and 'len(others) == 345' in ast.unparse(node(q,'original_others_complete')) and "proofs[condition]['original_fit_ids']" in reg and 'for seed in (11, 22, 33)' in reg and 'for fold in (1, 2, 3, 4, 5)' in reg)
 check('permanent_intent_then_immutable_bytes_before_lock_ready',reg.index("create(ROOT / 'registration_intent.json'")<reg.index('archive_blocked_statuses(blocked)')<reg.index('R.lock_condition')<reg.index("'ready'") and "destination.open('xb')" in ast.unparse(node(q,'archive_blocked_statuses')))
 check('no_training_queue_evaluation_or_old_qualifier_mutable_precondition', 'Q.check_boundary' not in q.read_text() and 'provider.check_boundary' not in d.read_text() and 'evaluate_v11' not in q.read_text() and "status='all_training_CPU_gate_complete_evaluation_pending'" in run)
 check('single_GPU_native_linux_handoff_unchanged',all(x in acquire for x in ('Q.admit_host','Q.linux_inventory() == []','Q.gpu_empty()','create(ROOT /')) and child.index('L.wait_capture')<child.index('N.capture')<child.index('owner.update(linux_worker_identity=') and 'process.wait()' in child and "child_pid=None" in child)
 # Exact old/new control functions, allowing solely numeric source labels/version migration.
 replacements={'continuation_queue_v14':'continuation_queue_v13','repair_owned_worker_v9':'repair_owned_worker_v8','repair_runtime_owner_v9':'repair_runtime_owner_v8','finetune_v12':'finetune_v11','Q14':'Q13','F12':'F11'}
 class ControlNormalize(ast.NodeTransformer):
  def visit_Constant(self,n):
   if isinstance(n.value,str):
    for a,b in replacements.items():n.value=n.value.replace(a,b)
   elif type(n.value)is int and n.value==14:n.value=13
   return n
 for name in ('validate_native','claim_worker_owner','queue_roles'):
  check('owner_'+name+'_exact_control',dump(ControlNormalize().visit(copy.deepcopy(node(M/'repair_runtime_owner_v9.py',name))))==dump(node(M/'repair_runtime_owner_v8.py',name)))
 check('worker_admitted_before_dynamic_training_import',ast.unparse(node(M/'repair_owned_worker_v9.py','run')).index('O.enter(')<ast.unparse(node(M/'repair_owned_worker_v9.py','run')).index('importlib.import_module') and "sys.platform == 'linux'" in ast.unparse(node(M/'repair_runtime_owner_v9.py','enter')) and 'os.getuid() == 1000' in ast.unparse(node(M/'repair_runtime_owner_v9.py','enter')))
 check('original_F3_F2_fit_micro1_chunk8_preserved','v2._FIT(condition, seed, fold, resume=resume, micro_subjects=1, window_chunk=8)' in ast.unparse(node(f,'fit')) and 'TEMPLATE.activate(trace, v2, batching)' in ast.unparse(node(f,'activate')) and "condition == D.REPAIR_CONDITIONS[1]" in ast.unparse(node(f,'run')))
 check('native_counts_only_supplementary_and_not_static_audit','native_backward_supplementary' in ast.unparse(node(f,'activate')) and "'supplementary_only': True" in ast.unparse(node(f,'native_delta')) and "'added_to_model_backward_or_seconds': False" in ast.unparse(node(f,'native_delta')))
 # Pure in-memory pre/post-registration lifecycle. No real fit or fixture file is written.
 base=Path('C:/review_fixture_only').resolve();rp=base/'continuation_queue_v14/registration.json';hp=base/'continuation_queue_v14/handoff/blocked_statuses.json';original=[];values={};bindings={};previous={'verified':15};qdesc={'path':'q14','sha256':'q14bytes'}
 def add(path,value,digest):
  path=Path(path).resolve();values[str(path)]=copy.deepcopy(value);b={'path':str(path),'sha256':digest};bindings[str(path)]=b;return b
 def checked(b):
  p=Path(b['path']).resolve();require(bindings.get(str(p))==b,'Fixture exact bound bytes');return p
 for i in range(15):original.append(add(base/'runs'/str(i)/'status.json',{'status':'blocked','condition':Q.CONDITIONS[0]},'original'+str(i)))
 values[str(base/'continuation_queue_v13/registration.json')]={'pending_mamba_statuses':original}
 rows=[]
 for i,o in enumerate(original):rows.append({'original':o,'archived':add(hp.parent/('blocked_'+str(i)+'.json'),{'status':'blocked','condition':Q.CONDITIONS[0]},o['sha256'])})
 handoff={'source':qdesc,'previous_CodeBrain_completion':previous,'rows':rows};hb=add(hp,handoff,'handoff')
 values[str(rp)]={'version':14,'registered_conditions':list(Q.CONDITIONS),'mamba_pre_registration_handoff':hb,'pending_mamba_statuses_at_registration':original}
 fake=types.SimpleNamespace(OUT=base,read=lambda p:copy.deepcopy(values[str(p)]),checked=checked,desc=lambda p:qdesc)
 ns={'S':fake,'SOURCE':d,'REPAIR_CONDITIONS':D.REPAIR_CONDITIONS,'previous_codebrain_completion':lambda:previous,'read_bound':lambda b:fake.read(checked(b)),'require':require};H.extract(d,['mamba_pre_registration_evidence'],ns);verify=ns['mamba_pre_registration_evidence']
 with patch.object(Path,'exists',lambda p:False):check('unregistered_requires_actual_original_blocked',verify() is None)
 values[original[0]['path']]['status']='ready'
 with patch.object(Path,'exists',lambda p:False):reject('unregistered_ready_rejected',verify)
 # Once registered, current statuses advance legitimately and the old immutable bytes govern history.
 with patch.object(Path,'exists',lambda p:p==rp):
  check('registered_current_ready_allowed_via_exact_old_snapshots',verify()==hb)
  values[str(hp)]['rows'][0]['archived']['sha256']='wrong';reject('tampered_archived_hash_rejected',verify);values[str(hp)]=copy.deepcopy(handoff)
  values[str(hp)]['rows']=rows[:-1];reject('missing_historical_fit_rejected',verify);values[str(hp)]=copy.deepcopy(handoff)
  values[str(hp)]['previous_CodeBrain_completion']={'verified':0};reject('changed_Q13_history_rejected',verify);values[str(hp)]=copy.deepcopy(handoff)
  values[str(hp.parent/'blocked_0.json')]['status']='ready';reject('false_historical_ready_rejected',verify);values[str(hp.parent/'blocked_0.json')]['status']='blocked'
  values[str(rp)]['registered_conditions']=['CodeBrain-lastblock-finetune'];reject('wrong_new_condition_rejected',verify)
 check('CPU16_mixed_true11_12_delegation_with_no_math_change',"versions == {11}" in (A/'audit_completed_finetune_checkpoints_v16.py').read_text() and "return OLD_V.trace_boundary(base.legacy_trace_boundary" in (A/'audit_completed_finetune_checkpoints_v16.py').read_text() and "return V.trace_boundary(base.legacy_trace_boundary" in (A/'audit_completed_finetune_checkpoints_v16.py').read_text() and "43ab48da0bd862399d13aed718ad87500a718e795e9742663d78ba97c958252f" in (A/'audit_completed_finetune_checkpoints_v16.py').read_text())
 check('E11_binds_actual_two_training_provenances','(11, \'CodeBrain-lastblock-finetune\'), (12, \'EEGMamba-lastblock-finetune\')' in ast.unparse(node(e,'verify_training_execution')) and 'D.previous_codebrain_completion()' in ast.unparse(node(e,'verify_gate')) and "terminal.get('execution_version') == 14" in ast.unparse(node(e,'verify_gate')))
 check('true360_and_actual135_405_gate_before_evaluator','training_complete\') == 360' in ast.unparse(node(q,'cpu_gate')) and "report['pending_fits'] == []" in ast.unparse(node(q,'cpu_gate')) and "cpu.get('schema_version') == 16" in ast.unparse(node(e,'validate_gate_data')) and ast.unparse(node(e,'worker')).index('verify_gate(gate_binding)')<ast.unparse(node(e,'worker')).index('driver = isolated_driver()'))
 # Actual immutable engineering evidence; no Tensor or raw-array deserialization.
 folder=OUT/'deterministic_repair_qualification_v10/EEGMamba-lastblock-finetune/1791824574948282100';actual=read(folder/'report.json');controller=read(folder/'controller.json');released=read(folder/'released_owner.json');worker=read(folder/'worker_owner.json');admission=read(folder/'admission.json')
 check('actual_success_report_attempt_source_exit0',desc(folder/'report.json')['sha256']=='85619b334e6eacf1983adee0c4b24c8d3be9f1f7e0a6fe8e49d8d9d224392ece' and read(folder/'attempt.json')==actual and actual['source']==desc(M/'qualify_deterministic_repairs_v10.py') and type(controller['returncode'])is int and controller['returncode']==0 and controller['status']=='worker_returned')
 check('actual_strict_phases_native_forward_resume',actual['GPU_engineering_checks_passed'] is True and actual['eligible_for_formal_registration'] is True and actual['diagnostic_only'] is False and actual['original_native_forward_bitwise_equal'] is True and actual['zero_step_optimizer_and_rng_resume_verified'] is True and [r['epoch_phase'] for r in actual['checks']]==[1,11] and all(r['repeat_bitwise_equal'] for r in actual['checks']) and all(r['loss_bitwise_equal'] and r['rng_exact_equal'] and r['all_gradients_bitwise_equal'] and not r['different_gradients'] for r in actual['repetition_diagnostics']))
 check('actual_6F4B48windows0steps_no_formal_no_test',all(actual[k]==n for k,n in [('forward_returned',6),('backward_returned',4),('encoder_windows_returned',48),('optimizer_step_started',0),('optimizer_step_returned',0),('formal_fits',0)]) and actual['outer_test_read'] is False)
 counts=actual['native_backward_function_counts'];check('native_calls_supplementary_last_layer16_each',len(counts)==12 and all(all(v==0 for v in x['counts'].values()) for x in counts[:11]) and counts[11]['counts']=={'chunk_wrappers_started':2,'chunk_native_calls_started':16,'chunk_native_calls_returned':16,'causal_wrappers_started':2,'causal_native_calls_started':16,'causal_native_calls_returned':16})
 check('actual_released_owner_preCUDA_admission_bound',not (folder.parent.parent/'owner.json').exists() and worker['native']['controller_lease']=={'path':(folder.parent.parent/'owner.json').relative_to(W).as_posix(),'sha256':desc(folder/'released_owner.json')['sha256']} and admission==controller['admission'] and actual['admission']==desc(folder/'admission.json') and worker['GPU_before_runtime']['context_count']==admission['GPU_admission']['context_count']==0)
 failed=OUT/'mamba_repeat_diagnosis_v1/EEGMamba-lastblock-finetune/1791823118874568500/attempt.json';diag=read(failed)
 check('true_failed_diagnostic_cost_remains_separate',diag['status']=='failed' and diag['primary_engineering_seconds']==38.444831497 and actual['primary_engineering_seconds']==66.17148993200001 and controller['controller_seconds_primary_cost'] is False)
 cost=A/'cost_aggregation_v17.py';body=node(cost,'collect');augment=[x for x in ast.walk(body) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='augment'];check('original_cost9_augment_once_no_16_nested_collector',len(augment)==1 and 'previous.load_v5()' in ast.unparse(body) and 'old.collect()' in ast.unparse(body))
 rows=c['actual_engineering_primary_rows'];check('seven_real_engineering_primaries_once_no_controller_sum',len(rows)==7 and len({x['artifact']['path'] for x in rows})==7 and sum(x['observed_counters']['optimizer_step_returned'] for x in rows)==0 and sum(x['observed_counters']['backward_returned'] for x in rows)==22 and all(x['aggregation_role']=='primary_observed_invocation' for x in rows) and any(x['artifact']==desc(failed) for x in rows) and any(x['artifact']==desc(folder/'attempt.json') for x in rows))
 for row in rows:require(desc(W/row['artifact']['path'])==row['artifact'],'Cost input changed')
 check('all_actual_cost_inputs_still_same_bytes',True)
 check('no_runtime_tensor_GPU_import_or_registration',not any(x in sys.modules for x in ('torch','numpy')) and not (Q.ROOT/'registration.json').exists() and not (Q.ROOT/'registration_intent.json').exists())
 report={'schema':'independent_mamba_final15_chain_source_review_v1','status':'new_repair_chain_sources_reviewed','passed':True,'reviewer':'/root/loss3_independent_review','reviewer_source':desc(SOURCE),'reviewed_sources':sources,'checks':checks,'author_contract':desc(contract),'actual_success_report':desc(folder/'report.json'),'actual_success_controller':desc(folder/'controller.json'),'actual_success_released_owner':desc(folder/'released_owner.json'),'actual_failed_diagnostic':desc(failed),'actual_engineering_primary_rows':rows,'original175_bytes_preserved':True,'scope':['Original Mamba15 only; completed original345 including genuine CodeBrain15 remain under original provenance.','Immutable original blocked-status byte handoff is created before any ready transition; post-registration verification accepts legitimate current-state progress only via exact archived history.','Actual successful Qual10 is required; no repeated CodeBrain or Mamba engineering probes introduced.','F12 delegates original F1/F2/F3 math/protocol and separately snapshots native backward calls; CPU16/E11 route real CodeBrain F11 and real Mamba F12 without relabeling.','True360 plus actual CPU16 135FT/405 loads and zero pending/deferred/error gate precedes owned E11; Q13/Q14 and all source bindings retained.','Cost17 preserves all actual failed/qualified/diagnostic attempts once, all old loss/unknown time, and nested timers/native-call counts as supplemental only.'],'model_calls':0,'Tensor_operations':0,'GPU_operations':0,'process_inventory_queries':0,'process_controls':0,'formal_registration_writes':0,'outer_test_reads':0,'actual_remaining_Mamba15_completed':False,'actual_full135_405_CPU_gate_complete':False,'actual_E11_evaluation_complete':False,'all_plan_complete':False,'limitations':['Actual training, final complete CPU audit, test evaluation, resource supplements and cost/full semantic gates still pending.','Original CUDA backward equivalence and speed improvement remain unestablished; native/autotuner internal kernel counts remain unknown.','EEGMamba external-pretraining participant overlap exclusion remains unknown.'],'captured_ns':time.time_ns()}
 require(Q.sources()==sources,'Source changed during independent review');target=A/('independent_mamba_final15_chain_source_review_'+str(report['captured_ns'])+'.json')
 with target.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 print(json.dumps({'report':desc(target),'passed':True,'checks':len(checks),'source_count':len(sources)}))
if __name__=='__main__':main()

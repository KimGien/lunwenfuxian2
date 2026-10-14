"""Narrow stdlib/AST contracts. No process, model, Tensor or real registration."""
from pathlib import Path
import ast, copy, hashlib, importlib.util, json, sys, time, types
WORK=Path(__file__).resolve().parents[3];sys.path.insert(0,str(WORK))
A=Path(__file__).resolve().parent;M=WORK/'tridep_ccml/eeg_all_candidates';OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
CHECKS={}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def check(name,value):require(value,name);CHECKS[name]=True
def rejects(name,fn):
 try:fn()
 except (ValueError,KeyError,AssertionError):CHECKS[name]=True;return
 raise AssertionError(name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def desc(p):return {'path':Path(p).relative_to(WORK).as_posix(),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def function(path,name,env):
 node=next(n for n in ast.walk(ast.parse(Path(path).read_text(encoding='utf-8'))) if isinstance(n,ast.FunctionDef) and n.name==name)
 space={'require':require,**env};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path)+':fake','exec'),space);return space[name]
def load(path):
 spec=importlib.util.spec_from_file_location('_fake_'+Path(path).stem,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def main():
 from tridep_ccml.eeg_all_candidates import continuation_queue_v14 as Q, deterministic_repair_execution_v9 as D
 source=Q.sources();check('exact_source_paths_unique',len(source)==len({x['path'] for x in source}))
 prior=read(OUT/'continuation_queue_v13/registration.json')
 check('old175_sources_preserved',len(prior['sources'])==175 and all(sha(WORK/x['path'])==x['sha256'] and x in source for x in prior['sources']))
 names=['finetune_v12','repair_runtime_owner_v9','repair_owned_worker_v9','continuation_queue_v14','deterministic_repair_execution_v9','repair_observation_v9','evaluate_v11']
 analysis=['audit_completed_finetune_checkpoints_v16','cost_aggregation_v17','review_actual_repair_engineering_v8','repair_postprocessing_gate_v8']
 for name in names+analysis:
  path=(M if name in names else A)/(name+'.py');compile(path.read_text(encoding='utf-8'),str(path),'exec');check('bound_'+name,desc(path) in source)
 check('Mamba_only15_scope',Q.CONDITIONS==('EEGMamba-lastblock-finetune',))
 for name in ('qualify_deterministic_repairs_v10','diagnose_mamba_repeat_v1','mamba_private_backward_graph_v4','mamba_backward_reductions_v1'):
  check('actual_numeric_or_diagnostic_source_'+name,desc(M/(name+'.py')) in source)
 qtext=Path(Q.__file__).read_text(encoding='utf-8');dtext=Path(D.__file__).read_text(encoding='utf-8')
 check('no_mutable_Qual10_boundary_after_registration','Q.check_boundary()' not in qtext and 'provider.check_boundary()' not in dtext)
 check('no_evaluation_in_training_controller','evaluate_v11' not in qtext and 'all_training_CPU_gate_complete_evaluation_pending' in qtext)
 register=next(n for n in ast.parse(qtext).body if isinstance(n,ast.FunctionDef) and n.name=='register')
 segment=ast.get_source_segment(qtext,register)
 check('intent_then_blocked_bytes_then_first_ready',segment.index("create(ROOT / 'registration_intent.json'")<segment.index('archive_blocked_statuses(blocked)')<segment.index('R.lock_condition')<segment.index("'ready'"))
 check('strict_original345_and_final360', 'len(others) == 345' in qtext and "get('training_complete') == 360" in qtext)
 check('blocked_byte_create_new',"destination.open('xb')" in qtext and "archived['sha256'] == original['sha256']" in qtext)
 check('native_tree_captured_after_linux_identity',qtext.index('linux_identity = L.wait_capture')<qtext.index('native_tree = N.capture')<qtext.index('owner.update(linux_worker_identity='))
 # Fake CPU and evaluator version routing: no imports of their runtime or tensors.
 route_calls=[]
 def route(version):return types.SimpleNamespace(trace_boundary=lambda *a:route_calls.append(version) or version,verify_execution=lambda *a,**k:version)
 cpu=function(A/'audit_completed_finetune_checkpoints_v16.py','boundary',{'base':types.SimpleNamespace(legacy_trace_boundary=None),'old_boundary':lambda *a:3,'V':route(12),'OLD_V':route(11)})
 inputs=lambda v:[{'record':{'implementation':{'version':v}}}]
 check('CPU16_routes_true_CB11',cpu({'condition':'CodeBrain-lastblock-finetune'},{'actual_deterministic_repair_execution':{}},inputs(11))==11)
 check('CPU16_routes_true_Mamba12',cpu({'condition':'EEGMamba-lastblock-finetune'},{'actual_deterministic_repair_execution':{}},inputs(12))==12)
 check('CPU16_old3_delegate',cpu({'condition':'CSBrain-lastblock-finetune'},{},inputs(3))==3)
 rejects('CPU16_rejects_false_CB12',lambda:cpu({'condition':'CodeBrain-lastblock-finetune'},{},inputs(12)))
 rejects('CPU16_rejects_mixed_versions',lambda:cpu({'condition':'EEGMamba-lastblock-finetune'},{},inputs(11)+inputs(12)))
 evalfn=function(M/'evaluate_v11.py','verify_training_execution',{'V':route(12),'OLD_V':route(11),'X':types.SimpleNamespace(verify_execution_acceleration=lambda *a,**kw:3)})
 md=lambda v,c:{'actual_deterministic_repair_execution':{'observer_version':v},'condition':c,'seed':11,'fold':1}
 check('E11_actual_CB11',evalfn(md(11,'CodeBrain-lastblock-finetune'))==11)
 check('E11_actual_Mamba12',evalfn(md(12,'EEGMamba-lastblock-finetune'))==12)
 rejects('E11_rejects_relabel_CB12',lambda:evalfn(md(12,'CodeBrain-lastblock-finetune')))
 gate=function(M/'evaluate_v11.py','validate_gate_data',{})
 good={'schema_version':16,'auditor_source':{'path':'cpu','sha256':'h'},'all_135_finetune_checkpoints_verified':True,'verified_completed_finetune_fits':135,'training_complete_finetune_observed':135,'registered_finetune_fits':135,'checkpoint_loads':405,'errors':[],'deferred_completed_fits':[],'pending_fits':[]}
 g={'passed':True,'original_training_complete':360,'FT':135,'CPU_loads':405};gate(g,good,good['auditor_source']);CHECKS['full_CPU16_gate']=True
 for key,value in [('schema_version',15),('verified_completed_finetune_fits',120),('checkpoint_loads',360),('errors',['x']),('deferred_completed_fits',['x']),('pending_fits',['x'])]:
  bad=dict(good);bad[key]=value;rejects('full_gate_rejects_'+key,lambda bad=bad:gate(g,bad,good['auditor_source']))
 # Plain dictionary native function counts only; these do not modify model counters.
 delta=function(M/'finetune_v12.py','native_delta',{})
 before=[{'layer_index':i,'counts':{'chunk_native_calls_started':0,'chunk_native_calls_returned':0}} for i in range(12)]
 after=copy.deepcopy(before);after[11]['counts']={'chunk_native_calls_started':8,'chunk_native_calls_returned':7}
 value=delta(before,after);check('native_counts_keep_unreturned_and_supplementary',value['delta'][11]['counts']['chunk_native_calls_returned']==7 and value['supplementary_only'] and not value['added_to_model_backward_or_seconds'])
 bad=copy.deepcopy(after);bad[0]['counts']['chunk_native_calls_started']=-1;rejects('native_negative_rejected',lambda:delta(before,bad))
 # Cost scope is actual immutable JSON only, no collector or binary/state audit.
 cost=load(A/'cost_aggregation_v17.py')
 rows=[]
 patterns=[]
 tree=ast.parse((A/'cost_aggregation_v17.py').read_text(encoding='utf-8'))
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='add_repair_ledgers')
 for n in fn.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='patterns' for t in n.targets):patterns.extend(ast.literal_eval(n.value))
  if isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='patterns':patterns.extend(ast.literal_eval(n.value))
 check('cost17_unique_patterns',len(patterns)==len(set(patterns)))
 for version in range(4,11):check('cost17_retains_Qual'+str(version),all('deterministic_repair_qualification_v'+str(version)+'/*/*/'+n in patterns for n in ['attempt.json','controller.json']))
 for pattern in patterns:
  if not pattern.endswith('attempt.json'):continue
  for path in OUT.glob(pattern):
   value=read(path);row=cost.repair_worker_record(desc(path),value,lambda d:{'observed_counters':{k:v for k,v in d.items() if k.endswith('_returned') and type(v)is int}});rows.append(row)
 check('qualifier_and_diagnostic_primary_unique',len(rows)==len({x['artifact']['path'] for x in rows}))
 diag=[x for x in rows if '/mamba_repeat_diagnosis_v1/' in x['artifact']['path']]
 check('actual_single_failed_diagnosis_counted_once',len(diag)==1 and diag[0]['seconds_observed']==38.444831497)
 passed=[x for x in rows if '/deterministic_repair_qualification_v10/' in x['artifact']['path']]
 check('actual_Qual10_success_primary_once',len(passed)==1 and passed[0]['seconds_observed']==read(OUT/'deterministic_repair_qualification_v10/EEGMamba-lastblock-finetune/1791824574948282100/report.json')['primary_engineering_seconds'] and passed[0]['status']=='engineering_verified')
 post=load(A/'repair_postprocessing_gate_v8.py')
 check('post8_pins_current_sources',all(sha(p)==h for p,h in post.PINS.items()))
 check('post8_requires_Q13_Q14_release',all(x in post.OWNERS for x in ['continuation_queue_v13/owner.json','continuation_queue_v14/owner.json']))
 check('old175_still_original',all(sha(WORK/x['path'])==x['sha256'] for x in prior['sources']))
 result={'schema':'mamba_final15_chain_stdlib_contract_v1','passed':True,'source':desc(Path(__file__)),'sources':source,'source_count':len(source),'new_sources':[desc((M if n in names else A)/(n+'.py')) for n in names+analysis],'checks':CHECKS,'actual_engineering_primary_rows':rows,'model_calls':0,'GPU_calls':0,'Tensor_operations':0,'process_queries':0,'formal_registration_writes':0,'old175_sources_preserved':True}
 target=A/('mamba_final15_chain_stdlib_contract_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'artifact':desc(target),'passed':True,'checks':len(CHECKS),'source_count':len(source)}))
if __name__=='__main__':main()

import ast,copy,hashlib,json,types,time
from pathlib import Path
W=Path(__file__).resolve().parents[3];A=W/'output/analysis/eeg_all_candidates_v1_20261008';M=W/'tridep_ccml/eeg_all_candidates';O=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008';checks=[]
names=['cost_aggregation_v18.py','repair_postprocessing_gate_v9.py','supplemental_head_resource_measurement_v8.py','finalize_full_plan_cost_v8.py','audit_full_plan_acceptance_v8.py','pause_memory_resume_cost_companion_v3.py','sync_boundary_v3_cost_companion_v3.py','audit_memory_pause_final_delivery_v3.py','system_standby_cost_companion_v3.py','audit_system_standby_final_delivery_v3.py']
def text(p):return p.read_text(encoding='utf-8')
def desc(p):p=p.resolve();return {'path':p.relative_to(W).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def ok(x,n):
 if not x:raise AssertionError(n)
 checks.append(n)
def funcs(p):return {n.name:n for n in ast.parse(text(p)).body if isinstance(n,ast.FunctionDef)}
def astsame(p,q,n):return ast.dump(funcs(p)[n],include_attributes=False)==ast.dump(funcs(q)[n],include_attributes=False)
for n in names:compile(text(A/n),str(A/n),'exec')
ok(True,'ten new sources parse; no execution imports')
for name in ('verify_scope_closure','all30'):
 ok(astsame(A/'audit_system_standby_final_delivery_v2.py',A/'audit_system_standby_final_delivery_v3.py',name),'original full-range/30 requirement function unchanged '+name)
ok(astsame(A/'pause_memory_resume_cost_companion_v2.py',A/'pause_memory_resume_cost_companion_v3.py','verify_loss_chain'),'all three original loss lifecycle rules unchanged')
resource=text(A/'supplemental_head_resource_measurement_v8.py')
ok("V2_SHA='351f67b6941b12cfb8139372783d92158c5040d7ad20abf4a8b2d435e5a9a54b'" in resource and 'probe_plan,exclusive_measurement_lease=D.probe_plan,D.exclusive_measurement_lease' in resource and 'len(conditions)==len(set(conditions))==12' in resource,'exact12 zero-update probe math delegated unchanged')
wide=text(A/'audit_full_plan_acceptance_v8.py');ok('CPU_BASE=CPU.load_delegate()' in wide and 'D.legacy_check_fit=CPU_BASE.load_wide().check_fit' in wide and 'CPU.B.' not in wide,'wide actually uses CPU17 patched original LR and preserved trace auditor')
fin=text(A/'finalize_full_plan_cost_v8.py');ok('proof["evaluator_execution"]["version"] == 12' in fin and 'value["resource_measurement_version"] == 8' in fin and 'snapshot.get("schema_version") == 18' in fin,'E12/resource8/Cost18 gates agree')
post=text(A/'repair_postprocessing_gate_v9.py');ok("gate['original_Q14_failed_terminal']" in post and "terminal.get('status') == 'failed'" in post and 'all_training_CPU_gate_complete_evaluation_pending' not in post,'post9 keeps Q14 failed with independent recovery gate')
constants={}
for node in ast.parse(text(M/'post_training_cpu_recovery_v1.py')).body:
 if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and isinstance(node.value,ast.Dict):
  try:constants[node.targets[0].id]=ast.literal_eval(node.value)
  except ValueError:pass
maps={}
for key in ('FAILED_CPU16','LINUX_REFERENCE','WINDOWS_REFERENCE'):
 d=constants[key];maps[d['path']]=json.loads(text(W/d['path']))
cpu17={'path':'synthetic/cpu17.json','sha256':'7'*64};value=copy.deepcopy(maps[constants['FAILED_CPU16']['path']]);value.update(schema_version=17,checkpoint_loads=405,status='all_finetune_verified');value['observed_auditor_operations_including_failed_checks']['cpu_checkpoint_loads_started']=405;value['observed_auditor_operations_including_failed_checks']['cpu_checkpoint_loads_returned']=405;maps[cpu17['path']]=value
gatebinding={'path':'synthetic/complete.json','sha256':'8'*64};gate={'CPU_audit':cpu17,'source':{'path':'synthetic/gate.py','sha256':'9'*64},'post_training_source_review':{'path':'synthetic/review.json','sha256':'a'*64},'post_training_sources':[],**{key:constants[name] for key,name in [('original_Q14_failed_terminal','Q14_FAILED'),('original_Q14_released_owner','Q14_RELEASED'),('original_child_complete','CHILD_COMPLETE')]}}
R=types.SimpleNamespace(ROOT=Path('synthetic'),verify_gate=lambda binding:gate,**{k:constants[k] for k in ('FAILED_CPU16','LINUX_REFERENCE','WINDOWS_REFERENCE')})
S=types.SimpleNamespace(desc=lambda p:gatebinding,checked=lambda d:d['path'],read=lambda p:maps[p])
ns={'RECOVERY':R,'S':S};nodes=[n for n in ast.parse(text(A/'cost_aggregation_v18.py')).body if isinstance(n,ast.FunctionDef) and n.name in ('require','category_totals','add_post_training_recovery')];exec(compile(ast.Module(body=nodes,type_ignores=[]),'cost18_fake','exec'),ns)
def normalized(data,**kwargs):return {'observed_counters':{},'seconds_observed':None,'unknown_counter_fields':[],'counter_issues':[]}
def start(preexisting=False):return {'inputs':[],'cost_records':[{'artifact':constants['FAILED_CPU16'],'category':'old_supplemental','aggregation_role':'supplemental_potentially_cumulative_test_report'}] if preexisting else [],'category_observations':{},'rules':[]}
for pre in (False,True):
 result=ns['add_post_training_recovery'](start(pre),normalized);rows=result['cost_records'];ok(len(rows)==4 and len({r['artifact']['path'] for r in rows})==4,'exact four primary scopes, existing inventory reused '+str(pre))
 ok([r['observed_counters']['cpu_checkpoint_loads_returned'] for r in rows[:2]]==[360,405] and all(r['seconds_observed'] is None for r in rows[:2]),'failed360/new405 counts; unknown seconds stay null '+str(pre))
 ok([r['seconds_observed'] for r in rows[2:]]==[maps[constants[k]['path']]['primary_stdlib_capture_seconds'] for k in ('LINUX_REFERENCE','WINDOWS_REFERENCE')],'both real capture seconds once '+str(pre))
bad=start(True);bad['cost_records'].append(copy.deepcopy(bad['cost_records'][0]))
try:ns['add_post_training_recovery'](bad,normalized)
except ValueError:ok(True,'duplicate artifact inventory rejected')
else:raise AssertionError('duplicate accepted')
for name in ('audit_memory_pause_final_delivery_v3.py','audit_system_standby_final_delivery_v3.py'):
 raw=ast.unparse(funcs(A/name)['main']);ok('require_complete' in raw and 'not args.run' in raw and 'parser.error' in raw,'strict require-complete cannot silently prepare '+name)
report={'schema':'post_training_final_compatibility_stdlib_preparation_v1','passed':True,'checks':checks,'reviewed_sources':[desc(A/n) for n in names],'training_registration':desc(O/'continuation_queue_v14/registration.json'),'source':desc(Path(__file__)),'scope':'New compatibility only; fake cost gate plus bound CPU16/platform JSON; no old contract suite rerun','Tensor_loads':0,'model_calls':0,'GPU_calls':0,'process_queries':0,'outer_test_reads':0,'actual_collect_run':False,'original_training_sources_modified':False}
p=A/('post_training_final_compatibility_stdlib_contract_'+str(time.time_ns())+'.json');p.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps({'artifact':desc(p),'checks':len(checks),'sources':report['reviewed_sources']}))

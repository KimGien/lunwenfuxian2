"""Independent Linux identity / successful-qualification reuse source review.
Actual execution entries, process inventories and GPU/model operations are never called.
Existing qualification verification hashes source/weight/probe bytes without Tensor loads.
"""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,os,sys,time,types
from pathlib import Path
SOURCE=Path(__file__).resolve(); A=SOURCE.parent; W=SOURCE.parents[3]; M=W/'tridep_ccml/eeg_all_candidates'; OUT=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
checks={}
def require(ok,msg):
    if not ok: raise ValueError(msg)
def check(label,value): require(value,label);checks[label]=True
def reject(label,fn):
    try:fn()
    except (ValueError,KeyError,AssertionError):checks[label]=True
    else:raise AssertionError('Unsafe acceptance: '+label)
def desc(path):
    p=Path(path).resolve(); return {'path':p.relative_to(W).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def checked(binding):
    p=W/binding['path'];require(desc(p)==binding,'Bound source changed '+str(p));return p
def tree(path):return ast.parse(path.read_text(encoding='utf-8-sig'))
def node(path,name):return next(x for x in tree(path).body if isinstance(x,ast.FunctionDef) and x.name==name)
def extract(path,names,ns):
    exec(compile(ast.Module(body=[copy.deepcopy(node(path,n)) for n in names],type_ignores=[]),str(path),'exec'),ns);return ns

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preparation',required=True); args=parser.parse_args()
    p=W/args.preparation; preparation=read(p)
    is_current_contract=preparation.get('passed') is True and 'candidate_execution_sources' in preparation
    check('source_preparation_or_current_contract_explicit_scope',is_current_contract or preparation.get('schema')=='read_only_linux_identity_and_success_reuse_preparation_v1')
    old=M/'qualify_deterministic_repairs_v5.py'; new=M/'qualify_deterministic_repairs_v6.py'; linux=M/'linux_worker_identity_v2.py'
    for name in ('runtime_worker','isolated_builder'):
        prior=ast.unparse(node(old,name)); current=ast.unparse(node(new,name)).replace('qualify_deterministic_repairs_v6','qualify_deterministic_repairs_v5')
        check(name+'_numeric_AST_unchanged_except_declared_module_label',current==prior)
    ns={'require':require,'CONDITIONS':('CodeBrain-lastblock-finetune','EEGMamba-lastblock-finetune')}
    extract(new,['safe_before_model'],ns)
    reject('new_qualification_rejects_already_successful_CodeBrain_before_any_IO',lambda:ns['safe_before_model'](ns['CONDITIONS'][0]))
    for path,name,args0,kwargs0 in [(new,'validate_worker',({},'EEGMamba-lastblock-finetune'),{}),(M/'repair_runtime_owner_v4.py','enter',({},'token'),{'condition':'EEGMamba-lastblock-finetune','repair_registration':{}}),(M/'evaluate_v7.py','worker',(),{'gate_binding':{},'admission_binding':{},'token':'x','mode':'mamba_linux'})]:
        ns={'require':require,'sys':types.SimpleNamespace(platform='linux'),'os':types.SimpleNamespace(getuid=lambda:0)}
        extract(path,[name],ns)
        reject(path.stem+'_rejects_root_model_before_IO',lambda ns=ns,name=name,args0=args0,kwargs0=kwargs0:ns[name](*args0,**kwargs0))
    # Root subprocess construction only: fake runner, no subprocess launched.
    snapshot_fn=node(linux,'snapshot')
    for platform in ('win32','linux'):
        calls=[]
        reply={'observer_uid':0,'observer_scope':'read_only_proc_no_models_or_gpu','query_succeeded':True,'errors':[],'boot_id':'boot','rows':[]}
        def fake_run(command,**kw):calls.append((command,kw));return types.SimpleNamespace(stdout=json.dumps(reply))
        ns={'require':require,'sys':types.SimpleNamespace(platform=platform),'SOURCE':linux,'WORK':W,'linux_path':lambda path:'/mnt/c/work space/'+Path(path).name,'subprocess':types.SimpleNamespace(run=fake_run),'json':json,'os':types.SimpleNamespace(getpid=lambda:70)}
        extract(linux,['snapshot'],ns); result=ns['snapshot'](); command,kw=calls[0]
        check(platform+'_root_observer_fixed_isolated_command',len(calls)==1 and command[0]==('wsl.exe' if platform=='win32' else '/mnt/c/Windows/System32/wsl.exe') and command[1:6]==['-d','TRI-DEP-Ubuntu-24.04','-u','root','--exec'] and command[7:11]==['-I','-S','-B','-c'] and kw['timeout']==30 and kw['check'] is True and ('creationflags' in kw)==(platform=='win32'))
        program=command[-1]
        imports=[alias.name for n in ast.walk(ast.parse(program)) if isinstance(n,ast.Import) for alias in n.names]
        check(platform+'_root_program_source_hash_bound_and_stdlib_only',imports==['hashlib','importlib.util','json','os'] and desc(linux)['sha256'] in program and 'snapshot_local(exclude_pid=os.getpid(), observer_caller_pid=' in program and 'torch' not in program)
        reply['observer_uid']=1000
        reject(platform+'_nonroot_observer_reply_rejected',lambda:ns['snapshot']())
    ns={'require':require};extract(linux,['select_unique','validate_bound'],ns)
    row={'pid':70,'parent':60,'start_ticks':'100','argv':['python','-m','m','--token','a'],'executable':'/venv/python','parent_identity':{'pid':60,'parent':1,'start_ticks':'80','argv':['/init'],'executable':'/init'}}
    value={'observer_uid':0,'observer_scope':'read_only_proc_no_models_or_gpu','query_succeeded':True,'errors':[],'boot_id':'boot','rows':[row]}
    binding=ns['select_unique'](value,module='m',flags={'--token':'a'})
    check('exact_linux_caller_accepted',ns['validate_bound'](value,binding,module='m',flags={'--token':'a'},worker_pid=70)==binding)
    reject('wrong_linux_caller_rejected',lambda:ns['validate_bound'](value,binding,module='m',flags={'--token':'a'},worker_pid=71))
    for field in ('boot','parent','duplicate','errors','nonce'):
        bad=copy.deepcopy(value)
        if field=='boot':bad['boot_id']='newboot'
        if field=='parent':bad['rows'][0]['parent_identity']['start_ticks']='81'
        if field=='duplicate':bad['rows'].append(copy.deepcopy(row))
        if field=='errors':bad['errors']=[{'error_type':'PermissionError'}]
        if field=='nonce':bad['rows'][0]['argv'][-1]='b'
        reject('changed_'+field+'_rejected',lambda bad=bad:ns['validate_bound'](bad,binding,module='m',flags={'--token':'a'},worker_pid=70))
    # Exact readonly relay exclusion: no broad wsl/init exemption, using fake /proc only.
    import re
    caller={'pid':70,'parent':60,'start_ticks':'100','argv':['python','-m','tridep_ccml.eeg_all_candidates.fixture','--token','a'],'executable':'/venv/python'}
    parent={'pid':60,'parent':1,'start_ticks':'80','argv':['/init'],'executable':'/init'}
    root_observer={'pid':900,'parent':899,'start_ticks':'130','argv':['python','-I','-S','-B','-c','pinned:eeg_all_candidates:sourceSHA'],'executable':'/venv/python'}
    bridge={'pid':80,'parent':70,'start_ticks':'120','argv':['/init','/mnt/c/Windows/System32/wsl.exe','/mnt/c/Windows/System32/wsl.exe','-d','TRI-DEP-Ubuntu-24.04','-u','root','--exec','/lexical/venv/python','-I','-S','-B','-c',root_observer['argv'][-1]],'executable':'/init'}
    mapping={70:caller,60:parent,900:root_observer,80:bridge}
    visible=[70,80]
    class FakeLeaf:
        def __init__(self,pid=None,kind=None):self.pid=pid;self.kind=kind;self.name=str(pid)
        def __truediv__(self,name):return FakeLeaf(self.pid,name)
        def read_bytes(self):return bytes(0).join(x.encode() for x in []) if self.kind!='cmdline' else b'\x00'.join(x.encode() for x in mapping[self.pid]['argv'])
        def read_text(self):return 'boot' if self.kind=='sys/kernel/random/boot_id' else ''
    class FakeProc:
        def iterdir(self):return [FakeLeaf(pid) for pid in visible]
        def __truediv__(self,name):return FakeLeaf(None,name)
    ns={'require':require,'Path':Path,'os':types.SimpleNamespace(getuid=lambda:0,getpid=lambda:900),'sys':types.SimpleNamespace(platform='linux'),'re':re,'MODULE_PATTERN':'eeg_all_candidates','WORK':W,'linux_path':lambda p:'/lexical/venv/python','process_identity':lambda pid,proc:copy.deepcopy(mapping[pid])}
    extract(linux,['snapshot_local'],ns)
    actual=ns['snapshot_local'](exclude_pid=900,proc_root=FakeProc(),observer_caller_pid=70)
    check('one_exact_readonly_interop_relay_excluded_with_full_identity',len(actual['rows'])==1 and actual['rows'][0]['pid']==70 and len(actual['excluded_exact_readonly_interop_relays'])==1 and actual['excluded_exact_readonly_interop_relays'][0]['start_ticks']=='120')
    for field in ('argv','parent','executable'):
        original=copy.deepcopy(mapping[80])
        if field=='argv':mapping[80]['argv'][-1]+=':not_the_current_program'
        if field=='parent':mapping[80]['parent']=60
        if field=='executable':mapping[80]['executable']='/different/init'
        rejected=ns['snapshot_local'](exclude_pid=900,proc_root=FakeProc(),observer_caller_pid=70)
        check('relay_changed_'+field+'_remains_visible_for_exclusivity_rejection',len(rejected['rows'])==2 and rejected['excluded_exact_readonly_interop_relays']==[])
        mapping[80]=original
    mapping[81]={**copy.deepcopy(mapping[80]),'pid':81,'start_ticks':'121'};visible.append(81)
    reject('two_matching_readonly_relays_rejected',lambda:ns['snapshot_local'](exclude_pid=900,proc_root=FakeProc(),observer_caller_pid=70));visible.pop()
    nested_path=A/'actual_unprivileged_root_observer_bridge_1791780223916.json';nested=read(nested_path);snap=nested['snapshot']
    check('actual_nested_bridge_evidence_uid1000_to_root0_no_model',nested['caller_uid']==1000 and nested['model_GPU_calls']==0 and snap['observer_uid']==0 and snap['errors']==[] and len(snap['rows'])==len(snap['excluded_exact_readonly_interop_relays'])==1)
    actual_relay=snap['excluded_exact_readonly_interop_relays'][0];actual_caller=snap['rows'][0]
    check('actual_nested_bridge_evidence_binds_current_helper_hash_and_caller',desc(linux)['sha256'] in actual_relay['argv'][-1] and actual_relay['parent']==actual_caller['pid'] and actual_relay['parent_identity']=={k:v for k,v in actual_caller.items() if k!='parent_identity'})
    # Current CPU/auditor/gate bindings; no actual CPU audit/evaluation.
    cpu=A/'audit_completed_finetune_checkpoints_v11.py'; raw=cpu.read_text(encoding='utf-8')
    check('CPU11_original_CPU8_math_and_current_observer7','audit_completed_finetune_checkpoints_v8.py' in raw and '43ab48da0bd862399d13aed718ad87500a718e795e9742663d78ba97c958252f' in raw and 'versions == {7}' in raw and 'schema_version=11' in raw and 'completed_finetune_checkpoint_audit_v11_' in raw)
    e=M/'evaluate_v7.py'; ns={'require':require}; extract(e,['validate_gate_data'],ns)
    gate={'passed':True,'original_training_complete':360,'FT':135,'CPU_loads':405}; cpu_binding=desc(cpu)
    cpu_value={'schema_version':11,'auditor_source':cpu_binding,'all_135_finetune_checkpoints_verified':True,'verified_completed_finetune_fits':135,'training_complete_finetune_observed':135,'registered_finetune_fits':135,'checkpoint_loads':405,'errors':[],'deferred_completed_fits':[],'pending_fits':[]}
    ns['validate_gate_data'](gate,cpu_value,cpu_binding);check('E7_strict_full_CPU11_shape_accepted',True)
    for field,val in [('schema_version',10),('checkpoint_loads',402),('errors',['error']),('pending_fits',['pending']),('deferred_completed_fits',['deferred'])]:
        bad=copy.deepcopy(cpu_value);bad[field]=val;reject('E7_reject_'+field,lambda bad=bad:ns['validate_gate_data'](gate,bad,cpu_binding))
    # Actual failed-before-model proof is immutable, with unknown controller elapsed preserved.
    closure_path=OUT/'deterministic_repair_qualification_v5/failure_closures/pre_model_linux_permission_1791778566044163800/closure.json';closure=read(closure_path)
    check('actual_Mamba5_failure_closure_bytes',desc(closure_path)['sha256']=='98bd04eb5256ef3d895eb750cb1a7ef9c4ac3f69ccce97b5dd37414739a21e37')
    controller_path=OUT/'deterministic_repair_qualification_v5/EEGMamba-lastblock-finetune/1791777298177356700/controller.json';controller=read(controller_path)
    check('actual_Mamba5_controller_preserved_nonterminal_without_elapsed',controller['status']=='worker_starting' and 'inclusive_controller_seconds' not in controller and not (controller_path.parent/'attempt.json').exists() and not (controller_path.parent/'worker_owner.json').exists())
    cost=A/'cost_aggregation_v12.py'
    sys.path.insert(0,str(W))
    from tridep_ccml.eeg_all_candidates import storage as S,qualify_deterministic_repairs_v4 as Q4,qualify_deterministic_repairs_v5 as Q5,qualify_deterministic_repairs_v6 as Q6,continuation_queue_v9 as Queue,deterministic_repair_execution_v4 as D,repair_observation_v4 as V
    def normalize(value,**kwargs):return {'observed_counters':{k:value.get(k+'_returned') for k in ('forward','backward','optimizer_step')},'unknown_counter_fields':[]}
    ns={'require':require,'copy':copy,'S':S,'Q':Q6,'PREVIOUS_Q':Q4,'PREVIOUS_SUCCESS_Q':Q5,'COUNTER_CATEGORY':'qualification','NESTED_CATEGORY':'nested','WORK':W}
    extract(cost,['repair_worker_record','category_totals','add_repair_ledgers'],ns)
    cost_result={'cost_records':[],'inputs':[],'input_read_errors':[],'category_observations':{},'unknown_or_opaque_records':[]}
    ns['add_repair_ledgers'](cost_result,normalize)
    rows=cost_result['cost_records']; primary=[x for x in rows if x['aggregation_role']=='primary_observed_invocation']
    check('Cost12_existing_two_real_primary_ledgers_once',len(primary)==2 and len({x['artifact']['path'] for x in rows})==len(rows) and cost_result['input_read_errors']==[])
    old_failed=next(x for x in primary if '/qualification_v4/' in x['artifact']['path'] or '/deterministic_repair_qualification_v4/' in x['artifact']['path'])
    old_success=next(x for x in primary if '/deterministic_repair_qualification_v5/' in x['artifact']['path'])
    check('Cost12_old_Qual4_failed_16seconds_preserved',old_failed['status']=='failed' and old_failed['seconds_observed']==16.700772199998028 and old_failed['observed_counters']['backward']==4)
    check('Cost12_old_CodeBrain5_success_once',old_success['status']=='engineering_verified' and old_success['observed_counters']=={'forward':6,'backward':4,'optimizer_step':0})
    unknown=next(x for x in rows if x['artifact']==desc(controller_path))
    check('Cost12_Mamba5_controller_unknown_not_fabricated_primary',unknown['status']=='worker_starting' and unknown['seconds_observed'] is None and unknown['aggregation_role']=='supplemental_nested_or_unmeasured')
    future=copy.deepcopy(read(W/old_success['artifact']['path']))
    future.update(source=desc(new),condition='EEGMamba-lastblock-finetune',primary_engineering_seconds=8.0)
    future_artifact={'path':(OUT/'deterministic_repair_qualification_v6/EEGMamba-lastblock-finetune/fixture/attempt.json').relative_to(W).as_posix(),'sha256':'fixture'}
    new_row=ns['repair_worker_record'](future_artifact,future,normalize)
    total=ns['category_totals']([*rows,new_row])
    check('Cost12_each_generation_primary_once_nested_never_added',total['primary_invocations']==3 and total['counter_totals']['backward']['known_observed_subtotal']==12 and total['known_recorded_scope_seconds_subtotal']==old_failed['seconds_observed']+old_success['seconds_observed']+8.0 and total['nested_driver_times_added'] is False and total['full_category_wall_or_gpu_busy_time'] is None)
    reject('Cost12_refuses_old_success_relabelled_asnew',lambda:ns['repair_worker_record'](old_success['artifact'],future,normalize))
    calls=[];base=types.SimpleNamespace(buffered_observation=lambda*a:None,collect=lambda:(calls.append('inventory') or {'rules':[]}),load_v4=lambda:types.SimpleNamespace(normalized='n'))
    previous=types.SimpleNamespace(load_v5=lambda:base,augment=lambda result,**kwargs:(calls.append('cost9') or result))
    ns={'delegate':lambda:previous,'buffered_observation':lambda*a:None,'add_repair_ledgers':lambda result,n:(calls.append('repair') or result),'L':types.SimpleNamespace(verify_lost_f3_attempt='v',read_json='r',__file__='L3'),'S':types.SimpleNamespace(desc=lambda p:{'path':str(p),'sha256':'fixture'}),'Path':Path,'SOURCE':'cost12','DELEGATE':'cost9','V':types.SimpleNamespace(SOURCE='V4')}
    extract(cost,['collect'],ns);fake=ns['collect']()
    check('Cost12_original_inventory_and_loss3_augment_exactly_once',calls==['inventory','cost9','repair'] and fake['schema_version']==12 and fake['all_plan_wall_seconds_exact'] is None)
    # Actual readonly old successful qualification acceptance, including official weight/probe byte hashing.
    report_path=OUT/'deterministic_repair_qualification_v5/CodeBrain-lastblock-finetune/1791777157876013500/report.json';proof=D.qualification(desc(report_path),'CodeBrain-lastblock-finetune')
    check('actual_CodeBrain_Qual5_success_reused_with_true_old_source',proof['successful_qualification_reused_without_new_model_work'] is True and proof['report']['sha256']=='89f0251f35539416ec7821ee959e502a70da51d91b24e3ca1e8748881bdb9052' and proof['observed_engineering']['source']==desc(old) and len(proof['original_fit_ids'])==15)
    V.source_check();sources=Queue.sources();prior_path=A/'independent_qualifier_fsync_repair_chain_source_review_1791777049037006700.json';prior=read(prior_path)
    check('all127_sources_inherit_all115_original_bytes',len(sources)==127 and len({x['path'] for x in sources})==127 and all(x in sources for x in prior['reviewed_sources']) and desc(prior_path) in sources)
    for item in sources:checked(item)
    check('all127_current_hashes_verified',True)
    check('D4_pin_actual_Qual6_current',D.QUALIFIER_SHA==desc(new)['sha256'])
    if is_current_contract:check('fresh_preparation_same_current127',preparation['candidate_execution_sources']==sources)
    else:check('draft_preparation_is_not_claimed_as_current_source_contract',preparation['new_sources'] != sources)
    check('all_imported_execution_modules_remain_no_Tensor',not any(x in sys.modules for x in ('torch','numpy')))
    post_path=A/'repair_postprocessing_gate_v4.py';ns={'S':S,'SOURCE':post_path}
    pin_node=next(x for x in tree(post_path).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in x.targets))
    exec(compile(ast.Module(body=[pin_node],type_ignores=[]),str(post_path),'exec'),ns)
    check('postgate4_all_current_queue_evaluator_CPU_cost_pins',all(desc(p)['sha256']==sha for p,sha in ns['PINS'].items()))
    check('actual_Mamba6_not_yet_run_or_registered',not (OUT/'deterministic_repair_qualification_v6/EEGMamba-lastblock-finetune/permanent_intent.json').exists() and not (Queue.ROOT/'registration.json').exists())
    result={'schema':'independent_linux_identity_reuse_chain_source_review_v1','status':'new_repair_chain_sources_reviewed','passed':True,'reviewer':'/root/loss3_independent_review','reviewer_source':desc(SOURCE),'reviewed_sources':sources,'auxiliary_sources':[desc(post_path),desc(A/'review_actual_repair_engineering_v4.py')],'checks':checks,'source_preparation_document':desc(p),'preparation_is_current_contract':is_current_contract,'prior115_review':desc(prior_path),'actual_CodeBrain_qual5_success':proof['report'],'actual_Mamba5_failure_closure':desc(closure_path),'actual_Mamba5_original_controller':desc(controller_path),'retained_qual4_failure_attempt':old_failed['artifact'],'review_scope':['The privileged observer runs only fixed source-hash-bound isolated stdlib code; reads /proc, bound observer source and Python stdlib, no model/GPU/control/write.','Linux model worker entrypoints reject uid0 and require original uid1000. Exact captured boot/pid/parent/start/argv/exe/nonce binding and exclusive claims remain.','Qual6 numerical runtime/builder AST equals Qual5 after explicit module-name migration; only Mamba may perform new qualification.','Actual original CodeBrain Qual5 success is accepted under original source/contract/path, with no new model work; current builder AST identity is enforced.','CPU11 delegates original CPU8 and verifies actual observer7; original360/135/405 test gate remains required.','Cost12 preserves failed Qual4, successful Qual5, unknown premodule Mamba5 controller and new Qual6 each once; cost9 loss augmentation once; nested elapsed excluded.'],'model_calls':0,'GPU_queries':0,'actual_process_inventory_queries':0,'process_controls':0,'checkpoint_tensor_loads':0,'outer_test_probability_reads':0,'actual_qualification_entries_invoked':0,'new_registration_created':False,'source_byte_hashing_performed':True,'official_weight_and_engineering_checkpoint_byte_hashing_performed':True,'byte_hash_counts_and_elapsed_unmeasured':None,'root_observer_actual_runtime_invoked_by_this_reviewer':False,'nested_Linux_Windows_WSL_observer_runtime_evidence_verified':True,'actual_nested_bridge_evidence':desc(nested_path),'actual_Mamba6_GPU_qualification_complete':False,'actual_new_formal_fits_completed':0,'all_plan_complete':False,'remaining_limitations':['Actual root-provided zero-model nested observation evidence was read and checked; each future model worker still needs its actual unique identity handshake.','Actual Mamba6 engineering, both original15 formal training groups, full CPU11/E7/postprocessing/final delivery remain pending.','Mamba backward accumulation order is intentionally changed; original CUDA backward equivalence and speed improvement are not claimed.','EEGMamba external-pretraining participant overlap exclusion remains unknown.','Final delivery must retain the bound Mamba5 premodule failure closure and original unknown controller elapsed.'],'captured_ns':time.time_ns()}
    require(sources==Queue.sources(),'Candidate changed during review')
    target=A/('independent_linux_identity_reuse_chain_source_review_'+str(result['captured_ns'])+'.json')
    with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'report':desc(target),'passed':True,'reviewed_sources':len(sources),'checks':len(checks)}))
if __name__=='__main__':main()

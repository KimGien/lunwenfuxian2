"""Independent native launcher tree / repair139 source review, zero actual model operations."""
from __future__ import annotations
import argparse,ast,copy,hashlib,json,os,sys,time,types
from pathlib import Path
SOURCE=Path(__file__).resolve();A=SOURCE.parent;W=SOURCE.parents[3];M=W/'tridep_ccml/eeg_all_candidates';OUT=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008';checks={}
def require(ok,msg):
    if not ok:raise ValueError(msg)
def check(label,value):require(value,label);checks[label]=True
def reject(label,fn):
    try:fn()
    except (ValueError,KeyError,AssertionError):checks[label]=True
    else:raise AssertionError('Unsafe acceptance '+label)
def desc(path):
    p=Path(path).resolve();return {'path':p.relative_to(W).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def checked(binding):
    p=W/binding['path'];require(desc(p)==binding,'Bound input changed');return p
def tree(path):return ast.parse(path.read_text(encoding='utf-8-sig'))
def node(path,name):return next(x for x in tree(path).body if isinstance(x,ast.FunctionDef) and x.name==name)
def extract(path,names,ns):
    exec(compile(ast.Module(body=[copy.deepcopy(node(path,n)) for n in names],type_ignores=[]),str(path),'exec'),ns);return ns

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--contract',required=True);args=parser.parse_args();contract_path=W/args.contract;contract=read(contract_path)
    check('current_contract_passed',contract.get('passed') is True)
    old=M/'qualify_deterministic_repairs_v6.py';new=M/'qualify_deterministic_repairs_v7.py';native=M/'native_launcher_tree_v1.py'
    for name in ('runtime_worker','isolated_builder'):
        check(name+'_numerical_AST_unchanged_from_Qual6',ast.unparse(node(new,name)).replace('qualify_deterministic_repairs_v7','qualify_deterministic_repairs_v6')==ast.unparse(node(old,name)))
    nns={'require':require};extract(native,['capture','verify'],nns);N=types.SimpleNamespace(**nns)
    diagnosis_path=A/'actual_wsl_launcher_identity_diagnosis_1791781021708556900.json';diagnosis=read(diagnosis_path)
    kwargs={'launcher_pid':diagnosis['launched_pid'],'controller_pid':diagnosis['controller_pid'],'module':'tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v6','nonce':'readonly_fixture_digest'}
    real_tree=N.capture(diagnosis['windows_query'],**kwargs)
    check('real_Windows_and_Linux_queries_capture_same_two_object_tree',len(real_tree)==2 and N.verify(diagnosis['linux_query']['host_inventory'],real_tree,**kwargs)==real_tree)
    for mutation in ('child_birth','child_command','child_parent','primary_birth','missing_primary','duplicate_shim'):
        bad=copy.deepcopy(real_tree)
        if mutation=='child_birth':bad[1]['birth']='999'
        if mutation=='child_command':bad[1]['command']+=' other'
        if mutation=='child_parent':bad[1]['parent']=1
        if mutation=='primary_birth':bad[0]['birth']='998'
        if mutation=='missing_primary':bad=bad[1:]
        if mutation=='duplicate_shim':bad.append({**bad[1],'pid':bad[1]['pid']+1})
        reject('native_'+mutation+'_rejected',lambda bad=bad:N.verify(bad,real_tree,**kwargs))
    controller={'pid':10,'parent':1,'birth':'100','command':'python -m tridep_ccml.eeg_all_candidates.continuation_queue_v10'}
    worker_module='tridep_ccml.eeg_all_candidates.repair_owned_worker_v5'
    def native_fixture(module,nonce):
        primary={'pid':20,'parent':10,'birth':'200','command':'wsl.exe -m '+module+' --worker --condition EEGMamba-lastblock-finetune --admission-sha256 ash --token '+nonce}
        shim={'pid':21,'parent':20,'birth':'201','command':primary['command']};return [primary,shim]
    linux={'boot_id':'fixture','identity':{'pid':303}}
    linux_calls=[]
    def linux_verify(snapshot,binding,**kwargs):
        require(binding==linux and kwargs['worker_pid']==303,'Linux identity lost');linux_calls.append(kwargs);return binding
    L=types.SimpleNamespace(snapshot=lambda:{'fixture':True},validate_bound=linux_verify)
    roles=native_fixture(worker_module,'tok');items=[controller,*roles]
    owner={'id':'owner','pid':10,'execution_version':10,'native_roles':[controller],'child_pid':20,'child_native_identity':roles[0],'child_native_roles':roles,'child_token':'tok','child_job':'job','linux_worker_identity':linux}
    admission={'queue_owner_id':'owner','token':'tok','job':'job','condition':'EEGMamba-lastblock-finetune','repair_registration':{'sha256':'repair'}}
    ons={'require':require,'N':N,'L':L,'WORKER_MODULE':worker_module};extract(M/'repair_runtime_owner_v5.py',['validate_native'],ons)
    def formal(values=items):return ons['validate_native'](values,owner,admission,worker_pid=303,platform='linux',admission_binding={'sha256':'admission'})
    check('formal_native_primary_and_shim_accepted_with_Linux_check',formal()==items and len(linux_calls)==1)
    for mutation in ('shim_birth','unrelated_worker','missing_parent','missing_shim'):
        bad=copy.deepcopy(items)
        if mutation=='shim_birth':bad[-1]['birth']='999'
        if mutation=='unrelated_worker':bad.append({'pid':30,'parent':1,'birth':'300','command':'python unrelated'})
        if mutation=='missing_parent':bad=bad[1:]
        if mutation=='missing_shim':bad=bad[:-1]
        reject('formal_'+mutation+'_rejected',lambda bad=bad:formal(bad))
    # Exercise actual qualification and evaluator admission functions in isolated fake namespaces.
    pure={'require':require};extract(new,['admit_host'],pure);admit_host=pure['admit_host']
    qmodule='tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v7';qroles=native_fixture(qmodule,'ash');qitems=[controller,*qroles]
    qowner={'pid':10,'condition':'EEGMamba-lastblock-finetune','native_roles':[controller],'child_pid':20,'child_native_roles':copy.deepcopy(qroles),'linux_worker_identity':linux}
    fake_desc=lambda p:{'path':str(p),'sha256':'fake'};queue=OUT/'continuation_queue_v5'
    qadmission={'condition':'EEGMamba-lastblock-finetune','source':fake_desc(new),'terminal':fake_desc(queue/'status.json'),'sources':[],'allowed_host':[controller]}
    qns={'require':require,'N':N,'L':L,'SOURCE':new,'QUEUE':queue,'ROOT':OUT/'qualification_fixture','sys':types.SimpleNamespace(platform='linux'),'os':types.SimpleNamespace(getuid=lambda:1000,getpid=lambda:303),'time':time,'desc':fake_desc,'read':lambda p:qowner,'checked':lambda v:v,'check_pause':lambda:None,'check_boundary':lambda:None,'source_check':lambda:None,'host_inventory':lambda:qitems,'admit_host':admit_host}
    extract(new,['validate_worker'],qns)
    qproof=qns['validate_worker'](qadmission,'EEGMamba-lastblock-finetune',admission_sha256='ash')
    check('qualification_accepts_captured_complete_tree_and_preserves_Linux_proof',qproof['allowed_host']==qitems and qproof['linux_worker_identity']==linux)
    qitems[-1]['birth']='changed';reject('qualification_changed_shim_rejected',lambda:qns['validate_worker'](qadmission,'EEGMamba-lastblock-finetune',admission_sha256='ash'));qitems[-1]['birth']='201'
    qitems.append({'pid':31,'parent':1,'birth':'301','command':'python another'});reject('qualification_extra_experiment_rejected',lambda:qns['validate_worker'](qadmission,'EEGMamba-lastblock-finetune',admission_sha256='ash'));qitems.pop()
    emodule='tridep_ccml.eeg_all_candidates.evaluate_v8';epath=M/'evaluate_v8.py';eroles=native_fixture(emodule,'tok');eitems=[controller,*eroles]
    eowner={'source':fake_desc(epath),'pid':10,'native_roles':[controller],'child_token':'tok','child_pid':20,'child_identity':eroles[0],'child_native_roles':eroles,'linux_worker_identity':linux,'child_mode':'mamba_linux','child_admission':{'sha256':'admission'},'CPU_gate':{'sha256':'cpu'}}
    ens={'require':require,'N':N,'L':L,'S':types.SimpleNamespace(desc=fake_desc),'SOURCE':epath,'MODULE':emodule,'Q':types.SimpleNamespace(admit_host=admit_host)};extract(epath,['admit_worker'],ens)
    check('evaluation_accepts_captured_complete_native_tree',ens['admit_worker'](eitems,eowner,'tok',303,'linux')==eitems)
    extra={'pid':32,'parent':1,'birth':'302','command':'wsl.exe unknown'}
    reject('evaluation_extra_experiment_rejected',lambda:ens['admit_worker']([*eitems,extra],eowner,'tok',303,'linux'))
    changed=copy.deepcopy(eitems);changed[-1]['birth']='999';reject('evaluation_changed_shim_rejected',lambda:ens['admit_worker'](changed,eowner,'tok',303,'linux'))
    for path,name in [(new,'run_controller'),(M/'continuation_queue_v10.py','child'),(epath,'controller')]:
        rendered=ast.unparse(node(path,name));capture=rendered.index('L.wait_capture(');nativepos=rendered.index('N.capture(',capture);admitpos=rendered.index('admit_host(',nativepos);publish=rendered.index('linux_worker_identity=linux_identity',admitpos)
        check(path.stem+'_publishes_after_Linux_capture_native_capture_and_exclusivity',capture<nativepos<admitpos<publish)
    cpu=A/'audit_completed_finetune_checkpoints_v12.py';raw=cpu.read_text()
    check('CPU12_original_CPU8_math_with_actual8_and_correct_filename','audit_completed_finetune_checkpoints_v8.py' in raw and '43ab48da0bd862399d13aed718ad87500a718e795e9742663d78ba97c958252f' in raw and 'versions == {8}' in raw and 'schema_version=12' in raw and 'completed_finetune_checkpoint_audit_v12_' in raw)
    for path,name,args0,kwargs0 in [(new,'validate_worker',({},'EEGMamba-lastblock-finetune'),{}),(M/'repair_runtime_owner_v5.py','enter',({},'tok'),{'condition':'EEGMamba-lastblock-finetune','repair_registration':{}}),(epath,'worker',(),{'gate_binding':{},'admission_binding':{},'token':'tok','mode':'mamba_linux'})]:
        ns={'require':require,'sys':types.SimpleNamespace(platform='linux'),'os':types.SimpleNamespace(getuid=lambda:0)};extract(path,[name],ns);reject(path.stem+'_root_model_still_rejected',lambda ns=ns,name=name,args0=args0,kwargs0=kwargs0:ns[name](*args0,**kwargs0))
    sys.path.insert(0,str(W))
    from tridep_ccml.eeg_all_candidates import storage as S,continuation_queue_v10 as Q10,deterministic_repair_execution_v5 as D,repair_observation_v5 as V,qualify_deterministic_repairs_v4 as Q4,qualify_deterministic_repairs_v5 as Q5,qualify_deterministic_repairs_v6 as Q6,qualify_deterministic_repairs_v7 as Q7
    cost=A/'cost_aggregation_v13.py'
    def normalize(value,**kwargs):return {'observed_counters':{k:value.get(k+'_returned') for k in ('forward','backward','optimizer_step')},'unknown_counter_fields':[]}
    cns={'require':require,'copy':copy,'S':S,'Q':Q7,'PREVIOUS_Q':Q4,'PREVIOUS_SUCCESS_Q':Q5,'PREVIOUS_IDENTITY_Q':Q6,'COUNTER_CATEGORY':'qualification','NESTED_CATEGORY':'nested','WORK':W};extract(cost,['repair_worker_record','category_totals','add_repair_ledgers'],cns)
    cresult={'cost_records':[],'inputs':[],'input_read_errors':[],'category_observations':{},'unknown_or_opaque_records':[]};cns['add_repair_ledgers'](cresult,normalize)
    rows=cresult['cost_records'];primary=[r for r in rows if r['aggregation_role']=='primary_observed_invocation'];check('Cost13_retains_only_two_actual_primary_attempts_once',len(primary)==2 and cresult['input_read_errors']==[] and len({r['artifact']['path'] for r in rows})==len(rows))
    failed6=OUT/'deterministic_repair_qualification_v6/EEGMamba-lastblock-finetune/1791780606091254700/controller.json';actual6=read(failed6);row6=next(r for r in rows if r['artifact']==desc(failed6))
    check('Qual6_zero_model_failed_controller_returncode1_preserved',actual6['admission']['source']==desc(old) and actual6['status']=='worker_returned' and actual6['returncode']==1 and row6['status']=='worker_returned' and row6['seconds_observed']==actual6['inclusive_controller_seconds'] and row6['aggregation_role']=='supplemental_nested_or_unmeasured' and not (failed6.parent/'attempt.json').exists() and not (failed6.parent/'worker_owner.json').exists())
    failed5=OUT/'deterministic_repair_qualification_v5/EEGMamba-lastblock-finetune/1791777298177356700/controller.json';row5=next(r for r in rows if r['artifact']==desc(failed5));check('Qual5_unknown_elapsed_still_unknown',row5['status']=='worker_starting' and row5['seconds_observed'] is None)
    failure4=next(r for r in primary if '/deterministic_repair_qualification_v4/' in r['artifact']['path']);success5=next(r for r in primary if '/deterministic_repair_qualification_v5/' in r['artifact']['path']);check('Qual4_failure_and_Qual5_success_retained',failure4['status']=='failed' and failure4['seconds_observed']==16.700772199998028 and success5['status']=='engineering_verified')
    future=copy.deepcopy(read(W/success5['artifact']['path']));future.update(source=desc(new),condition='EEGMamba-lastblock-finetune',primary_engineering_seconds=8.0)
    artifact={'path':(OUT/'deterministic_repair_qualification_v7/EEGMamba-lastblock-finetune/fixture/attempt.json').relative_to(W).as_posix(),'sha256':'fixture'};future_row=cns['repair_worker_record'](artifact,future,normalize);totals=cns['category_totals']([*rows,future_row]);check('all_four_qualification_generations_not_double_counted',totals['primary_invocations']==3 and totals['counter_totals']['backward']['known_observed_subtotal']==12 and totals['known_recorded_scope_seconds_subtotal']==failure4['seconds_observed']+success5['seconds_observed']+8.0 and totals['full_category_wall_or_gpu_busy_time'] is None and totals['nested_driver_times_added'] is False)
    reject('Cost13_no_relabelling_old_success_as_Qual7',lambda:cns['repair_worker_record'](success5['artifact'],future,normalize))
    calls=[];base=types.SimpleNamespace(buffered_observation=lambda*a:None,collect=lambda:(calls.append('inventory') or {'rules':[]}),load_v4=lambda:types.SimpleNamespace(normalized='n'));prior=types.SimpleNamespace(load_v5=lambda:base,augment=lambda result,**kw:(calls.append('cost9') or result));ns={'delegate':lambda:prior,'buffered_observation':lambda*a:None,'add_repair_ledgers':lambda result,n:(calls.append('repair') or result),'add_preserved_failure_closure':lambda result:(calls.append('closure') or result),'L':types.SimpleNamespace(verify_lost_f3_attempt='v',read_json='r',__file__='L3'),'S':types.SimpleNamespace(desc=lambda p:{'path':str(p),'sha256':'fixture'}),'Path':Path,'SOURCE':'cost13','DELEGATE':'cost9','V':types.SimpleNamespace(SOURCE='V5')};extract(cost,['collect'],ns);collection=ns['collect']();check('Cost13_inventory_and_three_loss_cost9_augmentation_once',calls==['inventory','cost9','repair','closure'] and collection['schema_version']==13)
    closure_path=OUT/'deterministic_repair_qualification_v5/failure_closures/pre_model_linux_permission_1791778566044163800/closure.json'
    fcns={'require':require,'S':S,'PREVIOUS_SUCCESS_Q':Q5,'FAILURE_CLOSURE':desc(closure_path)};extract(cost,['add_preserved_failure_closure'],fcns);fcns['add_preserved_failure_closure'](cresult)
    cproof=cresult['repair_qualification_failure_closures'][0];check('Cost13_premodel_closure_inputs_bound_without_extra_primary_cost',cproof['original_elapsed'] is None and cproof['extra_primary_counters_added'] is False and cproof['extra_primary_seconds_added'] is False and desc(closure_path) in cresult['inputs'] and len(cresult['cost_records'])==len(rows))
    report=OUT/'deterministic_repair_qualification_v5/CodeBrain-lastblock-finetune/1791777157876013500/report.json';proof=D.qualification(desc(report),'CodeBrain-lastblock-finetune');check('D5_true_old_CodeBrain_success_reused_with_original15',proof['successful_qualification_reused_without_new_model_work'] is True and proof['report']['sha256']=='89f0251f35539416ec7821ee959e502a70da51d91b24e3ca1e8748881bdb9052' and proof['observed_engineering']['source']==desc(M/'qualify_deterministic_repairs_v5.py') and len(proof['original_fit_ids'])==15)
    V.source_check();sources=Q10.sources();prior_review_path=A/'independent_linux_identity_reuse_chain_source_review_1791780532160912600.json';prior_review=read(prior_review_path)
    check('all139_inherit_original127_unchanged_and_previous_review',len(sources)==139 and len({r['path'] for r in sources})==139 and all(r in sources for r in prior_review['reviewed_sources']) and desc(prior_review_path) in sources)
    for source in sources:checked(source)
    check('actual_L2_root_observer_bytes_unchanged',desc(M/'linux_worker_identity_v2.py')['sha256']=='abe21d44b8bc663a6506e1d996c67db45212b4a6d5f4c42e4d6a8bc3cd0aec57')
    check('current_contract_exact139',contract.get('candidate_execution_sources',contract.get('source_chain'))==sources)
    check('current_Qual7_source_pin_matches',D.QUALIFIER_SHA==desc(new)['sha256'])
    post=A/'repair_postprocessing_gate_v5.py';ns={'S':S,'SOURCE':post};pin=next(n for n in tree(post).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets));exec(compile(ast.Module(body=[pin],type_ignores=[]),str(post),'exec'),ns);check('postgate5_pins_all_current_sources',all(desc(p)['sha256']==sha for p,sha in ns['PINS'].items()))
    check('no_Tensor_modules_imported',not any(name in sys.modules for name in ('torch','numpy')))
    cleanup_path=A/'qualification6_zero_model_cleanup_1791781276379.json';cleanup=read(cleanup_path);check('bound_actual_Qual6_cleanup_released_and_empty',cleanup['experimental_native_rows']==cleanup['gpu_contexts']==[] and cleanup['qualification6_owner_present'] is False)
    check('new_Qual7_and_Q10_not_yet_registered_or_run',not (OUT/'deterministic_repair_qualification_v7/EEGMamba-lastblock-finetune/permanent_intent.json').exists() and not (Q10.ROOT/'registration.json').exists())
    result={'schema':'independent_native_launcher_reuse_chain_source_review_v1','status':'new_repair_chain_sources_reviewed','passed':True,'reviewer':'/root/loss3_independent_review','reviewer_source':desc(SOURCE),'reviewed_sources':sources,'checks':checks,'current_contract':desc(contract_path),'prior127_review':desc(prior_review_path),'auxiliary_sources':[desc(post),desc(A/'review_actual_repair_engineering_v5.py')],'actual_WSL_tree_diagnosis':desc(diagnosis_path),'native_helper':desc(native),'actual_Qual6_failed_controller':desc(failed6),'actual_Qual6_failure_cleanup':desc(cleanup_path),'actual_Qual6_released_owner':desc(failed6.parent/'released_owner.json'),'actual_CodeBrain5_success_reused':proof['report'],'prior_Qual5_premodel_failure_closure':prior_review['actual_Mamba5_failure_closure'],'review_scope':['All three Linux controller paths independently capture exact Popen/direct-shim native identities after Linux capture and publish together; workers wait for and verify both identities.','Only one exact-command direct child is allowed; parent/birth/command/nonce changes, missing captured roles, ambiguous siblings and unknown experimental processes remain rejected.','Original qualifier numeric AST, ordinary model UID, immutable root observer, original matrix/roles/0-step qualification unchanged.','Actual original CodeBrain5 successful report is reused under its original source; Qual4/5/6 failures and unknown fields remain preserved.','CPU12 actual observer8 delegates unchanged CPU8; cost13 augments original cost9 exactly once and does not add nested controller elapsed to primary work.'],'model_calls':0,'GPU_queries':0,'actual_process_inventory_queries':0,'process_controls':0,'checkpoint_tensor_loads':0,'outer_test_probability_reads':0,'source_byte_hashing_performed':True,'official_weight_and_engineering_checkpoint_byte_hashing_performed':True,'byte_hash_counts_and_elapsed_unmeasured':None,'actual_Qual7_GPU_qualification_complete':False,'actual_formal_fits_completed':0,'new_registration_created':False,'all_plan_complete':False,'remaining_limitations':['Actual Qual7 Mamba engineering and original30 formal fits remain pending.','Full CPU12/E8 resource/cost/final acceptance and companion delivery remain pending.','Mamba backward accumulation order differs; original CUDA backward equivalence and speed improvement are not claimed.','EEGMamba external-pretraining participant overlap exclusion remains unknown.'],'captured_ns':time.time_ns()}
    require(sources==Q10.sources(),'Current source graph changed during review')
    target=A/('independent_native_launcher_reuse_chain_source_review_'+str(result['captured_ns'])+'.json')
    with target.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'report':desc(target),'passed':True,'reviewed_sources':len(sources),'checks':len(checks)}))
if __name__=='__main__':main()

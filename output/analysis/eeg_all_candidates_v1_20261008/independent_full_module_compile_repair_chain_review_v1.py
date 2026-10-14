"""Independent complete-module compilation/source-chain review; stdlib only."""
import ast,copy,hashlib,importlib.util,json,os,sys,time,types
from pathlib import Path
SOURCE=Path(__file__).resolve();A=SOURCE.parent;W=SOURCE.parents[3];M=W/'tridep_ccml/eeg_all_candidates';OUT=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008';sys.path.insert(0,str(W));checks={}
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);v=importlib.util.module_from_spec(s);s.loader.exec_module(v);return v
H=load(A/'independent_core_prefix_repair_chain_review_v1.py','_prior_source_reviewer_helpers')
desc,read,tree,node,dump,require=H.desc,H.read,H.tree,H.node,H.dump,H.require

def check(k,v):require(v,k);checks[k]=True
def reject(k,f):
 try:f()
 except (ValueError,KeyError,AssertionError):checks[k]=True
 else:raise AssertionError('Unsafe acceptance '+k)
def main():
 contract_path=A/'core_code_compat_chain_contracts_1791783695780664200.json';contract=read(contract_path)
 check('actual_new_contract108_passed',contract['passed'] is True and len(contract['checks'])==108)
 diagpath=A/'mamba_core_compile_scope_diagnosis_1791783596437499878.json';diag=read(diagpath)
 check('actual_Linux312_cached_matches_all18_full_compile_fields',diag['python'].startswith('3.12.3') and diag['comparisons'][0]['different_fields']==[] and len(diag['comparisons'][0]['equal_fields'])==18 and all(c['different_fields']==['co_code'] for c in diag['comparisons'][1:]) and diag['no_installed_module_executed'] is True and diag['model_constructions']==diag['tensor_operations']==diag['GPU_operations']==0)
 opcodepath=A/'mamba_core_compile_opcode_difference_1791783649364830171.json';opcode=read(opcodepath)
 G=load(M/'mamba_private_backward_graph_v3.py','_independent_G3');gp=M/'mamba_private_backward_graph_v3.py';oldgp=M/'mamba_private_backward_graph_v2.py'
 for n in ('clone','cell','replace_wrapped_leaf','rewrite_core_ast'):
  check('G3_'+n+'_AST_unchanged',dump(node(gp,n))==dump(node(oldgp,n)))
 def gate(p):
  return next(n for n in ast.walk(node(p,'private_core')) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require' and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='Loaded SSD core differs from pinned source function')
 check('loaded_source_all_original10_fields_strict_gate_unchanged',dump(gate(gp))==dump(gate(oldgp)))
 body=node(gp,'_code_for');check('full_module_compile_without_exec_or_eval',any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='compile' for n in ast.walk(body)) and not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('exec','eval') for n in ast.walk(tree(gp))))
 combined=OUT/'assets/eegmamba_linux_env/lib/python3.12/site-packages/mamba_ssm/ops/triton/ssd_combined.py';module=tree(combined);original=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name==G.CORE_KEY)
 code=G._code_for(copy.deepcopy(module),str(combined));sentinel=object();ns={'sentinel':sentinel};fn=types.FunctionType(code,ns,G.CORE_KEY);private=G.private_core(fn,{},lambda value,dim:value)
 check('full_source_fixture_passes_without_original_global_mutation',ns=={'sentinel':sentinel} and fn.__code__ is code and private.__globals__ is not ns and private.__code__.co_code!=code.co_code)
 for key,value in [('co_names',code.co_names+('unreviewed',)),('co_consts',code.co_consts+('unreviewed',)),('co_flags',code.co_flags^0x1000000)]:
  bad=types.FunctionType(code.replace(**{key:value}),{},G.CORE_KEY);reject('modified_loaded_'+key+'_rejected',lambda bad=bad:G.private_core(bad,{},lambda value,dim:value))
 reject('prechanged_loaded_core_rejected',lambda:G.private_core(private,{},lambda value,dim:value))
 reject('function_only_AST_no_longer_accepted',lambda:G._code_for(original,str(combined)))
 changed=copy.deepcopy(module);indices=[i for i,n in enumerate(changed.body) if isinstance(n,ast.FunctionDef) and n.name==G.CORE_KEY];changed.body[indices[0]]=G.rewrite_core_ast(changed.body[indices[0]])
 check('whole_module_other_nodes_preserved',all(dump(a)==dump(b) for i,(a,b) in enumerate(zip(module.body,changed.body)) if i!=indices[0]))
 check('FP32_C2_source_unchanged',desc(M/'mamba_cumsum_cpu_fp32_v2.py')['sha256']=='689eec011f11cdf4f54a4cf2bc2545ac115011dded729f4048bc84de8381cabe')
 for n in ('class_factory','bind_mixer','exact_finetune_parameter_names','adapt_raw','encode_windows'):
  check('M4_'+n+'_native_forward_route_unchanged',dump(node(M/'adapters/eegmamba_v4.py',n))==dump(node(M/'adapters/eegmamba_v3.py',n)))
 qold=M/'qualify_deterministic_repairs_v8.py';qnew=M/'qualify_deterministic_repairs_v9.py'
 check('qualification_calculation_AST_only_identity_version_changes',dump(node(qnew,'runtime_worker')).replace('qualify_deterministic_repairs_v9','qualify_deterministic_repairs_v8')==dump(node(qold,'runtime_worker')))
 check('private_builder_AST_only_Mamba_adapter34_route_changes',dump(node(qnew,'isolated_builder')).replace('.adapters.eegmamba_v4','.adapters.eegmamba_v3')==dump(node(qold,'isolated_builder')))
 newowner=copy.deepcopy(node(M/'repair_runtime_owner_v7.py','validate_native'))
 for n in ast.walk(newowner):
  if isinstance(n,ast.Compare) and isinstance(n.left,ast.Call) and isinstance(n.left.func,ast.Attribute) and n.left.func.attr=='get' and n.left.args and isinstance(n.left.args[0],ast.Constant) and n.left.args[0].value=='execution_version':
   require(len(n.comparators)==1 and isinstance(n.comparators[0],ast.Constant) and n.comparators[0].value==12,'exact newowner version');n.comparators[0].value=11
 check('formal_native_identity_guard_only_queue_version_changes',dump(newowner)==dump(node(M/'repair_runtime_owner_v6.py','validate_native')))
 from tridep_ccml.eeg_all_candidates import storage as S,continuation_queue_v12 as Q12,deterministic_repair_execution_v7 as D,repair_observation_v7 as V
 from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v4 as Q4,qualify_deterministic_repairs_v5 as Q5,qualify_deterministic_repairs_v6 as Q6,qualify_deterministic_repairs_v7 as Q7,qualify_deterministic_repairs_v8 as Q8,qualify_deterministic_repairs_v9 as Q9
 cb=OUT/'deterministic_repair_qualification_v5/CodeBrain-lastblock-finetune/1791777157876013500/report.json';proof=D.qualification(desc(cb),'CodeBrain-lastblock-finetune');check('actual_original_CodeBrain_qualification_reused_with_true_source',proof['successful_qualification_reused_without_new_model_work'] is True and proof['observed_engineering']['source']==desc(Q5.SOURCE) and len(proof['original_fit_ids'])==15)
 cost=A/'cost_aggregation_v15.py';cns={'require':require,'copy':copy,'S':S,'Q':Q9,'PREVIOUS_Q':Q4,'PREVIOUS_SUCCESS_Q':Q5,'PREVIOUS_IDENTITY_Q':Q6,'PREVIOUS_TREE_Q':Q7,'PREVIOUS_CORE_Q':Q8,'COUNTER_CATEGORY':'qualification','NESTED_CATEGORY':'nested','WORK':W}
 # Use the actual prior-generation import alias from source, never assume it.
 for n in tree(cost).body:
  if isinstance(n,ast.ImportFrom):
   for a in n.names:
    if a.name=='qualify_deterministic_repairs_v8':cns[a.asname or a.name]=Q8
 H.extract(cost,['repair_worker_record','category_totals','add_repair_ledgers'],cns)
 def normalize(v,**kw):return {'observed_counters':{k:v.get(k+'_returned') for k in ('forward','backward','optimizer_step')},'unknown_counter_fields':[]}
 result={'cost_records':[],'inputs':[],'input_read_errors':[],'category_observations':{},'unknown_or_opaque_records':[]};cns['add_repair_ledgers'](result,normalize);rows=result['cost_records'];primary=[r for r in rows if r['aggregation_role']=='primary_observed_invocation']
 check('four_actual_qualification_primary_invocations_once',len(primary)==4 and result['input_read_errors']==[] and len({r['artifact']['path'] for r in rows})==len(rows))
 f8=OUT/'deterministic_repair_qualification_v8/EEGMamba-lastblock-finetune/1791783245785244500/attempt.json';failure=read(f8);r8=next(r for r in primary if r['artifact']==desc(f8));controller=read(f8.parent/'controller.json');nested=next(r for r in rows if r['artifact']==desc(f8.parent/'controller.json'))
 check('actual_Qual8_failed_model_construction_cost_preserved',failure['source']==desc(Q8.SOURCE) and failure['status']=='failed' and failure['failure']=='Loaded SSD core differs from pinned source function' and all(failure[k+'_'+suffix]==0 for k in ('forward','backward','optimizer_step','encoder_forward','encoder_windows') for suffix in ('started','returned')) and failure['formal_fits']==0 and r8['seconds_observed']==6.125486071000001 and r8['unreturned_call_internal_work_unknown'] is True and 'adapter.load_encoder' in (f8.parent/'process.log').read_text())
 check('actual_Qual8_controller_nested23_seconds_excluded_from_primary',controller['returncode']==1 and nested['aggregation_role']=='supplemental_nested_or_unmeasured' and nested['seconds_observed']==23.894923999992898 and controller['controller_seconds_primary_cost'] is False)
 totals=cns['category_totals'](rows);check('four_generations_primary_sum_and_unknown_exact_total',totals['primary_invocations']==4 and totals['known_recorded_scope_seconds_subtotal']==sum(r['seconds_observed'] for r in primary) and totals['counter_totals']['backward']['known_observed_subtotal']==10 and totals['counter_totals']['optimizer_step']['known_observed_subtotal']==0 and totals['full_category_wall_or_gpu_busy_time'] is None and totals['nested_driver_times_added'] is False)
 false=copy.deepcopy(failure);false['source']=desc(Q9.SOURCE);reject('Qual8_failure_not_relabelled_Qual9',lambda:cns['repair_worker_record'](desc(f8),false,normalize))
 check('Cost15_collect_only_output_version_labels_changed',ast.unparse(node(cost,'collect')).replace('schema_version=15','schema_version=14').replace('actual10_versions_not_relabelled','actual9_versions_not_relabelled').replace('Actual observer10','Actual observer9').replace('nested Q12/','nested Q11/')==ast.unparse(node(A/'cost_aggregation_v14.py','collect')))
 collected=node(cost,'collect');calls=[n for n in ast.walk(collected) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='augment'];check('Cost9_augment_exactly_once',len(calls)==1 and isinstance(calls[0].func.value,ast.Name) and calls[0].func.value.id=='previous' and "cost_aggregation_v9.py" in cost.read_text() and '4d0863abf0a4dbfe30e8e2bd853ca7d7bb8242212fce65683be36d9cd70811e0' in cost.read_text())
 cpu=A/'audit_completed_finetune_checkpoints_v14.py';raw=cpu.read_text();check('CPU14_original_CPU8_math_actual10_filename',all(s in raw for s in ('audit_completed_finetune_checkpoints_v8.py','43ab48da0bd862399d13aed718ad87500a718e795e9742663d78ba97c958252f','versions == {10}','schema_version=14','completed_finetune_checkpoint_audit_v14_')))
 V.source_check();sources=Q12.sources();priorpath=A/'independent_core_prefix_repair_chain_source_review_1791783174737274500.json';prior=read(priorpath)
 check('exact166_inherit_all153_sources_and_prior_review',len(sources)==len({r['path'] for r in sources})==166 and all(r in sources for r in prior['reviewed_sources']) and desc(priorpath) in sources and sources==contract['source_chain'])
 for r in sources:require(desc(W/r['path'])==r,'Current source changed')
 check('all166_actual_bytes_bound',True)
 check('G3_M4_and_Qual9_pins_bound',D.QUALIFIER_SHA==desc(qnew)['sha256'] and desc(gp)['sha256'] in (M/'adapters/eegmamba_v4.py').read_text() and desc(M/'adapters/eegmamba_v4.py')['sha256'] in qnew.read_text())
 post=A/'repair_postprocessing_gate_v7.py';pns={'S':S,'SOURCE':post};pins=next(n for n in tree(post).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets));exec(compile(ast.Module(body=[pins],type_ignores=[]),str(post),'exec'),pns);check('all_auxiliary_current_pins',all(desc(p)['sha256']==sha for p,sha in pns['PINS'].items()))
 check('new_qualification_and_registration_unstarted',not (Q9.ROOT/'EEGMamba-lastblock-finetune/permanent_intent.json').exists() and not (Q12.ROOT/'registration.json').exists())
 check('no_Tensor_modules_imported',not any(x in sys.modules for x in ('torch','numpy')))
 preparations=[A/'mamba_complete_module_compile_stdlib_contract_1791783767937018090.json']
 require(preparations,'Actual Linux cached-code gate preparation pending');prep=read(preparations[-1]);require(prep.get('passed') is True and prep['tests_run']==9 and prep['errors']==prep['failures']==[] and prep['python'].startswith('3.12.3') and prep['actual_cache_sha256']==diag['cache']['sha256'],'Actual Linux cached-code gate not passed');require(all(desc(W/r['path'])==r for r in prep['reviewed_sources']),'Cached gate current source mismatch');require(desc(W/prep['test_source']['path'])==prep['test_source'],'Cached gate test source mismatch')
 result={'schema':'independent_full_module_compile_repair_chain_source_review_v1','status':'new_repair_chain_sources_reviewed','passed':True,'reviewer':'/root/loss3_independent_review','reviewer_source':desc(SOURCE),'reviewed_sources':sources,'checks':checks,'prior153_review':desc(priorpath),'current_contract':desc(contract_path),'actual_Linux312_compile_diagnosis':desc(diagpath),'actual_opcode_diagnosis':desc(opcodepath),'actual_Linux_cached_code_gate_preparation':desc(preparations[-1]),'actual_Qual8_failed_attempt':desc(f8),'actual_Qual8_nested_controller':desc(f8.parent/'controller.json'),'actual_Qual8_released_owner':desc(f8.parent/'released_owner.json'),'actual_CodeBrain_Qual5_reused':proof['report'],'auxiliary_sources':[desc(post),desc(A/'review_actual_repair_engineering_v7.py')],'scope':['Actual Linux3.12 cached core equals full-module source compilation across18 fields; single-function compile differs only in co_code due import/global compiler context.','G3 only changes compilation scope, retaining source hash and all10 strict loaded-code comparisons; full module is compiled but never executed.','C2 arithmetic, exact single-call rewrite and original forward/private wiring remain unchanged.','Real Qual8 construction failure had0 observed forward/backward/step, but model construction did occur;6.125486071000001 engineering seconds remains primary and23.894923999992898 controller seconds supplemental.','Original CodeBrain qualification reuse, all prior153 bytes, actual CPU14 observer10/source and Cost15 originalCost9 augmentation/failed scopes preserved.'],'model_calls':0,'GPU_operations':0,'actual_process_inventory_queries':0,'process_controls':0,'checkpoint_tensor_loads':0,'outer_test_probability_reads':0,'source_byte_hashing_performed':True,'official_weight_and_engineering_checkpoint_byte_hashing_performed':True,'byte_hash_count_time_unknown':None,'new_qualification_GPU_passed':False,'new_formal_fits_completed':0,'new_registration_created':False,'all_plan_complete':False,'remaining_limitations':['Actual Qual9 Mamba GPU engineering remains pending; compiled code-object tests do not establish GPU success.','Original30 formal fits and actual complete CPU/evaluation/resource/cost/acceptance remain pending.','Backward accumulation order change remains explicit; native CUDA gradient equivalence/speedup not claimed.','EEGMamba external-pretraining participant-overlap exclusion remains unknown.'],'captured_ns':time.time_ns()}
 require(sources==Q12.sources(),'Sources changed during review');target=A/('independent_full_module_compile_repair_chain_source_review_'+str(result['captured_ns'])+'.json')
 with target.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'report':desc(target),'passed':True,'checks':len(checks),'sources':len(sources)}))
if __name__=='__main__':main()

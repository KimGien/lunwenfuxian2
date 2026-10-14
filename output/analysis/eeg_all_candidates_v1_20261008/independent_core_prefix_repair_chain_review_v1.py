"""Independent review of the source-locked reverse-prefix repair; no model execution."""
from __future__ import annotations
import ast,copy,hashlib,importlib.util,json,os,struct,sys,time,types
from pathlib import Path
SOURCE=Path(__file__).resolve();A=SOURCE.parent;W=SOURCE.parents[3];M=W/'tridep_ccml/eeg_all_candidates';OUT=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008';checks={}
sys.path.insert(0,str(W))
def require(ok,msg):
 if not ok:raise ValueError(msg)
def check(label,ok):require(ok,label);checks[label]=True
def reject(label,fn):
 try:fn()
 except (ValueError,KeyError,AssertionError):checks[label]=True
 else:raise AssertionError('Unsafe acceptance: '+label)
def desc(p):
 p=Path(p).resolve();return {'path':p.relative_to(W).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def tree(p):return ast.parse(Path(p).read_text(encoding='utf-8-sig'))
def node(p,name):return next(n for n in tree(p).body if isinstance(n,ast.FunctionDef) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
def extract(p,names,ns):
 nodes=[copy.deepcopy(node(p,n)) for n in names]
 for n in nodes:n.decorator_list=[]
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),ns);return ns

def main():
 contract_path=A/'core_prefix_repair_chain_contracts_1791782568956296300.json';contract=read(contract_path)
 numeric_path=A/'mamba_core_reverse_cumsum_stdlib_preparation_1791782636336748000.json';numeric=read(numeric_path)
 check('author_contracts_are_real_passed103_and12',contract['passed'] is True and len(contract['checks'])==103 and numeric['passed'] is True and numeric['tests']==12 and numeric['failures']==numeric['errors']==0)
 G=load(M/'mamba_private_backward_graph_v2.py','_independent_G2_stdlib')
 combined=OUT/'assets/eegmamba_linux_env/lib/python3.12/site-packages/mamba_ssm/ops/triton/ssd_combined.py'
 original=node(combined,G.CORE_KEY);changed=G.rewrite_core_ast(original)
 target=dump(ast.parse(G.TARGET_EXPRESSION,mode='eval').body)
 calls=[n for n in ast.walk(original) if isinstance(n,ast.Call) and dump(n)==target]
 replacement=[n for n in ast.walk(changed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==G.PRIVATE_CUMSUM]
 check('actual_pinned_core_has_exactly_one_reviewed_method_prefix',desc(combined)['sha256']==G.COMBINED_SHA and len(calls)==len(replacement)==1)
 check('replacement_keeps_exact_flipped_operand_and_dim',dump(replacement[0].args[0])==dump(calls[0].func.value) and [(k.arg,ast.literal_eval(k.value)) for k in replacement[0].keywords]==[('dim',-1)])
 class Undo(ast.NodeTransformer):
  def visit_Call(self,n):
   if isinstance(n.func,ast.Name) and n.func.id==G.PRIVATE_CUMSUM:return copy.deepcopy(calls[0])
   return self.generic_visit(n)
 restored=Undo().visit(copy.deepcopy(changed))
 check('entire_core_AST_other_than_one_call_exact',dump(restored)==dump(original))
 bad=copy.deepcopy(original)
 class WrongDim(ast.NodeTransformer):
  def visit_Call(self,n):
   if dump(n)==target:n.keywords[0].value=ast.Constant(value=0)
   return self.generic_visit(n)
 reject('changed_target_dim_rejected',lambda:G.rewrite_core_ast(WrongDim().visit(bad)))
 bad=copy.deepcopy(original);bad.body.append(ast.Expr(value=copy.deepcopy(calls[0])))
 reject('duplicate_matching_call_rejected',lambda:G.rewrite_core_ast(bad))
 bad=copy.deepcopy(original);bad.body.append(ast.parse('x.cumsum(dim=-1)').body[0])
 reject('additional_unreviewed_method_call_rejected',lambda:G.rewrite_core_ast(bad))
 code=G._code_for(original,str(combined));globals0={'immutable_sentinel':object()};fn=types.FunctionType(code,globals0,G.CORE_KEY);before=dict(globals0);helper=lambda value,dim:value
 private=G.private_core(fn,{'known_helper':object()},helper)
 check('loaded_core_verified_private_copy_leaves_original_globals_and_code',globals0==before and fn.__code__ is code and private.__code__.co_code!=code.co_code and private.__globals__[G.PRIVATE_CUMSUM] is helper)
 foreign=types.FunctionType(G._code_for(changed,str(combined)),{},G.CORE_KEY)
 reject('already_changed_loaded_core_rejected',lambda:G.private_core(foreign,{},helper))
 reject('private_helper_name_collision_rejected',lambda:G.private_core(fn,{G.PRIVATE_CUMSUM:helper},helper))
 check('private_graph_original_forward_and_change_disclosure',"new_class.forward is original_class.forward" in (M/'mamba_private_backward_graph_v2.py').read_text() and "'original_function_bytecode_preserved':False" in (M/'mamba_private_backward_graph_v2.py').read_text())
 C=M/'mamba_cumsum_cpu_fp32_v2.py';oldC=M/'mamba_cumsum_cpu_fp32_v1.py';newnode=node(C,'cumsum');oldnode=node(oldC,'cumsum')
 check('existing_same_storage_out_path_AST_retained_exactly',dump(ast.Module(body=newnode.body[1:],type_ignores=[]))==dump(ast.Module(body=oldnode.body,type_ignores=[])))
 # Two-dimensional fake FP32 values exercise independent rows and out-of-place storage.
 def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
 class Grid:
  def __init__(self,values,shape,device='cuda:fake',dtype='f32',grad=False):self.values=list(values);self.shape=shape;self.ndim=len(shape);self.device=device;self.dtype=dtype;self.requires_grad=grad
  def numel(self):return len(self.values)
  def detach(self):return self
  def to(self,*,device):return Grid(self.values,self.shape,device,self.dtype)
  def contiguous(self):return self
  def clone(self):return Grid(self.values,self.shape,self.device,self.dtype)
  def __getitem__(self,index):
   i=index[-1];width=self.shape[-1];indices=list(range(i,len(self.values),width));parent=self
   class Column(Grid):
    def copy_(self,other):
     for k,j in enumerate(indices):parent.values[j]=other.values[k]
     return self
   return Column([self.values[j] for j in indices],self.shape[:-1],self.device,self.dtype)
  def __add__(self,other):return Grid([f32(a+b) for a,b in zip(self.values,other.values)],self.shape,self.device,self.dtype)
 T=types.SimpleNamespace(Tensor=Grid,float32='f32',empty_like=lambda x:Grid([0.]*x.numel(),x.shape,x.device),zeros_like=lambda x:Grid([0.]*x.numel(),x.shape,x.device))
 ns={'torch':T};extract(C,['require','cumsum'],ns);scan=ns['cumsum']
 value=Grid([16777216.,1.,-16777216.,2.,3.,4.],(2,3));before=value.values[:];result=scan(value,-1)
 check('new_result_independent_rows_FP32_rounding_shape_device_and_no_input_write',result.values==[16777216.,16777216.,0.,2.,5.,9.] and value.values==before and result.values is not value.values and (result.shape,result.device,result.dtype)==(value.shape,value.device,value.dtype))
 for label,kwargs,dim in [('float64',{'dtype':'f64'},-1),('higher_order',{'grad':True},-1),('nonlast',{},0),('bool_dim',{},True)]:
  reject('new_result_'+label+'_rejected',lambda kwargs=kwargs,dim=dim:scan(Grid([1.,2.],(1,2),**kwargs),dim))
 reject('promoted_dtype_argument_rejected',lambda:scan(value,-1,dtype='f64'))
 reject('empty_result_scan_rejected',lambda:scan(Grid([],(2,0)),-1))
 for name in ('class_factory','bind_mixer','exact_finetune_parameter_names','adapt_raw','encode_windows'):
  check('adapter_'+name+'_unchanged',dump(node(M/'adapters/eegmamba_v3.py',name))==dump(node(M/'adapters/eegmamba_v2.py',name)))
 raw=(M/'adapters/eegmamba_v3.py').read_text()
 check('adapter_exact_new_helper_pins_and_declared_changed_accumulation',all(desc(p)['sha256'] in raw for p in (C,M/'mamba_private_backward_graph_v2.py')) and "'original_core_bytecode_preserved':False" in raw and "'native_CUDA_numerical_equivalence_established':False" in raw and "'performance_improvement_claimed':False" in raw)
 # Entire tested worker retains its role/repetition/zero-step calculations.
 qold=M/'qualify_deterministic_repairs_v7.py';qnew=M/'qualify_deterministic_repairs_v8.py'
 check('qualifier_runtime_worker_entire_AST_unchanged',dump(node(qold,'runtime_worker'))==dump(node(qnew,'runtime_worker')).replace('qualify_deterministic_repairs_v8','qualify_deterministic_repairs_v7'))
 def normalized_builder(path,isnew):
  n=copy.deepcopy(node(path,'isolated_builder'));mapping=next(x.value for x in n.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mapping' for t in x.targets));values=ast.literal_eval(mapping)
  require(values=={'.adapters.codebrain_v2':'.adapters.codebrain_v3','.adapters.eegmamba':'.adapters.eegmamba_v3' if isnew else '.adapters.eegmamba_v2'},'exact adapter routing')
  if isnew:
   for k,v in zip(mapping.keys,mapping.values):
    if k.value=='.adapters.eegmamba':v.value='.adapters.eegmamba_v2'
  return dump(n)
 check('qualifier_builder_only_Mamba_adapter_route_changes',normalized_builder(qnew,True)==normalized_builder(qold,False))
 substitutions={'qualify_deterministic_repairs_v8':'qualify_deterministic_repairs_v7','continuation_queue_v11':'continuation_queue_v10','repair_runtime_owner_v6':'repair_runtime_owner_v5','repair_owned_worker_v6':'repair_owned_worker_v5','evaluate_v9':'evaluate_v8','finetune_v9':'finetune_v8','audit_completed_finetune_checkpoints_v13':'audit_completed_finetune_checkpoints_v12','timer_is_nested_supplemental_not_F9_primary_cost':'timer_is_nested_supplemental_not_F8_primary_cost','Actual E9':'Actual E8','Actual98-source recovery binding required':'Actual88-source recovery binding required'}
 def normalized_ast(path,name):
  n=copy.deepcopy(node(path,name))
  class Normalize(ast.NodeTransformer):
   def visit_Constant(self,x):
    if isinstance(x.value,str):
     for a,b in substitutions.items():x.value=x.value.replace(a,b)
    return x
  return dump(Normalize().visit(n))
 for path,prior,names in [('qualify_deterministic_repairs_v8.py','qualify_deterministic_repairs_v7.py',('validate_worker','admit_host','source_check')),('repair_owned_worker_v6.py','repair_owned_worker_v5.py',('run',)),('evaluate_v9.py','evaluate_v8.py',('admit_worker','worker'))]:
  for name in names:check(path+'_'+name+'_control_semantics_preserved',normalized_ast(M/path,name)==dump(node(M/prior,name)))
 from tridep_ccml.eeg_all_candidates import storage as S,continuation_queue_v11 as Q11,deterministic_repair_execution_v6 as D,repair_observation_v6 as V
 from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v4 as Q4,qualify_deterministic_repairs_v5 as Q5,qualify_deterministic_repairs_v6 as Q6,qualify_deterministic_repairs_v7 as Q7,qualify_deterministic_repairs_v8 as Q8
 report=OUT/'deterministic_repair_qualification_v5/CodeBrain-lastblock-finetune/1791777157876013500/report.json';proof=D.qualification(desc(report),'CodeBrain-lastblock-finetune')
 check('actual_CodeBrain_Qual5_reused_under_its_true_source_and_original15',proof['successful_qualification_reused_without_new_model_work'] is True and proof['report']['sha256']=='89f0251f35539416ec7821ee959e502a70da51d91b24e3ca1e8748881bdb9052' and proof['observed_engineering']['source']==desc(Q5.SOURCE) and len(proof['original_fit_ids'])==15)
 cost=A/'cost_aggregation_v14.py'
 def normalize(v,**kw):return {'observed_counters':{k:v.get(k+'_returned') for k in ('forward','backward','optimizer_step')},'unknown_counter_fields':[]}
 ns={'require':require,'copy':copy,'S':S,'Q':Q8,'PREVIOUS_Q':Q4,'PREVIOUS_SUCCESS_Q':Q5,'PREVIOUS_IDENTITY_Q':Q6,'PREVIOUS_TREE_Q':Q7,'COUNTER_CATEGORY':'qualification','NESTED_CATEGORY':'nested','WORK':W};extract(cost,['repair_worker_record','category_totals','add_repair_ledgers'],ns)
 result={'cost_records':[],'inputs':[],'input_read_errors':[],'category_observations':{},'unknown_or_opaque_records':[]};ns['add_repair_ledgers'](result,normalize);rows=result['cost_records'];primary=[r for r in rows if r['aggregation_role']=='primary_observed_invocation']
 check('only_three_actual_primary_qualifier_attempts_once',len(primary)==3 and result['input_read_errors']==[] and len({x['artifact']['path'] for x in rows})==len(rows))
 failed=OUT/'deterministic_repair_qualification_v7/EEGMamba-lastblock-finetune/1791782206453969000/attempt.json';failure=read(failed);record=next(r for r in primary if r['artifact']==desc(failed));controller=read(failed.parent/'controller.json')
 check('actual_Qual7_failure_count_duration_unknown_tail_preserved',failure['status']=='failed' and failure['source']==desc(Q7.SOURCE) and [failure[k] for k in ('forward_started','forward_returned','backward_started','backward_returned','encoder_windows_returned','optimizer_step_returned','formal_fits')]==[5,5,3,2,40,0,0] and failure['unreturned_call_internal_work_unknown'] is True and record['seconds_observed']==48.540251404 and record['unreturned_call_internal_work_unknown'] is True)
 nested=next(r for r in rows if r['artifact']==desc(failed.parent/'controller.json'));check('actual_Qual7_controller69_seconds_nested_not_added',nested['aggregation_role']=='supplemental_nested_or_unmeasured' and nested['seconds_observed']==controller['inclusive_controller_seconds']==69.25431119999848)
 totals=ns['category_totals'](rows);check('all_actual_qualification_primary_costs_counted_once_not_nested',totals['primary_invocations']==3 and totals['counter_totals']['backward']['known_observed_subtotal']==10 and totals['counter_totals']['optimizer_step']['known_observed_subtotal']==0 and totals['known_recorded_scope_seconds_subtotal']==sum(r['seconds_observed'] for r in primary) and totals['full_category_wall_or_gpu_busy_time'] is None and totals['nested_driver_times_added'] is False)
 wrong=copy.deepcopy(failure);wrong['source']=desc(Q8.SOURCE);reject('old_failure_cannot_be_relabelled_as_new_qualification',lambda:ns['repair_worker_record'](desc(failed),wrong,normalize))
 calls=[];base=types.SimpleNamespace(buffered_observation=lambda*a:None,collect=lambda:(calls.append('inventory') or {'rules':[]}),load_v4=lambda:types.SimpleNamespace(normalized='n'));prior=types.SimpleNamespace(load_v5=lambda:base,augment=lambda result,**kw:(calls.append('cost9') or result));cns={'delegate':lambda:prior,'buffered_observation':lambda*a:None,'add_repair_ledgers':lambda r,n:(calls.append('repair') or r),'add_preserved_failure_closure':lambda r:(calls.append('closure') or r),'L':types.SimpleNamespace(verify_lost_f3_attempt='v',read_json='r',__file__='L3'),'S':types.SimpleNamespace(desc=lambda p:{'path':str(p),'sha256':'fixture'}),'Path':Path,'SOURCE':'cost14','DELEGATE':'cost9','V':types.SimpleNamespace(SOURCE='V6')};extract(cost,['collect'],cns);r=cns['collect']();check('cost9_augmentation_exactly_once_after_original_inventory',calls==['inventory','cost9','repair','closure'] and r['schema_version']==14)
 closure=OUT/'deterministic_repair_qualification_v5/failure_closures/pre_model_linux_permission_1791778566044163800/closure.json';cns={'require':require,'S':S,'PREVIOUS_SUCCESS_Q':Q5,'FAILURE_CLOSURE':desc(closure)};extract(cost,['add_preserved_failure_closure'],cns);count=len(rows);cns['add_preserved_failure_closure'](result);check('zero_model_permission_loss_remains_unknown_and_inputs_only',len(result['cost_records'])==count and result['repair_qualification_failure_closures'][0]['original_elapsed'] is None and result['repair_qualification_failure_closures'][0]['extra_primary_seconds_added'] is False)
 cpu=A/'audit_completed_finetune_checkpoints_v13.py';raw=cpu.read_text();check('CPU13_keeps_CPU8_math_and_actual9_source_labels',all(x in raw for x in ('audit_completed_finetune_checkpoints_v8.py','43ab48da0bd862399d13aed718ad87500a718e795e9742663d78ba97c958252f','versions == {9}','schema_version=13','completed_finetune_checkpoint_audit_v13_')))
 V.source_check();sources=Q11.sources();previous_path=A/'independent_native_launcher_reuse_chain_source_review_1791782167310882100.json';previous=read(previous_path)
 check('exact153_unique_inherit_every_prior139_and_review',len(sources)==153 and len({x['path'] for x in sources})==153 and all(x in sources for x in previous['reviewed_sources']) and desc(previous_path) in sources)
 check('source_list_matches_current_contract',sources==contract.get('source_chain',contract.get('candidate_execution_sources')))
 for source in sources:require(desc(W/source['path'])==source,'source bytes changed')
 check('all153_current_bytes_match',True)
 check('unchanged_root_observer_and_native_launcher_guards',desc(M/'linux_worker_identity_v2.py')['sha256']=='abe21d44b8bc663a6506e1d996c67db45212b4a6d5f4c42e4d6a8bc3cd0aec57' and desc(M/'native_launcher_tree_v1.py')['sha256']=='09ad696871414baa47e9505ab450ca5ca7e5d2c62599c34dd0eaf5a2dfce8bea')
 check('new_qualifier_pin_current',D.QUALIFIER_SHA==desc(qnew)['sha256'])
 post=A/'repair_postprocessing_gate_v6.py';pns={'S':S,'SOURCE':post};pin=next(n for n in tree(post).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets));exec(compile(ast.Module(body=[pin],type_ignores=[]),str(post),'exec'),pns);check('all_postprocessing_current_pins',all(desc(p)['sha256']==sha for p,sha in pns['PINS'].items()))
 check('new_qualification_and_queue_unstarted',not (Q8.ROOT/'EEGMamba-lastblock-finetune/permanent_intent.json').exists() and not (Q11.ROOT/'registration.json').exists())
 check('no_torch_or_numpy_modules_imported',not any(x in sys.modules for x in ('torch','numpy')))
 result={'schema':'independent_core_prefix_repair_chain_source_review_v1','status':'new_repair_chain_sources_reviewed','passed':True,'reviewer':'/root/loss3_independent_review','reviewer_source':desc(SOURCE),'reviewed_sources':sources,'checks':checks,'current_contract':desc(contract_path),'numeric_preparation':desc(numeric_path),'prior139_review':desc(previous_path),'actual_CodeBrain_Qual5_success_reused':proof['report'],'actual_Qual7_failed_attempt':desc(failed),'actual_Qual7_nested_controller':desc(failed.parent/'controller.json'),'actual_Qual7_released_owner':desc(failed.parent/'released_owner.json'),'actual_Qual5_premodel_failure_closure':desc(closure),'new_auxiliary_sources':[desc(post),desc(A/'review_actual_repair_engineering_v6.py')],'scope':['The pinned installed backward core changes one Tensor.cumsum call to a private FP32 left-to-right CPU scan; all remaining core AST is exact.','The old same-storage out= path remains exact; the new-result path preserves shape/device, does not overwrite the input, and rejects higher-order, non-FP32 and non-last-axis inputs.','Native forward and instance-only private class/global wiring remain unchanged; backward accumulation order change and unavailable native CUDA gradient equivalence remain explicit.','Original CodeBrain Qual5 actual success is reused under its true source; the new Mamba adapter is the only builder-route change.','Prior Qual4/5/6/7 failure/source/counters/unknown fields are retained; actual Qual7 primary engineering cost is counted once and controller elapsed remains supplemental.','Current153 sources include all previous139 unchanged, CPU13 delegates original CPU8 math, and cost14 invokes original cost9 augmentation once.'],'process_controls':0,'actual_process_inventory_queries':0,'GPU_operations':0,'model_calls':0,'checkpoint_tensor_loads':0,'outer_test_probability_reads':0,'source_byte_hashing_performed':True,'official_weight_and_engineering_checkpoint_byte_hashing_performed':True,'byte_hash_count_time_unknown':None,'actual_Qual8_GPU_qualification_complete':False,'new_formal_fits_completed':0,'new_registration_created':False,'all_plan_complete':False,'nonblocking_observations':['Qual8 source_check error wording says Actual98 but the executable condition remains exactly len(sources)==88; old88 validation is unchanged.'],'remaining_limitations':['Actual Qual8 full Mamba GPU qualification is still pending; AST/fake tests do not establish GPU success.','The original30 formal fits and actual full CPU13/E9/resource/cost/final acceptance are still pending.','Native CUDA backward gradient equivalence and a speedup are not claimed.','EEGMamba external pretraining participant-overlap exclusion remains unknown.'],'captured_ns':time.time_ns()}
 require(sources==Q11.sources(),'source graph changed during review')
 path=A/('independent_core_prefix_repair_chain_source_review_'+str(result['captured_ns'])+'.json')
 with path.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
 print(json.dumps({'artifact':desc(path),'passed':True,'checks':len(checks),'sources':len(sources)}))
if __name__=='__main__':main()

"""Independent narrow review: frozen Mamba repeat diagnostic, no model/process operations."""
import ast,copy,hashlib,importlib.util,json,os,sys,time,types
from pathlib import Path
SOURCE=Path(__file__).resolve();A=SOURCE.parent;W=SOURCE.parents[3];M=W/'tridep_ccml/eeg_all_candidates';OUT=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008';sys.path.insert(0,str(W));checks={}
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);v=importlib.util.module_from_spec(s);s.loader.exec_module(v);return v
H=load(A/'independent_core_prefix_repair_chain_review_v1.py','_previous_review_helpers');desc,read,node,tree,dump,require=H.desc,H.read,H.node,H.tree,H.dump,H.require

def check(k,v):require(v,k);checks[k]=True
def reject(k,f):
 try:f()
 except (ValueError,KeyError,AssertionError):checks[k]=True
 else:raise AssertionError('Unsafe acceptance '+k)
def main():
 p=M/'diagnose_mamba_repeat_v1.py';old=M/'qualify_deterministic_repairs_v9.py'
 check('exact_frozen_diagnostic_and_Q9_sources',desc(p)['sha256']=='008981ad5948bfc885772ed0aceda46b8ca00ffc332797a5b0defc2aa874775c' and desc(old)['sha256']=='0c66d96313760334802e86da99771e14cf468f68f2908aa62f783ab4fa74f559')
 check('isolated_builder_entire_math_AST_exact',dump(node(p,'isolated_builder'))==dump(node(old,'isolated_builder')))
 class Normalize(ast.NodeTransformer):
  def visit_Constant(self,n):
   if isinstance(n.value,str):n.value=n.value.replace('diagnose_mamba_repeat_v1','qualify_deterministic_repairs_v9').replace('diagnostic_complete_no_formal_qualification','engineering_verified')
   return n
  def visit_Expr(self,n):
   if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='collect_repeat_diagnostic':return None
   return self.generic_visit(n)
  def visit_Call(self,n):
   n=self.generic_visit(n)
   if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='record' and n.func.attr=='update' and any(k.arg=='status' and isinstance(k.value,ast.Constant) and k.value.value=='engineering_verified' for k in n.keywords):
    n.keywords=[k for k in n.keywords if k.arg not in ('diagnostic_only','eligible_for_formal_registration')]
    for k in n.keywords:
     if k.arg=='GPU_engineering_checks_passed':require(k.value.value is False,'Only diagnostic nonqualification flag');k.value=ast.Constant(value=True)
   return n
 for name in ('runtime_worker','validate_worker','run_controller','own_host'):
  check(name+'_unchanged_except_diagnostic_metadata_hook_and_module_identity',dump(Normalize().visit(copy.deepcopy(node(p,name))))==dump(node(old,name)))
 check('strict_equality_gate_retained_exactly',any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require' and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='Repeated strict GPU loss/gradients/RNG differ' for n in ast.walk(node(p,'runtime_worker'))))
 code=p.read_text();check('success_never_formal_qualification',"status='diagnostic_complete_no_formal_qualification'" in code and 'GPU_engineering_checks_passed=False' in code and 'eligible_for_formal_registration=False' in code)
 check('permanent_intent_only_at_controller_admission',"permanent_intent.json" in ast.unparse(node(p,'safe_before_model')) and "permanent_intent.json" not in ast.unparse(node(p,'validate_worker')) and "safe_before_model(condition)" in ast.unparse(node(p,'run_controller')))
 check('all_definitions_precede_main_guard',isinstance(tree(p).body[-1],ast.If) and 'check_codebrain_terminal' in [n.name for n in tree(p).body[:-1] if isinstance(n,ast.FunctionDef)])
 # Only fake CPU captures: no Tensor runtime imported or constructed.
 class Vec:
  def __init__(self,values,device='cpu'):self.values=list(values);self.device=types.SimpleNamespace(type=device);self.shape=(len(values),);self.dtype='float32'
  def __sub__(self,other):return Vec([a-b for a,b in zip(self.values,other.values)])
  def abs(self):return Vec([abs(v) for v in self.values])
  def __ne__(self,other):return Vec([a!=b for a,b in zip(self.values,other.values)])
  def sum(self):return Vec([sum(self.values)])
  def max(self):return Vec([max(self.values)])
  def item(self):require(len(self.values)==1,'one fake item');return self.values[0]
  def numel(self):return len(self.values)
 torch=types.SimpleNamespace(equal=lambda a,b:a.values==b.values);F=types.SimpleNamespace(rng_equal=lambda a,b:a==b);saved=[]
 ns={'require':require,'desc':lambda p:{'path':'diagnostic','sha256':'fixture'},'SOURCE':p,'save':lambda p,v:saved.append((p,copy.deepcopy(v)))};H.extract(p,['collect_repeat_diagnostic'],ns);collect=ns['collect_repeat_diagnostic']
 a={'loss':Vec([1.]),'gradients':{'g':Vec([1.,2.])},'rng':{'cpu':'A'}};b={'loss':Vec([1.25]),'gradients':{'g':Vec([1.,2.5])},'rng':{'cpu':'B'}};record={'forward_returned':6,'backward_returned':4,'optimizer_step_returned':0,'formal_fits':0};collect(Path('fixture'),record,11,a,b,F,torch);row=record['repetition_diagnostics'][0]
 check('fake_loss_rng_gradient_mismatch_details_are_separate',row['loss_bitwise_equal'] is False and row['rng_exact_equal'] is False and row['all_gradients_bitwise_equal'] is False and row['different_gradients']==[{'parameter':'g','shape':[2],'dtype':'float32','unequal_elements':1,'elements':2,'max_abs_difference':0.5,'max_abs_left':2.,'max_abs_right':2.5}])
 check('existing_captures_and_model_call_counters_not_modified',a['gradients']['g'].values==[1.,2.] and b['gradients']['g'].values==[1.,2.5] and [record[k] for k in ('forward_returned','backward_returned','optimizer_step_returned','formal_fits')]==[6,4,0,0] and row['extra_GPU_model_calls']==0 and row['eligible_for_formal_registration'] is False and [x[0].name for x in saved]==['repeat_diagnostic.json','attempt.json'])
 cuda=copy.deepcopy(b);cuda['gradients']['g'].device.type='cuda';reject('diagnostic_rejects_nonCPU_gradient_capture',lambda:collect(Path('fixture'),{},11,a,cuda,F,torch))
 same={'loss':a['loss'],'gradients':{'g':a['gradients']['g'],'extra':Vec([1.])},'rng':a['rng']};r={};collect(Path('fixture'),r,11,a,same,F,torch);check('gradient_key_mismatch_reported_without_false_equal',r['repetition_diagnostics'][0]['gradient_keys_equal'] is False and r['repetition_diagnostics'][0]['all_gradients_bitwise_equal'] is False)
 ns={'require':require,'CONDITIONS':('CodeBrain-lastblock-finetune','EEGMamba-lastblock-finetune'),'ROOT':OUT/'nonexistent_review_only_fixture'};H.extract(p,['safe_before_model'],ns);reject('successful_CodeBrain_never_reprobed',lambda:ns['safe_before_model']('CodeBrain-lastblock-finetune'))
 from tridep_ccml.eeg_all_candidates import diagnose_mamba_repeat_v1 as D
 state=D.check_codebrain_terminal();q=OUT/'continuation_queue_v13';reg=read(q/'registration.json');cpu=read(W/state['partial_CPU_audit']['path']);child=read(W/state['outcomes']['CodeBrain-lastblock-finetune']['path'])
 check('actual_CB_natural_terminal_exit0_partial_CPU15_binding',state['status']=='codebrain15_complete_mamba15_blocked_no_evaluation' and child['condition']=='CodeBrain-lastblock-finetune' and child['source']==desc(qpath:=M/'continuation_queue_v13.py') and child['returncode']==0 and cpu['verified_completed_finetune_fits']==120 and cpu['checkpoint_loads']==360 and cpu['errors']==cpu['deferred_completed_fits']==[] and len(cpu['pending_fits'])==15 and not (q/'owner.json').exists())
 for binding in reg['sources']:require(desc(W/binding['path'])==binding,'Frozen175 source changed')
 check('all175_previous_source_bytes_unchanged',len(reg['sources'])==len({x['path'] for x in reg['sources']})==175)
 check('all_diagnostic_adapter_pins_current',all(desc(W/rel)['sha256']==sha for rel,sha in D.PINS.items()))
 oldfailure=OUT/'deterministic_repair_qualification_v9/EEGMamba-lastblock-finetune/1791783935769696400/attempt.json';failed=read(oldfailure);check('original_failed_Qual9_retained',desc(oldfailure)['sha256']=='77038a96f40f6fc0ecb6fa418b08a64b5a038cc67a7e790c0ab4723349d47325' and failed['status']=='failed' and failed['primary_engineering_seconds']==39.218419368)
 check('no_new_diagnostic_intent_or_Tensor_import',not (D.ROOT/D.CONDITIONS[1]/'permanent_intent.json').exists() and not any(n in sys.modules for n in ('torch','numpy')))
 result={'schema':'independent_mamba_repeat_diagnostic_source_review_v1','status':'diagnostic_source_reviewed_no_formal_qualification','passed':True,'reviewer':'/root/loss3_independent_review','reviewer_source':desc(SOURCE),'diagnostic_source':desc(p),'base_frozen_qualifier':desc(old),'checks':checks,'existing_Q13_registration':desc(q/'registration.json'),'existing_Q13_terminal':desc(q/'status.json'),'existing_CodeBrain_child_exit0':state['outcomes']['CodeBrain-lastblock-finetune'],'existing_partial_CPU15':state['partial_CPU_audit'],'preserved175_sources':reg['sources'],'preserved_Qual9_failed_attempt':desc(oldfailure),'diagnostic_dependencies':[desc(W/rel) for rel in D.PINS],'scope':['The old isolated builder, source-locked C2/G3/M4, sample/seed, native forward,6F4B48windows0step and strict repeat gate are unchanged.','Only statistics from already detached CPU loss/gradient/RNG captures are added before the unchanged strict gate.','Even a fully matching run produces only diagnostic_complete_no_formal_qualification, GPU_engineering_checks_passed=false and eligible_for_formal_registration=false.','Actual completed/released CodeBrain15, fixed175 source bytes, real child exit0, partial CPU15 120/360 and still blocked Mamba15 are prerequisites.','Existing native/Linux source/nonce/birth/UID1000/singleGPU checks remain unchanged; permanent intent is checked before controller launch only.'],'model_calls':0,'Tensor_operations':0,'GPU_operations':0,'actual_process_inventory_queries':0,'process_controls':0,'formal_fit_writes':0,'registration_writes':0,'outer_test_reads':0,'actual_diagnostic_run_completed':False,'eligible_for_formal_registration':False,'all_plan_complete':False,'future_cost_requirement':'Count each actual diagnostic attempt scope once as primary engineering; controller elapsed remains nested supplemental. Preserve failures, unknowns and original Qual9 cost; never add diagnostics to formal fit totals.','captured_ns':time.time_ns()}
 require(desc(p)==result['diagnostic_source'],'Diagnostic changed during review');target=A/('independent_mamba_repeat_diagnostic_source_review_'+str(result['captured_ns'])+'.json')
 with target.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'report':desc(target),'passed':True,'checks':len(checks)}))
if __name__=='__main__':main()

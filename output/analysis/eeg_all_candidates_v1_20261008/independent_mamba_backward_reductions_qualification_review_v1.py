"""Independent minimal four-source review. Stdlib AST/fake values only."""
import ast,copy,hashlib,importlib.util,json,os,sys,time,types
from pathlib import Path
SOURCE=Path(__file__).resolve();A=SOURCE.parent;W=SOURCE.parents[3];M=W/'tridep_ccml/eeg_all_candidates';OUT=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008';sys.path.insert(0,str(W));checks={}
def load(p,n):
 spec=importlib.util.spec_from_file_location(n,p);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);return v
H=load(A/'independent_core_prefix_repair_chain_review_v1.py','_minimal_review_helper');desc,read,node,tree,dump,require=H.desc,H.read,H.node,H.tree,H.dump,H.require

def check(k,v):require(v,k);checks[k]=True

def main():
 names=['mamba_backward_reductions_v1.py','mamba_private_backward_graph_v4.py','adapters/eegmamba_v5.py','qualify_deterministic_repairs_v10.py']
 hashes=['28d5d15b64f77a10c267f68b55f7439d1a64decd78278f45b684794d91891131','c96544ce2026a6c606f3736c3133f332dfd2ee5ab54f868a811313b08c3597c3','14e8d371696df13347e823db2e915be63826a77dd048810d934af9be6c0a03d6','e361be46244a9b0d2a07a29dfe634ee2f191a488dfc355eba1d6a17027c0e39b']
 sources=[desc(M/n) for n in names];check('exact_four_candidate_sources',all(d['sha256']==h for d,h in zip(sources,hashes)))
 q=M/names[-1];old=M/'diagnose_mamba_repeat_v1.py';g=M/names[1];g0=M/'mamba_private_backward_graph_v3.py';adapter=M/names[2]
 check('old_diagnostic_and_graph_bytes_preserved',desc(old)['sha256']=='008981ad5948bfc885772ed0aceda46b8ca00ffc332797a5b0defc2aa874775c' and desc(g0)['sha256']=='4ae48821d2350ebf1eaf5172e94ef20392efa931a7ff8f5afd73ba1bbdaa9ee9')
 class Normalize(ast.NodeTransformer):
  def visit_Constant(self,n):
   if isinstance(n.value,str):n.value=n.value.replace('qualify_deterministic_repairs_v10','diagnose_mamba_repeat_v1').replace('.adapters.eegmamba_v5','.adapters.eegmamba_v4').replace('engineering_verified','diagnostic_complete_no_formal_qualification')
   return n
  def visit_If(self,n):
   if any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='snapshot_native_backward_counts' for x in ast.walk(n)):
    require(ast.unparse(n.test)=='model is not None' and len(n.body)==2,'Only snapshot finally block removable');return None
   return self.generic_visit(n)
  def visit_Call(self,n):
   n=self.generic_visit(n)
   if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='record' and n.func.attr=='update' and any(k.arg=='status' and isinstance(k.value,ast.Constant) and k.value.value=='diagnostic_complete_no_formal_qualification' for k in n.keywords):
    for k in n.keywords:
     if k.arg in ('GPU_engineering_checks_passed','diagnostic_only','eligible_for_formal_registration'):
      require(isinstance(k.value,ast.Constant) and type(k.value.value) is bool,'Explicit qualification flags only');k.value=ast.Constant(value=k.arg=='diagnostic_only')
   return n
 oldfuncs={n.name:n for n in tree(old).body if isinstance(n,ast.FunctionDef)}
 for name,n in oldfuncs.items():check('qualifier_'+name+'_preserved_modulo_declared_route_status_snapshot',dump(Normalize().visit(copy.deepcopy(node(q,name))))==dump(n))
 for n in tree(g0).body:
  if isinstance(n,ast.FunctionDef) and n.name!='build_private_split_scan':check('graph_'+n.name+'_exact',dump(node(g,n.name))==dump(n))
 build=ast.unparse(node(g,'build_private_split_scan'));check('private_chunk_and_causal_only_global_overrides',"changes['_chunk_cumsum_bwd'] = private_chunk" in build and "'causal_conv1d_cuda': private_conv" in build and 'new_class.forward is original_class.forward' in build and "inner.__globals__['causal_conv1d_cuda'] is native_conv" in build and "core.__globals__['_chunk_cumsum_bwd'] is native_chunk" in build)
 check('native_source_code_and_extension_hash_guards_retained','c970c25ffcb5de3f0171dd035d73c0d59f62d711c0ebbc9a9c171c37194dc4be' in build and 'ac8ce3550175e716b150c091588a0e973062bd5ff2e695effff89fbe19ad9a84' in build and "'co_code', 'co_consts', 'co_names', 'co_varnames', 'co_argcount', 'co_posonlyargcount', 'co_kwonlyargcount', 'co_freevars', 'co_cellvars', 'co_flags'" in build)
 check('adapter_preserves_forward_parameter_identity_and_reports_changed_backward','mixer.forward.__func__.__code__ is original.__code__' in ast.unparse(node(adapter,'bind_mixer')) and "'native_backward_layout_and_reduction_order_changed': True" in ast.unparse(node(adapter,'load_encoder')) and "'performance_improvement_claimed': False" in ast.unparse(node(adapter,'load_encoder')))
 author=A/'mamba_backward_reductions_stdlib_contract_1791824126414413700.json';proof=read(author)
 check('actual_author_eleven_stdlib_contract_bound',desc(author)['sha256']=='2e471537072bfae01e187595158534c71c322a154b917dc2160aca14bf8d80ec' and proof['passed'] is True and proof['tests']==11 and proof['sources']==sources[:3] and desc(W/proof['test_source']['path'])==proof['test_source'])
 # Import only stdlib reduction source; extract fake classes without author runner or report writes.
 R=load(M/names[0],'_independent_reductions');test=W/proof['test_source']['path'];ns={'__builtins__':__builtins__}
 selected=[n for n in tree(test).body if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('f32','strides','Tensor','Torch')]
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(test),'exec'),ns);Tensor,T=ns['Tensor'],ns['Torch'];calls=[]
 def chunk(ddA,ddt_out,dt,A,dt_bias=None,dt_softplus=False,dt_limit=None,ddt=None):
  i=len(calls);calls.append(i);require(dt.shape==(1,95,8) and ddA.shape==(1,8,1,256),'B1 native invariant')
  return ddt.fill(10+i),Tensor((8,),i+1),None if dt_bias is None else Tensor((8,),2*(i+1))
 conv=types.SimpleNamespace(causal_conv1d_fwd=lambda:None,causal_conv1d_bwd=lambda *args:(_ for _ in ()).throw(ValueError('not invoked')))
 c,p,static=R.build_private_reductions(chunk,conv,torch_module=T);before=json.dumps(static,sort_keys=True);d=Tensor((8,95,8),stride=(1600,16,1))
 out,da,db=c(Tensor((8,8,1,256)),Tensor((8,8,1,256)),Tensor((8,95,8)),Tensor((8,)),Tensor((8,)),dt_softplus=True,ddt=d)
 check('independent_B8_single_chunk_native_calls_and_fixed_sum',calls==list(range(8)) and out is d and da.vals()==[36.]*8 and db.vals()==[72.]*8)
 check('static_proof_not_contaminated_by_live_counts',json.dumps(static,sort_keys=True)==before and R.snapshot_native_call_counts(c._native_call_counts)['chunk_native_calls_returned']==8)
 # One unreturned native call must retain started>returned, without fabricating completion.
 failed_calls=[]
 def fail(*args,**kw):
  failed_calls.append(1)
  if len(failed_calls)==2:raise ValueError('fixture native failure')
  return kw['ddt'],Tensor((8,),1),None
 fc,_,_=R.build_private_reductions(fail,conv,torch_module=T)
 try:fc(Tensor((2,8,1,256)),Tensor((2,8,1,256)),Tensor((2,95,8)),Tensor((8,)),dt_softplus=True)
 except ValueError:pass
 else:raise AssertionError('Native failure hidden')
 count=R.snapshot_native_call_counts(fc._native_call_counts);check('unreturned_native_failure_count_preserved',count['chunk_wrappers_started']==1 and count['chunk_native_calls_started']==2 and count['chunk_native_calls_returned']==1)
 ans={'R':R};H.extract(adapter,['snapshot_native_backward_counts'],ans);fake=types.SimpleNamespace(encoder=types.SimpleNamespace(layers=[types.SimpleNamespace(mixer=types.SimpleNamespace(_private_native_backward_counts=c._native_call_counts)) for _ in range(12)]));snapshot=ans['snapshot_native_backward_counts'](fake);c._native_call_counts['chunk_native_calls_started']+=1
 check('plain_Python_snapshot_is_copied_and_twelve_layers',len(snapshot)==12 and snapshot[0]['counts']['chunk_native_calls_started']==8 and c._native_call_counts['chunk_native_calls_started']==9)
 from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v10 as Q
 state=Q.check_codebrain_terminal();check('actual_Q13_CPU15_and_all175_still_bound',state['status']=='codebrain15_complete_mamba15_blocked_no_evaluation' and all(desc(W/rel)['sha256']==sha for rel,sha in Q.PINS.items()))
 require(not (Q.ROOT/'owner.json').exists() and not (Q.ROOT/Q.CONDITIONS[1]/'permanent_intent.json').exists(),'No actual Q10 run yet')
 check('new_qualification_intent_absent_and_no_Tensor_import',not any(k in sys.modules for k in ('torch','numpy')))
 diagnosis=OUT/'mamba_repeat_diagnosis_v1/EEGMamba-lastblock-finetune/1791823118874568500/attempt.json';actual=read(diagnosis)
 check('actual_failure_localized_loss_rng_equal_four_parameter_gradients',actual['status']=='failed' and actual['forward_returned']==6 and actual['backward_returned']==4 and actual['optimizer_step_returned']==0 and actual['primary_engineering_seconds']==38.444831497 and actual['repetition_diagnostics'][1]['loss_bitwise_equal'] is True and actual['repetition_diagnostics'][1]['rng_exact_equal'] is True and {x['parameter'].rsplit('.',1)[-1] for x in actual['repetition_diagnostics'][1]['different_gradients']}=={'A_log','bias','weight','dt_bias'})
 runtime=ast.unparse(node(q,'runtime_worker'));check('qualification_success_only_after_strict_repeat_resume_counts','engineering_verified' in runtime and runtime.index("'Repeated strict GPU loss/gradients/RNG differ'")<runtime.index("'Complete zero-step engineering resume differs'")<runtime.index("'Observed engineering counts differ'")<runtime.index("status='engineering_verified'") and 'separate_new_formal_execution_and_auditor_required' in runtime)
 report={'schema':'independent_mamba_backward_reductions_qualification_review_v1','status':'four_source_qualification_candidate_reviewed','passed':True,'reviewer':'/root/loss3_independent_review','reviewer_source':desc(SOURCE),'reviewed_sources':sources,'source_under_review':sources[-1],'checks':checks,'author_stdlib_contract':desc(author),'preceding_diagnostic_review':desc(A/'independent_mamba_repeat_diagnostic_source_review_1791822899143030100.json'),'actual_failed_diagnostic':desc(diagnosis),'actual_Q13_terminal':desc(OUT/'continuation_queue_v13/status.json'),'actual_CPU15':state['partial_CPU_audit'],'scope':['Only R1/G4/M5/Qual10 reviewed; no new formal pipeline.','Native forward/class identity retained. Backward chunk and causal-conv calls decompose registered batch into B1; causal buffers change to channels-first; parameter gradients accumulate on CPU FP32 in fixed batch order. This is explicitly changed rounding/layout/call order, not demonstrated original-CUDA-gradient equivalence.','Fixed seed11/fold1/first training person/8 windows and6forward4backward0optimizer strict loss/gradient/RNG/complete-checkpoint gates retained. No thresholds relaxed.','Native call counters are Python instrumentation separate from static audit; internal kernel/autotuner launches unknown. Calls are nested in model backward/attempt time and must not be added as new elapsed cost.','Actual GPU qualification remains unrun. Only actual successful report can enable a separately reviewed formal source/CPU/cost/evaluation chain; this review alone never qualifies training.'],'model_calls':0,'Tensor_operations':0,'GPU_operations':0,'process_inventory_queries':0,'process_controls':0,'formal_fit_writes':0,'outer_test_reads':0,'actual_GPU_qualification_complete':False,'eligible_for_formal_registration':False,'all_plan_complete':False,'limitations':['Standard-library fake tensors do not establish CUDA arithmetic or native kernel repeatability.','Transfer overhead and increased native-call costs are unmeasured; no speedup claim.','Original qualifier9 and the new failed diagnostic remain separate real engineering attempts; primary times each counted once, controller times nested.'],'captured_ns':time.time_ns()}
 require([desc(M/n) for n in names]==sources,'Candidate changed during review');target=A/('independent_mamba_backward_reductions_qualification_review_'+str(report['captured_ns'])+'.json')
 with target.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'report':desc(target),'passed':True,'checks':len(checks)}))
if __name__=='__main__':main()

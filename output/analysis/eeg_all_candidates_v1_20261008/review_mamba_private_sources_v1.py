"""Independent stdlib source/isolated graph review; no runtime Mamba imported."""
from pathlib import Path
import ast,functools,hashlib,inspect,json,time,types
W=Path(__file__).resolve().parents[3];SOURCE=Path(__file__).resolve();A=SOURCE.parent
def d(p):
 p=Path(p).resolve()
 assert p.is_relative_to(W)
 return {'path':p.relative_to(W).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
 graph=W/'tridep_ccml/eeg_all_candidates/mamba_private_backward_graph_v1.py'
 adapter=W/'tridep_ccml/eeg_all_candidates/adapters/eegmamba_v2.py'
 gp=A/'mamba_private_backward_graph_preparation_1791715032437300600.json'
 ap=A/'EEGMamba_private_adapter_v2_preparation_1791716575285075300.json'
 g=json.loads(gp.read_text(encoding='utf-8-sig'));a=json.loads(ap.read_text(encoding='utf-8-sig'))
 assert g['passed'] and a['passed'] and g['source']==d(graph) and a['source']==d(adapter)
 gt=graph.read_text(encoding='utf-8');at=adapter.read_text(encoding='utf-8')
 tree=ast.parse(gt)
 assert {n.names[0].name for n in tree.body if isinstance(n,ast.Import)}=={'copy','inspect','types'}
 ns={'__name__':'private_independent_review'}
 exec(compile(tree,str(graph)+':independent-review','exec'),ns)
 original_torch=types.SimpleNamespace(cumsum=lambda x:x+1)
 env={'torch':original_torch}
 exec('def _chunk_state_bwd_db(x):return torch.cumsum(x)\ndef _chunk_state_bwd_ddAcs_stable(x):return torch.cumsum(x)\ndef _mamba_chunk_scan_combined_bwd(x):return _chunk_state_bwd_db(x)+_chunk_state_bwd_ddAcs_stable(x)\ndef backward(ctx,x):return _mamba_chunk_scan_combined_bwd(x)\ndef forward(ctx,x):return x\n',env)
 original_inner=env['backward']
 def wrap(fn):
  @functools.wraps(fn)
  def call(*args,**kwargs):return fn(*args,**kwargs)
  return call
 old=type(ns['CLASS_KEY'],(object,),{'forward':staticmethod(env['forward']),'backward':staticmethod(wrap(original_inner))})
 env[ns['CLASS_KEY']]=old
 exec('def split(x):return MambaSplitConv1dScanCombinedFn.backward(None,x)\n',env)
 def factory(prev,fw,bw):return type('Private',(object,),{'forward':staticmethod(fw),'backward':staticmethod(bw)})
 repaired,proof=ns['build_private_split_scan'](env['split'],torch_namespace=types.SimpleNamespace(cumsum=lambda x:x+9),class_factory=factory)
 assert env['split'](7)==16 and repaired(7)==32
 assert old.forward is env['forward'] and old.backward.__wrapped__ is original_inner
 assert env['torch'] is original_torch and inspect.unwrap(repaired.__globals__[ns['CLASS_KEY']].backward).__code__ is original_inner.__code__
 assert proof['GPU_qualification_complete']==False
 funcs={n.name:n for n in ast.parse(at).body if isinstance(n,ast.FunctionDef)}
 calls=[ast.unparse(n.func) for n in ast.walk(funcs['load_encoder']) if isinstance(n,ast.Call)]
 assert calls.count('V1.load_encoder')==1 and calls.index('check_sources')<calls.index('V1.load_encoder')
 assert 'GPU_qualification_complete=False' in at and 'new_formal_execution_registration_required=True' in at
 assert 'native_CUDA_numerical_equivalence_established' in at and 'accumulation_order_changed' in at
 assert not any(x in at for x in ('use_deterministic_algorithms(', 'torch.load(', 'sys.modules['))
 manifest=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v5/recovery_registration_v2.json'
 for item in json.loads(manifest.read_text(encoding='utf-8-sig'))['sources']:assert d(W/item['path'])==item
 value={'schema':'independent_mamba_private_graph_and_adapter_source_review_v1','passed':True,
  'review_source':d(SOURCE),'graph_source':d(graph),'adapter_source':d(adapter),'producer_tests':[d(gp),d(ap)],
  'checks':{'private_graph_and_originals_independently_exercised':True,'exact_forward_reference_retained':True,
            'backward_bytecode_and_decorator_leaf_preserved':True,'source_guard_before_official_load':True,
            'order_change_and_pending_flags_explicit':True,'current84_sources_unchanged':True},
  'scope':'Source AST and independently exercised stdlib fake graph only; not an actual Linux class/model qualification',
  'tensor_operations':0,'model_constructions':0,'checkpoint_loads':0,'GPU_operations':0,
  'production_integration_applied':False,'GPU_qualification_complete':False,'final_plan_certificate':False,
  'remaining':['Actual pinned Linux AMP/autograd graph wiring','CUDA forward/gradient/frozen/recovery engineering',
               'Numerical/order and transfer-cost disclosure','New formal source and all final gates']}
 target=A/('independent_mamba_private_sources_review_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'passed':True,'artifact':d(target),'adapter':d(adapter)}))
if __name__=='__main__':main()

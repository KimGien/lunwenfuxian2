"""Pure stdlib fake-graph tests: no torch/Mamba/Tensor/model/GPU imports."""
from pathlib import Path
import functools,hashlib,io,json,sys,time,types,unittest
W=Path(__file__).resolve().parents[3];sys.path.insert(0,str(W))
from tridep_ccml.eeg_all_candidates import mamba_private_backward_graph_v1 as M

def decorator(fn):
 @functools.wraps(fn)
 def wrapper(*args,**kwargs):return fn(*args,**kwargs)
 return wrapper

def fixture():
 namespace={'torch':types.SimpleNamespace(cumsum=lambda x:x+1)}
 exec('def _chunk_state_bwd_db(x):return torch.cumsum(x)\ndef _chunk_state_bwd_ddAcs_stable(x):return torch.cumsum(x)\ndef _mamba_chunk_scan_combined_bwd(x):return _chunk_state_bwd_db(x)+_chunk_state_bwd_ddAcs_stable(x)\ndef backward(ctx,x):return _mamba_chunk_scan_combined_bwd(x)\ndef forward(ctx,x):return x\n',namespace)
 backward=decorator(decorator(namespace['backward']))
 cls=type(M.CLASS_KEY,(object,),{'forward':staticmethod(namespace['forward']),'backward':staticmethod(backward)})
 namespace[M.CLASS_KEY]=cls
 exec('def split(x):return MambaSplitConv1dScanCombinedFn.backward(None,x)\n',namespace)
 return namespace

def factory(old,forward,backward):
 return type('Private'+old.__name__,(object,),{'forward':staticmethod(forward),'backward':staticmethod(backward)})

class Guards(unittest.TestCase):
 def test_native_forward_reused_and_only_private_backward_changes(self):
  ns=fixture();old=ns[M.CLASS_KEY];wrapped=old.backward;core=ns[M.CORE_KEY]
  repaired,proof=M.build_private_split_scan(ns['split'],torch_namespace=types.SimpleNamespace(cumsum=lambda x:x+100),class_factory=factory)
  self.assertEqual(ns['split'](5),12);self.assertEqual(repaired(5),210)
  self.assertIs(ns[M.CLASS_KEY],old);self.assertIs(old.backward,wrapped);self.assertIs(ns[M.CORE_KEY],core)
  self.assertIs(repaired.__code__,ns['split'].__code__)
  self.assertFalse(proof['GPU_qualification_complete']);self.assertTrue(proof['new_formal_execution_binding_required'])
 def test_both_helper_globals_are_private(self):
  ns=fixture();repaired,_=M.build_private_split_scan(ns['split'],torch_namespace=types.SimpleNamespace(cumsum=lambda x:x+10),class_factory=factory)
  inner=__import__('inspect').unwrap(repaired.__globals__[M.CLASS_KEY].backward)
  core=inner.__globals__[M.CORE_KEY]
  for name in M.HELPERS:
   self.assertIs(core.__globals__[name].__code__,ns[name].__code__)
   self.assertIsNot(core.__globals__[name].__globals__,ns[name].__globals__)
   self.assertEqual(ns[name](2),3)
 def test_closure_defaults_and_annotation_preserved(self):
  def closure(v):
   def fn(x,*,more=1)->int:return v+x+more
   return fn
  original=closure(3);result=M.clone(original)
  self.assertIs(result.__closure__,original.__closure__)
  self.assertEqual(result.__kwdefaults__,original.__kwdefaults__)
  self.assertEqual(result.__annotations__,original.__annotations__)
  self.assertEqual(result(5),9)
 def test_no_wrapper_leaf_case(self):
  def f(x):return x
  def g(x):return x+1
  self.assertIs(M.replace_wrapped_leaf(f,f,g),g)
 def test_unknown_decorator_scope_rejected(self):
  def f(x):return x
  def leaf(x):return x
  with self.assertRaises(ValueError):M.replace_wrapped_leaf(f,leaf,leaf)
 def test_missing_helper_or_wrong_class_factory_rejected(self):
  ns=fixture();del ns[M.HELPERS[1]]
  with self.assertRaises(ValueError):M.build_private_split_scan(ns['split'],torch_namespace=types.SimpleNamespace(cumsum=lambda x:x),class_factory=factory)
  ns=fixture()
  with self.assertRaises(ValueError):M.build_private_split_scan(ns['split'],torch_namespace=types.SimpleNamespace(cumsum=lambda x:x),class_factory=lambda old,fw,bw:old)

def d(p):
 p=Path(p).resolve()
 return {'path':p.relative_to(W).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
if __name__=='__main__':
 stream=io.StringIO();start=time.perf_counter()
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
 value={'schema':'mamba_private_backward_graph_stdlib_preparation_v1','passed':result.wasSuccessful(),
 'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
 'observed_test_runner_seconds':time.perf_counter()-start,'source':d(M.__file__),'test_source':d(__file__),
 'scope':'Pure stdlib fake graph and class factory; no installed package imports or runtime wiring',
 'model_constructions':0,'checkpoint_loads':0,'tensor_operations':0,'GPU_operations':0,
 'production_integration_applied':False,'GPU_qualification_complete':False,'final_plan_certificate':False,
 'output':stream.getvalue()}
 target=Path(__file__).parent/('mamba_private_backward_graph_preparation_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'passed':value['passed'],'tests':result.testsRun,'artifact':d(target),
                   'output':stream.getvalue() if not result.wasSuccessful() else 'all passed'}))
 raise SystemExit(0 if result.wasSuccessful() else 1)

"""Standard-library AST and fake-value guards; no torch/Mamba/model/GPU imports."""
from pathlib import Path
import ast, copy, functools, hashlib, importlib.util, inspect, io, json, struct, sys, time, types, unittest
from unittest.mock import patch
WORK=Path(__file__).resolve().parents[3]
MOD=WORK/'tridep_ccml/eeg_all_candidates'
G_PATH=MOD/'mamba_private_backward_graph_v2.py'
spec=importlib.util.spec_from_file_location('_private_graph_v2_stdlib',G_PATH)
G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
C_PATH=MOD/'mamba_cumsum_cpu_fp32_v2.py'
A_PATH=MOD/'adapters/eegmamba_v3.py'
COMBINED=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets/eegmamba_linux_env/lib/python3.12/site-packages/mamba_ssm/ops/triton/ssd_combined.py'

def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
class FakeValue:
 def __init__(self,values,device='cuda:synthetic',shape=None,*,parent=None,index=None):
  self.values=list(values);self.device=device;self.dtype='float32';self.requires_grad=False
  self.shape=(len(self.values),) if shape is None else shape;self.ndim=len(self.shape)
  self.parent=parent;self.index=index
 def vals(self):return [self.parent.values[self.index]] if self.parent is not None else list(self.values)
 def numel(self):return len(self.vals())
 def detach(self):return self
 def to(self,*,device):return FakeValue(self.vals(),device,self.shape)
 def contiguous(self):return self
 def clone(self):return FakeValue(self.vals(),self.device,self.shape)
 def __getitem__(self,index):
  return FakeValue([self.vals()[index[-1]]],self.device,(),parent=self,index=index[-1])
 def __add__(self,other):return FakeValue([f32(x+y) for x,y in zip(self.vals(),other.vals())],self.device,self.shape)
 def copy_(self,other):
  if self.parent is not None:self.parent.values[self.index]=other.vals()[0]
  else:self.values[:]=other.vals()
  return self
 def stride(self):return (1,) if self.shape else ()
 def storage_offset(self):return 0
 def data_ptr(self):return id(self.values)
T=types.SimpleNamespace(Tensor=FakeValue,float32='float32',
 empty_like=lambda x:FakeValue([0]*x.numel(),x.device,x.shape),
 zeros_like=lambda x:FakeValue([0]*x.numel(),x.device,x.shape))
tree=ast.parse(C_PATH.read_text(encoding='utf-8'))
nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('require','cumsum')]
for node in nodes:node.decorator_list=[]
namespace={'torch':T}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(C_PATH),'exec'),namespace)
scan=namespace['cumsum']

def core_node():
 tree=ast.parse(COMBINED.read_text(encoding='utf-8'))
 return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==G.CORE_KEY)
def decorate(fn):
 @functools.wraps(fn)
 def wrapped(*args,**kwargs):return fn(*args,**kwargs)
 return wrapped
def fixture():
 ns={'torch':types.SimpleNamespace(cumsum=lambda x:x+1)}
 exec('def _chunk_state_bwd_db(x):return torch.cumsum(x)\ndef _chunk_state_bwd_ddAcs_stable(x):return torch.cumsum(x)\ndef _mamba_chunk_scan_combined_bwd(x):return _chunk_state_bwd_db(x)+_chunk_state_bwd_ddAcs_stable(x)\ndef backward(ctx,x):return _mamba_chunk_scan_combined_bwd(x)\ndef forward(ctx,x):return x\n',ns)
 cls=type(G.CLASS_KEY,(object,),{'forward':staticmethod(ns['forward']),'backward':staticmethod(decorate(decorate(ns['backward'])))})
 ns[G.CLASS_KEY]=cls;exec('def split(x):return MambaSplitConv1dScanCombinedFn.backward(None,x)',ns)
 return ns
def factory(old,fw,bw):return type('Private'+old.__name__,(object,),{'forward':staticmethod(fw),'backward':staticmethod(bw)})

class Guards(unittest.TestCase):
 def test_alias_out_original_statements_exact(self):
  old=next(n for n in ast.parse((MOD/'mamba_cumsum_cpu_fp32_v1.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='cumsum')
  new=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cumsum')
  self.assertIsInstance(new.body[0],ast.If)
  self.assertEqual(ast.dump(ast.Module(body=old.body,type_ignores=[])),ast.dump(ast.Module(body=new.body[1:],type_ignores=[])))
 def test_new_result_fp32_order_new_storage_device_shape(self):
  value=FakeValue([16777216.,1.,-16777216.]);before=value.vals()
  result=scan(value,-1)
  self.assertEqual(result.vals(),[16777216.,16777216.,0.])
  self.assertIsNot(result,value);self.assertNotEqual(result.data_ptr(),value.data_ptr())
  self.assertEqual((result.shape,result.dtype,result.device),(value.shape,value.dtype,value.device))
  self.assertEqual(value.vals(),before)
 def test_alias_out_returns_exact_same_object(self):
  value=FakeValue([1.,2.,3.])
  self.assertIs(scan(value,-1,out=value),value);self.assertEqual(value.vals(),[1.,3.,6.])
 def test_no_higher_order_wrong_dtype_or_dim(self):
  for key,bad in [('requires_grad',True),('dtype','float64')]:
   value=FakeValue([1.,2.]);setattr(value,key,bad)
   with self.assertRaises(ValueError):scan(value,-1)
  for dim in (True,1,0.0):
   with self.assertRaises(ValueError):scan(FakeValue([1.,2.]),dim)
  with self.assertRaises(ValueError):scan(FakeValue([]),-1)
 def test_exact_core_one_call_and_rest_AST_preserved(self):
  original=core_node();changed=G.rewrite_core_ast(original)
  calls=[n for n in ast.walk(changed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==G.PRIVATE_CUMSUM]
  self.assertEqual(len(calls),1)
  class Undo(ast.NodeTransformer):
   def visit_Call(self,n):
    if isinstance(n.func,ast.Name) and n.func.id==G.PRIVATE_CUMSUM:
     return ast.parse(G.TARGET_EXPRESSION,mode='eval').body
    return self.generic_visit(n)
  restored=Undo().visit(copy.deepcopy(changed))
  self.assertEqual(ast.dump(original,include_attributes=False),ast.dump(restored,include_attributes=False))
  self.assertEqual(hashlib.sha256(COMBINED.read_bytes()).hexdigest(),G.COMBINED_SHA)
 def test_source_bound_loaded_core_code_and_no_original_mutation(self):
  node=core_node();code=G._code_for(node,str(COMBINED))
  ns={'sentinel':object()};original=types.FunctionType(code,ns,G.CORE_KEY)
  before=dict(ns);helper=lambda x,dim:x
  actual=G.private_core(original,{'helper_override':17},helper)
  self.assertEqual(ns,before);self.assertIs(actual.__globals__[G.PRIVATE_CUMSUM],helper)
  self.assertEqual(actual.__globals__['helper_override'],17)
  self.assertIs(original.__code__,code);self.assertNotEqual(actual.__code__.co_code,code.co_code)
 def test_bad_source_or_changed_loaded_core_rejected(self):
  node=core_node();code=G._code_for(node,str(COMBINED));original=types.FunctionType(code,{},G.CORE_KEY)
  with patch.object(G,'COMBINED_SHA','0'*64),self.assertRaises(ValueError):G.private_core(original,{},lambda x:x)
  changed=G._code_for(G.rewrite_core_ast(node),str(COMBINED))
  foreign=types.FunctionType(changed,{},G.CORE_KEY)
  with self.assertRaises(ValueError):G.private_core(foreign,{},lambda x:x)
 def test_missing_duplicate_or_other_cumsum_rejected(self):
  node=core_node()
  for bad in ('def '+G.CORE_KEY+'(x):\n return x\n',
              'def '+G.CORE_KEY+'(x):\ a='+G.TARGET_EXPRESSION+'\n b='+G.TARGET_EXPRESSION+'\n return a+b\n',
              'def '+G.CORE_KEY+'(x):\ a='+G.TARGET_EXPRESSION+'\n return x.cumsum(dim=-1)+a\n'):
   with self.assertRaises(ValueError):G.rewrite_core_ast(ast.parse(bad).body[0])
 def test_global_name_collision_rejected(self):
  code=G._code_for(core_node(),str(COMBINED));original=types.FunctionType(code,{G.PRIVATE_CUMSUM:None},G.CORE_KEY)
  with self.assertRaises(ValueError):G.private_core(original,{},lambda x:x)
 def test_fake_closures_helpers_and_forward_remain_private(self):
  ns=fixture();old=ns[G.CLASS_KEY];wrapped=old.backward;core=ns[G.CORE_KEY]
  with patch.object(G,'private_core',side_effect=lambda original,changes,helper:G.clone(original,changes)) as bridge:
   private,proof=G.build_private_split_scan(ns['split'],torch_namespace=types.SimpleNamespace(cumsum=lambda x:x+100),class_factory=factory)
  self.assertEqual(ns['split'](5),12);self.assertEqual(private(5),210)
  self.assertIs(ns[G.CLASS_KEY],old);self.assertIs(old.backward,wrapped);self.assertIs(ns[G.CORE_KEY],core)
  self.assertIs(private.__globals__[G.CLASS_KEY].forward,old.forward)
  self.assertEqual(bridge.call_count,1);self.assertFalse(proof['original_function_bytecode_preserved'])
  self.assertTrue(proof['core_AST_single_cumsum_call_replaced']);self.assertFalse(proof['GPU_qualification_complete'])
 def test_adapter_pins_new_sources_and_keeps_original_forward_route(self):
  data=A_PATH.read_text(encoding='utf-8')
  for p in (C_PATH,G_PATH):self.assertIn(hashlib.sha256(p.read_bytes()).hexdigest(),data)
  self.assertIn('mamba_cumsum_cpu_fp32_v2 as C',data);self.assertIn('mamba_private_backward_graph_v2 as G',data)
  self.assertIn('adapter_version=3',data);self.assertIn('original_native_forward_function_reused',data)
  self.assertIn("'original_core_bytecode_preserved':False",data)
  self.assertNotIn('use_deterministic_algorithms',data)
 def test_old_sources_unchanged(self):
  pins={'mamba_cumsum_cpu_fp32_v1.py':'0159cec62acd3c7d07d60bbdeef961110e28d5b5579d47267655399da4e3c0e4',
        'mamba_private_backward_graph_v1.py':'417c2440d0ff5f1be7ea923d2f81bfff56faa3d8444019013b0df246a0efab5a',
        'adapters/eegmamba_v2.py':'c516c843db45ab987da560d1c952de1d888d0f873e57e5b2c262aa1c0b655741'}
  for name,sha in pins.items():self.assertEqual(hashlib.sha256((MOD/name).read_bytes()).hexdigest(),sha)

def desc(path):
 path=Path(path).resolve()
 return {'path':path.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
if __name__=='__main__':
 stream=io.StringIO();start=time.perf_counter()
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
 value={'schema':'mamba_core_reverse_cumsum_stdlib_preparation_v1','passed':result.wasSuccessful(),
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'source_descriptors':[desc(p) for p in (C_PATH,G_PATH,A_PATH,COMBINED,Path(__file__))],
        'scope':'stdlib AST/code-object and fake values/closures only; no torch/Mamba imports or real tensor operations',
        'model_constructions':0,'checkpoint_loads':0,'real_tensor_operations':0,'GPU_operations':0,'process_controls':0,
        'GPU_qualification_complete':False,'native_CUDA_gradient_equivalence_established':False,
        'accumulation_order_changed':True,'performance_improvement_claimed':False,
        'seconds':time.perf_counter()-start,'output':stream.getvalue()}
 target=Path(__file__).parent/('mamba_core_reverse_cumsum_stdlib_preparation_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as out:json.dump(value,out,indent=2);out.write('\n')
 print(json.dumps({'passed':value['passed'],'tests':result.testsRun,'artifact':desc(target),
                   'output':stream.getvalue() if not result.wasSuccessful() else 'all passed'}))
 raise SystemExit(0 if result.wasSuccessful() else 1)


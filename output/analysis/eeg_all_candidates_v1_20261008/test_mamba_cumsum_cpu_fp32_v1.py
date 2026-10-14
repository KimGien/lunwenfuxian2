"""Synthetic CPU tests; no installed Mamba import, author model or GPU."""
from pathlib import Path
import hashlib,io,json,sys,time,unittest
from unittest.mock import patch
W=Path(__file__).resolve().parents[3];sys.path.insert(0,str(W))
import torch
from tridep_ccml.eeg_all_candidates import mamba_cumsum_cpu_fp32_v1 as M

class Guards(unittest.TestCase):
 def setUp(self):
  self.guard=patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('No GPU allowed'))
  self.guard.start()
 def tearDown(self):self.guard.stop()
 def test_binary_exact_bounded_prefixes_and_alias(self):
  generator=torch.Generator(device='cpu').manual_seed(34)
  for n in (1,2,7,32,256):
   x=torch.randint(-8,9,(2,3,4,n),generator=generator).float()/16
   expected=M._NATIVE_CUMSUM(x,dim=-1);ptr=x.data_ptr()
   result=M.cumsum(x,dim=-1,out=x)
   self.assertIs(result,x);self.assertEqual(x.data_ptr(),ptr)
   self.assertEqual(x.dtype,torch.float32);self.assertTrue(torch.equal(x,expected))
 def test_noncontiguous_same_storage_views(self):
  x=torch.arange(2*3*17,dtype=torch.float32).reshape(2,3,17)/16
  before=x.clone();expected=M._NATIVE_CUMSUM(x[...,1:],dim=-1)
  result=M.cumsum(x[...,1:],dim=-1,out=x[...,1:])
  self.assertTrue(torch.equal(result,expected))
  self.assertTrue(torch.equal(x[...,0],before[...,0]))
  self.assertEqual(x.stride(),before.stride())
 def test_fixed_fp32_order_is_explicit_not_native_cpu_identity(self):
  x=torch.tensor([float(2**24),1.,-float(2**24)],dtype=torch.float32)
  native=M._NATIVE_CUMSUM(x,dim=-1)
  result=M.cumsum(x,dim=-1,out=x)
  self.assertTrue(torch.equal(result,torch.tensor([float(2**24),float(2**24),0.])))
  self.assertEqual(float(native[-1]),1.)
  self.assertFalse(torch.equal(result,native))
 def test_repeat_and_flags_unchanged(self):
  generator=torch.Generator(device='cpu').manual_seed(145)
  original=torch.randn(2,3,32,generator=generator)
  enabled=torch.are_deterministic_algorithms_enabled()
  warn=torch.is_deterministic_algorithms_warn_only_enabled()
  outputs=[M.cumsum(original.clone(),dim=-1,out=(x:=original.clone())) if False else None]
  outputs=[]
  for _ in range(3):
   x=original.clone();outputs.append(M.cumsum(x,dim=-1,out=x).clone())
  self.assertTrue(all(torch.equal(outputs[0],x) for x in outputs))
  self.assertEqual(torch.are_deterministic_algorithms_enabled(),enabled)
  self.assertEqual(torch.is_deterministic_algorithms_warn_only_enabled(),warn)
 def test_private_namespace_global_unchanged(self):
  original=torch.cumsum;namespace=M.torch_namespace()
  self.assertIs(torch.cumsum,original);self.assertIs(namespace.cumsum,M.cumsum)
  self.assertIs(namespace.empty,torch.empty);self.assertIs(namespace.cuda,torch.cuda)
 def test_scope_guards(self):
  x=torch.zeros(2,3)
  cases=[lambda:M.cumsum(x,0,out=x),lambda:M.cumsum(x,-1,out=x.clone()),
         lambda:M.cumsum(x,-1),lambda:M.cumsum(x.double(),-1,out=x.double()),
         lambda:M.cumsum(x,-1,out=x,dtype=torch.float64),
         lambda:M.cumsum(x.requires_grad_(),-1,out=x)]
  for call in cases:
   with self.assertRaises(ValueError):call()

def d(p):
 p=Path(p).resolve()
 return {'path':p.relative_to(W).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
if __name__=='__main__':
 stream=io.StringIO();start=time.perf_counter()
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
 value={'schema':'mamba_fixed_fp32_cumsum_CPU_preparation_v1','passed':result.wasSuccessful(),
        'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
        'observed_test_runner_seconds':time.perf_counter()-start,'source':d(M.__file__),'test_source':d(__file__),
        'scope':'Synthetic CPU FP32 tensors only; not a full Mamba gradient or CUDA comparison',
        'accumulation_order_changed_and_declared':True,
        'native_CPU_bitwise_identity_not_claimed':True,'native_CUDA_numerical_equivalence_established':False,
        'model_constructions':0,'checkpoint_loads':0,'GPU_operations':0,'raw_arrays_loaded':0,'optimizer_steps':0,
        'installed_packages_modified':False,'global_torch_function_patched':False,
        'production_integration_applied':False,'GPU_qualification_complete':False,'final_plan_certificate':False,
        'test_output':stream.getvalue()}
 target=Path(__file__).parent/('mamba_fixed_fp32_cumsum_CPU_preparation_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'passed':value['passed'],'tests':result.testsRun,'artifact':d(target),
                   'output':stream.getvalue() if not result.wasSuccessful() else 'all passed'}))
 raise SystemExit(0 if result.wasSuccessful() else 1)

"""CPU/fake loader guards only; never load official Linux Mamba/model/weights."""
from pathlib import Path
import hashlib,io,json,sys,time,types,unittest
from unittest.mock import patch
W=Path(__file__).resolve().parents[3];sys.path.insert(0,str(W))
import torch
from tridep_ccml.eeg_all_candidates.adapters import eegmamba_v2 as A

class Guards(unittest.TestCase):
 def setUp(self):
  self.guard=patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('No GPU allowed'));self.guard.start()
 def tearDown(self):self.guard.stop()
 def test_original_input_and_feature_dimension_preserved(self):
  self.assertEqual(A.FEATURE_DIM,200)
  for key in ('raw','model_input','physical_window_seconds','sample_rate_hz','pooling','direction'):
   self.assertEqual(A.INPUT_CONTRACT[key],A.V1.INPUT_CONTRACT[key])
 def test_fake_loader_preserves_parameter_names_and_marks_pending(self):
  model=types.SimpleNamespace(encoder=types.SimpleNamespace(layers=[types.SimpleNamespace(mixer=object()) for _ in range(12)]),
    named_parameters=lambda:[('last',types.SimpleNamespace(requires_grad=True))])
  with patch.object(A.V1,'load_encoder',return_value=(model,{})) as load, \
       patch.object(A,'bind_mixer',return_value={'fake':True}) as bind:
   got,audit=A.load_encoder(device='cpu',trainable=True)
   load.assert_called_once_with(device='cpu',trainable=True)
   self.assertIs(got,model);self.assertEqual(bind.call_count,12)
   self.assertFalse(audit['GPU_qualification_complete'])
   self.assertTrue(audit['new_formal_execution_registration_required'])
   self.assertTrue(audit['runtime_backward_adaptation']['accumulation_order_changed'])
   self.assertFalse(audit['runtime_backward_adaptation']['native_CUDA_numerical_equivalence_established'])
 def test_bad_pin_rejected_before_loader(self):
  original=A.desc
  def bad(path):
   d=original(path)
   if Path(path).resolve()==Path(A.V1.__file__).resolve():d['sha256']='0'*64
   return d
  with patch.object(A,'desc',side_effect=bad),patch.object(A.V1,'load_encoder',side_effect=AssertionError('model called')):
   with self.assertRaises(ValueError):A.load_encoder(device='cpu')
 def test_duplicate_and_foreign_mixer_rejected(self):
  duplicate=types.SimpleNamespace(_private_fp32_scan_bound=True)
  with self.assertRaises(ValueError):A.bind_mixer(duplicate)
  class Foreign:
   use_mem_eff_path=True
   def forward(self,x):return x
  with self.assertRaises(ValueError):A.bind_mixer(Foreign())
 def test_class_factory_keeps_native_forward_and_original_class(self):
  def forward(ctx,x):return x
  def backward(ctx,x):return x
  class Old(torch.autograd.Function):
   @staticmethod
   def forward(ctx,x):return x
   @staticmethod
   def backward(ctx,x):return x
  original=Old.forward
  cls=A.class_factory(Old,forward,backward)
  self.assertIsNot(cls,Old);self.assertIs(cls.forward,forward);self.assertIs(cls.backward,backward)
  self.assertIs(Old.forward,original)

def d(p):
 p=Path(p).resolve()
 return {'path':p.relative_to(W).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
if __name__=='__main__':
 stream=io.StringIO();start=time.perf_counter()
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
 value={'schema':'EEGMamba_private_adapter_v2_fake_preparation_v1','passed':result.wasSuccessful(),
 'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
 'observed_test_runner_seconds':time.perf_counter()-start,'source':d(A.__file__),'test_source':d(__file__),
 'scope':'Fake loader and class factory only; installed Mamba not imported and no official model/weights loaded',
 'model_constructions':0,'checkpoint_loads':0,'tensor_operations':0,'GPU_operations':0,'raw_arrays_loaded':0,
 'production_integration_applied':False,'GPU_qualification_complete':False,'final_plan_certificate':False,
 'output':stream.getvalue()}
 target=Path(__file__).parent/('EEGMamba_private_adapter_v2_preparation_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'passed':value['passed'],'tests':result.testsRun,'artifact':d(target),
                   'output':stream.getvalue() if not result.wasSuccessful() else 'all passed'}))
 raise SystemExit(0 if result.wasSuccessful() else 1)

"""Fake CPU/function tests; never load official CodeBrain model or weights."""
from pathlib import Path
import hashlib,io,json,sys,time,types,unittest
from unittest.mock import patch
W=Path(__file__).resolve().parents[3];sys.path.insert(0,str(W))
import torch
import torch.nn.functional as F
from tridep_ccml.eeg_all_candidates.adapters import codebrain_v3 as A
from tridep_ccml.eeg_all_candidates import codebrain_linear_backward_cpu_v1 as M

class FakeBlock:
    kernel_norm_initialized=True
    def forward(self,x,return_kernel=False):
        y=F.interpolate(x,scale_factor=2,mode='linear')
        return y if not return_kernel else (y,F.pad(x,(0,1)))

class Guards(unittest.TestCase):
    def setUp(self):
        self.guard=patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('No GPU permitted'))
        self.guard.start()
    def tearDown(self): self.guard.stop()

    def test_clone_private_bytecode_defaults_and_class_unchanged(self):
        original=FakeBlock.forward;original_F=original.__globals__['F']
        fn=A.clone_forward(original,M.functional_namespace())
        block=FakeBlock();block.forward=types.MethodType(fn,block)
        self.assertIs(fn.__code__,original.__code__)
        self.assertEqual(fn.__defaults__,original.__defaults__)
        self.assertIs(FakeBlock.forward,original)
        self.assertIs(original.__globals__['F'],original_F)
        self.assertIs(original_F.interpolate,M._NATIVE_INTERPOLATE)
        self.assertIsNot(fn.__globals__,original.__globals__)
        self.assertIs(fn.__globals__['F'].interpolate,M.interpolate)

    def test_cpu_output_gradient_and_forward_keyword_contract(self):
        one=FakeBlock();two=FakeBlock()
        two.forward=types.MethodType(A.clone_forward(FakeBlock.forward,M.functional_namespace()),two)
        x=torch.arange(12,dtype=torch.float32).reshape(1,3,4).requires_grad_()
        z=x.detach().clone().requires_grad_()
        y=one.forward(x);r,pad=two.forward(z,return_kernel=True)
        self.assertTrue(torch.equal(y,r));self.assertTrue(torch.equal(pad,F.pad(z,(0,1))))
        y.square().sum().backward();r.square().sum().backward()
        self.assertTrue(torch.equal(x.grad,z.grad))

    def test_foreign_and_duplicate_blocks_rejected(self):
        with self.assertRaises(ValueError): A.bind_author_block(FakeBlock())
        obj=FakeBlock();obj._codebrain_cpu_bridge_bound=True
        with self.assertRaises(ValueError): A.bind_author_block(obj)

    def test_changed_sources_rejected_before_model_load(self):
        actual=A.desc
        def wrong(path):
            d=actual(path)
            if Path(path).resolve()==Path(A.V2.__file__).resolve(): d['sha256']='0'*64
            return d
        with patch.object(A,'desc',side_effect=wrong),patch.object(A.V2,'load_encoder',side_effect=AssertionError('model called')):
            with self.assertRaises(ValueError): A.load_encoder(device='cpu')

    def test_fake_loader_preserves_names_and_marks_pending(self):
        blocks=[types.SimpleNamespace(S41=FakeBlock()) for _ in range(8)]
        param=types.SimpleNamespace(requires_grad=True)
        obj=types.SimpleNamespace(residual_layer=types.SimpleNamespace(residual_blocks=blocks),
                                  named_parameters=lambda:[('head',param)])
        with patch.object(A.V2,'load_encoder',return_value=(obj,{'implementation_sources':[]})) as loader, \
             patch.object(A,'bind_author_block',return_value={'fake_binding':True}) as bind:
            got,audit=A.load_encoder(device='cpu',trainable=True)
            loader.assert_called_once_with(device='cpu',trainable=True)
            self.assertIs(got,obj);self.assertEqual(bind.call_count,8)
            self.assertFalse(audit['GPU_qualification_complete'])
            self.assertTrue(audit['new_formal_execution_registration_required'])
            self.assertEqual(audit['adapter_version'],3)
            self.assertFalse(audit['runtime_gradient_backend_adaptation']['performance_improvement_claimed'])

    def test_no_parameters_constructed_or_global_namespace_patched(self):
        original=F.interpolate
        self.assertEqual(A.FEATURE_DIM,200)
        self.assertEqual(A.INPUT_CONTRACT['model_input'],A.V2.INPUT_CONTRACT['model_input'])
        self.assertEqual(A.INPUT_CONTRACT['physical_window_seconds'],5)
        self.assertIs(F.interpolate,original)

def desc(path):
 p=Path(path).resolve()
 return {'path':p.relative_to(W).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
if __name__=='__main__':
 stream=io.StringIO();start=time.perf_counter()
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
 value={'schema':'codebrain_private_adapter_v3_fake_cpu_preparation_v1','passed':result.wasSuccessful(),
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'observed_test_runner_seconds':time.perf_counter()-start,'source':desc(A.__file__),
        'test_source':desc(__file__),'bridge_source':desc(M.__file__),
        'scope':'Fake loader/function and synthetic CPU tensors; no official model/weight loading',
        'official_model_constructions':0,'checkpoint_loads':0,'GPU_operations':0,'raw_arrays_loaded':0,'optimizer_steps':0,
        'actual_current_queue_integration_applied':False,'actual_GPU_qualification_complete':False,
        'final_plan_certificate':False,'output':stream.getvalue()}
 target=Path(__file__).parent/('codebrain_private_adapter_v3_preparation_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'passed':value['passed'],'tests':result.testsRun,'artifact':desc(target),
                   'output':stream.getvalue() if not result.wasSuccessful() else 'all passed'}))
 raise SystemExit(0 if result.wasSuccessful() else 1)

"""Qualification safety/unit tests; no actual GPU/cohort/checkpoint execution."""
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

REPORT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_training_sync_qualifier_test',REPORT/'qualify_training_sync_v1.py')
K=importlib.util.module_from_spec(spec);spec.loader.exec_module(K)
CPU_TEST_COUNTS={'synthetic_backward_started':0,'synthetic_backward_returned':0,'synthetic_models':0}


def fixture():
    split={'train':['train'+str(i) for i in range(27)],'val':['val'+str(i) for i in range(3)],'test':['test'+str(i) for i in range(7)]}
    fit={'fit_id':'CSBrain-lastblock-finetune/seed_22/fold_5','condition':K.CONDITION,'seed':22,'fold':5,'stage':'3',
         **{k+'_ids':v for k,v in split.items()}}
    proof={'schema':'sync_qualification_safe_boundary_v1','passed':True,'condition':K.CONDITION,'seed':22,'fold':5,
           'checkpoint':{'path':(K.OUT/'runs'/fit['fit_id']/'latest.pt').relative_to(K.WORK).as_posix(),'sha256':'synthetic'},
           'registration':{'path':'q4.json','sha256':'synthetic'},'original_training_registration':{'path':'original.json','sha256':'synthetic'},
           'train_history':{'path':'history.json','sha256':'synthetic'}}
    proof['experimental_sources']=[K.descriptor(path) for path in (K.SOURCE,K.PROTOTYPE,K.GATE_SOURCE)]
    return proof,{'folds':{'fold_5':split}},fit,[{'epoch':1}],{'synthetic_gate_only':True}


def measurements():
    result=[]
    for plan in K.matrix_plan():
        row={**plan,'synchronized_wall_seconds':2. if plan['variant']=='original' else 1.}
        for key in ('ordered_logits_sha256','ordered_ce_sha256','pre_clip_gradients_sha256',
                    'post_clip_gradients_sha256','clip_norm_sha256','post_rng_sha256',
                    'parameter_buffer_state_sha256','optimizer_state_sha256'):
            row[key]='synthetic_same_'+key+str(plan['epoch_phase'])
        row.update(result={'subject_ce_sum':3.,'subjects':8,'backward_calls':8},module_modes=[('head',True)],
                   active_parameter_names=['synthetic'],operation_delta={'backward':8,'optimizer_step':0})
        result.append(row)
    return result


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='training_sync_gate_fixture_',dir=REPORT)
        self.root=Path(self.tmp.name)
        self.patch=patch.multiple(K,OUT=self.root/'out',OWNER=self.root/'owner.json',REPORT=self.root)
        self.patch.start()

    def tearDown(self):self.patch.stop();self.tmp.cleanup()

    def test_unflagged_pending_never_imports_tensor_or_calls_gate(self):
        with patch.object(K,'gate_module',side_effect=AssertionError('gate reached')),patch.object(K,'qualify',side_effect=AssertionError('run reached')):
            self.assertEqual(K.main([]),0)
        files=list(self.root.glob('training_sync_qualification_pending_*.json'))
        self.assertEqual(len(files),1)
        value=json.loads(files[0].read_text())
        self.assertFalse(value['passed']);self.assertEqual(value['GPU'],0);self.assertEqual(value['checkpoint_reads'],0)
        self.assertEqual(value['planned_engineering_budget']['backward'],96)

    def test_explicit_run_needs_exact_boundary_path_and_hash(self):
        with patch.object(K,'qualify',side_effect=AssertionError('run reached')):
            with self.assertRaises(ValueError):K.main(['--run'])
            boundary=self.root/'boundary.json';boundary.write_text('{}')
            with self.assertRaises(ValueError):K.main(['--run','--boundary-proof',str(boundary),'--boundary-sha','0'*64])

    def test_failed_gate_creates_zero_work_failed_trace_without_owner(self):
        gate=types.SimpleNamespace(OWNER=K.OWNER,preflight=lambda binding:(_ for _ in ()).throw(ValueError('not suspended')))
        with patch.object(K,'gate_module',return_value=gate),patch.object(K,'run_matrix',side_effect=AssertionError('GPU matrix reached')):
            with self.assertRaisesRegex(ValueError,'not suspended'):K.qualify({'path':'synthetic.json','sha256':'synthetic'})
        traces=list(K.OUT.glob('supervised_engineering/*/*.json'));self.assertEqual(len(traces),1)
        data=json.loads(traces[0].read_text())
        self.assertEqual(data['status'],'failed');self.assertEqual(data['backward_returned'],0)
        self.assertFalse(data['gpu_stage_entered']);self.assertFalse(K.OWNER.exists())

    def test_existing_foreign_lease_is_never_removed(self):
        K.OWNER.write_text(json.dumps({'id':'foreign'}))
        gate=types.SimpleNamespace(OWNER=K.OWNER,preflight=lambda binding:fixture())
        with patch.object(K,'gate_module',return_value=gate),patch.object(K,'run_matrix',side_effect=AssertionError('GPU matrix reached')):
            with self.assertRaises(FileExistsError):K.qualify({'path':'synthetic.json','sha256':'synthetic'})
        self.assertEqual(json.loads(K.OWNER.read_text()),{'id':'foreign'})

    def test_boundary_change_after_lease_stops_before_tensor_stage(self):
        values=fixture();modified=copy.deepcopy(values);modified[0]['extra_mutation']=True
        calls=iter([values,modified]);gate=types.SimpleNamespace(OWNER=K.OWNER,preflight=lambda binding:next(calls))
        with patch.object(K,'gate_module',return_value=gate),patch.object(K,'run_matrix',side_effect=AssertionError('GPU matrix reached')):
            with self.assertRaisesRegex(ValueError,'Boundary changed'):K.qualify({'path':'synthetic.json','sha256':'synthetic'})
        self.assertFalse(K.OWNER.exists())
        data=json.loads(next(K.OUT.glob('supervised_engineering/*/*.json')).read_text())
        self.assertFalse(data['gpu_stage_entered']);self.assertEqual(data['model_constructions_started'],0)

    def test_strict_identity_and_train_roles(self):
        p,r,f,h,_=fixture();people,split=K.validate_fixed_identity(p,r,f,h)
        self.assertEqual(people,split['train'][:8])
        for altered in ({**p,'fold':True},{**p,'condition':'REVE-lastblock-finetune'},
                        {**p,'checkpoint':{'path':'wrong/latest.pt','sha256':'synthetic'}}):
            with self.assertRaises(ValueError):K.validate_fixed_identity(altered,r,f,h)
        changed=copy.deepcopy(f);changed['train_ids'][0]=changed['test_ids'][0]
        with self.assertRaises(ValueError):K.validate_fixed_identity(p,r,changed,h)


class PlanTests(unittest.TestCase):
    def test_fixed_96_backward_budget_and_alternating_order(self):
        plan=K.matrix_plan();self.assertEqual(len(plan),12)
        self.assertEqual(sum(8 for _ in plan),96);self.assertEqual(sum(64 for _ in plan),768)
        for phase in (1,11):
            rows=[r for r in plan if r['epoch_phase']==phase]
            self.assertEqual(len(rows),6);self.assertEqual(sum(r['warmup'] for r in rows),2)
            self.assertNotEqual(rows[0]['variant'],rows[2]['variant'])
            self.assertNotEqual(rows[2]['variant'],rows[4]['variant'])
        self.assertEqual(K.FIXED_COUNTS['optimizer_step'],0)
        self.assertEqual(K.FIXED_COUNTS['optimizer_step_stub'],12)

    def test_equality_and_faster_must_hold_in_both_phases(self):
        self.assertTrue(K.summarize(measurements())['passed'])
        for field in ('pre_clip_gradients_sha256','post_clip_gradients_sha256','post_rng_sha256',
                      'ordered_ce_sha256','optimizer_state_sha256','parameter_buffer_state_sha256','result'):
            rows=measurements();rows[0][field]='changed'
            with self.subTest(field=field):self.assertFalse(K.summarize(rows)['passed'])
        rows=measurements()
        for row in rows:
            if row['epoch_phase']==11 and row['variant']=='candidate':row['synchronized_wall_seconds']=3.
        self.assertFalse(K.summarize(rows)['passed'])

    def test_duplicate_or_partial_matrix_never_qualifies(self):
        rows=measurements()
        with self.assertRaises(ValueError):K.summarize(rows[:-1])
        rows[-1]=rows[0]
        with self.assertRaises(ValueError):K.summarize(rows)

    def test_invalid_timing_does_not_become_speedup(self):
        for value in (True,0.,-1.,float('nan'),float('inf')):
            rows=measurements();rows[0]['synchronized_wall_seconds']=value
            with self.subTest(value=value),self.assertRaises(ValueError):K.summarize(rows)


class CpuHelperTests(unittest.TestCase):
    def test_actual_matrix_core_on_toy_cpu_uses_96_backwards_and_zero_steps(self):
        import numpy as np
        import torch
        sys.path.insert(0,str(K.WORK))
        from tridep_ccml.eeg_all_candidates import finetune as F
        from tridep_ccml.eeg_all_candidates import sync_subject_update_v1 as C
        torch.set_num_threads(2);torch.default_generator.manual_seed(541)
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weights=torch.nn.ParameterList([torch.nn.Parameter(torch.tensor([.01*(i+1)])) for i in range(20)])
            def forward(self,x):
                base=x.mean(dim=(1,2));weight=sum(self.weights)
                return torch.stack((base*weight,base+weight),dim=1)
        class Toy(F.RawSubjectModel):
            def __init__(self):
                encoder=Encoder()
                adapter=types.SimpleNamespace(__name__='synthetic_cpu_adapter',adapt_raw=lambda raw:raw,
                    encode_windows=lambda encoder,tensor:encoder(tensor))
                super().__init__(encoder,F.SubjectBagHead(2),adapter,[n for n,_ in encoder.named_parameters()],5,8)
        model=Toy();CPU_TEST_COUNTS['synthetic_models']+=1;optimizer=F.optimizer_for(model)
        before_model=F.cpu_state(model.state_dict())
        original_rng=F.rng_state('cpu')
        p,r,f,h,_=fixture();people,split=K.validate_fixed_identity(p,r,f,h)
        bindings={sid:{'path':'synthetic/'+sid,'sha256':'not-real-source'} for sid in split['train']+split['val']}
        raw={sid:np.full((60,19,1000),.05+i*.01,dtype=np.float32) for i,sid in enumerate(people)}
        r['labels']={sid:i%2 for i,sid in enumerate(split['train']+split['val']+split['test'])}
        lock={'path':'synthetic-lock','sha256':'not-real-source'}
        saved={'condition':K.CONDITION,'seed':22,'fold':5,'registry':p['original_training_registration'],
            'condition_source_lock':lock,'history':h,'epoch':len(h),'raw_sources':bindings,
            'state_dict':F.cpu_state(model.state_dict()),'optimizer_state_dict':F.cpu_state(optimizer.state_dict()),'rng':original_rng}
        fakeF=types.SimpleNamespace(**{name:getattr(F,name) for name in ('subject_update','optimizer_for','set_train_phase',
            'set_learning_rates','restore_rng','rng_equal','bag_indices','POLICY')})
        fakeF.C=types.SimpleNamespace(runtime=lambda seed:None)
        fakeF.rng_state=lambda device:F.rng_state('cpu')
        fakeF._raw_sources=lambda split,people:(raw,{sid:bindings[sid] for sid in people})
        fakeF2=types.SimpleNamespace(_ACTIVE=False,_TRACE=None,build_model=lambda *a,**kw:(model,{'synthetic':True}),_check_v2_lock=lambda condition:lock)
        trace={kind+'_'+state:0 for kind in K.FIXED_COUNTS for state in ('started','returned')}
        trace.update(actual_runtime={},model_constructions_started=0,model_constructions_returned=0,
            latest_checkpoint_load_started=0,latest_checkpoint_load_returned=0,forbidden_optimizer_attempts=0)
        snapshots=[]
        q=types.SimpleNamespace(checked=lambda descriptor,**kw:Path('SYNTHETIC_NO_FILE_READ'),
                              atomic_json=lambda path,value,**kw:snapshots.append(copy.deepcopy(value)))
        class CPUEvent:
            def __init__(self,*args,**kwargs):pass
            def record(self):self.t=time.perf_counter()
            def elapsed_time(self,other):return (other.t-self.t)*1000
        originals=(model.forward,model.adapter.encode_windows,torch.Tensor.backward,torch.nn.functional.cross_entropy,torch.nn.utils.clip_grad_norm_,torch.optim.AdamW.step)
        actual_backward=torch.Tensor.backward
        def count_cpu_backward(tensor,*args,**kwargs):
            CPU_TEST_COUNTS['synthetic_backward_started']+=1
            result=actual_backward(tensor,*args,**kwargs)
            CPU_TEST_COUNTS['synthetic_backward_returned']+=1
            return result
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('Actual GPU forbidden')))
            stack.enter_context(patch.object(torch.cuda,'Event',CPUEvent))
            stack.enter_context(patch.object(torch.cuda,'synchronize',lambda:None))
            stack.enter_context(patch.object(torch.cuda,'reset_peak_memory_stats',lambda:None))
            stack.enter_context(patch.object(torch.cuda,'max_memory_allocated',lambda:0))
            stack.enter_context(patch.object(torch,'load',return_value=saved))
            stack.enter_context(patch.object(torch.Tensor,'backward',count_cpu_backward))
            rows=K.run_matrix(torch,fakeF,fakeF2,C,p,r,f,h,people,split,trace,Path('NO_WRITE'),q)
        self.assertEqual(len(rows),12)
        self.assertTrue(K.summarize(rows)['bitwise_qualified'])
        for kind,expected in K.FIXED_COUNTS.items():
            self.assertEqual(trace[kind+'_started'],expected);self.assertEqual(trace[kind+'_returned'],expected)
        self.assertTrue(F.rng_equal(original_rng,F.rng_state('cpu')))
        self.assertEqual(K.fingerprint(before_model,torch),K.fingerprint(model.state_dict(),torch))
        self.assertEqual(optimizer.state,{})
        after=(model.forward,model.adapter.encode_windows,torch.Tensor.backward,torch.nn.functional.cross_entropy,torch.nn.utils.clip_grad_norm_,torch.optim.AdamW.step)
        self.assertEqual(originals,after);self.assertTrue(trace['external_rng_restored'])
        self.assertEqual(trace['forbidden_optimizer_attempts'],0)

    def test_exact_byte_fingerprint_distinguishes_signed_zero_and_dtype(self):
        import torch
        with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU forbidden')):
            a=torch.tensor([0.]);b=torch.tensor([-0.])
            self.assertNotEqual(K.fingerprint(a,torch),K.fingerprint(b,torch))
            self.assertNotEqual(K.fingerprint(a,torch),K.fingerprint(a.double(),torch))
            self.assertEqual(K.fingerprint({'x':a,'step':1},torch),K.fingerprint({'step':1,'x':a.clone()},torch))

    def test_all_optimizer_guards_restore_and_proxy_performs_no_update(self):
        import torch
        with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU forbidden')):
            p=torch.nn.Parameter(torch.tensor([2.]))
            optimizer=torch.optim.AdamW([p],lr=.1);p.grad=torch.ones_like(p)
            trace={'forbidden_optimizer_attempts':0,'optimizer_step_stub_started':0,'optimizer_step_stub_returned':0}
            original_adam,original_sgd=torch.optim.AdamW.step,torch.optim.SGD.step
            with K.forbid_optimizer_steps(torch,trace):
                with self.assertRaises(RuntimeError):optimizer.step()
                with self.assertRaises(RuntimeError):torch.optim.SGD.step(None)
                proxy=K.ZeroStepProxy(optimizer,trace);proxy.zero_grad(set_to_none=True);proxy.step()
            self.assertIs(torch.optim.AdamW.step,original_adam);self.assertIs(torch.optim.SGD.step,original_sgd)
            self.assertTrue(torch.equal(p,torch.tensor([2.])));self.assertEqual(optimizer.state,{})
            self.assertEqual(trace['forbidden_optimizer_attempts'],2);self.assertEqual(trace['optimizer_step_stub_returned'],1)


if __name__=='__main__':
    started=time.perf_counter();stream=io.StringIO()
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in (GateTests,PlanTests,CpuHelperTests))
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    record={'schema':'training_sync_qualifier_preflight_tests_v1','passed':result.wasSuccessful(),
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'seconds':time.perf_counter()-started,'output':stream.getvalue(),
        'sources':[K.descriptor(Path(__file__)),K.descriptor(K.SOURCE),K.descriptor(K.PROTOTYPE)],
        'scope':'Synthetic gate/budget/comparison fixtures, tiny CPU tensor byte/step guards, and actual runner matrix core on synthetic CPU tensors with fake CUDA timing/peak methods and in-memory fake checkpoint; no actual GPU qualification executed',
        'gpu_calls':0,'actual_checkpoint_loads':0,'cohort_reads':0,'actual_optimizer_updates':0,'actual_cpu_test_counts':CPU_TEST_COUNTS,
        'qualified_or_formal_execution_certificate':False}
    target=REPORT/('training_sync_qualifier_preflight_tests_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'passed':record['passed'],'tests_run':result.testsRun,'evidence':K.descriptor(target),
                      'output':'all passed' if result.wasSuccessful() else stream.getvalue()}))
    raise SystemExit(0 if result.wasSuccessful() else 1)

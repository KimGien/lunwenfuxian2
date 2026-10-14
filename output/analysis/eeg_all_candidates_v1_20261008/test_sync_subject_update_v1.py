"""Actual CPU toy-tensor regressions; no cohort, checkpoint, GPU or live patch."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
import numpy as np
import torch
from tridep_ccml.eeg_all_candidates import finetune as F
from tridep_ccml.eeg_all_candidates import sync_subject_update_v1 as C

TOY_ACTUAL_OPTIMIZER_STEPS = 0


def seed(value=914):
    torch.default_generator.manual_seed(value)


class Tiny(torch.nn.Module):
    def __init__(self, events=None):
        super().__init__()
        self.encoder = torch.nn.Linear(1, 3)
        self.head = torch.nn.Linear(3, 2)
        self.dropout = torch.nn.Dropout(.25)
        self.frozen = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
        self.window_chunk = 8
        self.events = events if events is not None else []
        self.bad_representation = False
        self.bad_logits = False

    @property
    def device(self): return torch.device('cpu')

    def forward(self, raw):
        self.events.append(('forward', float(raw[0,0,0])))
        x = torch.as_tensor(raw, device='cpu').mean(dim=1)
        x = self.encoder(x)
        if self.bad_representation: x = x * float('nan')
        F.require(torch.isfinite(x).all(), 'Encoder returned nonfinite representations')
        out = self.head(self.dropout(x))
        return out * float('nan') if self.bad_logits else out


class Adam(torch.optim.AdamW):
    def __init__(self, model, events):
        super().__init__([{'params':list(model.head.parameters()),'lr':.001},
                          {'params':list(model.encoder.parameters()),'lr':.0002}],weight_decay=.01)
        self.events = events

    def zero_grad(self, *args, **kwargs):
        self.events.append(('zero',kwargs))
        return super().zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        global TOY_ACTUAL_OPTIMIZER_STEPS
        TOY_ACTUAL_OPTIMIZER_STEPS += 1
        self.events.append(('step',))
        return super().step(*args, **kwargs)


class NoStep:
    """Qualification-shaped proxy: true zeroing, no optimizer update."""
    def __init__(self, optimizer): self.optimizer=optimizer; self.reached=0
    def zero_grad(self, *a, **kw): return self.optimizer.zero_grad(*a, **kw)
    def step(self): self.reached += 1


def inputs(n):
    people = ['toy'+str(i) for i in range(n)]
    raw = {sid:(np.arange(8,dtype=np.float32)[:,None] * .1 + i) for i,sid in enumerate(people)}
    return people,raw,{sid:i%2 for i,sid in enumerate(people)},np.arange(8)


def equal_tree(test, a, b):
    if isinstance(a,torch.Tensor):
        test.assertEqual(a.dtype,b.dtype);test.assertEqual(a.shape,b.shape);test.assertTrue(torch.equal(a,b))
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a: equal_tree(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b): equal_tree(test,x,y)
    else: test.assertEqual(a,b)


class SyncSubjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cuda_guard = patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('GPU is forbidden'))
        cls.cuda_guard.start()

    @classmethod
    def tearDownClass(cls): cls.cuda_guard.stop()

    def setUp(self):
        seed()
        self.candidate=C.make_subject_update(F.subject_update)

    def compare(self, n, warmup=False, repeats=1, no_step=False):
        a=Tiny();b=copy.deepcopy(a)
        if warmup:
            for model in (a,b):
                for p in model.encoder.parameters(): p.requires_grad_(False)
        a.events=[];b.events=[]
        oa=Adam(a,a.events);ob=Adam(b,b.events)
        ga=NoStep(oa) if no_step else oa;gb=NoStep(ob) if no_step else ob
        people,raw,labels,windows=inputs(n)
        original_backward=torch.Tensor.backward
        original_clip=torch.nn.utils.clip_grad_norm_
        def run(fn,model,opt):
            def backward(t,*args,**kwargs):
                model.events.append(('backward',));return original_backward(t,*args,**kwargs)
            def clip(parameters,*args,**kwargs):
                parameters=list(parameters)
                model.events.append(('preclip',tuple(p.grad.detach().clone() for p in parameters)))
                norm=original_clip(parameters,*args,**kwargs)
                model.events.append(('postclip',tuple(p.grad.detach().clone() for p in parameters)))
                return norm
            with patch.object(torch.Tensor,'backward',backward),patch.object(torch.nn.utils,'clip_grad_norm_',clip):
                return fn(model,opt,people,raw,labels,windows,micro_subjects=1)
        for _ in range(repeats):
            rng=torch.get_rng_state().clone()
            ra=run(F.subject_update,a,ga);rng_a=torch.get_rng_state().clone()
            torch.set_rng_state(rng)
            rb=run(self.candidate,b,gb);rng_b=torch.get_rng_state().clone()
            self.assertEqual(ra,rb);self.assertTrue(torch.equal(rng_a,rng_b))
            equal_tree(self,a.state_dict(),b.state_dict())
            equal_tree(self,oa.state_dict(),ob.state_dict())
            equal_tree(self,a.events,b.events)
            for pa,pb in zip(a.parameters(),b.parameters()):
                self.assertEqual(pa.grad is None,pb.grad is None)
                if pa.grad is not None:self.assertTrue(torch.equal(pa.grad,pb.grad))
        if no_step:
            self.assertEqual(ga.reached,repeats);self.assertEqual(gb.reached,repeats)
            self.assertEqual(oa.state,{});self.assertEqual(ob.state,{})
        return a,b

    def test_import_prototype_has_no_tensor_dependency(self):
        cmd=[sys.executable,'-B','-c',
             'import sys; from tridep_ccml.eeg_all_candidates import sync_subject_update_v1; assert "torch" not in sys.modules and "numpy" not in sys.modules']
        result=subprocess.run(cmd,cwd=WORK,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_source_factory_does_not_patch_original_or_forward(self):
        original=F.subject_update;forward=F.RawSubjectModel.forward
        candidate=C.make_subject_update(original)
        self.assertIs(F.subject_update,original);self.assertIs(F.RawSubjectModel.forward,forward)
        self.assertFalse(candidate.sync_candidate_provenance['gpu_qualified'])
        with self.assertRaises(ValueError):C.make_subject_update(lambda *a,**kw:None)
        with patch.object(C,'ORIGINAL_SHA256','0'*64),self.assertRaises(ValueError):C.make_subject_update(original)

    def test_full_batch_dropout_adamw_and_two_updates_are_bitwise(self):self.compare(8,repeats=2)
    def test_warmup_frozen_encoder_is_bitwise(self):self.compare(8,warmup=True,repeats=2)
    def test_remainders_keep_denominator_and_order(self):
        for n in (1,3,7):
            with self.subTest(n=n):self.compare(n)
    def test_qualification_shaped_probe_has_zero_optimizer_steps(self):
        before=TOY_ACTUAL_OPTIMIZER_STEPS
        self.compare(8,no_step=True)
        self.assertEqual(before,TOY_ACTUAL_OPTIMIZER_STEPS)

    def test_successful_host_observations_match_static_formula(self):
        def run(fn):
            seed();m=Tiny();o=Adam(m,m.events);counts={'cpu':0,'bool':0}
            cpu=torch.Tensor.cpu;boolean=torch.Tensor.__bool__
            def count_cpu(t,*a,**kw):counts['cpu']+=1;return cpu(t,*a,**kw)
            def count_bool(t):counts['bool']+=1;return boolean(t)
            with patch.object(torch.Tensor,'cpu',count_cpu),patch.object(torch.Tensor,'__bool__',count_bool):
                fn(m,o,*inputs(8),micro_subjects=1)
            return counts
        baseline=run(F.subject_update);candidate=run(self.candidate)
        self.assertEqual(baseline,{'cpu':8,'bool':20})
        self.assertEqual(candidate,{'cpu':1,'bool':16})
        costs=C.explicit_host_observation_counts(8,4)
        self.assertEqual(sum(baseline.values()),costs['original'])
        self.assertEqual(sum(candidate.values()),costs['candidate'])
        self.assertIsNone(costs['elapsed_seconds_saved'])

    def test_ce_is_python_sequential_sum_not_float32_reduction(self):
        def run(fn):
            seed();m=Tiny();o=NoStep(Adam(m,m.events));losses=iter([float(2**24)]+[1.]*7)
            def loss(logits,*a,**kw):return logits.sum()*0+next(losses)
            with patch.object(torch.nn.functional,'cross_entropy',loss):
                return fn(m,o,*inputs(8),micro_subjects=1)
        expected=run(F.subject_update);actual=run(self.candidate)
        self.assertEqual(expected,actual);self.assertEqual(actual['subject_ce_sum'],float(2**24+7))

    def check_failure(self, customize, match, expected_backwards):
        for fn in (F.subject_update,self.candidate):
            seed();m=Tiny();o=Adam(m,m.events);customize(m)
            before=copy.deepcopy(m.state_dict());calls=[]
            backward=torch.Tensor.backward
            def observe(t,*a,**kw):calls.append(1);return backward(t,*a,**kw)
            with patch.object(torch.Tensor,'backward',observe),patch.object(torch.nn.utils,'clip_grad_norm_',side_effect=AssertionError('clip after invalid')):
                with self.assertRaisesRegex(ValueError,match):fn(m,o,*inputs(3),micro_subjects=1)
            self.assertEqual(len(calls),expected_backwards);self.assertEqual(o.state,{})
            self.assertFalse(any(x[0]=='step' for x in m.events));equal_tree(self,m.state_dict(),before)

    def test_nonfinite_representation_still_stops_before_backward(self):
        self.check_failure(lambda m:setattr(m,'bad_representation',True),'representations',0)
    def test_nonfinite_ce_still_stops_before_backward(self):
        self.check_failure(lambda m:setattr(m,'bad_logits',True),'subject CE',0)
    def test_nonfinite_gradient_never_clips_or_steps(self):
        self.check_failure(lambda m:m.head.weight.register_hook(lambda g:g*float('nan')),'Nonfinite',3)
    def test_disconnected_parameter_never_clips_or_steps(self):
        self.check_failure(lambda m:m.register_parameter('unused',torch.nn.Parameter(torch.ones(2))),'disconnected',3)
    def test_frozen_gradient_never_clips_or_steps(self):
        def change(m):m.head.weight.register_hook(lambda g:(setattr(m.frozen,'grad',torch.ones_like(m.frozen)) or g))
        self.check_failure(change,'Frozen parameter',3)

    def test_copy_failure_occurs_after_backward_but_never_steps(self):
        m=Tiny();o=Adam(m,m.events);calls=[];backward=torch.Tensor.backward
        def observe(t,*a,**kw):calls.append(1);return backward(t,*a,**kw)
        with patch.object(torch.Tensor,'backward',observe),patch.object(torch.Tensor,'cpu',side_effect=RuntimeError('copy failed')):
            with self.assertRaisesRegex(RuntimeError,'copy failed'):self.candidate(m,o,*inputs(3),micro_subjects=1)
        self.assertEqual(len(calls),3);self.assertEqual(o.state,{})
        self.assertFalse(any(x[0]=='step' for x in m.events))

    def test_nonmicro1_or_non8chunk_rejected_before_zeroing(self):
        for micro,chunk in ((2,8),(1,16)):
            m=Tiny();m.window_chunk=chunk;o=Adam(m,m.events)
            with self.assertRaises(ValueError):self.candidate(m,o,*inputs(3),micro_subjects=micro)
            self.assertEqual(m.events,[])


def desc(path):
    path=Path(path).resolve()
    return {'path':path.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__=='__main__':
    started=time.perf_counter();stream=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(SyncSubjectTests)
    names=[case.id() for case in suite]
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    record={'schema':'sync_subject_update_cpu_prototype_tests_v1','passed':result.wasSuccessful(),
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'test_names':names,'output':stream.getvalue(),'seconds':time.perf_counter()-started,
        'sources':[desc(__file__),desc(C.SOURCE),desc(C.ORIGINAL_SOURCE)],'synthetic_cpu_only':True,
        'actual_synthetic_cpu_optimizer_steps':TOY_ACTUAL_OPTIMIZER_STEPS,
        'gpu_calls':0,'formal_optimizer_updates':0,'cohort_reads':0,'actual_checkpoint_loads':0,
        'formal_queue_or_execution_registration_changed':False,'gpu_qualification_passed':False}
    path=REPORT/('sync_subject_update_cpu_tests_'+str(time.time_ns())+'.json')
    with path.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'passed':record['passed'],'tests_run':result.testsRun,'evidence':desc(path),
                      'output':'all passed' if result.wasSuccessful() else stream.getvalue()},ensure_ascii=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)

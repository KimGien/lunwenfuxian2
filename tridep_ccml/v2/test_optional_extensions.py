import io
import unittest
import numpy as np
from scipy.special import digamma
import torch
from . import optional_extensions as x
from .common import set_seed, LOSS
from tridep_ccml.ccml import ccml_loss,evidence_dc
from tridep_ccml.train import state_digest


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        set_seed(11)
        self.inputs={m:torch.randn(4,d) for m,d in x.s.DIMENSIONS.items()}
        self.y=torch.tensor([0,1,0,1])

    def test_both_unit_initialized_paths_equal_untuned_control(self):
        control=x.s.ScaleModel(11,False)
        for kind,count in [('G',223375),('F',223366)]:
            model=x.ExtensionModel(11,kind)
            self.assertEqual(x.base_hash(model),state_digest(control.state_dict()))
            self.assertEqual(sum(p.numel() for p in model.parameters()),count)
            self.assertTrue(torch.equal(model(self.inputs),control(self.inputs)))

    def test_gate_quality_uses_no_gradient_and_is_class_symmetric(self):
        raw=torch.rand(4,3,2,requires_grad=True)*100
        q=x.quality_features(raw)
        self.assertFalse(q.requires_grad)
        self.assertTrue(torch.all(q>=0) and torch.all(q<=1))
        self.assertTrue(torch.equal(q,x.quality_features(raw.flip(-1))))

    def test_dynamic_bounds_identity_and_subject_dependence(self):
        model=x.ExtensionModel(11,'G')
        with torch.no_grad():model.gate_weights.copy_(torch.tensor([[10.,-5.],[-4.,10.],[2.,1.]]))
        _,logs,_=model.components(self.inputs);scales=logs.exp()
        self.assertTrue(torch.all(scales>=.5) and torch.all(scales<=2))
        self.assertGreater(float(scales.std(0).sum().detach()),0)

    def test_gates_do_not_mix_subjects_or_modalities(self):
        model=x.ExtensionModel(11,'G')
        with torch.no_grad():model.gate_weights.fill_(.7)
        a=model(self.inputs)
        changed={m:v.clone() for m,v in self.inputs.items()};changed['speech'][1:]*=100
        b=model(changed)
        self.assertTrue(torch.equal(a[0],b[0]))
        self.assertTrue(torch.equal(a[:,[0,2]],b[:,[0,2]]))

    def test_final_supervision_matches_independent_scipy_formula(self):
        e=torch.rand(4,3,2,dtype=torch.float64)
        logs=torch.zeros(4,3,dtype=torch.float64)
        total,extra=x.objective(e,logs,self.y,20,'F',.1)
        alpha=evidence_dc(e)['alpha'].numpy()
        expected=np.mean(digamma(alpha.sum(-1))-digamma(alpha[np.arange(4),self.y.numpy()]))
        self.assertAlmostEqual(float(extra),.1*expected,places=12)
        self.assertAlmostEqual(float(total-ccml_loss(e,self.y,20,**LOSS)),float(extra),places=12)

    def test_zero_beta_matches_control_value_and_gradient(self):
        e1=torch.rand(4,3,2,dtype=torch.float64,requires_grad=True)
        e2=e1.detach().clone().requires_grad_()
        baseline=ccml_loss(e1,self.y,15,**LOSS)
        candidate,_=x.objective(e2,torch.zeros(4,3),self.y,15,'F',0.)
        baseline.backward();candidate.backward()
        self.assertTrue(torch.equal(baseline,candidate))
        self.assertTrue(torch.equal(e1.grad,e2.grad))

    def test_extra_losses_have_finite_model_and_gate_gradients(self):
        for kind,value in [('G',.1),('F',1.)]:
            model=x.ExtensionModel(11,kind)
            raw,logs,_=model.components(self.inputs)
            loss,_=x.objective(raw*logs.exp()[...,None],logs,self.y,30,kind,value)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            if kind=='G':self.assertGreater(float(model.gate_weights.grad.norm()),0)

    def test_gate_shrinkage_points_to_zero(self):
        model=x.ExtensionModel(11,'G')
        with torch.no_grad():model.gate_bias.copy_(torch.tensor([-.4,0.,.4]))
        _,logs,_=model.components(self.inputs)
        logs.square().mean().backward()
        self.assertLess(model.gate_bias.grad[0],0)
        self.assertEqual(model.gate_bias.grad[1],0)
        self.assertGreater(model.gate_bias.grad[2],0)

    def test_checkpoint_roundtrip_is_exact(self):
        for kind in ['G','F']:
            model=x.ExtensionModel(11,kind);stream=io.BytesIO()
            torch.save(model.state_dict(),stream);stream.seek(0)
            other=x.ExtensionModel(22,kind);other.load_state_dict(torch.load(stream,weights_only=True),strict=True)
            self.assertTrue(torch.equal(model(self.inputs),other(self.inputs)))

    def candidates(self,kind):
        ids=list('abcdef')
        rows=[{'subject_id':sid,'inner_fold':i//2+1,'split':'inner_val','label':i%2,
               'p_final':[.7,.3] if i%2==0 else [.3,.7],
               'train_ids':[s for j,s in enumerate(ids) if j//2!=i//2]} for i,sid in enumerate(ids)]
        return ids,{value:[dict(r) for r in rows] for value in x.VALUES[kind]}

    def test_selection_tie_rules_and_nll(self):
        for kind in ['G','F']:
            ids,rows=self.candidates(kind)
            self.assertEqual(x.select_candidate(kind,rows,ids)['selected_value'],.1)
            v=x.VALUES[kind][0]
            for row in rows[v]:row['p_final']=[.9,.1] if row['label']==0 else [.1,.9]
            self.assertEqual(x.select_candidate(kind,rows,ids)['selected_value'],v)

    def test_selection_rejects_illegal_split_in_sample_or_missing_people(self):
        for kind in ['G','F']:
            for mutation in ['test','train','missing']:
                ids,rows=self.candidates(kind);first=rows[x.VALUES[kind][0]]
                if mutation=='missing':first.pop()
                else:first[0]['split']=mutation
                with self.assertRaises(ValueError):x.select_candidate(kind,rows,ids)


if __name__=='__main__':unittest.main()

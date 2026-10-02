import io
import unittest

import numpy as np
import torch
from torch import nn

from .scale import (ScaleModel, DIMENSIONS, LAMBDAS, require_label_free,
                    nested_splits, select_candidate)
from .common import RecordContext, set_seed, LOSS
from tridep_ccml.ccml import ccml_loss, evidence_dc
from tridep_ccml.train import state_digest


class ScaleTests(unittest.TestCase):
    def setUp(self):
        set_seed(11)
        self.inputs = {m: torch.randn(4, d) for m, d in DIMENSIONS.items()}

    def test_unit_scales_exactly_recover_control_and_initial_weights(self):
        a, b = ScaleModel(11, False), ScaleModel(11, True)
        self.assertEqual(state_digest(a.state_dict()), state_digest(b.state_dict()))
        self.assertTrue(torch.equal(a(self.inputs), b(self.inputs)))
        self.assertTrue(torch.equal(b(self.inputs), b.raw_evidence(self.inputs)))
        self.assertEqual(sum(p.numel() for p in a.parameters()), 223366)
        self.assertEqual(sum(p.numel() for p in b.parameters()), 223369)

    def test_bounds_and_shrinkage_toward_identity(self):
        model = ScaleModel(11)
        with torch.no_grad():
            model.scale_raw.copy_(torch.tensor([-100., 0., 100.]))
        self.assertTrue(torch.all(model.scales() >= .5) and torch.all(model.scales() <= 2.))
        with torch.no_grad():
            model.scale_raw.copy_(torch.tensor([-.4, 0., .4]))
        model.penalty().backward()
        self.assertLess(model.scale_raw.grad[0], 0)
        self.assertEqual(model.scale_raw.grad[1], 0)
        self.assertGreater(model.scale_raw.grad[2], 0)

    def test_common_scale_homogeneity_and_unchanged_argmax(self):
        e = torch.tensor([[[.3, 2.], [3., 4.], [.7, 1.]], [[2., .3], [4., 3.], [1., .7]]], dtype=torch.float64)
        original = evidence_dc(e)
        for s in [.5, 1.7, 2.]:
            after = evidence_dc(s*e)
            self.assertTrue(torch.allclose(after['e_final'], s*original['e_final'], atol=1e-12))
            self.assertTrue(torch.equal(after['p'].argmax(-1), original['p'].argmax(-1)))
        self.assertTrue(torch.all(evidence_dc(2*e)['u'] < original['u']))

    def test_modality_scaling_recomputes_min_and_residual(self):
        e = torch.tensor([[[2., 5.], [3., 2.], [4., 3.]]], dtype=torch.float64)
        s = torch.tensor([.5, 2., 1.], dtype=torch.float64)
        scaled = (e*s[None, :, None]).numpy()
        minimum = scaled.min(1)
        con = 3*minimum.sum(-1, keepdims=True)*minimum**1.25/(minimum**1.25).sum(-1, keepdims=True)
        cmp = (scaled-minimum[:, None, :]).mean(1)
        result = evidence_dc(e*s[None, :, None])
        np.testing.assert_allclose(result['m'].numpy(), minimum)
        np.testing.assert_allclose(result['e_final'].numpy(), con+cmp, atol=1e-12)
        self.assertFalse(torch.equal(evidence_dc(e)['m'], result['m']))

    def test_no_subject_or_modality_mixing_before_ccml(self):
        model = ScaleModel(11)
        original = model(self.inputs)
        changed = {m: x.clone() for m, x in self.inputs.items()}
        changed['speech'][1:] *= 200
        after = model(changed)
        self.assertTrue(torch.equal(original[0], after[0]))
        self.assertTrue(torch.equal(original[:, [0, 2]], after[:, [0, 2]]))

    def test_full_objective_finite_updates_all_modules(self):
        model = ScaleModel(11)
        before = {m: state_digest(model.projections[m].state_dict()) for m in DIMENSIONS}
        opt = torch.optim.Adamax(model.parameters(), lr=.001)
        loss = ccml_loss(model(self.inputs), torch.tensor([0, 1, 0, 1]), 20, **LOSS)+.1*model.penalty()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.scale_raw.grad.norm()), 0)
        nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        for m in DIMENSIONS:
            self.assertNotEqual(before[m], state_digest(model.projections[m].state_dict()))
        self.assertFalse(torch.equal(model.scales(), torch.ones(3)))

    def test_control_has_no_trainable_scale(self):
        model = ScaleModel(11, False)
        self.assertNotIn('scale_raw', dict(model.named_parameters()))
        loss = ccml_loss(model(self.inputs), torch.tensor([0, 1, 0, 1]), 1, **LOSS)
        loss.backward()
        self.assertIsNone(model.scale_raw.grad)
        self.assertTrue(torch.equal(model.scales(), torch.ones(3)))

    def test_restricted_checkpoint_roundtrip(self):
        model = ScaleModel(11)
        with torch.no_grad():
            model.scale_raw.copy_(torch.tensor([-.2, .3, .1]))
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored = ScaleModel(22)
        restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
        self.assertTrue(torch.equal(model(self.inputs), restored(self.inputs)))

    def test_supervised_parent_rejected(self):
        require_label_free({'fit_subject_ids': [], 'labels_used': False})
        for row in [{'fit_subject_ids': ['subject']}, {'fit_subject_ids': [], 'labels_used': True}]:
            with self.assertRaises(ValueError):
                require_label_free(row)

    def test_nested_subjects_disjoint_full_coverage_fixed_budget(self):
        ctx = RecordContext()
        nested = nested_splits(ctx)
        self.assertEqual(nested, nested_splits(ctx))
        total = 0
        for f, parts in nested['folds'].items():
            outer = ctx.folds['fold_'+f]
            seen = []
            for part in parts:
                self.assertFalse(set(part['train_ids']) & set(part['val_ids']+outer['test']+outer['val']))
                self.assertEqual(set(part['train_ids']+part['val_ids']), set(outer['train']))
                seen += part['val_ids']
                total += part['updates']
            self.assertCountEqual(seen, outer['train'])
        self.assertEqual(total*6, 16200)

    def choices(self):
        ids = [str(i) for i in range(6)]
        candidates = {}
        for lam in LAMBDAS:
            candidates[lam] = [{'subject_id': sid, 'label': i%2, 'split': 'inner_val',
                'inner_fold': i//2+1, 'train_ids': [s for j, s in enumerate(ids) if j//2 != i//2],
                'p_final': [.7, .3] if i%2 == 0 else [.3, .7]} for i, sid in enumerate(ids)]
        return ids, candidates

    def test_selection_uses_nll_and_prefers_stronger_shrinkage_on_tie(self):
        ids, rows = self.choices()
        self.assertEqual(select_candidate(rows, ids)['selected_lambda'], .1)
        for row in rows[.01]:
            row['p_final'] = [.8, .2] if row['label'] == 0 else [.2, .8]
        self.assertEqual(select_candidate(rows, ids)['selected_lambda'], .01)

    def test_selection_rejects_outer_test_and_in_sample_records(self):
        for bad in ['test', 'val', 'train']:
            ids, rows = self.choices()
            rows[.1][0]['split'] = bad
            with self.assertRaises(ValueError):
                select_candidate(rows, ids)
        ids, rows = self.choices()
        rows[.1][0]['train_ids'].append(rows[.1][0]['subject_id'])
        with self.assertRaises(ValueError):
            select_candidate(rows, ids)

    def test_selection_rejects_missing_or_duplicate_inner_oof(self):
        ids, rows = self.choices()
        rows[.1][-1] = rows[.1][0]
        with self.assertRaises(ValueError):
            select_candidate(rows, ids)


if __name__ == '__main__':
    unittest.main()

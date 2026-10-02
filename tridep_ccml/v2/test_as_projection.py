import io
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from .as_projection import NormalizedProjectionHeads, DIMENSIONS, PROJECTION_COUNT, TOTAL_COUNT
from .nr import CandidateHeads
from .common import set_seed, LOSS
from tridep_ccml.ccml import ccml_loss
from tridep_ccml.train import make_initial_state, state_digest


class ASProjectionTests(unittest.TestCase):
    def setUp(self):
        set_seed(11)
        self.initial = {f'{m}.0.{k}': v for m, d in DIMENSIONS.items() for k, v in nn.Linear(d, 128).state_dict().items()}
        self.model = NormalizedProjectionHeads(self.initial, 11)
        self.x = {m: torch.randn(4, d) for m, d in DIMENSIONS.items()}

    def test_parameter_scope_and_supplied_projection_identity(self):
        self.assertEqual(PROJECTION_COUNT, 475520)
        self.assertEqual(TOTAL_COUNT, 476294)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), TOTAL_COUNT)
        self.assertEqual(state_digest(self.initial), state_digest(self.model.projections.state_dict()))
        self.assertEqual(state_digest(self.model.heads.state_dict()), state_digest(make_initial_state(11)))
        self.assertEqual(sum(p.numel() for p in self.model.normalizer.parameters()), 0)
        self.assertFalse(any(isinstance(m, nn.Dropout) for m in self.model.projections['speech'].modules()))
        self.assertTrue(all(not m.training for m in self.model.modules()))
        self.assertFalse(any(isinstance(m, (nn.Conv1d, nn.LSTM)) for m in self.model.modules()))

    def test_frozen_projection_function_equals_matching_AS_heads(self):
        for normalize, variant in [(True, 'N'), (False, 'C')]:
            model = NormalizedProjectionHeads(self.initial, 11, normalize=normalize)
            head = CandidateHeads(variant)
            head.load_state_dict(make_initial_state(11), strict=True)
            self.assertTrue(torch.equal(model(self.x), head(model.representations(self.x))))

    def test_normalization_and_model_do_not_mix_people_or_modalities(self):
        original = self.model(self.x)
        changed = {m: v.clone() for m, v in self.x.items()}
        changed['speech'][1:] *= 100
        after = self.model(changed)
        self.assertTrue(torch.equal(original[0], after[0]))
        self.assertTrue(torch.equal(original[:, [0, 2]], after[:, [0, 2]]))

    def test_full_ccml_updates_all_projections_with_frozen_inputs(self):
        before = {m: state_digest(self.model.projections[m].state_dict()) for m in DIMENSIONS}
        inputs = {m: v.clone() for m, v in self.x.items()}
        optimizer = torch.optim.Adamax([
            {'params': self.model.projections.parameters(), 'lr': 1e-4},
            {'params': self.model.heads.parameters(), 'lr': 1e-3}], weight_decay=0.)
        loss = ccml_loss(F.softplus(self.model(self.x)), torch.tensor([0, 1, 0, 1]), 1, **LOSS)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        for m in DIMENSIONS:
            self.assertNotEqual(before[m], state_digest(self.model.projections[m].state_dict()))
            self.assertTrue(torch.equal(inputs[m], self.x[m]))
            self.assertIsNone(self.x[m].grad)

    def test_restricted_reload_and_wrong_legacy_speech_dimensions(self):
        stream = io.BytesIO()
        torch.save({'state_dict': self.model.state_dict(), 'torch': str(torch.__version__)}, stream)
        stream.seek(0)
        restored = NormalizedProjectionHeads(self.initial, 22)
        restored.load_state_dict(torch.load(stream, weights_only=True)['state_dict'], strict=True)
        self.assertTrue(torch.equal(self.model(self.x), restored(self.x)))
        with self.assertRaises(RuntimeError):
            self.model({**self.x, 'speech': torch.zeros(4, 3584)})


if __name__ == '__main__':
    unittest.main()

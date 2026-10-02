import io
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from .np_combo import NormalizedProjectionHeads
from .p import ProjectionHeads, DIMENSIONS, MODALITIES
from .nr import CandidateHeads
from .test_p import synthetic_projection_state
from .common import set_seed, LOSS
from tridep_ccml.ccml import ccml_loss
from tridep_ccml.train import make_initial_state, state_digest


class CombinationTests(unittest.TestCase):
    def setUp(self):
        set_seed(11)
        self.initial = synthetic_projection_state()
        self.model = NormalizedProjectionHeads(self.initial, 11)
        self.x = {m: torch.randn(4, d) for m, d in DIMENSIONS.items()}

    def test_initial_weights_and_trainable_scope_match_P(self):
        plain = ProjectionHeads(self.initial, 11)
        self.assertEqual(state_digest(plain.state_dict()), state_digest(self.model.state_dict()))
        self.assertEqual(state_digest(self.model.heads.state_dict()), state_digest(make_initial_state(11)))
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 836742)
        self.assertEqual(sum(p.numel() for p in self.model.normalizer.parameters()), 0)
        self.assertFalse(any(m.training for m in self.model.modules() if isinstance(m, nn.Dropout)))
        self.assertFalse(any(isinstance(m, (nn.LSTM, nn.Conv1d)) for m in self.model.modules()))

    def test_frozen_projection_path_matches_N_head(self):
        candidate = CandidateHeads('N')
        candidate.load_state_dict(self.model.heads.state_dict(), strict=True)
        reps = self.model.representations(self.x)
        self.assertTrue(torch.equal(self.model(self.x), candidate(reps)))
        self.assertTrue(torch.equal(self.model(self.x), self.model(self.x)))

    def test_normalization_does_not_mix_subjects_or_modalities(self):
        before = self.model(self.x).detach()
        changed = {m: v.clone() for m, v in self.x.items()}
        changed['eeg'][1:] *= 100
        after = self.model(changed).detach()
        self.assertTrue(torch.equal(before[0], after[0]))
        self.assertTrue(torch.equal(before[:, 1:], after[:, 1:]))
        zeros = self.model.normalizer(torch.zeros(2, 3, 128))
        self.assertTrue(torch.isfinite(zeros).all())
        self.assertTrue(torch.equal(zeros, torch.zeros_like(zeros)))

    def test_full_ccml_updates_all_projections_through_normalization(self):
        original_x = {m: t.clone() for m, t in self.x.items()}
        before = {m: state_digest(self.model.projections[m].state_dict()) for m in MODALITIES}
        optimizer = torch.optim.Adamax([
            {'params': self.model.projections.parameters(), 'lr': 1e-4},
            {'params': self.model.heads.parameters(), 'lr': 1e-3}], weight_decay=0.)
        loss = ccml_loss(F.softplus(self.model(self.x)), torch.tensor([0, 1, 0, 1]), 1, **LOSS)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        for m in MODALITIES:
            self.assertNotEqual(before[m], state_digest(self.model.projections[m].state_dict()))
            self.assertTrue(torch.equal(self.x[m], original_x[m]))
            self.assertIsNone(self.x[m].grad)

    def test_restricted_checkpoint_reload_is_exact(self):
        stream = io.BytesIO()
        torch.save({'state_dict': self.model.state_dict(), 'torch': str(torch.__version__)}, stream)
        stream.seek(0)
        saved = torch.load(stream, map_location='cpu', weights_only=True)
        restored = NormalizedProjectionHeads(self.initial, 22)
        restored.load_state_dict(saved['state_dict'], strict=True)
        self.assertTrue(torch.equal(self.model(self.x), restored(self.x)))

    def test_128d_cache_is_rejected(self):
        with self.assertRaises(RuntimeError):
            self.model({m: torch.zeros(4, 128) for m in MODALITIES})


if __name__ == '__main__':
    unittest.main()

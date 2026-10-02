import io
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from .audio import pool_question, pool_subject, SpeechAdaptation, make_model, initial_projection
from .common import set_seed, LOSS
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.ccml import ccml_loss


class SpeechPoolingTests(unittest.TestCase):
    def test_empty_question_is_zero_and_kept_in_29_question_denominator(self):
        empty = pool_question(np.empty((0, 768), dtype=np.float32), [])
        self.assertTrue(np.array_equal(empty, np.zeros(768, dtype=np.float32)))
        questions = np.zeros((29, 768), dtype=np.float32)
        questions[0] = 29.
        self.assertTrue(np.array_equal(pool_subject(questions), np.ones(768, dtype=np.float32)))

    def test_question_weight_is_independent_of_segment_count(self):
        questions = np.zeros((29, 768), dtype=np.float32)
        questions[0] = pool_question(np.full((1, 768), 29, np.float32), [[0, 80000]])
        questions[1] = pool_question(np.full((10, 768), 58, np.float32), [[i*40000, i*40000+80000] for i in range(10)])
        self.assertTrue(np.array_equal(pool_subject(questions), np.full(768, 3, np.float32)))

    def test_lexical_cache_order_and_duplicate_tail_remain_in_equal_mean(self):
        temporal = np.repeat(np.arange(12, dtype=np.float32)[:, None], 768, axis=1)
        temporal[-1] = temporal[-2]
        lexical = sorted(range(12), key=lambda i: f'part{i+1}.wav')
        windows = [[i*40000, i*40000+80000] for i in range(11)]
        windows.append(windows[-1])
        actual = pool_question(temporal[lexical], windows)
        self.assertTrue(np.array_equal(actual, temporal.mean(0, dtype=np.float64).astype(np.float32)))
        self.assertFalse(np.array_equal(actual, temporal[:-1].mean(0, dtype=np.float64).astype(np.float32)))

    def test_bad_shapes_nonfinite_and_mismatched_windows_fail(self):
        cases = [(np.zeros((1, 512), np.float32), [[0, 80000]]),
                 (np.full((1, 768), np.nan, np.float32), [[0, 80000]]),
                 (np.zeros((2, 768), np.float32), [[0, 80000]])]
        for vectors, windows in cases:
            with self.assertRaises(ValueError):
                pool_question(vectors, windows)
        with self.assertRaises(ValueError):
            pool_subject(np.zeros((28, 768), np.float32))


class SpeechAdaptationTests(unittest.TestCase):
    def setUp(self):
        set_seed(11)
        self.model = SpeechAdaptation(11)
        self.frozen = torch.randn(4, 3, 128)
        self.pooled = torch.randn(4, 768)

    def test_parameter_scope_and_stage_head_pairing(self):
        self.assertEqual(sum(p.numel() for p in self.model.speech_projection.parameters()), 98432)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 99206)
        self.assertEqual(state_digest(self.model.speech_projection.state_dict()), state_digest(initial_projection(11).state_dict()))
        for stage in ['control', 'N']:
            model = make_model(11, stage)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 774)
            self.assertEqual(state_digest(model.state_dict()), state_digest(make_initial_state(11)))
        self.assertEqual(state_digest(self.model.heads.state_dict()), state_digest(make_initial_state(11)))

    def test_old_speech_is_unused_and_eeg_text_are_exactly_preserved(self):
        reps = self.model.representations(self.pooled, self.frozen)
        self.assertTrue(torch.equal(reps[:, [0, 2]], self.frozen[:, [0, 2]]))
        changed = self.frozen.clone()
        changed[:, 1] *= 10000
        self.assertTrue(torch.equal(self.model(self.pooled, self.frozen), self.model(self.pooled, changed)))
        altered = self.pooled.clone()
        altered[1:] *= 100
        original_logits = self.model(self.pooled, self.frozen)
        altered_logits = self.model(altered, self.frozen)
        self.assertTrue(torch.equal(original_logits[0], altered_logits[0]))
        self.assertTrue(torch.equal(original_logits[:, [0, 2]], altered_logits[:, [0, 2]]))

    def test_ccml_updates_speech_projection_without_mutating_frozen_inputs(self):
        before = state_digest(self.model.speech_projection.state_dict())
        x_copy, pooled_copy = self.frozen.clone(), self.pooled.clone()
        optimizer = torch.optim.Adamax([
            {'params': self.model.speech_projection.parameters(), 'lr': 1e-4},
            {'params': self.model.heads.parameters(), 'lr': 1e-3}])
        loss = ccml_loss(F.softplus(self.model(self.pooled, self.frozen)), torch.tensor([0, 1, 0, 1]), 1, **LOSS)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        self.assertNotEqual(before, state_digest(self.model.speech_projection.state_dict()))
        self.assertTrue(torch.equal(self.frozen, x_copy))
        self.assertTrue(torch.equal(self.pooled, pooled_copy))
        self.assertIsNone(self.frozen.grad)
        self.assertIsNone(self.pooled.grad)

    def test_safe_reload_reproduces_adaptation_and_frozen_N_path(self):
        stream = io.BytesIO()
        torch.save({'state_dict': self.model.state_dict(), 'torch': str(torch.__version__)}, stream)
        stream.seek(0)
        restored = SpeechAdaptation(22)
        restored.load_state_dict(torch.load(stream, weights_only=True)['state_dict'])
        self.assertTrue(torch.equal(self.model(self.pooled, self.frozen), restored(self.pooled, self.frozen)))
        reps = restored.representations(self.pooled, self.frozen).detach()
        control, normal = make_model(11, 'control'), make_model(11, 'N')
        self.assertTrue(torch.equal(control(reps), restored.heads(reps)))
        self.assertTrue(torch.equal(normal(reps), restored.heads(F.layer_norm(reps, (128,), eps=1e-5))))


if __name__ == '__main__':
    unittest.main()

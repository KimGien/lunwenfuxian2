"""CPU-only synthetic head, stop-policy and interrupted-save regressions."""
from __future__ import annotations

import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from . import frozen_heads as F
from . import protocol as P
from . import storage as S
from tridep_ccml.tridep_eegpt.recovery_io import _same

np, torch = F.np, F.torch


def cpu_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': []}


def restore_cpu_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch_cpu'])


def cpu_runtime(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)


class FrozenHeadsCPU(unittest.TestCase):
    def setUp(self):
        self.before = cpu_rng()
        torch.set_num_threads(4)

    def tearDown(self):
        restore_cpu_rng(self.before)

    def test_subject_head_keeps_full60_window_axis_and_mean_pooling(self):
        model = F.SubjectHead(4).eval()
        x = torch.randn(2, 60, 4, requires_grad=True)
        actual = model(x)
        expected = model.classifier(model.window(x).mean(dim=1))
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.allclose(actual, model(x[:, torch.randperm(60)]), atol=1e-7, rtol=1e-6))
        torch.nn.functional.cross_entropy(actual, torch.tensor([0, 1])).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad.abs().sum(dim=-1) > 0).all())
        for invalid in (torch.randn(2, 59, 4), torch.randn(60, 4), torch.randn(2, 61, 4)):
            with self.assertRaises(ValueError):
                model(invalid)

    def test_validation_restores_mixed_modes_and_never_reads_test_features(self):
        model = F.SubjectHead(4).train()
        model.window[0].eval()
        modes = [m.training for m in model.modules()]
        features = torch.randn(5, 60, 4)
        features[4] = float('nan')  # Any accidental test input makes the diagnostic fail.
        ids = ['t0', 't1', 't2', 'v0', 'never_test']
        split = {'train': ids[:3], 'val': [ids[3]], 'test': [ids[4]]}
        labels = {'t0': 0, 't1': 1, 't2': 0, 'v0': 1}  # Test label deliberately absent.
        state, parameters = cpu_rng(), F.C.state_digest(model.state_dict())
        with patch.object(F, 'rng_state', cpu_rng), patch.object(F, 'restore_rng', restore_cpu_rng):
            result = F.validation(model, features, ids, split, labels)
        self.assertEqual(set(result), {'train_eval_ce', 'val_eval_ce'})
        self.assertTrue(F.rng_equal(state, cpu_rng()))
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(parameters, F.C.state_digest(model.state_dict()))

    def test_stop_extension_and_best_tie_rules(self):
        self.assertEqual(P.best_validation_epoch([1., 1., 1.0001]), 1)
        self.assertEqual(P.best_validation_epoch([1., .99995, .9998]), 3)
        flat = [1.] * 61
        self.assertFalse(P.training_decision('frozen', flat[:60])['stop'])
        self.assertEqual(P.training_decision('frozen', flat)['reason'], 'validation_patience')
        decreasing = [2. - i * .004 for i in range(300)]
        initial_cap = P.training_decision('frozen', decreasing)
        self.assertTrue(initial_cap['extend']); self.assertFalse(initial_cap['stop'])
        continued = P.training_decision('frozen', decreasing, extended=True)
        self.assertFalse(continued['stop']); self.assertFalse(continued['extend'])
        plateau_tail = decreasing + [decreasing[-1]] * 60
        self.assertTrue(P.training_decision('frozen', plateau_tail, extended=True)['stop'])
        self.assertEqual(P.learning_rate_multiplier('frozen', 300), .1)
        self.assertEqual(P.learning_rate_multiplier('frozen', 301), .1)
        self.assertEqual(P.learning_rate_multiplier('frozen', 500), .1)

    def test_actual_fit_cpu_resume_after_latest_save_matches_uninterrupted_best(self):
        # Synthetic identifiers and CPU features; all registry writes are mocked.
        with tempfile.TemporaryDirectory(prefix='frozen-head-cpu-', dir=S.REPORT) as task_dir:
            root = Path(task_dir)
            ids = [f'engineering_s{i}' for i in range(37)]
            split = {'train': ids[:27], 'val': ids[27:30], 'test': ids[30:]}
            labels = {s: i % 2 for i, s in enumerate(ids[:30])}
            features = torch.arange(37 * 60 * 4, dtype=torch.float32).reshape(37, 60, 4) / 9000
            features[30:] = float('nan')
            manifest = {'subject_ids': ids, 'dimension': 4, 'input_seconds': 4}
            source = root / 'manifest.json'
            S.save(source, manifest)
            descriptor = {'path': 'synthetic-registry', 'sha256': 'engineering'}
            chosen_folder = [root / 'continuous']
            original_save = S.save
            def flat_validation(*_):
                return {'train_eval_ce': 1., 'val_eval_ce': 1.}
            with patch.object(F.C, 'runtime', cpu_runtime), patch.object(F, 'rng_state', cpu_rng), \
                 patch.object(F, 'restore_rng', restore_cpu_rng), patch.object(F.torch.cuda, 'device_count', return_value=0), \
                 patch.object(F.R, 'registration_descriptor', return_value=descriptor), \
                 patch.object(F.R, 'fit_directory', side_effect=lambda *_: chosen_folder[0]), \
                 patch.object(F.R, 'record_fit_status'), patch.object(F.R, 'fit_status', return_value={'status': 'failed'}), \
                 patch.object(F.R, '_validate_complete'), patch.object(F, 'manifest_path', return_value=source), \
                 patch.object(F, 'validation', flat_validation):
                baseline_meta = F.fit('EEGPT-frozen-smallhead', 11, 1, features, manifest, split, labels)
                baseline = torch.load(chosen_folder[0] / 'latest.pt', map_location='cpu', weights_only=False)
                baseline_best = torch.load(chosen_folder[0] / 'best.pt', map_location='cpu', weights_only=False)
                self.assertEqual((baseline_meta['epochs'], baseline_meta['updates'], baseline_meta['best_epoch']), (61, 244, 1))
                chosen_folder[0] = root / 'interrupted'
                injected = [False]
                def save(path, value, **kwargs):
                    if Path(path).name == 'train_history.json' and len(value) == 2 and not injected[0]:
                        injected[0] = True
                        raise OSError('injected after durable latest epoch2')
                    return original_save(path, value, **kwargs)
                with patch.object(F.S, 'save', save):
                    with self.assertRaisesRegex(OSError, 'durable latest epoch2'):
                        F.fit('EEGPT-frozen-smallhead', 11, 1, features, manifest, split, labels)
                restored_meta = F.fit('EEGPT-frozen-smallhead', 11, 1, features, manifest, split, labels, resume=True)
                restored = torch.load(chosen_folder[0] / 'latest.pt', map_location='cpu', weights_only=False)
                restored_best = torch.load(chosen_folder[0] / 'best.pt', map_location='cpu', weights_only=False)
                self.assertTrue(_same(baseline_best, restored_best))
                for key in ('state_dict', 'optimizer_state_dict', 'history', 'order', 'best_state_dict', 'best_state_sha256', 'rng'):
                    self.assertTrue(_same(baseline[key], restored[key]), key)
                self.assertEqual(restored_meta['selected_state_sha256'], baseline_meta['selected_state_sha256'])
                with patch.object(F, 'SubjectHead', side_effect=AssertionError('Completed fit must not reconstruct')):
                    self.assertEqual(F.fit('EEGPT-frozen-smallhead', 11, 1, features, manifest, split, labels), restored_meta)


if __name__ == '__main__':
    unittest.main(verbosity=2)

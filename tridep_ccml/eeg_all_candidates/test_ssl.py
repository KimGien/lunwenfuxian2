"""CPU-only SSL sampling, masks, validation and interrupted recovery tests."""
from __future__ import annotations

from dataclasses import replace
import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from . import ssl as U
from . import protocol as P
from . import storage as S
from .adapters import eeg2rep as E
from .test_frozen_heads import cpu_rng, restore_cpu_rng, cpu_runtime
from tridep_ccml.tridep_eegpt.recovery_io import _same

np, torch = U.np, U.torch


class ToySSL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Module()
        self.core.contex_encoder = torch.nn.Linear(19, 16)
        self.core.target_encoder = copy.deepcopy(self.core.contex_encoder).requires_grad_(False)
        self.core.predict_head = torch.nn.Linear(16, 2).requires_grad_(False)

    def forward(self, x):
        return self.core.contex_encoder(x.mean(dim=-1))

    def pretrain_forward(self, x, *, mask=None):
        online = self(x)
        with torch.no_grad():
            target = self.core.target_encoder(x.mean(dim=-1))
        masked = len(mask['masked_indices'])
        return {'context': online[:, None].expand(-1, 250, -1),
                'prediction_masked': online[:, None].expand(-1, masked, -1),
                'target_masked': target[:, None].expand(-1, masked, -1)}

    @torch.no_grad()
    def momentum_update(self):
        for target, online in zip(self.core.target_encoder.parameters(), self.core.contex_encoder.parameters()):
            target.mul_(.99).add_(online, alpha=.01)


class SyntheticData:
    def __init__(self, path):
        self.manifest = S.desc(path)
        self.ids = ['engineering_t0', 'engineering_t1', 'engineering_v0']
        generator = torch.Generator(device='cpu').manual_seed(197)
        self.raw = torch.randn(3, 60, 19, 1000, generator=generator)
        self.reads = []

    def batch(self, people, indices, device):
        self.reads.append(tuple(people))
        return torch.stack([self.raw[self.ids.index(people[i // 60]), i % 60] for i in indices]).to(device)


class SSLCPU(unittest.TestCase):
    def setUp(self):
        self.before = cpu_rng()
        torch.set_num_threads(4)

    def tearDown(self):
        restore_cpu_rng(self.before)

    def test_predictor_or_ema_changes_alone_never_prove_online_backbone_training(self):
        original = {'core.contex_encoder.weight': torch.ones(2, 2), 'core.InputEmbedding.weight': torch.ones(2),
                    'core.Norm.weight': torch.ones(2), 'core.Predictor.weight': torch.ones(2),
                    'core.target_encoder.weight': torch.ones(2, 2), 'core.predict_head.weight': torch.ones(2)}
        changed = copy.deepcopy(original)
        changed['core.Predictor.weight'].add_(1)
        changed['core.target_encoder.weight'].add_(1)
        self.assertEqual(U.changed_online_names(original, changed), [])
        with self.assertRaisesRegex(ValueError, 'predictor/target EMA'):
            U.require_online_change(original, changed)
        changed['core.contex_encoder.weight'][0, 0] += .1
        self.assertEqual(U.require_online_change(original, changed), ['core.contex_encoder.weight'])

    def test_registered_fold_guard_and_initial_checkpoint_count_guards(self):
        registration = U.R.load_registration()
        entry = next(item for item in registration['fits'] if item['condition'] == U.CONDITION and item['seed'] == 11 and item['fold'] == 1)
        split = {role: entry[role + '_ids'] for role in ('train', 'val', 'test')}
        data = type('ReadOnlyPeople', (), {'ids': registration['subject_ids']})()
        U.validate_fit_context(11, 1, data, split)
        wrong = copy.deepcopy(split)
        wrong['train'][0], wrong['val'][0] = wrong['val'][0], wrong['train'][0]
        with self.assertRaisesRegex(ValueError, 'fold assignments'):
            U.validate_fit_context(11, 1, data, wrong)
        model, _ = E.build_model('cpu', stage='ssl')
        optimizer = U.optimizer_for(model)
        generator, masks = U.sampler_streams(11, 1)
        identity = {'engineering': True}
        with patch.object(U, 'rng_state', cpu_rng), patch.object(U.torch.cuda, 'device_count', return_value=0):
            initial = U.checkpoint(model, optimizer, generator, masks, 0, [], [], None, False, identity)
            U.validate_resume(initial, model, identity, split['train'], 64)
            altered = copy.deepcopy(initial)
            altered['ema_updates'] = 1
            with self.assertRaisesRegex(ValueError, 'counts/EMA'):
                U.validate_resume(altered, model, identity, split['train'], 64)
            altered = copy.deepcopy(initial)
            altered['configuration']['mask_ratio'] = .4
            with self.assertRaisesRegex(ValueError, 'hyperparameters'):
                U.validate_resume(altered, model, identity, split['train'], 64)

    def test_train_interleaving_uses_every_person_window_once_and_has_no_single_tail(self):
        for people in (26, 27):
            generator, _ = U.sampler_streams(11, 1)
            for _ in range(3):
                order = U.training_order(people, generator)
                self.assertEqual(sorted(order), list(range(people * 60)))
                for turn in range(60):
                    self.assertEqual(sorted(i // 60 for i in order[turn * people:(turn + 1) * people]), list(range(people)))
                for size in (4, 32, 64):
                    groups = U.logical_batches(order, size)
                    self.assertEqual([i for group in groups for i in group], order)
                    self.assertTrue(all(2 <= len(group) <= size for group in groups))
            self.assertEqual(len(U.logical_batches(order, 64)), 25 if people == 26 else 26)
        self.assertEqual([len(g) for g in U.logical_batches(list(range(129)), 64)], [64, 63, 2])

    def test_separate_sampling_and_ssp_streams_restore_exact_next_epoch(self):
        generator, masks = U.sampler_streams(22, 3)
        U.training_order(27, generator)
        for _ in range(26):
            E.make_ssp_mask(rng=masks)
        saved = U.sampler_state(generator, masks)
        expected_order = U.training_order(27, generator)
        expected_masks = [U.mask_record(E.make_ssp_mask(rng=masks)) for _ in range(26)]
        U.restore_sampler(saved, generator, masks)
        self.assertEqual(U.training_order(27, generator), expected_order)
        self.assertEqual([U.mask_record(E.make_ssp_mask(rng=masks)) for _ in range(26)], expected_masks)
        for seed in range(40):
            mask = E.make_ssp_mask(rng=random.Random(seed))
            self.assertEqual(U.restore_mask(U.mask_record(mask)), mask)

    def test_fixed_validation_restores_author_state_and_does_not_ema_or_read_test(self):
        with tempfile.TemporaryDirectory(prefix='ssl-val-cpu-', dir=S.REPORT) as task_dir:
            folder = Path(task_dir)
            path = folder / 'source.json'; S.save(path, {'synthetic': True})
            data = SyntheticData(path)
            split = {'train': [data.ids[0]], 'val': data.ids[1:], 'test': ['never_read']}
            with patch.object(U.R, 'registration_descriptor', return_value={'path': 'synthetic', 'sha256': 'engineering'}):
                manifest = U.validation_manifest(folder, split, 11, 1, 64, data.manifest)
            model = ToySSL().train()
            model.core.target_encoder.eval()
            modes = [m.training for m in model.modules()]
            state = cpu_rng(); before = U.C.state_digest(model.state_dict())
            with patch.object(U, 'rng_state', cpu_rng), patch.object(U, 'restore_rng', restore_cpu_rng), \
                 patch.object(model, 'momentum_update', side_effect=AssertionError('validation must never EMA')):
                actual = U.validation(model, data, split, manifest, 'float32')
                repeat = U.validation(model, data, split, manifest, 'float32')
            self.assertEqual(actual, repeat)
            self.assertTrue(all(people == tuple(split['val']) for people in data.reads))
            self.assertEqual(modes, [m.training for m in model.modules()])
            self.assertEqual(before, U.C.state_digest(model.state_dict()))
            self.assertTrue(U.rng_equal(state, cpu_rng()))

    def test_actual_ssl_fit_resume_matches_uninterrupted_model_ema_optimizer_sampling_best(self):
        with tempfile.TemporaryDirectory(prefix='ssl-resume-cpu-', dir=S.REPORT) as task_dir:
            root = Path(task_dir)
            path = root / 'source.json'; S.save(path, {'synthetic': True})
            data = SyntheticData(path)
            split = {'train': data.ids[:2], 'val': data.ids[2:], 'test': ['unread_test_without_label']}
            settings_path = root / 'settings.json'
            S.save(settings_path, {'actual_logical_batch': 64, 'precision': 'float32'})
            chosen_folder = [root / 'continuous']
            synthetic_policy = replace(P.POLICIES['ssl'], initial_epochs=3, minimum_stop_epoch=1, patience=2,
                                       final_epochs=4, warmup_epochs=1)
            def build(*_, **__):
                model = ToySSL()
                return model, {'parameters': sum(p.numel() for p in model.parameters())}
            def flat_validation(*_):
                return {'loss': 1., 'alignment_mse': .2, 'variance_hinge': .8,
                        'offdiagonal_covariance': 0., 'context_feature_std_mean': .2, 'prediction_feature_std_mean': .2}
            with patch.object(U, 'SETTINGS', settings_path), patch.object(U.C, 'runtime', cpu_runtime), \
                 patch.object(U, 'rng_state', cpu_rng), patch.object(U, 'restore_rng', restore_cpu_rng), \
                 patch.object(U.torch.cuda, 'device_count', return_value=0), patch.object(U.R, '_condition_lock'), \
                 patch.object(U, 'validate_fit_context'), \
                 patch.object(U.R, 'fit_directory', side_effect=lambda *_: chosen_folder[0]), patch.object(U.R, 'record_fit_status'), \
                 patch.object(U.R, 'registration_descriptor', return_value={'path': 'synthetic', 'sha256': 'engineering'}), \
                 patch.object(U.R, '_validate_complete'), patch.object(U.R, 'fit_status', return_value={'status': 'failed'}), \
                 patch.object(U.E, 'build_model', side_effect=build), patch.object(U, 'validation', flat_validation), \
                 patch.dict(P.POLICIES, {'ssl': synthetic_policy}):
                baseline_meta = U.fit_one(11, 1, data, split, device='cpu')
                baseline = torch.load(chosen_folder[0] / 'latest.pt', map_location='cpu', weights_only=False)
                baseline_best = torch.load(chosen_folder[0] / 'best.pt', map_location='cpu', weights_only=False)
                self.assertEqual((baseline_meta['epochs'], baseline_meta['updates'], baseline_meta['ema_updates']), (3, 6, 6))
                self.assertTrue(baseline_meta['online_backbone_update_verified'])
                self.assertGreater(baseline_meta['online_changed_parameter_count'], 0)
                self.assertNotEqual(baseline_meta['initial_online_state_sha256'], baseline_meta['selected_online_state_sha256'])
                chosen_folder[0] = root / 'interrupted'
                original_save = S.save
                injected = [False]
                def save(path, value, **kwargs):
                    if Path(path).name == 'train_history.json' and len(value) == 1 and not injected[0]:
                        injected[0] = True
                        raise OSError('injected after SSL latest epoch1')
                    return original_save(path, value, **kwargs)
                with patch.object(U.S, 'save', save):
                    with self.assertRaisesRegex(OSError, 'SSL latest epoch1'):
                        U.fit_one(11, 1, data, split, device='cpu')
                restored_meta = U.fit_one(11, 1, data, split, resume=True, device='cpu')
                restored = torch.load(chosen_folder[0] / 'latest.pt', map_location='cpu', weights_only=False)
                restored_best = torch.load(chosen_folder[0] / 'best.pt', map_location='cpu', weights_only=False)
                self.assertTrue(_same({k: v for k, v in baseline_best.items() if k != 'identity'},
                                      {k: v for k, v in restored_best.items() if k != 'identity'}))
                for key in ('state_dict', 'optimizer_state_dict', 'sampling', 'rng', 'history', 'order', 'best', 'ema_updates'):
                    self.assertTrue(_same(baseline[key], restored[key]), key)
                self.assertEqual(baseline_meta['selected_state_sha256'], restored_meta['selected_state_sha256'])
                self.assertEqual(baseline_meta['selected_online_state_sha256'], restored_meta['selected_online_state_sha256'])
                self.assertEqual(baseline_meta['online_changed_parameter_names'], restored_meta['online_changed_parameter_names'])
                with patch.object(U.E, 'build_model', side_effect=AssertionError('Completed SSL fit must not construct')):
                    self.assertEqual(U.fit_one(11, 1, data, split, device='cpu'), restored_meta)


if __name__ == '__main__':
    unittest.main(verbosity=2)

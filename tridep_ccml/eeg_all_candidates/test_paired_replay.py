"""Meaningful replay/IO regressions. Real CUDA inputs are original train people only."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from . import paired_replay as P
from . import storage as S
from tridep_ccml.tridep_eegpt.recovery_io import _same

np, torch = P.np, P.torch


class ReplayContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = S.read(S.WORK / 'S00/configs/cohort.json')
        cls.ids = cls.lock['subject_ids']
        cls.folds = S.read(S.WORK / 'S00' / cls.lock['fold_file'])
        cls.labels = {r['subject_id']: r['label'] for r in S.read(S.WORK / 'S00/manifests/subjects.json')}
        cls.audit = S.read(S.REPORT / 'replay_readonly_audit.json')
        S.REPORT.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.rng = P.rng_state()

    def tearDown(self):
        P.restore_rng(self.rng)

    def test_all_300_orders_match_independent_cpu_rng_recipe_and_old_prefix(self):
        torch.set_num_threads(4)
        for seed in (11, 22, 33):
            for fold in range(1, 6):
                torch.manual_seed(seed)
                model = P.T.original_model()
                reference = next(r for r in self.audit['checks'] if r['seed'] == seed and r['fold'] == fold)
                self.assertEqual(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                                 reference['cpu_rng_after_initialization_sha256'])
                old_folder = P.T._folder(P.T.CONDITIONS[0], seed, fold)
                old_initial = torch.load(old_folder / 'initial.pt', map_location='cpu', weights_only=True)
                self.assertTrue(_same(dict(model.state_dict()), old_initial['state_dict']))
                _, dataset = P.T.training_dataset(self.ids, self.folds[f'fold_{fold}']['train'], self.labels)
                loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
                order = []
                for epoch in range(1, 301):
                    order.append([int(i) for batch, _ in loader for i in batch])
                    if epoch in P.SNAPSHOTS:
                        self.assertEqual(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
                                         reference['independent_cpu_rng_endpoint_sha256'][str(epoch)])
                for condition in P.T.CONDITIONS:
                    old_order = np.load(P.T._folder(condition, seed, fold) / 'batch_order.npy', allow_pickle=False)
                    self.assertTrue(np.array_equal(np.asarray(order[:60], dtype=np.int64), old_order))
                self.assertEqual(len(order), 300)
                self.assertTrue(all(sorted(row) == sorted(order[0]) for row in order))

    def _training_only_diagnostic_inputs(self):
        chosen = self.folds['fold_1']['train'][:4]
        manifest = S.read(P.C.OUT / 'features_manifest.json')
        records = {r['subject_id']: r for r in manifest['records']}
        features = torch.zeros((37, 60, 200), dtype=torch.float32, device='cuda')
        for sid in chosen:
            with np.load(S.checked(records[sid]['features']), allow_pickle=False) as bundle:
                features[self.ids.index(sid)] = torch.as_tensor(bundle['labram'], device='cuda')
        # Both diagnostic roles deliberately contain only original outer-train people.
        split = {'train': chosen[:2], 'val': chosen[2:], 'test': []}
        return chosen, features, split

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA engineering contract')
    def test_diagnostic_preserves_rng_parameters_modes_and_next_dropout(self):
        P.C.runtime(11)
        model = P.T.original_model().to('cuda').train()
        chosen, features, split = self._training_only_diagnostic_inputs()
        model.conv[0].eval()  # Exercise mixed module-mode restoration.
        modes = [m.training for m in model.modules()]
        before = P.C.state_digest(model.state_dict())
        state = P.rng_state()
        result = P.diagnostic(model, features, self.ids, split, self.labels)
        self.assertEqual(set(result), {'train_eval_ce', 'val_eval_ce'})
        self.assertTrue(P.rng_equal(state, P.rng_state()))
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertEqual(before, P.C.state_digest(model.state_dict()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        x = features[torch.tensor([self.ids.index(s) for s in chosen], device='cuda')]
        with torch.no_grad():
            actual = model(x).clone()
            P.restore_rng(state)
            expected = model(x).clone()
        self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA engineering contract')
    def test_diagnostic_restores_all_rng_and_modes_even_on_failure(self):
        P.C.runtime(11)
        model = P.T.original_model().to('cuda').train()
        _, features, split = self._training_only_diagnostic_inputs()
        state = P.rng_state()
        modes = [m.training for m in model.modules()]
        def fail(*_):
            random.random(); np.random.random(); torch.rand(1); torch.rand(1, device='cuda')
            raise RuntimeError('injected diagnostic failure')
        with patch.object(model, 'forward', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'injected diagnostic failure'):
                P.diagnostic(model, features, self.ids, split, self.labels)
        self.assertTrue(P.rng_equal(state, P.rng_state()))
        self.assertEqual(modes, [m.training for m in model.modules()])

    def test_rng_equal_rejects_missing_cuda_device_state(self):
        original = P.rng_state()
        altered = copy.deepcopy(original)
        altered['torch_cuda'] = [] if original['torch_cuda'] else [torch.zeros(8, dtype=torch.uint8)]
        self.assertFalse(P.rng_equal(original, altered))

    def test_checkpoint_roundtrip_produces_exact_uninterrupted_next_epoch(self):
        # Toy CPU updates exercise continuation without creating any candidate fit.
        torch.manual_seed(194)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.4), torch.nn.Linear(4, 2))
        optimizer = torch.optim.Adamax(model.parameters(), lr=.005)
        x = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 18
        y = torch.tensor([0, 1, 0, 1, 0, 1])
        def epoch(m, opt):
            m.train()
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=3, shuffle=True)
            losses = []
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(m(xb), yb)
                loss.backward(); opt.step()
                losses.append(float(loss.detach()))
            return losses
        epoch(model, optimizer)
        descriptor = {'path': 'engineering-test-registration', 'sha256': 'test'}
        with tempfile.TemporaryDirectory(prefix='replay-resume-test-', dir=S.REPORT) as task_dir:
            path = Path(task_dir) / 'latest.pt'
            with patch.object(P.R, 'registration_descriptor', return_value=descriptor):
                payload = P.checkpoint_value(model, optimizer, 1, [], [], 'engineering-initial')
            S.save_torch(path, payload)
            reference_losses = epoch(model, optimizer)
            reference_state = copy.deepcopy(model.state_dict())
            reference_optimizer = copy.deepcopy(optimizer.state_dict())
            reference_rng = P.rng_state()
            resumed = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.4), torch.nn.Linear(4, 2))
            resumed_optimizer = torch.optim.Adamax(resumed.parameters(), lr=.005)
            saved = torch.load(path, map_location='cpu', weights_only=False)
            resumed.load_state_dict(saved['state_dict']); resumed_optimizer.load_state_dict(saved['optimizer_state_dict'])
            P.restore_rng(saved['rng'])
            self.assertEqual(reference_losses, epoch(resumed, resumed_optimizer))
            self.assertTrue(_same(reference_state, resumed.state_dict()))
            self.assertTrue(_same(reference_optimizer, resumed_optimizer.state_dict()))
            self.assertTrue(P.rng_equal(reference_rng, P.rng_state()))
            self.assertEqual(saved['sampler_position'], {'epoch_completed': 1, 'next_epoch': 2, 'next_batch': 0})

    def test_storage_retries_replacement_without_serializing_again(self):
        with tempfile.TemporaryDirectory(prefix='replay-io-test-', dir=S.REPORT) as task_dir:
            path = Path(task_dir) / 'checkpoint.bin'
            original_replace = Path.replace
            attempts, serializations = [], []
            def replace(current, target):
                attempts.append(str(current))
                if len(attempts) == 1:
                    raise PermissionError('injected transient sharing failure')
                return original_replace(current, target)
            def serialize(stream):
                serializations.append(1)
                stream.write(b'exact serialized state')
            with patch.object(Path, 'replace', replace):
                S.write(path, serialize)
            self.assertEqual(len(serializations), 1)
            self.assertEqual(len(attempts), 2)
            self.assertEqual(path.read_bytes(), b'exact serialized state')


if __name__ == '__main__':
    unittest.main(verbosity=2)

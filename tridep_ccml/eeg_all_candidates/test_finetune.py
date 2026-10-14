"""CPU synthetic tests; no participant data, GPU work or formal fit records."""
from __future__ import annotations

import copy
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

import numpy as np
import torch

from . import finetune as F


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Sequential(torch.nn.Linear(1, 4), torch.nn.BatchNorm1d(4), torch.nn.Dropout(.8))
        self.last = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.3))

    def forward(self, windows):
        return self.last(self.frozen(windows.mean(dim=(1, 2)).unsqueeze(-1)))


class TinyAdapter:
    @staticmethod
    def adapt_raw(raw):
        return raw

    @staticmethod
    def encode_windows(model, tensor):
        return model(tensor)


def model_fixture():
    torch.manual_seed(31)
    encoder = TinyEncoder()
    selected = [name for name, _ in encoder.named_parameters() if name.startswith("last.")]
    for name, parameter in encoder.named_parameters():
        parameter.requires_grad_(name in selected)
    return F.RawSubjectModel(encoder, F.SubjectBagHead(4), TinyAdapter, selected, 5, window_chunk=8)


def data_fixture(count=5):
    people = [f"synthetic_{i}" for i in range(count)]
    values = np.sin(np.arange(1000, dtype=np.float32) * .04)[None, None, :]
    base = np.broadcast_to(values, (60, 19, 1000)).copy()
    raw = {person: base + np.float32(i / 10) for i, person in enumerate(people)}
    labels = {person: i % 2 for i, person in enumerate(people)}
    return people, raw, labels


class FinetuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_eight_epoch_window_coverage_and_explicit_padding(self):
        cycle = np.concatenate([F.bag_indices(epoch) for epoch in range(1, 9)])
        self.assertEqual(len(cycle), 64)
        self.assertEqual(sorted(set(cycle)), list(range(60)))
        self.assertEqual(cycle[-8:].tolist(), [56, 57, 58, 59, 0, 1, 2, 3])
        self.assertEqual(F.bag_indices(9).tolist(), F.bag_indices(1).tolist())

    def test_warmup_changes_head_only_and_then_last_block(self):
        model = model_fixture()
        optimizer = F.optimizer_for(model)
        people, raw, labels = data_fixture(2)
        frozen = F.state_digest(F.frozen_state(model))
        before = F.cpu_state(model.encoder.state_dict())
        F.set_train_phase(model, 1)
        F.set_learning_rates(optimizer, 1)
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))
        self.assertEqual(optimizer.param_groups[1]["lr"], 0.)
        F.subject_update(model, optimizer, people, raw, labels, F.bag_indices(1), micro_subjects=1)
        self.assertTrue(all(torch.equal(before[name], tensor) for name, tensor in model.encoder.state_dict().items()))
        F.set_train_phase(model, 11)
        F.set_learning_rates(optimizer, 11)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 2e-6)
        self.assertFalse(model.encoder.frozen.training)
        self.assertTrue(model.encoder.last.training)
        F.subject_update(model, optimizer, people, raw, labels, F.bag_indices(11), micro_subjects=2)
        self.assertNotEqual(F.state_digest(before), F.state_digest(model.encoder.state_dict()))
        self.assertEqual(frozen, F.state_digest(F.frozen_state(model)))

    def test_actual_five_person_ce_accumulation_matches_one_batch(self):
        people, raw, labels = data_fixture(5)
        micro = model_fixture()
        F.set_train_phase(micro, 11)
        # Deterministic comparison isolates CE weighting from independent
        # stochastic dropout masks in differently shaped microbatches.
        for module in micro.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        full = copy.deepcopy(micro)
        opt_micro = torch.optim.SGD([p for p in micro.parameters() if p.requires_grad], lr=.01)
        opt_full = torch.optim.SGD([p for p in full.parameters() if p.requires_grad], lr=.01)
        result = F.subject_update(micro, opt_micro, people, raw, labels, F.bag_indices(2), micro_subjects=2)
        opt_full.zero_grad(set_to_none=True)
        logits = full(np.stack([raw[sid][F.bag_indices(2)] for sid in people]))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([labels[sid] for sid in people]))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in full.parameters() if p.requires_grad], 1.)
        opt_full.step()
        self.assertEqual(result["subjects"], 5)
        self.assertEqual(result["backward_calls"], 3)
        self.assertAlmostEqual(result["subject_ce_sum"] / 5, float(loss.detach()), places=6)
        for name, tensor in micro.state_dict().items():
            torch.testing.assert_close(tensor, full.state_dict()[name], rtol=2e-6, atol=2e-7)

    def test_train_val_all60_diagnostics_restore_rng_modes_and_state(self):
        model = model_fixture()
        F.set_train_phase(model, 11)
        people, raw, labels = data_fixture(3)
        split = {"train": people[:2], "val": people[2:], "test": ["outer_test_never_loaded"]}
        before_rng = F.rng_state("cpu")
        before_state = F.state_digest(model.state_dict())
        before_modes = [module.training for module in model.modules()]
        output = F.diagnostics(model, raw, split, labels)
        self.assertEqual(set(output), {"train_eval_ce", "val_eval_ce"})
        self.assertEqual(before_state, F.state_digest(model.state_dict()))
        self.assertTrue(F.rng_equal(before_rng, F.rng_state("cpu")))
        self.assertEqual(before_modes, [module.training for module in model.modules()])

    def test_resume_roundtrip_preserves_optimizer_rng_and_selected_best(self):
        model = model_fixture()
        F.set_train_phase(model, 11)
        optimizer = F.optimizer_for(model)
        F.set_learning_rates(optimizer, 11)
        people, raw, labels = data_fixture(2)
        F.subject_update(model, optimizer, people, raw, labels, F.bag_indices(11), micro_subjects=1)
        best = F.selected_state(model)
        state = F.rng_state("cpu")
        payload = {"state_dict": F.cpu_state(model.state_dict()), "optimizer": F.cpu_state(optimizer.state_dict()),
                   "rng": state, "best": best, "scheduler": {"completed_epoch": 11, "extended": False},
                   "sampler": {"epoch_completed": 11, "next_batch": 0}}
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        reloaded = model_fixture()
        optimizer_new = F.optimizer_for(reloaded)
        reloaded.load_state_dict(restored["state_dict"], strict=True)
        optimizer_new.load_state_dict(restored["optimizer"])
        F.restore_rng(restored["rng"])
        self.assertTrue(F.rng_equal(state, F.rng_state("cpu")))
        self.assertEqual(F.state_digest(model.state_dict()), F.state_digest(reloaded.state_dict()))
        self.assertEqual([float(value["step"]) for value in optimizer.state.values()],
                         [float(value["step"]) for value in optimizer_new.state.values()])
        F.load_selected_state(reloaded, restored["best"])
        self.assertEqual(F.state_digest(best), F.state_digest(F.selected_state(reloaded)))
        # Resume must reproduce the NEXT shuffled logical batch and stochastic
        # dropout update, not merely deserialize equal tensors.
        def next_update(target_model, target_optimizer):
            F.restore_rng(restored["rng"])
            F.set_train_phase(target_model, 12)
            F.set_learning_rates(target_optimizer, 12)
            shuffled = [people[i] for i in torch.randperm(len(people)).tolist()]
            F.subject_update(target_model, target_optimizer, shuffled, raw, labels, F.bag_indices(12), micro_subjects=1)
            return shuffled, F.rng_state("cpu")
        original_order, original_rng = next_update(model, optimizer)
        resumed_order, resumed_rng = next_update(reloaded, optimizer_new)
        self.assertEqual(original_order, resumed_order)
        self.assertTrue(F.rng_equal(original_rng, resumed_rng))
        self.assertEqual(F.state_digest(model.state_dict()), F.state_digest(reloaded.state_dict()))

    def test_missing_initial_epoch_zero_rebuild_and_next_step_replay(self):
        model = model_fixture()
        optimizer = F.optimizer_for(model)
        state = F.rng_state("cpu")
        payload = {"state_dict": F.cpu_state(model.state_dict()), "optimizer_state_dict": F.cpu_state(optimizer.state_dict()),
                   "rng": state, "epoch": 0, "optimizer_updates": 0, "backward_calls": 0,
                   "history": [], "subject_order": [], "best_selected_state_dict": None}
        # Keep synthetic checkpoints inside the explicitly writable workspace;
        # AppData's sandbox Temp can create a directory without write access.
        folder = F.S.WORK / "tmp/eeg_all_candidates_cpu_tests" / ("epoch0_" + uuid.uuid4().hex)
        folder.mkdir(parents=True)
        self.assertTrue(folder.resolve().is_relative_to((F.S.WORK / "tmp").resolve()))
        def temporary_save(path, value, *, immutable=False):
            self.assertTrue(immutable)
            with Path(path).open("xb") as stream:
                torch.save(value, stream)
        with patch.object(F.S, "save_torch", side_effect=temporary_save):
            F._recover_epoch_zero(folder, payload)
        restored = torch.load(folder / "initial.pt", map_location="cpu", weights_only=False)
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            F._recover_epoch_zero(folder, payload)
        blocked_folder = folder / "blocked_after_progress"
        blocked_folder.mkdir()
        (blocked_folder / "progress.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "optimizer batch started"):
            F._recover_epoch_zero(blocked_folder, payload)
        resumed = model_fixture()
        resumed_optimizer = F.optimizer_for(resumed)
        resumed.load_state_dict(restored["state_dict"], strict=True)
        resumed_optimizer.load_state_dict(restored["optimizer_state_dict"])
        people, raw, labels = data_fixture(2)
        def next_update(target_model, target_optimizer):
            F.restore_rng(restored["rng"])
            F.set_train_phase(target_model, 1)
            F.set_learning_rates(target_optimizer, 1)
            shuffled = [people[i] for i in torch.randperm(len(people)).tolist()]
            F.subject_update(target_model, target_optimizer, shuffled, raw, labels, F.bag_indices(1), micro_subjects=1)
            return shuffled, F.rng_state("cpu")
        original_order, original_rng = next_update(model, optimizer)
        resumed_order, resumed_rng = next_update(resumed, resumed_optimizer)
        self.assertEqual(original_order, resumed_order)
        self.assertTrue(F.rng_equal(original_rng, resumed_rng))
        self.assertEqual(F.state_digest(model.state_dict()), F.state_digest(resumed.state_dict()))


if __name__ == "__main__":
    unittest.main()

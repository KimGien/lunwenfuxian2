"""CPU-only semantic tests; no participant input, readiness lock or GPU fit."""
from __future__ import annotations

import copy
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import supervised as U
from . import finetune as F


class ToyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Linear(1, 4)
        self.norm = torch.nn.BatchNorm1d(4)
        self.attention = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.2))
        self.unused_target = torch.nn.Linear(4, 4)
        self.unused_target.requires_grad_(False)

    def forward(self, windows):
        return self.attention(self.norm(self.stem(windows.mean(dim=(1, 2)).unsqueeze(-1))))


class ToyAdapter:
    @staticmethod
    def adapt_raw(raw):
        return raw

    @staticmethod
    def encode_windows(model, tensor):
        return model(tensor)


def model_fixture():
    torch.manual_seed(71)
    encoder = ToyEncoder()
    selected = [name for name, p in encoder.named_parameters() if p.requires_grad]
    return U.SupervisedModel(encoder, F.SubjectBagHead(4), ToyAdapter, selected, None, 8)


def data_fixture():
    people = ["synthetic_a", "synthetic_b"]
    signal = np.sin(np.arange(1000, dtype=np.float32) * .09)[None, None, :]
    raw = {sid: np.broadcast_to(signal, (60, 19, 1000)).copy() + np.float32(index / 4) for index, sid in enumerate(people)}
    return people, raw, {people[0]: 0, people[1]: 1}


class SupervisedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_scalar_stats_ignore_validation_test_and_reject_mixed_fit_ids(self):
        raw = {"train_a": np.full((60, 19, 1000), 1., dtype=np.float32),
               "train_b": np.full((60, 19, 1000), 3., dtype=np.float32),
               "val": np.full((60, 19, 1000), 10000., dtype=np.float32),
               "test": np.full((60, 19, 1000), -10000., dtype=np.float32)}
        first = U.train_statistics(raw, ["train_a", "train_b"])
        raw["val"].fill(np.nan)
        raw["test"].fill(np.inf)
        second = U.train_statistics(raw, ["train_a", "train_b"])
        self.assertEqual(first, second)
        self.assertEqual(first["mean"], 2.)
        self.assertEqual(first["std"], 1.)
        self.assertEqual(first["fit_subject_ids"], ["train_a", "train_b"])
        with self.assertRaisesRegex(ValueError, "exactly match declared train"):
            U.EC.fit_train_statistics(np.stack([raw["train_a"], raw["train_b"]]), ["train_a", "val"], ["train_a", "train_b"])

    def test_ssl_identity_and_learned_state_binding_reject_other_fold_seed_people(self):
        registry = {"path": "synthetic_registration", "sha256": "1" * 64}
        source = {"path": "synthetic_raw_manifest", "sha256": "2" * 64}
        split = {"train": ["train_a", "train_b"], "val": ["val_a"], "test": ["test_a"]}
        identity = {"condition": U.SSL_CONDITION, "seed": 11, "fold": 1, "train_ids": split["train"],
                    "val_ids": split["val"], "encoder_fit_subject_ids": split["train"], "registry": registry,
                    "source": source, "settings": {"path": "synthetic_settings", "sha256": "3" * 64},
                    "validation_mask_manifest": {"path": "synthetic_masks", "sha256": "4" * 64}}
        state = {"synthetic_weight": torch.tensor([2., 3.])}
        fingerprint = F.state_digest(state)
        saved = {"stage": "ssl", "identity": identity, "configuration": U.ER.DEFAULT_CONFIG,
                 "ssl_contract": U.ER.SSL_CONTRACT, "epoch": 2, "optimizer_updates": 3, "ema_updates": 3,
                 "state_dict": state, "state_sha256": fingerprint}
        metadata = {**identity, "best_epoch": 2, "selected_state_sha256": fingerprint,
                    "initial_state_sha256": "0" * 64, "test_forward_count": 0, "validation_updates": 0}
        kwargs = {"seed": 11, "fold": 1, "split": split, "registration": registry, "raw_manifest": source}
        self.assertEqual(U.validate_ssl_payload(saved, metadata, **kwargs), fingerprint)
        for wrong in ({"seed": 22}, {"fold": 2}, {"encoder_fit_subject_ids": ["train_a", "val_a"]}):
            tampered = copy.deepcopy(saved)
            tampered["identity"].update(wrong)
            with self.assertRaisesRegex(ValueError, "exact seed, fold"):
                U.validate_ssl_payload(tampered, metadata, **kwargs)
        tampered = copy.deepcopy(saved)
        tampered["state_dict"]["synthetic_weight"][0] += 1.
        with self.assertRaisesRegex(ValueError, "best state/hash"):
            U.validate_ssl_payload(tampered, metadata, **kwargs)
        tampered = copy.deepcopy(saved)
        tampered["ssl_contract"] = {**U.ER.SSL_CONTRACT, "input_normalization": "invented zscore"}
        with self.assertRaisesRegex(ValueError, "normalization contract"):
            U.validate_ssl_payload(tampered, metadata, **kwargs)

    def test_pending_ssl_cannot_enter_supervised_binding(self):
        with patch.object(U.R, "fit_status", return_value={"status": "engineering_pending"}):
            with self.assertRaisesRegex(ValueError, "Same-seed/fold SSL must actually finish"):
                U.bind_ssl_checkpoint(11, 1, {"train": ["a"], "val": ["b"], "test": ["c"]})

    def test_active_bn_buffers_in_best_checkpoint_and_unused_target_frozen(self):
        model = model_fixture()
        U.set_train_mode(model)
        self.assertTrue(model.encoder.norm.training)
        self.assertFalse(model.encoder.unused_target.training)
        keys = U.training_state_keys(model)
        self.assertIn("encoder.norm.running_mean", keys)
        self.assertIn("encoder.norm.num_batches_tracked", keys)
        self.assertNotIn("encoder.unused_target.weight", keys)
        frozen = F.state_digest(U.frozen_state(model))
        initial_buffers = F.cpu_state(model.encoder.norm.state_dict())
        people, raw, labels = data_fixture()
        optimizer = U.optimizer_for(model, "EEGConformer-scratch")
        U.set_learning_rates(optimizer, "EEGConformer-scratch", 1)
        F.subject_update(model, optimizer, people, raw, labels, F.bag_indices(1), micro_subjects=1)
        best = U.selected_state(model)
        self.assertFalse(torch.equal(initial_buffers["running_mean"], model.encoder.norm.running_mean))
        F.subject_update(model, optimizer, people, raw, labels, F.bag_indices(2), micro_subjects=2)
        U.load_selected_state(model, best)
        self.assertEqual(F.state_digest(best), F.state_digest(U.selected_state(model)))
        self.assertEqual(frozen, F.state_digest(U.frozen_state(model)))

    def test_full_resume_next_dropout_update_and_phase_one_encoder_learning(self):
        model = model_fixture()
        U.set_train_mode(model)
        optimizer = U.optimizer_for(model, "EEG2Rep-supervised")
        U.set_learning_rates(optimizer, "EEG2Rep-supervised", 1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 3e-5)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-5)
        self.assertTrue(model.encoder.stem.weight.requires_grad)
        people, raw, labels = data_fixture()
        F.subject_update(model, optimizer, people, raw, labels, F.bag_indices(1), micro_subjects=1)
        payload = {"state_dict": F.cpu_state(model.state_dict()), "optimizer": F.cpu_state(optimizer.state_dict()),
                   "rng": F.rng_state("cpu"), "best": U.selected_state(model),
                   "scheduler": {"policy": "ssl_supervised", "completed_epoch": 1, "extended": False},
                   "sampler_position": {"epoch_completed": 1, "next_batch": 0}}
        stream = io.BytesIO()
        torch.save(payload, stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=False)
        reloaded = model_fixture()
        optimizer_new = U.optimizer_for(reloaded, "EEG2Rep-supervised")
        reloaded.load_state_dict(saved["state_dict"], strict=True)
        optimizer_new.load_state_dict(saved["optimizer"])
        self.assertEqual(F.state_digest(saved["best"]), F.state_digest(U.selected_state(reloaded)))
        def next_update(target, optim):
            F.restore_rng(saved["rng"])
            U.set_train_mode(target)
            U.set_learning_rates(optim, "EEG2Rep-supervised", 2)
            order = [people[i] for i in torch.randperm(len(people)).tolist()]
            F.subject_update(target, optim, order, raw, labels, F.bag_indices(2), micro_subjects=1)
            return order, F.rng_state("cpu")
        order_old, rng_old = next_update(model, optimizer)
        order_new, rng_new = next_update(reloaded, optimizer_new)
        self.assertEqual(order_old, order_new)
        self.assertTrue(F.rng_equal(rng_old, rng_new))
        self.assertEqual(F.state_digest(model.state_dict()), F.state_digest(reloaded.state_dict()))


if __name__ == "__main__":
    unittest.main()

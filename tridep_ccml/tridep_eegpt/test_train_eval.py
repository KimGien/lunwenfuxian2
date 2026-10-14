"""Pure contract tests: no encoder/head forward, optimizer step or cohort fit."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import train_eval as T


def synthetic_context():
    ids = [f"synthetic_{i:02d}" for i in range(37)]
    labels = {sid: i % 2 for i, sid in enumerate(ids)}
    folds, start = {}, 0
    for fold, count in enumerate((7, 8, 7, 8, 7), 1):
        test = ids[start:start + count]
        remaining = [sid for sid in ids if sid not in test]
        folds[f"fold_{fold}"] = {"train": remaining[3:], "val": remaining[:3], "test": test}
        start += count
    return {"subject_ids": ids, "cohort_id": "synthetic-only", "folds_sha256": "synthetic-folds"}, folds, labels


class Contracts(unittest.TestCase):
    def setUp(self):
        self.lock, self.folds, self.labels = synthetic_context()

    def test_original_role_counts_and_once_test(self):
        T.validate_context(self.lock, self.folds, self.labels)
        self.assertEqual(Counter(sum((v["test"] for v in self.folds.values()), [])), Counter(self.lock["subject_ids"]))

    def test_person_overlap_rejected(self):
        self.folds["fold_1"]["train"][0] = self.folds["fold_1"]["test"][0]
        with self.assertRaisesRegex(ValueError, "isolation"):
            T.validate_context(self.lock, self.folds, self.labels)

    def test_training_dataset_never_reads_val_or_test_labels(self):
        train = self.folds["fold_1"]["train"]
        class TrainOnly(dict):
            def __getitem__(self, key):
                if key not in train:
                    raise AssertionError("Held-out label read")
                return super().__getitem__(key)
        indices, dataset = T.training_dataset(self.lock["subject_ids"], train, TrainOnly(self.labels))
        self.assertEqual(dataset.tensors[1].tolist(), [self.labels[s] for s in train])
        self.assertEqual(indices, dataset.tensors[0].tolist())

    def test_batch_order_trainonly_once_every_epoch(self):
        train = [self.lock["subject_ids"].index(s) for s in self.folds["fold_1"]["train"]]
        order = T.np.tile(T.np.asarray(train, dtype=T.np.int64), (60, 1))
        T.validate_batch_order(order, train)
        order[0, 0] = self.lock["subject_ids"].index(self.folds["fold_1"]["val"][0])
        with self.assertRaisesRegex(ValueError, "original train"):
            T.validate_batch_order(order, train)

    def test_pair_rng_and_loader_order_without_model_forward(self):
        train = self.folds["fold_1"]["train"]
        def simulate():
            T.torch.manual_seed(11)
            # Stand in for identical constructor RNG consumption; no model.
            initial = T.torch.rand(19, 7)
            _, dataset = T.training_dataset(self.lock["subject_ids"], train, self.labels)
            loader = T.torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
            order = [[i for batch, _ in loader for i in batch.tolist()] for _ in range(60)]
            return initial, T.np.asarray(order, dtype=T.np.int64)
        initial1, order1 = simulate(); initial2, order2 = simulate()
        self.assertTrue(T.torch.equal(initial1, initial2))
        self.assertTrue(T.np.array_equal(order1, order2))
        T.validate_batch_order(order1, [self.lock["subject_ids"].index(s) for s in train])

    def test_batch_hash_covers_order_and_shape(self):
        a = T.np.arange(24, dtype=T.np.int64).reshape(3, 8)
        self.assertNotEqual(T.batch_order_digest(a), T.batch_order_digest(a[:, ::-1]))

    def test_gate_requires_two_seeds_and_full_classification_checks(self):
        base = dict.fromkeys(T.KEYS, .6)
        candidate = {**base, "macro_f1": .62, "nll": .5, "brier": .5}
        cs = {str(s): {"test": {"macro_f1": .7 if s == 11 else .6}} for s in T.SEEDS}
        bs = {str(s): {"test": {"macro_f1": .6}} for s in T.SEEDS}
        self.assertFalse(T.gate(candidate, base, cs, bs)["passed"])
        cs["22"]["test"]["macro_f1"] = .7
        self.assertTrue(T.gate(candidate, base, cs, bs)["passed"])
        candidate["balanced_accuracy"] = .59
        self.assertFalse(T.gate(candidate, base, cs, bs)["passed"])

    def test_original_tie_policy_and_nll_clip(self):
        m = T.metric([0, 1], [.5, .5])
        self.assertEqual(m["confusion_hc_mdd"], [[1, 0], [1, 0]])
        self.assertEqual(m["auc"], .5)
        m = T.metric([0, 1], [1., 0.])
        self.assertAlmostEqual(m["nll"], -T.math.log(1e-15), places=12)
        self.assertEqual(m["brier"], 1.)


class FilesystemContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out, self.report = self.root / "OUT", self.root / "REPORT"
        self.out.mkdir(); self.report.mkdir()
        self.lock, self.folds, self.labels = synthetic_context()
        self.reg = {"training": {c: T.TRAINING for c in T.CONDITIONS}, "formal_fits": 30,
                    "formal_updates": 3600, "training_gradient_passes": 3600, "fusion_selections": 0}
        def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
        def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        def desc(path): return {"path": Path(path).relative_to(self.root).as_posix(), "sha256": digest(path)}
        def checked(record):
            p = self.root / record["path"]
            if digest(p) != record["sha256"]:
                raise ValueError("Source changed")
            return p
        self.write(self.out / "registration.json", self.reg)
        self.patch = patch.multiple(T.C, WORK=self.root, OUT=self.out, REPORT=self.report,
                read=read, digest=digest, desc=desc, checked=checked, registration=lambda: self.reg,
                context=lambda: (self.lock, self.folds, self.labels))
        self.patch.start()
        self.reg_desc = desc(self.out / "registration.json")
        self.write(self.report / "input_independent_verification.json", {"passed": True, "registration": self.reg_desc})

    def tearDown(self):
        self.patch.stop(); self.temp.cleanup()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def action(self, name, status="started"):
        self.write(self.out / "actions" / (name + ".json"), {"status": status, "registration": self.reg_desc})

    def test_independent_inputs_required_before_any_model_construction(self):
        self.action("fit_all")
        self.write(self.report / "input_independent_verification.json", {"passed": False, "registration": self.reg_desc})
        with patch.object(T, "original_model", side_effect=AssertionError("No model allowed")) as construct:
            with self.assertRaisesRegex(ValueError, "independent input"):
                T.fit_all()
            construct.assert_not_called()

    def test_partial_fit_refused_before_features_or_model(self):
        self.action("fit_all")
        T._folder(T.CONDITIONS[0], T.SEEDS[0], 1).mkdir(parents=True)
        with patch.object(T, "original_model", side_effect=AssertionError("No model allowed")) as construct:
            with self.assertRaisesRegex(ValueError, "automatic retry"):
                T.fit_all()
            construct.assert_not_called()

    def test_all30_barrier_before_validation_test_or_model(self):
        self.action("fit_all", "complete"); self.action("evaluate")
        self.write(self.out / "training_ledger.json", {"fits_completed": 29, "updates_completed": 3480,
                   "gradient_passes_completed": 3480, "fits": [], "registration": self.reg_desc})
        with patch.object(T, "original_model", side_effect=AssertionError("No model allowed")) as construct:
            with self.assertRaisesRegex(ValueError, "all30"):
                T.evaluate()
            construct.assert_not_called()

    def feature_manifest(self, extra=None, dtype=None):
        arrays = {k: T.np.zeros(shape, dtype=T.np.float32) for k, shape in T.FEATURE_ARRAY_SHAPES.items()}
        if extra is not None: arrays[extra] = T.np.array([0], dtype=T.np.float32)
        if dtype is not None: arrays["eegpt"] = arrays["eegpt"].astype(dtype)
        bundle = self.out / "synthetic_features.npz"
        T.np.savez_compressed(bundle, **arrays)
        record = T.C.desc(bundle)
        self.write(self.out / "features_manifest.json", {"passed": True, "registration": self.reg_desc,
                    "records": [{"subject_id": sid, "features": record, "bundle": record} for sid in self.lock["subject_ids"]]})

    def test_only_two_arrays_enter_head_audit_arrays_stay_separate(self):
        self.feature_manifest()
        features, _ = T.load_features(self.lock)
        self.assertEqual(set(features), {"labram", "eegpt"})
        self.assertTrue(all(x.shape == (37, 60, 200) for x in features.values()))

    def test_unknown_key_or_label_payload_refused(self):
        self.feature_manifest(extra="labels")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            T.load_features(self.lock)

    def test_float64_head_features_refused(self):
        self.feature_manifest(dtype=T.np.float64)
        with self.assertRaisesRegex(ValueError, "Invalid frozen"):
            T.load_features(self.lock)

    def test_source_mismatch_refused(self):
        self.feature_manifest()
        bundle = self.out / "synthetic_features.npz"
        with bundle.open("ab") as f: f.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Source changed"):
            T.load_features(self.lock)


if __name__ == "__main__":
    unittest.main()

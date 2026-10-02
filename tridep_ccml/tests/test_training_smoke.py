"""Synthetic runner and metric checks; never reads MODMA subject features."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tridep_ccml.ccml import MODALITIES, ThreeViewHeads
from tridep_ccml.metrics import binary_metrics, reliability_ece5, summarize, uncertainty_diagnostics
from tridep_ccml.records import STAGES, check_fusion, digest
from tridep_ccml import train


class MetricContractChecks(unittest.TestCase):
    def test_binary_metrics_match_hand_calculation_with_hc_tie(self):
        values = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.5, 0.9])
        self.assertAlmostEqual(values["macro_f1"], 0.5)
        self.assertAlmostEqual(values["mdd_recall"], 0.5)
        self.assertAlmostEqual(values["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(values["nll"], -(2 * math.log(.9) + math.log(.4) + math.log(.5)) / 4)
        self.assertAlmostEqual(values["brier"], (.01 + .36 + .25 + .01) / 4)

    def test_ece_bins_include_endpoints_and_use_mdd_event_rate(self):
        values = reliability_ece5([0, 1, 0, 1, 0, 1], [0., .2, .4, .6, .8, 1.])
        self.assertEqual([b["n"] for b in values["bins"]], [1, 1, 1, 1, 2])
        self.assertAlmostEqual(values["ece5"], 0.4)

    def test_risk_coverage_keeps_uncertainty_ties_together(self):
        rows = [
            {"subject_id": "synthetic-a", "u_final": .2, "label": 0, "prediction": 0},
            {"subject_id": "synthetic-b", "u_final": .2, "label": 1, "prediction": 0},
            {"subject_id": "synthetic-c", "u_final": .9, "label": 1, "prediction": 1},
        ]
        values = uncertainty_diagnostics(rows)
        self.assertEqual(len(values["risk_coverage"]), 2)
        self.assertEqual(values["risk_coverage"][0]["accepted"], 2)
        self.assertAlmostEqual(values["risk_coverage"][0]["coverage"], 2 / 3)
        self.assertAlmostEqual(values["risk_coverage"][0]["risk"], .5)
        self.assertAlmostEqual(values["risk_coverage"][1]["risk"], 1 / 3)
        for row in rows:
            row["u_final"] = None
        self.assertIsNone(uncertainty_diagnostics(rows))

    def test_fold_mean_is_not_substituted_by_pooled_oof(self):
        rows = []
        for seed in (11, 22):
            examples = [(1, 0, .1), (1, 1, .9)] + [(2, 0, .9)] * 3 + [(2, 1, .1)] * 3
            for i, (fold, label, probability) in enumerate(examples):
                rows.append({"experiment": "SYNTHETIC-SMOKE", "seed": seed, "outer_fold": fold,
                             "subject_id": f"synthetic-{i}", "split": "test", "label": label,
                             "p_final": [1 - probability, probability], "prediction": int(probability > .5),
                             "u_final": None})
        report = summarize(rows)
        self.assertEqual(report["independent_subjects"], 8)
        self.assertAlmostEqual(report["primary_seed_mean_of_fold_means"]["macro_f1"], .5)
        self.assertAlmostEqual(report["separate_seed_mean_of_pooled_oof"]["macro_f1"], .25)


class SyntheticRunnerSmoke(unittest.TestCase):
    """Two-epoch smoke of actual runner functions using synthetic metadata.

    The production CLI retains the locked 60-epoch policy. Only this in-memory
    fixture has ten synthetic IDs and two epochs; no MODMA file is opened.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tc-synthetic-smoke-")
        self.addCleanup(self.temporary.cleanup)
        ids = [f"synthetic-{i:02d}" for i in range(10)]
        train_ids = [ids[i] for i in (8, 1, 4, 2, 9, 0)]
        val_ids = [ids[3], ids[5]]
        test_ids = [ids[6], ids[7]]
        config = {"epochs": 2, "batch_size": 4, "learning_rate": .001,
                  "weight_decay": 0., "gradient_clip": 1., "rho": 1.25,
                  "delta": 1., "gamma": 1., "anneal_epochs": 50,
                  "seeds": [11], "folds": [1]}
        labels = np.array([0, 1] * 5, dtype=np.int64)
        manifest = {"npz_sha256": "a" * 64,
                    "parent_checkpoints": {m: {"path": f"synthetic/{m}.pt", "sha256": "b" * 64}
                                           for m in MODALITIES},
                    "source_features": {m: {sid: {"path": f"synthetic/{m}/{sid}.npy", "sha256": "c" * 64}
                                             for sid in ids} for m in MODALITIES}}
        self.data = {"features": np.random.default_rng(11).normal(size=(10, 3, 128)).astype(np.float32),
                     "subject_ids": ids, "labels": labels,
                     "splits": ["train" if sid in train_ids else "val" if sid in val_ids else "test" for sid in ids],
                     "manifest": manifest, "manifest_hash": "d" * 64}
        self.context = SimpleNamespace(root=Path(self.temporary.name), config=config,
            lock={"cohort_id": "SYNTHETIC-SMOKE-NOT-MODMA"}, cohort_sha256="e" * 64,
            folds_sha256="f" * 64, config_sha256="1" * 64,
            folds={"fold_1": {"train": train_ids, "val": val_ids, "test": test_ids}},
            ids=ids, labels=dict(zip(ids, labels.tolist())), experiments=list(STAGES),
            manifest=lambda seed, fold: (manifest, self.data["manifest_hash"]))
        self.context.check_file = lambda descriptor: self.assertEqual(digest(descriptor["path"]), descriptor["sha256"])
        self.device = torch.device("cpu")

    def read_rows(self, experiment):
        path = train.run_directory(self.context.root, experiment, 11, 1) / "predictions.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_paired_training_split_isolation_reload_and_tc03_reuse(self):
        stages = ("TC-01-37-fulltext", "TC-02-37-fulltext", "TC-04-37-fulltext")
        original_features = self.data["features"].copy()
        original_labels = self.data["labels"].copy()
        captured = {}
        original_fit = train.fit_heads

        def capture_fit(context, data, experiment, seed, fold, directory, device):
            result = original_fit(context, data, experiment, seed, fold, directory, device)
            captured[experiment] = result
            return result

        with patch.object(train, "fit_heads", side_effect=capture_fit):
            reports = {stage: train.run_one(self.context, self.data, stage, 11, 1, self.device)
                       for stage in stages}

        initial_hash = train.state_digest(train.make_initial_state(11))
        self.assertEqual({report["source_metadata"]["initialization_sha256"] for report in reports.values()}, {initial_hash})
        self.assertEqual(len({report["source_metadata"]["batch_order_sha256"] for report in reports.values()}), 1)
        # Independently reconstruct the expected shuffle of the six-item train list.
        generator = torch.Generator().manual_seed(11)
        expected_batch_hash = hashlib.sha256()
        for _ in range(2):
            expected_batch_hash.update(torch.randperm(6, generator=generator).numpy().astype("<i8").tobytes())
        for stage, report in reports.items():
            with self.subTest(stage=stage):
                model, checkpoint_path, metadata = captured[stage]
                self.assertEqual(metadata["batch_order_sha256"], expected_batch_hash.hexdigest())
                self.assertEqual(metadata["optimizer_updates"], 4)
                self.assertEqual(metadata["train_ids"], self.context.folds["fold_1"]["train"])
                self.assertEqual(metadata["completed_epochs"], 2)
                self.assertTrue(report["training_performed"])
                self.assertNotEqual(train.state_digest(model.state_dict()), initial_hash)

                # Changing every held-out feature and label must not change fitted weights.
                altered = copy.deepcopy(self.data)
                held_out = [i for i, sid in enumerate(self.data["subject_ids"])
                            if sid not in self.context.folds["fold_1"]["train"]]
                altered["features"][held_out] *= -1e6
                altered["labels"][held_out] = 1 - altered["labels"][held_out]
                counterfactual_dir = self.context.root / "synthetic_counterfactual" / stage
                counterfactual_dir.mkdir(parents=True)
                counterfactual, _, _ = original_fit(self.context, altered, stage, 11, 1,
                                                    counterfactual_dir, self.device)
                self.assertEqual(train.state_digest(counterfactual.state_dict()), train.state_digest(model.state_dict()))

                restored = ThreeViewHeads()
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                restored.load_state_dict(checkpoint["state_dict"])
                before = train.predict_records(self.context, self.data, model, stage, 11, 1,
                                               checkpoint_path, metadata, self.device)
                after = train.predict_records(self.context, self.data, restored, stage, 11, 1,
                                              checkpoint_path, metadata, self.device)
                self.assertEqual(before, after)
                self.assertEqual(after, self.read_rows(stage))
                for row in after:
                    check_fusion(row)

        np.testing.assert_array_equal(self.data["features"], original_features)
        np.testing.assert_array_equal(self.data["labels"], original_labels)

        tc02_checkpoint = captured["TC-02-37-fulltext"][1]
        tc02_hash = digest(tc02_checkpoint)
        tc03_stages = ("TC-03-mean-37-fulltext", "TC-03-sum-37-fulltext", "TC-03-ccml-37-fulltext")
        # Exercise the real dispatch path; any optimization attempt must fail.
        with patch.object(train, "fit_heads", side_effect=AssertionError("TC03 cannot train")), \
             patch.object(torch.optim, "Adamax", side_effect=AssertionError("TC03 cannot create an optimizer")):
            tc03_reports = {stage: train.run_one(self.context, self.data, stage, 11, 1, self.device)
                            for stage in tc03_stages}
        tc02_views = [row["views"] for row in self.read_rows("TC-02-37-fulltext")]
        for stage, report in tc03_reports.items():
            self.assertFalse(report["training_performed"])
            self.assertEqual(report["checkpoint_sha256"], tc02_hash)
            self.assertEqual(Path(report["checkpoint_path"]), tc02_checkpoint)
            self.assertEqual(report["source_metadata"]["experiment"], "TC-02-37-fulltext")
            directory = train.run_directory(self.context.root, stage, 11, 1)
            self.assertFalse((directory / "checkpoint.pt").exists())
            self.assertFalse((directory / "train_history.json").exists())
            rows = self.read_rows(stage)
            self.assertEqual([row["views"] for row in rows], tc02_views)
            for row in rows:
                self.assertEqual(row["checkpoint_sha256"], tc02_hash)
                check_fusion(row)
        self.assertEqual(digest(tc02_checkpoint), tc02_hash)
        mean_rows, sum_rows = self.read_rows(tc03_stages[0]), self.read_rows(tc03_stages[1])
        self.assertEqual([row["prediction"] for row in mean_rows], [row["prediction"] for row in sum_rows])
        self.assertTrue(all(total["u_final"] < mean["u_final"] for mean, total in zip(mean_rows, sum_rows)))

    def test_ablation_dispatch_preserves_common_initialization_and_record_parameters(self):
        settings = {"TC-05a-37-fulltext": (1., 1., 1.),
                    "TC-05b-37-fulltext": (1.25, 0., 1.),
                    "TC-05c-37-fulltext": (1.25, 1., 0.)}
        initial_hash = train.state_digest(train.make_initial_state(11))
        for stage, parameters in settings.items():
            report = train.run_one(self.context, self.data, stage, 11, 1, self.device)
            self.assertEqual(report["source_metadata"]["initialization_sha256"], initial_hash)
            self.assertEqual(report["source_metadata"]["optimizer_updates"], 4)
            for row in self.read_rows(stage):
                self.assertEqual(tuple(row[name] for name in ("rho", "delta", "gamma")), parameters)
                self.assertIsNotNone(row["e_cmp"])
                check_fusion(row)


if __name__ == "__main__":
    unittest.main(verbosity=2)

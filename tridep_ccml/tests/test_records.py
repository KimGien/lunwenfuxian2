"""Synthetic prediction-contract rejection tests; never trains on real data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tridep_ccml.records import (MODALITIES, STAGES, RecordContext, check_fusion,
                                 digest, opinion, validate_prediction_records)


def basic_row():
    return {"experiment": "TC-00-mean-37-fulltext", "cohort_id": "MODMA-37-v1",
            "cohort_sha256": "a" * 64, "folds_sha256": "b" * 64,
            "subject_id": "00000003", "seed": 11, "outer_fold": 1, "split": "test", "label": 1,
            "train_ids": ["00000001"], "parent_checkpoints": {}, "source_features": {},
            "feature_manifest_sha256": None, "feature_npz_sha256": None,
            "training_config_sha256": "c" * 64, "checkpoint_sha256": None,
            "initialization_sha256": None, "fusion_method": "mean_probability",
            "fusion_weights": [1 / 3] * 3,
            "views": {m: {"e": None, "alpha": None, "p": [.25, .75], "u": None} for m in MODALITIES},
            "m": None, "e_con": None, "e_cmp": None, "e_final": None,
            "p_final": [.25, .75], "u_final": None, "prediction": 1,
            "rho": None, "delta": None, "gamma": None}


def ccml_row(experiment="TC-04-37-fulltext"):
    row = basic_row()
    row.update(experiment=experiment, fusion_method="ccml", fusion_weights=None,
               rho=1.25, delta=1., gamma=0. if experiment == "TC-05c-37-fulltext" else 1.)
    es = [[2., 1.], [1., 3.], [4., 2.]]
    for m, e in zip(MODALITIES, es):
        a, p, u = opinion(e)
        row["views"][m] = {"e": e, "alpha": a, "p": p, "u": u}
    row.update(m=[1., 1.], e_con=[3., 3.], e_cmp=[4 / 3, 1.], e_final=[13 / 3, 4.])
    _, row["p_final"], row["u_final"] = opinion(row["e_final"])
    row["prediction"] = 0
    return row


class FusionContractTests(unittest.TestCase):
    def test_valid_probability_and_evidence_records(self):
        check_fusion(basic_row())
        check_fusion(ccml_row())

    def test_probability_fusion_cannot_invent_uncertainty(self):
        row = basic_row()
        row["u_final"] = .3
        with self.assertRaisesRegex(ValueError, "fabricate"):
            check_fusion(row)

    def test_evidence_probability_consistency(self):
        row = ccml_row()
        row["views"]["eeg"]["p"] = [.2, .8]
        with self.assertRaisesRegex(ValueError, "eeg.p"):
            check_fusion(row)

    def test_decomposition_rejects_subtracting_scaled_minimum(self):
        row = ccml_row()
        row["e_cmp"] = [0., 0.]
        with self.assertRaisesRegex(ValueError, "e_cmp"):
            check_fusion(row)

    def test_gamma_zero_still_requires_complementary_inference(self):
        row = ccml_row("TC-05c-37-fulltext")
        check_fusion(row)
        row["e_final"] = row["e_con"]
        with self.assertRaisesRegex(ValueError, "e_final"):
            check_fusion(row)

    def test_nonfinite_values_and_wrong_prediction_rejected(self):
        for value in (float("nan"), float("inf"), -1.):
            row = ccml_row()
            row["views"]["text"]["e"][0] = value
            with self.assertRaises(ValueError):
                check_fusion(row)
        row = basic_row()
        row["prediction"] = 0
        with self.assertRaisesRegex(ValueError, "threshold"):
            check_fusion(row)

    def test_all_zero_evidence_is_uniform_maximum_uncertainty_hc(self):
        row = ccml_row()
        for m in MODALITIES:
            row["views"][m] = {"e": [0., 0.], "alpha": [1., 1.], "p": [.5, .5], "u": 1.}
        for key in ("m", "e_con", "e_cmp", "e_final"):
            row[key] = [0., 0.]
        row.update(p_final=[.5, .5], u_final=1., prediction=0)
        check_fusion(row)
        row["prediction"] = 1
        with self.assertRaises(ValueError):
            check_fusion(row)


class ProvenanceContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "source.bin"
        source.write_bytes(b"synthetic source")
        descriptor = {"path": str(source), "sha256": digest(source)}
        self.row = basic_row()
        self.row["parent_checkpoints"] = {m: dict(descriptor) for m in MODALITIES}
        self.row["source_features"] = {m: dict(descriptor) for m in MODALITIES}
        self.context = SimpleNamespace(root=self.root, workspace=self.root, _hashes={},
            experiments=list(STAGES), lock={"cohort_id": "MODMA-37-v1"},
            cohort_sha256="a" * 64, folds_sha256="b" * 64, config_sha256="c" * 64,
            ids=["00000001", "00000002", "00000003"],
            config={"seeds": [11], "folds": [1]},
            labels={"00000001": 0, "00000002": 0, "00000003": 1},
            folds={"fold_1": {"train": ["00000001"], "val": ["00000002"], "test": ["00000003"]}})
        self.context.check_file = lambda d: RecordContext.check_file(self.context, d)

    def validate(self, rows):
        return validate_prediction_records(rows, self.row["experiment"], context=self.context,
                                           require_complete=False)

    def test_real_source_digest_and_valid_row(self):
        self.assertTrue(self.validate([self.row])["passed"])

    def test_wrong_cohort_fold_subject_label_split_and_train_ids(self):
        changes = {"cohort_sha256": "d" * 64, "folds_sha256": "e" * 64,
                   "subject_id": "02010004", "label": 0, "split": "train",
                   "train_ids": ["00000003"], "training_config_sha256": "f" * 64}
        for key, value in changes.items():
            with self.subTest(key=key):
                row = copy.deepcopy(self.row)
                row[key] = value
                with self.assertRaises(ValueError):
                    self.validate([row])

    def test_unregistered_experiment_and_duplicate_subject(self):
        with self.assertRaises(ValueError):
            validate_prediction_records([self.row], "TC-06-37-fulltext", context=self.context,
                                        require_complete=False)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.validate([self.row, self.row])

    def test_changed_source_hash_rejected(self):
        self.row["source_features"]["speech"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Source hash changed"):
            self.validate([self.row])

    def test_partial_results_cannot_claim_complete_cohort(self):
        with self.assertRaisesRegex(ValueError, "37 unique"):
            validate_prediction_records([self.row], self.row["experiment"], context=self.context)

    def test_schema_is_valid_json_and_matches_record_fields(self):
        schema = json.loads((Path(__file__).resolve().parents[1] / "configs/prediction.schema.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(set(schema["required"]), set(self.row))
        self.assertEqual(set(schema["properties"]), set(self.row))
        self.assertEqual(set(schema["properties"]["experiment"]["enum"]), set(STAGES))


if __name__ == "__main__":
    unittest.main()

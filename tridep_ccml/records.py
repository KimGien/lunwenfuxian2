"""Independent TC prediction contract; never registers experiments in S00.

JSON Schema checks structure; these validators additionally check the locked
cohort, physical source hashes, split membership and evidence arithmetic.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
MODALITIES = ("eeg", "speech", "text")
FOLDS_SHA256 = "a28ebe92c94428128313586d3a5524b2f03befcbc473215f4b0b2dcb49c3284d"
STAGES = {
    "TC-00-weighted-37-fulltext": "weighted_probability",
    "TC-00-mean-37-fulltext": "mean_probability",
    "TC-01-37-fulltext": "mean_probability",
    "TC-02-37-fulltext": "mean_probability",
    "TC-03-mean-37-fulltext": "mean_evidence",
    "TC-03-sum-37-fulltext": "sum_evidence",
    "TC-03-ccml-37-fulltext": "ccml",
    "TC-04-37-fulltext": "ccml",
    "TC-05a-37-fulltext": "ccml",
    "TC-05b-37-fulltext": "ccml",
    "TC-05c-37-fulltext": "ccml",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name):
    if isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected),
                f"{name}: invalid shape")
        for a, b in zip(actual, expected):
            close(a, b, name)
    else:
        require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=2e-5, abs_tol=2e-6),
                f"{name}: inconsistent value")


def vector(value, name, probability=False):
    require(isinstance(value, list) and len(value) == 2, f"{name}: expected [HC,MDD]")
    require(all(isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) and v >= 0 for v in value), f"{name}: invalid values")
    if probability:
        require(all(v <= 1 for v in value), f"{name}: invalid probability")
        close(sum(value), 1., name)
    return value


def opinion(evidence):
    alpha = [v + 1. for v in evidence]
    strength = sum(alpha)
    return alpha, [v / strength for v in alpha], 2. / strength


def check_fusion(row):
    """Check all applicable fields, including exact method/stage semantics."""
    method = row["fusion_method"]
    require(row["experiment"] in STAGES and STAGES[row["experiment"]] == method,
            "Experiment/fusion_method mismatch")
    require(set(row["views"]) == set(MODALITIES), "Expected exactly three views")
    evidential = not row["experiment"].startswith(("TC-00-", "TC-01-"))
    es, ps = [], []
    for modality in MODALITIES:
        view = row["views"][modality]
        require(set(view) == {"e", "alpha", "p", "u"}, "Incomplete view fields")
        p = vector(view["p"], f"{modality}.p", probability=True)
        ps.append(p)
        if evidential:
            e = vector(view["e"], f"{modality}.e")
            a, expected_p, u = opinion(e)
            close(view["alpha"], a, f"{modality}.alpha")
            close(p, expected_p, f"{modality}.p")
            close(view["u"], u, f"{modality}.u")
            es.append(e)
        else:
            require(all(view[k] is None for k in ("e", "alpha", "u")),
                    "Ordinary probability has no Dirichlet evidence or uncertainty")
    p_final = vector(row["p_final"], "p_final", probability=True)
    joint = row["experiment"].startswith(("TC-04-", "TC-05"))
    expected_parameters = {"rho": 1.25 if method == "ccml" else None,
                           "delta": 1. if joint else None, "gamma": 1. if joint else None}
    if row["experiment"] == "TC-05a-37-fulltext":
        expected_parameters["rho"] = 1.
    if row["experiment"] == "TC-05b-37-fulltext":
        expected_parameters["delta"] = 0.
    if row["experiment"] == "TC-05c-37-fulltext":
        expected_parameters["gamma"] = 0.
    for k, value in expected_parameters.items():
        if value is None:
            require(row[k] is None, f"Unused {k} must be null")
        else:
            close(row[k], value, k)
    if method in ("mean_probability", "weighted_probability"):
        require(all(row[k] is None for k in ("m", "e_con", "e_cmp", "e_final", "u_final")),
                "Probability fusion must not fabricate evidence or u")
        weights = row["fusion_weights"]
        require(isinstance(weights, list) and len(weights) == 3 and
                all(isinstance(w, (int, float)) and math.isfinite(w) and 0 <= w <= 1
                    for w in weights), "Invalid probability fusion weights")
        close(sum(weights), 1., "fusion_weights")
        if method == "mean_probability":
            close(weights, [1 / 3] * 3, "equal probability weights")
        close(p_final, [sum(weights[v] * ps[v][k] for v in range(3)) for k in range(2)],
              "p_final")
    else:
        require(evidential and row["fusion_weights"] is None,
                "Evidence fusion has no scalar probability weights")
        if method == "ccml":
            m = [min(e[k] for e in es) for k in range(2)]
            cmp = [sum(e[k] - m[k] for e in es) / 3 for k in range(2)]
            # Scale first to avoid overflow/underflow in the independent verifier.
            maximum = max(m)
            powers = [(v / maximum) ** row["rho"] for v in m] if maximum else [0., 0.]
            denominator = sum(powers)
            con = [3 * sum(m) * v / denominator for v in powers] if denominator else [0., 0.]
            close(row["m"], m, "m")
            close(row["e_con"], con, "e_con")
            close(row["e_cmp"], cmp, "e_cmp")
            final = [a + b for a, b in zip(con, cmp)]
        else:
            require(all(row[k] is None for k in ("m", "e_con", "e_cmp")),
                    "Non-CCML inference must leave decomposition fields null")
            divisor = 3 if method == "mean_evidence" else 1
            final = [sum(e[k] for e in es) / divisor for k in range(2)]
        close(vector(row["e_final"], "e_final"), final, "e_final")
        _, expected_p, expected_u = opinion(final)
        close(p_final, expected_p, "p_final")
        close(row["u_final"], expected_u, "u_final")
    require(row["prediction"] == int(p_final[1] > .5), "Prediction/threshold mismatch")


class RecordContext:
    """Load immutable source lock and TC registry once; cache physical file hashes."""

    def __init__(self, root=ROOT, s00_root=None):
        self.root = Path(root).resolve()
        self.workspace = self.root.parent
        self.s00 = Path(s00_root).resolve() if s00_root else self.workspace / "S00"
        self.cohort_path = self.s00 / "configs/cohort.json"
        self.lock = read_json(self.cohort_path)
        self.cohort_sha256 = digest(self.cohort_path)
        self.folds_path = self.s00 / self.lock["fold_file"]
        self.folds_sha256 = digest(self.folds_path)
        require(self.folds_sha256 == self.lock["folds_sha256"] == FOLDS_SHA256,
                "Locked S00 fold hash changed")
        self.folds = read_json(self.folds_path)
        self.ids = self.lock["subject_ids"]
        require(len(self.ids) == len(set(self.ids)) == 37 and "02010004" not in self.ids,
                "TC requires exactly locked MODMA-37")
        original = self.s00 / self.lock["official_fold_file"]
        require(digest(original) == self.lock["official_folds_sha256"],
                "Archived official fold hash changed")
        expected = {n: {p: [s for s in ids if s in self.ids] for p, ids in fold.items()}
                    for n, fold in read_json(original).items()}
        require(self.folds == expected, "TC must retain original fold assignments and order")
        subjects_path = self.s00 / "manifests/subjects.json"
        subjects = read_json(subjects_path)
        self.labels = {r["subject_id"]: r["label"] for r in subjects}
        require(len(subjects) == 37 and set(self.labels) == set(self.ids), "Subject manifest differs")
        require(dict(Counter(str(v) for v in self.labels.values())) == self.lock["class_counts"],
                "Subject diagnoses differ")
        audit = read_json(self.s00 / "reports/modma_audit.json")
        require(audit.get("passed") and audit.get("cohort_id") == self.lock["cohort_id"]
                and audit.get("folds_sha256") == self.folds_sha256
                and audit.get("subject_manifest_sha256") == digest(subjects_path)
                and audit.get("cohort") == 37 and audit.get("wav_count") == 1073,
                "Missing, failed, or stale MODMA data audit")
        self.registry = read_json(self.root / "configs/registry.json")
        require(self.registry["cohort_id"] == self.lock["cohort_id"]
                and self.registry["source_cohort_sha256"] == self.cohort_sha256
                and self.registry["folds_sha256"] == self.folds_sha256,
                "Independent TC registry must bind original cohort and folds")
        self.experiments = self.registry["experiment_ids"]
        require(set(self.experiments) == set(STAGES), "Unexpected TC registry stage set")
        self.config_path = self.root / "configs/experiment.json"
        self.config = read_json(self.config_path)
        self.config_sha256 = digest(self.config_path)
        self._hashes = {}
        self._manifests = {}

    def check_file(self, descriptor):
        require(isinstance(descriptor, dict) and set(descriptor) >= {"path", "sha256"},
                "Source descriptor requires path and SHA256")
        path = Path(descriptor["path"])
        if not path.is_absolute():
            path = self.workspace / path
        key = str(path.resolve())
        if key not in self._hashes:
            self._hashes[key] = digest(path)
        require(self._hashes[key] == descriptor["sha256"], f"Source hash changed: {path}")

    def manifest(self, seed, fold):
        key = (seed, fold)
        if key not in self._manifests:
            path = self.root / "features" / f"seed_{seed}" / f"fold_{fold}.json"
            manifest = read_json(path)
            require(manifest["seed"] == seed and manifest["fold"] == fold, "Feature seed/fold mismatch")
            require(manifest["cohort_id"] == self.lock["cohort_id"]
                    and manifest["cohort_sha256"] == self.cohort_sha256
                    and manifest["folds_sha256"] == self.folds_sha256,
                    "Feature cohort/fold hash mismatch")
            require(manifest["train_ids"] == self.folds[f"fold_{fold}"]["train"],
                    "Feature fitting train IDs mismatch")
            require(manifest["modalities"] == list(MODALITIES) and manifest["frozen_eval"] is True,
                    "Feature modality order or frozen/eval policy changed")
            require(manifest["exporter_sha256"] == digest(self.root / "prepare.py"),
                    "Feature exporter source changed; re-audit preparation")
            require(manifest["upstream_model_source_sha256"] ==
                    digest(self.s00 / "upstream-dev/lib/models/models.py"),
                    "Parent model source changed; re-audit preparation")
            self.check_file({"path": str(path.with_suffix(".npz")), "sha256": manifest["npz_sha256"]})
            require(set(manifest["parent_checkpoints"]) == set(MODALITIES), "Missing parent modality")
            require(set(manifest["source_features"]) == set(MODALITIES), "Missing source modality")
            for modality in MODALITIES:
                self.check_file(manifest["parent_checkpoints"][modality])
                require(set(manifest["source_features"][modality]) == set(self.ids),
                        "Source feature subject set mismatch")
                for descriptor in manifest["source_features"][modality].values():
                    self.check_file(descriptor)
            self._manifests[key] = (manifest, digest(path))
        return self._manifests[key]


def validate_prediction_records(rows, experiment, root=ROOT, require_complete=True,
                                context=None):
    context = context or RecordContext(root)
    require(experiment in context.experiments and experiment in STAGES, "Unregistered experiment")
    require(bool(rows), "Empty prediction records")
    seen, tests, folds_seen = set(), {}, {}
    required = {"experiment", "cohort_id", "cohort_sha256", "folds_sha256", "subject_id", "seed",
                "outer_fold", "split", "label", "train_ids", "parent_checkpoints", "source_features",
                "feature_manifest_sha256", "feature_npz_sha256", "training_config_sha256",
                "checkpoint_sha256", "initialization_sha256", "fusion_method", "fusion_weights",
                "views", "m", "e_con", "e_cmp", "e_final", "p_final", "u_final", "prediction",
                "rho", "delta", "gamma"}
    for row in rows:
        require(set(row) == required, "Prediction record has missing or unexpected fields")
        require(row["experiment"] == experiment and row["cohort_id"] == context.lock["cohort_id"]
                and row["cohort_sha256"] == context.cohort_sha256
                and row["folds_sha256"] == context.folds_sha256, "Prediction cohort provenance mismatch")
        seed, fold, sid = row["seed"], row["outer_fold"], row["subject_id"]
        require(seed in context.config["seeds"] and fold in context.config["folds"],
                "Unregistered seed/fold")
        assignment = context.folds[f"fold_{fold}"]
        require(row["split"] in assignment and sid in assignment[row["split"]]
                and context.labels.get(sid) == row["label"], "Subject/label/split mismatch")
        require(row["train_ids"] == assignment["train"], "Training subject provenance mismatch")
        key = (seed, fold, sid)
        require(key not in seen, "Duplicate subject prediction")
        seen.add(key)
        folds_seen.setdefault((seed, fold), []).append(sid)
        tests.setdefault(seed, [])
        if row["split"] == "test":
            tests[seed].append(sid)
        require(row["training_config_sha256"] == context.config_sha256, "Training config hash mismatch")
        if experiment.startswith("TC-00-"):
            # Historical baseline provenance is its original checkpoint/features;
            # no artificial new head checkpoint or representation is required.
            require(all(row[k] is None for k in ("checkpoint_sha256", "initialization_sha256",
                        "feature_manifest_sha256", "feature_npz_sha256")), "TC00 cannot claim new heads")
            require(set(row["parent_checkpoints"]) == set(MODALITIES)
                    and set(row["source_features"]) == set(MODALITIES), "Missing TC00 source modality")
            for descriptor in list(row["parent_checkpoints"].values()) + list(row["source_features"].values()):
                context.check_file(descriptor)
        else:
            manifest, manifest_hash = context.manifest(seed, fold)
            require(row["feature_manifest_sha256"] == manifest_hash
                    and row["feature_npz_sha256"] == manifest["npz_sha256"], "Feature provenance mismatch")
            require(row["parent_checkpoints"] == manifest["parent_checkpoints"], "Parent checkpoint mismatch")
            expected_source = {m: manifest["source_features"][m][sid] for m in MODALITIES}
            require(row["source_features"] == expected_source, "Subject source feature mismatch")
            for k in ("checkpoint_sha256", "initialization_sha256"):
                require(isinstance(row[k], str) and len(row[k]) == 64
                        and all(c in "0123456789abcdef" for c in row[k]), f"Invalid {k}")
            source_experiment = "TC-02-37-fulltext" if experiment.startswith("TC-03-") else experiment
            checkpoint = context.root / "runs" / source_experiment / f"seed_{seed}" / f"fold_{fold}" / "checkpoint.pt"
            context.check_file({"path": str(checkpoint), "sha256": row["checkpoint_sha256"]})
            sidecar = read_json(checkpoint.with_suffix(".json"))
            require(sidecar["checkpoint_sha256"] == row["checkpoint_sha256"], "Head checkpoint sidecar mismatch")
            expected_meta = {"experiment": source_experiment, "seed": seed, "fold": fold,
                             "cohort_id": context.lock["cohort_id"], "cohort_sha256": context.cohort_sha256,
                             "folds_sha256": context.folds_sha256, "train_ids": row["train_ids"],
                             "feature_manifest_sha256": row["feature_manifest_sha256"],
                             "feature_npz_sha256": row["feature_npz_sha256"],
                             "parent_checkpoints": row["parent_checkpoints"],
                             "training_config_sha256": row["training_config_sha256"],
                             "initialization_sha256": row["initialization_sha256"]}
            require(all(sidecar["metadata"].get(k) == value for k, value in expected_meta.items()),
                    "Head checkpoint provenance differs from prediction")
        check_fusion(row)
    if require_complete:
        require(set(tests) == set(context.config["seeds"]), "Missing planned seeds")
        require(all(len(sids) == 37 and set(sids) == set(context.ids) for sids in tests.values()),
                "Each seed must include exactly 37 unique test subjects")
        require(set(folds_seen) == {(s, f) for s in context.config["seeds"] for f in context.config["folds"]},
                "Missing planned folds")
    return {"passed": True, "experiment": experiment, "records": len(rows),
            "seeds": sorted(tests), "complete": require_complete,
            "test_subjects_by_seed": {str(s): len(v) for s, v in tests.items()}}

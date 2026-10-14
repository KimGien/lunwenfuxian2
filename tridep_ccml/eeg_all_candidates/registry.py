"""Immutable 360-fit registration and an evidence-based global test gate.

CLI: ``python -m tridep_ccml.eeg_all_candidates.registry register|status|test-gate``.
Runtime fit statuses live separately from registration.json, which is never
updated. A failed/blocked/engineering probe cannot unlock external testing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from . import protocol as P

REGISTRATION = P.OUT / "registration.json"
STATUS_VALUES = ("engineering_pending", "ready", "training", "training_complete", "blocked", "failed")
COMPLETION_ARTIFACTS = ("checkpoint", "train_history", "validation_history", "metadata")
COMPLETION_CHECKS = ("passed", "checkpoint_reload_verified", "parameter_update_verified", "frozen_parameters_verified")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def descriptor(path) -> dict:
    resolved = Path(path).resolve()
    _require(resolved.is_relative_to(P.WORK), "Artifact is outside the EEG workspace")
    return {"path": resolved.relative_to(P.WORK).as_posix(), "sha256": digest(resolved)}


def checked(record) -> Path:
    _require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "Expected path/SHA256 descriptor")
    resolved = (P.WORK / record["path"]).resolve()
    _require(resolved.is_relative_to(P.WORK) and resolved.is_file(), "Source missing or outside workspace")
    _require(digest(resolved) == record["sha256"], "Source bytes changed: " + str(resolved))
    return resolved


def _write_new(path: Path, value) -> None:
    """Exclusive creation never replaces an existing immutable record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _replace_status(path: Path, value) -> None:
    """Serialize once; file replacement retries never repeat training work."""
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.1 * (attempt + 1))


def fit_id(condition: str, seed: int, fold: int) -> str:
    _require(condition in P.CONDITION_BY_ID and seed in P.SEEDS and fold in P.FOLDS,
             "Unregistered condition, seed or fold")
    return f"{condition}/seed_{seed}/fold_{fold}"


def fit_directory(condition: str, seed: int, fold: int) -> Path:
    return P.OUT / "runs" / fit_id(condition, seed, fold)


def _cohort_context() -> tuple[dict, dict, dict, dict]:
    cohort_path = P.WORK / "S00/configs/cohort.json"
    cohort = read_json(cohort_path)
    folds_path = P.WORK / "S00" / cohort["fold_file"]
    folds = read_json(folds_path)
    copied_folds = P.WORK / "tridep_ccml/manifests/fold_assignments.json"
    copied_cohort = P.WORK / "tridep_ccml/manifests/cohort.json"
    _require(digest(folds_path) == cohort["folds_sha256"] == P.FOLDS_SHA256,
             "Original37 five-fold source changed")
    _require(read_json(copied_folds) == folds and read_json(copied_cohort) == cohort,
             "TriDep copies and active S00 cohort/folds disagree")
    ids = cohort["subject_ids"]
    labels = {r["subject_id"]: r["label"] for r in read_json(P.WORK / "S00/manifests/subjects.json")}
    _require(len(ids) == len(set(ids)) == 37 and set(labels) == set(ids)
             and Counter(labels.values()) == Counter({0: 21, 1: 16}), "Locked37 people/labels differ")
    _require(set(folds) == {f"fold_{f}" for f in P.FOLDS}, "Requires unchanged five folds")
    held_out = []
    for split in folds.values():
        _require(set(split) == {"train", "val", "test"}, "Fold roles differ")
        _require(len(split["train"]) in (26, 27) and len(split["val"]) == 3 and len(split["test"]) in (7, 8), "Fold role counts differ")
        flat = [s for part in ("train", "val", "test") for s in split[part]]
        _require(len(flat) == len(set(flat)) == 37 and set(flat) == set(ids), "People cross fold roles")
        _require(all({labels[s] for s in split[part]} == {0, 1} for part in split), "Each fold role must contain both classes")
        held_out.extend(split["test"])
    _require(Counter(held_out) == Counter(ids), "Each person must have exactly one outer test role")
    official = P.WORK / "S00" / cohort["official_fold_file"]
    _require(digest(official) == cohort["official_folds_sha256"], "Archived38 source changed")
    retained = {key: {part: [sid for sid in values if sid in set(ids)] for part, values in split.items()}
                for key, split in read_json(official).items()}
    _require(retained == folds, "Fixed folds no longer retain original order after exclusion")
    sources = {name: descriptor(path) for name, path in {
        "cohort": cohort_path, "folds": folds_path, "cohort_copy": copied_cohort,
        "folds_copy": copied_folds, "labels": P.WORK / "S00/manifests/subjects.json",
        "official38_folds": official,
    }.items()}
    return cohort, folds, labels, sources


def create_registration() -> dict:
    """Create once; an existing registration is an error, never overwritten."""
    _require(not REGISTRATION.exists(), "Registration already exists; load it instead of overwriting")
    cohort, folds, labels, sources = _cohort_context()
    raw_path = P.WORK / "tridep_ccml/experiments/TriDep-Stage1-EEG-37-fulltext/input_manifest.json"
    raw = read_json(raw_path)
    _require(raw["subject_ids"] == cohort["subject_ids"] and raw["folds"] == folds
             and raw["channel_names"] == list(P.CHANNELS), "Accepted E2v2 raw people/folds/channels differ")
    raw_records = raw["records"]
    _require(len(raw_records) == 37 and [row["subject_id"] for row in raw_records] == cohort["subject_ids"], "Raw manifest coverage/order differs")
    _require(all(row["label"] == labels[row["subject_id"]] for row in raw_records), "Raw manifest labels differ")
    sources.update({"plan": descriptor(P.PLAN), "protocol_source": descriptor(Path(P.__file__)),
                    "registry_source": descriptor(Path(__file__)), "accepted_raw_manifest": descriptor(raw_path),
                    "e2v2_input_manifest": raw["input_manifest"],
                    "legacy_pair_registration": descriptor(P.WORK / "tridep_ccml/experiments/TriDep-Stage4-EEGPT-Matched4s-37-fulltext-20261006/registration.json"),
                    "legacy_pair_features_manifest": descriptor(P.WORK / "tridep_ccml/experiments/TriDep-Stage4-EEGPT-Matched4s-37-fulltext-20261006/features_manifest.json")})
    for source in sources.values():
        checked(source)
    configuration = P.configuration()
    fits = []
    for condition in P.CONDITIONS:
        for seed in P.SEEDS:
            for fold in P.FOLDS:
                split = folds[f"fold_{fold}"]
                fits.append({"fit_id": fit_id(condition.condition_id, seed, fold),
                             "condition": condition.condition_id, "method": condition.method,
                             "stage": condition.stage, "objective": condition.objective,
                             "seed": seed, "fold": fold,
                             "train_ids": split["train"], "val_ids": split["val"], "test_ids": split["test"],
                             "policy": asdict(P.POLICIES[condition.policy_key]),
                             "dependency_fit_id": fit_id(condition.dependency, seed, fold) if condition.dependency else None,
                             "engineering_status": "pending", "training_status": "pending",
                             "blocker_status": "not_yet_assessed", "initially_complete": False,
                             "test_evaluation_status": "frozen"})
    counts = Counter(f["objective"] for f in fits)
    _require(len(fits) == len({f["fit_id"] for f in fits}) == 360 and counts == {"supervised": 345, "self_supervised": 15}, "Formal fit arithmetic differs")
    result = {"schema_version": 1, "experiment_id": P.EXPERIMENT_ID, "registered_at": now(),
              "status": "registered_training_pending", "configuration": configuration,
              "configuration_sha256": canonical_digest(configuration), "sources": sources,
              "cohort_id": cohort["cohort_id"], "subject_ids": cohort["subject_ids"],
              "labels": labels, "class_counts": cohort["class_counts"], "folds": folds,
              "folds_sha256": P.FOLDS_SHA256, "formal_fits": 360, "supervised_fits": 345,
              "ssl_fits": 15, "conditions_count": 24, "candidate_papers_count": 9,
              "formal_fits_completed_at_registration": 0, "test_evaluation_frozen": True,
              "test_unlock_rule": "all_360_training_complete_with_verified_checkpoint_curve_and_reload_evidence",
              "condition_source_locks": {c.condition_id: {"status": "pending", "path": f"engineering/{c.condition_id}.json"} for c in P.CONDITIONS},
              "raw_sources": [{"subject_id": row["subject_id"], "raw": row["raw"]} for row in raw_records],
              "fits": fits}
    _write_new(REGISTRATION, result)
    return result


def load_registration(verify_sources: bool = True) -> dict:
    result = read_json(REGISTRATION)
    _require(result["configuration"] == P.configuration()
             and result["configuration_sha256"] == canonical_digest(P.configuration()), "Static protocol differs from immutable registration")
    _require(result["formal_fits"] == 360 and result["supervised_fits"] == 345 and result["ssl_fits"] == 15
             and len(result["fits"]) == 360 and result["test_evaluation_frozen"] is True, "Registration counts/test freeze differ")
    expected = {fit_id(c.condition_id, s, f) for c in P.CONDITIONS for s in P.SEEDS for f in P.FOLDS}
    _require({fit["fit_id"] for fit in result["fits"]} == expected, "Registration fit identities differ")
    if verify_sources:
        for source in result["sources"].values():
            checked(source)
    return result


def registration_descriptor() -> dict:
    return descriptor(REGISTRATION)


def lock_condition(condition: str, contract: dict) -> dict:
    """Lock successful engineering evidence, exact input and parameter names."""
    load_registration()
    _require(condition in P.CONDITION_BY_ID, "Condition not registered")
    _require(isinstance(contract.get("input_contract"), dict) and bool(contract["input_contract"]), "Exact input contract required")
    _require(isinstance(contract.get("trainable_parameter_names"), list) and bool(contract["trainable_parameter_names"]), "Exact trainable parameter names required")
    _require(isinstance(contract.get("frozen_parameter_names"), list), "Exact frozen parameter names required")
    _require(len(set(contract["trainable_parameter_names"])) == len(contract["trainable_parameter_names"])
             and not set(contract["trainable_parameter_names"]) & set(contract["frozen_parameter_names"]), "Parameter sets duplicate or overlap")
    for name in ("forward_checked", "gradient_checked", "resume_checked", "training_only_engineering_checked"):
        _require(contract.get(name) is True, "Engineering check pending: " + name)
    _require(isinstance(contract.get("implementation_sources"), list) and bool(contract["implementation_sources"]), "Implementation source hashes required")
    _require(isinstance(contract.get("weight_sources"), list), "Weight source list required; scratch/SSL may have none")
    if P.CONDITION_BY_ID[condition].policy_key not in ("scratch", "ssl"):
        _require(bool(contract["weight_sources"]), "Pretrained or fold-local SSL weight provenance required")
    for source in contract["implementation_sources"] + contract["weight_sources"]:
        checked(source)
    value = {"condition": condition, "registration": registration_descriptor(), "locked_at": now(),
             "configuration_sha256": canonical_digest(contract), "contract": contract}
    path = P.OUT / "engineering" / f"{condition}.json"
    _write_new(path, value)
    return descriptor(path)


def _condition_lock(condition: str) -> dict:
    path = P.OUT / "engineering" / f"{condition}.json"
    value = read_json(path)
    _require(value["condition"] == condition and value["registration"] == registration_descriptor()
             and value["configuration_sha256"] == canonical_digest(value["contract"]), "Engineering source lock identity/hash differs")
    for source in value["contract"]["implementation_sources"] + value["contract"]["weight_sources"]:
        checked(source)
    return descriptor(path)


def _validate_complete(evidence: dict, folder: Path) -> None:
    _require(isinstance(evidence, dict), "Completion evidence required")
    for name in COMPLETION_CHECKS:
        _require(evidence.get(name) is True, "Completion check missing: " + name)
    for name in COMPLETION_ARTIFACTS:
        path = checked(evidence.get(name))
        _require(path.is_relative_to(folder), "Completion artifact belongs to another fit: " + name)
    _require(isinstance(evidence.get("epochs"), int) and evidence["epochs"] > 0
             and isinstance(evidence.get("optimizer_updates"), int) and evidence["optimizer_updates"] > 0,
             "Actual epochs and optimizer updates required")
    _require(bool(evidence.get("stop_reason")), "Registered training stop reason required")


def record_fit_status(condition: str, seed: int, fold: int, status: str, *, evidence: dict | None = None, reason: str | None = None) -> dict:
    """Append a status event; completion requires hash-bound real evidence."""
    registration = load_registration()
    identity = fit_id(condition, seed, fold)
    _require(status in STATUS_VALUES, "Unknown fit status")
    _require(status not in ("blocked", "failed") or bool(reason), "Blocked/failed status requires actual reason")
    folder = fit_directory(condition, seed, fold)
    current_path = folder / "status.json"
    previous = read_json(current_path) if current_path.exists() else None
    _require(previous is None or previous["registration"] == registration_descriptor(), "Previous fit status belongs to another registration")
    _require(previous is None or previous["status"] != "training_complete", "Completed fit is sealed; preserve it")
    lock = _condition_lock(condition) if status in ("ready", "training", "training_complete") else None
    if status == "training_complete":
        _validate_complete(evidence, folder)
        entry = next(item for item in registration["fits"] if item["fit_id"] == identity)
        if entry["policy"]["early_stopping"] is False:
            _require(evidence["epochs"] == 300 and evidence["optimizer_updates"] == 600,
                     "Fixed diagnostic requires 300 epochs and 600 real updates")
    dependency = P.CONDITION_BY_ID[condition].dependency
    if status in ("training", "training_complete") and dependency:
        dependent = read_json(fit_directory(dependency, seed, fold) / "status.json")
        _require(dependent["status"] == "training_complete", "Fold-local SSL must finish before supervised fitting")
        _validate_complete(dependent["evidence"], fit_directory(dependency, seed, fold))
    event = {"fit_id": identity, "condition": condition, "seed": seed, "fold": fold,
             "registration": registration_descriptor(), "condition_source_lock": lock,
             "status": status, "recorded_at": now(), "reason": reason, "evidence": evidence,
             "previous_status": previous["status"] if previous else "engineering_pending",
             "test_evaluation_status": "frozen"}
    event_path = folder / "status_events" / f"{uuid.uuid4().hex}.json"
    _write_new(event_path, event)
    event["event_record"] = descriptor(event_path)
    _replace_status(current_path, event)
    return event


def fit_status(condition: str, seed: int, fold: int) -> dict:
    path = fit_directory(condition, seed, fold) / "status.json"
    if not path.exists():
        return {"fit_id": fit_id(condition, seed, fold), "status": "engineering_pending", "test_evaluation_status": "frozen"}
    value = read_json(path)
    _require(value["fit_id"] == fit_id(condition, seed, fold) and value["registration"] == registration_descriptor(), "Fit status identity differs")
    event = read_json(checked(value["event_record"]))
    _require(event == {k: v for k, v in value.items() if k != "event_record"}, "Current fit status differs from immutable event")
    return value


def status_summary() -> dict:
    registration = load_registration()
    counts = Counter(fit_status(f["condition"], f["seed"], f["fold"])["status"] for f in registration["fits"])
    return {"experiment_id": P.EXPERIMENT_ID, "formal_fits": 360, "supervised_fits": 345, "ssl_fits": 15,
            "statuses": dict(counts), "test_evaluation_frozen": counts.get("training_complete", 0) != 360}


def assert_test_unlocked() -> dict:
    """Call BEFORE any outer-test model forward, predictions or scoring.

    This intentionally has no override argument. All 360 completions and all
    their byte hashes are checked, including SSL and control conditions.
    """
    registration = load_registration()
    incomplete = []
    statuses = []
    for fit in registration["fits"]:
        value = fit_status(fit["condition"], fit["seed"], fit["fold"])
        if value["status"] != "training_complete":
            incomplete.append((fit["fit_id"], value["status"]))
        else:
            folder = fit_directory(fit["condition"], fit["seed"], fit["fold"])
            _validate_complete(value["evidence"], folder)
            _require(value["condition_source_lock"] == _condition_lock(fit["condition"]), "Completed engineering source lock changed")
            statuses.append(descriptor(folder / "status.json"))
    _require(not incomplete, f"External testing remains frozen: {len(incomplete)}/360 fits incomplete; first={incomplete[:3]}")
    return {"passed": True, "registration": registration_descriptor(), "formal_fits_verified": len(statuses),
            "statuses": statuses, "checked_at": now()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "status", "test-gate"))
    args = parser.parse_args()
    if args.action == "register":
        value = create_registration()
        output = {"registration": registration_descriptor(), "formal_fits": value["formal_fits"],
                  "supervised_fits": value["supervised_fits"], "ssl_fits": value["ssl_fits"],
                  "completed": 0, "test_evaluation_frozen": True}
    elif args.action == "status":
        output = status_summary()
    else:
        output = assert_test_unlocked()
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

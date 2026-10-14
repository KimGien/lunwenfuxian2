"""Capture stable epoch1..120 JSON prefixes and verify the registered extension."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys
import time

REPORT = Path(__file__).resolve().parent
WORK = REPORT.parents[2]
OUT = WORK / "tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008"
FIT = OUT / "runs/CSBrain-lastblock-finetune/seed_11/fold_1"
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import protocol as P


def read(path):
    assert Path(path).suffix == ".json"
    for attempt in range(4):
        try:
            return json.loads(Path(path).read_bytes().decode("utf-8-sig"))
        except (PermissionError, FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
            if attempt == 3: raise
            time.sleep((.02, .04, .08)[attempt])


def desc(path):
    path = Path(path)
    assert path.name not in ("train_history.json", "validation_history.json", "status.json"), "Never bind a mutable full-history/status SHA"
    assert path.suffix in (".json", ".py")
    return {"path": path.relative_to(WORK).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    protocol = desc(Path(P.__file__))
    assert protocol["sha256"] == "7449f21e82e5a81c71f46fe7ac98681f1308c670dab4598750fbd0192e88d114"
    registration = read(OUT / "registration.json")
    registry_desc = desc(OUT / "registration.json")
    assert registry_desc["sha256"] == "27bf2972c08f77457959cf3cba5bc6056c320f223c2426af5da3ae88d55ce090"
    assert registration["configuration"] == P.configuration() and registration["formal_fits"] == 360
    q3 = read(OUT / "continuation_queue_v3/registration.json")
    q3_desc = desc(OUT / "continuation_queue_v3/registration.json")
    assert q3_desc["sha256"] == "e135585ca475b733bbc6bfbaa1616fe71f9b8fb1af8f2c0b413d2a9c8565b3a0"
    sources = {item["path"]: item for item in q3["sources"]}
    training_sources = []
    for name in ("finetune.py", "finetune_v2.py"):
        value = desc(WORK / "tridep_ccml/eeg_all_candidates" / name)
        assert value == sources[value["path"]]
        training_sources.append(value)
    started = read(FIT / "started.json")
    assert (started["condition"], started["seed"], started["fold"]) == ("CSBrain-lastblock-finetune", 11, 1)
    split = registration["folds"]["fold_1"]
    assert started["train_ids"] == split["train"] and started["val_ids"] == split["val"] and started["test_forward_count"] == 0
    assert (len(split["train"]), len(split["val"])) == (27, 3)
    assert set(started["raw_sources"]) == set(split["train"] + split["val"])
    lock_desc = desc(WORK / started["condition_source_lock"]["path"])
    assert lock_desc == started["condition_source_lock"]
    lock = read(WORK / lock_desc["path"])
    assert lock["registration"] == registry_desc
    train_path, val_path = FIT / "train_history.json", FIT / "validation_history.json"
    history, validation = read(train_path), read(val_path)
    assert len(history) >= 121 and len(validation) >= 120 and len(history) <= 200 and len(validation) <= 200
    first120, val120, epoch121 = history[:120], validation[:120], history[120]
    assert read(train_path)[:121] == first120 + [epoch121] and read(val_path)[:120] == val120
    losses = []
    for epoch, (training, val) in enumerate(zip(first120, val120), 1):
        assert training["epoch"] == val["epoch"] == epoch and training["val_eval_ce"] == val["val_ce"]
        assert math.isfinite(val["val_ce"]) and val["val_ce"] >= 0
        assert training["optimizer_updates"] == epoch * 4 and training["backward_calls"] == epoch * 27
        losses.append(val["val_ce"])
        decision = P.training_decision("finetune", losses, extended=False)
        assert decision["stop"] is False and decision["extend"] is (epoch == 120)
    policy = P.POLICIES["finetune"]
    assert (policy.initial_epochs, policy.final_epochs, policy.minimum_stop_epoch, policy.patience) == (120, 200, 40, 30)
    previous, recent = mean(losses[80:100]), mean(losses[100:120])
    improvement = (previous - recent) / previous
    assert decision == {"epoch": 120, "best_epoch": 117, "stop": False, "extend": True,
                        "reason": "registered_extension", "last_20_relative_validation_improvement": improvement}
    assert decision["best_epoch"] > .8 * 120 and improvement >= .01
    assert math.isclose(improvement, .030884556606166236, rel_tol=0, abs_tol=1e-15)
    assert (epoch121["epoch"], epoch121["optimizer_updates"], epoch121["backward_calls"]) == (121, 484, 3267)
    assert epoch121["head_lr"] == policy.head_lr * P.learning_rate_multiplier("finetune", 121)
    assert epoch121["backbone_lr"] == policy.backbone_lr * P.learning_rate_multiplier("finetune", 121, parameter_group="backbone")
    assert P.learning_rate_multiplier("finetune", 121) == P.learning_rate_multiplier("finetune", 200) == .1
    fit_status, queue_status = read(FIT / "status.json"), read(OUT / "continuation_queue_v3/status.json")
    assert fit_status["test_evaluation_status"] == "frozen" and queue_status["test_evaluation_frozen"] is True
    timestamp = str(time.time_ns())
    snapshot_path = REPORT / ("csbrain_fold1_epoch120_stable_prefix_" + timestamp + ".json")
    snapshot = {"kind": "immutable_prefix_of_active_training_history", "captured_at": datetime.now(timezone.utc).isoformat(),
                "fit_id": "CSBrain-lastblock-finetune/seed_11/fold_1", "registration": registry_desc, "protocol": protocol,
                "condition_source_lock": lock_desc, "started": desc(FIT / "started.json"), "training_sources": training_sources,
                "source_history_paths_only": [train_path.relative_to(WORK).as_posix(), val_path.relative_to(WORK).as_posix()],
                "full_active_history_sha_bound": False, "prefix_equal_across_two_reads": True,
                "train_history_first120": first120, "validation_history_first120": val120,
                "first_post_extension_completed_epoch": epoch121}
    save(snapshot_path, snapshot)
    result = {"kind": "independent_registered_extension_event_review", "passed": True,
              "created_at": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/resume_review", "review_source": desc(Path(__file__)),
              "stable_prefix_snapshot": desc(snapshot_path), "registration": registry_desc, "q3_registration": q3_desc, "protocol": protocol,
              "condition_source_lock": lock_desc, "training_sources": training_sources,
              "decision_recomputed_from_only_first120_validation_losses": decision,
              "eligibility": {"best_epoch": 117, "must_be_strictly_after_epoch": 96, "in_final20_percent": True,
                              "mean_epoch81_to100": previous, "mean_epoch101_to120": recent,
                              "relative_improvement": improvement, "minimum_relative_improvement": .01},
              "unchanged_budget": {"initial_cap": 120, "extended_cap": 200, "additional_possible_epochs_max": 80,
                                   "same_registered_fit": True, "formal_fit_matrix_total": 360,
                                   "further_extension_allowed_by_policy": False, "extension_lr_multiplier_held": .1},
              "role_checks": {"train_subjects": 27, "validation_subjects": 3, "raw_source_descriptor_keys_only_train_and_val": True,
                              "started_test_forward_count": 0, "fit_test_evaluation_status_observed": fit_status["test_evaluation_status"],
                              "global_test_evaluation_frozen_observed": queue_status["test_evaluation_frozen"],
                              "outer_test_scores_used_for_extension": False},
              "observation_only": {"history_rows_at_first_read": len(history), "validation_rows_at_first_read": len(validation),
                                   "post_extension_complete_epoch121_seen": True, "maximum_observed_completed_epoch_within200": True,
                                   "fit_status_at_review": fit_status["status"], "full_mutable_history_or_status_sha_is_permanent_binding": False},
              "source_evidence": [{"source": protocol, "line": 197, "end_line": 220, "description": "Registered cap/validation-only extension; no repeated extension when extended=True"},
                                  {"source": training_sources[0], "line": 531, "end_line": 538, "description": "Actual formal loop sets extended on the registered decision then persists full epoch state"}],
              "review_operations": {"active_pt_reads": 0, "models_constructed": 0, "gpu_calls": 0, "raw_or_feature_arrays_read": 0,
                                    "test_probabilities_read": 0, "sealed_source_changes": 0},
              "scope": "Extension event and fixed-budget eligibility only; no claim of final fit completion or final checkpoint state",
              "fit_completion_claimed": False, "full_plan_completion_claimed": False}
    report_path = REPORT / ("independent_csbrain_registered_extension_review_" + timestamp + ".json")
    save(report_path, result)
    print(json.dumps({"artifact": desc(report_path), "prefix_snapshot": desc(snapshot_path), "passed": True, "decision": decision}, ensure_ascii=True))


if __name__ == "__main__": main()

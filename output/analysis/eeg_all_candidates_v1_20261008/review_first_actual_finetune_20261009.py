"""Independent JSON/byte audit of the first actual FT CPU checkpoint proof.

No torch import, tensor deserialization, raw samples, predictions or model calls.
Only source-bound training reports and three sealed checkpoint byte hashes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

WORK = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import protocol as P

VERIFIED = {}


def desc(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": path.relative_to(WORK).as_posix(), "sha256": digest.hexdigest()}


def check(value):
    path = WORK / value["path"]
    assert path.suffix not in (".npy", ".npz"), "Raw/feature arrays are outside this review"
    actual = desc(path)
    assert actual["sha256"] == value["sha256"], value["path"]
    VERIFIED[value["path"]] = actual
    return path


def read(value):
    path = check(value)
    assert path.suffix == ".json"
    return json.loads(path.read_text(encoding="utf-8"))


def source(location, start, end, claim):
    return {"source": desc(WORK / location), "line": start, "end_line": end, "claim": claim}


def main():
    actual_desc = {"path": "output/analysis/eeg_all_candidates_v1_20261008/completed_finetune_checkpoint_audit_v2_1791510698316559500.json",
                   "sha256": "ecf09d43809ec47e133c68cf6bb85dd13eb395d249ef1d8cb737acd14c3ebdae"}
    actual = read(actual_desc)
    prior_desc = {"path": "output/analysis/eeg_all_candidates_v1_20261008/independent_completed_finetune_auditor_v2_review_1791510105296177400.json",
                  "sha256": "b463b7edfdaaf348e1969742edf7569821c9139e90d12999da32c6fa40ff0a3b"}
    prior = read(prior_desc)
    assert actual["auditor_source"] == prior["reviewed_sources"]["auditor"]
    check(actual["auditor_source"])
    for value in actual["delegated_sources"].values():
        check(value)
    registry = read(actual["registration"])
    assert actual["status"] == "partial_verified" and actual["registered_finetune_fits"] == 135
    assert actual["verified_completed_finetune_fits"] == actual["logical_completed_fits_counted"] == 1
    assert actual["training_complete_finetune_observed"] == 1 and len(actual["pending_fits"]) == 134
    assert not actual["errors"] and not actual["deferred_completed_fits"]
    assert actual["checkpoint_loads"] == 3 and actual["model_constructions"] == actual["gpu_operations"] == 0
    assert actual["observed_auditor_operations_including_failed_checks"] == {
        "cpu_checkpoint_loads_started": 3, "cpu_checkpoint_loads_returned": 3,
        "model_construction_attempts_blocked": 0, "gpu_initialization_attempts_blocked": 0}
    assert actual["full_360_plan_completion_claimed"] is False and actual["all_135_finetune_checkpoints_verified"] is False
    for name in ("raw_arrays_loaded", "outer_prediction_artifacts_read", "test_gate_called", "evaluation_driver_called", "wide_all_plan_audit_called"):
        assert actual[name] is False
    row, = actual["completed_rows"]
    assert row["fit_id"] == "EEGPT-lastblock-finetune/seed_11/fold_1" and row["passed"] is True
    metadata, completion, status, lock = (read(row[name]) for name in ("metadata", "completion", "status_artifact", "condition_source_lock"))
    contract = lock["contract"]
    assert completion == status["evidence"] and status["status"] == "training_complete"
    assert status["registration"] == completion["registry"] == lock["registration"] == actual["registration"]
    assert status["condition_source_lock"] == metadata["condition_source_lock"] == row["condition_source_lock"]
    assert completion["metadata"] == row["metadata"] and metadata["test_forward_count"] == 0
    check(status["event_record"])
    history = read(completion["train_history"])
    validation = read(completion["validation_history"])
    split = registry["folds"]["fold_1"]
    for role in ("train", "val", "test"):
        assert metadata[role + "_ids"] == split[role]
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (27, 3, 7)
    assert set(metadata["raw_sources"]) == set(split["train"] + split["val"])
    # Raw-source descriptors are compared, never dereferenced.
    registered_raw = {item["subject_id"]: item["raw"] for item in registry["raw_sources"]}
    assert all(metadata["raw_sources"][sid] == registered_raw[sid] for sid in metadata["raw_sources"])
    assert metadata["fresh_official_pretrained"] is metadata["fresh_head"] is True
    assert metadata["frozen_best_head_inherited"] is False
    assert metadata["micro_subjects"] == 1 and metadata["window_chunk"] == 8
    assert metadata["effective_subject_batch"] == metadata["window_bag_size"] == 8
    assert metadata["eval_window_count"] == 60 and metadata["reload_probe_subject_ids"] == split["train"][:1]
    assert (row["epochs"], row["updates"], row["best_epoch"]) == (67, 268, 37)
    assert metadata["epochs"] == completion["epochs"] == 67 and len(history) == len(validation) == 67
    assert metadata["optimizer_updates"] == metadata["head_optimizer_updates"] == completion["optimizer_updates"] == 268
    assert metadata["backbone_optimizer_updates"] == 4 * (67 - 10) == 228
    assert metadata["backward_calls"] == 67 * 27 == 1809
    losses = []
    for epoch, (training, val) in enumerate(zip(history, validation), 1):
        assert training["epoch"] == val["epoch"] == epoch
        assert training["optimizer_updates"] == epoch * 4 and training["backward_calls"] == epoch * 27
        assert training["window_indices"] == [(((epoch - 1) % 8) * 8 + k) % 60 for k in range(8)]
        assert training["cycle_epoch"] == (epoch - 1) % 8 + 1 and training["unfrozen"] is (epoch > 10)
        assert training["repeated_padding_windows_this_epoch"] == ([0, 1, 2, 3] if epoch % 8 == 0 else [])
        assert training["training_module_subtrees"] == (["blocks.7"] if epoch > 10 else [])
        assert training["head_lr"] == 3e-4 * P.learning_rate_multiplier("finetune", epoch)
        assert training["backbone_lr"] == 1e-5 * P.learning_rate_multiplier("finetune", epoch, parameter_group="backbone")
        assert all(math.isfinite(training[name]) for name in ("train_bag_ce", "train_eval_ce", "val_eval_ce"))
        assert val["val_ce"] == training["val_eval_ce"]
        losses.append(val["val_ce"])
        decision = P.training_decision("finetune", losses, extended=False)
        assert decision["stop"] is (epoch == 67) and decision["extend"] is False
    assert decision["best_epoch"] == 37 and decision["reason"] == metadata["stop_reason"] == "validation_patience"
    assert metadata["extended"] is False
    active = contract["trainable_parameter_names"]
    heads = [name for name in active if name.startswith("head.")]
    encoder = [name for name in active if name.startswith("encoder.")]
    assert len(heads) == 6 and len(encoder) == 12
    assert set(metadata["parameter_changed_after_training"]) == set(active)
    assert all(value is True for value in metadata["parameter_changed_after_training"].values())
    assert metadata["selected_checkpoint_phase"] == "lastblock_finetune"
    for phase in ("selected", "terminal"):
        assert metadata[phase + "_encoder_changed_parameter_names"] == encoder
        assert metadata[phase + "_head_changed_parameter_names"] == heads
        assert metadata[phase + "_encoder_changed_parameter_count"] == 12
        assert metadata[phase + "_head_changed_parameter_count"] == 6
    assert len({metadata[name] for name in ("selected_encoder_initial_state_sha256", "selected_encoder_state_sha256", "terminal_encoder_state_sha256")}) == 3
    assert metadata["frozen_initial_sha256"] == metadata["frozen_final_sha256"]
    implementation = metadata["versioned_finetune_implementation"]
    assert implementation == contract["versioned_finetune_implementation"]
    for name in ("wrapper_source", "delegated_runner_source", "luna_adapter_source"):
        check(implementation[name])
    # Author descriptors remain bound to engineering; raw arrays/author weight tensors are not loaded.
    assert metadata["adapter_audit"]["checkpoint"] == contract["input_contract"]["adapter"]["checkpoint"]
    assert metadata["adapter_audit"]["selected_encoder_parameter_names"] == [name.removeprefix("encoder.") for name in encoder]
    for value in contract["implementation_sources"]:
        check(value)
    proof = row["cpu_checkpoint_audit"]
    assert proof["cpu_tensors_verified"] is True and proof["checkpoint_loads"] == 3
    assert proof["only_map_location_cpu"] is True and proof["model_constructions"] == proof["gpu_operations"] == 0
    checkpoints = {Path(value["path"]).name: value for value in proof["checkpoints"]}
    assert set(checkpoints) == {"initial.pt", "latest.pt", "best.pt"} and checkpoints["best.pt"] == completion["checkpoint"]
    for value in checkpoints.values():
        check(value)  # streaming bytes only, no torch.load
    boundaries = {value["path"]: value for value in proof["durable_counter_boundaries"]}
    assert (boundaries["initial.pt"]["epoch"], boundaries["initial.pt"]["optimizer_updates"], boundaries["initial.pt"]["backward_calls"]) == (0, 0, 0)
    assert (boundaries["latest.pt"]["epoch"], boundaries["latest.pt"]["optimizer_updates"], boundaries["latest.pt"]["backward_calls"]) == (67, 268, 1809)
    assert boundaries["best.pt"]["epoch"] == 37 and boundaries["best.pt"]["optimizer_updates"] is boundaries["best.pt"]["backward_calls"] is None
    for name in ("initial.pt", "latest.pt"):
        groups = boundaries[name]["optimizer_groups"]
        assert groups["group_names"] == ["head", "backbone"] and (groups["head_parameters"], groups["backbone_parameters"]) == (6, 12)
        assert groups["serialized_id_parameter_names"] == heads + encoder
        assert groups["moment_shapes_verified"] is (name == "latest.pt")
    boundary = row["f2_physical_durable_boundary"]
    attempt, = boundary["attempts"]
    trace = read(attempt["artifact"])
    assert trace["status"] == attempt["status"] == "complete" and trace["kind"] == "formal"
    assert (trace["condition"], trace["seed"], trace["fold"]) == (row["condition"], 11, 1)
    assert trace["implementation"] == implementation
    expected = {"forward": 67 * (27 + 27 + 3) + 2,
                "encoder_forward": 67 * (27 + (27 + 3) * math.ceil(60 / 8)) + 2 * math.ceil(60 / 8),
                "encoder_windows": 67 * (27 * 8 + (27 + 3) * 60) + 2 * 60,
                "encode": 17905, "backward": 1809, "optimizer_step": 268}
    for name, value in expected.items():
        for suffix in ("_started", "_returned"):
            key = name + suffix
            assert trace[key] == attempt["observed_counters"][key] == boundary["recorded_physical_call_subtotals"][key] == value
    assert expected["forward"] == 3821 and expected["encoder_windows"] == 135192
    assert not boundary["duplicate_content_aliases"] and not row["legacy_failure_records_not_added"]
    assert boundary["durable_optimizer_updates_counted_once"] == 268 and boundary["durable_backward_calls_counted_once"] == 1809
    assert boundary["optimizer_returns_above_durable_known_lower_bound"] == boundary["backward_returns_above_durable_known_lower_bound"] == 0
    assert boundary["legacy_failure_counters_added_to_f2"] is boundary["physical_and_durable_counts_added_together"] is False
    assert boundary["nested_metadata_seconds_added_to_attempt_seconds"] is False
    assert boundary["physical_optimizer_calls_exact"] is boundary["all_historical_wall_seconds_exact"] is None
    assert 0 < metadata["seconds_this_attempt"] < trace["seconds"] == attempt["seconds_this_attempt"]
    assert metadata["peak_allocated_bytes"] == 354929152
    assert actual["durable_optimizer_updates_verified_subtotal"] == 268 and actual["durable_backward_calls_verified_subtotal"] == 1809
    for value in list(VERIFIED.values()):
        check(value)
    record = {
        "kind": "independent_first_actual_finetune_evidence_review", "created_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": "/root/resume_review", "review_source": desc(Path(__file__)), "passed": True,
        "actual_cpu_audit": actual_desc, "preceding_source_only_review": prior_desc,
        "scope": "Independent JSON, source and checkpoint-byte binding review of one actual completed CPU checkpoint audit; tensor checks are those executed by the bound CPU auditor, not repeated here",
        "first_actual_cpu_check_not_previous_pending_test": True,
        "fit_id": row["fit_id"], "training_complete_recorded_at": status["recorded_at"],
        "actual_cpu_audit_created_at": actual["created_at"], "epochs": 67, "best_epoch": 37,
        "stop_reason": "validation_patience", "verified_durable_optimizer_updates": 268,
        "verified_durable_backward_calls": 1809, "train_subjects": 27, "validation_subjects": 3,
        "terminal_adamw_groups": {"head_parameters": 6, "head_steps_each": 268,
                                  "backbone_parameters": 12, "backbone_steps_each": 228,
                                  "empty_initial_state_and_terminal_moment_shapes_verified_by_bound_cpu_auditor": True,
                                  "terminal_head_lr": history[-1]["head_lr"], "terminal_backbone_lr": history[-1]["backbone_lr"],
                                  "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8},
        "encoder_update_proof": {"selected_changed_parameter_count": 12, "terminal_changed_parameter_count": 12,
                                 "selected_phase": "lastblock_finetune", "terminal_all_18_active_tensors_changed": True,
                                 "selected_and_terminal_encoder_states_are_distinct": True, "frozen_tensors_unchanged": True,
                                 "selected_encoder_state_sha256": metadata["selected_encoder_state_sha256"],
                                 "terminal_encoder_state_sha256": metadata["terminal_encoder_state_sha256"],
                                 "proof_basis": "Actual bound wide.deep_checkpoint CPU assertions plus this review's exact source, three checkpoint hashes and metadata/contract names"},
        "independently_recomputed_observed_call_counts": expected,
        "forward_count_components": {"training": 67 * 27, "all_window_train_validation_diagnostics": 67 * 30, "train_only_reload_checks": 2},
        "encoder_window_count_components": {"training": 67 * 27 * 8, "all_window_train_validation_diagnostics": 67 * 30 * 60, "train_only_reload_checks": 120},
        "resource_scope": {"formal_f2_attempt_seconds": trace["seconds"], "nested_f1_metadata_seconds": metadata["seconds_this_attempt"],
                           "formal_training_peak_allocated_bytes": metadata["peak_allocated_bytes"],
                           "overlapping_seconds_not_summed": True, "old_head_probe_resource_unknowns_not_backfilled": True},
        "cpu_auditor_actual_operations": actual["observed_auditor_operations_including_failed_checks"],
        "this_review_operations": {"checkpoint_byte_hashes": 3, "tensor_checkpoint_deserializations": 0,
                                   "model_constructions": 0, "gpu_calls": 0, "raw_arrays_read": 0,
                                   "test_probability_artifacts_read": 0, "training_sources_modified": 0},
        "source_locations": [
            source("output/analysis/eeg_all_candidates_v1_20261008/audit_completed_finetune_checkpoints_v2.py", 161, 247, "Actual CPU load wrapper binds checkpoint identities, durable counters and AdamW group order/moment shapes"),
            source("output/analysis/eeg_all_candidates_v1_20261008/audit_full_plan_acceptance_v3.py", 234, 252, "The actual CPU delegate asserts every optimizer group's hyperparameters, LR and per-parameter steps"),
            source("output/analysis/eeg_all_candidates_v1_20261008/audit_full_plan_acceptance_v3.py", 269, 281, "The actual CPU delegate asserts frozen tensor equality, all intended terminal changes, selected encoder changed names and selected/terminal hashes"),
            source("tridep_ccml/eeg_all_candidates/finetune.py", 499, 561, "Registered train and diagnostic calls per epoch plus two train-only reload forwards explain observed trace counts"),
            source("tridep_ccml/eeg_all_candidates/finetune_v2.py", 56, 86, "Durable started/returned observer and actual wrapper elapsed cost"),
        ],
        "verified_descriptors": list(VERIFIED.values()),
        "full_135_finetune_completion_claimed": False, "full_360_plan_completion_claimed": False,
        "snapshot_status": "one_completed_finetune_verified", "remaining_ft_fits_in_audit_snapshot": 134,
        "test_gate_opened_by_this_review": False, "final_semantic_companion": False,
        "historical_physical_exact_totals_remain_unknown": True,
    }
    destination = REPORT / ("independent_first_actual_finetune_review_" + str(time.time_ns()) + ".json")
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"artifact": desc(destination), "passed": True, "verified_descriptors": len(VERIFIED),
                      "actual_cpu_audit_loads": 3, "review_tensor_loads": 0}, ensure_ascii=True))


if __name__ == "__main__":
    main()

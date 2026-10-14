"""Read-only independent review of the first real head-warmup selected FT.

Stdlib and registered policy only. No tensor deserialization or model operation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import time

REPORT = Path(__file__).resolve().parent
WORK = REPORT.parents[2]
spec = importlib.util.spec_from_file_location("first_ft_review_helpers", REPORT / "review_first_actual_finetune_20261009.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
P = B.P


def main():
    first_review = B.read({"path": "output/analysis/eeg_all_candidates_v1_20261008/independent_first_actual_finetune_review_1791510902423001600.json",
                           "sha256": "9570c96017cb7574b8b7f08cfe806a39715a0aa0f1c2e9583d4eeed149107b14"})
    B.check(first_review["review_source"])
    actual_desc = {"path": "output/analysis/eeg_all_candidates_v1_20261008/completed_finetune_checkpoint_audit_v2_1791512527325217600.json",
                   "sha256": "2ae8c94e4dc51f51177350b29fa432c7bef06211108ed7001ea007c45864721a"}
    actual = B.read(actual_desc)
    assert actual["auditor_source"]["sha256"] == "12bbb62d025bb2bbacfefa68677a9707185ae5943927b245b1e817a0124802bb"
    B.check(actual["auditor_source"])
    for value in actual["delegated_sources"].values(): B.check(value)
    registry = B.read(actual["registration"])
    assert actual["status"] == "partial_verified" and actual["verified_completed_finetune_fits"] == 3
    assert actual["checkpoint_loads"] == 9 and not actual["errors"] and not actual["deferred_completed_fits"]
    assert actual["observed_auditor_operations_including_failed_checks"] == {
        "cpu_checkpoint_loads_started": 9, "cpu_checkpoint_loads_returned": 9,
        "model_construction_attempts_blocked": 0, "gpu_initialization_attempts_blocked": 0}
    assert len(actual["pending_fits"]) == 132 and actual["all_135_finetune_checkpoints_verified"] is False
    assert actual["full_360_plan_completion_claimed"] is False
    for name in ("raw_arrays_loaded", "outer_prediction_artifacts_read", "test_gate_called", "evaluation_driver_called"):
        assert actual[name] is False
    row, = [item for item in actual["completed_rows"] if item["fit_id"] == "EEGPT-lastblock-finetune/seed_11/fold_3"]
    assert row["passed"] is True and (row["epochs"], row["updates"], row["best_epoch"]) == (40, 160, 4)
    metadata, completion, status, lock = (B.read(row[name]) for name in ("metadata", "completion", "status_artifact", "condition_source_lock"))
    assert status["evidence"] == completion and status["status"] == "training_complete"
    assert metadata["condition_source_lock"] == status["condition_source_lock"] == row["condition_source_lock"]
    assert completion["metadata"] == row["metadata"] and completion["registry"] == lock["registration"] == actual["registration"]
    B.check(status["event_record"])
    history, validation = (B.read(completion[name]) for name in ("train_history", "validation_history"))
    split = registry["folds"]["fold_3"]
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (26, 3, 8)
    for role in ("train", "val", "test"): assert metadata[role + "_ids"] == split[role]
    assert len(set(split["train"])) == 26 and set(split["train"]).isdisjoint(split["val"] + split["test"])
    raw = {item["subject_id"]: item["raw"] for item in registry["raw_sources"]}
    assert set(metadata["raw_sources"]) == set(split["train"] + split["val"])
    assert all(value == raw[sid] for sid, value in metadata["raw_sources"].items())
    assert metadata["test_forward_count"] == 0 and metadata["reload_probe_subject_ids"] == split["train"][:1]
    assert metadata["micro_subjects"] == 1 and metadata["effective_subject_batch"] == metadata["window_bag_size"] == 8
    assert metadata["window_chunk"] == 8 and metadata["eval_window_count"] == 60
    assert (metadata["epochs"], metadata["optimizer_updates"], metadata["head_optimizer_updates"], metadata["backbone_optimizer_updates"], metadata["backward_calls"]) == (40, 160, 160, 120, 1040)
    assert metadata["best_epoch"] == 4 and metadata["selected_checkpoint_phase"] == "head_only_warmup"
    assert metadata["extended"] is False and len(history) == len(validation) == 40
    policy = P.POLICIES["finetune"]
    assert (policy.warmup_epochs, policy.minimum_stop_epoch, policy.patience, policy.minimum_improvement) == (10, 40, 30, 1e-4)
    losses = []
    for epoch, (training, val) in enumerate(zip(history, validation), 1):
        assert training["epoch"] == val["epoch"] == epoch
        assert training["optimizer_updates"] == epoch * math.ceil(26 / 8) == epoch * 4
        assert training["backward_calls"] == epoch * 26 and training["unfrozen"] is (epoch > 10)
        assert training["window_indices"] == [(((epoch - 1) % 8) * 8 + index) % 60 for index in range(8)]
        assert training["cycle_epoch"] == (epoch - 1) % 8 + 1
        assert training["repeated_padding_windows_this_epoch"] == ([0, 1, 2, 3] if epoch % 8 == 0 else [])
        assert training["training_module_subtrees"] == (["blocks.7"] if epoch > 10 else [])
        assert training["head_lr"] == policy.head_lr * P.learning_rate_multiplier("finetune", epoch)
        assert training["backbone_lr"] == policy.backbone_lr * P.learning_rate_multiplier("finetune", epoch, parameter_group="backbone")
        assert val["val_ce"] == training["val_eval_ce"] and math.isfinite(val["val_ce"])
        losses.append(val["val_ce"])
        decision = P.training_decision("finetune", losses, extended=False)
        assert decision["stop"] is (epoch == 40) and decision["extend"] is False
        if epoch >= 4: assert decision["best_epoch"] == 4
    assert decision["reason"] == completion["stop_reason"] == metadata["stop_reason"] == "validation_patience"
    assert history[3]["optimizer_updates"] == 16 and history[3]["backward_calls"] == 104 and history[3]["backbone_lr"] == 0
    assert history[9]["optimizer_updates"] == 40 and history[9]["backward_calls"] == 260
    assert history[-1]["optimizer_updates"] - history[9]["optimizer_updates"] == 120
    assert history[-1]["backward_calls"] - history[9]["backward_calls"] == 780
    contract = lock["contract"]
    active = contract["trainable_parameter_names"]
    heads = [name for name in active if name.startswith("head.")]
    encoder = [name for name in active if name.startswith("encoder.")]
    assert (len(heads), len(encoder)) == (6, 12)
    assert set(metadata["parameter_changed_after_training"]) == set(active) and all(metadata["parameter_changed_after_training"].values())
    assert metadata["selected_encoder_parameters_changed"] is False
    assert metadata["selected_encoder_changed_parameter_names"] == [] and metadata["selected_encoder_changed_parameter_count"] == metadata["selected_encoder_parameter_change_count"] == 0
    assert metadata["selected_encoder_state_sha256"] == metadata["selected_encoder_initial_state_sha256"]
    assert metadata["terminal_encoder_state_sha256"] != metadata["selected_encoder_state_sha256"]
    assert metadata["terminal_encoder_changed_parameter_count"] == 12 and metadata["terminal_encoder_changed_parameter_names"] == encoder
    for phase in ("selected", "terminal"):
        assert metadata[phase + "_head_changed_parameter_count"] == 6 and metadata[phase + "_head_changed_parameter_names"] == heads
    assert metadata["frozen_initial_sha256"] == metadata["frozen_final_sha256"]
    assert metadata["fresh_official_pretrained"] is metadata["fresh_head"] is True and metadata["frozen_best_head_inherited"] is False
    implementation = metadata["versioned_finetune_implementation"]
    assert implementation == contract["versioned_finetune_implementation"]
    for name in ("wrapper_source", "delegated_runner_source", "luna_adapter_source"): B.check(implementation[name])
    for value in contract["implementation_sources"]: B.check(value)
    proof = row["cpu_checkpoint_audit"]
    assert proof["cpu_tensors_verified"] is True and proof["checkpoint_loads"] == 3 and proof["only_map_location_cpu"] is True
    checkpoints = {Path(value["path"]).name: value for value in proof["checkpoints"]}
    assert set(checkpoints) == {"initial.pt", "latest.pt", "best.pt"} and checkpoints["best.pt"] == completion["checkpoint"]
    for value in checkpoints.values(): B.check(value)
    bounds = {item["path"]: item for item in proof["durable_counter_boundaries"]}
    assert (bounds["initial.pt"]["epoch"], bounds["initial.pt"]["optimizer_updates"], bounds["initial.pt"]["backward_calls"]) == (0, 0, 0)
    assert (bounds["latest.pt"]["epoch"], bounds["latest.pt"]["optimizer_updates"], bounds["latest.pt"]["backward_calls"]) == (40, 160, 1040)
    assert bounds["best.pt"]["epoch"] == 4 and bounds["best.pt"]["optimizer_updates"] is bounds["best.pt"]["backward_calls"] is None
    for name in ("initial.pt", "latest.pt"):
        groups = bounds[name]["optimizer_groups"]
        assert groups["group_names"] == ["head", "backbone"] and groups["serialized_id_parameter_names"] == heads + encoder
        assert (groups["head_parameters"], groups["backbone_parameters"]) == (6, 12)
        assert groups["moment_shapes_verified"] is (name == "latest.pt")
    boundary = row["f2_physical_durable_boundary"]
    attempt, = boundary["attempts"]
    trace = B.read(attempt["artifact"])
    assert trace["status"] == "complete" and trace["kind"] == "formal"
    assert (trace["condition"], trace["seed"], trace["fold"]) == (row["condition"], 11, 3) and trace["implementation"] == implementation
    expected = {"forward": 40 * (26 + 29) + 2, "encoder_forward": 40 * (26 + 29 * 8) + 16,
                "encode": 10336, "encoder_windows": 40 * (26 * 8 + 29 * 60) + 120,
                "backward": 1040, "optimizer_step": 160}
    for name, value in expected.items():
        for suffix in ("_started", "_returned"):
            key = name + suffix
            assert trace[key] == attempt["observed_counters"][key] == boundary["recorded_physical_call_subtotals"][key] == value
    assert expected["forward"] == 2202 and expected["encoder_windows"] == 78040
    assert not boundary["duplicate_content_aliases"] and not row["legacy_failure_records_not_added"]
    assert boundary["durable_optimizer_updates_counted_once"] == 160 and boundary["durable_backward_calls_counted_once"] == 1040
    assert boundary["optimizer_returns_above_durable_known_lower_bound"] == boundary["backward_returns_above_durable_known_lower_bound"] == 0
    assert 0 < metadata["seconds_this_attempt"] < trace["seconds"] and metadata["peak_allocated_bytes"] == 354929152
    prior_reports = []
    for path in sorted(REPORT.glob("completed_finetune_checkpoint_audit_v2_*.json")):
        if path.name >= Path(actual_desc["path"]).name: continue
        descriptor = B.desc(path)
        value = B.read(descriptor)
        prior_reports.append({"artifact": descriptor, "verified_fits": value["verified_completed_finetune_fits"]})
    assert {0, 1, 2}.issubset({item["verified_fits"] for item in prior_reports})
    for value in list(B.VERIFIED.values()): B.check(value)
    result = {
        "kind": "independent_first_head_warmup_selected_finetune_review", "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/resume_review",
        "review_source": B.desc(Path(__file__)), "helper_source": first_review["review_source"], "actual_cpu_audit": actual_desc,
        "fit_id": row["fit_id"], "scope": "Fold3 training JSON/source/three checkpoint-byte binding review of genuine completed CPU tensor proof; no new tensor deserialization",
        "actual_cpu_audit_scope": {"verified_ft_fits": 3, "cpu_loads": 9, "errors": 0,
                                   "this_fold_cpu_loads": 3, "remaining_ft_fits_in_snapshot": 132},
        "selection": {"selected_epoch": 4, "phase": "head_only_warmup", "selected_encoder_changed_parameters": 0,
                      "selected_encoder_initial_and_selected_state_sha256": metadata["selected_encoder_state_sha256"],
                      "selected_head_changed_parameters_reported": 6, "selected_checkpoint_replaced_or_constrained_to_after_warmup": False,
                      "proof_basis": "Actual bound CPU comparison finds no changed encoder tensor names, and verifies selected encoder SHA; frozen tensors also equal initial"},
        "terminal": {"epoch": 40, "head_steps_each": 160, "backbone_steps_each": 120,
                     "encoder_changed_parameters": 12, "all18_intended_parameters_changed": True,
                     "encoder_state_sha256": metadata["terminal_encoder_state_sha256"], "frozen_tensors_unchanged": True},
        "phase_counts": {"head_only_epochs": 10, "head_only_optimizer_steps": 40, "head_only_backwards": 260,
                         "head_and_lastblock_epochs": 30, "head_and_lastblock_optimizer_steps": 120,
                         "head_and_lastblock_backwards": 780, "selected_epoch4_history_updates": 16,
                         "selected_epoch4_history_backwards": 104, "selected_checkpoint_optimizer_payload_counts_remain_null": True},
        "sampling": {"train_subjects": 26, "validation_subjects": 3, "test_subjects_declared_only": 8,
                     "logical_batch_sizes_from_sealed_sampler": [8, 8, 8, 2], "micro_subjects": 1,
                     "optimizer_steps_per_epoch": 4, "backwards_per_epoch": 26,
                     "cyclic_8_of_60_training_windows_verified_all_40_epochs": True,
                     "raw_source_descriptors_compared_without_reading_arrays": True,
                     "subject_order_basis": "Registered train IDs, actual history/traces and sealed randperm complete-traversal implementation; subject_order tensor payload not reloaded in this review"},
        "stopping": {"best_epoch_recomputed_from_all40_validation_losses": 4, "effective_improvement": 1e-4,
                     "minimum_stop_epoch": 40, "patience": 30, "first_policy_stop": 40,
                     "epochs_since_best_at_stop": 36, "reason": "validation_patience", "extension": False,
                     "no_test_information_used": True},
        "recomputed_actual_trace_counts": expected,
        "cost": {"formal_f2_attempt_seconds": trace["seconds"], "nested_f1_seconds": metadata["seconds_this_attempt"],
                 "formal_peak_allocated_bytes": metadata["peak_allocated_bytes"], "nested_seconds_added": False,
                 "old_probe_resources_backfilled": False, "exact_unobserved_historical_calls_remain_unknown": True},
        "preserved_prior_cpu_reports": prior_reports, "verified_descriptors": list(B.VERIFIED.values()),
        "source_evidence": [
            B.source("tridep_ccml/eeg_all_candidates/protocol.py", 186, 220, "Effective validation improvement and earliest registered stop are independent of training phase"),
            B.source("tridep_ccml/eeg_all_candidates/finetune.py", 503, 539, "Full train randperm, groups of8, micro1, per-epoch histories and validation-only selection"),
            B.source("output/analysis/eeg_all_candidates_v1_20261008/audit_full_plan_acceptance_v3.py", 269, 281, "Actual CPU selected-versus-initial differences and terminal all-active changes checked separately"),
            B.source("tridep_ccml/eeg_all_candidates/evaluate_v2.py", 54, 85, "Head-warmup selected disclosure requires zero encoder changes; terminal evidence does not replace selected evidence"),
        ],
        "this_review_operations": {"checkpoint_bytes_hashed": 3, "tensor_deserializations": 0, "models_constructed": 0,
                                   "gpu_calls": 0, "raw_arrays_read": 0, "outer_probabilities_read": 0, "sealed_sources_modified": 0},
        "full_135_ft_completion_claimed": False, "full_360_plan_completion_claimed": False, "final_semantic_companion": False,
    }
    destination = REPORT / ("independent_head_warmup_selected_finetune_review_" + str(time.time_ns()) + ".json")
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"artifact": B.desc(destination), "passed": True, "verified_descriptors": len(B.VERIFIED),
                      "selected_encoder_changes": 0, "terminal_encoder_changes": 12, "tensor_loads_by_this_review": 0}, ensure_ascii=True))


if __name__ == "__main__":
    main()

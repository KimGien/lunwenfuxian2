"""Independent JSON/source review of CSBrain's first extended FT completion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

REPORT = Path(__file__).resolve().parent
WORK = REPORT.parents[2]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import protocol as P
verified = {}


def desc(path):
    path = Path(path).resolve()
    assert path.suffix in (".json", ".py"), "No tensor/raw/feature reads"
    return {"path": path.relative_to(WORK).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(binding):
    actual = desc(WORK / binding["path"])
    assert actual["sha256"] == binding["sha256"], binding["path"]
    verified[actual["path"]] = actual
    return WORK / actual["path"]


def read(binding):
    return json.loads(check(binding).read_text(encoding="utf-8"))


def main():
    actual_binding = {"path": "output/analysis/eeg_all_candidates_v1_20261008/completed_finetune_checkpoint_audit_v3_1791525043354058900.json", "sha256": "a57599602c5e7e8b5835f3e90665541a2b1b152715d6dd7dfebd0518057f35ef"}
    actual = read(actual_binding)
    assert actual["status"] == "partial_verified" and actual["verified_completed_finetune_fits"] == 16 and actual["checkpoint_loads"] == 48
    assert not actual["errors"] and not actual["deferred_completed_fits"] and len(actual["pending_fits"]) == 119
    assert actual["auditor_source"]["sha256"] == "307d6feabb9b4964db335fd9a0390e3026015cd57f6205117de125c79508c6d7"
    check(actual["auditor_source"])
    for binding in actual["delegated_sources"].values(): check(binding)
    registration = read(actual["registration"])
    row, = [row for row in actual["completed_rows"] if row["condition"] == "CSBrain-lastblock-finetune"]
    assert row["fit_id"] == "CSBrain-lastblock-finetune/seed_11/fold_1" and row["passed"] is True
    metadata, completion, status, lock = (read(row[name]) for name in ("metadata", "completion", "status_artifact", "condition_source_lock"))
    assert completion == status["evidence"] and status["status"] == "training_complete"
    assert completion["metadata"] == row["metadata"] and completion["registry"] == lock["registration"] == actual["registration"]
    assert metadata["condition_source_lock"] == row["condition_source_lock"] == status["condition_source_lock"]
    check(status["event_record"])
    contract = lock["contract"]
    assert (metadata["condition"], metadata["seed"], metadata["fold"]) == (row["condition"], 11, 1)
    for role in ("train", "val", "test"): assert metadata[role + "_ids"] == registration["folds"]["fold_1"][role]
    assert len(metadata["train_ids"]) == 27 and len(metadata["val_ids"]) == 3
    assert metadata["test_forward_count"] == 0 and set(metadata["raw_sources"]) == set(metadata["train_ids"] + metadata["val_ids"])
    assert metadata["versioned_finetune_implementation"] == contract["versioned_finetune_implementation"]
    for value in metadata["adapter_audit"]["implementation_sources"]: check(value)
    assert metadata["adapter_audit"]["source"] == contract["input_contract"]["adapter"]["source"]
    assert (metadata["epochs"], metadata["best_epoch"], metadata["optimizer_updates"], metadata["head_optimizer_updates"], metadata["backbone_optimizer_updates"], metadata["backward_calls"]) == (147, 117, 588, 588, 548, 3969)
    assert metadata["extended"] is True and metadata["stop_reason"] == "validation_patience"
    assert metadata["micro_subjects"] == 1 and metadata["window_chunk"] == 8 and metadata["reload_probe_subject_ids"] == metadata["train_ids"][:1]
    history, validation = (read(completion[name]) for name in ("train_history", "validation_history"))
    old_extension_binding = {"path": "output/analysis/eeg_all_candidates_v1_20261008/independent_csbrain_registered_extension_review_1791524264695918300.json", "sha256": "24b0657fc2231e0643c2c7a8bfb234f77c99ca3f28ebf6ed9b40f556dd81e4fd"}
    old_extension = read(old_extension_binding)
    prefix = read(old_extension["stable_prefix_snapshot"])
    assert history[:120] == prefix["train_history_first120"] and validation[:120] == prefix["validation_history_first120"]
    assert history[120] == prefix["first_post_extension_completed_epoch"]
    assert len(history) == len(validation) == 147 and old_extension["protocol"] == actual["delegated_sources"]["protocol"]
    losses, extended, extension_epochs = [], False, []
    for epoch, (train, val) in enumerate(zip(history, validation), 1):
        assert train["epoch"] == val["epoch"] == epoch and train["val_eval_ce"] == val["val_ce"]
        assert train["optimizer_updates"] == 4 * epoch and train["backward_calls"] == 27 * epoch
        assert train["unfrozen"] is (epoch > 10) and train["training_module_subtrees"] == (["backbone.encoder.layers.11"] if epoch > 10 else [])
        assert train["window_indices"] == [(((epoch - 1) % 8) * 8 + index) % 60 for index in range(8)]
        assert train["head_lr"] == 3e-4 * P.learning_rate_multiplier("finetune", epoch)
        assert train["backbone_lr"] == 1e-5 * P.learning_rate_multiplier("finetune", epoch, parameter_group="backbone")
        losses.append(val["val_ce"])
        decision = P.training_decision("finetune", losses, extended=extended)
        assert decision["stop"] is (epoch == 147)
        if decision["extend"]:
            extension_epochs.append(epoch)
            assert epoch == 120 and decision == old_extension["decision_recomputed_from_only_first120_validation_losses"]
            extended = True
    assert extended is True and extension_epochs == [120] and decision["best_epoch"] == 117 and decision["reason"] == "validation_patience"
    assert 147 - 117 == P.POLICIES["finetune"].patience == 30 and P.POLICIES["finetune"].final_epochs == 200
    encoder = [name for name in contract["trainable_parameter_names"] if name.startswith("encoder.")]
    heads = [name for name in contract["trainable_parameter_names"] if name.startswith("head.")]
    assert len(encoder) == 20 and len(heads) == 6 and len(contract["frozen_parameter_names"]) == 273
    assert all(name.startswith("encoder.backbone.encoder.layers.11.") for name in encoder)
    assert set(metadata["parameter_changed_after_training"]) == set(encoder + heads) and all(value is True for value in metadata["parameter_changed_after_training"].values())
    assert metadata["selected_checkpoint_phase"] == "lastblock_finetune"
    for phase in ("selected", "terminal"):
        assert metadata[phase + "_encoder_changed_parameter_count"] == 20 and metadata[phase + "_encoder_changed_parameter_names"] == encoder
        assert metadata[phase + "_head_changed_parameter_count"] == 6 and metadata[phase + "_head_changed_parameter_names"] == heads
    assert len({metadata[name] for name in ("selected_encoder_initial_state_sha256", "selected_encoder_state_sha256", "terminal_encoder_state_sha256")}) == 3
    assert metadata["frozen_initial_sha256"] == metadata["frozen_final_sha256"]
    proof = row["cpu_checkpoint_audit"]
    assert proof["cpu_tensors_verified"] is True and proof["checkpoint_loads"] == 3 and proof["only_map_location_cpu"] is True
    assert proof["gpu_operations"] == proof["model_constructions"] == 0
    bounds = {value["path"]: value for value in proof["durable_counter_boundaries"]}
    assert (bounds["latest.pt"]["epoch"], bounds["latest.pt"]["optimizer_updates"], bounds["latest.pt"]["backward_calls"]) == (147, 588, 3969)
    assert bounds["best.pt"]["epoch"] == 117 and bounds["best.pt"]["optimizer_updates"] is bounds["best.pt"]["backward_calls"] is None
    assert (bounds["initial.pt"]["epoch"], bounds["initial.pt"]["optimizer_updates"], bounds["initial.pt"]["backward_calls"]) == (0, 0, 0)
    for name in ("initial.pt", "latest.pt"):
        groups = bounds[name]["optimizer_groups"]
        assert groups["group_names"] == ["head", "backbone"] and groups["serialized_id_parameter_names"] == heads + encoder
        assert groups["moment_shapes_verified"] is (name == "latest.pt")
    boundary = row["f2_physical_durable_boundary"]
    attempt, = boundary["attempts"]
    trace = read(attempt["artifact"])
    assert trace["status"] == "complete" and trace["kind"] == "formal" and trace["seed"] == 11 and trace["fold"] == 1
    assert trace["implementation"] == metadata["versioned_finetune_implementation"] and trace["condition"] == row["condition"]
    expected = {"forward": 147 * (27 + 30) + 2, "encoder_forward": 147 * (27 + 30 * 8) + 16,
                "encode": 39265, "encoder_windows": 147 * (27 * 8 + 30 * 60) + 120, "backward": 3969, "optimizer_step": 588}
    for name, count in expected.items():
        for suffix in ("_started", "_returned"): assert trace[name + suffix] == boundary["recorded_physical_call_subtotals"][name + suffix] == count
    assert not boundary["duplicate_content_aliases"] and not row["legacy_failure_records_not_added"]
    assert boundary["physical_optimizer_calls_exact"] is boundary["all_historical_wall_seconds_exact"] is None
    assert boundary["optimizer_returns_above_durable_known_lower_bound"] == boundary["backward_returns_above_durable_known_lower_bound"] == 0
    assert boundary["physical_and_durable_counts_added_together"] is boundary["legacy_failure_counters_added_to_f2"] is boundary["nested_metadata_seconds_added_to_attempt_seconds"] is False
    assert trace["seconds"] > metadata["seconds_this_attempt"] > 0 and metadata["peak_allocated_bytes"] == 223574528
    for binding in list(verified.values()): check(binding)
    report = {"kind": "independent_first_csbrain_extended_completion_review", "passed": True,
              "created_at": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/resume_review", "review_source": desc(Path(__file__)),
              "actual_cpu_audit": actual_binding, "fit_id": row["fit_id"], "scope": {"csbrain_verified_fits": 1, "csbrain_registered_fits": 15,
              "all_ft_verified_in_actual_cpu_snapshot": 16, "all_ft_registered": 135, "actual_cpu_loads_total": 48, "this_fit_cpu_loads": 3},
              "preserved_extension_review": old_extension_binding, "preserved_prefix": old_extension["stable_prefix_snapshot"],
              "final_history_matches_preserved_first120_and_epoch121": True,
              "policy": {"extended": True, "extension_epoch": 120, "best_epoch": 117, "completed_epochs": 147, "stop_reason": "validation_patience",
                         "patience": 30, "final_budget": 200, "actual_lr_after120_head": history[-1]["head_lr"], "actual_lr_after120_backbone": history[-1]["backbone_lr"],
                         "scheduler_extended_true_in_latest_asserted_by_existing_cpu_auditor": True, "no_second_extension_or_test_guided_selection": True},
              "parameter_and_optimizer_proof": {"head_parameters": 6, "head_steps_each": 588, "layer11_parameters": 20, "backbone_steps_each": 548,
                                                "frozen_parameter_names": 273, "selected_encoder_changes": 20, "terminal_encoder_changes": 20,
                                                "selected_and_terminal_heads_changed": 6, "frozen_state_unchanged": True,
                                                "proof_basis": "Existing bound actual CPU tensor audit plus metadata/contract/source agreement; no tensor reload in this review"},
              "actual_trace_counts_recomputed": expected,
              "cost_scope": {"physical_optimizer_step_returns": 588, "backwards": 3969, "group_step_counts_not_summed_as_physical_calls": True,
                             "formal_attempt_count": 1, "reported_failed_attempts": 0, "duplicate_attempts_counted": 0,
                             "f2_seconds": trace["seconds"], "nested_metadata_seconds": metadata["seconds_this_attempt"], "peak_allocated_bytes": metadata["peak_allocated_bytes"],
                             "nested_seconds_or_physical_and_durable_counts_added": False, "engineering_resources_or_old_probe_unknowns_backfilled": False,
                             "per_attempt_resume_baseline_and_historical_exact_cost_remain_unknown": True},
              "recovery_scope": {"actual_cpu_verified_zero_epoch_initial_and_complete_epoch_latest": True,
                                 "latest_rng_sampler_scheduler_history_checked_by_existing_cpu_auditor": True,
                                 "completed_train_only_reload_probe_subject_ids": metadata["reload_probe_subject_ids"], "recovery_or_reload_reexecuted_here": False},
              "checkpoint_bindings_carried_from_actual_cpu_proof_not_reread": proof["checkpoints"],
              "source_evidence": [{"source": actual["delegated_sources"]["wide_acceptance"], "line": 218, "end_line": 281, "description": "Existing CPU assertions bind initial/latest/best, scheduler extended flag, optimizer steps and actual frozen/changed tensors"},
                                  {"source": actual["delegated_sources"]["protocol"], "line": 197, "end_line": 239, "description": "Single registered extension, later patience stop and held learning rate"}],
              "verified_json_and_source_descriptors": list(verified.values()),
              "review_operations": {"pt_files_read": 0, "tensor_deserializations": 0, "model_constructions": 0, "gpu_calls": 0, "raw_or_feature_arrays_read": 0,
                                    "outer_test_probabilities_read": 0, "sealed_sources_modified": 0, "other15_fits_reaudited_here": False},
              "all_csbrain15_completion_claimed": False, "full_plan_completion_claimed": False, "final_semantic_companion": False}
    destination = REPORT / ("independent_csbrain_first_extended_completion_review_" + str(time.time_ns()) + ".json")
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"artifact": desc(destination), "passed": True, "scope_csbrain": "1/15", "cpu_snapshot_scope": "16/135", "pt_reads": 0}, ensure_ascii=True))


if __name__ == "__main__": main()

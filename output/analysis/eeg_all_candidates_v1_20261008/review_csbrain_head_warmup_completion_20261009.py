"""CSBrain fold3 head-warmup selection: existing JSON/source/CPU proof only."""
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
    assert path.suffix in (".json", ".py"), "Only JSON/source reads permitted"
    return {"path": path.relative_to(WORK).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(binding):
    value = desc(WORK / binding["path"])
    assert value["sha256"] == binding["sha256"], binding["path"]
    verified[value["path"]] = value
    return WORK / value["path"]


def read(binding):
    return json.loads(check(binding).read_text(encoding="utf-8"))


def main():
    audit_binding = {"path": "output/analysis/eeg_all_candidates_v1_20261008/completed_finetune_checkpoint_audit_v3_1791528577327694600.json", "sha256": "b79b4144c71ad30f7eec5c750fceaf37811cd6dcfe16b2a1e5cb869b0a818dab"}
    audit = read(audit_binding)
    assert audit["status"] == "partial_verified" and audit["verified_completed_finetune_fits"] == 18 and audit["checkpoint_loads"] == 54
    assert not audit["errors"] and not audit["deferred_completed_fits"] and len(audit["pending_fits"]) == 117
    assert audit["auditor_source"]["sha256"] == "307d6feabb9b4964db335fd9a0390e3026015cd57f6205117de125c79508c6d7"
    check(audit["auditor_source"])
    for binding in audit["delegated_sources"].values(): check(binding)
    registration = read(audit["registration"])
    cs_rows = [item for item in audit["completed_rows"] if item["condition"] == "CSBrain-lastblock-finetune"]
    assert len(cs_rows) == 3
    row, = [item for item in cs_rows if item["fit_id"] == "CSBrain-lastblock-finetune/seed_11/fold_3"]
    assert row["passed"] is True and (row["epochs"], row["updates"], row["best_epoch"]) == (40, 160, 8)
    metadata, completion, status, lock = (read(row[name]) for name in ("metadata", "completion", "status_artifact", "condition_source_lock"))
    assert completion == status["evidence"] and status["status"] == "training_complete"
    assert completion["registry"] == lock["registration"] == audit["registration"] and completion["metadata"] == row["metadata"]
    assert metadata["condition_source_lock"] == row["condition_source_lock"] == status["condition_source_lock"]
    assert (metadata["condition"], metadata["seed"], metadata["fold"]) == (row["condition"], 11, 3)
    contract = lock["contract"]
    assert metadata["versioned_finetune_implementation"] == contract["versioned_finetune_implementation"]
    for binding in metadata["adapter_audit"]["implementation_sources"]: check(binding)
    assert metadata["adapter_audit"]["source"] == contract["input_contract"]["adapter"]["source"]
    split = registration["folds"]["fold_3"]
    for role in ("train", "val", "test"): assert metadata[role + "_ids"] == split[role]
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (26, 3, 8)
    assert set(metadata["raw_sources"]) == set(split["train"] + split["val"]) and metadata["test_forward_count"] == 0
    assert metadata["micro_subjects"] == 1 and metadata["effective_subject_batch"] == metadata["window_bag_size"] == metadata["window_chunk"] == 8
    assert metadata["eval_window_count"] == 60 and metadata["reload_probe_subject_ids"] == split["train"][:1]
    assert (metadata["head_optimizer_updates"], metadata["backbone_optimizer_updates"], metadata["backward_calls"]) == (160, 120, 1040)
    assert metadata["best_epoch"] == 8 and metadata["selected_checkpoint_phase"] == "head_only_warmup" and metadata["extended"] is False
    history, validation = (read(completion[name]) for name in ("train_history", "validation_history"))
    assert len(history) == len(validation) == 40
    losses = []
    for epoch, (train, val) in enumerate(zip(history, validation), 1):
        assert train["epoch"] == val["epoch"] == epoch and train["val_eval_ce"] == val["val_ce"]
        assert train["optimizer_updates"] == epoch * 4 and train["backward_calls"] == epoch * 26
        assert train["unfrozen"] is (epoch > 10) and train["training_module_subtrees"] == (["backbone.encoder.layers.11"] if epoch > 10 else [])
        assert train["window_indices"] == [(((epoch - 1) % 8) * 8 + index) % 60 for index in range(8)]
        assert train["head_lr"] == 3e-4 * P.learning_rate_multiplier("finetune", epoch)
        assert train["backbone_lr"] == 1e-5 * P.learning_rate_multiplier("finetune", epoch, parameter_group="backbone")
        losses.append(val["val_ce"])
        decision = P.training_decision("finetune", losses, extended=False)
        assert decision["stop"] is (epoch == 40) and decision["extend"] is False
    assert decision["best_epoch"] == 8 and decision["reason"] == metadata["stop_reason"] == "validation_patience"
    assert P.POLICIES["finetune"].minimum_stop_epoch == 40 and 40 - 8 == 32 >= P.POLICIES["finetune"].patience == 30
    assert (history[7]["optimizer_updates"], history[7]["backward_calls"], history[7]["backbone_lr"]) == (32, 208, 0)
    assert (history[9]["optimizer_updates"], history[9]["backward_calls"]) == (40, 260)
    heads = [name for name in contract["trainable_parameter_names"] if name.startswith("head.")]
    encoder = [name for name in contract["trainable_parameter_names"] if name.startswith("encoder.")]
    assert len(heads) == 6 and len(encoder) == 20 and len(contract["frozen_parameter_names"]) == 273
    assert all(name.startswith("encoder.backbone.encoder.layers.11.") for name in encoder)
    assert set(metadata["parameter_changed_after_training"]) == set(heads + encoder) and all(value is True for value in metadata["parameter_changed_after_training"].values())
    assert metadata["selected_encoder_changed_parameter_count"] == 0 and metadata["selected_encoder_changed_parameter_names"] == []
    assert metadata["selected_encoder_parameters_changed"] is False and metadata["selected_encoder_state_sha256"] == metadata["selected_encoder_initial_state_sha256"]
    assert metadata["terminal_encoder_changed_parameter_count"] == 20 and metadata["terminal_encoder_changed_parameter_names"] == encoder
    assert metadata["terminal_encoder_state_sha256"] != metadata["selected_encoder_state_sha256"]
    for phase in ("selected", "terminal"):
        assert metadata[phase + "_head_changed_parameter_count"] == 6 and metadata[phase + "_head_changed_parameter_names"] == heads
    assert metadata["frozen_initial_sha256"] == metadata["frozen_final_sha256"]
    proof = row["cpu_checkpoint_audit"]
    assert proof["cpu_tensors_verified"] is True and proof["checkpoint_loads"] == 3 and proof["only_map_location_cpu"] is True
    bounds = {item["path"]: item for item in proof["durable_counter_boundaries"]}
    assert (bounds["latest.pt"]["epoch"], bounds["latest.pt"]["optimizer_updates"], bounds["latest.pt"]["backward_calls"]) == (40, 160, 1040)
    assert bounds["best.pt"]["epoch"] == 8 and bounds["best.pt"]["optimizer_updates"] is bounds["best.pt"]["backward_calls"] is None
    groups = bounds["latest.pt"]["optimizer_groups"]
    assert groups["group_names"] == ["head", "backbone"] and groups["serialized_id_parameter_names"] == heads + encoder and groups["moment_shapes_verified"] is True
    boundary = row["f2_physical_durable_boundary"]
    attempt, = boundary["attempts"]
    trace = read(attempt["artifact"])
    assert (trace["condition"], trace["seed"], trace["fold"], trace["kind"], trace["status"]) == (row["condition"], 11, 3, "formal", "complete")
    assert trace["implementation"] == metadata["versioned_finetune_implementation"]
    expected = {"forward": 40 * (26 + 29) + 2, "encoder_forward": 40 * (26 + 29 * 8) + 16,
                "encode": 10336, "encoder_windows": 40 * (26 * 8 + 29 * 60) + 120, "backward": 1040, "optimizer_step": 160}
    for name, value in expected.items():
        for suffix in ("_started", "_returned"): assert trace[name + suffix] == boundary["recorded_physical_call_subtotals"][name + suffix] == value
    assert not row["legacy_failure_records_not_added"] and not boundary["duplicate_content_aliases"]
    assert trace["seconds"] > metadata["seconds_this_attempt"] > 0 and metadata["peak_allocated_bytes"] == 223574528
    for binding in list(verified.values()): check(binding)
    result = {"kind": "independent_csbrain_fold3_head_warmup_selection_review", "passed": True,
              "created_at": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/resume_review", "review_source": desc(Path(__file__)),
              "actual_cpu_audit": audit_binding, "fit_id": row["fit_id"],
              "scope": {"csbrain_complete_in_actual_cpu_snapshot": "3/15", "ft_complete_in_actual_cpu_snapshot": "18/135", "actual_cpu_loads": 54,
                        "this_fold_cpu_loads": 3, "this_review_inspected_fit_only": row["fit_id"], "other_cases_reaudited": False},
              "selection": {"epoch": 8, "phase": "head_only_warmup", "encoder_changes": 0,
                            "encoder_initial_and_selected_sha256": metadata["selected_encoder_state_sha256"], "head_changes_reported": 6,
                            "selected_epoch_history_head_updates": 32, "selected_epoch_history_backwards": 208,
                            "selected_checkpoint_missing_optimizer_counts_not_filled_from_terminal": True, "best_forced_after_warmup": False},
              "terminal": {"epochs": 40, "encoder_changes": 20, "head_changes": 6, "frozen_parameter_names": 273, "frozen_state_unchanged": True,
                           "terminal_encoder_sha256": metadata["terminal_encoder_state_sha256"], "head_steps_each": 160, "backbone_steps_each": 120, "backwards": 1040},
              "sampling_and_policy": {"train_people": 26, "validation_people": 3, "logical_batch_sizes_from_sealed_sampler": [8, 8, 8, 2], "micro_subjects": 1,
                                      "head_only_first10_steps": 40, "head_only_first10_backwards": 260, "post_warmup_steps": 120, "post_warmup_backwards": 780,
                                      "best_epoch_recomputed": 8, "first_registered_stop": 40, "minimum_stop_epoch": 40, "patience": 30, "epochs_since_best": 32, "extended": False,
                                      "sampler_evidence": "Registered role IDs, actual per-epoch counters, sealed full randperm traversal; no tensor payload reread"},
              "actual_trace_counts_recomputed": expected,
              "cost_scope": {"f2_attempt_seconds": trace["seconds"], "nested_metadata_seconds": metadata["seconds_this_attempt"], "peak_allocated_bytes": metadata["peak_allocated_bytes"],
                             "overlapping_costs_or_optimizer_group_counts_added": False, "observed_failed_records": 0, "duplicate_attempts": 0, "historical_exact_unknowns_not_backfilled": True},
              "checkpoint_bindings_from_existing_actual_cpu_proof": proof["checkpoints"],
              "proof_basis": "Existing bound actual CPU assertions verify selected versus initial encoder equality, terminal all-active changes, frozen state and optimizer moments/steps; this review validates the JSON/source bindings without loading tensors",
              "source_evidence": [{"source": audit["delegated_sources"]["wide_acceptance"], "line": 234, "end_line": 281, "description": "Actual optimizer and selected/terminal CPU tensor assertions"},
                                  {"source": audit["delegated_sources"]["protocol"], "line": 186, "end_line": 220, "description": "Validation-only effective best and registered stop without post-warmup selection constraint"}],
              "verified_json_and_source_descriptors": list(verified.values()),
              "review_operations": {"pt_reads": 0, "tensor_deserializations": 0, "models_constructed": 0, "gpu_calls": 0, "raw_or_features_read": 0, "test_probabilities_read": 0, "sealed_sources_modified": 0},
              "all_csbrain15_completion_claimed": False, "full_plan_completion_claimed": False}
    destination = REPORT / ("independent_csbrain_head_warmup_completion_review_" + str(time.time_ns()) + ".json")
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"artifact": desc(destination), "passed": True, "scope": ["CSBrain3/15", "FT18/135"], "pt_reads": 0}, ensure_ascii=True))


if __name__ == "__main__": main()

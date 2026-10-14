"""JSON/source-only independent review of the first CSBrain extended-region best."""
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

VERIFIED = {}
ACTUAL = {"path": "output/analysis/eeg_all_candidates_v1_20261008/completed_finetune_checkpoint_audit_v3_1791535793189396100.json", "sha256": "826a1e5bba628fc16d298df3fccab0e558bb469670b0364ca13fe93b858c5756"}
FIT = "CSBrain-lastblock-finetune/seed_22/fold_1"


def desc(path):
    path = Path(path).resolve()
    assert path.suffix in (".json", ".py"), "No checkpoint/raw/feature reads"
    return {"path": path.relative_to(WORK).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(binding):
    observed = desc(WORK / binding["path"])
    assert observed == binding, binding
    VERIFIED[observed["path"]] = observed
    return WORK / observed["path"]


def read(binding):
    return json.loads(check(binding).read_text(encoding="utf-8"))


def main():
    actual = read(ACTUAL)
    assert actual["status"] == "partial_verified" and actual["verified_completed_finetune_fits"] == 21 and actual["checkpoint_loads"] == 63
    assert not actual["errors"] and not actual["deferred_completed_fits"] and len(actual["pending_fits"]) == 114
    assert actual["auditor_source"]["sha256"] == "307d6feabb9b4964db335fd9a0390e3026015cd57f6205117de125c79508c6d7"
    assert actual["registration"]["sha256"] == "27bf2972c08f77457959cf3cba5bc6056c320f223c2426af5da3ae88d55ce090"
    assert actual["delegated_sources"]["protocol"]["sha256"] == "7449f21e82e5a81c71f46fe7ac98681f1308c670dab4598750fbd0192e88d114"
    check(actual["auditor_source"])
    for binding in actual["delegated_sources"].values(): check(binding)
    registration = read(actual["registration"])
    assert sum(row["condition"] == "CSBrain-lastblock-finetune" for row in actual["completed_rows"]) == 6
    row, = [row for row in actual["completed_rows"] if row["fit_id"] == FIT]
    assert row["passed"] is True and row["training_json_verified"] is True
    meta, complete, status, lock = (read(row[key]) for key in ("metadata", "completion", "status_artifact", "condition_source_lock"))
    assert status["status"] == "training_complete" and status["evidence"] == complete
    assert complete["metadata"] == row["metadata"] and complete["registry"] == lock["registration"] == actual["registration"]
    assert meta["condition_source_lock"] == row["condition_source_lock"] == status["condition_source_lock"]
    assert all(complete[key] is True for key in ("passed", "checkpoint_reload_verified", "parameter_update_verified", "frozen_parameters_verified"))
    check(status["event_record"])
    assert (meta["condition"], meta["seed"], meta["fold"]) == (row["condition"], 22, 1)
    for role in ("train", "val", "test"): assert meta[role + "_ids"] == registration["folds"]["fold_1"][role]
    assert len(meta["train_ids"]) == 27 and len(meta["val_ids"]) == 3
    assert meta["test_forward_count"] == 0 and set(meta["raw_sources"]) == set(meta["train_ids"] + meta["val_ids"])
    contract = lock["contract"]
    assert meta["versioned_finetune_implementation"] == contract["versioned_finetune_implementation"]
    for binding in meta["adapter_audit"]["implementation_sources"]: check(binding)
    for key in ("wrapper_source", "delegated_runner_source", "luna_adapter_source"):
        check(meta["versioned_finetune_implementation"][key])
    assert meta["adapter_audit"]["source"] == contract["input_contract"]["adapter"]["source"]
    assert (meta["epochs"], meta["best_epoch"], meta["optimizer_updates"], meta["head_optimizer_updates"], meta["backbone_optimizer_updates"], meta["backward_calls"]) == (175, 145, 700, 700, 660, 4725)
    assert meta["extended"] is True and meta["stop_reason"] == "validation_patience"
    assert (meta["micro_subjects"], meta["window_chunk"], meta["effective_subject_batch"], meta["eval_window_count"]) == (1, 8, 8, 60)
    assert meta["reload_probe_subject_ids"] == meta["train_ids"][:1]
    history, validation = (read(complete[key]) for key in ("train_history", "validation_history"))
    assert len(history) == len(validation) == 175
    policy = P.POLICIES["finetune"]
    assert (policy.initial_epochs, policy.final_epochs, policy.patience, policy.minimum_stop_epoch) == (120, 200, 30, 40)
    assert policy.optimizer == "AdamW" and policy.checkpoint_selection == "validation_subject_mean_ce"
    losses, extended, extension_events, decision_samples = [], False, [], {}
    for epoch, (train, val) in enumerate(zip(history, validation), 1):
        assert train["epoch"] == val["epoch"] == epoch and train["val_eval_ce"] == val["val_ce"]
        assert train["optimizer_updates"] == 4 * epoch and train["backward_calls"] == 27 * epoch
        assert train["unfrozen"] is (epoch > 10) and train["training_module_subtrees"] == (["backbone.encoder.layers.11"] if epoch > 10 else [])
        assert train["window_indices"] == [(((epoch - 1) % 8) * 8 + index) % 60 for index in range(8)]
        assert train["head_lr"] == policy.head_lr * P.learning_rate_multiplier("finetune", epoch)
        assert train["backbone_lr"] == policy.backbone_lr * P.learning_rate_multiplier("finetune", epoch, parameter_group="backbone")
        losses.append(val["val_ce"])
        decision = P.training_decision("finetune", losses, extended=extended)
        assert decision["stop"] is (epoch == 175), (epoch, decision)
        if decision["extend"]:
            assert epoch == 120 and decision["best_epoch"] == 120 and decision["reason"] == "registered_extension"
            assert decision["last_20_relative_validation_improvement"] == 0.015098763731107833
            assert decision["best_epoch"] > .8 * policy.initial_epochs and decision["last_20_relative_validation_improvement"] >= .01
            extension_events.append(decision)
            extended = True
        if epoch in (120, 121, 145, 174, 175): decision_samples[str(epoch)] = decision
    assert extended is True and len(extension_events) == 1
    assert decision["best_epoch"] == 145 and decision["reason"] == "validation_patience"
    assert policy.initial_epochs < 145 < 175 <= policy.final_epochs and 175 - 145 == policy.patience
    assert decision_samples["145"]["best_epoch"] == decision_samples["174"]["best_epoch"] == 145
    assert all(r["head_lr"] == policy.head_lr * .1 and r["backbone_lr"] == policy.backbone_lr * .1 for r in history[119:])
    encoder = [n for n in contract["trainable_parameter_names"] if n.startswith("encoder.")]
    heads = [n for n in contract["trainable_parameter_names"] if n.startswith("head.")]
    assert (len(encoder), len(heads), len(contract["frozen_parameter_names"])) == (20, 6, 273)
    assert all(n.startswith("encoder.backbone.encoder.layers.11.") for n in encoder)
    assert set(meta["parameter_changed_after_training"]) == set(encoder + heads) and all(v is True for v in meta["parameter_changed_after_training"].values())
    assert meta["selected_checkpoint_phase"] == "lastblock_finetune"
    for phase in ("selected", "terminal"):
        assert meta[phase + "_encoder_changed_parameter_count"] == 20 and meta[phase + "_encoder_changed_parameter_names"] == encoder
        assert meta[phase + "_head_changed_parameter_count"] == 6 and meta[phase + "_head_changed_parameter_names"] == heads
    assert len({meta[k] for k in ("selected_encoder_initial_state_sha256", "selected_encoder_state_sha256", "terminal_encoder_state_sha256")}) == 3
    assert meta["frozen_initial_sha256"] == meta["frozen_final_sha256"]
    proof = row["cpu_checkpoint_audit"]
    assert proof["cpu_tensors_verified"] is True and proof["checkpoint_loads"] == 3 and proof["only_map_location_cpu"] is True
    assert proof["gpu_operations"] == proof["model_constructions"] == 0
    bounds = {r["path"]: r for r in proof["durable_counter_boundaries"]}
    assert (bounds["initial.pt"]["epoch"], bounds["initial.pt"]["optimizer_updates"], bounds["initial.pt"]["backward_calls"]) == (0, 0, 0)
    assert (bounds["latest.pt"]["epoch"], bounds["latest.pt"]["optimizer_updates"], bounds["latest.pt"]["backward_calls"]) == (175, 700, 4725)
    assert bounds["best.pt"]["epoch"] == 145 and bounds["best.pt"]["optimizer_updates"] is bounds["best.pt"]["backward_calls"] is None
    assert complete["checkpoint"] in proof["checkpoints"]
    for name in ("initial.pt", "latest.pt"):
        groups = bounds[name]["optimizer_groups"]
        assert groups["group_names"] == ["head", "backbone"] and groups["serialized_id_parameter_names"] == heads + encoder
        assert groups["moment_shapes_verified"] is (name == "latest.pt")
    boundary = row["f2_physical_durable_boundary"]
    attempt, = boundary["attempts"]
    trace = read(attempt["artifact"])
    assert (trace["status"], trace["kind"], trace["condition"], trace["seed"], trace["fold"]) == ("complete", "formal", row["condition"], 22, 1)
    assert trace["implementation"] == meta["versioned_finetune_implementation"]
    expected = {"forward": 175 * (27 + 30) + 2, "encoder_forward": 175 * (27 + 30 * 8) + 16,
                "encode": 46741, "encoder_windows": 175 * (27 * 8 + 30 * 60) + 120, "backward": 175 * 27, "optimizer_step": 175 * 4}
    for name, count in expected.items():
        for suffix in ("_started", "_returned"): assert trace[name + suffix] == boundary["recorded_physical_call_subtotals"][name + suffix] == count
    assert not boundary["duplicate_content_aliases"] and not row["legacy_failure_records_not_added"]
    for key in ("physical_optimizer_calls_exact", "physical_backward_calls_exact", "all_historical_wall_seconds_exact"): assert boundary[key] is None
    assert attempt["resume_durable_baseline"] is attempt["durable_updates_attributed_to_this_attempt"] is None
    assert boundary["optimizer_returns_above_durable_known_lower_bound"] == boundary["backward_returns_above_durable_known_lower_bound"] == 0
    for key in ("physical_and_durable_counts_added_together", "legacy_failure_counters_added_to_f2", "nested_metadata_seconds_added_to_attempt_seconds"): assert boundary[key] is False
    assert trace["seconds"] == attempt["seconds_this_attempt"] > meta["seconds_this_attempt"] > 0 and meta["peak_allocated_bytes"] == 223574528
    for binding in list(VERIFIED.values()): check(binding)
    report = {
        "kind": "independent_csbrain_first_extended_region_selected_checkpoint_review", "passed": True,
        "created_at": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/resume_review", "review_source": desc(Path(__file__)),
        "actual_cpu_audit": ACTUAL, "fit_id": FIT,
        "scope": {"reviewed_fits_here": 1, "csbrain_verified_in_cpu_snapshot": 6, "csbrain_registered": 15,
                  "ft_verified_in_cpu_snapshot": 21, "ft_registered": 135, "cpu_loads_in_existing_snapshot": 63, "target_fit_existing_cpu_loads": 3},
        "policy": {"registered": policy.__dict__, "extension_event": extension_events[0], "decisions_recomputed_every_epoch": 175,
                   "decision_samples": decision_samples, "selected_epoch": 145, "terminal_epoch": 175, "first_stop": 175,
                   "selected_inside_extended_region": True, "single_extension_only": True, "extension_is_same_fit": True,
                   "lr_from_epoch120_through175": {"head": history[-1]["head_lr"], "backbone": history[-1]["backbone_lr"]},
                   "held_terminal_lr_no_scheduler_restart": True, "selection_input_only_registered_validation_subject_mean_ce": True},
        "roles": {"train_ids": meta["train_ids"], "val_ids": meta["val_ids"], "train_only_reload_ids": meta["reload_probe_subject_ids"],
                  "test_forwards_reported": 0, "train_subjects": 27, "logical_batch_tail": [8, 8, 8, 3]},
        "actual_parameter_proof": {"selected_encoder_changes": 20, "terminal_encoder_changes": 20,
                                   "terminal_head_changes": 6, "selected_head_changes_reported_by_f2": 6,
                                   "selected_head_individual_tensor_changes_independently_reaudited_here": False,
                                   "frozen_parameter_names": 273, "actual_selected_and_terminal_frozen_tensors_equal_initial_in_cpu_proof": True,
                                   "encoder_hashes": {k: meta[k] for k in ("selected_encoder_initial_state_sha256", "selected_encoder_state_sha256", "terminal_encoder_state_sha256")},
                                   "frozen_initial_and_final_sha256": meta["frozen_initial_sha256"],
                                   "proof_basis": "Previously executed source-bound CPU audit; JSON/source checks here, no checkpoint reads"},
        "update_scope": {"durable_terminal_head_steps_each": 700, "durable_terminal_backbone_steps_each": 660, "physical_optimizer_returns": 700,
                         "warmup_head_steps": 40, "post_warmup_head_and_backbone_steps": 660, "backward_returns": 4725,
                         "selected_epoch_history_head_steps": history[144]["optimizer_updates"], "selected_epoch_history_backbone_steps": 4 * (145 - 10),
                         "selected_epoch_history_backward_calls": history[144]["backward_calls"],
                         "selected_history_counters_not_injected_into_best_checkpoint": True, "optimizer_groups_not_summed_as_physical_calls": True},
        "actual_f2_counts_recomputed": expected,
        "resource_and_recovery_scope": {"formal_attempt_count": 1, "failed_attempts_reported": 0, "duplicate_attempts_counted": 0,
                                        "f2_attempt_seconds": trace["seconds"], "nested_metadata_seconds": meta["seconds_this_attempt"],
                                        "peak_allocated_bytes": meta["peak_allocated_bytes"], "nested_seconds_added": False,
                                        "known_physical_returns_and_durable_steps_added": False, "engineering_costs_added": False,
                                        "unknown_historical_head_probe_resources_backfilled": False,
                                        "per_attempt_resume_durable_baseline": None, "all_historical_wall_seconds_exact": None,
                                        "actual_terminal_rng_sampler_history_extended_scheduler_checked_by_existing_cpu_audit": True,
                                        "recovery_or_reload_reexecuted_here": False},
        "checkpoint_descriptors_carried_from_cpu_proof_without_reads": proof["checkpoints"],
        "source_evidence": [{"source": actual["delegated_sources"]["protocol"], "line": 186, "end_line": 239, "description": "Validation-only effective best, single 120-to-200 extension, patience stop and held LR"},
                            {"source": meta["versioned_finetune_implementation"]["delegated_runner_source"], "line": 503, "end_line": 590, "description": "All registered train subjects, epoch-complete resume, best checkpoint capture, train-only reload and completion"},
                            {"source": actual["delegated_sources"]["wide_acceptance"], "line": 218, "end_line": 281, "description": "Previously executed actual CPU initial/latest/best tensors, scheduler, optimizer moments/steps and frozen/changed parameter checks"}],
        "verified_json_and_source_descriptors": list(VERIFIED.values()),
        "review_operations": {"pt_reads": 0, "tensor_deserializations": 0, "model_constructions": 0, "gpu_calls": 0,
                              "raw_or_feature_array_reads": 0, "outer_test_probability_reads": 0, "sealed_source_modifications": 0,
                              "other20_completed_fits_reaudited": False},
        "all_csbrain15_completion_claimed": False, "full_plan_completion_claimed": False, "final_semantic_companion": False,
    }
    path = REPORT / ("independent_csbrain_extended_region_selection_review_" + str(time.time_ns()) + ".json")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"artifact": desc(path), "passed": True, "csbrain_snapshot_scope": "6/15", "ft_snapshot_scope": "21/135", "pt_reads": 0}, ensure_ascii=True))


if __name__ == "__main__": main()

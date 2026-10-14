"""Small cross-seed review from existing CPU evidence; JSON/source reads only."""
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
    assert path.suffix in (".json", ".py"), "Only JSON and source are allowed"
    return {"path": path.relative_to(WORK).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def check(value):
    actual = desc(WORK / value["path"])
    assert actual["sha256"] == value["sha256"], value["path"]
    verified[actual["path"]] = actual
    return WORK / actual["path"]


def read(value):
    return json.loads(check(value).read_text(encoding="utf-8"))


def main():
    binding = {"path": "output/analysis/eeg_all_candidates_v1_20261008/completed_finetune_checkpoint_audit_v2_1791514413482603500.json", "sha256": "33bf976db07d77bc238cbff1a6d7a1d77521378f636db3a28dcf55f93b9854db"}
    audit = read(binding)
    assert audit["status"] == "partial_verified" and audit["verified_completed_finetune_fits"] == 6
    assert audit["checkpoint_loads"] == 18 and not audit["errors"] and len(audit["pending_fits"]) == 129
    assert audit["full_360_plan_completion_claimed"] is audit["all_135_finetune_checkpoints_verified"] is False
    check(audit["auditor_source"])
    assert audit["auditor_source"]["sha256"] == "12bbb62d025bb2bbacfefa68677a9707185ae5943927b245b1e817a0124802bb"
    for source in audit["delegated_sources"].values(): check(source)
    registry = read(audit["registration"])
    rows = {row["seed"]: row for row in audit["completed_rows"] if row["condition"] == "EEGPT-lastblock-finetune" and row["fold"] == 1}
    assert set(rows) == {11, 22}
    metadata, initializer_evidence = {}, {}
    for seed, row in rows.items():
        meta = metadata[seed] = read(row["metadata"])
        completion, status, lock = (read(row[key]) for key in ("completion", "status_artifact", "condition_source_lock"))
        assert row["passed"] is True and meta["seed"] == seed and meta["fold"] == 1 and meta["condition"] == row["condition"]
        assert completion == status["evidence"] and status["status"] == "training_complete"
        assert completion["registry"] == lock["registration"] == audit["registration"]
        assert completion["metadata"] == row["metadata"] and meta["condition_source_lock"] == row["condition_source_lock"]
        for role in ("train", "val", "test"): assert meta[role + "_ids"] == registry["folds"]["fold_1"][role]
        assert meta["test_forward_count"] == 0 and meta["fresh_head"] is meta["fresh_official_pretrained"] is True
        assert meta["frozen_best_head_inherited"] is False and meta["adapter_audit"]["fresh_head_seed"] == seed
        assert meta["adapter_audit"]["initialized_from_frozen_best_head"] is False
        for key in ("wrapper_source", "delegated_runner_source", "luna_adapter_source"): check(meta["versioned_finetune_implementation"][key])
        proof = row["cpu_checkpoint_audit"]
        assert proof["cpu_tensors_verified"] is True and proof["checkpoint_loads"] == 3 and proof["only_map_location_cpu"] is True
        assert proof["model_constructions"] == proof["gpu_operations"] == 0
        initial, = [value for value in proof["checkpoints"] if Path(value["path"]).name == "initial.pt"]
        zero, = [value for value in proof["durable_counter_boundaries"] if value["path"] == "initial.pt"]
        assert (zero["epoch"], zero["optimizer_updates"], zero["backward_calls"]) == (0, 0, 0)
        initializer_evidence[str(seed)] = {"metadata": row["metadata"], "initial_checkpoint_bound_by_existing_cpu_audit": initial,
                                         "actual_cpu_verified_full_state_digest": meta["initial_state_sha256"], "fresh_head_seed": seed}
    left, right = metadata[11], metadata[22]
    assert left["initial_state_sha256"] != right["initial_state_sha256"]
    assert left["selected_encoder_initial_state_sha256"] == right["selected_encoder_initial_state_sha256"]
    assert left["adapter_audit"]["encoder_state_sha256"] == right["adapter_audit"]["encoder_state_sha256"]
    assert left["adapter_audit"]["checkpoint"] == right["adapter_audit"]["checkpoint"]
    assert left["adapter_audit"]["weight_sources"] == right["adapter_audit"]["weight_sources"]
    assert left["frozen_initial_sha256"] == right["frozen_initial_sha256"] == right["frozen_final_sha256"]
    assert (right["epochs"], right["best_epoch"], right["head_optimizer_updates"], right["backbone_optimizer_updates"], right["backward_calls"]) == (82, 52, 328, 288, 2214)
    assert right["selected_checkpoint_phase"] == "lastblock_finetune" and right["selected_encoder_changed_parameter_count"] == right["terminal_encoder_changed_parameter_count"] == 12
    assert right["selected_head_changed_parameter_count"] == right["terminal_head_changed_parameter_count"] == 6
    assert len({right[name] for name in ("selected_encoder_initial_state_sha256", "selected_encoder_state_sha256", "terminal_encoder_state_sha256")}) == 3
    completion = read(rows[22]["completion"])
    history, validation = (read(completion[name]) for name in ("train_history", "validation_history"))
    assert len(history) == len(validation) == 82
    losses = []
    for epoch, (train, val) in enumerate(zip(history, validation), 1):
        assert train["epoch"] == val["epoch"] == epoch and val["val_ce"] == train["val_eval_ce"]
        assert train["optimizer_updates"] == 4 * epoch and train["backward_calls"] == 27 * epoch
        assert train["unfrozen"] is (epoch > 10)
        losses.append(val["val_ce"])
        decision = P.training_decision("finetune", losses, extended=False)
        assert decision["stop"] is (epoch == 82) and decision["extend"] is False
    assert decision["best_epoch"] == 52 and decision["reason"] == right["stop_reason"] == "validation_patience"
    assert right["extended"] is False
    terminal, = [value for value in rows[22]["cpu_checkpoint_audit"]["durable_counter_boundaries"] if value["path"] == "latest.pt"]
    assert (terminal["epoch"], terminal["optimizer_updates"], terminal["backward_calls"]) == (82, 328, 2214)
    groups = terminal["optimizer_groups"]
    assert groups["group_names"] == ["head", "backbone"] and (groups["head_parameters"], groups["backbone_parameters"]) == (6, 12) and groups["moment_shapes_verified"] is True
    for value in list(verified.values()): check(value)
    result = {"kind": "independent_first_cross_seed_initializer_review", "passed": True,
              "created_at": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/resume_review", "review_source": desc(Path(__file__)),
              "actual_cpu_audit": binding, "scope": "seed11 versus seed22, EEGPT fold1; JSON/source and already-executed CPU proof only",
              "snapshot": {"verified_ft_fits": 6, "registered_ft_fits": 135, "actual_cpu_loads": 18, "errors": 0},
              "same_registered_split": {"fold": 1, "train": 27, "validation": 3, "test_ids_declared_only": 7},
              "initializers": initializer_evidence,
              "common_encoder_initial_state_sha256": right["selected_encoder_initial_state_sha256"],
              "common_author_adapter_unprefixed_encoder_state_sha256": right["adapter_audit"]["encoder_state_sha256"],
              "common_author_weight_descriptor": right["adapter_audit"]["checkpoint"],
              "full_initial_digests_distinct_after_new_head_seed": True,
              "proof_basis": "Existing actual wide.deep_checkpoint verified each initial.pt full tensor-state digest against its metadata; independent wrapper verified actual seed/fold and zero-update boundary. Shared encoder initializer disclosure/weight binding plus seeded fresh-head source explains distinct full-state digests; not a filename-only seed claim.",
              "new_seed22_fold1": {"epochs": 82, "best_epoch": 52, "first_registered_stop": 82, "patience": 30,
                                    "head_steps_each": 328, "backbone_steps_each": 288, "backwards": 2214,
                                    "head_parameters": 6, "backbone_parameters": 12, "selected_phase": "lastblock_finetune",
                                    "selected_encoder_changed": 12, "terminal_encoder_changed": 12, "frozen_state_unchanged": True},
              "source_evidence": [
                  {"source": audit["delegated_sources"]["finetune_v1"], "line": 124, "end_line": 148, "claim": "Load official encoder then create a fresh native head inside fork_rng/manual_seed(seed)"},
                  {"source": audit["delegated_sources"]["wide_acceptance"], "line": 218, "end_line": 252, "claim": "Actual CPU loads verify initial tensor-state digest and terminal AdamW steps/LRs/groups"},
                  {"source": audit["auditor_source"], "line": 190, "end_line": 247, "claim": "Existing CPU load wrapper validates checkpoint seed/fold/registration and scalar boundaries"}],
              "verified_json_and_source_descriptors": list(verified.values()),
              "review_operations": {"pt_files_read": 0, "tensor_deserializations": 0, "models_constructed": 0, "gpu_calls": 0,
                                    "raw_arrays_read": 0, "test_probabilities_read": 0, "sealed_sources_changed": 0},
              "full_plan_completion_claimed": False, "final_semantic_companion": False}
    destination = REPORT / ("independent_first_cross_seed_finetune_review_" + str(time.time_ns()) + ".json")
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"artifact": desc(destination), "passed": True, "verified_descriptors": len(verified), "pt_files_read": 0}, ensure_ascii=True))


if __name__ == "__main__": main()

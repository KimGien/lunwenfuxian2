"""Read-only CPU audit of completed registered FT fits, including partial queues.

Only training_complete FT identities can reach checkpoint loading. The wide
acceptance audit's check_fit/deep_checkpoint are reused; its all-plan audit,
evaluation methods and test gate are never called. No prediction/raw arrays,
model constructor or GPU operation is used. Reports are new immutable snapshots.
"""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import protocol as P, registry as R, storage as S

SOURCE = Path(__file__).resolve()
WIDE_SOURCE = SOURCE.with_name("audit_full_plan_acceptance_v3.py")
WIDE_SHA256 = "892e7a71265e8730c4deaf08f4a84579369ea639865ba5def1233b68fa26e64e"
F2_SOURCE = WORK / "tridep_ccml/eeg_all_candidates/finetune_v2.py"
F2_SHA256 = "fa53183ea4927daed5dcec0204e8a28ab09940f589742debda884402581c1a8b"
F1_SOURCE = F2_SOURCE.with_name("finetune.py")
F1_SHA256 = "cda19474654233c95072acae475174d9011bb0739be650a597194f04c4b0aa24"
COUNTERS = ("forward", "encoder_forward", "encoder_windows", "encode", "backward", "optimizer_step")


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def seconds(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def delegated_sources():
    for path, expected in ((WIDE_SOURCE, WIDE_SHA256), (F2_SOURCE, F2_SHA256), (F1_SOURCE, F1_SHA256)):
        require(S.digest(path) == expected, "Reviewed source changed: " + str(path))
    return {"wide_acceptance": S.desc(WIDE_SOURCE), "finetune_v2": S.desc(F2_SOURCE),
            "finetune_v1": S.desc(F1_SOURCE), "registry": S.desc(Path(R.__file__)),
            "protocol": S.desc(Path(P.__file__)), "storage": S.desc(Path(S.__file__))}


def load_wide():
    require(S.digest(WIDE_SOURCE) == WIDE_SHA256, "Reviewed wide acceptance source changed")
    spec = importlib.util.spec_from_file_location("completed_ft_wide_v3", WIDE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finetune_matrix(registration):
    conditions = {condition.condition_id for condition in P.CONDITIONS if condition.policy_key == "finetune"}
    expected = {R.fit_id(condition, seed, fold) for condition in conditions for seed in P.SEEDS for fold in P.FOLDS}
    fits = [fit for fit in registration["fits"] if fit["condition"] in conditions]
    require(len(conditions) == 9 and len(expected) == 135 and len(fits) == len(expected)
            and {fit["fit_id"] for fit in fits} == expected, "Registered FT matrix is not the exact nine by fifteen identities")
    for fit in fits:
        require(fit["fit_id"] == R.fit_id(fit["condition"], fit["seed"], fit["fold"]), "FT identity fields differ")
        split = registration["folds"][f"fold_{fit['fold']}"]
        require(all(fit[role + "_ids"] == split[role] for role in ("train", "val", "test")), "Registered FT split identity differs")
    return fits


def trace_inventory(fit):
    """Only F2 formal attempts for this completed identity; no engineering costs."""
    rows = []
    root = S.OUT / "finetune_v2_attempts" / fit["condition"]
    for path in sorted(root.glob("*/attempt.json")):
        before = S.desc(path)
        record = S.read(path)
        if record.get("kind") != "formal" or (record.get("seed"), record.get("fold")) != (fit["seed"], fit["fold"]):
            continue
        require(before == S.desc(path), "Formal attempt changed during JSON read")
        require(record.get("condition") == fit["condition"], "Attempt path and condition differ")
        rows.append({"artifact": before, "record": record})
    return rows


def trace_boundary(fit, metadata, inputs):
    """Compare physical observations with durable counts without adding the two.

F2 has no persisted resume-baseline per attempt, so per-attempt durable
attribution and exact unobserved physical calls remain null. Identical copied
records are aliases; F1 failure counters are never inputs to this subtotal.
"""
    updates, backwards = metadata["optimizer_updates"], metadata["backward_calls"]
    require(all(integer(metadata[key]) for key in ("epochs", "optimizer_updates", "backward_calls", "head_optimizer_updates", "backbone_optimizer_updates"))
            and metadata["epochs"] > 10, "Invalid completed FT integer counters/epochs")
    require(integer(updates) and updates > 0 and integer(backwards) and backwards > 0, "Invalid durable FT counters")
    require(metadata["head_optimizer_updates"] == updates, "Head counter differs from durable updates")
    require(metadata["backbone_optimizer_updates"] == math.ceil(len(fit["train_ids"]) / 8) * (metadata["epochs"] - 10),
            "Backbone counter differs from post-warmup durable epochs")
    require(inputs, "Completed FT has no observed F2 formal attempt trace")
    unique, aliases, seen_paths, seen_hashes = [], [], set(), {}
    for item in inputs:
        artifact, record = item["artifact"], item["record"]
        require(artifact["path"] not in seen_paths, "Same attempt path supplied twice")
        seen_paths.add(artifact["path"])
        require((record.get("condition"), record.get("kind"), record.get("seed"), record.get("fold")) ==
                (fit["condition"], "formal", fit["seed"], fit["fold"]), "F2 formal attempt identity differs")
        require(type(record.get("seed")) is int and type(record.get("fold")) is int, "Attempt seed/fold must be strict integers")
        provenance = record["implementation"]
        require(provenance["version"] == 2 and provenance["wrapper_source"] == S.desc(F2_SOURCE)
                and provenance["delegated_runner_source"] == S.desc(F1_SOURCE)
                and provenance["luna_adapter_source"] == S.desc(F2_SOURCE.parent / "adapters/luna_v2.py")
                and provenance["training_math_sampler_warmup_optimizer_stop_and_resume_unchanged"] is True,
                "Attempt implementation is not the sealed F2 delegation")
        state = record["status"]
        require(state in ("running", "complete", "failed"), "Unknown F2 attempt status")
        require(isinstance(record.get("started_at"), str) and record["started_at"], "Attempt start observation missing")
        for kind in COUNTERS:
            started, returned = record[kind + "_started"], record[kind + "_returned"]
            require(integer(started) and integer(returned) and returned <= started, "Invalid F2 counter: " + kind)
            if kind != "encoder_windows":
                require(started - returned <= 1, "F2 serial observer has more than one outstanding call: " + kind)
            if state == "complete":
                require(started == returned, "Main-return complete attempt retains an unreturned call")
        require(record["encode_started"] == record["encoder_forward_started"]
                and record["encode_returned"] == record["encoder_forward_returned"], "F2 encode aliases differ")
        require(record["forward_returned"] >= record["backward_returned"] >= record["optimizer_step_returned"],
                "F2 formal forward/backward/optimizer return boundary differs")
        if state != "running":
            require(seconds(record.get("seconds")) and isinstance(record.get("finished_at"), str) and record["finished_at"],
                    "Terminal F2 attempt lacks actual elapsed observation")
        if artifact["sha256"] in seen_hashes:
            aliases.append({"artifact": artifact, "alias_of": seen_hashes[artifact["sha256"]], "cost_counted_again": False})
            continue
        seen_hashes[artifact["sha256"]] = artifact
        unique.append({"artifact": artifact, "status": state, "seconds_this_attempt": record.get("seconds"),
                       "observed_counters": {key: record[key] for kind in COUNTERS for key in (kind + "_started", kind + "_returned")},
                       "resume_durable_baseline": None, "durable_updates_attributed_to_this_attempt": None})
    totals = {key: sum(row["observed_counters"][key] for row in unique) for kind in COUNTERS for key in (kind + "_started", kind + "_returned")}
    require(any(row["status"] == "complete" for row in unique), "No observed main-return successful F2 formal attempt")
    require(all(row["status"] != "running" for row in unique), "Completed fit still has a mutable running F2 attempt")
    require(totals["optimizer_step_returned"] >= updates and totals["backward_returned"] >= backwards,
            "Recorded F2 returns do not cover the verified durable trajectory")
    return {"passed": True, "fit_id": fit["fit_id"], "logical_completed_fits_counted": 1,
            "durable_optimizer_updates_counted_once": updates, "durable_backward_calls_counted_once": backwards,
            "attempts": unique, "duplicate_content_aliases": aliases, "recorded_physical_call_subtotals": totals,
            "optimizer_returns_above_durable_known_lower_bound": totals["optimizer_step_returned"] - updates,
            "backward_returns_above_durable_known_lower_bound": totals["backward_returned"] - backwards,
            "physical_optimizer_calls_exact": None, "physical_backward_calls_exact": None,
            "all_historical_wall_seconds_exact": None, "per_attempt_resume_baselines_known": False,
            "unreturned_started_calls_are_not_claimed_as_completed": True,
            "legacy_failure_counters_added_to_f2": False, "physical_and_durable_counts_added_together": False,
            "nested_metadata_seconds_added_to_attempt_seconds": False,
            "zero_step_complete_attempts_can_be_existing_completion_rechecks": True}


def optimizer_group_boundary(payload, contract):
    """Use sealed optimizer_for ordering: head first, selected encoder second."""
    names = contract["trainable_parameter_names"]
    heads = [name for name in names if name.startswith("head.")]
    backbone = [name for name in names if name.startswith("encoder.")]
    require(heads and backbone and len(names) == len(set(names)) == len(heads) + len(backbone), "FT contract parameter groups are not exact head/encoder names")
    ordered = heads + backbone
    optimizer = payload["optimizer_state_dict"]
    groups = optimizer["param_groups"]
    require(len(groups) == 2 and [group.get("group_name") for group in groups] == ["head", "backbone"],
            "Actual optimizer groups are not ordered head/backbone")
    ids = [identifier for group in groups for identifier in group["params"]]
    require(all(type(identifier) is int for identifier in ids) and ids == list(range(len(ordered)))
            and len(groups[0]["params"]) == len(heads) and len(groups[1]["params"]) == len(backbone),
            "Actual optimizer serialized IDs/group coverage differ from sealed constructor order")
    require(set(ordered).issubset(payload["state_dict"]), "Contract optimizer parameter missing from actual tensor state")
    if payload["epoch"] == 0:
        require(optimizer["state"] == {}, "Initial AdamW state is not empty before training")
    else:
        require(set(optimizer["state"]) == set(ids), "Actual optimizer state IDs differ from parameter group IDs")
        for identifier, name in enumerate(ordered):
            state = optimizer["state"][identifier]
            require(all(tuple(state[key].shape) == tuple(payload["state_dict"][name].shape) for key in ("exp_avg", "exp_avg_sq")),
                    "Actual AdamW moment shape differs from source-bound parameter order")
    return {"group_names": ["head", "backbone"], "head_parameters": len(heads), "backbone_parameters": len(backbone),
            "serialized_id_parameter_names": ordered, "moment_shapes_verified": payload["epoch"] > 0,
            "ordering_basis": "sealed finetune.optimizer_for plus condition contract; no model reconstructed"}


def checkpoint_payload(path, payload, fit, row, metadata, lock, registry):
    require(all(payload.get(key) == value for key, value in {"condition": fit["condition"], "seed": fit["seed"],
            "fold": fit["fold"], "registry": registry, "condition_source_lock": lock}.items()), "CPU checkpoint identity/source binding differs")
    if path.name == "initial.pt":
        require(all(type(payload.get(name)) is int and payload[name] == 0 for name in ("epoch", "optimizer_updates", "backward_calls")),
                "Initial FT checkpoint is not an exact zero-update boundary")
    elif path.name == "latest.pt":
        require(all(type(payload.get(name)) is int and payload[name] == expected for name, expected in {
                "epoch": row["epochs"], "optimizer_updates": metadata["optimizer_updates"], "backward_calls": metadata["backward_calls"]}.items()),
                "Latest checkpoint durable counters differ from terminal metadata")
        require(payload["history"][-1]["optimizer_updates"] == payload["optimizer_updates"]
                and payload["history"][-1]["backward_calls"] == payload["backward_calls"], "Latest checkpoint/history counter boundary differs")
    elif path.name == "best.pt":
        require(type(payload.get("best_epoch")) is int and payload["best_epoch"] == row["best_epoch"], "Selected checkpoint epoch identity differs")
    else:
        raise ValueError("Unexpected checkpoint in completed FT audit")
    return {"path": path.name, "epoch": payload.get("epoch", payload.get("best_epoch")),
            "optimizer_updates": payload.get("optimizer_updates"), "backward_calls": payload.get("backward_calls"),
            "selected_checkpoint_counters_not_inferred_from_terminal": True}


def cpu_audit(wide, fit, row, contract, metadata, lock, operations):
    import torch
    folder = R.fit_directory(fit["condition"], fit["seed"], fit["fold"]).resolve()
    paths = {folder / name for name in ("initial.pt", "latest.pt", "best.pt")}
    bindings = [S.desc(path) for path in sorted(paths)]
    original_load, observed = torch.load, []

    def load(path, *args, **kwargs):
        path = Path(path).resolve()
        require(path in paths and not args and kwargs == {"map_location": "cpu", "weights_only": False},
                "Delegated load is not one of the three exact CPU FT checkpoint reads")
        require(path.name not in {item["path"] for item in observed}, "Delegated CPU checkpoint loaded twice")
        operations["cpu_checkpoint_loads_started"] += 1
        payload = original_load(path, **kwargs)
        operations["cpu_checkpoint_loads_returned"] += 1
        boundary = checkpoint_payload(path, payload, fit, row, metadata, lock, R.registration_descriptor())
        if path.name in ("initial.pt", "latest.pt"):
            boundary["optimizer_groups"] = optimizer_group_boundary(payload, contract)
        observed.append(boundary)
        return payload

    def forbidden_gpu(*args, **kwargs):
        operations["gpu_initialization_attempts_blocked"] += 1
        raise RuntimeError("GPU use forbidden in this CPU audit")

    def forbidden_model(*args, **kwargs):
        operations["model_construction_attempts_blocked"] += 1
        raise RuntimeError("Model construction forbidden in this CPU audit")

    with patch.object(torch, "load", load), patch.object(torch.cuda, "_lazy_init", forbidden_gpu), \
            patch.object(torch.nn.Module, "__init__", forbidden_model):
        result = wide.deep_checkpoint(fit, row, contract, metadata)
    require({item["path"] for item in observed} == {"initial.pt", "latest.pt", "best.pt"}, "Incomplete actual CPU checkpoint read set")
    require(bindings == [S.desc(path) for path in sorted(paths)], "Completed checkpoint changed during CPU inspection")
    require(result.get("cpu_tensors_verified") is True and result.get("model_constructions") == 0, "Delegated CPU tensor proof missing")
    return {**result, "checkpoints": bindings, "durable_counter_boundaries": observed,
            "checkpoint_loads": len(observed), "gpu_operations": 0, "only_map_location_cpu": True}


def completed_fit_audit(wide, fit, status, registration, locks, operations):
    folder = R.fit_directory(fit["condition"], fit["seed"], fit["fold"])
    inputs = trace_inventory(fit)
    if any(item["record"].get("status") == "running" for item in inputs):
        return {"deferred": True, "fit_id": fit["fit_id"], "reason": "training_complete recorded but F2 wrapper attempt not yet terminal",
                "attempts": [item["artifact"] for item in inputs], "checkpoint_loads": 0}
    status_binding = S.desc(folder / "status.json")
    if fit["condition"] not in locks:
        lock = R._condition_lock(fit["condition"])
        locks[fit["condition"]] = (lock, S.read(S.checked(lock))["contract"])
    lock, contract = locks[fit["condition"]]
    row = wide.check_fit(fit, status, registration, lock, contract)
    metadata = S.read(S.checked(row["metadata"]))
    boundary = trace_boundary(fit, metadata, inputs)
    legacy = [{"artifact": S.desc(path), "observations": S.read(path), "added_to_any_cost_counter": False,
               "resume_baseline_disambiguated": False} for path in sorted(folder.glob("failure_*.json"))]
    deep = cpu_audit(wide, fit, row, contract, metadata, lock, operations)
    require(status_binding == S.desc(folder / "status.json") and R.fit_status(fit["condition"], fit["seed"], fit["fold"]) == status,
            "Completed fit status changed during audit")
    require([item["artifact"] for item in inputs] == [item["artifact"] for item in trace_inventory(fit)], "F2 attempt inventory changed during audit")
    for record in (row["metadata"], row["completion"], lock, *[item["artifact"] for item in inputs], *[item["artifact"] for item in legacy]):
        S.checked(record)
    return {**row, "status_artifact": status_binding, "cpu_checkpoint_audit": deep, "f2_physical_durable_boundary": boundary,
            "legacy_failure_records_not_added": legacy, "passed": True}


def audit():
    sources = delegated_sources()
    registration = R.load_registration(verify_sources=False)
    registry = R.registration_descriptor()
    fits = finetune_matrix(registration)
    pending, deferred, errors, rows, locks, counts = [], [], [], [], {}, Counter()
    operations = {"cpu_checkpoint_loads_started": 0, "cpu_checkpoint_loads_returned": 0,
                  "model_construction_attempts_blocked": 0, "gpu_initialization_attempts_blocked": 0}
    wide = None
    completed_observed = 0
    for fit in fits:
        try:
            status = R.fit_status(fit["condition"], fit["seed"], fit["fold"])
            counts.update([status["status"]])
            if status["status"] != "training_complete":
                pending.append({"fit_id": fit["fit_id"], "status": status["status"], "reason": status.get("reason")})
                continue
            completed_observed += 1
            require(status.get("condition") == fit["condition"] and type(status.get("seed")) is int and status["seed"] == fit["seed"]
                    and type(status.get("fold")) is int and status["fold"] == fit["fold"], "Completed FT status identity differs")
            if wide is None:
                wide = load_wide()
            result = completed_fit_audit(wide, fit, status, registration, locks, operations)
            if result.get("deferred"):
                deferred.append(result)
            else:
                rows.append(result)
        except Exception as error:
            errors.append({"fit_id": fit["fit_id"], "error_type": type(error).__name__, "error": str(error)})
    require(registry == R.registration_descriptor(), "Immutable registration changed during audit")
    require(sources == delegated_sources(), "Reviewed audit sources changed during audit")
    all_verified = len(rows) == len(fits) and not pending and not deferred and not errors
    state = "failed" if errors else "all_finetune_verified" if all_verified else "partial_verified" if rows else "pending"
    return {"schema_version": 2, "created_at": S.now(), "auditor_source": S.desc(SOURCE), "delegated_sources": sources,
            "registration": registry, "status": state, "scope": "registered completed finetune checkpoints only",
            "registered_total_fits": registration["formal_fits"], "registered_finetune_fits": len(fits),
            "registered_finetune_conditions": len({fit["condition"] for fit in fits}), "finetune_status_counts": dict(counts),
            "training_complete_finetune_observed": completed_observed, "verified_completed_finetune_fits": len(rows),
            "all_135_finetune_checkpoints_verified": all_verified, "partial_snapshot": not all_verified,
            "not_full_plan_completion_certificate": True, "full_360_plan_completion_claimed": False,
            "completed_rows": rows, "pending_fits": pending, "deferred_completed_fits": deferred, "errors": errors,
            "durable_optimizer_updates_verified_subtotal": sum(row["updates"] for row in rows),
            "durable_backward_calls_verified_subtotal": sum(row["f2_physical_durable_boundary"]["durable_backward_calls_counted_once"] for row in rows),
            "logical_completed_fits_counted": len(rows), "checkpoint_loads": operations["cpu_checkpoint_loads_returned"],
            "observed_auditor_operations_including_failed_checks": operations,
            "model_constructions": 0, "gpu_operations": 0, "non_finetune_checkpoints_read": 0,
            "raw_arrays_loaded": False, "outer_prediction_artifacts_read": False, "test_gate_called": False,
            "evaluation_driver_called": False, "wide_all_plan_audit_called": False,
            "historical_physical_optimizer_calls_exact": None, "historical_wall_seconds_exact": None,
            "method": "Reuse reviewed wide v3 JSON/math and CPU tensors on genuine completed FT only; observer inspects those same three CPU loads for scalar boundaries. F2 calls and durable counters remain separate; failure JSON is supplementary only."}


def main():
    result = audit()
    path = S.REPORT / ("completed_finetune_checkpoint_audit_v2_" + str(time.time_ns()) + ".json")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"artifact": S.desc(path), "status": result["status"], "completed_finetune_observed": result["training_complete_finetune_observed"],
                      "verified": result["verified_completed_finetune_fits"], "expected": result["registered_finetune_fits"],
                      "checkpoint_loads": result["checkpoint_loads"], "errors": result["errors"]}, ensure_ascii=True))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

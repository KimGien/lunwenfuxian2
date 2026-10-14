"""Read-only v5 CPU audit of completed FT fits, including active partial queues.

Only training_complete FT identities can reach checkpoint loading. The wide
acceptance audit's check_fit/deep_checkpoint are reused; its all-plan audit,
evaluation methods and test gate are never called. No prediction/raw arrays,
model constructor or GPU operation is used. Reports are new immutable snapshots.
Trace identity is screened before binding; bounded transient JSON reads never
turn unknown or changing attempt evidence into a completed-fit certificate.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
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
from tridep_ccml.eeg_all_candidates import execution_acceleration_v1 as X
from tridep_ccml.eeg_all_candidates import lost_f3_closure_v1 as L

SOURCE = Path(__file__).resolve()
WIDE_SOURCE = SOURCE.with_name("audit_full_plan_acceptance_v3.py")
WIDE_SHA256 = "892e7a71265e8730c4deaf08f4a84579369ea639865ba5def1233b68fa26e64e"
F2_SOURCE = WORK / "tridep_ccml/eeg_all_candidates/finetune_v2.py"
F2_SHA256 = "fa53183ea4927daed5dcec0204e8a28ab09940f589742debda884402581c1a8b"
F1_SOURCE = F2_SOURCE.with_name("finetune.py")
F1_SHA256 = "cda19474654233c95072acae475174d9011bb0739be650a597194f04c4b0aa24"
COUNTERS = ("forward", "encoder_forward", "encoder_windows", "encode", "backward", "optimizer_step")
TRACE_PERMISSION_RETRY_DELAYS = (.02, .04, .08)


class TraceDeferred(RuntimeError):
    """A known matching trace cannot yet support a stable completed snapshot."""

    def __init__(self, path, reason):
        self.path = Path(path).resolve().relative_to(WORK).as_posix()
        self.reason = reason
        super().__init__(reason + ": " + self.path)


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def seconds(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def delegated_sources():
    require(S.digest(SOURCE.with_name("audit_completed_finetune_checkpoints_v4.py")) ==
            "17c7a4074429b1e69366c1b66f0178f577dc29f9c8b7e822a3d4061b80973ff0", "Sealed CPUv4 changed")
    for path, expected in ((WIDE_SOURCE, WIDE_SHA256), (F2_SOURCE, F2_SHA256), (F1_SOURCE, F1_SHA256)):
        require(S.digest(path) == expected, "Reviewed source changed: " + str(path))
    require(S.digest(SOURCE.with_name("audit_completed_finetune_checkpoints_v3.py")) ==
            "307d6feabb9b4964db335fd9a0390e3026015cd57f6205117de125c79508c6d7", "Original reviewed CPU auditor changed")
    return {"previous_cpu_auditor_v4": S.desc(SOURCE.with_name("audit_completed_finetune_checkpoints_v4.py")),
            "lost_f3_closure_verifier": S.desc(Path(L.__file__)), "wide_acceptance": S.desc(WIDE_SOURCE), "finetune_v2": S.desc(F2_SOURCE),
            "finetune_v1": S.desc(F1_SOURCE), "registry": S.desc(Path(R.__file__)),
            "protocol": S.desc(Path(P.__file__)), "storage": S.desc(Path(S.__file__)), "execution_acceleration": S.desc(Path(X.__file__)),
            "previous_cpu_auditor": S.desc(SOURCE.with_name("audit_completed_finetune_checkpoints_v3.py"))}


def load_wide():
    require(S.digest(WIDE_SOURCE) == WIDE_SHA256, "Reviewed wide acceptance source changed")
    spec = importlib.util.spec_from_file_location("completed_ft_wide_v4", WIDE_SOURCE)
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


def read_trace_bytes(path):
    """Four bounded read attempts for transient replacement/open races."""
    for index in range(len(TRACE_PERMISSION_RETRY_DELAYS) + 1):
        try:
            return Path(path).read_bytes()
        except (PermissionError, FileNotFoundError):
            if index == len(TRACE_PERMISSION_RETRY_DELAYS):
                raise
            time.sleep(TRACE_PERMISSION_RETRY_DELAYS[index])


def read_trace_snapshot(path):
    """Initial x-write may temporarily expose missing/partial JSON bytes."""
    for index in range(len(TRACE_PERMISSION_RETRY_DELAYS) + 1):
        try:
            raw = Path(path).read_bytes()
            return raw, json.loads(raw.decode("utf-8-sig"))
        except (PermissionError, FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError):
            if index == len(TRACE_PERMISSION_RETRY_DELAYS):
                raise
            time.sleep(TRACE_PERMISSION_RETRY_DELAYS[index])


def trace_inventory(fit):
    """Screen one decoded byte snapshot before binding only matching traces.

Unrelated active records need no second read or hash-file operation. An
unreadable/invalid identity is never silently skipped. Matching running or
changing traces defer this fit, including during the final post-CPU recheck.
"""
    rows = []
    root = S.OUT / "finetune_v2_attempts" / fit["condition"]
    for path in sorted(root.glob("*/attempt.json")):
        raw, record = read_trace_snapshot(path)
        require(isinstance(record, dict) and record.get("condition") == fit["condition"], "Attempt path and condition differ or identity is unreadable")
        require(record.get("kind") in ("formal", "engineering"), "Unknown attempt kind cannot be omitted")
        if record["kind"] == "engineering":
            require(record.get("seed") is None and record.get("fold") is None, "Engineering attempt identity differs")
            continue
        require(type(record.get("seed")) is int and record["seed"] in P.SEEDS
                and type(record.get("fold")) is int and record["fold"] in P.FOLDS, "Unknown formal attempt identity cannot be omitted")
        if (record["seed"], record["fold"]) != (fit["seed"], fit["fold"]):
            continue
        before = {"path": path.resolve().relative_to(WORK).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}
        stopped_proof = None
        lost_proof = None
        if record.get("status") == "running":
            lost_proof = L.verify_lost_f3_attempt(before, record, fit)
            if lost_proof is None:
                stopped_proof = X.verify_stopped_attempt(before, record, fit)
            if stopped_proof is None and lost_proof is None:
                raise TraceDeferred(path, "Matching F2 attempt is still running")
        try:
            after = read_trace_bytes(path)
        except (PermissionError, FileNotFoundError) as error:
            raise TraceDeferred(path, "Matching F2 attempt is unreadable after identity screening; bounded retry exhausted") from error
        if raw != after:
            raise TraceDeferred(path, "Matching F2 attempt changed during byte binding")
        item = {"artifact": before, "record": record}
        if stopped_proof is not None: item["proven_stopped_handoff"] = stopped_proof
        if lost_proof is not None: item["proven_process_loss"] = lost_proof
        rows.append(item)
    return rows


def recheck_trace_inventory(fit, inputs):
    current = trace_inventory(fit)
    if [item["artifact"] for item in inputs] != [item["artifact"] for item in current]:
        raise TraceDeferred(S.OUT / "finetune_v2_attempts" / fit["condition"], "Matching F2 attempt inventory changed during CPU audit")


def legacy_trace_boundary(fit, metadata, inputs):
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


def trace_boundary(fit, metadata, inputs):
    """Validate actual execution first, reuse unchanged F2 counter arithmetic.

    The temporary normalized copy exists only for the old counter validator.
    Actual input descriptors and actual F3 provenance remain in the result.
    A prior complete F2 fit may have a zero-call F3 completion recheck.
    """
    normalized, execution_rows = [], []
    metadata_execution = None
    if "execution_acceleration" in metadata:
        metadata_execution = X.verify_execution_acceleration(metadata, condition=fit["condition"], seed=fit["seed"], fold=fit["fold"])
    for item in inputs:
        record = item["record"]
        version = record.get("implementation", {}).get("version")
        require(type(version) is int and version in (2, 3), "Unknown actual observer version")
        clone = deepcopy(item)
        if version == 3:
            proof = X.verify_observer_record(record, item["artifact"], fit)
            lost = item.get("proven_process_loss")
            if lost is not None:
                require(lost == L.verify_lost_f3_attempt(item["artifact"], record, fit), "Process-loss closure changed")
                # Adapter values exist only in memory for the sealed legacy shape check.
                # They are never persisted as observations and are restored below.
                clone["record"].update(status="failed", seconds=0, finished_at="process_absence_observed_elapsed_unknown")
            any_work = any(record[key] > 0 for key in X.COUNTER_FIELDS)
            training_work = any(record[kind + suffix] > 0 for kind in ("backward", "optimizer_step") for suffix in ("_started", "_returned"))
            if training_work or (any_work and metadata_execution is not None):
                require(metadata_execution == proof["execution_acceleration"], "Actual F3 work requires matching completed metadata execution binding")
                if record["status"] == "complete":
                    durable = record.get("latest_durable_checkpoint")
                    if durable is not None:
                        require(all(durable.get(key) == metadata[other] for key, other in
                                (("epoch", "epochs"), ("optimizer_updates", "optimizer_updates"), ("backward_calls", "backward_calls"))),
                                "F3 completed process terminal checkpoint commit differs")
                    else:
                        require(record["optimizer_step_returned"] == 0,
                                "F3 completed process with new updates lacks a checkpoint commit observation")
            elif metadata_execution is not None:
                require(metadata_execution == proof["execution_acceleration"], "F3 recheck execution differs from accelerated completed metadata")
            clone["record"]["implementation"] = X.f2_provenance()
            execution_rows.append({"artifact": item["artifact"], "actual_implementation": record["implementation"], **proof,
                                   "proven_process_loss": lost,
                                   "zero_call_existing_completion_recheck": not any_work,
                                   "no_new_training_completion_or_metadata_recovery": not training_work})
        else:
            require("execution_acceleration" not in record, "F2 trace cannot silently claim F3 execution")
            stopped = item.get("proven_stopped_handoff")
            if stopped is not None:
                require(stopped == X.verify_stopped_attempt(item["artifact"], record, fit), "Historical handoff proof changed")
                # Only adapt terminal-shape preconditions for the unchanged
                # counter arithmetic. This placeholder is removed below and is
                # never reported as a measured elapsed value or source status.
                clone["record"].update(status="failed", seconds=0, finished_at="proven_exit_elapsed_unknown")
            execution_rows.append({"artifact": item["artifact"], "observer_version": 2,
                                   "actual_implementation": record["implementation"], "unflushed_tail_unknown": None,
                                   "counter_snapshot_exact_for_process": None, "proven_stopped_handoff": stopped})
        normalized.append(clone)
    if metadata_execution is not None:
        require(any(row["observer_version"] == 3 and not row["zero_call_existing_completion_recheck"] for row in execution_rows),
                "Accelerated metadata has no actual nonzero F3 attempt")
    result = legacy_trace_boundary(fit, metadata, normalized)
    by_path = {row["artifact"]["path"]: row for row in execution_rows}
    for attempt in result["attempts"]:
        attempt["actual_execution_observation"] = by_path[attempt["artifact"]["path"]]
        if attempt["actual_execution_observation"].get("proven_process_loss"):
            attempt.update(status="process_loss_observed", original_trace_status="running", seconds_this_attempt=None,
                           final_process_elapsed_seconds_unknown=True, graceful_terminal_flush=False,
                           unflushed_tail_counters=None, counters_are_persisted_lower_bounds=True)
        if attempt["actual_execution_observation"].get("proven_stopped_handoff"):
            attempt.update(status="stopped_by_verified_handoff", original_trace_status="running", seconds_this_attempt=None,
                           final_process_elapsed_seconds_unknown=True)
    result.update(actual_observer_versions=sorted({row["observer_version"] for row in execution_rows}),
                  execution_acceleration_verified=metadata_execution,
                  actual_execution_observations=execution_rows,
                  legacy_counter_arithmetic_reused_after_strict_actual_source_verification=True,
                  unflushed_buffered_tails_unknown=any(row["unflushed_tail_unknown"] is True for row in execution_rows),
                  historical_or_unflushed_physical_work_zero_filled=False)
    return result


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
    if any(item["record"].get("status") == "running" and not item.get("proven_stopped_handoff") and not item.get("proven_process_loss") for item in inputs):
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
    recheck_trace_inventory(fit, inputs)
    require(boundary == trace_boundary(fit, metadata, inputs), "Actual observer/qualification source proof changed during CPU audit")
    # Matching traces have just had their same-byte SHA bindings rechecked
    # with bounded retries. Do not reopen them via un-retried S.checked here.
    for record in (row["metadata"], row["completion"], lock, *[item["artifact"] for item in legacy]):
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
        reads_before_fit = operations["cpu_checkpoint_loads_returned"]
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
        except TraceDeferred as error:
            deferred.append({"deferred": True, "fit_id": fit["fit_id"], "reason": error.reason,
                             "trace_path": error.path, "checkpoint_loads": operations["cpu_checkpoint_loads_returned"] - reads_before_fit})
        except Exception as error:
            errors.append({"fit_id": fit["fit_id"], "error_type": type(error).__name__, "error": str(error)})
    require(registry == R.registration_descriptor(), "Immutable registration changed during audit")
    require(sources == delegated_sources(), "Reviewed audit sources changed during audit")
    all_verified = len(rows) == completed_observed == len(fits) == 135 and not pending and not deferred and not errors
    state = "failed" if errors else "all_finetune_verified" if all_verified else "partial_verified" if rows else "pending"
    return {"schema_version": 5, "created_at": S.now(), "auditor_source": S.desc(SOURCE), "delegated_sources": sources,
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
            "trace_read_policy": {"identity_from_single_json_byte_read": True, "unrelated_records_not_reopened_for_hashing": True,
                                  "matching_records_bound_to_two_equal_byte_reads_before_and_after_cpu_audit": True,
                                  "permission_read_attempts_max": len(TRACE_PERMISSION_RETRY_DELAYS) + 1,
                                  "permission_retry_sleep_seconds_max_per_read": sum(TRACE_PERMISSION_RETRY_DELAYS),
                                  "initial_json_retry_errors": ["PermissionError", "FileNotFoundError", "UnicodeDecodeError", "JSONDecodeError"],
                                  "initial_json_attempts_max": len(TRACE_PERMISSION_RETRY_DELAYS) + 1,
                                  "unreadable_unknown_identity_is_error": True, "matching_mutable_trace_defers_fit": True},
            "method": "Reuse unchanged reviewed wide v3 JSON/math and same three CPU tensor loads; separately verify actual F2/F3 observer and immutable execution qualification. Physical observations and durable counters remain separate; buffered unflushed tails remain unknown; only source-bound process-loss closure permits a historical F3 running record."}


def main():
    result = audit()
    path = S.REPORT / ("completed_finetune_checkpoint_audit_v5_" + str(time.time_ns()) + ".json")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"artifact": S.desc(path), "status": result["status"], "completed_finetune_observed": result["training_complete_finetune_observed"],
                      "verified": result["verified_completed_finetune_fits"], "expected": result["registered_finetune_fits"],
                      "checkpoint_loads": result["checkpoint_loads"], "errors": result["errors"]}, ensure_ascii=True))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

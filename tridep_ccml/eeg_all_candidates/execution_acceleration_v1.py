"""Read-only source/JSON contract for explicitly registered FT execution changes.

No tensors, checkpoint bytes, raw arrays, models, GPU APIs or process operations.
Original F2 mathematical provenance is preserved separately from F3 execution.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from . import protocol as P, registry as R, storage as S

SOURCE = Path(__file__).resolve()
SCHEMA = "eeg_finetune_execution_acceleration_v1"
F1_SHA = "cda19474654233c95072acae475174d9011bb0739be650a597194f04c4b0aa24"
F2_SHA = "fa53183ea4927daed5dcec0204e8a28ab09940f589742debda884402581c1a8b"
Q3_SHA = "e135585ca475b733bbc6bfbaa1616fe71f9b8fb1af8f2c0b413d2a9c8565b3a0"
COUNTERS = ("forward", "encoder_forward", "encoder_windows", "encode", "backward", "optimizer_step")
COUNTER_FIELDS = tuple(kind + suffix for kind in COUNTERS for suffix in ("_started", "_returned"))
FLUSH_BOUNDARIES = ("subject_update_before", "subject_update_after", "diagnostics_before", "diagnostics_after",
                    "latest_checkpoint_committed", "fit_before", "fit_returned", "exception")
COUNTER_SEMANTICS = "persisted_lower_bounds_until_graceful_terminal_flush"
SAFE_STOP_SOURCE = S.REPORT / "safe_stop_acceleration_v1.py"
SAFE_STOP_SHA = "6c0c6d084b5463ed3cfa2326cc17bb105fa3a9369fd75527d4912d198489071c"


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def descriptor(path):
    path = Path(path).resolve()
    require(path.is_relative_to(S.WORK) and path.suffix in (".json", ".py"), "Execution verifier only opens workspace JSON/source")
    return {"path": path.relative_to(S.WORK).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(binding):
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "Expected exact source descriptor")
    path = (S.WORK / binding["path"]).resolve()
    require(descriptor(path) == binding, "Execution evidence/source bytes differ: " + str(path))
    return path


def read_bound(binding):
    path = checked(binding)
    require(path.suffix == ".json", "Expected bound JSON")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def f2_provenance():
    """Exact sealed F2 value without importing F2, torch or its model modules."""
    f1, f2 = SOURCE.with_name("finetune.py"), SOURCE.with_name("finetune_v2.py")
    require(descriptor(f1)["sha256"] == F1_SHA and descriptor(f2)["sha256"] == F2_SHA, "Sealed mathematical delegates changed")
    return {"version": 2, "wrapper_source": descriptor(f2), "delegated_runner_source": descriptor(f1),
            "luna_adapter_source": descriptor(SOURCE.parent / "adapters/luna_v2.py"),
            "repairs": ["retain author fixed rotary frequencies", "durable per-call attempt cost and independent engineering probe",
                        "separate terminal training updates from validation-selected checkpoint changes"],
            "training_math_sampler_warmup_optimizer_stop_and_resume_unchanged": True,
            "no_validation_or_test_score_configuration_selection": True}


def load_execution_acceleration(registration_descriptor, *, condition, seed, fold):
    require(condition in P.CONDITION_BY_ID and P.CONDITION_BY_ID[condition].policy_key == "finetune", "Acceleration is registered FT only")
    require(type(seed) is int and seed in P.SEEDS and type(fold) is int and fold in P.FOLDS, "Unregistered strict fit identity")
    path = checked(registration_descriptor)
    expected = S.OUT / "execution_acceleration_v1" / condition / "registration.json"
    require(path == expected.resolve(), "Acceleration registration must be the immutable condition sidecar")
    value = read_bound(registration_descriptor)
    require(value.get("schema") == SCHEMA and type(value.get("version")) is int and value["version"] == 1
            and value.get("status") == "qualified", "Acceleration registration is not qualified v1")
    require(value.get("registry") == R.registration_descriptor(), "Original360 registration differs")
    require(value.get("training_math_unchanged") is True and value.get("single_gpu") is True,
            "Execution sidecar cannot change training math or single-GPU policy")
    require(type(value.get("training_micro_subjects")) is int and value["training_micro_subjects"] == 1
            and type(value.get("training_window_chunk")) is int and value["training_window_chunk"] == 8,
            "Original training microbatch/window chunk differs")
    q3_binding = value.get("previous_execution_registration")
    q3_path = S.OUT / "continuation_queue_v3/registration.json"
    require(q3_binding == descriptor(q3_path) and q3_binding["sha256"] == Q3_SHA, "Original sealed Q3 registration differs")
    q3 = read_bound(q3_binding)
    require(len(q3["sources"]) == 47, "Expected original47 sealed source descriptors")
    for binding in q3["sources"]: checked(binding)
    paths = {"finetune_v3": SOURCE.with_name("finetune_v3.py"), "finetune_v2": SOURCE.with_name("finetune_v2.py"),
             "finetune": SOURCE.with_name("finetune.py"), "validation_batching_v1": SOURCE.with_name("validation_batching_v1.py"),
             "execution_acceleration_v1": SOURCE}
    sources = value.get("sources")
    require(isinstance(sources, dict) and set(sources) == set(paths), "Actual acceleration source set differs")
    for key, source_path in paths.items():
        require(sources[key] == descriptor(source_path), "Acceleration source identity differs: " + key)
    entries = value.get("conditions")
    require(isinstance(entries, dict) and set(entries) == {condition}, "Condition sidecar has an unexpected condition scope")
    entry = entries[condition]
    lock = R._condition_lock(condition)
    require(entry.get("condition_source_lock") == lock, "Original engineering contract differs")
    original = read_bound(lock)
    provenance = f2_provenance()
    require(original["registration"] == value["registry"] and original["contract"]["versioned_finetune_implementation"] == provenance,
            "Original qualified F2 mathematical contract differs")
    chunk = entry.get("diagnostic_window_chunk")
    require(type(chunk) is int and chunk in (8, 16, 32, 60), "Invalid diagnostic-only chunk")
    qualification = entry.get("qualification")
    if qualification is None:
        require(chunk == 8, "Expanded diagnostics require actual qualification")
    else:
        from .validation_batching_v1 import verify_qualification
        registry = read_bound(value["registry"])
        proof = verify_qualification(qualification, condition, registry["folds"]["fold_" + str(fold)], allow_tolerance=False)
        require(proof["selected_chunk"] == chunk, "Qualified and registered diagnostic chunk differ")
        require(proof.get("condition_source_lock") == lock, "Qualification and original condition lock differ")
    return {"schema": SCHEMA, "version": 3, "observer_version": 3, "source": sources["finetune_v3"],
            "delegated_finetune_implementation": provenance, "registration": registration_descriptor,
            "condition_source_lock": lock, "condition": condition, "seed": seed, "fold": fold,
            "diagnostics": {"window_chunk": chunk, "qualification": qualification, "bitwise_equivalence_required": True},
            "training": {"micro_subjects": 1, "window_chunk": 8, "effective_subject_batch": 8, "window_bag_size": 8, "unchanged": True},
            "observer": {"per_call_persistence": False, "flush_boundaries": list(FLUSH_BOUNDARIES), "hard_kill_unflushed_tail": "unknown"}}


def verify_execution_acceleration(value, *, condition, seed, fold, require_terminal=False):
    """Verify a complete metadata object or its explicit execution value."""
    require(isinstance(value, dict), "Expected execution or metadata mapping")
    execution = value.get("execution_acceleration", value)
    require(isinstance(execution, dict), "Missing actual execution acceleration")
    expected = load_execution_acceleration(execution.get("registration"), condition=condition, seed=seed, fold=fold)
    require(execution == expected, "Actual execution differs from immutable qualified sidecar")
    if "execution_acceleration" in value and "versioned_finetune_implementation" in value:
        require(value["versioned_finetune_implementation"] == expected["delegated_finetune_implementation"], "F2 mathematical provenance was redefined")
        require(type(value.get("window_chunk")) is int and value["window_chunk"] == 8
                and type(value.get("micro_subjects")) is int and value["micro_subjects"] == 1,
                "Training/reload/test chunk or microbatch differs")
    if require_terminal:
        require(value.get("status") == "complete" and value.get("graceful_terminal_flush") is True,
                "Expected observed graceful main-return terminal trace")
    return expected


def verify_observer_record(record, artifact, fit):
    """Validate a same-byte-bound F3 snapshot, leaving hard-kill tails unknown."""
    require(isinstance(artifact, dict) and set(artifact) == {"path", "sha256"}, "Bound attempt descriptor required")
    require((record.get("condition"), record.get("kind"), record.get("seed"), record.get("fold")) ==
            (fit["condition"], "formal", fit["seed"], fit["fold"]), "F3 formal attempt identity differs")
    require(type(record.get("seed")) is int and type(record.get("fold")) is int, "Attempt seed/fold must be strict integers")
    execution = verify_execution_acceleration(record["execution_acceleration"], condition=fit["condition"], seed=fit["seed"], fold=fit["fold"])
    implementation = record.get("implementation", {})
    require(type(implementation.get("version")) is int and implementation["version"] == 3
            and implementation.get("wrapper_source") == execution["source"]
            and implementation.get("delegated_wrapper_source") == execution["delegated_finetune_implementation"]["wrapper_source"]
            and implementation.get("delegated_runner_source") == execution["delegated_finetune_implementation"]["delegated_runner_source"],
            "Observer is not source-bound actual F3 over sealed F2/F1")
    require(implementation.get("delegated_finetune_implementation") == execution["delegated_finetune_implementation"]
            and implementation.get("training_math_sampler_warmup_optimizer_stop_and_resume_unchanged") is True
            and implementation.get("no_validation_or_test_score_configuration_selection") is True,
            "F3 mathematical delegation disclosure differs")
    require(record.get("counter_semantics") == COUNTER_SEMANTICS, "Buffered counter semantics missing")
    require(integer(record.get("flush_sequence")) and record["flush_sequence"] > 0
            and record.get("last_flush_reason") in (*FLUSH_BOUNDARIES, "attempt_created"), "Invalid durable flush sequence/boundary")
    flushes = record.get("flush_counts")
    require(isinstance(flushes, dict) and set(flushes).issubset({*FLUSH_BOUNDARIES, "attempt_created"})
            and all(integer(n) and n > 0 for n in flushes.values()) and sum(flushes.values()) == record["flush_sequence"]
            and flushes.get("attempt_created") == 1 and flushes.get(record["last_flush_reason"], 0) > 0,
            "Flush counts/sequence differ")
    require(record.get("dirty_since_last_flush") is False and record.get("dirty_flag_scope") == "snapshot_instant_only"
            and record.get("counter_scope") == "this_process_invocation_not_resumed_trajectory"
            and record.get("encode_is_encoder_forward_alias") is True, "Buffered snapshot scope differs")
    snapshot = record.get("durable_counter_snapshot")
    require(isinstance(snapshot, dict) and set(snapshot) == set(COUNTER_FIELDS)
            and all(integer(record.get(key)) and snapshot[key] == record[key] and type(snapshot[key]) is int for key in COUNTER_FIELDS),
            "Buffered snapshot does not bind all persisted counter fields")
    state = record.get("status")
    require(state in ("running", "complete", "failed"), "Unknown F3 observer state")
    terminal = record.get("graceful_terminal_flush")
    require(type(terminal) is bool and type(record.get("counter_snapshot_exact_for_process")) is bool
            and type(record.get("unflushed_tail_unknown")) is bool, "Buffered exactness flags must be explicit booleans")
    require(record["counter_snapshot_exact_for_process"] is terminal and record["unflushed_tail_unknown"] is not terminal,
            "Counter exactness contradicts graceful terminal flush")
    if terminal:
        require(state in ("complete", "failed") and record["last_flush_reason"] == ("fit_returned" if state == "complete" else "exception"),
                "Only graceful finish/exception flush can close process observation")
        require(record.get("unflushed_tail_counters") == {key: 0 for key in COUNTER_FIELDS}
                and all(type(n) is int for n in record["unflushed_tail_counters"].values()), "Graceful tail must be explicit observed zero counters")
        require(seconds(record.get("seconds")) and isinstance(record.get("finished_at"), str) and record["finished_at"], "Graceful terminal elapsed observation missing")
    else:
        require(state != "complete" and record.get("unflushed_tail_counters") is None, "Unflushed tail cannot be silently zero-filled")
    require(isinstance(record.get("started_at"), str) and record["started_at"], "Attempt start observation missing")
    for kind in COUNTERS:
        started, returned = record[kind + "_started"], record[kind + "_returned"]
        require(returned <= started and (kind == "encoder_windows" or started - returned <= 1), "Invalid serial buffered call boundary: " + kind)
        if state == "complete": require(started == returned, "Main-return complete trace retains unreturned call")
    require(record["encode_started"] == record["encoder_forward_started"] and record["encode_returned"] == record["encoder_forward_returned"], "Encoder aliases differ")
    require(record["forward_returned"] >= record["backward_returned"] >= record["optimizer_step_returned"], "Formal invocation ordering differs")
    require(record.get("started_without_observed_return") == {kind: record[kind + "_started"] - record[kind + "_returned"]
            for kind in ("forward", "encoder_forward", "backward", "optimizer_step")}, "Unreturned-call disclosure differs")
    durable = record.get("latest_durable_checkpoint")
    if durable is not None:
        require(isinstance(durable, dict) and durable.get("path") ==
                (R.fit_directory(fit["condition"], fit["seed"], fit["fold"]) / "latest.pt").relative_to(S.WORK).as_posix()
                and all(integer(durable.get(k)) for k in ("epoch", "optimizer_updates", "backward_calls"))
                and durable.get("checkpoint_save_returned") is True and durable.get("checkpoint_hash") is None
                and durable.get("hash_scope") == "not_rehashed_each_epoch; final_checkpoint_audit_binds_bytes",
                "Durable checkpoint commit observation differs")
    return {"passed": True, "execution_acceleration": execution, "observer_version": 3,
            "persisted_counters_are_lower_bounds": not terminal, "counter_snapshot_exact_for_process": terminal,
            "unflushed_tail_unknown": not terminal, "unflushed_tail_counters": record.get("unflushed_tail_counters"),
            "started_not_returned_calls_are_not_completed_work": True}


def verify_stopped_attempt(artifact, record, fit):
    """Recognize a proven exited historical F2 process without rewriting its trace.

    Checkpoint descriptors are carried as prior CPU proof, never opened here.
    The known counter delta is separate from durable trajectory and observer
    subtotals. The interrupted call outcome and elapsed endpoint stay unknown.
    """
    proofs = []
    for path in sorted((S.OUT / "continuation_queue_v4/handoff").glob("safe_stop_*.json")):
        binding = descriptor(path)
        proof = read_bound(binding)
        if proof.get("f2_attempt", {}).get("path") == artifact["path"]:
            proofs.append((binding, proof))
    if not proofs:
        return None
    require(len(proofs) == 1, "Historical stopped attempt has ambiguous handoff proofs")
    binding, proof = proofs[0]
    require(proof.get("schema") == "acceleration_safe_stop_v1" and proof.get("mode") == "stopped_training"
            and proof.get("passed") is True and proof.get("source") == descriptor(SAFE_STOP_SOURCE)
            and proof["source"]["sha256"] == SAFE_STOP_SHA, "Stopped-process proof lacks the reviewed exact controller")
    require(proof.get("f2_attempt") == artifact and proof["f2_attempt_snapshot"]["sha256"] == artifact["sha256"]
            and read_bound(proof["f2_attempt_snapshot"]) == record, "Original F2 trace and preserved bytes differ")
    require(record.get("status") == "running" and record.get("implementation") == f2_provenance(), "Expected original unchanged running F2 trace")
    require(all(proof.get(k) == fit[k] and type(proof.get(k)) is type(fit[k]) for k in ("condition", "seed", "fold")), "Stopped-process fit identity differs")
    require(proof.get("registration") == R.registration_descriptor()
            and all(proof.get(k) is True for k in ("all_training_processes_stopped", "gpu_idle", "resume_full_state_verified",
                "unreturned_calls_remain_unknown", "old_attempt_seconds_remain_unknown",
                "checkpoint_history_is_authoritative_if_json_write_lagged", "formal_training_batch_precision_selection_budget_unchanged")),
            "Stopped-process exit/full-state/cost-boundary proof incomplete")
    require(proof.get("checkpoint_load_device") == "cpu" and type(proof.get("model_operations")) is int and proof["model_operations"] == 0,
            "Handoff checkpoint proof must be CPU without model operations")
    require(proof.get("stopped_process_ids") == [29932, 47936, 40360]
            and all(type(pid) is int for pid in proof["stopped_process_ids"]), "Stopped process set differs from reviewed controller")
    identities = proof.get("held_process_identities", {})
    require(set(identities) == {"dispatcher", "launcher", "gpu_child"}
            and [identities[k].get("pid") for k in ("dispatcher", "launcher", "gpu_child")] == proof["stopped_process_ids"],
            "Held process identities differ")
    owner = read_bound(proof["old_owner_snapshot"])
    queue_attempt = read_bound(proof["old_q3_attempt_status"])
    require(owner.get("pid") == 29932 and owner.get("child_pid") == 47936
            and owner.get("id") == "a45f0f1e767e44dc83150ab00a3971bc"
            and owner.get("child_job") == "CSBrain-lastblock-finetune-fit"
            and queue_attempt.get("child_pid") == 47936 and queue_attempt.get("job", {}).get("module") == "finetune_v2",
            "Archived owner/child route differs")
    for key in ("epoch", "optimizer_updates", "backward_calls", "recovery_extra_optimizer_updates", "recovery_extra_backward_calls"):
        require(integer(proof.get(key)), "Invalid stopped/durable integer boundary")
    require(record["optimizer_step_returned"] - proof["optimizer_updates"] == proof["recovery_extra_optimizer_updates"]
            and record["backward_returned"] - proof["backward_calls"] == proof["recovery_extra_backward_calls"], "Recovery physical/durable delta differs")
    require(isinstance(proof.get("checkpoint"), dict) and proof["checkpoint"] in proof.get("checkpoint_evidence", [])
            and proof.get("checkpoints") == [proof["checkpoint"]], "Preserved checkpoint descriptor scope differs")
    protected = proof.get("protected_sources", [])
    q3 = read_bound(descriptor(S.OUT / "continuation_queue_v3/registration.json"))
    require(protected == q3["sources"] + [descriptor(S.OUT / "continuation_queue_v3/registration.json")]
            and len(q3["sources"]) == 47, "Protected original source scope differs")
    for source in protected: checked(source)
    return {"passed": True, "proof": binding, "actual_original_status_preserved": "running",
            "process_exit_proven": True, "final_elapsed_seconds": None, "unreturned_callable_outcome": None,
            "durable_checkpoint_descriptor_not_reopened": proof["checkpoint"],
            "durable_epoch": proof["epoch"], "durable_optimizer_updates": proof["optimizer_updates"], "durable_backward_calls": proof["backward_calls"],
            "known_extra_optimizer_returns_outside_checkpoint": proof["recovery_extra_optimizer_updates"],
            "known_extra_backward_returns_outside_checkpoint": proof["recovery_extra_backward_calls"],
            "extra_work_not_added_again_to_observed_trace": True}

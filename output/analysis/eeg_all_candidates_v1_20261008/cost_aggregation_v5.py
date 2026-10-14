"""Read-only cost v5: retain v4 inventory and verify buffered F3 observation scope."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import execution_acceleration_v1 as X

DELEGATE = SOURCE.with_name("cost_aggregation_v4.py")
DELEGATE_SHA = "46af6b28f80332f99b22f9373bb5902c0f7c42e9676c54a56da9c2f62eddbd40"


def load_v4():
    if X.descriptor(DELEGATE)["sha256"] != DELEGATE_SHA:
        raise ValueError("Sealed v4 cost delegate changed")
    spec = importlib.util.spec_from_file_location("cost_v5_sealed_v4", DELEGATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bound_json(binding):
    path = WORK / binding["path"]
    if path.suffix != ".json":
        raise ValueError("Cost review only opens JSON observations")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("Cost input changed after the delegated same-byte snapshot")
    return json.loads(raw.decode("utf-8-sig"))


def buffered_observation(data, artifact):
    if data.get("implementation", {}).get("version") != 3:
        if "execution_acceleration" in data:
            raise ValueError("Execution acceleration requires actual observer version3")
        return None
    fit = {name: data[name] for name in ("condition", "seed", "fold")}
    fit["fit_id"] = fit["condition"] + "/seed_" + str(fit["seed"]) + "/fold_" + str(fit["fold"])
    proof = X.verify_observer_record(data, artifact, fit)
    return {**proof, "flush_sequence": data["flush_sequence"], "last_flush_reason": data["last_flush_reason"],
            "counter_semantics": data["counter_semantics"], "durable_counter_snapshot": data["durable_counter_snapshot"],
            "latest_durable_checkpoint_commit_observation": data.get("latest_durable_checkpoint"),
            "phase_counters_are_not_inferred_from_total_calls": True,
            "training_window_chunk": 8, "diagnostic_window_chunk": proof["execution_acceleration"]["diagnostics"]["window_chunk"],
            "unobserved_optimizer_side_effects_of_unreturned_calls_are_unknown": True}


def augment(result):
    """Never convert persisted buffered subtotals into exact physical totals."""
    result["schema_version"] = 5
    result["delegated_cost_source"] = result["source"]
    result["source"] = X.descriptor(SOURCE)
    result["execution_verifier_source"] = X.descriptor(Path(X.__file__))
    observations, stopped_attempts, errors, affected_categories = [], [], [], set()
    for row in result["cost_records"]:
        if "/finetune_v2_attempts/" not in row["artifact"]["path"]:
            continue
        try:
            data = bound_json(row["artifact"])
            proof = buffered_observation(data, row["artifact"])
            if proof is None:
                if data.get("kind") == "formal" and data.get("status") == "running":
                    fit = {key: data[key] for key in ("condition", "seed", "fold")}
                    stopped = X.verify_stopped_attempt(row["artifact"], data, fit)
                    if stopped is not None:
                        row["proven_stopped_handoff"] = stopped
                        row["seconds_observed"] = None
                        row["seconds_scope"] = "Original running F2 process exited; final elapsed endpoint is unknown"
                        row["original_trace_status_preserved"] = data["status"]
                        stopped_attempts.append({"artifact": row["artifact"], **stopped})
                        affected_categories.add(row["category"])
                continue
            row["actual_execution_observer"] = proof
            row["counter_scope"] = "persisted buffered lower bounds" if proof["unflushed_tail_unknown"] else "gracefully flushed started/returned process observations"
            row["unflushed_tail_counters"] = proof["unflushed_tail_counters"]
            row["known_counts_do_not_cover_unflushed_tail"] = proof["unflushed_tail_unknown"]
            observations.append({"artifact": row["artifact"], **proof})
            if proof["unflushed_tail_unknown"]:
                affected_categories.add(row["category"])
                # A zero observed difference says nothing about a lost buffered tail.
                for name in ("optimizer", "backward", "model_forward", "encoder_forward", "encoder_windows"):
                    row.setdefault("persisted_started_not_returned_lower_bounds", {})[name] = row["observed_counters"][name + "_started_not_returned"]
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
            errors.append({"artifact": row["artifact"], "error_type": type(error).__name__, "error": str(error)})
            affected_categories.add(row["category"])
            row["actual_execution_observer"] = {"passed": False, "scope": "mutable_or_unverified_source_snapshot", "unflushed_tail_unknown": None}
    for category in affected_categories:
        totals = result["category_observations"][category]
        totals["buffered_or_unverified_tail_coverage_unknown"] = True
        for field in totals["counter_totals"].values():
            field["exact_category_total"] = None
            field["known_observed_subtotal_excludes_unknown_buffered_tails"] = True
    result["execution_observer_verification"] = {"source": X.descriptor(Path(X.__file__)), "observations": observations, "errors": errors,
                                                "passed_for_observed_stable_f3_records": not errors,
                                                "hard_kill_unflushed_tail_never_zero_filled": True}
    result["source_bound_stopped_attempts"] = stopped_attempts
    result["all_plan_physical_optimizer_returns_exact"] = None
    result["all_plan_wall_seconds_exact"] = None
    result["rules"].extend(["F3 keeps training/reload/test chunk8; diagnostic chunk changes only with immutable bitwise qualification.",
                            "Buffered counters are persisted lower bounds until graceful terminal flush; lost tails remain unknown.",
                            "Even graceful failed started/returned observations do not establish the side effects of an unreturned call.",
                            "No per-stage forward totals are synthesized from the changed diagnostic chunk."])
    return result


def collect():
    v4 = load_v4()
    result = augment(v4.collect())
    added, issues, seen_attempts = [], [], {}
    for path in sorted(v4.OUT.glob("execution_acceleration_v1/*/qualification_attempts/*/attempt.json")):
        try:
            raw = path.read_bytes()
            artifact = {"path": path.relative_to(WORK).as_posix(), "sha256": hashlib.sha256(raw).hexdigest()}
            value = json.loads(raw.decode("utf-8-sig"))
            if value.get("schema") != "diagnostic_batching_qualification_attempt_v1" or value.get("kind") != "engineering_qualification":
                raise ValueError("Unknown actual qualification attempt schema")
            if value.get("qualifier_source") != X.descriptor(SOURCE.with_name("qualify_acceleration_20261009.py")):
                raise ValueError("Qualification attempt source differs")
            if value.get("status") not in ("running", "complete", "failed") or value.get("unflushed_crash_tail_exact", "missing") is not None:
                raise ValueError("Qualification state or crash-tail disclosure differs")
            normalized = v4.normalized(value)
            peaks = [item.get("peak_allocated_bytes") for item in value.get("measurements", [])]
            failure_peak = value.get("peak_allocated_bytes_observed_at_failure")
            if failure_peak is not None: peaks.append(failure_peak)
            if any(type(n) is not int or n < 0 for n in peaks):
                raise ValueError("Qualification measured peak must be explicit allocated bytes")
            normalized["peak_allocated_bytes"] = max(peaks, default=None)
            record = {"artifact": artifact, "category": "diagnostic_batching_qualification",
                      "aggregation_role": "primary_observed_invocation", "condition": value.get("condition", path.parents[2].name),
                      "seed": value.get("seed"), "fold": value.get("fold"), "status": value["status"],
                      "attempt_id": artifact["path"], "started_at": value.get("started_at"), "finished_at": value.get("finished_at"),
                      "counter_scope": "Phase/repetition snapshots; hard-crash unflushed tail remains unknown",
                      "unflushed_crash_tail_exact": None, "nested_candidate_measurement_seconds_added": False,
                      "gpu_stage_entered": value.get("gpu_stage_entered"), "recorded_measurements": value.get("measurements", []),
                      "failure_peak_allocated_bytes_observed": failure_peak,
                      "peak_source_fields": (["measurements[*].peak_allocated_bytes"] if value.get("measurements") else [])
                                            + (["peak_allocated_bytes_observed_at_failure"] if failure_peak is not None else []),
                      "recorded_cost_fields": {k: v for k, v in value.items() if any(word in k for word in ("seconds", "forward", "windows", "updates", "backward", "forbidden"))},
                      "note": "Whole-attempt seconds once; candidate timings are nested evidence. This is zero-update engineering, never a formal fit.", **normalized}
            identity = (artifact["sha256"], value.get("started_at"))
            if identity in seen_attempts:
                record.update(aggregation_role="duplicate_invocation_artifact", alias_of=seen_attempts[identity])
            else:
                seen_attempts[identity] = artifact
            added.append(record)
            result["inputs"].append(artifact)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
            issues.append({"path": path.relative_to(WORK).as_posix(), "error": str(error)})
    result["cost_records"].extend(added)
    result["input_read_errors"].extend(issues)
    if added:
        primary = [row for row in added if row["aggregation_role"] == "primary_observed_invocation"]
        counters = {}
        for key in added[0]["observed_counters"]:
            known = [row["observed_counters"][key] for row in primary if row["observed_counters"][key] is not None]
            counters[key] = {"known_observed_subtotal": sum(known) if known else None, "records_with_value": len(known),
                             "records_unknown": len(primary) - len(known), "exact_category_total": None,
                             "known_observed_subtotal_excludes_unknown_buffered_tails": True}
        known_seconds = [row["seconds_observed"] for row in primary if row["seconds_observed"] is not None]
        result["category_observations"]["diagnostic_batching_qualification"] = {
            "primary_invocations": len(primary), "supplemental_records": len(added) - len(primary), "counter_totals": counters,
            "known_recorded_scope_seconds_subtotal": sum(known_seconds) if known_seconds else None,
            "seconds_unknown_records": sum(row["seconds_observed"] is None for row in primary), "full_category_wall_or_gpu_busy_time": None,
            "candidate_measurements_nested_not_added": True}
        result["max_observed_peak_allocated_bytes"] = max([n for n in [result["max_observed_peak_allocated_bytes"],
            *[row["peak_allocated_bytes"] for row in added]] if type(n) is int], default=None)
        result["unknown_or_opaque_records"].extend(row["artifact"]["path"] for row in added)
    result["qualification_cost_coverage"] = {"pattern": "execution_acceleration_v1/*/qualification_attempts/*/attempt.json",
                                           "actual_attempts_observed": len(added), "read_errors": issues,
                                           "failed_attempts_retained": True, "nested_candidate_times_added": False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = collect()
    target = (args.output or SOURCE.parent / ("actual_cost_v5_" + str(time.time_ns()) + ".json")).resolve()
    if not target.is_relative_to(SOURCE.parent):
        raise ValueError("Write only a new standalone report")
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"artifact": X.descriptor(target), "formal_status_counts": result["formal_status_counts"],
                      "buffered_observers": len(result["execution_observer_verification"]["observations"]),
                      "execution_errors": result["execution_observer_verification"]["errors"], "draft": True}))


if __name__ == "__main__": main()

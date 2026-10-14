"""Read-only version4 cost inventory; separate model, test and runtime setup evidence.

Keep per-fit durable training counters, observed invocation counters and opaque
legacy/recovery records separate. Missing fields remain null. This does not
certify full plan completion or manufacture a precise all-work total.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

WORK = Path(__file__).resolve().parents[3]
OUT = WORK / "tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008"
REPORT = Path(__file__).resolve().parent
ALIASES = {
    "optimizer_started": ("optimizer_step_started", "sgd_step_started"),
    "optimizer_returned": ("optimizer_step_returned", "sgd_step_returned", "optimizer_updates", "engineering_optimizer_updates", "engineering_optimizer_steps"),
    "backward_started": ("backward_started", "engineering_gradient_passes_started"),
    "backward_returned": ("backward_returned", "engineering_gradient_passes_completed", "backward_calls", "engineering_backward_calls", "engineering_gradient_passes", "gradient_passes"),
    "model_forward_started": ("outer_forward_started", "forward_started", "model_forward_started"),
    "outer_test_forward_started": ("outer_test_forward_started",),
    "model_forward_returned": ("outer_forward_returned", "forward_returned", "model_forward_returned"),
    "outer_test_forward_returned": ("outer_test_forward_returned",),
    "encoder_forward_started": ("encoder_forward_started", "encode_started", "started_batches"),
    "encoder_forward_returned": ("encoder_forward_returned", "encode_returned", "returned_batches", "gpu_encode_calls"),
    "encoder_windows_started": ("encoder_windows_started", "started_windows", "encoder_window_forwards_started"),
    "encoder_windows_returned": ("encoder_windows_returned", "returned_windows", "encoder_window_forwards_completed"),
    "prediction_people_started": ("prediction_people_started",),
    "prediction_people_returned": ("prediction_people_returned",),
    "ema_started": ("ema_updates_started",),
    "ema_returned": ("ema_updates_completed", "engineering_ema_updates", "ema_updates"),
}


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def counter(source, aliases):
    found = [(key, source[key]) for key in aliases if key in source]
    if not found:
        return None, [], None
    if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for _, value in found):
        return None, [key for key, _ in found], "invalid integer counter"
    if len({value for _, value in found}) != 1:
        return None, [key for key, _ in found], "conflicting aliases; do not add or choose silently"
    return found[0][1], [key for key, _ in found], None


def elapsed(source):
    for key in ("seconds", "seconds_this_attempt", "forward_backward_seconds"):
        if key in source and numeric(source[key]):
            scope = "producer-defined engineering timer; may include model construction, source checks, serialization and restore" if key == "forward_backward_seconds" else "recorded invocation scope"
            return float(source[key]), key, scope
    if source.get("started_at") and source.get("finished_at"):
        try:
            seconds = (datetime.fromisoformat(source["finished_at"]) - datetime.fromisoformat(source["started_at"])).total_seconds()
            if numeric(seconds):
                return seconds, "finished_at minus started_at", "ledger endpoint interval, not GPU busy time"
        except (ValueError, TypeError):
            pass
    return None, None, "unrecorded or still running; never zero-filled"


def normalized(source, *, invocation_aliases=True):
    flat = {**source.get("contract", {}), **source, **source.get("actual_engineering_cost", {})}
    fields, used, issues = {}, {}, []
    for name, aliases in ALIASES.items():
        if not invocation_aliases:
            aliases = tuple(key for key in aliases if key.endswith(("_started", "_returned")))
        fields[name], used[name], issue = counter(flat, aliases)
        if issue:
            issues.append({"field": name, "problem": issue})
    for name in ("optimizer", "backward", "model_forward", "outer_test_forward", "encoder_forward", "encoder_windows", "prediction_people", "ema"):
        started, returned = fields[name + "_started"], fields[name + "_returned"]
        fields[name + "_started_not_returned"] = started - returned if started is not None and returned is not None and started >= returned else None
        if started is not None and returned is not None and returned > started:
            issues.append({"field": name, "problem": "returned exceeds started"})
    seconds, seconds_field, scope = elapsed(flat)
    peak = max((flat[key] for key in ("peak_allocated_bytes", "peak_gpu_allocated_bytes") if key in flat and numeric(flat[key])), default=None)
    return {"observed_counters": fields, "counter_source_fields": used, "counter_issues": issues,
            "seconds_observed": seconds, "seconds_source_field": seconds_field, "seconds_scope": scope,
            "peak_allocated_bytes": peak, "peak_memory_units_unresolved": flat.get("peak_memory"), "unknown_counter_fields": [name for name, value in fields.items() if value is None]}


def rollback_lower_bound(failure, history):
    """Use recorded rows only; D0 after early resume can have stale locals."""
    epoch = failure.get("last_durable_epoch")
    total, _, issue = counter(failure, ("optimizer_updates", "updates_completed", "updates_total_process"))
    process = failure.get("updates_completed_this_process")
    if issue or type(epoch) is not int or epoch <= 0:
        return None, "D0/missing checkpoint boundary is opaque: resumed saved counters may precede validation and process baseline setup"
    row = next((row for row in history if isinstance(row, dict) and row.get("epoch") == epoch), None)
    durable, _, issue = counter(row or {}, ("optimizer_updates", "updates_completed"))
    if issue or total is None or durable is None or type(process) is not int or process < 0 or process > total or total < durable or total - process > durable:
        return None, "recorded global/process/history counters do not prove a consistent boundary"
    return total - durable, "acknowledged completed logical updates beyond recorded checkpoint row; not complete physical optimizer-return accounting"



def is_test_audit(data):
    """Recognize test/probe content; a CPU-looking filename proves no runtime."""
    if not isinstance(data, dict) or data.get("draft_not_completion_certificate"):
        return False
    return any(key in data for key in ("tests_run", "tests_passed", "test_module", "test_suite", "test_measurements")) or (
        any(key in data for key in ("data_source", "scope", "mode"))
        and any(key in data for key in ("synthetic_optimizer_updates", "engineering_optimizer_updates", "cuda_calls", "gpu_model_forwards")))


def test_runtime_category(data):
    if data.get("real_cuda_input_role") or data.get("real_cuda_inputs_used") is True:
        return "mixed_engineering_test_audit"
    gpu_keys = ("gpu_calls", "gpu_forwards", "gpu_model_forwards", "cuda_calls", "actual_process_actions_spawns_or_gpu")
    if data.get("gpu_used") is True or any(numeric(data.get(key)) and data[key] > 0 for key in gpu_keys):
        return "mixed_engineering_test_audit"
    if data.get("gpu_used") is False or any(type(data.get(key)) is int and data[key] == 0 for key in gpu_keys) or data.get("pure_standard_library") is True or data.get("pure_standard_library_simulation") is True:
        return "cpu_or_synthetic_test_audit"
    return "test_audit_runtime_unknown"


def supplemental_test_evidence(data):
    # Preserve current and prior/cumulative scopes without adding them together.
    return {key: value for key, value in data.items() if key not in ("test_output",)
            and (any(token in key for token in ("seconds", "cost", "updates", "backward", "forward", "gradient", "earlier", "prior", "cumulative", "test_measurements"))
                 or key in ("scope", "data_source", "mode", "gpu_used", "participant_inputs_used", "real_cuda_input_role", "real_cuda_inputs_used", "tests_run", "failures", "errors"))}


def runtime_setup_evidence(data):
    before, after = data.get("packages_before", {}), data.get("packages_after", {})
    return {"scope": data.get("scope"), "invocation_created_at_unix_ns": data.get("created_at_unix_ns"),
            "install_returncode": data.get("install", {}).get("returncode"),
            "pip_check_returncode": data.get("pip_check", {}).get("returncode"),
            "wheel": data.get("wheel"), "added_packages": data.get("added_packages"),
            "package_version_changes": {key: {"before": before.get(key), "after": after.get(key)} for key in before.keys() | after.keys() if before.get(key) != after.get(key)},
            "prior_package_versions_unchanged": data.get("prior_package_versions_unchanged"),
            "model_constructions": data.get("model_constructions"), "model_forwards": data.get("model_forwards"),
            "optimizer_updates": data.get("optimizer_updates"), "test_forwards": data.get("test_forwards"),
            "gpu_qualification": data.get("gpu_qualification"), "duration_seconds": elapsed(data)[0]}



def recovery_extra_cost(data, history=None, progress=None):
    if data.get("kind") != "user_authorized_resume_after_paused_process_exit":
        return None
    fields = ("durable_epoch", "durable_optimizer_updates", "durable_backward_calls", "paused_epoch", "paused_acknowledged_optimizer_updates", "paused_acknowledged_backward_calls", "acknowledged_updates_not_in_checkpoint", "acknowledged_backward_calls_not_in_checkpoint")
    valid = all(type(data.get(key)) is int and data[key] >= 0 for key in fields)
    valid = valid and data["paused_epoch"] >= data["durable_epoch"]
    valid = valid and data["paused_acknowledged_optimizer_updates"] - data["durable_optimizer_updates"] == data["acknowledged_updates_not_in_checkpoint"]
    valid = valid and data["paused_acknowledged_backward_calls"] - data["durable_backward_calls"] == data["acknowledged_backward_calls_not_in_checkpoint"]
    row = next((row for row in history or [] if isinstance(row, dict) and row.get("epoch") == data.get("durable_epoch")), {})
    durable_matches = row.get("optimizer_updates") == data.get("durable_optimizer_updates") and row.get("backward_calls") == data.get("durable_backward_calls") and bool(row)
    paused_matches = isinstance(progress, dict) and all(progress.get(a) == data.get(b) for a,b in (("epoch","paused_epoch"),("last_durable_epoch","durable_epoch"),("optimizer_updates_completed","paused_acknowledged_optimizer_updates"),("backward_calls_completed","paused_acknowledged_backward_calls")))
    verified = bool(valid and durable_matches and paused_matches)
    identity = {key:data.get(key) for key in ("condition","seed","fold","durable_epoch","paused_epoch")}
    identity["pause_record"] = data.get("pause_record")
    return {"boundary_arithmetic_consistent":bool(valid), "preserved_history_boundary_matches":bool(durable_matches), "preserved_progress_boundary_matches":bool(paused_matches),
            "boundary_verified":verified, "recovery_identity":identity, "reported_boundary_counters":{key:data.get(key) for key in fields},
            "acknowledged_optimizer_updates_outside_checkpoint":data["acknowledged_updates_not_in_checkpoint"] if verified else None,
            "acknowledged_backward_calls_outside_checkpoint":data["acknowledged_backward_calls_not_in_checkpoint"] if verified else None,
            "uncheckpointed_diagnostic_forward_count":data.get("uncheckpointed_diagnostic_forward_count"),
            "prior_partial_attempt_model_seconds":data.get("prior_partial_attempt_model_seconds"),
            "old_process_exit_code":data.get("old_process_exit_code"), "old_process_exit_time":data.get("old_process_exit_time"),
            "overnight_suspension_in_training_time":False, "source_cost_semantics":data.get("cost_semantics"),
            "aggregation_semantics":"Only acknowledged work beyond retained checkpoint; durable baseline is excluded. This is separate from fit durable counters and cannot be added to an overlapping physical attempt trace. It remains extra even after replay reaches the same logical epoch."}


def collect():
    cache, read_errors = {}, []

    def read(path):
        path = Path(path).resolve()
        key = path.relative_to(WORK).as_posix()
        if key not in cache:
            try:
                raw = path.read_bytes()
                cache[key] = (json.loads(raw.decode("utf-8-sig")), {"path": key, "sha256": hashlib.sha256(raw).hexdigest()})
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                read_errors.append({"path": key, "error": str(error)})
                cache[key] = (None, {"path": key, "sha256": None})
        return cache[key]

    registration, registration_descriptor = read(OUT / "registration.json")
    if registration is None:
        raise ValueError("The authoritative static registration is required")
    records, formal = [], []
    seen, primary_identities = set(), {}

    def add(path, category, *, role="primary_observed_invocation", view=None, identity=None, note=None):
        data, descriptor = read(path)
        if data is None:
            return
        identity_key = (descriptor["path"], category, identity or "")
        if identity_key in seen:
            return
        seen.add(identity_key)
        value = data if view is None else view
        if not isinstance(value, dict):
            return
        condition = value.get("condition")
        identity_basis = "record field"
        if condition is None:
            condition = next((item["condition_id"] for item in registration["configuration"]["conditions"]
                              if item["condition_id"] in path.parts), None)
            identity_basis = "registered condition directory" if condition else "shared/unknown scope"
        if condition is None and category.startswith("ssl_engineering"):
            condition, identity_basis = "EEG2Rep-fold-local-ssl", "dedicated registered SSL engineering settings/attempt layout"
        fit_id = condition + "/seed_" + str(value["seed"]) + "/fold_" + str(value["fold"]) if condition and value.get("seed") is not None and value.get("fold") is not None else None
        strong_identity = value.get("started_at") or value.get("time") or value.get("created_at_unix_ns") or (path.name if path.name.startswith("attempt_") else None)
        if role == "primary_observed_invocation" and strong_identity:
            view_bytes = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
            invocation = (category, hashlib.sha256(view_bytes).hexdigest(), strong_identity, identity if view is not None else None)
            if invocation in primary_identities:
                role = "duplicate_invocation_artifact"
                note = "Exact invocation bytes/identity also at " + primary_identities[invocation] + "; retained as provenance, counted only once."
            else:
                primary_identities[invocation] = descriptor["path"]
        records.append({"artifact": descriptor, "category": category, "aggregation_role": role,
                        "condition": condition, "seed": value.get("seed"), "fold": value.get("fold"), "fit_id": fit_id, "identity_basis": identity_basis,
                        "attempt_id": identity or descriptor["path"], "status": value.get("status", "complete" if value.get("passed") is True else "failed" if value.get("passed") is False else "unknown"),
                        "started_at": value.get("started_at"), "finished_at": value.get("finished_at"),
                        "note": note, "recorded_cost_fields": {key: item for key, item in value.items()
                            if any(token in key for token in ("cost", "seconds", "updates", "gradient_passes", "backward_calls", "peak_", "forward", "encode", "window", "step", "memory"))},
                        "counter_scope": "recorded invocation" if role == "primary_observed_invocation" else "supplemental; cumulative counters remain only in recorded_cost_fields/formal_durable table",
                        **normalized(value, invocation_aliases=role == "primary_observed_invocation")})

    for pattern, category in (
        ("finetune_v2_attempts/*/*/attempt.json", "finetune_engineering_or_formal"),
        ("**/export_attempts/*.json", "feature_export"),
        ("supervised_engineering/*/*.json", "supervised_engineering"),
        ("ssl/engineering_attempts/*.json", "ssl_engineering_failure"),
        ("engineering_attempts/EEGMamba/*.json", "mamba_gpu_engineering"),
        ("evaluation/*/seed_*/fold_*/*/attempts/*.json", "evaluation"),
    ):
        for path in sorted(OUT.glob(pattern)):
            data, _ = read(path)
            resolved_category = ("formal_training_attempt" if data and data.get("kind") == "formal" else "finetune_engineering") if category == "finetune_engineering_or_formal" else category
            add(path, resolved_category)
    for path in sorted((OUT / "assets/eegmamba_dependency_repair").glob("*/installation_*.json")):
        data, _ = read(path)
        if data:
            add(path, "runtime_environment_repair", note="Actual isolated environment install; no model cost is inferred and unrecorded duration remains null.")
            records[-1]["runtime_setup_evidence"] = runtime_setup_evidence(data)
    settings = OUT / "ssl/engineering_settings.json"
    if settings.exists():
        add(settings, "ssl_engineering_success", note="Producer timer begins before runtime/model construction and includes forward/backward and checkpoint roundtrip; not GPU busy time.")
    paired = OUT / "paired_replay/engineering_checks.json"
    if paired.exists():
        data, _ = read(paired)
        for index, item in enumerate(data.get("checks", []) if data else []):
            add(paired, "paired_engineering", view=item, identity="check_" + str(index), note="Each recorded check once; aggregate gradient_passes is a duplicate, not added.")
    for path in sorted((OUT / "engineering").glob("*.json")):
        data, _ = read(path)
        condition = data.get("condition") if data else None
        # Frozen head engineering has only this exact numeric contract; other
        # contract files mirror their dedicated primary attempts/settings.
        spec = next((c for c in registration["configuration"]["conditions"] if c["condition_id"] == condition), {})
        role = "primary_observed_invocation" if spec.get("policy_key") == "frozen" else "supplemental_contract_duplicate"
        add(path, "frozen_head_engineering" if role.startswith("primary") else "engineering_contract", role=role,
            view={**data.get("contract", {}), "condition": condition, "passed": True} if data else None,
            note="A checked boolean does not manufacture absent forward/time counters.")
    for fit in registration["fits"]:
        folder = OUT / "runs" / fit["fit_id"]
        status, status_desc = read(folder / "status.json") if (folder / "status.json").exists() else ({"status": "engineering_pending"}, None)
        meta, meta_desc = read(folder / "metadata.json") if (folder / "metadata.json").exists() else ({}, None)
        failures = sorted(folder.glob("failure_*.json"))
        history, history_desc = read(folder / "train_history.json") if (not meta or failures) and (folder / "train_history.json").exists() else ([], None)
        row = meta or (history[-1] if isinstance(history, list) and history else {})
        updates, used, issue = counter(row, ("optimizer_updates", "updates", "updates_completed"))
        backwards, back_used, back_issue = counter(row, ("backward_calls", "gradient_passes", "gradient_passes_completed"))
        formal.append({"fit_id": fit["fit_id"], "condition": fit["condition"], "seed": fit["seed"], "fold": fit["fold"], "objective": fit["objective"],
                       "status": status.get("status") if status else "unreadable", "status_artifact": status_desc,
                       "metadata_artifact": meta_desc, "history_artifact": history_desc,
                       "metadata_matches_bound_completion_sha": bool(meta_desc and status and status.get("status") == "training_complete" and status.get("evidence", {}).get("metadata") == meta_desc),
                       "durable_epoch": row.get("epochs", row.get("epoch")), "formal_durable_updates": updates,
                       "formal_durable_backward_calls": backwards, "counter_source_fields": used + back_used, "counter_issues": [x for x in (issue, back_issue) if x],
                       "best_epoch": row.get("best_epoch"), "stop_reason": row.get("stop_reason", row.get("stopping_reason")),
                       "last_attempt_seconds": row.get("seconds_this_attempt"), "peak_allocated_bytes": row.get("peak_allocated_bytes"),
                       "aggregation_role": "one_durable_counter_per_fit_not_added_to_physical_attempt_counters"})
        if meta:
            add(folder / "metadata.json", "formal_legacy_metadata", role="supplemental_durable_metadata",
                note="Cumulative durable counters and last-success time are not a full process-history ledger and must not be added to F2 trace.")
        for failure in failures:
            data, _ = read(failure)
            add(failure, "legacy_failure", role="supplemental_opaque_or_overlapping_failure",
                note="Process counter is not added to final fit counter or F2 invocation; resumed locals may be stale before baseline setup.")
            if data:
                extra, reason = rollback_lower_bound(data, history if isinstance(history, list) else [])
                records[-1].update(fit_id=fit["fit_id"], acknowledged_checkpoint_external_updates_lower_bound=extra, rollback_semantics=reason,
                                   raw_failure_counters={k: data[k] for k in ("last_durable_epoch", "optimizer_updates", "updates_completed", "updates_total_process", "updates_completed_this_process") if k in data})
    for pattern, category in (("resume_events/*.json", "recovery_event"), ("ready_recovery/*.json", "recovery_event"),
                              ("ready_recovery/*/status.json", "recovery_event"), ("progress_events/*.json", "progress_event"),
                              ("continuation_queue*/attempts/*/status.json", "dispatcher_elapsed")):
        for path in sorted(OUT.glob(pattern)):
            add(path, category, role="supplemental_nested_or_unmeasured", note="Nested queue elapsed/recovery statements are evidence; never added to model invocation elapsed or assumed zero work.")
            data, _ = read(path)
            if data and data.get("kind") == "user_authorized_resume_after_paused_process_exit":
                snapshots, snapshot_bindings = {}, []
                for item in data.get("preserved_artifacts", []):
                    saved = item.get("preserved_copy", {})
                    saved_path = (WORK / saved.get("path", "")).resolve()
                    if saved_path.name not in ("paused_fit_train_history.json", "paused_fit_progress.json"):
                        continue
                    if not saved_path.is_relative_to((OUT / "resume_events").resolve()):
                        raise ValueError("Recovery JSON snapshot is outside the declared resume-event directory")
                    snapshot, actual = read(saved_path)
                    matches = saved == actual and item.get("original_at_resume", {}).get("sha256") == actual["sha256"]
                    snapshot_bindings.append({"expected":saved,"actual":actual,"matches":matches})
                    if matches:
                        snapshots[saved_path.name] = snapshot
                extra = recovery_extra_cost(data, snapshots.get("paused_fit_train_history.json"), snapshots.get("paused_fit_progress.json"))
                extra["preserved_json_bindings"] = snapshot_bindings
                records[-1]["recovery_extra_cost"] = extra
                records[-1]["note"] = extra["aggregation_semantics"]
    mamba_report = REPORT / "eegmamba_official_gpu_check.json"
    if mamba_report.exists():
        add(mamba_report, "mamba_engineering_report", role="supplemental_check_report", note="Caller engineering ledger is primary; output may mirror one successful attempt, not all failures.")
    for path in sorted(REPORT.glob("*.json")):
        # File names only select likely diagnostic reports, never determine CPU/GPU.
        # Do not open training predictions, probability artifacts or prior aggregates.
        name = path.name.lower()
        if any(token in name for token in ("probabil", "prediction", "actual_cost", "cost_aggregation")) or not any(token in name for token in ("test", "cpu", "engineering", "adapter", "verification", "strict_load")):
            continue
        data, _ = read(path)
        if is_test_audit(data):
            add(path, test_runtime_category(data), role="supplemental_potentially_cumulative_test_report", note="Content-classified test evidence, possibly mixed CPU/GPU; no filename-based CPU-only assertion or additive cumulative test totals.")
            records[-1]["test_cost_evidence"] = supplemental_test_evidence(data)
    totals = {}
    categories = sorted({r["category"] for r in records})
    for category in categories:
        selected = [r for r in records if r["category"] == category and r["aggregation_role"] == "primary_observed_invocation"]
        counters = {}
        for field in next(iter(records))["observed_counters"] if records else []:
            values = [r["observed_counters"][field] for r in selected]
            known = [value for value in values if value is not None]
            counters[field] = {"known_observed_subtotal": sum(known) if known else None, "records_with_value": len(known),
                               "records_unknown": sum(v is None for v in values), "exact_category_total": sum(values) if values and all(v is not None for v in values) else None}
        totals[category] = {"primary_invocations": len(selected), "primary_status_counts": dict(Counter(r["status"] for r in selected)),
                            "supplemental_records": sum(r["category"] == category and r["aggregation_role"] != "primary_observed_invocation" for r in records),
                            "counter_totals": counters, "known_recorded_scope_seconds_subtotal": sum(r["seconds_observed"] for r in selected if r["seconds_observed"] is not None) if any(r["seconds_observed"] is not None for r in selected) else None,
                            "seconds_unknown_records": sum(r["seconds_observed"] is None for r in selected), "full_category_wall_or_gpu_busy_time": None}
    peaks = [r["peak_allocated_bytes"] for r in records if r["category"] not in ("cpu_or_synthetic_test_audit", "mixed_engineering_test_audit", "test_audit_runtime_unknown", "runtime_environment_repair")] + [r["peak_allocated_bytes"] for r in formal]
    recovery_work, recovery_seen = [], set()
    for record in records:
        extra = record.get("recovery_extra_cost")
        if not extra:
            continue
        key = json.dumps(extra["recovery_identity"],sort_keys=True)
        duplicate = key in recovery_seen
        if extra["boundary_verified"] and not duplicate:
            recovery_seen.add(key)
        recovery_work.append({"artifact":record["artifact"], "duplicate_checkpoint_boundary":duplicate, **extra})
    unique_recovery_work = [row for row in recovery_work if row["boundary_verified"] and not row["duplicate_checkpoint_boundary"]]
    statuses = Counter(row["status"] for row in formal)
    script_bytes = Path(__file__).read_bytes()
    return {"schema_version": 4, "artifact_kind": "read-only cost aggregation draft; known observations and unknown coverage", "draft_not_completion_certificate": True,
            "captured_at": datetime.now().astimezone().isoformat(), "source": {"path": Path(__file__).relative_to(WORK).as_posix(), "sha256": hashlib.sha256(script_bytes).hexdigest()},
            "registration": registration_descriptor, "formal_fits_expected": len(registration["fits"]), "formal_status_counts": dict(statuses),
            "formal_durable_updates_known_subtotal": sum(row["formal_durable_updates"] for row in formal if row["formal_durable_updates"] is not None) if any(row["formal_durable_updates"] is not None for row in formal) else None,
            "formal_durable_backward_calls_known_subtotal": sum(row["formal_durable_backward_calls"] for row in formal if row["formal_durable_backward_calls"] is not None) if any(row["formal_durable_backward_calls"] is not None for row in formal) else None,
            "formal_last_success_seconds_known_subtotal": sum(row["last_attempt_seconds"] for row in formal if numeric(row["last_attempt_seconds"])) if any(numeric(row["last_attempt_seconds"]) for row in formal) else None,
            "formal_last_success_seconds_missing_fits": [row["fit_id"] for row in formal if not numeric(row["last_attempt_seconds"])],
            "formal_last_success_seconds_scope": "One recorded last-success sub-scope per fit; not all previous calls or GPU busy time. Never add to overlapping F2 invocation seconds without invocation binding.",
            "formal_durable_counter_missing_fits": [row["fit_id"] for row in formal if row["formal_durable_updates"] is None],
            "formal_durable_counts_are_not_total_physical_attempt_cost": True, "formal_fits": formal, "cost_records": records, "category_observations": totals,
            "max_observed_peak_allocated_bytes": max((v for v in peaks if numeric(v)), default=None),
            "all_plan_physical_optimizer_returns_exact": None, "all_plan_wall_seconds_exact": None,
            "acknowledged_checkpoint_external_recovery_work":recovery_work,
            "recovery_acknowledged_optimizer_updates_known_subtotal":sum(row["acknowledged_optimizer_updates_outside_checkpoint"] for row in unique_recovery_work) if unique_recovery_work else None,
            "recovery_acknowledged_backward_calls_known_subtotal":sum(row["acknowledged_backward_calls_outside_checkpoint"] for row in unique_recovery_work) if unique_recovery_work else None,
            "recovery_extra_work_subtotal_scope":"Unique verified pause/checkpoint boundaries only. Separate extra physical work beyond durable fit history; do not add again to overlapping F2 traces. Unknown diagnostics and timing stay null; overnight suspension excluded.",
            "unknown_or_opaque_records": [r["artifact"]["path"] for r in records if r["unknown_counter_fields"] or r["aggregation_role"] != "primary_observed_invocation"],
            "input_read_errors": read_errors, "inputs": [descriptor for _, descriptor in cache.values()],
            "rules": ["Missing stays null; a known subtotal is not a complete total.", "Started and returned are separate; started-not-returned is not returned work.",
                      "Counter aliases, manifests, contracts and per-fit cumulative copies are not additive.", "Generic model_forward includes train/val/test model invocations; outer_test remains unknown unless explicitly recorded.", "Runtime setup/install records and mixed CPU/GPU tests remain separate supplemental costs; absent durations are unknown.", "Only explicit allocated-byte fields enter GPU peak; generic peak_memory units are unresolved.", "Formal durable counters, physical attempts, engineering, exports, failures, recovery and evaluation stay separate.",
                      "Legacy failures are not zero-filled, nor automatically added to formal/F2 updates; D0 resumed locals can be stale.", "Opaque and overlapping phases remain visible; nested dispatcher elapsed is not model wall time.",
                      "Peak GPU memory is a maximum, never a sum; duration sub-scopes are explicitly labeled.", "No raw/features/checkpoint/model/test probability was read and no GPU/process operation was performed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = collect()
    target = (args.output or REPORT / ("actual_cost_v4_" + str(time.time_ns()) + ".json")).resolve()
    if not target.is_relative_to(REPORT):
        raise ValueError("Write only a new standalone audit JSON in this analysis directory")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": target.relative_to(WORK).as_posix(), "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "formal_status_counts": value["formal_status_counts"], "observed_records": len(value["cost_records"]), "draft": True}))


if __name__ == "__main__":
    main()

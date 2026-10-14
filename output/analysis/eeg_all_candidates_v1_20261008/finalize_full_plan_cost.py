"""Finalize an independent cost companion only after quiescent full evaluation.

The pending path uses standard-library JSON/status reads only. The final path
calls E2's read-only completed_report_verified; it never invokes evaluation,
constructs a model, loads raw data, or starts a training/evaluation process.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

WORK = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
OUT = WORK / "tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008"
QUEUE = OUT / "continuation_queue_v3"
COST_SOURCE = REPORT / "cost_aggregation_v4.py"
COST_SHA256 = "46af6b28f80332f99b22f9373bb5902c0f7c42e9676c54a56da9c2f62eddbd40"
KNOWN_RESUME = "tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/resume_events/1791506856841705600.json"
KNOWN_RESUME_SHA256 = "1fea3e308156b2e9f3f86f2d28b9e4ffa8816657703bb956f2c3b014dadce0dd"
PHASE_PATTERNS = {
    "formal_metadata": "runs/*/seed_*/fold_*/metadata.json",
    "formal_failures": "runs/*/seed_*/fold_*/failure_*.json",
    "ft_attempts": "finetune_v2_attempts/*/*/attempt.json",
    "exports": "**/export_attempts/*.json",
    "supervised_engineering": "supervised_engineering/*/*.json",
    "ssl_failed_engineering": "ssl/engineering_attempts/*.json",
    "mamba_gpu_engineering": "engineering_attempts/EEGMamba/*.json",
    "evaluation": "evaluation/*/seed_*/fold_*/*/attempts/*.json",
    "environment_repair": "assets/eegmamba_dependency_repair/*/installation_*.json",
    "engineering_contracts": "engineering/*.json",
    "resume_events": "resume_events/*.json",
    "ready_recovery_events": "ready_recovery/*.json",
    "ready_recovery_statuses": "ready_recovery/*/status.json",
    "progress_events": "progress_events/*.json",
    "queue_attempt_statuses": "continuation_queue*/attempts/*/status.json",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def desc(path):
    path = Path(path).resolve()
    require(path.is_relative_to(WORK.resolve()), "Artifact outside this workspace")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": path.relative_to(WORK).as_posix(), "sha256": digest}


def checked(record):
    require(isinstance(record, dict) and set(record) == {"path", "sha256"}, "Expected a path/SHA256 descriptor")
    path = (WORK / record["path"]).resolve()
    require(path.is_file() and desc(path) == record, "Artifact is missing or changed: " + str(path))
    return path


def count(value):
    return type(value) is int and value >= 0


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def head_resource_gate(registration):
    """New measurements close head-resource evidence gaps, never old timings."""
    result = {"passed": True, "conditions_requiring_new_measurement": [], "verified_measurements": [], "pending": []}
    for condition in registration.get("configuration", {}).get("conditions", []):
        if condition["policy_key"] not in ("diagnostic", "frozen"):
            continue
        name = condition["condition_id"]
        lock_path = OUT / "engineering" / (name + ".json")
        if not lock_path.exists():
            result["pending"].append({"condition": name, "reason": "head engineering contract missing"})
            continue
        lock = read(lock_path)
        contract = lock["contract"]
        seconds = next((contract[k] for k in ("seconds", "forward_backward_seconds") if number(contract.get(k))), None)
        peak = next((contract[k] for k in ("peak_allocated_bytes", "peak_gpu_allocated_bytes") if number(contract.get(k))), None)
        if seconds is not None and peak is not None:
            continue
        result["conditions_requiring_new_measurement"].append(name)
        verified, issues = [], []
        for path in sorted((OUT / "supervised_engineering" / name).glob("measurement_*.json")):
            value = read(path)
            try:
                require(value.get("passed") is True and value.get("subtype") == "supplemental_head_resource_measurement"
                        and value.get("condition") == name and value.get("new_measurement_not_historical_backfill") is True,
                        "Not an actual successful new head resource measurement")
                require(value.get("registration") == desc(OUT / "registration.json")
                        and value.get("condition_source_lock") == desc(lock_path), "Resource measurement registration/contract binding differs")
                checked(value["source"])
                require(number(value.get("seconds")) and value["seconds"] > 0
                        and count(value.get("peak_allocated_bytes")) and value["peak_allocated_bytes"] > 0,
                        "Resource measurement actual seconds/peak missing")
                require(type(value.get("optimizer_updates")) is int and value["optimizer_updates"] == 0,
                        "Resource measurement must not update the optimizer")
                require(type(value.get("gradient_passes")) is int and value["gradient_passes"] == 1
                        and number(value.get("forward_backward_seconds")) and value["forward_backward_seconds"] > 0,
                        "Resource measurement must record one actual gradient pass and its timer")
                require(value.get("input_role") == "train" and type(value.get("seed")) is int and value["seed"] in (11, 22, 33)
                        and type(value.get("fold")) is int and value["fold"] in (1, 2, 3, 4, 5), "Resource measurement role/seed/fold invalid")
                ids = value.get("probe_subject_ids")
                train = registration["folds"][f"fold_{value['fold']}"]["train"]
                require(isinstance(ids, list) and ids and len(ids) == len(set(ids)) and set(ids).issubset(train),
                        "Resource measurement includes non-training people")
                verified.append({"artifact": desc(path), "condition": name, "source": value["source"],
                    "seconds": value["seconds"], "peak_allocated_bytes": value["peak_allocated_bytes"],
                    "optimizer_updates": 0, "gradient_passes": 1, "forward_backward_seconds": value["forward_backward_seconds"],
                    "input_role": "train", "probe_subject_ids": ids,
                    "old_missing_seconds_remains_unknown": seconds is None,
                    "old_missing_peak_remains_unknown": peak is None,
                    "new_measurement_not_historical_backfill": True})
            except (KeyError, ValueError, TypeError) as error:
                issues.append({"artifact": desc(path), "reason": str(error)})
        if verified:
            result["verified_measurements"].extend(verified)
        else:
            result["pending"].append({"condition": name, "reason": "Required new head seconds/peak measurement pending", "observed_invalid_records": issues})
    result["passed"] = not result["pending"]
    return result


def preflight():
    """No evaluator import or probability reads until all cheap gates pass."""
    value = {"status": "pending", "captured_at": now(), "final_snapshot": False,
             "draft_not_completion_certificate": True, "outer_prediction_artifacts_read": False,
             "completed_report_verifier_called": False, "reasons": [], "gate_artifacts": []}
    registration_path = OUT / "registration.json"
    if not registration_path.exists():
        value["reasons"].append("Immutable registration is absent")
        return value
    registration = read(registration_path)
    fits = registration.get("fits", [])
    counts = Counter(fit.get("objective") for fit in fits)
    conditions = Counter(fit.get("condition") for fit in fits)
    if len(fits) != 360 or len({f.get("fit_id") for f in fits}) != 360 or counts != {"supervised": 345, "self_supervised": 15} or len(conditions) != 24 or set(conditions.values()) != {15}:
        value["reasons"].append("Registration is not the fixed 360-fit/24-condition matrix")
        return value
    value["gate_artifacts"].append(desc(registration_path))
    statuses, status_descriptors = Counter(), []
    for fit in fits:
        path = OUT / "runs" / fit["fit_id"] / "status.json"
        status = read(path) if path.exists() else {}
        statuses[status.get("status", "engineering_pending")] += 1
        if path.exists():
            status_descriptors.append(desc(path))
    value["formal_status_counts"] = dict(statuses)
    if statuses != {"training_complete": 360}:
        value["reasons"].append("All 360 actual fit statuses must be training_complete")
        return value
    value["training_status_descriptors"] = status_descriptors
    queue_path = QUEUE / "status.json"
    if not queue_path.exists():
        value["reasons"].append("Final Q3 status is absent")
        return value
    queue = read(queue_path)
    if not (queue.get("status") == "all_training_complete" and queue.get("test_evaluation_completed") is True
            and queue.get("test_evaluation_status") == "complete" and queue.get("test_evaluation_frozen") is False):
        value["reasons"].append("Q3 has not recorded final completed test evaluation")
        return value
    if (QUEUE / "owner.json").exists():
        value["reasons"].append("Q3 owner has not been released")
        return value
    queue_registration = read(QUEUE / "registration.json")
    jobs = {job["id"] for job in queue_registration["jobs"]}
    if set(queue.get("outcomes", {})) != jobs or set(queue["outcomes"].values()) != {"complete"}:
        value["reasons"].append("Every registered Q3 job must have a complete outcome")
        return value
    value["gate_artifacts"].extend((desc(queue_path), desc(QUEUE / "registration.json")))
    resource_gate = head_resource_gate(registration)
    value["supplemental_head_resource_gate"] = resource_gate
    if not resource_gate["passed"]:
        value["reasons"].append("Actual supplementary head resource measurements remain pending")
        return value
    value["gate_artifacts"].extend(item["artifact"] for item in resource_gate["verified_measurements"])
    completion_path = OUT / "evaluation/complete.json"
    if not completion_path.exists():
        value["reasons"].append("Actual evaluation/complete.json is absent")
        return value
    completion = read(completion_path)
    required = {"passed": True, "formal_fits_verified": 360, "supervised_conditions": 23,
                "supervised_fits_evaluated": 345, "evaluated_checkpoints": 435, "candidate_papers_count": 9}
    if any(completion.get(k) != v or type(completion.get(k)) is not type(v) for k, v in required.items()):
        value["reasons"].append("Actual evaluation completion arithmetic is invalid")
        return value
    stable_gate = {"passed": True, "registration": desc(registration_path), "formal_fits_verified": 360,
                   "statuses": status_descriptors}
    canonical = hashlib.sha256(json.dumps(stable_gate, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    original_gate_path = OUT / "evaluation" / ("test_gate_" + canonical + ".json")
    if not original_gate_path.exists() or completion.get("test_gate") != desc(original_gate_path) or read(original_gate_path) != stable_gate:
        value["reasons"].append("Bound original test gate must already exist and match all 360 status hashes; no gate may be recreated here")
        return value
    value["gate_artifacts"].append(desc(original_gate_path))
    value["gate_artifacts"].append(desc(completion_path))
    value.update(status="ready_for_readonly_verification", reasons=[], queue_owner_released=True)
    return value


def verify_existing_evaluation():
    """Only the completed-report verifier is called; never E2.run/materialize."""
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))
    from tridep_ccml.eeg_all_candidates import evaluate_v2 as E2
    value = E2.completed_report_verified()
    require(value.get("passed") is True and value.get("formal_fits_verified") == 360
            and value.get("supervised_fits_evaluated") == 345 and value.get("supervised_conditions") == 23
            and value.get("model_constructions") == value.get("gpu_forward_calls") == 0,
            "Read-only E2 completion verification did not pass")
    return value


def collect_cost():
    require(desc(COST_SOURCE)["sha256"] == COST_SHA256, "Reviewed v4 cost collector changed")
    spec = importlib.util.spec_from_file_location("_reviewed_final_cost_v4", COST_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect(), module


def verify_training_cost(snapshot, registration, collector):
    require(snapshot["formal_fits_expected"] == 360 and snapshot["formal_status_counts"] == {"training_complete": 360}
            and len(snapshot["formal_fits"]) == 360 and not snapshot["input_read_errors"], "Cost snapshot training/read coverage incomplete")
    fit_rows = {row["fit_id"]: row for row in snapshot["formal_fits"]}
    require(len(fit_rows) == 360 and set(fit_rows) == {fit["fit_id"] for fit in registration["fits"]}, "Cost fit identity set differs")
    inputs = {item["path"]: item for item in snapshot["inputs"]}
    manifest_records, manifests = [], set()
    summaries = []
    configuration = registration["configuration"]["conditions"]
    for condition in configuration:
        name = condition["condition_id"]
        rows = [fit_rows[fit["fit_id"]] for fit in registration["fits"] if fit["condition"] == name]
        require(len(rows) == 15, "Condition does not have 15 cost rows")
        for row in rows:
            folder = OUT / "runs" / row["fit_id"]
            status = read(checked(row["status_artifact"]))
            evidence = status["evidence"]
            completion_path = folder / "complete.json"
            require(read(completion_path) == evidence, "Actual completion differs from bound status evidence")
            inputs[desc(completion_path)["path"]] = desc(completion_path)
            metadata_path = checked(evidence["metadata"])
            metadata = read(metadata_path)
            history_path = checked(evidence["train_history"])
            history = read(history_path)
            inputs[evidence["metadata"]["path"]] = evidence["metadata"]
            inputs[evidence["train_history"]["path"]] = evidence["train_history"]
            updates, _, issue = collector.counter(metadata, ("optimizer_updates", "updates", "updates_completed"))
            terminal, _, terminal_issue = collector.counter(history[-1], ("optimizer_updates", "updates_completed", "updates"))
            require(row["metadata_matches_bound_completion_sha"] is True and row["metadata_artifact"] == evidence["metadata"], "Cost metadata completion binding differs")
            require(not issue and not terminal_issue and count(updates) and updates > 0
                    and updates == terminal == evidence["optimizer_updates"] == row["formal_durable_updates"], "Actual durable update counters differ")
            require(count(metadata["epochs"]) and metadata["epochs"] > 0 and len(history) == metadata["epochs"]
                    and metadata["epochs"] == evidence["epochs"] == row["durable_epoch"], "Actual epoch counters differ")
            require(metadata["condition"] == name and metadata["seed"] == row["seed"] and metadata["fold"] == row["fold"], "Cost metadata identity differs")
            require(not row["counter_issues"], "Formal cost counter aliases conflict")
            manifest = metadata.get("feature_manifest")
            if manifest and manifest["path"] not in manifests:
                data = read(checked(manifest))
                manifests.add(manifest["path"])
                inputs[manifest["path"]] = manifest
                manifest_records.append({"artifact": manifest, "category": "feature_export_manifest",
                    "aggregation_role": "supplemental_export_or_inherited_cache_manifest", "condition": name,
                    "new_encoder_forwards": data.get("new_encoder_forwards"), "execution_attempts": data.get("execution_attempts"),
                    "seconds_this_attempt": data.get("seconds_this_attempt"), "peak_allocated_bytes": data.get("peak_allocated_bytes"),
                    "note": "Manifest counters/time mirror export attempts or describe an inherited cache; never added again.",
                    **collector.normalized(data, invocation_aliases=False)})
        durations = [row["last_attempt_seconds"] for row in rows if number(row["last_attempt_seconds"])]
        peaks = [row["peak_allocated_bytes"] for row in rows if number(row["peak_allocated_bytes"])]
        summaries.append({"condition": name, "policy": condition["policy_key"], "fits": 15,
            "durable_optimizer_updates": sum(row["formal_durable_updates"] for row in rows),
            "actual_epochs_by_fit": [{"seed": row["seed"], "fold": row["fold"], "epochs": row["durable_epoch"]} for row in rows],
            "actual_epochs_min": min(row["durable_epoch"] for row in rows), "actual_epochs_max": max(row["durable_epoch"] for row in rows),
            "last_success_subscope_seconds_known_sum": sum(durations) if durations else None,
            "last_success_subscope_seconds_unknown_fits": 15-len(durations),
            "seconds_scope": snapshot["formal_last_success_seconds_scope"],
            "peak_allocated_bytes_max": max(peaks) if peaks else None, "peak_unknown_fits": 15-len(peaks)})
    require(len(summaries) == 24 and snapshot["formal_durable_updates_known_subtotal"] == sum(r["durable_optimizer_updates"] for r in summaries), "24-condition durable total differs")
    snapshot["inputs"] = list(inputs.values())
    snapshot["cost_records"].extend(manifest_records)
    snapshot["condition_cost_summaries"] = summaries
    snapshot["feature_manifest_cost_evidence"] = manifest_records
    snapshot["category_observations"]["feature_export_manifest"] = {
        "primary_invocations": 0, "primary_status_counts": {}, "supplemental_records": len(manifest_records),
        "counter_totals": {}, "known_recorded_scope_seconds_subtotal": None,
        "seconds_unknown_records": 0, "full_category_wall_or_gpu_busy_time": None}
    peaks = [snapshot.get("max_observed_peak_allocated_bytes")] + [r["peak_allocated_bytes"] for r in manifest_records]
    snapshot["max_observed_peak_allocated_bytes"] = max((v for v in peaks if number(v)), default=None)


def verify_phase_coverage(snapshot, registration):
    records = snapshot["cost_records"]
    by_path = defaultdict(list)
    for row in records:
        by_path[row["artifact"]["path"]].append(row)
    inventory = []
    for phase, pattern in PHASE_PATTERNS.items():
        for path in sorted(OUT.glob(pattern)):
            descriptor = desc(path)
            require(descriptor["path"] in by_path and any(r["artifact"] == descriptor for r in by_path[descriptor["path"]]), "Unaccounted available cost artifact: " + descriptor["path"])
            inventory.append({"phase": phase, "artifact": descriptor, "categories": sorted({r["category"] for r in by_path[descriptor["path"]]})})
    for relative in ("paired_replay/engineering_checks.json", "ssl/engineering_settings.json"):
        descriptor = desc(OUT / relative)
        require(descriptor["path"] in by_path, "Required actual engineering phase missing")
        inventory.append({"phase": "required_engineering", "artifact": descriptor, "categories": sorted({r["category"] for r in by_path[descriptor["path"]]})})
    unknown_attempt_roots = [p.name for p in OUT.iterdir() if p.is_dir() and p.name.startswith("finetune_") and p.name.endswith("_attempts") and p.name != "finetune_v2_attempts"]
    require(not unknown_attempt_roots, "Unreviewed attempt families: " + str(unknown_attempt_roots))
    category_counts = Counter(row["category"] for row in records)
    required_categories = {"formal_legacy_metadata", "formal_training_attempt", "feature_export", "feature_export_manifest",
        "frozen_head_engineering", "paired_engineering", "finetune_engineering", "supervised_engineering", "ssl_engineering_success",
        "mamba_gpu_engineering", "evaluation", "runtime_environment_repair", "recovery_event",
        "cpu_or_synthetic_test_audit", "mixed_engineering_test_audit"}
    require(required_categories.issubset(category_counts), "Actual phase missing: " + str(required_categories - category_counts.keys()))
    required_ft = {fit["fit_id"] for fit in registration["fits"] if fit["condition"].endswith("-lastblock-finetune")}
    actual_ft = {row["fit_id"] for row in records if row["category"] == "formal_training_attempt"}
    require(len(required_ft) == 135 and required_ft.issubset(actual_ft), "Not all 135 actual FT attempt ledgers are accounted")
    required_ft_conditions = {fit["condition"] for fit in registration["fits"] if fit["fit_id"] in required_ft}
    require(required_ft_conditions.issubset({row["condition"] for row in records if row["category"] == "finetune_engineering"}), "An actual FT engineering condition is missing")
    recovery = snapshot["acknowledged_checkpoint_external_recovery_work"]
    known = [r for r in recovery if r["artifact"]["path"] == KNOWN_RESUME]
    require(len(known) == 1 and known[0]["artifact"]["sha256"] == KNOWN_RESUME_SHA256 and known[0]["boundary_verified"]
            and known[0]["acknowledged_optimizer_updates_outside_checkpoint"] == 4
            and known[0]["acknowledged_backward_calls_outside_checkpoint"] == 27
            and known[0]["overnight_suspension_in_training_time"] is False, "Known 4-update/27-backward recovery work not accounted")
    require(all(r["boundary_verified"] for r in recovery), "An actual recovery boundary remains unverified")
    completion = read(OUT / "evaluation/complete.json")
    evaluation_report = read(checked(completion["evaluation_report"]))
    evaluation_attempts = evaluation_report["execution_attempts"]
    require(evaluation_attempts and all(item["path"] in by_path and any(r["artifact"] == item and r["category"] == "evaluation" for r in by_path[item["path"]]) for item in evaluation_attempts), "A bound evaluation attempt was not accounted")
    for row in records:
        if row["category"] == "evaluation":
            data = read(checked(row["artifact"]))
            require(data.get("optimizer_updates") == data.get("ema_updates") == 0, "Evaluation recorded a training/EMA update")
    snapshot["available_artifact_coverage"] = {"scope": "All named actual cost artifact families in PHASE_PATTERNS plus required engineering records, bound feature manifests, and v4 content-classified test audits; not a claim to recover unrecorded historical work.",
        "patterns": PHASE_PATTERNS, "inventory": inventory, "category_record_counts": dict(category_counts),
        "bound_evaluation_attempt_descriptors": evaluation_attempts, "all_bound_evaluation_attempts_accounted": True,
        "all135_finetune_attempt_fit_ids_accounted": True, "known_recovery_boundary_accounted": True}
    return inventory


def disclose_gaps(snapshot):
    records = snapshot["cost_records"]
    unknown = [row["artifact"]["path"] for row in records if row.get("unknown_counter_fields") or not row.get("aggregation_role", "").startswith("primary")]
    snapshot["unknown_or_opaque_records"] = sorted(set(unknown))
    snapshot["coverage_gaps"] = [
        "旧训练 metadata 的累计逻辑计数不能恢复全部物理尝试；旧失败的 D0 恢复局部计数可能尚未设好基线。",
        "每次成功 metadata 的 seconds_this_attempt 只覆盖生产者记录的子范围；不能与 F2 整次尝试、调度器或父进程时间相加。",
        "CodeBrain 首次导出留下实际 2368 窗口/592 批记录但无结束时刻；缺失耗时保持 null。",
        "额外恢复工作只记封存 progress 超过 checkpoint 的部分；已知 4 次优化器更新和 27 次反向单列，诊断前向及旧退出/耗时未知；暂停过夜不计训练。",
        "模型前向涵盖 train/val/test 的实际调用；不能从通用 outer_forward 字段推出外测试调用量。",
        "CPU、混合 CUDA/CPU 测试和历史累计测试统计均保留原作用范围，不能与正式 GPU 训练相加。",
        "Ninja 安装完成证据保留；未记录安装耗时，保持 null。已存在的特征缓存不制造本轮新导出成本。",
        "峰值显存仅对显式字节字段取最大值；峰值不相加，未注明单位的 peak_memory 不转换。",
        "原 head 工程探针缺少耗时/峰值的条件，在队列结束后执行独立训练人员补测；新测量单列成本，旧未知记录继续保留，不能回填为旧探针资源。",
        "覆盖声明针对已逐项列出的现存成本证据族；未记录的历史工作不可逆推出精确总量。",
    ]
    snapshot["all_plan_physical_optimizer_returns_exact"] = None
    snapshot["all_plan_wall_seconds_exact"] = None
    snapshot["rules"] = [rule for rule in snapshot["rules"] if not rule.startswith("No raw/features/checkpoint/model/test probability")]
    snapshot["rules"].append("The v4 cost collector reads JSON ledgers only. After the final gate, E2 verifies existing predictions and CPU checkpoint headers and rehashes bound raw/features bytes; no signal arrays are loaded, no new predictions are computed, and no model construction, GPU call or process launch occurs.")


def show(value, digits=3):
    return "未知" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def render_report(snapshot):
    lines = ["# EEG 九候选全计划成本伴随报告", "", f"生成时间：{snapshot['captured_at']}（北京时间）。",
        "", "全 360 次拟合、435 个固定评价检查点、Q3 最终完成且 owner 释放后，已用 E2 的只读完成核验检查既存结果；核验包括原始/特征文件的字节哈希，不加载信号数组或重算预测。本报告独立附带，不修改已绑定的九候选统一报告。",
        "", f"正式持久逻辑更新已记录小计：{snapshot['formal_durable_updates_known_subtotal']}。这不是包含全部失败与重算的物理更新总量。全计划精确物理更新总量和总耗时均为未知（JSON 为 null）。",
        "", "时间列仅为每个拟合最后一次成功 metadata 所记录的子范围之和，不含所有历史尝试，也不代表 GPU 忙碌时间；不得再加 F2 尝试或调度器嵌套时间。",
        "", "| 条件 | 拟合 | 持久更新 | 实际轮数范围 | 已记录成功子范围秒数 | 耗时缺失拟合 | 峰值显存 bytes（最大） | 峰值缺失拟合 |",
        "|---|---:|---:|---|---:|---:|---:|---:|"]
    for row in snapshot["condition_cost_summaries"]:
        lines.append(f"| {row['condition']} | 15 | {row['durable_optimizer_updates']} | {row['actual_epochs_min']}–{row['actual_epochs_max']} | {show(row['last_success_subscope_seconds_known_sum'])} | {row['last_success_subscope_seconds_unknown_fits']} | {show(row['peak_allocated_bytes_max'])} | {row['peak_unknown_fits']} |")
    lines += ["", "## 各次拟合实际轮数", "", "顺序按 seed/fold 明示，完整元数据和原始成本记录由最终 JSON 绑定。", ""]
    for row in snapshot["condition_cost_summaries"]:
        values = "；".join(f"s{x['seed']}/f{x['fold']}={x['epochs']}" for x in row["actual_epochs_by_fit"])
        lines.append(f"- {row['condition']}：{values}")
    lines += ["", "## 工程、导出、失败恢复、环境、测试与评价", "", "以下阶段之间不相加。主尝试记录小计与旁证分别列出；缓存/manifest、失败与最终累计计数、父子进程时间可能重叠。", "",
        "| 阶段 | 主尝试记录 | 旁证/重复记录 | 已知记录范围秒数小计 | 缺失耗时主记录 |",
        "|---|---:|---:|---:|---:|"]
    for name, row in snapshot["category_observations"].items():
        lines.append(f"| {name} | {row['primary_invocations']} | {row['supplemental_records']} | {show(row['known_recorded_scope_seconds_subtotal'])} | {row['seconds_unknown_records']} |")
    lines += ["", "主尝试已知调用小计如下；括号为该字段未记录的主尝试数，故小计不一定是阶段完整总量。正式 F2 尝试与上方每拟合持久计数相互重叠，不能相加。", "",
        "| 阶段 | 模型前向返回 | 编码窗口返回 | 反向返回 | 优化器返回 |",
        "|---|---:|---:|---:|---:|"]
    for name, row in snapshot["category_observations"].items():
        if row["primary_invocations"]:
            cells = []
            for field in ("model_forward_returned", "encoder_windows_returned", "backward_returned", "optimizer_returned"):
                counter = row["counter_totals"].get(field, {})
                cells.append(show(counter.get("known_observed_subtotal")) + "（缺 " + str(counter.get("records_unknown", row["primary_invocations"])) + "）")
            lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += ["", f"恢复造成的 checkpoint 外已确认更新小计：{show(snapshot['recovery_acknowledged_optimizer_updates_known_subtotal'])}；反向调用小计：{show(snapshot['recovery_acknowledged_backward_calls_known_subtotal'])}。2026-10-09 已知恢复边界为 108−104=4 次优化器更新、729−702=27 次反向；已保存的 104/702 不重复加。之后重算完成不会抹去此前额外工作。",
        "", "通用模型前向不等于外测试前向。评价阶段执行的是固定 train/val/test 预测，所有评价尝试均核验为零优化器更新、零 EMA 更新。",
        "", "replay_engineering_test_results 同时含原 fold 训练人员的真实 CUDA probe 与 toy CPU 测试；原记录的每次完整测试 6 次 toy 更新及此前失败 5.354 秒分别保留，不推断未测量量，也不混入正式训练。",
        "", f"全局已观察显存峰值（显式 bytes 字段最大值）：{show(snapshot['max_observed_peak_allocated_bytes'])}。",
        "", "## 新增 head 资源补测", "", "以下是队列结束后独立执行的新测量；输入仅来自登记训练人员，优化器更新为 0。原探针缺失的资源值继续为未知，不用新值回填。", "",
        "| 条件 | 新测秒数 | 新测峰值 bytes | 优化器更新 | 旧秒数未知 | 旧峰值未知 |",
        "|---|---:|---:|---:|---|---|"]
    for row in snapshot.get("completion_gate", {}).get("supplemental_head_resource_gate", {}).get("verified_measurements", []):
        lines.append(f"| {row['condition']} | {show(row['seconds'])} | {row['peak_allocated_bytes']} | 0 | {row['old_missing_seconds_remains_unknown']} | {row['old_missing_peak_remains_unknown']} |")
    lines += ["", "## 缺口与计数边界", ""]
    lines.extend("- " + text for text in snapshot["coverage_gaps"])
    lines += ["", "成本 JSON 保留全部 360 个拟合、逐个来源哈希、完整主尝试与旁证、阶段覆盖清单、恢复边界和未知字段。所有缺失值均未填零。", ""]
    return "\n".join(lines)


def finalize():
    gate = preflight()
    if gate["status"] != "ready_for_readonly_verification":
        return gate
    verification = verify_existing_evaluation()
    snapshot, collector = collect_cost()
    registration = read(OUT / "registration.json")
    verify_training_cost(snapshot, registration, collector)
    inventory = verify_phase_coverage(snapshot, registration)
    disclose_gaps(snapshot)
    require(not snapshot["input_read_errors"], "Cost reads failed")
    after = preflight()
    require(after["status"] == "ready_for_readonly_verification"
            and after["gate_artifacts"] == gate["gate_artifacts"]
            and after["training_status_descriptors"] == gate["training_status_descriptors"], "Final gate changed during read-only cost verification")
    for item in snapshot["inputs"]:
        checked(item)
    for item in inventory:
        checked(item["artifact"])
    snapshot.update(schema_version=5, artifact_kind="Independent verified final full-plan cost snapshot",
        status="verified_final_cost", captured_at=now(), draft_not_completion_certificate=False, final_snapshot=True,
        all_available_artifacts_accounted=True, coverage_gaps_disclosed=True, completion_gate=gate,
        existing_evaluation_verification=verification, finalizer_source=desc(Path(__file__)),
        gpu_calls_by_finalizer=0, model_constructions_by_finalizer=0, optimizer_updates_by_finalizer=0,
        new_predictions_computed=0, existing_evaluation_outputs_modified=False)
    stamp = str(time.time_ns())
    human_path = REPORT / ("九候选全计划成本伴随报告_" + stamp + ".md")
    final_path = REPORT / ("final_full_plan_cost_" + stamp + ".json")
    require(not human_path.exists() and not final_path.exists(), "Final artifacts already exist")
    with human_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render_report(snapshot))
    snapshot["human_readable_cost_report"] = desc(human_path)
    with final_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(snapshot, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"status": "verified_final_cost", "final_snapshot": True, "final_cost": desc(final_path),
            "human_readable_cost_report": desc(human_path), "formal_fits_verified": 360,
            "unknown_historical_totals_remain_null": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check cheap gates only; never read predictions or create a final artifact")
    args = parser.parse_args()
    value = preflight() if args.check_only else finalize()
    if value["status"] != "verified_final_cost":
        path = REPORT / ("full_plan_cost_pending_" + str(time.time_ns()) + ".json")
        value["finalizer_source"] = desc(Path(__file__))
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": value["status"], "reasons": value["reasons"], "evidence": desc(path)}, ensure_ascii=True))
    else:
        print(json.dumps(value, ensure_ascii=True))


if __name__ == "__main__":
    main()

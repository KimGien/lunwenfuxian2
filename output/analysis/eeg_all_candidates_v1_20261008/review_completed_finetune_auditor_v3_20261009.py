"""Independent, source-bound review of the transient trace-read repair."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

REPORT = Path(__file__).resolve().parent
WORK = REPORT.parents[2]


def desc(path):
    path = Path(path)
    assert path.suffix in (".py", ".json")
    return {"path": path.relative_to(WORK).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verified(name, sha):
    value = desc(REPORT / name)
    assert value["sha256"] == sha, name
    return value


def main():
    bindings = {
        "auditor_v3": verified("audit_completed_finetune_checkpoints_v3.py", "307d6feabb9b4964db335fd9a0390e3026015cd57f6205117de125c79508c6d7"),
        "tests_v3": verified("test_audit_completed_finetune_checkpoints_v3.py", "2726e8136d7e41263b83bbc9f9ec1d94a78fbc2a1b294313947e38e6453e8b64"),
        "preserved_auditor_v2": verified("audit_completed_finetune_checkpoints_v2.py", "12bbb62d025bb2bbacfefa68677a9707185ae5943927b245b1e817a0124802bb"),
        "preserved_tests_v2": verified("test_audit_completed_finetune_checkpoints_v2.py", "7976f77e5ebb7c55cb455cc3b8b890c21cecc841417b13b4fe3508bd71979df8"),
        "preserved_failed_actual_v2_report": verified("completed_finetune_checkpoint_audit_v2_1791516059601364100.json", "b460ebc0f06730b4340463855d4636f3c9b8bf5ce4e822adfc04ffb7ad21ee0c"),
        "author_stdlib_test_report": verified("completed_ft_audit_v3_stdlib_tests_1791516483040181300.json", "5dfbb369b637a646126cca6e733d2a257a92fb3eeebe70be37aa9b5febb7a68b"),
        "independent_stdlib_test_report": verified("completed_ft_audit_v3_stdlib_tests_1791516543413544200.json", "826f2631fa5216242c0ca8dbedf7e8434bf8fb44f660baf38720d0c2110e89f0"),
    }
    for key in ("author_stdlib_test_report", "independent_stdlib_test_report"):
        data = json.loads((WORK / bindings[key]["path"]).read_text(encoding="utf-8"))
        assert data["source"] == bindings["auditor_v3"] and data["tests"] == bindings["tests_v3"]
        assert data["tests_run"] == 31 and data["errors"] == data["failures"] == 0 and data["passed"] is True
        assert data["actual_checkpoints_loaded"] == data["actual_models_constructed"] == data["actual_gpu_calls"] == 0
        assert data["regression_evidence"] == bindings["preserved_failed_actual_v2_report"]
    parsed = [ast.parse((WORK / bindings[key]["path"]).read_text(encoding="utf-8")) for key in ("preserved_auditor_v2", "auditor_v3")]
    functions = [{node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)} for tree in parsed]
    identical = ("require", "integer", "seconds", "delegated_sources", "load_wide", "finetune_matrix", "trace_boundary", "optimizer_group_boundary", "checkpoint_payload", "cpu_audit")
    assert all(ast.dump(functions[0][name]) == ast.dump(functions[1][name]) for name in identical)
    changed = [name for name in functions[0] if ast.dump(functions[0][name]) != ast.dump(functions[1][name])]
    assert set(changed) == {"trace_inventory", "completed_fit_audit", "audit", "main"}
    failed = json.loads((WORK / bindings["preserved_failed_actual_v2_report"]["path"]).read_text(encoding="utf-8"))
    assert failed["status"] == "failed" and failed["training_complete_finetune_observed"] == 8
    assert failed["verified_completed_finetune_fits"] == 7 and failed["checkpoint_loads"] == 21
    assert len(failed["errors"]) == 1 and failed["errors"][0]["error_type"] == "PermissionError"
    checks = [
        ("identity_before_binding", 111, 140, "One raw JSON snapshot establishes condition/kind/strict registered seed/fold. A valid unrelated identity is skipped without a second read or file hash."),
        ("bounded_transient_reads", 88, 108, "Initial permission/missing/UTF8/JSON failures get at most4 reads with .02/.04/.08-second waits. Unknown identity after exhaustion remains an error, never silently skipped."),
        ("mutable_target_deferred", 131, 147, "Matching running, inaccessible second read, differing bytes or changed final inventory yields TraceDeferred rather than accepting stale completion."),
        ("binding_not_weakened", 137, 140, "SHA is calculated from the same bytes that supplied identity and counters, only after exact equal-byte confirmation. Final inventory rechecks these SHA descriptors."),
        ("final_recheck_before_counting", 310, 335, "The final inventory passes the same bounded binding logic; duplicate un-retried trace S.checked is removed, while metadata/complete/lock/failure checks remain."),
        ("consumed_cpu_work_retained", 348, 385, "Post-CPU deferred fits retain actual load count in operations/deferred row but never enter verified rows or durable subtotals."),
        ("completion_gate_not_relaxed", 366, 381, "A full135 certificate requires verified rows and observed completed identities both equal135, and no pending/deferred/error. Full360 remains explicitly unclaimed."),
        ("new_output_only", 401, 410, "Reports use a new v3 prefix and exclusive creation; existing v2 source/reports stay unchanged."),
    ]
    result = {"kind": "independent_completed_finetune_auditor_v3_source_review", "review_passed": True,
              "created_at": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/resume_review",
              "review_source": desc(Path(__file__)), "bindings": bindings, "definite_blocking_findings": [],
              "checks": [{"id": key, "passed": True, "source": bindings["auditor_v3"], "line": start, "end_line": end, "observation": detail} for key, start, end, detail in checks],
              "independently_ast_identical_core_functions": list(identical), "changed_existing_functions": changed,
              "independently_repeated_stdlib_tests": 31, "test_failures": 0, "test_errors": 0,
              "preserved_failure_scope": {"observed_complete": 8, "verified": 7, "actual_cpu_loads": 21, "training_failure_claimed": False},
              "scope": "Source and stdlib regression review only. The v3 actual completed-fit CPU audit must be run separately by root; no result is inferred from synthetic tests.",
              "review_operations": {"actual_checkpoint_reads": 0, "models_constructed": 0, "gpu_calls": 0, "raw_arrays_read": 0, "outer_test_artifacts_read": 0, "sealed_sources_modified": 0},
              "future_actions": ["Root runs this bound v3 on genuine completed fits while preserving all older v2 reports.",
                                 "Use v3 in README/heartbeat instructions after actual verification; this review does not edit those targets.",
                                 "Persistent unidentified read errors remain visible; retry is bounded, not a guarantee that concurrent snapshots always succeed."],
              "full_135_actual_verification_claimed": False, "full_360_plan_completion_claimed": False, "final_semantic_companion": False}
    destination = REPORT / ("independent_completed_finetune_auditor_v3_review_" + str(time.time_ns()) + ".json")
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"artifact": desc(destination), "review_passed": True, "checks": len(checks), "independent_stdlib_tests": 31}, ensure_ascii=True))


if __name__ == "__main__": main()

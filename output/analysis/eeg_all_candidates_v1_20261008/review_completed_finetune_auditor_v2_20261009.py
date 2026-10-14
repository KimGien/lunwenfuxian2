"""Seal an independent source review; stdlib/JSON only, no training audit run."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

WORK = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
MOD = WORK / "tridep_ccml/eeg_all_candidates"


def descriptor(path):
    path = Path(path)
    return {"path": path.relative_to(WORK).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(path, digest):
    value = descriptor(path)
    assert value["sha256"] == digest, str(path)
    return value


def evidence(path, line, end, description):
    return {"source": descriptor(path), "line": line, "end_line": end,
            "description": description}


def main():
    auditor = REPORT / "audit_completed_finetune_checkpoints_v2.py"
    wide = REPORT / "audit_full_plan_acceptance_v3.py"
    tests = REPORT / "test_audit_completed_finetune_checkpoints_v2.py"
    f1, f2 = MOD / "finetune.py", MOD / "finetune_v2.py"
    sources = {
        "auditor": checked(auditor, "12bbb62d025bb2bbacfefa68677a9707185ae5943927b245b1e817a0124802bb"),
        "tests": checked(tests, "7976f77e5ebb7c55cb455cc3b8b890c21cecc841417b13b4fe3508bd71979df8"),
        "wide_cpu_delegate": checked(wide, "892e7a71265e8730c4deaf08f4a84579369ea639865ba5def1233b68fa26e64e"),
        "finetune_v1": checked(f1, "cda19474654233c95072acae475174d9011bb0739be650a597194f04c4b0aa24"),
        "finetune_v2": checked(f2, "fa53183ea4927daed5dcec0204e8a28ab09940f589742debda884402581c1a8b"),
    }
    pending = REPORT / "completed_finetune_checkpoint_audit_v2_1791509909435237700.json"
    pending_binding = checked(pending, "e9be99d8cc1376316571f5a76ad6703b5af3ffecb5c8c876a4fd3a8f8cdecf0c")
    observed = json.loads(pending.read_text(encoding="utf-8"))
    assert observed["auditor_source"] == sources["auditor"]
    assert observed["status"] == "pending" and observed["registered_finetune_fits"] == 135
    assert observed["training_complete_finetune_observed"] == observed["checkpoint_loads"] == 0
    assert len(observed["pending_fits"]) == 135 and not observed["errors"]
    test_bindings = []
    for filename, digest in (
        ("completed_ft_audit_v2_stdlib_tests_1791509897542249900.json", "d382262446bb923d893641ed6444ecb8be54f3e7505ee87b1f0ab6a7008b758f"),
        ("completed_ft_audit_v2_stdlib_tests_1791509976971282600.json", "735bddcda3c081b38186b64eedb21582b5c6609c5e1fbc48035eb89e245e320a"),
    ):
        path = REPORT / filename
        binding = checked(path, digest)
        value = json.loads(path.read_text(encoding="utf-8"))
        assert value["source"] == sources["auditor"] and value["tests"] == sources["tests"]
        assert value["passed"] is True and value["tests_run"] == 16
        assert value["errors"] == value["failures"] == value["actual_checkpoints_loaded"] == value["actual_gpu_calls"] == 0
        test_bindings.append(binding)
    checks = [
        ("exact_registered_scope", "Nine registered FT conditions by three seeds by five folds, with exact registered role IDs. Non-complete status cannot reach checkpoint inspection.",
         [(auditor, 62, 72), (auditor, 286, 302)]),
        ("wrapper_return_boundary", "A complete.json/status transition precedes F2 main return. A still-running formal attempt defers the identity before all checkpoint reads.",
         [(f1, 584, 592), (f2, 262, 277), (auditor, 250, 255)]),
        ("real_started_and_returned", "F2 persists each call start before invocation and each return after invocation. Strict integer counters and encode aliases are validated; unreturned calls are not completed calls.",
         [(f2, 41, 86), (auditor, 119, 135), (tests, 129, 136)]),
        ("retry_costs_separate_from_durable", "Unique content is counted once; failed and successful attempts retain physical known-return subtotals separately from the single durable trajectory. Physical calls may exceed durable updates after redo.",
         [(auditor, 90, 158), (tests, 112, 121), (tests, 138, 147)]),
        ("zero_step_complete_recheck", "F1 returns existing valid completion before constructing a model or updating. A new successful F2 trace may therefore have zero calls and must not be counted as another fit.",
         [(f1, 405, 415), (f2, 262, 277), (tests, 123, 127)]),
        ("unknown_resume_baselines_preserved", "No fabricated per-attempt durable baseline or exact physical total. Legacy failure observations are retained without addition because they can include resumed cumulative counters before start_updates is assigned.",
         [(f1, 450, 450), (f1, 453, 499), (f1, 593, 597), (auditor, 140, 158), (auditor, 264, 265)]),
        ("durable_history_and_scalar_boundaries", "Terminal metadata/history/update math and CPU latest scalar counts agree. Initial counters are integer zero. Selected best is bound to its own epoch and receives no invented terminal counters.",
         [(wide, 85, 139), (auditor, 97, 103), (auditor, 190, 208)]),
        ("adamw_head_backbone_identity", "Two ordered head/backbone groups, sequential serialized IDs, source-contract parameter order and moment shapes supplement delegate checks of learning rates, hyperparameters and actual per-parameter warmup step counts.",
         [(f1, 171, 175), (auditor, 161, 187), (wide, 234, 252)]),
        ("same_three_cpu_loads", "Wrapper observes only initial/latest/best loads delegated by wide, requires exact CPU kwargs, verifies identities and scalar boundaries, and does not load an additional checkpoint. GPU initialization and model construction are blocked during the delegate call.",
         [(wide, 213, 233), (auditor, 211, 247), (tests, 229, 258)]),
        ("terminal_updates_and_selected_disclosure", "CPU tensor checks require every intended terminal parameter to differ, all frozen tensors to remain equal, and actual selected/terminal encoder hashes and selected names to match disclosure. A head-warmup selected best is not misrepresented as a changed selected encoder.",
         [(wide, 188, 196), (wide, 269, 281), (MOD / "evaluate_v2.py", 54, 85)]),
        ("concurrent_read_stability", "Complete status, checkpoint bytes and formal attempt inventory must remain unchanged through inspection. Inputs/source descriptors are revalidated before success is returned.",
         [(auditor, 79, 86), (auditor, 243, 244), (auditor, 267, 271), (auditor, 305, 306)]),
        ("limited_report_and_failed_audit_accounting", "Only verified rows contribute durable totals; failed CPU read operations remain reported. Partial verification and full135 verification are explicitly distinct from full360 plan completion. No evaluation driver or outer-test gate is called.",
         [(auditor, 303, 325), (tests, 91, 103)]),
    ]
    report = {
        "kind": "independent_auxiliary_auditor_source_review", "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/resume_review",
        "review_builder": descriptor(Path(__file__)), "reviewed_sources": sources,
        "review_passed": True, "definite_blocking_findings": [],
        "scope": "Read-only source/JSON review and independent rerun of sixteen stdlib tests; no real FT checkpoint inspection in this review",
        "checks": [{"id": key, "passed": True, "observation": desc,
                    "evidence": [evidence(path, start, end, desc) for path, start, end in refs]}
                   for key, desc, refs in checks],
        "test_reports": test_bindings, "independent_test_rerun_report": test_bindings[1],
        "captured_real_pending_report": pending_binding,
        "captured_snapshot": {"created_at": observed["created_at"], "status": observed["status"],
                              "completed_fits": 0, "pending_fits": 135, "checkpoint_loads": 0,
                              "snapshot_not_claimed_current": True},
        "actual_review_operations": {"models_constructed": 0, "gpu_calls": 0,
                                    "tensor_checkpoints_loaded": 0, "raw_arrays_loaded": 0,
                                    "outer_test_probability_artifacts_read": 0, "training_sources_modified": 0},
        "future_actual_evidence_required": [
            "Run the bound auditor on genuine training_complete identities only after their F2 formal traces are terminal; pending snapshots are not actual tensor proof.",
            "The first genuine completed fit still needs successful CPU initial/latest/best inspection and exact trace/durable reconciliation; synthetic tests cannot validate actual tensor formats.",
            "All135 requires 135 verified completed rows, no pending/deferred/errors and three actual CPU loads per verified row; failed audit operations must remain visible.",
            "Full360 completion, fixed E2 evaluation, final cost accounting, supplemental head resource measurements and final semantic companion remain separate future requirements.",
            "If a historic attempt trace/baseline is missing or a call return was not observed, retain unknown exact totals; do not substitute projected counters or F1 failure fields.",
        ],
        "full_plan_completed": False, "all_135_finetune_checkpoints_verified_by_this_review": False,
        "final_semantic_companion": False, "historical_unknowns_backfilled": False,
    }
    path = REPORT / ("independent_completed_finetune_auditor_v2_review_" + str(time.time_ns()) + ".json")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"artifact": descriptor(path), "checks": len(checks), "review_passed": True,
                      "actual_tensor_checkpoints_loaded": 0}, ensure_ascii=True))


if __name__ == "__main__":
    main()

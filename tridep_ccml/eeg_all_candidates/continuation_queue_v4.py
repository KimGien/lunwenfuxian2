"""Single-GPU execution acceleration, reusing all original formal fit identities."""
from __future__ import annotations

import argparse
import ast
import importlib.util
from pathlib import Path
import time

from . import continuation_queue as Q1, continuation_queue_v3 as Q3, registry as R, storage as S

SOURCE = Path(__file__).resolve()
ROOT = S.OUT / "continuation_queue_v4"
NEW_SOURCES = ("continuation_queue_v4.py", "linux_entry_v3.py", "evaluate_v3.py",
               "acceleration_prepare_v1.py", "finetune_v3.py", "validation_batching_v1.py",
               "execution_acceleration_v1.py", "test_continuation_queue_v4.py")
ANALYSIS_SOURCES = ("qualify_acceleration_20261009.py", "safe_stop_acceleration_v1.py", "audit_completed_finetune_checkpoints_v4.py")


def acceleration_registration(condition):
    return S.OUT / "execution_acceleration_v1" / condition / "registration.json"


def jobs():
    result = []
    for original in Q3.jobs():
        job = {**original, "args": list(original["args"]), "dependencies": list(original["dependencies"])}
        if original.get("fits") and original["conditions"][0] in Q1.FINETUNE:
            condition = original["conditions"][0]
            qualification_id = condition + "-acceleration-prepare"
            linux = original.get("linux", False)
            qualification = {"id": qualification_id,
                             "module": "linux_entry_v3" if linux else "acceleration_prepare_v1",
                             "args": ["--action", "mamba-acceleration-prepare"] if linux else ["--condition", condition],
                             "conditions": [condition], "dependencies": list(original["dependencies"])}
            if linux:
                qualification["linux"] = True
            result.append(qualification)
            job["dependencies"] = [qualification_id]
            job["module"] = "linux_entry_v3" if linux else "finetune_v3"
        result.append(job)
    originals = {job["id"]: job for job in Q3.jobs()}
    fits = [job for job in result if job.get("fits")]
    if sum(job["fits"] for job in fits) != 195:
        raise ValueError("Exact original195-fit matrix required")
    for job in fits:
        old = originals[job["id"]]
        if any(job.get(key) != old.get(key) for key in ("id", "fits", "conditions", "args", "linux")):
            raise ValueError("Only execution routing and qualification dependencies may change")
    return result


def sources():
    previous = S.read(Q3.ROOT / "registration.json")
    for source in previous["sources"]:
        S.checked(source)
    added = ([S.desc(SOURCE.parent / name) for name in NEW_SOURCES]
             + [S.desc(S.REPORT / name) for name in ANALYSIS_SOURCES]
             + [S.desc(Q3.ROOT / "registration.json")])
    return list({record["path"]: record for record in previous["sources"] + added}.values())


def register():
    R.load_registration()
    path = ROOT / "registration.json"
    if not path.exists():
        S.save(path, {"version": 4, "registry": R.registration_descriptor(), "registered_at": S.now(),
                      "jobs": jobs(), "sources": sources(), "previous_execution_registration": S.desc(Q3.ROOT / "registration.json"),
                      "formal_fits_in_reused_matrix": 195, "one_gpu_sequential": True,
                      "training_protocol_matrix_or_hyperparameters_changed": False,
                      "training_micro_subjects": 1, "training_window_chunk": 8,
                      "only_changes": ["buffered physical-call observer", "individually qualified diagnostic window chunks"],
                      "test_gate_unchanged": True, "requires_safe_handoff": True}, immutable=True)
    value = S.read(path)
    if value["registry"] != R.registration_descriptor() or value["jobs"] != jobs():
        raise ValueError("Q4 execution registration differs")
    for source in value["sources"]:
        S.checked(source)
    return value


def corrected_run_child(driver):
    tree = ast.parse(Path(Q1.__file__).read_text(encoding="utf-8"))
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_child")
    changed = 0
    for node in ast.walk(original):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "linux_owner":
            if ast.dump(node.value, include_attributes=False) != ast.dump(ast.parse("folder / 'linux_owner.json'", mode="eval").body, include_attributes=False):
                raise ValueError("Original owner routing differs")
            node.value = ast.parse("(S.OUT / 'continuation_queue/attempts' / ('v4_' + folder.name) / 'linux_owner.json') if job.get('linux') else (folder / 'linux_owner.json')", mode="eval").body
            changed += 1
    if changed != 1:
        raise ValueError("Expected exactly one owner path assignment")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])), str(SOURCE), "exec"), driver.__dict__)


def isolated_driver():
    driver = Q3.isolated_driver()
    driver.ROOT = ROOT
    driver.register = register
    driver.jobs = jobs
    driver.sources = sources
    corrected_run_child(driver)
    run_child = driver.run_child
    def reused_or_run(job, owner, *, resume=False):
        reusable = bool(job.get("fits")) or job["id"].endswith("-acceleration-prepare")
        if reusable and job.get("conditions") and Q1.all_complete(job["conditions"]):
            # Completed fits retain their original real execution provenance;
            # qualification is only needed for work that will actually run.
            S.save(ROOT / "reused_jobs" / (job["id"] + ".json"),
                   {"job": job, "status": "verified_completed_condition_reused", "verified_at": S.now(),
                    "q4_registration": S.desc(ROOT / "registration.json"),
                    "new_model_calls": 0, "new_optimizer_updates": 0,
                    "qualification_for_future_work_skipped_because_all_fits_complete": job["id"].endswith("-acceleration-prepare")})
            return "complete"
        return run_child(job, owner, resume=resume)
    driver.run_child = reused_or_run
    # Use original command construction, then provide immutable per-condition execution bindings.
    def command(job, *, resume=False):
        mapping = {"linux_entry": "linux_entry_v3", "evaluate": "evaluate_v3"}
        routed = {**job, "module": mapping.get(job["module"], job["module"]), "args": list(job["args"])}
        if job.get("fits") and job["conditions"][0] in Q1.FINETUNE:
            descriptor = S.desc(acceleration_registration(job["conditions"][0]))
            routed["args"].extend(["--acceleration-registration", descriptor["path"],
                                   "--acceleration-registration-sha256", descriptor["sha256"]])
        return Q1.command(routed, resume=resume)
    driver.command = command
    original_postcondition = driver.postcondition
    def postcondition(job):
        if job["id"].endswith("-acceleration-prepare"):
            from . import execution_acceleration_v1 as A
            condition = job["conditions"][0]
            A.load_execution_acceleration(S.desc(acceleration_registration(condition)), condition=condition, seed=11, fold=1)
            return True
        if job["id"] in ("evaluate-mamba-finetune", "evaluate-all"):
            from . import evaluate_v3 as E
            if job["id"] == "evaluate-mamba-finetune":
                value = E.completed_condition_verified("EEGMamba-lastblock-finetune")
                return value.get("passed") is True and value.get("supervised_fits_evaluated") == 15 and value.get("primary_oof_predictions") == 111
            value = E.completed_report_verified()
            return value.get("passed") is True and value.get("formal_fits_verified") == 360 and value.get("supervised_conditions") == 23 and value.get("supervised_fits_evaluated") == 345
        return original_postcondition(job)
    driver.postcondition = postcondition
    evaluate_tail = driver.evaluate_tail
    def cpu_verified_evaluation_tail(owner):
        if Q1.all_complete(tuple(c.condition_id for c in Q1.P.CONDITIONS)):
            path = S.REPORT / "audit_completed_finetune_checkpoints_v4.py"
            spec = importlib.util.spec_from_file_location("q4_pre_test_ft_cpu_audit", path)
            auditor = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(auditor)
            result = auditor.audit()
            target = ROOT / ("pre_test_finetune_cpu_audit_" + str(time.time_ns()) + ".json")
            S.save(target, result, immutable=True)
            if not (result["all_135_finetune_checkpoints_verified"] is True and not result["errors"]
                    and not result["deferred_completed_fits"] and result["checkpoint_loads"] == 405):
                raise ValueError("All actual135 FT CPU checkpoints must verify before outer-test inference")
            S.save(ROOT / "pre_test_finetune_cpu_gate.json", {"passed": True, "audit": S.desc(target),
                                                            "checkpoint_loads": 405, "test_forward_calls_this_check": 0})
        return evaluate_tail(owner)
    driver.evaluate_tail = cpu_verified_evaluation_tail
    return driver


def require_handoff():
    proof = S.read(ROOT / "handoff_complete.json")
    if not (proof.get("passed") is True and proof.get("all_training_processes_stopped") is True
            and proof.get("gpu_idle") is True and proof.get("q4_registration") == S.desc(ROOT / "registration.json")):
        raise ValueError("Exact checkpoint safe-stop proof required")
    binding = proof.get("safe_stop")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("Actual source-bound safe-stop artifact is required")
    safe = S.read(S.checked(binding))
    expected_protected = S.read(Q3.ROOT / "registration.json")["sources"] + [S.desc(Q3.ROOT / "registration.json")]
    if not (safe.get("schema") == "acceleration_safe_stop_v1" and safe.get("mode") == "stopped_training"
            and safe.get("source") == S.desc(S.REPORT / "safe_stop_acceleration_v1.py")
            and safe.get("passed") is True and safe.get("all_training_processes_stopped") is True
            and safe.get("gpu_idle") is True and safe.get("resume_full_state_verified") is True
            and safe.get("checkpoint_load_device") == "cpu" and safe.get("model_operations") == 0
            and safe.get("protected_sources") == expected_protected
            and proof.get("protected_sources") == safe["protected_sources"]
            and proof.get("checkpoint_evidence") == safe.get("checkpoint_evidence")):
        raise ValueError("Handoff assertions differ from actual reviewed safe-stop proof")
    identities = safe.get("held_process_identities", {})
    if (set(identities) != {"dispatcher", "launcher", "gpu_child"}
            or safe.get("stopped_process_ids") != [identities[role]["pid"] for role in ("dispatcher", "launcher", "gpu_child")]
            or not safe.get("checkpoint_evidence") or safe.get("checkpoint") not in safe["checkpoint_evidence"]):
        raise ValueError("Exact stopped process objects and immutable durable checkpoint are required")
    for descriptor in proof["protected_sources"] + proof["checkpoint_evidence"]:
        S.checked(descriptor)
    if (Q3.ROOT / "owner.json").exists():
        raise ValueError("Old Q3 owner has not been safely archived")
    return proof


def run(*, resume=False):
    register()
    require_handoff()
    isolated_driver().run(resume=resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    register() if args.register_only else run(resume=args.resume)

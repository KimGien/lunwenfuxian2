"""Per-architecture zero-update diagnostic qualification before buffered FT."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import time

from . import storage as S, registry as R
from . import execution_acceleration_v1 as A

SOURCE = Path(__file__).resolve()
ROOT = S.OUT / "execution_acceleration_v1"
QUALIFIER = S.REPORT / "qualify_acceleration_20261009.py"


def load_qualifier():
    spec = importlib.util.spec_from_file_location("eeg_diagnostic_qualification_runtime", QUALIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def registration_path(condition):
    return ROOT / condition / "registration.json"


def create_registration(condition, qualification_result):
    path = registration_path(condition)
    if path.exists():
        descriptor = S.desc(path)
        A.load_execution_acceleration(descriptor, condition=condition, seed=11, fold=1)
        return descriptor
    passed = qualification_result.get("passed") is True
    descriptor = qualification_result["artifact"]
    report = S.read(S.checked(descriptor))
    failed_oom = (report.get("schema") == "diagnostic_batching_qualification_attempt_v1"
                  and report.get("status") == "failed" and report.get("gpu_stage_entered") is True
                  and "cuda" in report.get("error", "").lower() and "out of memory" in report.get("error", "").lower())
    if report.get("condition") != condition or (report.get("passed") is not passed and not (not passed and failed_oom)):
        raise ValueError("Actual qualification result identity differs")
    chunk = report["selected_chunk"] if passed else 8
    entry = {"condition_source_lock": R._condition_lock(condition), "diagnostic_window_chunk": chunk,
             "qualification": descriptor if passed else None,
             "tested_qualification": descriptor,
             "batching_status": "qualified" if passed else "retained_original8_candidate_oom" if failed_oom else "retained_original8_no_bitwise_speedup"}
    sources = {name: S.desc(SOURCE.parent / (name + ".py")) for name in
               ("finetune_v3", "finetune_v2", "finetune", "validation_batching_v1", "execution_acceleration_v1")}
    S.save(path, {"schema": "eeg_finetune_execution_acceleration_v1", "version": 1, "status": "qualified",
                  "registry": R.registration_descriptor(), "registered_at": S.now(),
                  "previous_execution_registration": S.desc(S.OUT / "continuation_queue_v3/registration.json"),
                  "single_gpu": True, "training_math_unchanged": True,
                  "training_micro_subjects": 1, "training_window_chunk": 8,
                  "sources": sources, "conditions": {condition: entry},
                  "preparation_source": S.desc(SOURCE), "qualifier_source": S.desc(QUALIFIER),
                  "logging_status": "buffered_call_observer_qualified",
                  "qualification_failure_is_preserved_not_counted_as_formal_fit": not passed}, immutable=True)
    binding = S.desc(path)
    A.load_execution_acceleration(binding, condition=condition, seed=11, fold=1)
    return binding


def queue_exclusive_proof(condition, *, linux_owner=None):
    owner_path = S.OUT / "continuation_queue_v4/owner.json"
    owner = S.read(owner_path)
    expected_job = condition + "-acceleration-prepare"
    if owner.get("child_job") != expected_job:
        raise ValueError("Queue is not preparing this architecture")
    if os.name == "nt" and owner.get("child_pid") not in (os.getpid(), os.getppid()):
        raise ValueError("This qualifier is not the actual queue child")
    process_list = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=20, check=True).stdout.strip()
    if process_list:
        raise RuntimeError("Existing GPU compute owner; qualification refused")
    value = {"schema": "acceleration_safe_stop_v1", "mode": "queue_exclusive_post_prepare",
             "passed": True, "source": S.desc(SOURCE), "condition": condition, "seed": 11, "fold": 1,
             "registration": R.registration_descriptor(), "gpu_idle": True, "stopped_process_ids": [],
             "queue_owner": S.desc(owner_path), "queue_child_job": expected_job,
             "checkpoints": [], "created_at": S.now(), "zero_optimizer_qualification_only": True}
    if linux_owner is not None:
        value["linux_owner"] = linux_owner
        value["windows_queue_owner"] = value["queue_owner"]
    path = ROOT / condition / ("queue_exclusive_proof_" + str(time.time_ns()) + ".json")
    S.save(path, value, immutable=True)
    return S.desc(path)


def run(condition, *, stop_proof=None, linux_owner=None):
    path = registration_path(condition)
    if path.exists():
        value = S.desc(path)
        A.load_execution_acceleration(value, condition=condition, seed=11, fold=1)
        return value
    R._condition_lock(condition)
    qualifier = load_qualifier()
    proof = stop_proof or queue_exclusive_proof(condition, linux_owner=linux_owner)
    value = S.read(S.checked(proof))
    attempts = ROOT / condition / "qualification_attempts"
    previous = set(attempts.glob("*/attempt.json"))
    try:
        result = qualifier.run_qualification(condition, value["seed"], value["fold"], safe_stop_proof=proof)
    except RuntimeError as error:
        # Optional expanded diagnostic chunks may exceed memory even though
        # the unchanged F2 training has already passed its real engineering.
        # Keep that failed attempt and enable only independently verified logging.
        if "cuda" not in str(error).lower() or "out of memory" not in str(error).lower():
            raise
        created = set(attempts.glob("*/attempt.json")) - previous
        if len(created) != 1:
            raise
        attempt_path = created.pop()
        observed = S.read(attempt_path)
        if not (observed.get("status") == "failed" and observed.get("gpu_stage_entered") is True
                and observed.get("condition") == condition and observed.get("optimizer_updates") == 0
                and observed.get("backward_calls") == 0 and observed.get("forbidden_optimizer_attempts") == 0
                and observed.get("forbidden_backward_attempts") == 0
                and observed.get("qualifier_source") == S.desc(QUALIFIER)):
            raise
        result = {"artifact": S.desc(attempt_path), "passed": False, "selected_chunk": None}
    return create_registration(condition, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    args = parser.parse_args()
    print(run(args.condition), flush=True)

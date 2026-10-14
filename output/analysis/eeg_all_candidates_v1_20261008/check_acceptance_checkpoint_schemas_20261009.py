"""Check actual completed training schemas on CPU; no evaluation/gate imports."""
from pathlib import Path
import importlib.util
import json
import sys
import time

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import registry as R, storage as S


def main():
    source = HERE / "audit_full_plan_acceptance_v3.py"
    expected = "892e7a71265e8730c4deaf08f4a84579369ea639865ba5def1233b68fa26e64e"
    assert S.desc(source)["sha256"] == expected
    spec = importlib.util.spec_from_file_location("wide_schema_preflight", source)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    import torch
    torch.set_num_threads(2)
    registration = R.load_registration()
    conditions = (
        "EEGPT-mean-projection4s-originalCNN-300ep", "EEGPT-frozen-smallhead",
        "EEGConformer-scratch", "EEG2Rep-fold-local-ssl", "EEG2Rep-supervised",
    )
    results = []
    for condition in conditions:
        fit = next(fit for fit in registration["fits"]
                   if fit["condition"] == condition and fit["seed"] == 11 and fit["fold"] == 1)
        try:
            status = R.fit_status(condition, 11, 1)
            assert status["status"] == "training_complete"
            lock = R._condition_lock(condition)
            contract = S.read(S.checked(lock))["contract"]
            row = audit.check_fit(fit, status, registration, lock, contract)
            metadata = S.read(S.checked(row["metadata"]))
            proof = audit.deep_checkpoint(fit, row, contract, metadata)
            results.append({"fit_id": fit["fit_id"], "passed": True,
                            "checkpoint_proof": proof, "metadata": row["metadata"]})
        except Exception as error:
            results.append({"fit_id": fit["fit_id"], "passed": False,
                            "error_type": type(error).__name__, "error": str(error)})
    value = {"source": S.desc(Path(__file__)), "auditor": S.desc(source),
             "captured_at": S.now(), "registration": R.registration_descriptor(),
             "scope": "One already completed seed11/fold1 actual training checkpoint per available policy; FT pending",
             "model_constructions": 0, "gpu_operations": 0, "evaluation_gate_calls": 0,
             "test_probability_reads": 0, "signal_or_feature_array_loads": 0,
             "full_plan_completed": False, "results": results,
             "passed": all(row["passed"] for row in results)}
    path = HERE / ("acceptance_checkpoint_schema_preflight_" + str(time.time_ns()) + ".json")
    S.save(path, value, immutable=True)
    print(json.dumps({"artifact": S.desc(path), "passed": value["passed"], "results": results}, ensure_ascii=False))
    if not value["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

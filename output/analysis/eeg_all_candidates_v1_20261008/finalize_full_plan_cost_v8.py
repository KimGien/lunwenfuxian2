"""Finalize v8 mixed Q13/Q14/E12 independent cost companion only after quiescent full evaluation.

Before full training/Q14/CPU17/E12 completion, the pending path reads JSON/status only.
Later resource gates may hash bound feature/checkpoint bytes without loading
arrays or tensors; no pending path reads predictions. The final path
calls E12's read-only completed_report_verified; it never invokes evaluation,
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
QUEUE = OUT / "continuation_queue_v5"
COST_SOURCE = REPORT / "cost_aggregation_v18.py"
RESOURCE_RUNNER = REPORT / "supplemental_head_resource_measurement_v8.py"
RESOURCE_SHA256 = "5df2ed56381a5179caac5592fcd1d4b044a913de8ce55b49839c967e717f8cea"
COST_SHA256 = "8f8d37c3fae0e408c36ddeb85643f4c2d37334bf973e0d5da80e62124e62ccd0"
PREVIOUS_FINALIZER = REPORT / "finalize_full_plan_cost_v7.py"
PREVIOUS_FINALIZER_SHA256 = "ab5d4d892f3d6a0e75c2c82db5bd3bbcfdffd086b4be4f92858234612982a712"
RESOURCE_DELEGATE = REPORT / "supplemental_head_resource_measurement_v2.py"
RESOURCE_DELEGATE_SHA256 = "351f67b6941b12cfb8139372783d92158c5040d7ad20abf4a8b2d435e5a9a54b"
CPU_AUDITOR = REPORT / "audit_completed_finetune_checkpoints_v17.py"
Q4_REGISTRATION_SHA256 = "c831bca3afad5574a7bc084fa4a0ebcaec7bea629679d7c1b426171092f8b089"
Q5_SOURCE = WORK / "tridep_ccml/eeg_all_candidates/continuation_queue_v5.py"
Q5_SOURCE_SHA256 = "e9a281d44b2e5361215ad178bc6eff96e46327c56cf3b9c2c7050081c9356971"
CPU_AUDITOR_SHA256 = "3845dcf2e23a924700a519d6f687213604b4456b836d9f9b9d7262ddcb406eb8"
QUALIFIER_V2 = REPORT / "qualify_acceleration_v2.py"
QUALIFIER_V2_SHA256 = "3f63b8eb63c33782a972494af0278e8924fa0b866511c7b217f16a2d6e4416e2"
QUALIFIER_V1 = REPORT / "qualify_acceleration_20261009.py"
QUALIFIER_V1_SHA256 = "878fda0f97d0a6ec801501baf1bad084d77d2d319897609a2a39600e01568a5e"
EXPECTED_RESOURCE_MEASUREMENTS = 12
KNOWN_HANDOFF = "tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v4/handoff/safe_stop_1791547625592196000.json"
KNOWN_HANDOFF_SHA256 = "3fc46e1aed14bf418b1ed940a32db4c878ce14bfbabec9cdb48b5f735c9bb406"
KNOWN_RESUME = "tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/resume_events/1791506856841705600.json"
KNOWN_RESUME_SHA256 = "1fea3e308156b2e9f3f86f2d28b9e4ffa8816657703bb956f2c3b014dadce0dd"
KNOWN_MAMBA_RESTORE = {"path": "tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/progress_events/mamba_recheck_report_restore_1791548331506439200.json", "sha256": "185ceeae5ac12805309680d252dd5e21660ee61b262bbb8a7d87401a0c4292f1"}
KNOWN_MAMBA_ARCHIVE = {"path": "tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets/eegmamba_recheck_report_history/q4_recheck_1791547936287947400.json", "sha256": "ea8a12c1315b51c39a18f02e7955ddf58cb151c7e51a6652a8ffc36d3d3cfd3a"}
KNOWN_MAMBA_LEDGER = {"path": "tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/engineering_attempts/EEGMamba/1791547890770467006_4c86e13ed7424198b3e9c439a6a85b91.json", "sha256": "e0ea75607ee48e7e85a8879c747a8e5cfcd36cd3d22d8836d28c9390f3101f8d"}
KNOWN_G1_CANCEL = {"path":"tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/execution_acceleration_v1/sync_boundary/cancel_waiting_complete_1791553589427394800.json","sha256":"986d4e3d82cb51e0d697cae20a50309378daaa6d68c75565aae403263ef01da0"}
KNOWN_MAMBA_OLD_SHA256 = "2a499ff86ff460c54b16cf8167f880e4d12e3d05a2902300833284dee26ff99c"
PHASE_PATTERNS = {
    "formal_metadata": "runs/*/seed_*/fold_*/metadata.json",
    "formal_failures": "runs/*/seed_*/fold_*/failure_*.json",
    "ft_attempts": "finetune_v2_attempts/*/*/attempt.json",
    "exports": "**/export_attempts/*.json",
    "supervised_engineering": "supervised_engineering/*/*.json",
    "ssl_failed_engineering": "ssl/engineering_attempts/*.json",
    "mamba_gpu_engineering": "engineering_attempts/EEGMamba/*.json",
    "mamba_recheck_inner_probe": "assets/eegmamba_recheck_report_history/*.json",
    "evaluation": "evaluation/*/seed_*/fold_*/*/attempts/*.json",
    "environment_repair": "assets/eegmamba_dependency_repair/*/installation_*.json",
    "engineering_contracts": "engineering/*.json",
    "resume_events": "resume_events/*.json",
    "ready_recovery_events": "ready_recovery/*.json",
    "ready_recovery_statuses": "ready_recovery/*/status.json",
    "progress_events": "progress_events/*.json",
    "queue_attempt_statuses": "continuation_queue*/attempts/*/status.json",
    "diagnostic_batching_qualification": "execution_acceleration_v1/*/qualification_attempts/*/attempt.json",
    "sync_boundary_inspections": "execution_acceleration_v1/sync_boundary*/inspection_*.json",
    "sync_boundary_proofs": "execution_acceleration_v1/sync_boundary*/boundary_*.json",
    "sync_boundary_cancellations": "execution_acceleration_v1/sync_boundary*/cancel_waiting_complete_*.json",
    "sync_boundary_deferrals": "execution_acceleration_v1/sync_boundary*/optional_sync_qualification_deferred_*.json",
    "q5_natural_handoff": "continuation_queue_v5/handoff_complete.json",
    "optional_diagnostic_batching_qualification_v2": "execution_acceleration_v1/*/qualification_attempts_v2/*/attempt.json",
}

for _version in range(4,11):
    PHASE_PATTERNS['repair_qualification_'+str(_version)]='deterministic_repair_qualification_v'+str(_version)+'/*/*/attempt.json'
    PHASE_PATTERNS['repair_controller_'+str(_version)]='deterministic_repair_qualification_v'+str(_version)+'/*/*/controller.json'
for _name,_pattern in {'mamba_repeat_diagnosis':'mamba_repeat_diagnosis_v1/*/*/attempt.json','mamba_repeat_controller':'mamba_repeat_diagnosis_v1/*/*/controller.json','CodeBrain_worker':'continuation_queue_v13/attempts/*/worker.json','Mamba_worker':'continuation_queue_v14/attempts/*/worker.json'}.items():PHASE_PATTERNS[_name]=_pattern
ACTUAL_REVIEW=None
_CURRENT_LIFECYCLE=None
if hashlib.sha256(RESOURCE_RUNNER.read_bytes()).hexdigest()!=RESOURCE_SHA256:raise ValueError('Reviewed resource7 source changed before import')
_spec=importlib.util.spec_from_file_location('_final8_resource',RESOURCE_RUNNER)
R7=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(R7)

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


def measurement_plan(condition, registration, lock_path):
    """Independent source-bound description of the exact original head probe."""
    name = condition["condition_id"]
    fits = [r for r in registration["fits"] if r["condition"] == name and r["seed"] == 11 and r["fold"] == 1]
    require(len(fits) == 1, "Expected exactly one registered seed11/fold1 identity")
    fit = fits[0]
    n = 16 if condition["policy_key"] == "diagnostic" else 8
    people = fit["train_ids"][:n]
    require(len(people) == n and len(set(people)) == n and not set(people) & set(fit["val_ids"] + fit["test_ids"]), "Original probe roles differ")
    lock = read(lock_path); contract = lock["contract"]
    require(lock["condition"] == name and lock["registration"] == desc(OUT / "registration.json"), "Head contract identity differs")
    canonical = hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(lock["configuration_sha256"] == canonical and contract["probe_subject_ids"] == people, "Original head contract/probe changed")
    for item in contract["implementation_sources"] + contract["weight_sources"]:
        checked(item)
    paired = condition["policy_key"] == "diagnostic"
    manifest_binding = contract["input_contract"]["frozen_cache"] if paired else desc(OUT / "frozen_features" / name / "manifest.json")
    require(manifest_binding in contract["weight_sources"], "Probe feature manifest not bound by original contract")
    manifest = read(checked(manifest_binding))
    if not paired:
        require(manifest.get("passed") is True and manifest.get("condition") == name and manifest.get("encoder_fit_subject_ids") == [], "Frozen probe manifest differs")
    by_id = {row["subject_id"]: row for row in manifest["records"]}
    require(len(by_id) == len(manifest["records"]) and set(people).issubset(by_id), "Duplicate/missing probe feature person")
    key = ("eegpt" if name.startswith("EEGPT-") else "labram") if paired else None
    records = [{"subject_id": person, "features": by_id[person]["features"],
                "archive_key": key if paired else by_id[person].get("archive_key")} for person in people]
    return {"condition": name, "paired": paired, "people": people, "contract": contract,
            "lock": desc(lock_path), "manifest": manifest_binding, "features": records}


def verified_resource_record(value, plan):
    """Reject labels without a complete, correctly bound actual attempt trace."""
    require(desc(RESOURCE_RUNNER)["sha256"] == RESOURCE_SHA256, "Reviewed supplementary runner bytes changed")
    require(value.get("source") == desc(RESOURCE_RUNNER), "Measurement source must be the reviewed v8 runner")
    require(desc(RESOURCE_DELEGATE)["sha256"] == RESOURCE_DELEGATE_SHA256
            and value.get("delegated_resource_measurement_source") == desc(RESOURCE_DELEGATE)
            and type(value.get("resource_measurement_version")) is int and value["resource_measurement_version"] == 8,
            "Actual new resource measurement must delegate the sealed v2 mathematics")
    require(value.get("status") == "complete" and value.get("passed") is True
            and value.get("subtype") == "supplemental_head_resource_measurement"
            and value.get("condition") == plan["condition"]
            and value.get("new_measurement_not_historical_backfill") is True
            and value.get("historical_resource_fields_modified") is False, "Measurement is not a completed independent new probe")
    require(value.get("registration") == desc(OUT / "registration.json") and value.get("condition_source_lock") == plan["lock"], "Measurement registration/contract differs")
    require(type(value.get("seed")) is int and value["seed"] == 11 and type(value.get("fold")) is int and value["fold"] == 1
            and value.get("input_role") == "train" and value.get("probe_subject_ids") == plan["people"], "Measurement must use exact original seed11/fold1 probe people")
    require(value.get("feature_manifest") == plan["manifest"]
            and value.get("loaded_feature_sources") == [r["features"] for r in plan["features"]]
            and value.get("feature_archive_keys_read") == [{"subject_id":r["subject_id"], "key":r["archive_key"]} for r in plan["features"]], "Actual feature descriptors/archive keys differ")
    for item in value["loaded_feature_sources"]:
        checked(item)
    require(value.get("actual_trainable_parameter_names") == plan["contract"]["trainable_parameter_names"], "Measurement head parameter names differ")
    expected_counts = {"optimizer_updates":0,"gradient_passes":1,"optimizer_step_started":0,"optimizer_step_returned":0,
        "formal_fits":0,"new_encoder_forwards":0,"forward_started":1,"forward_returned":1,"backward_started":1,"backward_returned":1,
        "model_constructions_started":1,"model_constructions_returned":1}
    require(all(type(value.get(k)) is int and value[k] == v for k,v in expected_counts.items()), "Actual head operation trace differs")
    require(number(value.get("seconds")) and value["seconds"] > 0 and number(value.get("forward_backward_seconds"))
            and 0 < value["forward_backward_seconds"] <= value["seconds"] and count(value.get("peak_allocated_bytes"))
            and value["peak_allocated_bytes"] > 0, "Actual positive resource measurements required")
    require(value.get("resume_checked") is True, "Actual checkpoint roundtrip missing")
    initial = value.get("initial_state_sha256")
    require(isinstance(initial,str) and len(initial)==64 and all(c in "0123456789abcdef" for c in initial)
            and value.get("restored_state_sha256") == initial, "Roundtrip state fingerprints differ")
    probe_path = checked(value["resume_probe"])
    require(probe_path.parent == (OUT / "supervised_engineering" / plan["condition"]).resolve() and probe_path.suffix == ".pt", "Resume probe belongs to another condition")
    parameters = plan["contract"].get("parameters", plan["contract"]["input_contract"].get("parameters"))
    require(type(value.get("head_parameters")) is int and value["head_parameters"] == parameters > 0, "Actual head parameter count differs")
    require(value.get("actual_optimizer",{}).get("name") == ("Adamax" if plan["paired"] else "AdamW"), "Actual probe optimizer type differs")
    gate = value.get("gate",{})
    require(gate.get('status')=='ready_after_full_e12_verification' and gate.get('registration')==desc(OUT/'registration.json'), 'Actual E12 measurement gate required')
    require(gate.get('training_lifecycle')==_CURRENT_LIFECYCLE and _CURRENT_LIFECYCLE is not None and gate.get('postprocessing_source_review')==ACTUAL_REVIEW,'Measurement must bind actual Q13/Q14/CPU17/E12 and this reviewed postprocessing source')
    R7.source_review(gate['postprocessing_source_review'])
    proof = gate.get("evaluation_verification",{})
    require(proof.get("passed") is True and proof.get("complete_manifest") == desc(OUT/"evaluation/complete.json")
            and type(proof.get("formal_fits_verified")) is int and proof["formal_fits_verified"] == 360
            and type(proof.get("supervised_fits_evaluated")) is int and proof["supervised_fits_evaluated"] == 345
            and type(proof.get("supervised_conditions")) is int and proof["supervised_conditions"] == 23
            and type(proof.get("model_constructions")) is int and proof["model_constructions"] == 0
            and type(proof.get("gpu_forward_calls")) is int and proof["gpu_forward_calls"] == 0, "Measurement does not bind verified existing full E12 completion")
    require(type(proof.get("evaluator_execution",{}).get("version")) is int
            and proof["evaluator_execution"]["version"] == 12
            and proof["evaluator_execution"].get("wrapper_source") == desc(WORK/"tridep_ccml/eeg_all_candidates/evaluate_v11.py"),
            "Head resource verification used another evaluation execution")
    require(isinstance(value.get("measurement_scope"),dict) and all(value["measurement_scope"].get(k) for k in ("seconds","forward_backward_seconds","peak_allocated_bytes","features","no_historical_backfill")), "Measurement timing/memory scope absent")
    return True


def head_resource_gate(registration):
    """Require exact independent head measurements and a released lease."""
    result = {"passed": True, "conditions_requiring_new_measurement": [], "verified_measurements": [], "pending": []}
    if (OUT/"supervised_engineering/supplemental_head_resource_owner.json").exists():
        result.update(passed=False, pending=[{"condition":None,"reason":"Supplemental measurement owner has not been released"}])
        return result
    for condition in registration.get("configuration", {}).get("conditions", []):
        if condition["policy_key"] not in ("diagnostic", "frozen"):
            continue
        name = condition["condition_id"]
        lock_path = OUT / "engineering" / (name + ".json")
        if not lock_path.exists():
            result["pending"].append({"condition": name, "reason": "head engineering contract missing"})
            continue
        contract = read(lock_path)["contract"]
        seconds = next((contract[k] for k in ("seconds", "forward_backward_seconds") if number(contract.get(k)) and contract[k]>0), None)
        peak = next((contract[k] for k in ("peak_allocated_bytes", "peak_gpu_allocated_bytes") if count(contract.get(k)) and contract[k]>0), None)
        if seconds is not None and peak is not None:
            continue
        result["conditions_requiring_new_measurement"].append(name)
        verified, issues = [], []
        paths = sorted((OUT / "supervised_engineering" / name).glob("measurement_*.json"))
        for path in paths:
            value = read(path)
            try:
                plan = measurement_plan(condition, registration, lock_path)
                verified_resource_record(value, plan)
                verified.append({"artifact": desc(path), "condition": name, "source": value["source"],
                    "seconds": value["seconds"], "peak_allocated_bytes": value["peak_allocated_bytes"],
                    "optimizer_updates": 0, "gradient_passes": 1, "forward_backward_seconds": value["forward_backward_seconds"],
                    "input_role": "train", "probe_subject_ids": value["probe_subject_ids"],
                    "resume_probe":value["resume_probe"], "gate":value["gate"],
                    "old_missing_seconds_remains_unknown": seconds is None,
                    "old_missing_peak_remains_unknown": peak is None,
                    "new_measurement_not_historical_backfill": True})
            except (KeyError, ValueError, TypeError, OSError) as error:
                issues.append({"artifact": desc(path), "reason": str(error)})
        if len(verified)==1:
            result["verified_measurements"].extend(verified)
        else:
            result["pending"].append({"condition": name, "reason": "Exactly one complete, bound new head resource measurement required", "observed_invalid_records": issues, "valid_records_count":len(verified)})
    result["passed"] = not result["pending"]
    if result["passed"] and len(result["conditions_requiring_new_measurement"]) != EXPECTED_RESOURCE_MEASUREMENTS:
        result.update(passed=False)
        result["pending"].append({"condition": None, "reason": "Actual original missing-resource condition set must contain the registered twelve probes"})
    return result


def final_source_gate(queue_registration):
    R7.source_review(ACTUAL_REVIEW)
    required={PREVIOUS_FINALIZER:PREVIOUS_FINALIZER_SHA256,COST_SOURCE:COST_SHA256,RESOURCE_RUNNER:RESOURCE_SHA256,RESOURCE_DELEGATE:RESOURCE_DELEGATE_SHA256,CPU_AUDITOR:CPU_AUDITOR_SHA256}
    bindings=[]
    for path,digest in required.items():
        binding=desc(path);require(binding['sha256']==digest,'Reviewed final source changed: '+str(path));bindings.append(binding)
    lifecycle=_CURRENT_LIFECYCLE
    require(lifecycle is not None,'Verified current lifecycle required')
    bindings.extend(lifecycle['training_sources'])
    bindings.extend(lifecycle['post_training_sources']);bindings.extend([lifecycle['post_training_source_review'],lifecycle['training_registration'],lifecycle['training_source_review'],ACTUAL_REVIEW])
    return list({x['path']:x for x in bindings}.values())



def recovery_execution_proof():
    from tridep_ccml.eeg_all_candidates import lost_f3_closure_v3 as L
    proof=L.verify_recovery_registration(); current=proof['value']; previous=read(checked(current['previous_recovery_registration']))
    terminal=read(QUEUE/'status.json')
    require(terminal['recovery_registration']==proof['registration'] and terminal['recovery_execution_source']==current['source'],'Actual recovery3 terminal source differs')
    result={'recovery_registration':proof['registration'],'recovery_execution_source':current['source'],'recovery_closure':terminal['recovery_closure'],
      'previous_recovery_registration':current['previous_recovery_registration'],'previous_recovery_closure':current['previous_recovery_closure'],
      'original_recovery_registration':previous['previous_recovery_registration'],'original_recovery_closure':{'path':'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/process_loss_closures/lost_f3_1791597116209422300.json','sha256':'e56bbb201c067bf0bab31c0d405520fb2860deb12796f088072bb30bdd73de65'}}
    for binding in result.values():checked(binding)
    return result



def completed_cpu_gate(registration):
    lifecycle=_CURRENT_LIFECYCLE
    require(lifecycle is not None and lifecycle.get('CPU_checkpoint_same_bytes_verified')==405,'Current verified CPU17 same-byte lifecycle required')
    return {'passed':True,'gate':lifecycle['CPU_gate'],'audit':lifecycle['CPU_audit'],'auditor_source':desc(CPU_AUDITOR),'verified_fits':135,'prior_actual_cpu_checkpoint_loads':405,'new_cpu_checkpoint_loads':0}


def preflight():
    global _CURRENT_LIFECYCLE
    _CURRENT_LIFECYCLE=None
    value=R7.inspect_gate(review=ACTUAL_REVIEW,verify_report=False)
    value.update(final_snapshot=False,draft_not_completion_certificate=True,outer_prediction_artifacts_read=False,completed_report_verifier_called=False)
    if value['status']!='ready_for_readonly_verification':return value
    lifecycle=value['training_lifecycle']; _CURRENT_LIFECYCLE=lifecycle
    registration=read(OUT/'registration.json')
    value['formal_status_counts']=value['actual_fit_statuses']
    value['training_status_descriptors']=[desc(OUT/'runs'/fit['fit_id']/'status.json') for fit in registration['fits']]
    value['verified_final_sources']=final_source_gate(read(checked(lifecycle['training_registration'])))
    value['recovery_execution_proof']=recovery_execution_proof()
    value['completed_finetune_cpu_gate']=completed_cpu_gate(registration)
    value['supplemental_head_resource_gate']=head_resource_gate(registration)
    value['gate_artifacts']=value['verified_final_sources']+[lifecycle[k] for k in ('registration','training_registration','training_terminal','CPU_gate','CPU_audit','evaluation_control_completion','evaluation_runtime','evaluation_released_owner')]+[x['artifact'] for x in value['supplemental_head_resource_gate']['verified_measurements']]
    if not value['supplemental_head_resource_gate']['passed']:
        value.update(status='pending',reasons=['Actual twelve zero-update head resource measurements remain pending'])
    return value



def verify_existing_evaluation():
    return R7.P.verify_existing()['evaluation_verification']



def collect_cost():
    require(desc(COST_SOURCE)['sha256']==COST_SHA256,'Reviewed Cost18 changed')
    module=R7.load_pinned(COST_SOURCE,COST_SHA256,'_final8_actual_cost18')
    return module.collect(),module.delegate().load_v5().load_v4()



def verify_training_cost(snapshot, registration, collector):
    require(snapshot.get("schema_version") == 18 and snapshot.get("source") == desc(COST_SOURCE)
            and snapshot.get("execution_observer_verification", {}).get("passed_for_observed_stable_f3_records") is True
            and snapshot["execution_observer_verification"]["errors"] == [],
            "Cost18 actual execution source/observer verification missing")
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


def bind_mamba_recheck_evidence(snapshot):
    """Bind the archived repeated probe without counting its parent twice."""
    proof = read(checked(KNOWN_MAMBA_RESTORE))
    probe = read(checked(KNOWN_MAMBA_ARCHIVE))
    ledger = read(checked(KNOWN_MAMBA_LEDGER))
    require(proof.get("passed") is True and proof.get("real_new_recheck_report_archived") == KNOWN_MAMBA_ARCHIVE
            and proof.get("new_recheck_primary_ledger") == KNOWN_MAMBA_LEDGER
            and proof.get("original_restored_report", {}).get("sha256") == KNOWN_MAMBA_OLD_SHA256
            and proof.get("original_exact_recovered_copy", {}).get("sha256") == KNOWN_MAMBA_OLD_SHA256,
            "Actual Mamba recheck and exact old-byte recovery provenance differs")
    old = read(checked(proof["original_restored_report"]))
    checked(proof["original_exact_recovered_copy"])
    checked(proof["recovery_provenance"])
    require(all(proof.get(key) is True for key in ("new_recheck_is_engineering_not_formal_training",
            "old_source_bound_reports_and_new_real_cost_both_preserved", "original_report_unchanged_after_exact_byte_recovery",
            "nested_time_and_counter_reports_are_not_added_again", "no_model_checkpoint_or_raw_array_operations")),
            "Mamba recovery scope is not explicit")
    require(probe.get("passed") is True and ledger.get("passed") is True and ledger.get("status") == "complete"
            and ledger.get("engineering") is True and ledger.get("cohort_data_or_labels_used") is False,
            "Repeated Mamba invocation must be completed synthetic engineering")
    expected_ledger = {"encode_started":4,"encode_returned":4,"backward_started":1,"backward_returned":1,
                       "sgd_step_started":1,"sgd_step_returned":1,"formal_fits":0,"formal_optimizer_updates":0}
    require(all(type(ledger.get(key)) is int and ledger[key] == value for key,value in expected_ledger.items())
            and all(type(probe.get(key)) is int and probe[key] == value for key,value in
                    {"gpu_encode_calls":4,"engineering_optimizer_steps":1,"formal_fits":0}.items()),
            "Actual Mamba synthetic counters differ")
    require(number(probe.get("seconds")) and probe["seconds"] == proof.get("new_measured_seconds") == 20.169247279
            and number(old.get("seconds")) and old["seconds"] == proof.get("old_measured_seconds") == 140.135474809
            and type(probe.get("peak_gpu_allocated_bytes")) is int
            and probe["peak_gpu_allocated_bytes"] == proof.get("new_peak_allocated_bytes") == 309073408,
            "Mamba original/new measured resources differ")
    records = snapshot["cost_records"]
    primary = [row for row in records if row["artifact"] == KNOWN_MAMBA_LEDGER]
    require(len(primary) == 1 and primary[0]["category"] == "mamba_gpu_engineering"
            and primary[0]["aggregation_role"] == "primary_observed_invocation"
            and not primary[0].get("counter_issues"), "Mamba primary invocation missing, duplicated or misclassified")
    expected_normalized = {"encoder_forward_started":4,"encoder_forward_returned":4,"backward_started":1,
                           "backward_returned":1,"optimizer_started":1,"optimizer_returned":1}
    require(all(type(primary[0]["observed_counters"].get(key)) is int and primary[0]["observed_counters"][key] == value
                for key,value in expected_normalized.items()), "Mamba primary cost counters do not match actual invocation")
    endpoint_seconds = (datetime.fromisoformat(ledger["finished_at"]) - datetime.fromisoformat(ledger["started_at"])).total_seconds()
    require("seconds" not in ledger and endpoint_seconds == 21.727767
            and primary[0].get("seconds_observed") == endpoint_seconds
            and primary[0].get("seconds_source_field") == "finished_at minus started_at",
            "Mamba primary duration must preserve recorded endpoint scope")
    require(not any(row["artifact"] == KNOWN_MAMBA_ARCHIVE for row in records), "Mamba archived inner probe already counted")
    bindings = [KNOWN_MAMBA_RESTORE, KNOWN_MAMBA_ARCHIVE, KNOWN_MAMBA_LEDGER,
                proof["original_restored_report"], proof["original_exact_recovered_copy"], proof["recovery_provenance"]]
    inputs = {item["path"]: item for item in snapshot["inputs"]}
    for binding in bindings:
        require(binding["path"] not in inputs or inputs[binding["path"]] == binding, "Mamba cost input binding changed")
        inputs[binding["path"]] = binding
    snapshot["inputs"] = list(inputs.values())
    records.append({"artifact": KNOWN_MAMBA_ARCHIVE, "category": "mamba_recheck_inner_probe",
        "aggregation_role": "supplemental_inner_probe_already_in_engineering_invocation",
        "primary_invocation": KNOWN_MAMBA_LEDGER, "seconds_observed": probe["seconds"],
        "seconds_scope": "nested actual probe timer; already within primary ledger endpoint interval, never added again",
        "peak_allocated_bytes": probe["peak_gpu_allocated_bytes"], "counts_added_again":False,
        "recorded_cost_fields": {"encoder_returns":4,"backward_returns_in_primary":1,"synthetic_sgd_returns":1,"formal_fits":0},
        "unknown_counter_fields":[], "counter_issues":[]})
    snapshot["category_observations"]["mamba_recheck_inner_probe"] = {"primary_invocations":0,
        "primary_status_counts":{}, "supplemental_records":1,"counter_totals":{},
        "known_recorded_scope_seconds_subtotal":None,"seconds_unknown_records":0,"full_category_wall_or_gpu_busy_time":None}
    peaks = [snapshot.get("max_observed_peak_allocated_bytes"), probe["peak_gpu_allocated_bytes"]]
    snapshot["max_observed_peak_allocated_bytes"] = max(v for v in peaks if count(v))
    snapshot["mamba_recheck_cost_evidence"] = {"restoration_proof":KNOWN_MAMBA_RESTORE,
        "original_restored_report":proof["original_restored_report"],"original_probe_seconds":old["seconds"],
        "primary_invocation":KNOWN_MAMBA_LEDGER,"archived_inner_probe":KNOWN_MAMBA_ARCHIVE,
        "primary_ledger_endpoint_seconds":endpoint_seconds,"inner_probe_seconds":probe["seconds"],
        "new_peak_allocated_bytes":probe["peak_gpu_allocated_bytes"],
        "known_actual_encoder_returns":4,"known_actual_backward_returns":1,"known_synthetic_sgd_returns":1,
        "formal_fits":0,"formal_optimizer_updates":0,"counts_added_again":False,"nested_seconds_added_again":False,
        "full_invocation_precise_wall_seconds":None,"gpu_busy_seconds":None}


def append_optional_qualification_v2_cost(snapshot, collector):
    """Extend the sealed v5 inventory; never relabel optional deferral as qualification."""
    require(desc(QUALIFIER_V2)['sha256']==QUALIFIER_V2_SHA256
            and desc(QUALIFIER_V1)['sha256']==QUALIFIER_V1_SHA256,'Optional qualifier source changed')
    category='optional_diagnostic_batching_qualification_v2'
    require(category not in snapshot['category_observations'],'Optional V2 costs already aggregated')
    added=[];seen={};existing={r['artifact']['path'] for r in snapshot['cost_records']}
    pattern=PHASE_PATTERNS[category]
    zero_fields=('backward_calls','optimizer_updates','formal_optimizer_updates','test_forward_count','checkpoint_weight_deserializations')
    counter_fields=('forward_started','forward_returned','encoder_forward_started','encoder_forward_returned',
                    'encoder_windows_started','encoder_windows_returned','forbidden_backward_attempts','forbidden_optimizer_attempts')
    for path in sorted(OUT.glob(pattern)):
        artifact=desc(path);value=read(checked(artifact))
        require(artifact['path'] not in existing,'Optional V2 attempt already appears in sealed collector inventory')
        require(value.get('schema')=='diagnostic_batching_qualification_attempt_v2'
                and value.get('kind')=='engineering_qualification'
                and value.get('qualifier_source')==desc(QUALIFIER_V2)
                and value.get('delegated_measurement_source')==desc(QUALIFIER_V1),'Unknown optional qualification provenance')
        require(value.get('status') in ('complete','failed','deferred'),'Optional attempt is still mutable or unknown')
        require(value.get('condition')==path.parents[2].name
                and type(value.get('seed')) is int and value['seed']==11
                and type(value.get('fold')) is int and value['fold']==1,'Optional attempt identity differs')
        require(value.get('actual_queue')=='continuation_queue_v5'
                and value.get('expanded_batch_activation_authorized') is False
                and type(value.get('production_diagnostic_chunk')) is int and value['production_diagnostic_chunk']==8
                and value.get('nested_measurement_timings_are_additive') is False,'Optional observation is not original8 production execution')
        require(all(type(value.get(k)) is int and value[k]==0 for k in zero_fields),'Optional qualification recorded forbidden training/test work')
        require(all(type(value.get(k)) is int and value[k]>=0 for k in counter_fields),'Optional qualification has invalid counters')
        require(type(value.get('gpu_stage_entered')) is bool,'Optional GPU entry state unknown')
        if 'seconds' in value:require(number(value['seconds']),'Invalid optional whole-attempt seconds')
        if value.get('model_constructions') is not None:require(count(value['model_constructions']),'Invalid optional model-construction count')
        peaks=[m['peak_allocated_bytes'] for m in value.get('measurements',[])]
        failure_peak=value.get('peak_allocated_bytes_observed_at_failure')
        if failure_peak is not None:peaks.append(failure_peak)
        require(all(count(n) for n in peaks),'Optional peak must be explicit integer bytes')
        peak=max(peaks,default=None)
        require(value.get('peak_allocated_bytes_observed')==peak
                and (value.get('peak_allocated_bytes_observed') is None or count(value['peak_allocated_bytes_observed'])),
                'Optional peak aggregate differs from recorded repetitions/failure')
        if value['status']=='deferred':
            deferred_zero=('model_constructions','raw_reads','checkpoint_reads','torch_imports')+counter_fields
            require(value['gpu_stage_entered'] is False and value.get('qualification_performed') is False
                    and type(value.get('unflushed_crash_tail_exact')) is int and value['unflushed_crash_tail_exact']==0
                    and all(type(value.get(k)) is int and value[k]==0 for k in deferred_zero)
                    and not value.get('measurements') and peak is None
                    and value.get('peak_scope')=='no_GPU_or_model_work','Deferred observation conceals actual work')
            observation=value.get('external_gpu_observation',{})
            require(observation.get('idle') is False and observation.get('contexts'),'Deferral lacks external GPU context evidence')
        else:
            require(value.get('unflushed_crash_tail_exact','missing') is None,'Actual optional probe tail cannot be zero-filled')
        normalized=collector.normalized(value)
        require(not normalized['counter_issues'],'Conflicting/invalid optional qualification counters')
        normalized['peak_allocated_bytes']=peak
        record={'artifact':artifact,'category':category,'aggregation_role':'primary_observed_invocation',
                'attempt_id':artifact['path'],'condition':value['condition'],'seed':value['seed'],'fold':value['fold'],
                'status':value['status'],'gpu_stage_entered':value['gpu_stage_entered'],
                'formal_fits':0,'formal_optimizer_updates':0,'production_diagnostic_chunk':8,
                'counter_scope':'Producer snapshots; deferred pre-GPU zero work is explicit; actual crash tails unknown',
                'unflushed_crash_tail_exact':value['unflushed_crash_tail_exact'],
                'recorded_model_constructions':value.get('model_constructions'),
                'explicit_zero_work_fields':{key:value[key] for key in zero_fields},
                'recorded_measurements':value.get('measurements',[]),'failure_peak_allocated_bytes_observed':failure_peak,
                'external_gpu_observation':value.get('external_gpu_observation'),
                'nested_candidate_measurement_seconds_added':False,'qualification_activation_authorized':False,**normalized}
        identity=(artifact['sha256'],value.get('started_at'))
        if identity in seen:record.update(aggregation_role='duplicate_invocation_artifact',alias_of=seen[identity])
        else:seen[identity]=artifact
        added.append(record);snapshot['inputs'].append(artifact)
    primary=[r for r in added if r['aggregation_role']=='primary_observed_invocation']
    snapshot['cost_records'].extend(added)
    if added:
        totals={}
        for name in added[0]['observed_counters']:
            known=[r['observed_counters'][name] for r in primary if r['observed_counters'][name] is not None]
            totals[name]={'known_observed_subtotal':sum(known) if known else None,'records_with_value':len(known),
                          'records_unknown':len(primary)-len(known),'exact_category_total':None}
        seconds=[r['seconds_observed'] for r in primary if r['seconds_observed'] is not None]
        snapshot['category_observations'][category]={'primary_invocations':len(primary),'supplemental_records':len(added)-len(primary),
            'counter_totals':totals,'primary_status_counts':dict(Counter(r['status'] for r in primary)),
            'known_recorded_scope_seconds_subtotal':sum(seconds) if seconds else None,
            'seconds_unknown_records':len(primary)-len(seconds),'full_category_wall_or_gpu_busy_time':None,
            'candidate_measurements_nested_not_added':True,'deferred_is_not_qualification_passed':True}
        snapshot['max_observed_peak_allocated_bytes']=max((n for n in
            [snapshot['max_observed_peak_allocated_bytes'],*[r['peak_allocated_bytes'] for r in added]] if count(n)),default=None)
    snapshot['optional_v2_cost_companion']={'source':desc(Path(__file__)),'qualifier_source':desc(QUALIFIER_V2),
        'delegated_measurement_source':desc(QUALIFIER_V1),'pattern':pattern,'artifacts':[r['artifact'] for r in added],
        'primary_invocations':len(primary),'duplicates_retained':len(added)-len(primary),'read_errors':[],
        'deferred_is_not_qualification_passed':True,'nested_measurement_times_added':False,'formal_training_cost_added':False}
    return added


def append_boundary_control_evidence(snapshot):
    """Immutable control proofs are supplemental; parent/wait intervals never add."""
    category='qualification_boundary_control_evidence'
    require(category not in snapshot['category_observations'],'Boundary control evidence already inventoried')
    names=('sync_boundary_inspections','sync_boundary_proofs','sync_boundary_cancellations','sync_boundary_deferrals','q5_natural_handoff')
    paths=sorted({path for name in names for path in OUT.glob(PHASE_PATTERNS[name])})
    require(checked(KNOWN_G1_CANCEL) in [p.resolve() for p in paths],'Known original guardian cancellation is missing')
    cancelled=read(checked(KNOWN_G1_CANCEL))
    require(cancelled.get('passed') is True and cancelled.get('optional_guardian_exit_verified') is True
            and all(type(cancelled.get(k)) is int and cancelled[k]==0 for k in
                    ('qualification_model_optimizer_GPU_calls','training_child_process_actions')),
            'Original cancellation must retain confirmed zero model/GPU work and zero training-child actions')
    rows=[];existing={row['artifact']['path'] for row in snapshot['cost_records']}
    for path in paths:
        artifact=desc(path);value=read(checked(artifact))
        require(artifact['path'] not in existing,'Boundary proof already represented in primary cost inventory')
        source=value.get('source')
        require(isinstance(source,dict) and Path(source['path']).suffix=='.py','Boundary evidence requires its actual controller source')
        checked(source)
        archived=[]
        for name in ('archived_owner','archived_state','old_owner_snapshot','old_attempt_snapshot'):
            if name in value:
                binding=value[name]
                require(Path(binding['path']).suffix=='.json','Retained control snapshot must be JSON')
                checked(binding);archived.append(binding)
        row={'artifact':artifact,'category':category,'aggregation_role':'supplemental_control_evidence_no_increment',
             'source':source,'retained_archived_json_bindings':archived,'status':value.get('status',value.get('phase')),'passed_as_recorded':value.get('passed'),
             'recorded_direct_cost_or_control_fields':{k:v for k,v in value.items() if not isinstance(v,(dict,list))
                 and any(term in k.lower() for term in ('seconds','calls','updates','fits','gpu','model','backward','actions'))},
             'observed_counters':{},'seconds_observed':None,'peak_allocated_bytes':None,
             'seconds_scope':'Control/parent/wait evidence; no independent CPU timer is inferred and no training wall interval is added',
             'unknown_counter_fields':['independent_control_cpu_seconds_not_inferred'],
             'counts_added_again':False,'qualification_work_counted_only_in_its_primary_attempt':True,
             'waiting_or_parent_seconds_added_to_training':False}
        rows.append(row);snapshot['inputs'].extend([artifact,source,*archived])
    snapshot['cost_records'].extend(rows)
    snapshot['category_observations'][category]={'primary_invocations':0,'supplemental_records':len(rows),
        'counter_totals':{},'known_recorded_scope_seconds_subtotal':None,'seconds_unknown_records':0,
        'full_category_wall_or_gpu_busy_time':None}
    snapshot['qualification_boundary_control_evidence']={'artifacts':[row['artifact'] for row in rows],
        'mutable_state_and_owner_files_are_not_primary_cost_records':True,'primary_qualifier_costs_added_again':False,
        'waiting_or_parent_seconds_added_to_training':False}
    return rows


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
        "cpu_or_synthetic_test_audit", "mixed_engineering_test_audit", "diagnostic_batching_qualification", "mamba_recheck_inner_probe"}
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
    stopped = snapshot.get("source_bound_stopped_attempts", [])
    handoff_files = sorted({*list((OUT / "continuation_queue_v4/handoff").glob("safe_stop_*.json")),
                            *list((QUEUE / "handoff").glob("safe_stop_*.json"))})
    require(handoff_files and {row["proof"]["path"] for row in stopped} == {desc(path)["path"] for path in handoff_files},
            "Actual handoff recovery proof inventory is incomplete")
    known_stop = [row for row in stopped if row["proof"] == {"path": KNOWN_HANDOFF, "sha256": KNOWN_HANDOFF_SHA256}]
    require(len(known_stop) == 1 and known_stop[0].get("passed") is True
            and known_stop[0].get("known_extra_optimizer_returns_outside_checkpoint") == 0
            and known_stop[0].get("known_extra_backward_returns_outside_checkpoint") == 2
            and known_stop[0].get("durable_epoch") == 31
            and known_stop[0].get("durable_optimizer_updates") == 124
            and known_stop[0].get("durable_backward_calls") == 837
            and known_stop[0].get("extra_work_not_added_again_to_observed_trace") is True
            and known_stop[0].get("final_elapsed_seconds") is None,
            "Known source-bound CSBrain 0-update/2-backward handoff boundary missing")
    for row in stopped:
        proof = read(checked(row["proof"]))
        require(row["artifact"] in [r["artifact"] for r in records if r["category"] == "formal_training_attempt"],
                "Stopped physical trace is not accounted in primary FT invocation inventory")
        snapshot["inputs"].append(row["proof"])
        records.append({"artifact": row["proof"], "category": "recovery_handoff_evidence",
                        "aggregation_role": "supplemental_checkpoint_boundary_already_in_formal_attempt",
                        "condition": proof["condition"], "seed": proof["seed"], "fold": proof["fold"],
                        "known_checkpoint_external_optimizer_returns": row["known_extra_optimizer_returns_outside_checkpoint"],
                        "known_checkpoint_external_backward_returns": row["known_extra_backward_returns_outside_checkpoint"],
                        "counts_added_again": False, "seconds_observed": None, "unknown_counter_fields": ["unreturned_callable_outcome"]})
        inventory.append({"phase": "recovery_handoff_evidence", "artifact": row["proof"], "categories": ["recovery_handoff_evidence"]})
    snapshot["category_observations"]["recovery_handoff_evidence"] = {"primary_invocations":0,
        "supplemental_records":len(stopped), "counter_totals":{}, "known_recorded_scope_seconds_subtotal":None,
        "seconds_unknown_records":0,"full_category_wall_or_gpu_busy_time":None}
    snapshot["acknowledged_acceleration_handoff_work_already_in_observed_traces"] = stopped
    category_counts.update({"recovery_handoff_evidence": len(stopped)})
    completion = read(OUT / "evaluation/complete.json")
    evaluation_report = read(checked(completion["evaluation_report"]))
    evaluation_attempts = evaluation_report["execution_attempts"]
    require(evaluation_attempts and all(item["path"] in by_path and any(r["artifact"] == item and r["category"] == "evaluation" for r in by_path[item["path"]]) for item in evaluation_attempts), "A bound evaluation attempt was not accounted")
    for row in records:
        if row["category"] == "evaluation":
            data = read(checked(row["artifact"]))
            require(data.get("optimizer_updates") == data.get("ema_updates") == 0, "Evaluation recorded a training/EMA update")
    snapshot["available_artifact_coverage"] = {"scope": "All named actual cost artifact families in PHASE_PATTERNS plus required engineering records, bound feature manifests, and v5 content-classified test audits; not a claim to recover unrecorded historical work.",
        "patterns": PHASE_PATTERNS, "inventory": inventory, "category_record_counts": dict(category_counts),
        "bound_evaluation_attempt_descriptors": evaluation_attempts, "all_bound_evaluation_attempts_accounted": True,
        "all135_finetune_attempt_fit_ids_accounted": True, "known_recovery_boundary_accounted": True}
    return inventory


def bind_lost_f3_recovery_cost(snapshot, recovery, collector):
    """All three immutable loss boundaries are subsets of their original traces."""
    status = snapshot.get('lost_f3_closure_verification', {})
    require(status.get('passed_for_observed_closures') is True and status.get('errors') == [],
            'Lost-process cost verification failed')
    parent_status = snapshot.get('lost_parent_observer_verification', {})
    require(parent_status.get('passed_for_observed_closures') is True and parent_status.get('errors') == [],
            'Lost-parent observer verification failed')
    rows = snapshot.get('source_bound_lost_f3_attempts', [])
    primary = [r for r in rows if r.get('aggregation_role') == 'primary_observed_invocation']
    expected = {recovery[k]['path']: recovery[k] for k in ('original_recovery_closure','previous_recovery_closure','recovery_closure')}
    require(len(expected) == len(primary) == 3
            and {r['closure']['path']: r['closure'] for r in primary} == expected,
            'Exactly the three independently bound process-loss events are required')
    files = [desc(path) for path in sorted((OUT / 'process_loss_closures').glob('lost_f3_*.json'))]
    require({r['path']: r for r in files} == expected, 'An actual process-loss closure is omitted or ambiguous')
    require(all(r['closure']['path'] in expected and r['closure'] == expected[r['closure']['path']]
                and r.get('aggregation_role') in ('primary_observed_invocation', 'duplicate_invocation_artifact') for r in rows),
            'An unexplained closure alias is present')
    inventory = []; cpu_rows = []; cpu_seen = set(); evidence_rows = []
    for binding in recovery.values():
        checked(binding)
        if binding not in snapshot['inputs']: snapshot['inputs'].append(binding)
    for row in primary:
        is_previous = row['closure'] == recovery['original_recovery_closure']
        registration = recovery['original_recovery_registration'] if is_previous else (recovery['previous_recovery_registration'] if row['closure']==recovery['previous_recovery_closure'] else recovery['recovery_registration'])
        require(row.get('passed') is True and row.get('recovery_registration') == registration,
                'Lost F3 recovery registration differs')
        manifest = read(checked(registration))
        require(row.get('closure_source') == manifest['source']
                and (row['closure'] != recovery['recovery_closure'] or row['closure_source'] == recovery['recovery_execution_source']),
                'Lost F3 actual recovery source differs')
        for key in ('known_extra_optimizer_returns_outside_checkpoint', 'known_extra_backward_returns_outside_checkpoint',
                    'durable_epoch', 'durable_optimizer_updates', 'durable_backward_calls'):
            require(count(row.get(key)), 'Event-specific durable/lower-bound count invalid: '+key)
        require(row.get('unflushed_tail_unknown') is True and row.get('unflushed_tail_counters', 'missing') is None
                and row.get('final_elapsed_seconds', 'missing') is None and row.get('graceful_terminal_flush') is False
                and row.get('extra_work_not_added_again') is True, 'Unknown lost process tail/time must stay null')
        actual = [r for r in snapshot['cost_records'] if r['artifact'] == row['artifact']
                  and r.get('aggregation_role') == 'primary_observed_invocation']
        require(len(actual) == 1 and actual[0]['category'] == 'formal_training_attempt'
                and actual[0]['status'] == 'running' and actual[0]['seconds_observed'] is None,
                'Original lost physical invocation must be retained once without fabricated finish')
        original = read(checked(row['artifact'])); normalized = collector.normalized(original)
        require(original.get('status') == 'running' and original.get('seconds') is None
                and not normalized['counter_issues'] and actual[0]['observed_counters'] == normalized['observed_counters'],
                'Original persisted physical counters changed')
        for key, durable_key, delta_key in (
                ('optimizer_returned', 'durable_optimizer_updates', 'known_extra_optimizer_returns_outside_checkpoint'),
                ('backward_returned', 'durable_backward_calls', 'known_extra_backward_returns_outside_checkpoint')):
            observed = actual[0]['observed_counters'].get(key)
            require(count(observed) and observed >= row[durable_key] and observed-row[durable_key] == row[delta_key],
                    'Persisted physical return and checkpoint boundary differ')
        if is_previous:
            require([row[k] for k in ('durable_epoch','durable_optimizer_updates','durable_backward_calls',
                    'known_extra_optimizer_returns_outside_checkpoint','known_extra_backward_returns_outside_checkpoint')]
                    == [8,32,216,4,27], 'Original CSBrain loss boundary changed')
        else:
            parents = [x for x in snapshot.get('source_bound_lost_parent_observer_scopes', []) if x['closure']==row['closure']]
            require(len(parents) == 1 and parents[0]['artifact'] == row.get('original_queue_attempt')
                    and parents[0]['closure'] == row['closure'] and parents[0]['final_elapsed_seconds'] is None
                    and parents[0]['added_again'] is False, 'New lost parent observer scope differs')
        identity = {key: original[key] for key in ('condition','seed','fold')}
        evidence_rows.append({'artifact': row['closure'], 'category': 'lost_f3_recovery_evidence',
            'aggregation_role': 'supplemental_checkpoint_boundary_already_in_formal_attempt', **identity,
            'known_checkpoint_external_optimizer_returns_lower_bound': row['known_extra_optimizer_returns_outside_checkpoint'],
            'known_checkpoint_external_backward_returns_lower_bound': row['known_extra_backward_returns_outside_checkpoint'],
            'counts_added_again': False, 'seconds_observed': None, 'unflushed_tail_counters': None,
            'unknown_counter_fields': ['lost_unflushed_tail','original_final_elapsed']})
        inventory.append({'phase':'lost_f3_recovery_evidence','artifact':row['closure'],'categories':['lost_f3_recovery_evidence']})
        cpu_binding = row['checkpoint_cpu_resume_proof']; cpu_key = (cpu_binding['path'],cpu_binding['sha256'])
        if cpu_key in cpu_seen: continue
        cpu_seen.add(cpu_key); cpu = read(checked(cpu_binding))
        require(cpu.get('passed') is True and count(cpu.get('checkpoint_loads')) and cpu['checkpoint_loads'] > 0
                and all(type(cpu.get(k)) is int and cpu[k] == 0 for k in
                    ('model_constructions','GPU_operations','raw_arrays_loaded','test_probability_reads'))
                and all(k not in cpu or (type(cpu[k]) is int and cpu[k] == 0) for k in ('optimizer_updates','backward_calls'))
                and ('seconds' not in cpu or cpu['seconds'] is None or number(cpu['seconds'])),
                'Actual independent zero-update CPU recovery preflight required')
        require(not any(r['artifact'] == cpu_binding for r in snapshot['cost_records']),
                'CPU recovery preflight already classified; do not add another primary scope')
        normalized = collector.normalized(cpu)
        require(not normalized['counter_issues'], 'CPU observer counter conflict')
        # A missing explicit timer is unknown, even if wall endpoints are present.
        normalized['seconds_observed'] = cpu.get('seconds')
        normalized['seconds_source_field'] = 'seconds' if cpu.get('seconds') is not None else None
        cpu_rows.append({'artifact':cpu_binding,'category':'recovery_cpu_preflight',
            'aggregation_role':'primary_observed_invocation', **identity,
            'cpu_checkpoint_loads':cpu['checkpoint_loads'],'gpu_calls':0,
            'counter_scope':'Independent CPU resume-safety validation, never formal training',
            'nested_checkpoint_counts_added':False, **normalized})
        inventory.append({'phase':'recovery_cpu_preflight','artifact':cpu_binding,'categories':['recovery_cpu_preflight']})
        if cpu_binding not in snapshot['inputs']: snapshot['inputs'].append(cpu_binding)
    require(not any(r['artifact'] == e['artifact'] for r in snapshot['cost_records'] for e in evidence_rows),
            'Loss evidence already accounted')
    snapshot['acknowledged_lost_f3_work_already_in_observed_traces'] = rows
    snapshot['cost_records'].extend(evidence_rows+cpu_rows)
    snapshot['category_observations']['lost_f3_recovery_evidence'] = {'primary_invocations':0,'supplemental_records':len(evidence_rows),
        'counter_totals':{},'known_recorded_scope_seconds_subtotal':None,'seconds_unknown_records':0,'full_category_wall_or_gpu_busy_time':None}
    counters = {}
    for key in {key for r in cpu_rows for key in r['observed_counters']}:
        known = [r['observed_counters'].get(key) for r in cpu_rows if r['observed_counters'].get(key) is not None]
        counters[key] = {'known_observed_subtotal':sum(known) if known else None,'records_with_value':len(known),
                         'records_unknown':len(cpu_rows)-len(known),'exact_category_total':None}
    seconds = [r['seconds_observed'] for r in cpu_rows if r['seconds_observed'] is not None]
    snapshot['category_observations']['recovery_cpu_preflight'] = {'primary_invocations':len(cpu_rows),'supplemental_records':0,
        'counter_totals':counters,'known_recorded_scope_seconds_subtotal':sum(seconds) if seconds else None,
        'seconds_unknown_records':len(cpu_rows)-len(seconds),'full_category_wall_or_gpu_busy_time':None}
    return inventory


def disclose_gaps(snapshot):
    records = snapshot["cost_records"]
    unknown = [row["artifact"]["path"] for row in records if row.get("unknown_counter_fields") or not row.get("aggregation_role", "").startswith("primary")]
    snapshot["unknown_or_opaque_records"] = sorted(set(unknown))
    snapshot["coverage_gaps"] = [
        "2026-10-10 从 CSBrain s33/f4 第8轮32次更新/216次反向恢复；原F3持久观测为36次更新/243次反向，额外4次更新和27次反向仅为已确认返回下界，已包含主尝试计数不重加。它与此前108−104的4/27及s22/f5的0/2是三个不同边界。原running记录保持原字节，丢失的诊断前向等未落盘尾部、原退出码/结束时刻/耗时均为null；暂停过夜不计训练。",
        "每个恢复事件的独立CPU预检按其真实报告分别计一次审查成本，不加入正式训练耗时；已记录秒数保留，未记录时长为null，嵌套checkpoint_validation计数不重加。",
        "新增LUNA s33/f3失联事件由独立第二恢复登记绑定；额外返回下界仅为该原尝试已保存物理计数减持久检查点边界，不重加，也不与旧CSBrain事件混同。原LUNA父任务仍running且最终耗时null；恢复后的重算只来自新的真实尝试。",
        "同进程午间暂停与系统低功耗重叠继续单独披露；成功尝试的原始包含暂停秒数保留为recorded elapsed，不代表active训练耗时。精确active及暂停/待机贡献保持null，最终交付须另绑定时间范围附页和未配对事件范围复核。",
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
        "F3 逐调用在内存记录，在逻辑更新/诊断/检查点边界落盘；正常结束计数完整，硬终止未落盘尾部继续未知。开始但未返回不能当作已完成更新。",
        "同一物理GPU上的单个本计划EEG工作进程，不等于设备没有外部C/G/C+G上下文；已观察外部占用时，可选性能测量延期并保留零探针工作证据，不声明未实测的GPU加速。",
        "自然边界控制器的检查、取消、GPU占用延后和Q5交接证明仅作旁证；等待与父进程区间不计为训练耗时，已存在的资格探针主账本不重复加，未独立测得的控制器CPU开销保持未知。",
        "Q5可选诊断资格检查单列V2主尝试：有外部GPU上下文而延后时，检查耗时保留，但模型/数据读取/前向/反向/更新为实证零，峰值未知且不视为资格通过；正式诊断仍使用chunk8。实测失败保留其已发生工作与未知尾部。",
        "诊断批处理资格只统计整次工程尝试耗时；各候选、预热与重复测量嵌套在该尝试内，不重复相加；失败实测峰值保留并取最大值。",
        "CSBrain s22/f5 安全交接保存第31轮124次更新/837次反向，另有0次更新与2次反向已确认返回；该额外工作已包含旧F2调用记录，不再加入小计。旧尝试结束耗时与未返回调用结局为null。",
        "Mamba 原工程报告140.135474809秒已恢复到原始字节；Q4新增复测的主账本端点区间21.727767秒与内层探针20.169247279秒分列，后者不再相加，也不加入正式训练或Q4父进程时间。新增4次编码、1次反向和1次合成SGD仅由主账本计一次，正式拟合与更新均为0，309073408 bytes峰值只取最大值。",
    ]
    snapshot['coverage_gaps'].extend([
        '第三次 LaBraM4s s33/f1 失联保留原55轮220/1485观察与54轮216/1458持久边界，额外至少4/27已在原主尝试中，不重加；退出时点、退出码和未落盘尾未知。',
        'Qual4至10全部实际尝试及单次Mamba重复诊断各以原worker主账计一次；父控制器时间为嵌套旁证。Qual8已发生模型构造但零前反向，不称零模型；未返回的内部工作未知。',
        'Mamba真实通过的确定性资格使用受审CPU FP32反向累加顺序修复，保持原生forward；不声称与原CUDA反向逐位相同或GPU加速。native函数调用是子计数，不等于CUDA kernel/autotune次数，不再加到模型反向总数。',
        'Q13 CodeBrain和Q14 Mamba分别保留真实来源与终态；旧Q5仍为其原要求剩余状态，绝不伪造为执行了新修复训练或E12评价。'])
    snapshot["all_plan_physical_optimizer_returns_exact"] = None
    snapshot["all_plan_wall_seconds_exact"] = None
    snapshot["rules"] = [rule for rule in snapshot["rules"] if not rule.startswith("No raw/features/checkpoint/model/test probability")]
    snapshot["rules"].append("The Cost18 collector reads ledgers and source-bound closure evidence; delegated provenance verification can hash official weight bytes, without tensor deserialization. Hash call counts/bytes/time were not separately instrumented. After the final gate, E12 verifies existing predictions and CPU checkpoint headers and rehashes bound raw/features bytes; no signal arrays are loaded, no new predictions are computed, and no model construction, GPU call or process launch occurs.")


def show(value, digits=3):
    return "未知" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def render_report(snapshot):
    lines = ["# EEG 九候选全计划成本伴随报告", "", f"生成时间：{snapshot['captured_at']}（北京时间）。",
        "", "全 360 次拟合、435 个固定评价检查点、Q13完成、Q14训练子任务exit0且owner释放（Q14原CPU16失败保留，独立CPU17恢复门实际通过）、E12 实际完成后，已用 E12 的只读完成核验检查既存结果；核验包括原始/特征文件的字节哈希，不加载信号数组或重算预测。此前真实135个微调拟合的405次CPU检查点读取已通过；此生成器仅复核其绑定证据，不重跑该审计。12个新增head资源补测均完成且释放owner。本报告独立附带，不修改已绑定的九候选统一报告。",
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
    bind_mamba_recheck_evidence(snapshot)
    append_optional_qualification_v2_cost(snapshot, collector)
    append_boundary_control_evidence(snapshot)
    lost_inventory = bind_lost_f3_recovery_cost(snapshot, gate['recovery_execution_proof'], collector)
    inventory = verify_phase_coverage(snapshot, registration)
    inventory.extend(lost_inventory)
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
    snapshot.update(schema_version=18, artifact_kind="Independent verified final full-plan cost snapshot",
        status="verified_final_cost", captured_at=now(), draft_not_completion_certificate=False, final_snapshot=True,
        all_available_artifacts_accounted=True, coverage_gaps_disclosed=True, completion_gate=gate,
        completion_gate_scope="Cheap prerequisite observations made before E12 verification; its false read/call flags apply only to that earlier stage.",
        existing_evaluation_verification=verification, finalizer_source=desc(Path(__file__)),
        previous_finalizer_source=desc(PREVIOUS_FINALIZER), actual_cost_collector_source=desc(COST_SOURCE),
        source_bound_cpu_checkpoint_verification=gate["completed_finetune_cpu_gate"],
        postprocessing_source_review=ACTUAL_REVIEW, actual_training_lifecycle=gate["training_lifecycle"],
        actual_completed_report_verifier_called=True, existing_outer_prediction_artifacts_read=True,
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


def main(argv=None):
    global ACTUAL_REVIEW
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group();group.add_argument('--run',action='store_true');group.add_argument('--check-only',action='store_true')
    parser.add_argument('--source-review');parser.add_argument('--source-review-sha256');args=parser.parse_args(argv)
    if not args.run and not args.check_only:
        print(json.dumps({'status':'preparation_only','Tensor_loads':0,'model_calls':0,'GPU_queries':0,'new_outputs':0}));return 0
    if args.source_review or args.source_review_sha256:
        require(args.source_review and args.source_review_sha256,'Both source-review fields required')
        ACTUAL_REVIEW=desc(WORK/args.source_review)
        require(ACTUAL_REVIEW['sha256']==args.source_review_sha256,'Explicit source-review SHA differs')
    value=preflight() if args.check_only else finalize()
    print(json.dumps(value,ensure_ascii=True));return 0 if value['status']=='verified_final_cost' else 2



if __name__ == "__main__":
    main()

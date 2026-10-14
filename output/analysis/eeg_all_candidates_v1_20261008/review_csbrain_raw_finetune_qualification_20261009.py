"""Verify existing CSBrain raw-FT engineering evidence without model execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

REPORT = Path(__file__).resolve().parent
WORK = REPORT.parents[2]
OUT = WORK / "tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008"
verified = {}


def desc(path):
    path = Path(path).resolve()
    assert path.suffix not in (".npy", ".npz") and "/runs/" not in path.as_posix(), "No raw/features/formal checkpoint access"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    value = {"path": path.relative_to(WORK).as_posix(), "sha256": digest.hexdigest()}
    verified[value["path"]] = value
    return value


def check(value):
    actual = desc(WORK / value["path"])
    assert actual["sha256"] == value["sha256"], value["path"]
    return actual


def read(path):
    binding = desc(path)
    assert Path(path).suffix == ".json"
    return json.loads(Path(path).read_text(encoding="utf-8")), binding


def main():
    lock, lock_desc = read(OUT / "engineering/CSBrain-lastblock-finetune.json")
    contract = lock["contract"]
    assert hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest() == lock["configuration_sha256"]
    registry, registry_desc = read(OUT / "registration.json")
    assert lock["condition"] == "CSBrain-lastblock-finetune" and lock["registration"] == registry_desc
    trace, trace_desc = read(OUT / "finetune_v2_attempts/CSBrain-lastblock-finetune/e_1791520704920988100_1e347180/attempt.json")
    assert trace["condition"] == lock["condition"] and trace["kind"] == "engineering" and trace["status"] == "complete"
    assert trace["seed"] is trace["fold"] is None and trace["implementation"] == contract["versioned_finetune_implementation"]
    assert contract["engineering_attempt_path"] == trace_desc["path"]
    q3, q3_desc = read(OUT / "continuation_queue_v3/registration.json")
    assert q3_desc["sha256"] == "e135585ca475b733bbc6bfbaa1616fe71f9b8fb1af8f2c0b413d2a9c8565b3a0"
    assert len(q3["sources"]) == 47
    for value in q3["sources"]: check(value)
    prepare, prepare_desc = read(OUT / "continuation_queue_v3/attempts/1791520703166322400-CSBrain-lastblock-finetune-prepare/status.json")
    job, = [job for job in q3["jobs"] if job["id"] == "CSBrain-lastblock-finetune-prepare"]
    assert prepare["job"] == job and prepare["registration"] == q3_desc and prepare["status"] == "complete" and prepare["returncode"] == 0
    assert job["module"] == "finetune_v2" and job["args"] == ["--conditions", lock["condition"], "--prepare"]
    check(prepare["log"])
    chronology = [prepare["started_at"], trace["started_at"], lock["locked_at"], trace["finished_at"], prepare["finished_at"]]
    assert [datetime.fromisoformat(value) for value in chronology] == sorted(datetime.fromisoformat(value) for value in chronology)
    for name in ("forward_checked", "gradient_checked", "resume_checked", "training_only_engineering_checked"): assert contract[name] is True
    assert contract["probe_subject_ids"] == registry["folds"]["fold_1"]["train"][:1] == ["02010024"]
    assert not set(contract["probe_subject_ids"]) & set(registry["folds"]["fold_1"]["val"] + registry["folds"]["fold_1"]["test"])
    raw_desc = {item["subject_id"]: item["raw"] for item in registry["raw_sources"]}
    assert contract["raw_sources"] == {sid: raw_desc[sid] for sid in contract["probe_subject_ids"]}
    input_contract, adapter = contract["input_contract"], contract["input_contract"]["adapter"]
    assert (input_contract["micro_subjects"], input_contract["window_chunk"], input_contract["windows_per_training_subject"], input_contract["all_window_validation"]) == (1, 8, 8, 60)
    assert adapter["formal_pretrained_strict_loaded"] is True and adapter["loaded_backbone_keys"] == 295
    assert (adapter["full_pretrained_parameters"], adapter["retained_encoder_parameters"], adapter["native_dimension"]) == (8902000, 8861800, 200)
    assert adapter["removed_reconstruction_projector_keys"] == ["proj_out.0.weight", "proj_out.0.bias"]
    assert adapter["source"]["repository"] == "https://github.com/yuchen2199/CSBrain" and adapter["source"]["revision"] == "185aee55b24d0410a830df8dd08d03f675616998"
    assert adapter["fresh_head_seed"] == 11 and adapter["fresh_head"] is True and adapter["initialized_from_frozen_best_head"] is False
    author_weight = check(adapter["source"]["weight"])
    for value in contract["implementation_sources"] + contract["weight_sources"]: check(value)
    author_input = adapter["input_contract"]
    assert author_input["source_shape"] == ["W", 19, 1000] and author_input["encoder_shape"] == ["W", 19, 5, 200]
    assert author_input["seconds_per_window"] == 5 and author_input["sample_rate_hz"] == 200
    assert author_input["new_statistical_normalization"] is None and author_input["author_reference_reproduced"] is False
    assert author_input["physical_amplitude_calibration_confirmed"] is False
    active = contract["trainable_parameter_names"]
    encoder = [name for name in active if name.startswith("encoder.")]
    heads = [name for name in active if name.startswith("head.")]
    assert len(active) == len(set(active)) == 26 and (len(encoder), len(heads), len(contract["frozen_parameter_names"])) == (20, 6, 273)
    assert all(name.startswith("encoder.backbone.encoder.layers.11.") for name in encoder)
    assert not set(active) & set(contract["frozen_parameter_names"])
    assert ["encoder." + name for name in adapter["selected_encoder_parameter_names"]] == encoder
    assert contract["gradient_checks"] == [{"epoch_phase": 1, "actual_trainable_parameter_names": heads}, {"epoch_phase": 11, "actual_trainable_parameter_names": encoder + heads}]
    expected = {"forward": 2, "encoder_forward": 2, "encode": 2, "encoder_windows": 16, "backward": 2, "optimizer_step": 0}
    for kind, count in expected.items():
        for suffix in ("_started", "_returned"):
            assert trace[kind + suffix] == contract["engineering_cost_counts"][kind + suffix] == count
    assert contract["engineering_optimizer_updates"] == 0 and contract["engineering_backward_calls"] == 2
    assert 0 < trace["seconds"] < prepare["seconds"] and contract["peak_allocated_bytes"] == 214993920 and contract["baseline_allocated_bytes"] == 35619328
    resume = check(contract["resume_probe"])
    assert Path(resume["path"]).parent == Path(trace_desc["path"]).parent
    frozen, frozen_desc = read(OUT / "engineering/CSBrain-frozen-smallhead.json")
    old = frozen["contract"]
    assert old["parameters"] == 26386 and len(old["trainable_parameter_names"]) == 6 and len(old["probe_subject_ids"]) == 8
    assert old["engineering_gradient_passes"] == 1 and old["engineering_optimizer_updates"] == 0
    assert old["frozen_basis"] == "hash-bound fixed native feature arrays; external encoder not invoked while fitting"
    source = lambda name, start, end, description: {"source": desc(WORK / "tridep_ccml/eeg_all_candidates" / name), "line": start, "end_line": end, "description": description}
    report = {"kind": "independent_csbrain_raw_finetune_engineering_review", "passed": True,
              "created_at": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/resume_review", "review_source": desc(Path(__file__)),
              "bindings": {"registration": registry_desc, "q3_registration": q3_desc, "engineering_lock": lock_desc, "actual_f2_attempt": trace_desc,
                           "q3_prepare_complete": prepare_desc, "resume_probe": resume, "official_weight": author_weight, "prior_frozen_head_lock": frozen_desc},
              "scope": "Completed CSBrain raw-input FT engineering only; no formal fit/checkpoint inspected or completion inferred",
              "source_checks": {"q3_registered_sources_verified": 47, "engineering_implementation_sources_verified": len(contract["implementation_sources"]),
                                "unique_verified_files": len(verified), "sealed_training_sources_modified": 0},
              "strict_author_load": {"repository": adapter["source"]["repository"], "revision": adapter["source"]["revision"], "loaded_backbone_keys": 295,
                                     "pretrained_parameters_before_projector_removal": 8902000, "retained_parameters": 8861800,
                                     "removed_after_strict_load": adapter["removed_reconstruction_projector_keys"], "fresh_head_seed": 11},
              "actual_input_and_mask": {"raw_shape": [19, 1000], "encoder_shape": [19, 5, 200], "seconds": 5, "hz": 200, "native_dimension": 200,
                                        "fixed_scale_divisor": 100, "probe_role": "fold1 train only", "probe_subject_ids": contract["probe_subject_ids"],
                                        "head_parameters": 6, "lastblock_parameters": 20, "lastblock_prefix": "encoder.backbone.encoder.layers.11.",
                                        "head_only_phase": 1, "head_and_lastblock_phase": 11, "unintended_frozen_gradients_absent_in_actual_source_assertions": True},
              "actual_engineering_calls": expected,
              "resource_measurements": {"f2_attempt_seconds": trace["seconds"], "q3_dispatcher_seconds": prepare["seconds"],
                                        "baseline_allocated_bytes": 35619328, "peak_allocated_bytes": 214993920,
                                        "nested_seconds_summed": False, "prior_frozen_head_unknown_resources_backfilled": False},
              "recovery": {"completed": True, "model_optimizer_rng_scheduler_sampler_saved": True,
                           "same_model_state_and_rng_after_reload_asserted_by_actual_f2_execution": True,
                           "resume_probe_bytes_verified_only_here": True, "checkpoint_deserialized_by_this_review": False},
              "prior_frozen_head_distinction": {"frozen_head_parameters": 6, "frozen_head_scalar_parameters": 26386, "probe_people": 8,
                                               "encoder_calls_in_head_probe": 0, "gradient_passes": 1,
                                               "new_raw_ft_probe_encoder_calls": 2, "new_raw_ft_probe_backwards": 2,
                                               "prior_frozen_head_lock_is_not_raw_finetune_qualification": True},
              "adaptation_limits_retained": author_input["adaptation_differences"] + ["Fixed numeric /100 scaling; physical amplitude calibration unconfirmed", "Author left-ear reference not reproduced"],
              "source_evidence": [source("adapters/csbrain.py", 128, 159, "Strict official loading and equality before removing reconstruction projector"),
                                  source("adapters/csbrain.py", 97, 125, "Only last block trainable; frozen backbone eval; native token mean pooling"),
                                  source("adapters/csbrain.py", 163, 172, "Real raw adaptation and encoder entry"),
                                  source("finetune_v2.py", 198, 251, "Train-only two-phase backward, zero optimizer, state/RNG roundtrip and actual resource record"),
                                  source("finetune.py", 124, 175, "Fresh native head and explicit phase/AdamW parameter groups"),
                                  source("frozen_heads.py", 118, 153, "Earlier cached-feature head-only qualification scope")],
              "this_review_operations": {"model_constructions": 0, "gpu_calls": 0, "tensor_deserializations": 0, "active_formal_checkpoints_read": 0,
                                         "raw_or_feature_arrays_read": 0, "test_probabilities_read": 0, "official_weight_and_engineering_resume_byte_hashes_only": True},
              "formal_fits_completed_by_engineering": 0, "csbrain_15_fits_completion_claimed": False, "full_plan_completion_claimed": False}
    destination = REPORT / ("independent_csbrain_raw_finetune_engineering_review_" + str(time.time_ns()) + ".json")
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"artifact": desc(destination), "passed": True, "q3_sources_verified": 47, "engineering_sources_verified": 62, "actual_backwards": 2, "optimizer_steps": 0}, ensure_ascii=True))


if __name__ == "__main__": main()

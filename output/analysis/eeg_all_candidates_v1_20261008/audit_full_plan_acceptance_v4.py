"""Repeatable wide EEG-plan acceptance audit, without constructing models.

While any of the registered 360 fits is incomplete, this reads only training
JSON/provenance and leaves every outer-test prediction and test gate untouched.
Only after all real training evidence passes can fixed evaluation bundles be
audited. Final costs are independent companion artifacts; E2 outputs stay sealed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import io
import importlib.util
import json
import math
import re
from pathlib import Path
import sys
import time

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import protocol as P, registry as R, storage as S
from tridep_ccml.eeg_all_candidates import execution_acceleration_v1 as X

PREPARATION = S.REPORT / "full_plan_acceptance_preparation_v2_1791454204721868700.json"
SOURCE = Path(__file__).resolve()
DELEGATED_SOURCE = SOURCE.with_name("audit_full_plan_acceptance.py")
PREVIOUS_SOURCE = SOURCE.with_name("audit_full_plan_acceptance_v2.py")
SEMANTIC_REVIEW_IDS = {"R04", "R06", "R10", "R27", "R29"}
ONLINE_PREFIXES = ("core.InputEmbedding.", "core.Norm.", "core.Norm2.", "core.contex_encoder.")


def require(value, message):
    if not value:
        raise ValueError(message)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def value_alias(data, names):
    values = [data[name] for name in names if name in data]
    require(values and all(value == values[0] for value in values), "Missing/conflicting actual aliases: " + "/".join(names))
    return values[0]


def hash_shape(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def bag_indices(epoch):
    start = ((epoch - 1) % 8) * 8
    return [(start + index) % 60 for index in range(8)]


def policy_curve(policy, history, validation, metadata, evidence):
    epochs = metadata["epochs"]
    require(isinstance(epochs, int) and epochs > 0 and len(history) == len(validation) == epochs, "Incomplete per-epoch curves")
    losses, extended, last = [], False, None
    key = "val_ssl_loss" if policy == "ssl" else "val_eval_ce"
    for epoch, (row, val) in enumerate(zip(history, validation), 1):
        require(row["epoch"] == val["epoch"] == epoch and number(row[key]), "Curve epoch/loss invalid")
        require(val["val_ssl_loss" if policy == "ssl" else "val_ce"] == row[key], "Validation history is not the actual training diagnostic")
        losses.append(row[key])
        last = P.training_decision(policy, losses, extended=extended)
        require(not last["stop"] or epoch == epochs, "Training continued after registered stopping condition")
        if last["extend"]:
            extended = True
    require(last["stop"] is True, "Final curve has not reached a registered stopping condition")
    if policy == "diagnostic":
        require(epochs == 300 and evidence["optimizer_updates"] == 600 and metadata["early_stopping"] is False,
                "Paired diagnostic is not300 epochs/600 actual updates")
        require(metadata["checkpoint_selection"] == "fixed_epoch_300" and metadata["stopping_reason"] == "fixed_300_epoch_diagnostic", "Diagnostic selection/stop differs")
    else:
        require(metadata["best_epoch"] == last["best_epoch"] and metadata["extended"] is extended,
                "Selected validation epoch or extension differs from full registered curve")
        require(value_alias(metadata, ("stop_reason", "stopping_reason")) == evidence["stop_reason"] == last["reason"], "Actual registered stop reason differs")
    return last


def legacy_check_fit(fit, status, registration, lock, contract):
    """Actual complete-file schemas and mathematical history checks, no tensors."""
    folder = R.fit_directory(fit["condition"], fit["seed"], fit["fold"])
    evidence = status["evidence"]
    R._validate_complete(evidence, folder)
    require(status["condition_source_lock"] == lock and S.read(folder / "complete.json") == evidence, "Completion/source/event evidence differs")
    metadata = S.read(S.checked(evidence["metadata"]))
    history = S.read(S.checked(evidence["train_history"]))
    validation = S.read(S.checked(evidence["validation_history"]))
    condition, seed, fold = fit["condition"], fit["seed"], fit["fold"]
    split = registration["folds"][f"fold_{fold}"]
    specification = P.CONDITION_BY_ID[condition]
    policy = specification.policy_key
    require(all(metadata[name] == expected for name, expected in {"condition": condition, "seed": seed, "fold": fold,
            "train_ids": split["train"], "val_ids": split["val"]}.items()), "Actual fit identity/train/validation people differ")
    if "test_ids" in metadata:
        require(metadata["test_ids"] == split["test"], "Declared outer-test roles differ")
    require(metadata["test_forward_count"] == 0, "Training metadata reports an outer-test forward")
    require(metadata["epochs"] == evidence["epochs"], "Completion epoch differs")
    updates = value_alias(metadata, ("updates", "optimizer_updates"))
    require(updates == evidence["optimizer_updates"] > 0, "Actual optimizer count differs")
    decision = policy_curve(policy, history, validation, metadata, evidence)
    train_count = len(split["train"])
    for row in history:
        epoch = row["epoch"]
        row_updates = value_alias(row, ("updates_completed", "optimizer_updates"))
        if policy == "ssl":
            sizes = row["logical_batch_sizes"]
            require(sizes and all(isinstance(n, int) and 2 <= n <= 64 for n in sizes)
                    and all(n == 64 for n in sizes[:-1]) and sum(sizes) == train_count * 60,
                    "SSL actual logical-batch coverage/size differs")
            require(row["train_windows"] == train_count * 60 and len(row["masks"]) == len(sizes), "SSL balanced window/mask count differs")
            steps = len(sizes)
            require(row["ema_updates_completed"] == row["gradient_passes_completed"] == row_updates == epoch * steps, "SSL optimizer/EMA/gradient counts differ")
            require(all(number(row[group][key]) for group in ("train_components", "val_components")
                        for key in ("loss", "alignment_mse", "variance_hinge", "offdiagonal_covariance", "context_feature_std_mean", "prediction_feature_std_mean")), "SSL components/noncollapse evidence invalid")
        else:
            steps = 2 if policy == "diagnostic" else math.ceil(train_count / 8)
            require(row_updates == epoch * steps, "Actual complete train traversal/update count differs")
            if policy in ("scratch", "ssl_supervised", "finetune"):
                micro = metadata["micro_subjects"]
                require(micro in (1, 2) and row["window_indices"] == bag_indices(epoch), "Raw8-window cycle/microbatch differs")
                expected_backwards = epoch * ((train_count // 8) * math.ceil(8 / micro) + math.ceil((train_count % 8) / micro))
                require(row["backward_calls"] == expected_backwards and row["cycle_epoch"] == (epoch - 1) % 8 + 1,
                        "Actual CE backward/coverage cycle differs")
                require(row["repeated_padding_windows_this_epoch"] == ([0, 1, 2, 3] if epoch % 8 == 0 else []), "Cyclic4 padding differs")
                if policy == "finetune":
                    require(row["unfrozen"] is (epoch > 10), "FT head warmup phase differs")
                else:
                    require(row["supervised_ema_updates"] == 0, "Supervised target EMA was updated")
                require(row["head_lr"] == P.POLICIES[policy].head_lr * P.learning_rate_multiplier(policy, epoch), "Actual head learning-rate schedule differs")
                require(row["backbone_lr"] == P.POLICIES[policy].backbone_lr * P.learning_rate_multiplier(policy, epoch, parameter_group="backbone"), "Actual backbone learning-rate schedule differs")
            elif policy != "diagnostic":
                require(row["lr"] == P.POLICIES[policy].head_lr * P.learning_rate_multiplier(policy, epoch), "Actual learning-rate schedule differs")
    require(history[-1]["updates_completed" if "updates_completed" in history[-1] else "optimizer_updates"] == updates, "Terminal history counter differs")
    if policy == "diagnostic":
        replay = S.read(folder / "first60_replay_verification.json")
        require(all(replay[name] is True for name in ("passed", "exact_initial_state", "exact_all_120_batch_losses", "exact_batch_order", "exact_epoch60_state", "exact_optimizer_state")), "Historical60 strict replay not established")
        S.checked(replay["historical_checkpoint"])
        require(metadata["first60_replay_exact"] is True and metadata["optimizer_rng_isolated_validation"] is True
                and metadata["encoder_forwards"] == 0 and set(metadata["snapshots"]) == {str(e) for e in P.SNAPSHOT_EPOCHS}, "Fixed paired/RNG/frozen/snapshot evidence differs")
        for descriptor in metadata["snapshots"].values(): S.checked(descriptor)
    elif policy == "frozen":
        manifest = S.read(S.checked(metadata["feature_manifest"]))
        require(manifest["subject_ids"] == registration["subject_ids"] and manifest["encoder_fit_subject_ids"] == [] and manifest["dimension"] == metadata["dimension"], "Frozen manifest scope/native dimension differs")
        require(metadata["parameters"] == 130 * metadata["dimension"] + 386 and metadata["new_encoder_forwards"] == 0
                and metadata["selected_state_sha256"] != metadata["initial_state_sha256"], "Native small head/frozen/actual-update evidence differs")
    elif policy == "ssl":
        require(metadata["encoder_fit_subject_ids"] == split["train"] and metadata["actual_logical_batch"] == 64
                and metadata["precision"] == "bfloat16" and metadata["loss_precision"] == "float32"
                and metadata["loss_autocast_enabled"] is False and metadata["validation_updates"] == 0,
                "Fold-local64/BF16/FP32-loss/no-validation-update contract differs")
        require(metadata["ema_updates"] == updates and metadata["online_backbone_update_verified"] is True
                and metadata["online_changed_parameter_count"] == len(set(metadata["online_changed_parameter_names"])) > 0
                and all(name.startswith(ONLINE_PREFIXES) for name in metadata["online_changed_parameter_names"])
                and metadata["selected_online_state_sha256"] != metadata["initial_online_state_sha256"], "Online encoder learned proof missing")
        require(number(metadata["noncollapse_validation_context_std_mean"]) and metadata["noncollapse_validation_context_std_mean"] > 0,
                "Selected validation context variance is not positive")
        masks = S.read(S.checked(metadata["validation_mask_manifest"]))
        require(masks is not None and metadata["ssl_precision_repair"]["version"] == 2, "Fixed validation masks/precision source missing")
        S.checked(metadata["source"]); S.checked(metadata["settings"]); S.checked(metadata["ssl_precision_repair"]["wrapper_source"])
    else:
        require(metadata["window_bag_size"] == 8 and value_alias(metadata, ("effective_subject_batch", "effective_subjects")) == 8
                and value_alias(metadata, ("eval_window_count", "eval_windows")) == 60, "Raw subject8/all60 contract differs")
        changed = metadata["parameter_changed_after_training"]
        require(changed and set(changed) == set(contract["trainable_parameter_names"]) and all(value is True for value in changed.values()), "Intended trained parameter coverage/change is incomplete")
        require(metadata["frozen_initial_sha256"] == metadata["frozen_final_sha256"], "Frozen state changed")
        require(set(metadata["raw_sources"]) == set(split["train"] + split["val"]), "Raw trainer included a test person or omitted train/val source")
        if policy == "scratch":
            bound = S.read(S.checked(metadata["statistics"]))
            require(bound["statistics"]["fit_subject_ids"] == split["train"] and bound["validation_or_test_fit_people"] == [], "Conformer scalar statistics fitted outside train people")
            require(metadata["ssl_binding"] is None and metadata["batchnorm_microbatch_statistics_not_true_batch8"] is True, "Scratch/BN adaptation declaration differs")
        elif policy == "ssl_supervised":
            binding = metadata["ssl_binding"]
            dependency = R.fit_directory("EEG2Rep-fold-local-ssl", seed, fold)
            ssl_status = R.fit_status("EEG2Rep-fold-local-ssl", seed, fold)
            require(ssl_status["status"] == "training_complete" and binding["checkpoint"] == ssl_status["evidence"]["checkpoint"]
                    and S.checked(binding["checkpoint"]).parent == dependency, "Same seed/fold actual selected SSL checkpoint differs")
            ssl_meta = S.read(S.checked(binding["metadata"]))
            require(ssl_meta["train_ids"] == split["train"] and ssl_meta["val_ids"] == split["val"]
                    and ssl_meta["selected_state_sha256"] == binding["selected_state_sha256"] and metadata["statistics"] is None
                    and metadata["supervised_ema_updates"] == 0, "SSL binding/source/normalization/target freeze differs")
        else:
            from tridep_ccml.eeg_all_candidates import evaluate_v2 as E2
            E2.selection_evidence(metadata)
            require(metadata["fresh_official_pretrained"] is True and metadata["fresh_head"] is True and metadata["frozen_best_head_inherited"] is False,
                    "FT inherited another validation-selected head/encoder")
            source = metadata["versioned_finetune_implementation"]
            require(source["version"] == 2 and source["wrapper_source"] == S.desc(P.WORK / "tridep_ccml/eeg_all_candidates/finetune_v2.py"), "Actual FTv2 implementation source differs")
            if specification.method == "LUNA":
                names = metadata["adapter_audit"]["selected_encoder_parameter_names"]
                require(len(names) == 17 and not any(name.endswith("rotary_emb.freqs") for name in names), "Author fixed LUNA frequency was trained")
    require(number(metadata["seconds_this_attempt"]), "Actual attempt elapsed time missing")
    return {"fit_id": fit["fit_id"], "condition": condition, "seed": seed, "fold": fold, "policy": policy,
            "epochs": metadata["epochs"], "updates": updates, "best_epoch": metadata.get("best_epoch", 300),
            "stop": decision["reason"], "metadata": evidence["metadata"], "completion": S.desc(folder / "complete.json"),
            "condition_source_lock": lock, "training_json_verified": True}


def check_fit(fit, status, registration, lock, contract):
    """Preserve sealed math checks and separately bind actual FT execution."""
    row = legacy_check_fit(fit, status, registration, lock, contract)
    metadata = S.read(S.checked(row["metadata"]))
    if P.CONDITION_BY_ID[fit["condition"]].policy_key == "finetune":
        path = SOURCE.with_name("audit_completed_finetune_checkpoints_v4.py")
        spec = importlib.util.spec_from_file_location("wide_v4_ft_observer", path)
        auditor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(auditor)
        inputs = auditor.trace_inventory(fit)
        proof = auditor.trace_boundary(fit, metadata, inputs)
        auditor.recheck_trace_inventory(fit, inputs)
        row["actual_execution_observer_proof"] = proof
        row["execution_auditor_source"] = S.desc(path)
        if "execution_acceleration" in metadata:
            row["execution_acceleration_verified"] = X.verify_execution_acceleration(metadata, condition=fit["condition"], seed=fit["seed"], fold=fit["fold"])
    else:
        require("execution_acceleration" not in metadata, "FT execution acceleration appeared on a non-FT fit")
    return row


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def deep_checkpoint(fit, row, contract, metadata):
    """Future-only CPU tensor audit; never instantiate an encoder/head."""
    import torch
    folder = R.fit_directory(fit["condition"], fit["seed"], fit["fold"])
    policy = P.CONDITION_BY_ID[fit["condition"]].policy_key
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=False)
    terminal = torch.load(folder / "latest.pt", map_location="cpu", weights_only=False)
    selected = torch.load(S.checked(R.fit_status(fit["condition"], fit["seed"], fit["fold"])["evidence"]["checkpoint"]), map_location="cpu", weights_only=False)
    require(initial["epoch"] == 0 and terminal["epoch"] == row["epochs"] and terminal.get("rng") and terminal.get("sampler_position"), "Initializer/latest epoch/RNG/sampler incomplete")
    require(value_alias(selected, ("epoch", "best_epoch")) == row["best_epoch"], "Actual selected checkpoint epoch differs")
    require(set(terminal["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
            and terminal["rng"]["torch_cuda"], "Formal CUDA fit must retain every RNG family")
    sampler = {"epoch_completed": row["epochs"], "next_batch": 0}
    if policy in ("ssl", "diagnostic"): sampler["next_epoch"] = row["epochs"] + 1
    require(terminal["sampler_position"] == sampler, "Terminal sampler boundary differs")
    expected_scheduler = None if policy == "diagnostic" else {"policy": policy, "completed_epoch": row["epochs"], "extended": metadata["extended"]}
    require(terminal["scheduler_state_dict"] == expected_scheduler, "Terminal scheduler state differs")
    require(terminal["history"] == S.read(folder / "train_history.json"), "Terminal history and complete-file history differ")
    require(state_digest(initial["state_dict"]) == metadata["initial_state_sha256"], "Actual initializer hash differs")
    signature = metadata["final_state_sha256"] if policy == "diagnostic" else metadata["selected_state_sha256"]
    require(state_digest(selected["state_dict"]) == signature and all(bool(torch.isfinite(value).all()) for value in selected["state_dict"].values()), "Actual selected tensor state/hash differs")
    opt = terminal["optimizer_state_dict"]
    groups, states = opt["param_groups"], opt["state"]
    ids = [identifier for group in groups for identifier in group["params"]]
    require(ids and len(ids) == len(set(ids)) == len(contract["trainable_parameter_names"]) and set(states) == set(ids),
            "Optimizer coverage is empty/incomplete or contains unintended state")
    for group in groups:
        require(group["weight_decay"] == P.POLICIES[policy].weight_decay and group["betas"] == (.9, .999)
                and group["eps"] == 1e-8, "Actual optimizer hyperparameters differ from registered/default values")
        name = group.get("group_name", "head")
        base_lr = P.POLICIES[policy].backbone_lr if name == "backbone" else P.POLICIES[policy].head_lr
        expected_lr = base_lr * P.learning_rate_multiplier(policy, row["epochs"], parameter_group=name)
        require(group["lr"] == expected_lr, "Terminal actual optimizer LR differs")
        expected = row["updates"]
        if policy == "finetune" and group.get("group_name") == "backbone":
            expected = math.ceil(len(fit["train_ids"]) / 8) * (row["epochs"] - 10)
        require(group["params"] and expected > 0, "Vacuous parameter group/update verification")
        for identifier in group["params"]:
            state = states[identifier]
            require(float(state["step"]) == expected and all(bool(torch.isfinite(value).all()) for value in state.values() if isinstance(value, torch.Tensor)), "Optimizer actual step/moment differs")
    if policy == "diagnostic":
        for epoch in P.SNAPSHOT_EPOCHS:
            snapshot = torch.load(S.checked(metadata["snapshots"][str(epoch)]), map_location="cpu", weights_only=False)
            require(snapshot["epoch"] == epoch and snapshot["registry"] == R.registration_descriptor() and snapshot["rng"], "Fixed diagnostic snapshot identity/RNG differs")
        replay = S.read(folder / "first60_replay_verification.json")
        old = torch.load(S.checked(replay["historical_checkpoint"]), map_location="cpu", weights_only=False)
        replay60 = torch.load(S.checked(metadata["snapshots"]["60"]), map_location="cpu", weights_only=False)
        from tridep_ccml.tridep_eegpt.recovery_io import _same
        require(_same(old["state_dict"], replay60["state_dict"]) and _same(old["optimizer_state_dict"], replay60["optimizer_state_dict"]), "Actual historical60 tensors/Adamax differ")
    elif policy == "ssl":
        online = lambda state: {name: value for name, value in state.items() if name.startswith(ONLINE_PREFIXES)}
        require(state_digest(online(initial["state_dict"])) == metadata["initial_online_state_sha256"]
                and state_digest(online(selected["state_dict"])) == metadata["selected_online_state_sha256"], "Actual selected online encoder hash differs")
        changed = [name for name, value in online(initial["state_dict"]).items() if not torch.equal(value, selected["state_dict"][name])]
        require(set(changed) == set(metadata["online_changed_parameter_names"]) and changed, "Actual SSL online changed names differ")
        require(selected["epoch"] == row["best_epoch"] and selected["stage"] == "ssl" and selected["optimizer_updates"] == selected["ema_updates"], "Selected SSL checkpoint phase/count differs")
    elif policy in ("scratch", "ssl_supervised", "finetune"):
        active = set(contract["trainable_parameter_names"])
        mutable = set(metadata.get("active_running_stat_buffer_keys", []))
        frozen = {name for name in initial["state_dict"] if name not in active | mutable}
        require(all(torch.equal(initial["state_dict"][name], terminal["state_dict"][name]) and torch.equal(initial["state_dict"][name], selected["state_dict"][name]) for name in frozen), "Actual frozen tensors differ")
        changed = {name for name in active if not torch.equal(initial["state_dict"][name], terminal["state_dict"][name])}
        require(changed == active == {name for name, value in metadata["parameter_changed_after_training"].items() if value}, "Actual intended terminal parameter updates differ")
        if policy == "finetune":
            names = [name for name in initial["state_dict"] if name.startswith("encoder.")]
            selected_changed = [name for name in names if not torch.equal(initial["state_dict"][name], selected["state_dict"][name])]
            require(set(selected_changed) == set(metadata["selected_encoder_changed_parameter_names"])
                    and state_digest({name: selected["state_dict"][name] for name in names}) == metadata["selected_encoder_state_sha256"]
                    and state_digest({name: terminal["state_dict"][name] for name in names}) == metadata["terminal_encoder_state_sha256"], "Actual selected/terminal FT encoder proof differs")
    return {"fit_id": fit["fit_id"], "cpu_tensors_verified": True, "model_constructions": 0}


def evaluation_audit(registration):
    """One real full360 gate, then actual cached E2 proofs and report arithmetic."""
    from tridep_ccml.eeg_all_candidates import evaluate_v3 as E2
    driver = E2.isolated_driver()
    checked = R.assert_test_unlocked()
    stable = {key: checked[key] for key in ("passed", "registration", "formal_fits_verified", "statuses")}
    existing_gate = E2.ROOT / ("test_gate_" + driver.canonical(stable) + ".json")
    require(existing_gate.exists() and S.read(existing_gate) == stable,
            "Actual completed evaluation's source-bound gate must already exist; audit cannot create it")
    gate = driver._gate()  # Never reachable from a current pending training run.
    root = E2.ROOT
    complete = S.read(root / "complete.json")
    require(complete["passed"] is True and complete["test_gate"] == gate
            and (complete["formal_fits_verified"], complete["supervised_conditions"], complete["supervised_fits_evaluated"], complete["evaluated_checkpoints"], complete["candidate_papers_count"]) == (360, 23, 345, 435, 9), "Actual global evaluation completion arithmetic differs")
    E2.verify_execution(complete["evaluator_execution"])
    report = S.read(S.checked(complete["evaluation_report"]))
    E2.verify_execution(report["evaluator_execution"])
    require(report["test_gate"] == gate and set(report["conditions"]) == set(driver.CONDITIONS), "Evaluation report gate/condition set differs")
    cohort = {"subject_ids": registration["subject_ids"]}
    entries = []
    for condition in driver.CONDITIONS:
        snapshots = P.SNAPSHOT_EPOCHS if P.CONDITION_BY_ID[condition].policy_key == "diagnostic" else ("best",)
        for seed in P.SEEDS:
            for fold in P.FOLDS:
                split = registration["folds"][f"fold_{fold}"]
                for snapshot in snapshots:
                    _, _, binding, destination = driver._fit_context(condition, seed, fold, snapshot, gate, cohort, split, registration)
                    result, rows = driver._verified_completed_fit(destination / "complete.json", binding, split, registration["labels"])
                    entries.append({"condition": condition, "seed": seed, "fold": fold, "snapshot": snapshot,
                                    "result": result, "rows": rows, "path": destination / "complete.json"})
    require(len(entries) == len(report["probability_batch_artifacts"]) == 435, "Actual fixed-checkpoint bundle set is incomplete")
    require({tuple(record.items()) for record in report["probability_batch_artifacts"]} == {tuple(S.desc(e["path"]).items()) for e in entries}, "Report bundle hashes/set differ")
    summaries = {}
    for condition in driver.CONDITIONS:
        observed = report["conditions"][condition]
        variants = {}
        for snapshot in (P.SNAPSHOT_EPOCHS if P.CONDITION_BY_ID[condition].policy_key == "diagnostic" else ("best",)):
            variants[str(snapshot)] = driver.condition_summary([e for e in entries if e["condition"] == condition and e["snapshot"] == snapshot], registration["subject_ids"])
        primary = "300" if "300" in variants else "best"
        require(observed["variants"] == variants and observed["primary_snapshot"] == primary, "Actual report/37-person OOF/111 repeats/fold-role metrics differ")
        summaries[condition] = (variants, primary)
    for condition, (variants, primary) in summaries.items():
        observed = report["conditions"][condition]
        control = driver.matched_control(condition)
        retention = driver.retention(variants[primary], summaries[control][0][summaries[control][1]]) if control else None
        require(observed["matched_control"] == control and observed["engineering_retention"] == retention, "Fixed matched-control/retention criterion differs")
        if primary == "300":
            duration = observed["predefined_duration_comparison"]
            require(duration["test_best_snapshot_selected"] is False and duration["secondary_snapshots"] == [120, 200]
                    and duration["deltas"] == {key: variants["300"]["primary"][key] - variants["60"]["primary"][key] for key in variants["300"]["primary"]}, "Prespecified300-minus60 comparison differs")
    require(set(report["nine_papers"]) == set(P.METHODS), "Nine candidate conclusion set differs")
    for method, paper in report["nine_papers"].items():
        routes = [c.condition_id for c in P.CONDITIONS if c.method == method and c.objective == "supervised" and c.policy_key != "diagnostic"]
        passing = [condition for condition in routes if report["conditions"][condition]["engineering_retention"]["passed"]]
        require(paper["routes"] == routes and paper["passing_routes"] == passing and paper["all_prespecified_routes_completed"] is True
                and paper["negative_results_count_as_completed"] is True and paper["status"] == ("改善" if passing else "未改善"), "A candidate route/negative result/conclusion was omitted or reselected")
    require(report["threshold"] == .5 and report["tie"] == "HC" and report["distinct_people"] == 37
            and report["seed_repeats_not_new_people"] is True and report["test_guided_model_or_threshold_selection"] is False, "Fixed threshold/cohort scope differs")
    comparison = S.checked(complete["comparison_table"])
    with comparison.open(encoding="utf-8-sig", newline="") as stream: table = list(csv.DictReader(stream))
    require(len(table) == 23 and {row["condition"] for row in table} == set(driver.CONDITIONS), "Unified comparison table is not all23 conditions")
    oof_path = S.checked(complete["oof_probabilities"])
    with oof_path.open(encoding="utf-8-sig", newline="") as stream: oof = list(csv.DictReader(stream))
    require(len(oof) == sum(len(P.SNAPSHOT_EPOCHS) if P.CONDITION_BY_ID[c].policy_key == "diagnostic" else 1 for c in driver.CONDITIONS) * 111, "All fixed-snapshot OOF rows were not delivered")
    verify_exported_tables(driver, report, entries, complete)
    S.checked(complete["nine_paper_report"])
    return {"passed": True, "complete": S.desc(root / "complete.json"), "gate": gate,
            "checkpoint_bundles": 435, "conditions": 23, "supervised_fits": 345, "oof_distinct_people": 37,
            "oof_repeats_per_primary_condition": 111, "model_constructions": 0, "gpu_forwards": 0,
            "report_and_retention_recomputed_from_real_bound_probabilities": True}


def cost_recovery_audit(data, companion):
    """Validate the acknowledged pause cost independently of logical fit totals."""
    event_path = S.OUT / "resume_events/1791506856841705600.json"
    event = S.read(event_path)
    descriptor = S.desc(event_path)
    records = data["cost_records"]
    known = [row for row in data["acknowledged_checkpoint_external_recovery_work"] if row["artifact"] == descriptor]
    require(len(known) == 1, "Known pause recovery boundary must occur exactly once")
    row = known[0]
    require(row["boundary_verified"] is True and row["duplicate_checkpoint_boundary"] is False
            and row["boundary_arithmetic_consistent"] is True and row["preserved_history_boundary_matches"] is True
            and row["preserved_progress_boundary_matches"] is True, "Known recovery boundary was not verified")
    observed = row["reported_boundary_counters"]
    require(all(observed[key] == event[key] for key in observed), "Recovery summary changed original acknowledged counters")
    require((event["condition"], event["seed"], event["fold"], event["durable_epoch"], event["paused_epoch"]) ==
            ("EEG2Rep-supervised", 22, 4, 26, 27), "Known pause fit/epoch identity differs")
    require(event["paused_acknowledged_optimizer_updates"] - event["durable_optimizer_updates"] ==
            event["acknowledged_updates_not_in_checkpoint"] == row["acknowledged_optimizer_updates_outside_checkpoint"] == 4,
            "Acknowledged four updates outside checkpoint were lost or redefined")
    require(event["paused_acknowledged_backward_calls"] - event["durable_backward_calls"] ==
            event["acknowledged_backward_calls_not_in_checkpoint"] == row["acknowledged_backward_calls_outside_checkpoint"] == 27,
            "Acknowledged27 backward calls outside checkpoint were lost or redefined")
    preserved = {Path(item["preserved_copy"]["path"]).name: item["preserved_copy"] for item in event["preserved_artifacts"]}
    history = S.read(S.checked(preserved["paused_fit_train_history.json"]))
    progress = S.read(S.checked(preserved["paused_fit_progress.json"]))
    require((history[-1]["epoch"], history[-1]["optimizer_updates"], history[-1]["backward_calls"]) == (26, 104, 702)
            and (progress["epoch"], progress["optimizer_updates_completed"], progress["backward_calls_completed"], progress["last_durable_epoch"]) == (27, 108, 729, 26),
            "Preserved actual history/progress do not support4/27 extra physical work")
    actual_bindings = {item["expected"]["path"]: item for item in row["preserved_json_bindings"]}
    for name in ("paused_fit_train_history.json", "paused_fit_progress.json"):
        binding = actual_bindings[preserved[name]["path"]]
        require(binding["expected"] == binding["actual"] == preserved[name] and binding["matches"] is True,
                "Cost snapshot used unbound recovery JSON copies")
    require(row["overnight_suspension_in_training_time"] is False
            and row["prior_partial_attempt_model_seconds"] is None and row["uncheckpointed_diagnostic_forward_count"] is None,
            "Unrecorded pause diagnostics/time must remain unknown; overnight suspension is not training")
    matching_records = [r for r in records if r["category"] == "recovery_event" and r["artifact"] == descriptor]
    require(len(matching_records) == 1 and matching_records[0].get("recovery_extra_cost") is not None,
            "Known recovery work is absent from actual cost records")
    unique, keys = [], set()
    for item in data["acknowledged_checkpoint_external_recovery_work"]:
        if not item["boundary_verified"]: continue
        key = json.dumps(item["recovery_identity"], sort_keys=True)
        require(item["duplicate_checkpoint_boundary"] is (key in keys), "Recovery boundary de-duplication differs")
        if key not in keys: unique.append(item)
        keys.add(key)
    require(data["recovery_acknowledged_optimizer_updates_known_subtotal"] == sum(r["acknowledged_optimizer_updates_outside_checkpoint"] for r in unique)
            and data["recovery_acknowledged_backward_calls_known_subtotal"] == sum(r["acknowledged_backward_calls_outside_checkpoint"] for r in unique),
            "Recovery known subtotals lost or double-counted a checkpoint boundary")
    repair_path = S.OUT / "assets/eegmamba_dependency_repair/ninja_v1/installation_1791442246509965726.json"
    repairs = [r for r in records if r["category"] == "runtime_environment_repair" and r["artifact"] == S.desc(repair_path)]
    require(len(repairs) == 1 and repairs[0].get("runtime_setup_evidence"), "Actual isolated runtime repair is missing")
    require(repairs[0]["seconds_observed"] is None and repairs[0]["runtime_setup_evidence"]["duration_seconds"] is None,
            "Missing dependency-install duration must not be converted to zero")
    if data["unknown_or_opaque_records"]:
        require(data["all_plan_physical_optimizer_returns_exact"] is None and data["all_plan_wall_seconds_exact"] is None,
                "Opaque historical records cannot support an invented exact physical total")
    text = Path(companion).read_text(encoding="utf-8-sig").lower()
    # The root agent writes this prose after inspecting these records. This is
    # a presence check, not a claim to infer prose semantics automatically.
    require(re.search(r"(?<!\d)4\s*(次)?\s*(优化器|optimizer)", text)
            and re.search(r"(?<!\d)27\s*(次)?\s*(反向|backward)", text)
            and any(term in text for term in ("检查点", "checkpoint"))
            and any(term in text for term in ("未知", "unknown"))
            and any(term in text for term in ("暂停", "suspension")),
            "Readable cost report must explain the4 updates/27 backwards beyond checkpoint, unknowns and suspension")
    return {"passed": True, "known_recovery_event": descriptor, "extra_physical_optimizer_updates": 4,
            "extra_physical_backward_calls": 27, "known_runtime_repair": S.desc(repair_path),
            "extra_cost_separate_from_durable_fit_counters": True,
            "historical_unknowns_preserved": True, "overnight_suspension_not_training_time": True}


def final_cost_audit(path, companion, training_rows):
    require(path is not None and Path(path).exists(), "Independent final full-scope cost snapshot pending")
    data = S.read(path)
    require(data.get("schema_version") == 5 and data.get("execution_observer_verification", {}).get("passed_for_observed_stable_f3_records") is True
            and not data["execution_observer_verification"]["errors"], "Final cost must verify the actual F3 buffered observers")
    require(data.get("source") == S.desc(SOURCE.with_name("cost_aggregation_v5.py")), "Final cost execution source differs")
    require(data.get("draft_not_completion_certificate") is False and data.get("final_snapshot") is True,
            "A cost draft/current partial snapshot is not final cost delivery")
    require(data["formal_fits_expected"] == 360 and data["formal_status_counts"] == {"training_complete": 360}, "Cost snapshot is not all360 real training completions")
    require({row["fit_id"] for row in data["formal_fits"]} == {row["fit_id"] for row in training_rows}
            and len(data["formal_fits"]) == 360, "Final cost fit identity set differs")
    updates = {row["fit_id"]: row["updates"] for row in training_rows}
    require(all(row["formal_durable_updates"] == updates[row["fit_id"]] for row in data["formal_fits"]), "Final cost durable logical counters differ")
    require(data["formal_durable_updates_known_subtotal"] == sum(updates.values()) and not data["input_read_errors"], "Known formal cost subtotal/read coverage differs")
    for descriptor in data["inputs"]: S.checked(descriptor)
    require(data.get("all_available_artifacts_accounted") is True and data.get("coverage_gaps_disclosed") is True,
            "Final engineering/export/failure/recovery/evaluation cost coverage and limits need explicit review")
    categories = {row["category"] for row in data["cost_records"]}
    require({"formal_legacy_metadata", "feature_export", "evaluation", "recovery_event", "runtime_environment_repair"}.issubset(categories), "An actual cost phase is missing")
    require(categories & {"cpu_or_synthetic_test_audit", "mixed_engineering_test_audit", "test_audit_runtime_unknown"}, "Content-classified test cost evidence is missing")
    require(data.get("unknown_or_opaque_records") is not None and data.get("rules"), "Unknown history/overlap boundaries were not disclosed")
    # Irretrievable old physical totals/elapsed may legitimately remain null.
    # Full available-record coverage and truthful limits, not invented zeros,
    # are the acceptance requirement.
    if companion is None and data.get("human_readable_cost_report"):
        companion = S.checked(data["human_readable_cost_report"])
    require(companion is not None and Path(companion).exists() and Path(companion).stat().st_size > 0,
            "Independent human-readable full-scope cost companion pending")
    recovery = cost_recovery_audit(data, companion)
    return {"passed": True, "snapshot": S.desc(path), "companion": S.desc(companion), "recovery_and_environment": recovery,
            "physical_all_plan_exact_total": data.get("all_plan_physical_optimizer_returns_exact"),
            "historical_unknowns_are_allowed_when_explicitly_covered_and_disclosed": True}


def verify_registered_identity(registration):
    """R02/R03 are identity checks, not statements that any training completed."""
    from dataclasses import asdict
    cohort, folds, labels, sources = R._cohort_context()  # Only JSON and hashes.
    require(registration["subject_ids"] == cohort["subject_ids"] and registration["folds"] == folds
            and registration["labels"] == labels, "Actual original cohort/folds/labels differ")
    require(all(registration["sources"][key] == value for key, value in sources.items()), "Cohort source identities differ")
    require(len(registration["fits"]) == 360 and len({f["fit_id"] for f in registration["fits"]}) == 360,
            "Complete unique immutable 360 identity set required")
    for fit in registration["fits"]:
        spec = P.CONDITION_BY_ID[fit["condition"]]
        split = folds[f"fold_{fit['fold']}"]
        expected = {"fit_id": R.fit_id(spec.condition_id, fit["seed"], fit["fold"]),
                    "method": spec.method, "stage": spec.stage, "objective": spec.objective,
                    "policy": asdict(P.POLICIES[spec.policy_key]),
                    "dependency_fit_id": R.fit_id(spec.dependency, fit["seed"], fit["fold"]) if spec.dependency else None,
                    "train_ids": split["train"], "val_ids": split["val"], "test_ids": split["test"]}
        require(all(fit[key] == value for key, value in expected.items()), "A registered fit identity/role/policy/dependency differs")
    require(Counter(f["objective"] for f in registration["fits"]) == {"supervised": 345, "self_supervised": 15},
            "Supervised/SSL identity arithmetic differs")
    raw = S.read(S.checked(registration["sources"]["accepted_raw_manifest"]))
    require(raw["subject_ids"] == cohort["subject_ids"] and raw["folds"] == folds
            and raw["channel_names"] == list(P.CHANNELS)
            and registration["raw_sources"] == [{"subject_id": r["subject_id"], "raw": r["raw"]} for r in raw["records"]]
            and all(r["label"] == labels[r["subject_id"]] for r in raw["records"]), "Registered raw manifest identity differs")


def csv_cells(rows):
    """Apply the same standard CSV scalar serialization without writing files."""
    require(bool(rows), "Cannot verify an empty deliverable")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    stream.seek(0)
    reader = csv.DictReader(stream)
    return reader.fieldnames, list(reader)


def verify_csv(path, expected):
    fields, cells = csv_cells(expected)
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        actual = list(reader)
        require(reader.fieldnames == fields and actual == cells,
                "Delivered CSV cells/order differ from bound probability/checkpoint evidence: " + Path(path).name)


def verify_exported_tables(driver, report, entries, complete):
    """Check every delivered CSV cell, including FT selected/terminal disclosure."""
    comparison, oof = [], []
    for condition in driver.CONDITIONS:
        spec = P.CONDITION_BY_ID[condition]
        summary = report["conditions"][condition]
        primary = summary["variants"][summary["primary_snapshot"]]
        criterion = summary["engineering_retention"]
        comparison.append({"condition": condition, "method": spec.method, "route": spec.route,
                           "seconds_per_window": spec.input_seconds, "primary_snapshot": summary["primary_snapshot"],
                           "matched_control": summary["matched_control"] or "",
                           **driver._comparison_contract(condition, entries), **primary["primary"],
                           "macro_f1_delta": criterion["deltas"]["macro_f1"] if criterion else "",
                           "retention_passed": criterion["passed"] if criterion else "",
                           "probability_benefit": criterion["probability_benefit"] if criterion else ""})
        for snapshot in P.SNAPSHOT_EPOCHS if spec.policy_key == "diagnostic" else ("best",):
            for entry in entries:
                if entry["condition"] == condition and entry["snapshot"] == snapshot:
                    oof.extend({"condition": condition, "snapshot": snapshot, "seed": entry["seed"], "fold": entry["fold"],
                                **row, "logits": json.dumps(row["logits"])} for row in entry["rows"] if row["role"] == "test")
    verify_csv(S.checked(complete["comparison_table"]), comparison)
    verify_csv(S.checked(complete["oof_probabilities"]), oof)
    text = S.checked(complete["nine_paper_report"]).read_text(encoding="utf-8")
    for method, paper in report["nine_papers"].items():
        require(f"| {method} | {paper['status']} | 完成 |" in text, "Human report omitted/changed a verified candidate conclusion")
    require("不是统计显著性结论" in text and "阴性结果全部保留" in text
            and "不称为完整论文复现" in text, "Human report scientific scope disclosures missing")


def semantic_review_binding(training_rows, evaluation, costs):
    """Binds a prose/code review to the final evidence actually examined."""
    return {"registration": R.registration_descriptor(), "audit_source": S.desc(SOURCE),
            "formal_complete_artifacts": sorted((row["completion"] for row in training_rows), key=lambda row: row["path"]),
            "evaluation_complete": evaluation["complete"], "cost_snapshot": costs["snapshot"]}


def semantic_review_audit(path, training_rows, evaluation, costs):
    """Do not promote source existence or a generic complete flag to semantic proof.

    These five requirements need an explicit final review of actual contracts,
    engineering attempts, executed code and queue lifetime evidence. The root
    agent can produce this companion after doing that review; user approval is
    not required. It must not claim that this function re-performs human/code
    interpretation. A prose finding and exact line/evidence references remain
    available for scrutiny beside the independently automated assertions.
    """
    require(path is not None and Path(path).is_file(), "Final independent semantic review is missing")
    data = S.read(path)
    require(data.get("kind") == "independent_final_semantic_review" and data.get("passed") is True
            and data.get("reviewer") and data.get("binding") == semantic_review_binding(training_rows, evaluation, costs),
            "Semantic review is partial, stale, or unbound to these actual final artifacts")
    items = data.get("items", [])
    require(len(items) == len(SEMANTIC_REVIEW_IDS) and {r["id"] for r in items} == SEMANTIC_REVIEW_IDS,
            "Every non-automatic requirement needs an explicit separate review")
    for item in items:
        require(item.get("conclusion") == "verified" and isinstance(item.get("finding"), str) and item["finding"].strip()
                and item.get("checks") and item.get("evidence"), "Semantic conclusion requires substantive checks, finding and sources")
        for check in item["checks"]:
            require(isinstance(check, dict) and check.get("claim") and check.get("observed")
                    and check.get("evidence_locations"), "A semantic check cannot be an unsubstantiated passed flag")
        for source in item["evidence"]:
            S.checked(source)
    return {"passed": True, "artifact": S.desc(path), "verified_ids": sorted(SEMANTIC_REVIEW_IDS),
            "verification_kind": "source_bound_explicit_independent_review_attestation",
            "semantic_findings_not_independently_inferred_from_registration_text": True}


def audit(*, cost_artifact=None, cost_report=None, semantic_review=None):
    previous = SOURCE.with_name("audit_full_plan_acceptance_v3.py")
    require(S.digest(previous) == "892e7a71265e8730c4deaf08f4a84579369ea639865ba5def1233b68fa26e64e", "Sealed wide v3 delegate changed")
    preparation = S.read(PREPARATION)
    for key in ("plan", "protocol_source", "immutable_registration", "versioned_execution_registration"):
        S.checked(preparation[key])
    registration = R.load_registration()
    verify_registered_identity(registration)
    q3 = S.read(S.OUT / "continuation_queue_v3/registration.json")
    for descriptor in q3["sources"]: S.checked(descriptor)
    counts, missing, errors, rows, locks = Counter(), [], [], [], {}
    for fit in registration["fits"]:  # Includes virtual pending identities, never glob-count the matrix.
        try:
            status = R.fit_status(fit["condition"], fit["seed"], fit["fold"])
            counts.update([status["status"]])
            if status["status"] != "training_complete":
                missing.append({"fit_id": fit["fit_id"], "status": status["status"], "reason": status.get("reason")})
                continue
            if fit["condition"] not in locks:
                lock = R._condition_lock(fit["condition"])
                locks[fit["condition"]] = (lock, S.read(S.checked(lock))["contract"])
            lock, contract = locks[fit["condition"]]
            rows.append(check_fit(fit, status, registration, lock, contract))
        except Exception as error:
            errors.append({"fit_id": fit["fit_id"], "error_type": type(error).__name__, "error": str(error)})
    all_training = len(rows) == 360 and not missing and not errors
    evaluation_access_attempted = False
    deep, evaluation, costs = [], {"status": "pending", "reason": "All360 real training proofs must pass before any outer-test artifact read."}, {"status": "pending", "reason": "Final independent cost snapshot/companion pending."}
    if all_training:
        try:
            for fit in registration["fits"]:
                row = next(row for row in rows if row["fit_id"] == fit["fit_id"])
                metadata = S.read(S.checked(row["metadata"]))
                deep.append(deep_checkpoint(fit, row, locks[fit["condition"]][1], metadata))
            if (S.OUT / "evaluation/complete.json").exists():
                evaluation_access_attempted = True
                evaluation = evaluation_audit(registration)
            else:
                evaluation = {"status": "pending", "reason": "All training verified; genuine fixed-checkpoint evaluation delivery still missing."}
        except Exception as error:
            errors.append({"stage": "future_cpu_checkpoint_or_gated_evaluation", "error_type": type(error).__name__, "error": str(error)})
        try:
            costs = final_cost_audit(cost_artifact, cost_report, rows)
        except Exception as error:
            costs = {"status": "pending", "reason": str(error)}
    by_condition = Counter(row["condition"] for row in rows)
    by_policy = Counter(row["policy"] for row in rows)
    complete_train = all_training and len(deep) == 360 and not errors
    complete_eval = evaluation.get("passed") is True
    complete_cost = costs.get("passed") is True
    complete = complete_train and complete_eval and complete_cost
    semantic = {"status": "pending", "reason": "Independent source-bound final semantic review pending", "required_ids": sorted(SEMANTIC_REVIEW_IDS)}
    if complete_train and complete_eval and complete_cost:
        try:
            semantic = semantic_review_audit(semantic_review, rows, evaluation, costs)
        except Exception as error:
            semantic["reason"] = str(error)
    per_requirement = []
    train_ids = {"R04", "R05", "R06", "R09", "R10", "R11", "R17", "R26", "R27", "R29"}
    eval_ids = {"R08", "R19", "R20", "R21", "R22", "R23", "R24", "R25"}
    for requirement in preparation["requirements"]:
        ident = requirement["id"]
        ok = ident in ("R02", "R03")
        pending = "Await complete real scope; narrow preparation/code tests do not satisfy this requirement."
        if ident in train_ids: ok = complete_train
        if ident in eval_ids: ok = complete_train and complete_eval
        if ident == "R07": ok = by_policy["diagnostic"] == 30 and complete_train
        if ident == "R12": ok = by_policy["scratch"] == 15 and complete_eval
        if ident in ("R13", "R14"): ok = by_policy["ssl"] == 15 and complete_train
        if ident == "R15": ok = by_policy["ssl_supervised"] == 15 and complete_train
        if ident == "R16": ok = by_policy["finetune"] == 135 and complete_train
        if ident == "R18":
            proof = S.REPORT / "eegmamba_official_gpu_check.json"
            ok = proof.exists() and S.read(proof).get("passed") is True and by_condition["EEGMamba-frozen-smallhead"] == 15 and by_condition["EEGMamba-lastblock-finetune"] == 15 and complete_train
        if ident == "R28": ok = complete_cost
        if ident in SEMANTIC_REVIEW_IDS: ok = complete_train and semantic.get("passed") is True
        if ident in ("R01", "R30"): ok = complete
        per_requirement.append({"id": ident, "requirement": requirement["requirement"], "status": "verified_complete" if ok else "pending",
                                "reason": None if ok else pending, "authoritative_evidence_locations": requirement["authoritative_evidence_locations"],
                                "scope_limit": "Train/metadata and actual saved proofs only; no constructor/forward or hidden test unlock."})
    # Every requirement is a prerequisite. In v1, R18 could remain pending
    # while the top-level full_plan_completed flag was nevertheless true.
    complete = all(r["status"] == "verified_complete" for r in per_requirement if r["id"] not in ("R01", "R30"))
    for row in per_requirement:
        if row["id"] in ("R01", "R30"):
            row["status"] = "verified_complete" if complete else "pending"
            row["reason"] = None if complete else "Every R02-R29 evidence requirement must pass."
        if row["id"] in SEMANTIC_REVIEW_IDS:
            row["verification_kind"] = "automated_artifact_checks_plus_explicit_independent_semantic_review"
            if not semantic.get("passed"):
                row["reason"] = semantic["reason"]
    references = []
    for pattern in ("paired_completed_readonly_audit_30fits.json", "ssl_cumulative_readonly_audit_15fits_*.json", "conformer_cumulative_audit_15fits_*.json", "eeg2rep_supervised_cumulative_audit_*fits_*.json"):
        for path in sorted(S.REPORT.glob(pattern)):
            data = S.read(path)
            if data.get("source"): S.checked(data["source"])
            references.append({"artifact": S.desc(path), "actual_verified_fits": data.get("completed_fits_verified", data.get("completed_fits_checked")),
                               "all15_complete": data.get("all15_complete"), "scope": data.get("scope"),
                               "partial_scope_is_not_promoted_to_full15": True})
    return {"schema_version": 4, "artifact_kind": "wide final acceptance audit; explicit pending when not fully achieved", "captured_at": S.now(),
            "source": S.desc(SOURCE), "prior_auditor_source": S.desc(DELEGATED_SOURCE), "previous_v2_auditor_source": S.desc(PREVIOUS_SOURCE),
            "previous_v3_auditor_source": S.desc(previous), "execution_verifier_source": S.desc(Path(X.__file__)),
            "preparation": S.desc(PREPARATION), "registration": R.registration_descriptor(),
            "objective_satisfied": complete, "full_plan_completed": complete, "formal_identity_count": 360,
            "status_counts": dict(counts), "completed_training_json_verified": len(rows), "incomplete_fits": missing, "verification_errors": errors,
            "training_rows": rows, "by_condition_verified": dict(by_condition), "by_policy_verified": dict(by_policy),
            "semantic_review": semantic, "deep_cpu_checkpoint_scope_verified": len(deep), "cpu_checkpoint_proofs": deep, "evaluation": evaluation, "final_cost": costs,
            "requirements": per_requirement, "prior_actual_audit_references": references,
            "outer_test_artifacts_read": evaluation_access_attempted,
            "outer_test_read_flag_semantics": "conservative: access attempted, including a failed verification after partial reads",
            "current_pending_run_never_calls_test_gate_or_reads_predictions": not all_training,
            "audit_operations": {"models_constructed": 0, "gpu_operations": 0, "signal_or_feature_array_loads": 0, "new_neural_forwards": 0, "optimizer_steps": 0,
                                 "sealed_source_edits": 0, "registry_status_edits": 0},
            "cost_final_artifact_contract": {"required_markers": {"draft_not_completion_certificate": False, "final_snapshot": True, "all_available_artifacts_accounted": True, "coverage_gaps_disclosed": True},
                                             "independent_human_cost_companion_required": True, "unknown_historical_physical_totals_may_remain_null": True,
                                             "original_E2_report_hash_must_not_be_modified_for_cost_delivery": True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cost-artifact", type=Path)
    parser.add_argument("--cost-report", type=Path)
    parser.add_argument("--semantic-review", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    value = audit(cost_artifact=args.cost_artifact, cost_report=args.cost_report, semantic_review=args.semantic_review)
    path = (args.output or S.REPORT / ("full_plan_acceptance_audit_v4_" + str(time.time_ns()) + ".json")).resolve()
    require(path.is_relative_to(S.REPORT), "Save a new separate audit only in this analysis directory")
    S.save(path, value, immutable=True)
    print(json.dumps({"artifact": S.desc(path), "full_plan_completed": value["full_plan_completed"], "status_counts": value["status_counts"],
                      "training_verified": value["completed_training_json_verified"], "pending_requirements": sum(r["status"] == "pending" for r in value["requirements"]),
                      "errors": len(value["verification_errors"]), "test_predictions_read": value["outer_test_artifacts_read"]}))
    if args.require_complete and not value["full_plan_completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Fold-local Conformer scratch and EEG2Rep post-SSL subject classifiers.

No outer-test forward is exposed. Formal fitting requires a prior successful
CUDA train-only engineering lock; EEG2Rep additionally requires the same
seed/fold's genuinely completed SSL best checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import time

import numpy as np
import torch

from . import finetune as F, protocol as P, registry as R, storage as S
from .adapters import eegconformer as EC, eeg2rep as ER

CONDITIONS = ("EEGConformer-scratch", "EEG2Rep-supervised")
SSL_CONDITION = "EEG2Rep-fold-local-ssl"
require = F.require


def policy_key(condition):
    require(condition in CONDITIONS, "Unknown scratch/post-SSL supervised condition")
    return P.CONDITION_BY_ID[condition].policy_key


def train_statistics(raw, train_ids):
    """Select only declared training people before fitting any scalar stats."""
    require(len(train_ids) == len(set(train_ids)) > 0 and all(sid in raw for sid in train_ids), "Invalid normalization train identities")
    return EC.fit_train_statistics(np.stack([raw[sid] for sid in train_ids]), train_ids, train_ids)


def validate_ssl_payload(saved, metadata, *, seed, fold, split, registration, raw_manifest):
    """Pure identity/value validation; byte/file/status gates are in binding."""
    expected = {"condition": SSL_CONDITION, "seed": seed, "fold": fold,
                "train_ids": split["train"], "val_ids": split["val"],
                "encoder_fit_subject_ids": split["train"], "registry": registration, "source": raw_manifest}
    require(saved.get("stage") == "ssl" and isinstance(saved.get("identity"), dict), "Supervision requires an SSL selected-best checkpoint")
    require(all(saved["identity"].get(key) == value and metadata.get(key) == value for key, value in expected.items()),
            "SSL checkpoint/metadata must match this exact seed, fold, training and validation people")
    require(saved.get("configuration") == ER.DEFAULT_CONFIG and saved.get("ssl_contract") == ER.SSL_CONTRACT,
            "SSL configuration or author normalization contract differs")
    require(saved.get("epoch") == metadata.get("best_epoch") and isinstance(saved["epoch"], int) and saved["epoch"] > 0,
            "SSL selected best epoch differs from completed metadata")
    require(saved.get("optimizer_updates", 0) > 0 and saved.get("ema_updates") == saved["optimizer_updates"],
            "SSL selected state has no genuine logical updates/EMA evidence")
    require(isinstance(saved.get("state_dict"), dict) and saved["state_dict"]
            and all(torch.isfinite(value).all() for value in saved["state_dict"].values()), "SSL learned state is missing or nonfinite")
    fingerprint = F.state_digest(saved["state_dict"])
    require(fingerprint == saved.get("state_sha256") == metadata.get("selected_state_sha256")
            and fingerprint != metadata.get("initial_state_sha256"), "SSL best state/hash unchanged or mismatched")
    require(metadata.get("test_forward_count") == 0 and metadata.get("validation_updates") == 0,
            "SSL completion must have no test forwards or validation updates")
    for key in ("settings", "validation_mask_manifest"):
        require(saved["identity"].get(key) == metadata.get(key) and isinstance(metadata.get(key), dict), "SSL source contract incomplete: " + key)
    return fingerprint


def bind_ssl_checkpoint(seed, fold, split):
    folder = R.fit_directory(SSL_CONDITION, seed, fold)
    status = R.fit_status(SSL_CONDITION, seed, fold)
    require(status["status"] == "training_complete", "Same-seed/fold SSL must actually finish before supervision")
    evidence = status["evidence"]
    R._validate_complete(evidence, folder)
    checkpoint_path = S.checked(evidence["checkpoint"])
    require(checkpoint_path == folder / "best.pt", "EEG2Rep must load this fold's selected SSL best.pt")
    metadata = S.read(S.checked(evidence["metadata"]))
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    registration = R.load_registration()
    fingerprint = validate_ssl_payload(saved, metadata, seed=seed, fold=fold, split=split,
                                       registration=R.registration_descriptor(),
                                       raw_manifest=registration["sources"]["accepted_raw_manifest"])
    for key in ("settings", "validation_mask_manifest"):
        S.checked(metadata[key])
    source_lock = R._condition_lock(SSL_CONDITION)
    require(status["condition_source_lock"] == source_lock, "Completed SSL source lock differs")
    binding = {"condition": SSL_CONDITION, "seed": seed, "fold": fold,
               "train_ids": split["train"], "val_ids": split["val"], "encoder_fit_subject_ids": split["train"],
               "checkpoint": evidence["checkpoint"], "metadata": evidence["metadata"],
               "completion": S.desc(folder / "complete.json"), "status": S.desc(folder / "status.json"),
               "condition_source_lock": source_lock, "selected_state_sha256": fingerprint,
               "best_epoch": saved["epoch"], "settings": metadata["settings"],
               "validation_mask_manifest": metadata["validation_mask_manifest"],
               "normalization": ER.SSL_CONTRACT["input_normalization"], "cross_fold_weights_shared": False}
    return saved["state_dict"], binding


class SupervisedModel(F.RawSubjectModel):
    def __init__(self, encoder, head, adapter, selected_names, statistics, window_chunk):
        super().__init__(encoder, head, adapter, selected_names, 5, window_chunk)
        self.statistics = statistics

    def adapt(self, raw):
        return self.adapter.adapt_raw(raw, self.statistics) if self.adapter is EC else self.adapter.adapt_raw(raw)


def build_model(condition, *, seed, fold, split, raw, device="cuda", window_chunk=8, statistics=None):
    require(condition in CONDITIONS, "Condition not in supervised scratch/SSL routes")
    binding = None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if condition == "EEGConformer-scratch":
            statistics = train_statistics(raw, split["train"]) if statistics is None else statistics
            require(statistics["fit_subject_ids"] == split["train"], "Normalizer comes from another fold's people")
            encoder, audit = EC.load_encoder(device=device, trainable=True)
            adapter = EC
        else:
            require(statistics is None, "EEG2Rep follows its SSL normalization; no new cohort statistics allowed")
            learned, binding = bind_ssl_checkpoint(seed, fold, split)
            encoder, audit = ER.build_model(device=device, stage="supervised")
            encoder.load_state_dict(learned, strict=True)
            require(F.state_digest(encoder.state_dict()) == binding["selected_state_sha256"], "Strict EEG2Rep SSL weight loading differs")
            adapter = ER
    selected = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    require(selected == audit["trainable_parameter_names"], "Adapter and supervised encoder trainable parameters disagree")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = F.SubjectBagHead(audit["feature_dim"]).to(device=device, dtype=torch.float32)
    model = SupervisedModel(encoder, head, adapter, selected, statistics, window_chunk)
    audit.update(fresh_subject_head=True, head_seed=seed, ssl_binding=binding,
                 planned_update_parameter_names=list(model.update_names), statistics=statistics,
                 initialization="seeded_random_entire_backbone" if binding is None else "same_seed_fold_completed_SSL_best",
                 normalization=EC.INPUT_CONTRACT["normalization"] if binding is None else ER.SSL_CONTRACT["input_normalization"])
    return model, audit


def set_train_mode(model):
    """Train active encoder subtrees; unused/target/predictor modules stay eval."""
    selected = set(model.selected_names)
    for name, parameter in model.encoder.named_parameters():
        parameter.requires_grad_(name in selected)
    model.eval()
    model.head.train()
    enabled = []
    for name, module in model.encoder.named_modules():
        if any(name == scope or (scope and name.startswith(scope + ".")) for scope in enabled):
            continue
        prefix = name + "." if name else ""
        parameters = {prefix + parameter_name for parameter_name, _ in module.named_parameters()}
        if parameters and parameters.issubset(selected):
            module.train(True)
            enabled.append(name)
    return enabled


def training_state_keys(model):
    """Include active BatchNorm running state in validation-selected checkpoints."""
    keys = set(model.update_names)
    for name, _ in model.named_buffers():
        owner = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        parameters = list(owner.parameters(recurse=False))
        if parameters and all(parameter.requires_grad for parameter in parameters):
            keys.add(name)
    return keys


def selected_state(model):
    keys = training_state_keys(model)
    return F.cpu_state({name: tensor for name, tensor in model.state_dict().items() if name in keys})


def frozen_state(model):
    keys = training_state_keys(model)
    return {name: tensor for name, tensor in model.state_dict().items() if name not in keys}


def load_selected_state(model, state):
    require(set(state) == training_state_keys(model), "Best supervised parameters/running-stat buffers differ")
    complete = model.state_dict()
    complete.update(state)
    model.load_state_dict(complete, strict=True)


def optimizer_for(model, condition):
    policy = P.POLICIES[policy_key(condition)]
    parameters = dict(model.encoder.named_parameters())
    return torch.optim.AdamW([{"params": list(model.head.parameters()), "lr": policy.head_lr, "group_name": "head"},
                             {"params": [parameters[name] for name in model.selected_names], "lr": policy.backbone_lr, "group_name": "backbone"}],
                            weight_decay=policy.weight_decay)


def set_learning_rates(optimizer, condition, epoch):
    key = policy_key(condition)
    policy = P.POLICIES[key]
    for group in optimizer.param_groups:
        base = policy.head_lr if group["group_name"] == "head" else policy.backbone_lr
        group["lr"] = base * P.learning_rate_multiplier(key, epoch, parameter_group=group["group_name"])


def _sources(condition, model):
    sources = F._implementation_sources(condition, model, [])
    for path in (Path(__file__), Path(F.__file__), Path(model.adapter.ROOT / "author_source_binding.json")):
        sources.append(S.desc(path))
    for name in (ER.SOURCE_SHA256 if model.adapter is ER else ("conformer.py",)):
        sources.append(S.desc(model.adapter.ROOT / name))
    return list({item["path"]: item for item in sources}.values())


def prepare_condition(condition, *, seed=11, fold=1, micro_subjects=1, window_chunk=8):
    require(condition in CONDITIONS and micro_subjects in (1, 2) and window_chunk > 0, "Invalid engineering setting")
    require(torch.cuda.is_available(), "Real supervised engineering requires CUDA")
    R.load_registration()
    if (S.OUT / "engineering" / (condition + ".json")).exists():
        R._condition_lock(condition)
        return
    started = time.perf_counter()
    backward_calls = 0
    model, optimizer, raw = None, None, None
    attempt_path = S.OUT / "supervised_engineering" / condition / f"attempt_{time.time_ns()}.json"
    try:
        torch.cuda.reset_peak_memory_stats()
        _, folds, labels = F.C.context()
        split = folds[f"fold_{fold}"]
        # Fit engineering normalization on training people only; source reads,
        # model construction and normalization are included in attempt cost.
        people = split["train"] if condition == "EEGConformer-scratch" else split["train"][:micro_subjects]
        raw, raw_sources = F._raw_sources(split, people=people)
        model, audit = build_model(condition, seed=seed, fold=fold, split=split, raw=raw, window_chunk=window_chunk)
        F.C.runtime(seed)
        optimizer = optimizer_for(model, condition)
        initial = F.cpu_state(model.state_dict())
        set_train_mode(model)
        chosen = split["train"][:micro_subjects]
        target = torch.tensor([labels[sid] for sid in chosen], device=model.device)
        loss = torch.nn.functional.cross_entropy(model(np.stack([raw[sid][F.bag_indices(1)] for sid in chosen])), target)
        require(torch.isfinite(loss), "Nonfinite supervised engineering CE")
        loss.backward()
        backward_calls += 1
        require(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.requires_grad), "Supervised encoder/head engineering gradients disconnected")
        require(all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad), "Supervised frozen parameters received gradients")
        # Running statistics may update in a real from-scratch BN probe; reset
        # all values so this zero-optimizer engineering does not seed training.
        observed_buffers = {name: not torch.equal(initial[name], tensor.cpu()) for name, tensor in model.named_buffers()}
        model.load_state_dict(initial, strict=True)
        state = F.rng_state(model.device)
        probe_path = attempt_path.with_suffix(".pt")
        probe = {"state_dict": F.cpu_state(model.state_dict()), "optimizer_state_dict": F.cpu_state(optimizer.state_dict()),
                 "scheduler_state_dict": {"policy": policy_key(condition), "completed_epoch": 0, "extended": False},
                 "rng": state, "sampler_position": {"epoch_completed": 0, "next_batch": 0}}
        S.save_torch(probe_path, probe, immutable=True)
        loaded = torch.load(probe_path, map_location="cpu", weights_only=False)
        model.load_state_dict(loaded["state_dict"], strict=True)
        optimizer.load_state_dict(loaded["optimizer_state_dict"])
        F.restore_rng(loaded["rng"])
        require(F.state_digest(initial) == F.state_digest(model.state_dict()) and F.rng_equal(state, F.rng_state(model.device)), "Supervised engineering full-state restore differs")
        selected = list(model.update_names)
        binding = audit["ssl_binding"]
        contract = {"input_contract": {"adapter_audit": audit, "micro_subjects": micro_subjects,
                    "window_chunk": window_chunk, "training_windows_per_subject": 8,
                    "eval_windows_per_subject": 60, "effective_subjects": 8, "precision": "float32",
                    "normalization_scope": "every_fit_train_people_only" if binding is None else "exact_fold_SSL_contract",
                    "probe_ssl_binding_not_shared_across_other_fits": binding},
                    "trainable_parameter_names": selected,
                    "frozen_parameter_names": [name for name, _ in model.named_parameters() if name not in selected],
                    "implementation_sources": _sources(condition, model),
                    "weight_sources": [] if binding is None else [binding["checkpoint"], binding["metadata"], binding["completion"], binding["status"]],
                    "forward_checked": True, "gradient_checked": True, "resume_checked": True,
                    "training_only_engineering_checked": True, "probe_subject_ids": chosen,
                    "probe_raw_sources": {sid: raw_sources[sid] for sid in chosen},
                    "probe_checkpoint": S.desc(probe_path), "engineering_optimizer_updates": 0,
                    "engineering_backward_calls": backward_calls, "buffer_changes_during_probe_reset_before_exit": observed_buffers,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "seconds": time.perf_counter() - started}
        S.save(attempt_path, {"passed": True, "condition": condition, "seed": seed, "fold": fold,
               "optimizer_updates": 0, "backward_calls": backward_calls, "formal_fits": 0,
               "seconds": time.perf_counter() - started, "contract": contract}, immutable=True)
        R.lock_condition(condition, contract)
        for value in P.SEEDS:
            for index in P.FOLDS:
                R.record_fit_status(condition, value, index, "ready")
        return contract
    except BaseException as error:
        if not attempt_path.exists():
            S.save(attempt_path, {"passed": False, "condition": condition, "seed": seed, "fold": fold,
                   "optimizer_updates": 0, "backward_calls": backward_calls, "formal_fits": 0,
                   "seconds": time.perf_counter() - started, "error": f"{type(error).__name__}: {error}"}, immutable=True)
        raise
    finally:
        del model, optimizer, raw
        gc.collect()
        torch.cuda.empty_cache()


def validate_resume(saved, *, identity, model, condition, train_ids, micro_subjects):
    require(saved["identity"] == identity, "Supervised resume fold/normalization/SSL checkpoint/source binding differs")
    epoch = saved["epoch"]
    key = policy_key(condition)
    require(0 <= epoch <= P.POLICIES[key].final_epochs and len(saved["history"]) == len(saved["subject_order"]) == epoch,
            "Supervised resume epoch/history lengths differ")
    require(saved["scheduler_state_dict"] == {"policy": key, "completed_epoch": epoch, "extended": saved["extended"]}
            and saved["sampler_position"] == {"epoch_completed": epoch, "next_batch": 0}, "Supervised scheduler/sampler differs")
    steps = math.ceil(len(train_ids) / 8)
    backwards = len(train_ids) // 8 * math.ceil(8 / micro_subjects) + math.ceil(len(train_ids) % 8 / micro_subjects)
    require(saved["optimizer_updates"] == steps * epoch and saved["backward_calls"] == backwards * epoch,
            "Supervised optimizer/backward totals differ")
    for index, (row, people) in enumerate(zip(saved["history"], saved["subject_order"]), 1):
        require(row["epoch"] == index and row["optimizer_updates"] == steps * index and row["backward_calls"] == backwards * index
                and row["window_indices"] == F.bag_indices(index).tolist() and sorted(people) == sorted(train_ids), "Supervised resumed sampling/history differs")
    require(all(torch.isfinite(value).all() for value in saved["state_dict"].values()), "Nonfinite supervised resume state")
    best = saved["best_selected_state_dict"]
    require((epoch == 0 and best is None) or (best is not None and F.state_digest(best) == saved["best_selected_state_sha256"]
            and set(best) == training_state_keys(model)), "Supervised best state/running statistics differ")
    groups = saved["optimizer_state_dict"]["param_groups"]
    require([group["group_name"] for group in groups] == ["head", "backbone"], "Supervised optimizer groups differ")
    states = saved["optimizer_state_dict"]["state"]
    for group in groups:
        for parameter_id in group["params"]:
            state = states.get(parameter_id, {})
            require((epoch == 0 and not state) or (state and float(state["step"]) == steps * epoch), "Supervised optimizer step state differs")


def fit_one(condition, seed, fold, *, resume=False, micro_subjects=1, window_chunk=8):
    require(condition in CONDITIONS and seed in P.SEEDS and fold in P.FOLDS and micro_subjects in (1, 2), "Unregistered supervised fit/engineering setting")
    require(torch.cuda.is_available(), "Formal supervised fitting requires CUDA")
    attempt_started = time.perf_counter()
    source_lock = R._condition_lock(condition)
    folder = R.fit_directory(condition, seed, fold)
    if (folder / "complete.json").exists():
        evidence = S.read(folder / "complete.json")
        R._validate_complete(evidence, folder)
        if R.fit_status(condition, seed, fold)["status"] != "training_complete":
            R.record_fit_status(condition, seed, fold, "training_complete", evidence=evidence)
        return S.read(folder / "metadata.json")
    partial = (folder / "started.json").exists()
    require(not partial or resume, "Partial supervised fit requires complete-state --resume")
    cohort, folds, labels = F.C.context()
    split = folds[f"fold_{fold}"]
    raw, raw_sources = F._raw_sources(split)
    stats_path = folder / "train_statistics.json"
    statistics = S.read(stats_path)["statistics"] if stats_path.exists() else None
    model, audit = build_model(condition, seed=seed, fold=fold, split=split, raw=raw, window_chunk=window_chunk, statistics=statistics)
    F._contract_matches(condition, model, micro_subjects, window_chunk)
    if condition == "EEGConformer-scratch":
        statistics_record = {"condition": condition, "seed": seed, "fold": fold, "statistics": model.statistics,
                             "fit_raw_sources": {sid: raw_sources[sid] for sid in split["train"]},
                             "registry": R.registration_descriptor(), "validation_or_test_fit_people": []}
        S.save(stats_path, statistics_record, immutable=True)
    else:
        require(not stats_path.exists(), "Unexpected learned cohort normalization in EEG2Rep supervised route")
    F.C.runtime(seed)
    optimizer = optimizer_for(model, condition)
    key = policy_key(condition)
    initial = F.state_digest(model.state_dict())
    initial_selected = selected_state(model)
    frozen_initial = F.state_digest(frozen_state(model))
    identity = {"condition": condition, "seed": seed, "fold": fold, "registry": R.registration_descriptor(),
                "condition_source_lock": source_lock, "train_ids": split["train"], "val_ids": split["val"],
                "raw_sources": raw_sources, "statistics": S.desc(stats_path) if condition == "EEGConformer-scratch" else None,
                "ssl_binding": audit["ssl_binding"], "initial_state_sha256": initial, "frozen_initial_sha256": frozen_initial}
    history, order, best, epoch, extended, updates, backwards = [], [], None, 0, False, 0, 0
    R.record_fit_status(condition, seed, fold, "training")
    if not partial:
        S.save(folder / "started.json", {**identity, "started_at": S.now()}, immutable=True)

    def payload():
        return {"identity": identity, "state_dict": F.cpu_state(model.state_dict()),
                "optimizer_state_dict": F.cpu_state(optimizer.state_dict()),
                "scheduler_state_dict": {"policy": key, "completed_epoch": epoch, "extended": extended},
                "rng": F.rng_state(model.device), "sampler_position": {"epoch_completed": epoch, "next_batch": 0},
                "epoch": epoch, "history": history, "subject_order": order, "best_selected_state_dict": best,
                "best_selected_state_sha256": F.state_digest(best) if best is not None else None,
                "extended": extended, "optimizer_updates": updates, "backward_calls": backwards}

    durable_epoch, start_updates, started = 0, 0, attempt_started
    torch.cuda.reset_peak_memory_stats()
    try:
        if partial:
            target = folder / "latest.pt" if (folder / "latest.pt").exists() else folder / "initial.pt"
            if not target.exists():
                started_record = S.read(folder / "started.json")
                require(all(started_record.get(name) == value for name, value in identity.items()), "Epoch-zero supervised start identity differs")
                F._recover_epoch_zero(folder, payload())
            saved = torch.load(target, map_location="cpu", weights_only=False)
            validate_resume(saved, identity=identity, model=model, condition=condition, train_ids=split["train"], micro_subjects=micro_subjects)
            model.load_state_dict(saved["state_dict"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            F.restore_rng(saved["rng"])
            require(F.rng_equal(saved["rng"], F.rng_state(model.device)) and F.state_digest(frozen_state(model)) == frozen_initial, "Supervised resume RNG/frozen state differs")
            epoch, history, order, best = saved["epoch"], saved["history"], saved["subject_order"], saved["best_selected_state_dict"]
            extended, updates, backwards = saved["extended"], saved["optimizer_updates"], saved["backward_calls"]
        else:
            S.save_torch(folder / "initial.pt", payload(), immutable=True)
        durable_epoch, start_updates = epoch, updates
        S.save(folder / "train_history.json", history)
        S.save(folder / "validation_history.json", [{"epoch": row["epoch"], "val_ce": row["val_eval_ce"]} for row in history])
        decision = P.training_decision(key, [row["val_eval_ce"] for row in history], extended=extended) if history else {"stop": False}
        while not decision["stop"]:
            epoch += 1
            modes = set_train_mode(model)
            set_learning_rates(optimizer, condition, epoch)
            people = [split["train"][i] for i in torch.randperm(len(split["train"])).tolist()]
            windows = F.bag_indices(epoch)
            epoch_ce = 0.
            for start in range(0, len(people), 8):
                logical = people[start:start + 8]
                S.save(folder / "progress.json", {"epoch": epoch, "optimizer_updates_started": updates + 1,
                       "optimizer_updates_completed": updates, "batch_subject_ids": logical,
                       "window_indices": windows.tolist(), "last_durable_epoch": durable_epoch})
                outcome = F.subject_update(model, optimizer, logical, raw, labels, windows, micro_subjects=micro_subjects)
                updates += 1
                backwards += outcome["backward_calls"]
                epoch_ce += outcome["subject_ce_sum"]
                S.save(folder / "progress.json", {"epoch": epoch, "optimizer_updates_completed": updates,
                       "backward_calls_completed": backwards, "last_durable_epoch": durable_epoch})
            require(F.state_digest(frozen_state(model)) == frozen_initial, "Frozen EEG2Rep predictor/target or buffers changed in supervision")
            order.append(people)
            history.append({"epoch": epoch, "train_bag_ce": epoch_ce / len(people),
                            "optimizer_updates": updates, "backward_calls": backwards,
                            "window_indices": windows.tolist(), "cycle_epoch": (epoch - 1) % 8 + 1,
                            "repeated_padding_windows_this_epoch": [0, 1, 2, 3] if epoch % 8 == 0 else [],
                            "head_lr": optimizer.param_groups[0]["lr"], "backbone_lr": optimizer.param_groups[1]["lr"],
                            "training_module_subtrees": modes, "supervised_ema_updates": 0,
                            **F.diagnostics(model, raw, split, labels)})
            decision = P.training_decision(key, [row["val_eval_ce"] for row in history], extended=extended)
            if decision["best_epoch"] == epoch:
                best = selected_state(model)
            if decision["extend"]:
                extended = True
            S.save_torch(folder / "latest.pt", payload())
            durable_epoch = epoch
            S.save(folder / "train_history.json", history)
            S.save(folder / "validation_history.json", [{"epoch": row["epoch"], "val_ce": row["val_eval_ce"]} for row in history])
            if epoch % 20 == 0:
                print(f"SUPERVISED {condition} seed={seed} fold={fold} epoch={epoch} best={decision['best_epoch']}", flush=True)
        require(best is not None, "Validation-selected supervised state missing")
        final_training = selected_state(model)
        changed = {name: not torch.equal(final_training[name], initial_selected[name]) for name in model.update_names}
        require(all(changed.values()), "An intended CNN/attention/encoder/head parameter never updated")
        load_selected_state(model, best)
        require(F.state_digest(frozen_state(model)) == frozen_initial, "Validation best restoration changed frozen state")
        model.eval()
        best_path = folder / "best.pt"
        S.save_torch(best_path, {"identity": identity, "state_dict": F.cpu_state(model.state_dict()),
                     "statistics": model.statistics, "best_epoch": decision["best_epoch"]}, immutable=True)
        probe_people = split["train"][:1]
        with torch.inference_mode():
            reference = model(np.stack([raw[sid] for sid in probe_people])).cpu()
            reloaded = torch.load(best_path, map_location="cpu", weights_only=False)
            require(reloaded["identity"] == identity and reloaded["statistics"] == model.statistics, "Best normalization/source binding differs")
            model.load_state_dict(reloaded["state_dict"], strict=True)
            require(torch.equal(reference, model(np.stack([raw[sid] for sid in probe_people])).cpu()), "Supervised full reload prediction differs")
        metadata = {**identity, "epochs": epoch, "updates": updates, "optimizer_updates": updates,
                    "head_optimizer_updates": updates, "encoder_optimizer_updates": updates,
                    "backward_calls": backwards, "best_epoch": decision["best_epoch"], "stop_reason": decision["reason"], "extended": extended,
                    "adapter_audit": audit, "selected_state_sha256": F.state_digest(model.state_dict()),
                    "frozen_final_sha256": F.state_digest(frozen_state(model)), "parameter_changed_after_training": changed,
                    "active_running_stat_buffer_keys": sorted(training_state_keys(model) - set(model.update_names)),
                    "micro_subjects": micro_subjects, "window_chunk": window_chunk, "effective_subjects": 8,
                    "window_bag_size": 8, "eval_windows": 60, "cycle_unique_windows": 60, "cycle_repeats": [0, 1, 2, 3],
                    "batchnorm_microbatch_statistics_not_true_batch8": condition == "EEGConformer-scratch",
                    "supervised_ema_updates": 0, "test_forward_count": 0, "precision": "float32",
                    "fresh_subject_head": True, "reload_probe_subject_ids": probe_people,
                    "seconds_this_attempt": time.perf_counter() - started, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        if (folder / "metadata.json").exists():
            prior = S.read(folder / "metadata.json")
            require(all(prior[name] == metadata[name] for name in ("condition", "seed", "fold", "epochs", "updates", "best_epoch", "selected_state_sha256", "ssl_binding", "statistics")), "Existing supervised completion metadata differs")
            metadata = prior
        else:
            S.save(folder / "metadata.json", metadata, immutable=True)
        evidence = {"passed": True, "checkpoint_reload_verified": True, "parameter_update_verified": True,
                    "frozen_parameters_verified": True, "epochs": epoch, "optimizer_updates": updates,
                    "stop_reason": decision["reason"], "registry": R.registration_descriptor(),
                    **{name: S.desc(folder / filename) for name, filename in (("checkpoint", "best.pt"), ("train_history", "train_history.json"),
                            ("validation_history", "validation_history.json"), ("metadata", "metadata.json"))}}
        S.save(folder / "complete.json", evidence, immutable=True)
        R.record_fit_status(condition, seed, fold, "training_complete", evidence=evidence)
        return metadata
    except BaseException as error:
        S.save(folder / f"failure_{time.time_ns()}.json", {"error": f"{type(error).__name__}: {error}", "last_durable_epoch": durable_epoch,
               "optimizer_updates": updates, "updates_completed_this_process": updates - start_updates,
               "epoch_started": epoch, "seconds_this_attempt": time.perf_counter() - started, "time": S.now()}, immutable=True)
        R.record_fit_status(condition, seed, fold, "failed", reason=str(error))
        raise
    finally:
        del model, optimizer, raw
        gc.collect()
        torch.cuda.empty_cache()


def run(conditions=CONDITIONS, *, seed=None, fold=None, prepare=False, resume=False, limit=None, micro_subjects=1, window_chunk=8):
    require(limit is None or limit > 0, "Limit must be positive")
    count = 0
    for condition in conditions:
        require(condition in CONDITIONS, "Unregistered supervised condition")
        if prepare:
            prepare_condition(condition, seed=11 if seed is None else seed, fold=1 if fold is None else fold,
                              micro_subjects=micro_subjects, window_chunk=window_chunk)
            continue
        R._condition_lock(condition)
        for value in P.SEEDS if seed is None else (seed,):
            for index in P.FOLDS if fold is None else (fold,):
                fit_one(condition, value, index, resume=resume, micro_subjects=micro_subjects, window_chunk=window_chunk)
                count += 1
                if limit is not None and count >= limit:
                    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, choices=P.SEEDS)
    parser.add_argument("--fold", type=int, choices=P.FOLDS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--micro-subjects", type=int, choices=(1, 2), default=1)
    parser.add_argument("--window-chunk", type=int, default=8)
    args = parser.parse_args()
    run(args.conditions, seed=args.seed, fold=args.fold, prepare=args.prepare, resume=args.resume,
        limit=args.limit, micro_subjects=args.micro_subjects, window_chunk=args.window_chunk)

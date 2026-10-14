"""Fresh-pretrained, fold-local limited fine-tuning of subject window bags.

Run ``--prepare`` separately to perform real training-person engineering
probes and lock a condition. Fitting refuses to create a synthetic readiness
lock. Training/evaluation in this module never forwards an outer-test person.
"""
from __future__ import annotations

import argparse
import gc
import importlib
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from tridep_ccml.tridep_stage1 import common as C
from . import protocol as P, registry as R, storage as S

CONDITIONS = tuple(c.condition_id for c in P.CONDITIONS if c.policy_key == "finetune")
POLICY = P.POLICIES["finetune"]


def require(value, message):
    if not value:
        raise ValueError(message)


class SubjectBagHead(torch.nn.Module):
    """Same registered head form for 8-window training and 60-window scoring."""
    def __init__(self, dimension):
        super().__init__()
        self.window = torch.nn.Sequential(torch.nn.LayerNorm(dimension), torch.nn.Linear(dimension, 128),
                                         torch.nn.GELU(), torch.nn.Dropout(.2))
        self.classifier = torch.nn.Linear(128, 2)

    def forward(self, windows):
        require(windows.ndim == 3 and windows.shape[1] in (8, 60), "Expected 8/60 window representations per subject")
        return self.classifier(self.window(windows).mean(dim=1))


def bag_indices(epoch):
    require(isinstance(epoch, int) and epoch >= 1, "Epoch must be one-based")
    return ((epoch - 1) % 8 * 8 + np.arange(8, dtype=np.int64)) % 60


def rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def rng_equal(left, right):
    return (left["python"] == right["python"] and left["numpy"][0] == right["numpy"][0]
            and np.array_equal(left["numpy"][1], right["numpy"][1]) and left["numpy"][2:] == right["numpy"][2:]
            and torch.equal(left["torch_cpu"], right["torch_cpu"])
            and len(left["torch_cuda"]) == len(right["torch_cuda"])
            and all(torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"])))


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    return value


def state_digest(state):
    return C.state_digest(state)


class RawSubjectModel(torch.nn.Module):
    def __init__(self, encoder, head, adapter, selected_names, seconds, window_chunk=8):
        super().__init__()
        self.encoder, self.head = encoder, head
        self.adapter = adapter
        self.selected_names = tuple(selected_names)
        self.seconds = seconds
        self.window_chunk = window_chunk
        require(window_chunk > 0, "Window chunk must be positive")

    @property
    def device(self):
        return next(self.head.parameters()).device

    @property
    def update_names(self):
        return tuple("encoder." + name for name in self.selected_names) + tuple("head." + name for name, _ in self.head.named_parameters())

    def adapt(self, raw):
        if getattr(self.adapter, "__name__", "").endswith(".labram"):
            return self.adapter.adapt_raw(raw, seconds=self.seconds)
        return self.adapter.adapt_raw(raw)

    def forward(self, raw):
        require(isinstance(raw, np.ndarray) and raw.dtype == np.float32 and raw.ndim == 4
                and raw.shape[1] in (8, 60) and tuple(raw.shape[2:]) == (19, 1000), "Expected raw [subjects,8/60,19,1000]")
        subjects, windows = raw.shape[:2]
        adapted = self.adapt(np.ascontiguousarray(raw.reshape(-1, 19, 1000)))
        features = []
        for start in range(0, len(adapted), self.window_chunk):
            tensor = torch.as_tensor(adapted[start:start + self.window_chunk], device=self.device)
            features.append(self.adapter.encode_windows(self.encoder, tensor))
        representation = torch.cat(features).reshape(subjects, windows, -1)
        require(torch.isfinite(representation).all(), "Encoder returned nonfinite representations")
        return self.head(representation)


def build_model(condition, *, seed=11, device="cuda", window_chunk=8):
    require(condition in CONDITIONS, "Only registered limited-finetune conditions supported")
    specification = P.CONDITION_BY_ID[condition]
    adapter_name = "codebrain_v2" if specification.method == "CodeBrain" else specification.method.lower()
    adapter = importlib.import_module(".adapters." + adapter_name, __package__)
    # Author constructors are isolated from the fresh head initialization.
    with torch.random.fork_rng(devices=[]):
        encoder, audit = adapter.load_encoder(device=device, trainable=True)
    selected = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    require(selected and len(selected) == len(set(selected)), "Adapter did not expose its exact last-block parameters")
    if hasattr(adapter, "exact_finetune_parameter_names"):
        require(selected == adapter.exact_finetune_parameter_names(encoder), "Adapter trainability differs from exact registered module list")
    for key in ("trainable_parameter_names", "lastblock_parameter_names", "finetune_parameter_names"):
        if key in audit:
            require(selected == audit[key], "Adapter audit and loaded trainable parameter list disagree")
    dimension = audit.get("native_dimension", audit.get("feature_dim", getattr(encoder, "output_dimension", None)))
    require(isinstance(dimension, int) and dimension > 0, "Adapter must report its native representation dimension")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = SubjectBagHead(dimension).to(device)
    model = RawSubjectModel(encoder, head, adapter, selected, specification.input_seconds, window_chunk)
    audit.update(native_dimension=dimension, fresh_head_seed=seed, fresh_head=True,
                 initialized_from_frozen_best_head=False, selected_encoder_parameter_names=selected,
                 planned_update_parameter_names=list(model.update_names))
    return model, audit


def set_train_phase(model, epoch):
    """Activate only complete trainable subtrees; frozen dropout stays eval."""
    selected = set(model.selected_names) if epoch > POLICY.warmup_epochs else set()
    for name, parameter in model.encoder.named_parameters():
        parameter.requires_grad_(name in selected)
    model.eval()
    model.head.train()
    enabled = []
    for name, module in model.encoder.named_modules():
        if any(name == scope or name.startswith(scope + ".") for scope in enabled):
            continue
        prefix = name + "." if name else ""
        parameters = {prefix + parameter_name for parameter_name, _ in module.named_parameters()}
        if parameters and parameters.issubset(selected):
            module.train(True)
            enabled.append(name)
    require(all(not parameter.requires_grad for name, parameter in model.encoder.named_parameters() if name not in selected), "Frozen parameter trainability changed")
    return enabled


def optimizer_for(model):
    parameters = dict(model.encoder.named_parameters())
    return torch.optim.AdamW([{"params": list(model.head.parameters()), "lr": POLICY.head_lr, "group_name": "head"},
                             {"params": [parameters[name] for name in model.selected_names], "lr": 0., "group_name": "backbone"}],
                            weight_decay=POLICY.weight_decay)


def set_learning_rates(optimizer, epoch):
    for group in optimizer.param_groups:
        name = group["group_name"]
        base = POLICY.head_lr if name == "head" else POLICY.backbone_lr
        group["lr"] = base * P.learning_rate_multiplier("finetune", epoch, parameter_group=name)


def frozen_state(model):
    excluded = set(model.update_names)
    # Frozen buffers are deliberately included. Mode control keeps their state
    # unchanged; a model with trainable running-stat buffers needs a separately
    # documented adapter/contract instead of a silent exemption.
    return {name: tensor for name, tensor in model.state_dict().items() if name not in excluded}


def selected_state(model):
    names = set(model.update_names)
    return cpu_state({name: tensor for name, tensor in model.state_dict().items() if name in names})


def load_selected_state(model, state):
    require(set(state) == set(model.update_names), "Selected state parameter coverage differs")
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current, strict=True)


def _raw_sources(split, *, people=None):
    manifest = R.load_registration()
    by_id = {row["subject_id"]: row["raw"] for row in manifest["raw_sources"]}
    chosen = split["train"] + split["val"] if people is None else people
    require(len(chosen) == len(set(chosen)) and not set(chosen) & set(split["test"]), "Raw load roles overlap outer test")
    require(set(chosen).issubset(set(split["train"] + split["val"])), "Raw request includes an unregistered role")
    raw, sources = {}, {}
    for sid in chosen:
        source = S.checked(by_id[sid])
        array = np.load(source, allow_pickle=False)
        require(array.shape == (60, 19, 1000) and array.dtype == np.float32 and np.isfinite(array).all(), "Accepted raw shape/dtype changed")
        raw[sid], sources[sid] = array, by_id[sid]
    return raw, sources


def subject_update(model, optimizer, people, raw, labels, window_ids, *, micro_subjects):
    """Sum microbatch subject CE / actual logical people, then one update."""
    require(1 <= len(people) <= 8 and micro_subjects in (1, 2), "Logical batch must contain 1..8 people and microbatch 1/2")
    require(len(people) == len(set(people)), "A person must contribute exactly one CE to this logical batch")
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    backward_calls = 0
    for start in range(0, len(people), micro_subjects):
        chosen = people[start:start + micro_subjects]
        batch = np.stack([raw[sid][window_ids] for sid in chosen])
        target = torch.tensor([labels[sid] for sid in chosen], device=model.device, dtype=torch.long)
        ce_sum = torch.nn.functional.cross_entropy(model(batch), target, reduction="sum")
        require(torch.isfinite(ce_sum), "Nonfinite raw-bag subject CE")
        (ce_sum / len(people)).backward()
        total += float(ce_sum.detach().cpu())
        backward_calls += 1
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require(parameter.grad is not None and torch.isfinite(parameter.grad).all(), "Nonfinite or disconnected trainable gradient: " + name)
        else:
            require(parameter.grad is None, "Frozen parameter acquired a gradient: " + name)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], POLICY.gradient_clip)
    optimizer.step()
    return {"subject_ce_sum": total, "subjects": len(people), "backward_calls": backward_calls}


def diagnostics(model, raw, split, labels):
    """Read-only all60-window train/val diagnostics, with full RNG isolation."""
    state = rng_state(model.device)
    modes = [(module, module.training) for module in model.modules()]
    before = state_digest(model.state_dict())
    output = {}
    try:
        model.eval()
        with torch.inference_mode():
            for role in ("train", "val"):
                people = split[role]
                ce = 0.
                for sid in people:
                    require(sid in raw and sid not in split["test"], "Outer-test person reached diagnostics")
                    logits = model(raw[sid][None])
                    target = torch.tensor([labels[sid]], device=model.device)
                    loss = torch.nn.functional.cross_entropy(logits, target)
                    require(torch.isfinite(loss), "Nonfinite all-window diagnostic CE")
                    ce += float(loss.cpu())
                output[role + "_eval_ce"] = ce / len(people)
        require(before == state_digest(model.state_dict()), "Diagnostics mutated parameter/buffer state")
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(state)
    require(rng_equal(state, rng_state(model.device)), "Diagnostics failed to restore RNG")
    return output


def _contract_matches(condition, model, micro_subjects, window_chunk):
    descriptor = R._condition_lock(condition)
    contract = S.read(S.checked(descriptor))["contract"]
    require(set(contract["trainable_parameter_names"]) == set(model.update_names), "Engineering and runtime parameter contracts differ")
    require(contract["input_contract"]["micro_subjects"] == micro_subjects
            and contract["input_contract"]["window_chunk"] == window_chunk, "Microbatch/window chunk changed after engineering lock")
    return descriptor


def _implementation_sources(condition, model, audit_sources):
    """Pin dynamically imported vendor code, not a mutable download manifest."""
    records = {record["path"]: record for record in audit_sources}
    for path in (Path(__file__), Path(model.adapter.__file__), Path(S.__file__),
                 Path(C.__file__), Path(P.__file__), Path(R.__file__)):
        record = S.desc(path)
        records[record["path"]] = record
    support = Path(model.adapter.__file__).parent / "_author_support.py"
    if support.exists():
        record = S.desc(support)
        records[record["path"]] = record
    manifest_path = S.OUT / "assets/download_manifest.json"
    if manifest_path.exists():
        source = S.read(manifest_path).get("sources", {}).get(P.CONDITION_BY_ID[condition].method.lower(), {})
        for item in source.get("files", []):
            if item["path"].endswith(".py"):
                record = {"path": item["path"], "sha256": item["sha256"]}
                S.checked(record)
                records[record["path"]] = record
    for record in records.values():
        S.checked(record)
    return list(records.values())


def _recover_epoch_zero(folder, value):
    """Rebuild the deterministic initial state only before any batch started."""
    require(not (folder / "initial.pt").exists() and not (folder / "latest.pt").exists(),
            "Epoch-zero recovery cannot replace an existing checkpoint")
    require(not (folder / "progress.json").exists(), "Missing state after an optimizer batch started; epoch-zero reconstruction refused")
    require(value["epoch"] == value["optimizer_updates"] == value["backward_calls"] == 0
            and not value["history"] and not value["subject_order"] and value["best_selected_state_dict"] is None,
            "Initial reconstruction must contain no training progress")
    S.save_torch(folder / "initial.pt", value, immutable=True)


def prepare_condition(condition, *, micro_subjects=1, window_chunk=8):
    """Explicit real training-person probe, with zero optimizer updates."""
    require(torch.cuda.is_available(), "Engineering probes require the declared CUDA execution environment")
    R.load_registration()
    existing = S.OUT / "engineering" / (condition + ".json")
    if existing.exists():
        R._condition_lock(condition)
        return S.read(existing)
    cohort, folds, labels = C.context()
    split = folds["fold_1"]
    chosen = split["train"][:micro_subjects]
    raw, sources = _raw_sources(split, people=chosen)
    model, audit = build_model(condition, device="cuda", window_chunk=window_chunk)
    C.runtime(11)
    selected = list(model.update_names)
    initial = state_digest(model.state_dict())
    optimizer = optimizer_for(model)
    peak_start = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    checks = []
    try:
        for epoch in (1, 11):
            set_train_phase(model, epoch)
            optimizer.zero_grad(set_to_none=True)
            target = torch.tensor([labels[sid] for sid in chosen], device=model.device)
            logits = model(np.stack([raw[sid][bag_indices(epoch)] for sid in chosen]))
            loss = torch.nn.functional.cross_entropy(logits, target)
            require(torch.isfinite(loss), "Engineering forward loss nonfinite")
            loss.backward()
            actual = [n for n, p in model.named_parameters() if p.requires_grad]
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad), "Engineering last-block/head gradients disconnected")
            require(all(p.grad is None for p in model.parameters() if not p.requires_grad), "Engineering frozen gradients changed")
            checks.append({"epoch_phase": epoch, "actual_trainable_parameter_names": actual, "ce": float(loss.cpu())})
        require(state_digest(model.state_dict()) == initial, "Zero-update engineering probe changed model buffers")
        state = rng_state(model.device)
        probe = S.OUT / "finetune_engineering" / condition / "resume_probe.pt"
        value = {"state_dict": cpu_state(model.state_dict()), "optimizer_state_dict": cpu_state(optimizer.state_dict()),
                 "scheduler_state_dict": {"policy": "finetune", "completed_epoch": 0, "extended": False},
                 "rng": state, "sampler_position": {"epoch_completed": 0, "next_batch": 0}}
        S.save_torch(probe, value, immutable=True)
        restored = torch.load(probe, map_location="cpu", weights_only=False)
        model.load_state_dict(restored["state_dict"], strict=True)
        optimizer.load_state_dict(restored["optimizer_state_dict"])
        restore_rng(restored["rng"])
        require(state_digest(model.state_dict()) == initial and rng_equal(state, rng_state(model.device)), "Engineering resume roundtrip failed")
        # Gather hash-bound files from heterogeneous adapter audit structures.
        def descriptors(value):
            if isinstance(value, dict):
                if "path" in value and "sha256" in value:
                    yield {"path": value["path"], "sha256": value["sha256"]}
                else:
                    for child in value.values():
                        yield from descriptors(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    yield from descriptors(child)
        provenance = {item["path"]: item for item in descriptors(audit)}
        for item in provenance.values():
            S.checked(item)
        require(provenance, "Adapter did not supply checkpoint/source descriptors")
        contract = {"input_contract": {"adapter": audit, "windows_per_training_subject": 8,
                    "all_window_validation": 60, "coverage_cycle_epochs": 8, "micro_subjects": micro_subjects,
                    "window_chunk": window_chunk, "effective_subjects": 8, "precision": "float32"},
                    "trainable_parameter_names": selected,
                    "frozen_parameter_names": [n for n, _ in model.named_parameters() if n not in selected],
                    "implementation_sources": _implementation_sources(condition, model,
                                             [record for record in provenance.values() if record["path"].endswith(".py")]),
                    "weight_sources": list(provenance.values()), "forward_checked": True, "gradient_checked": True,
                    "resume_checked": True, "training_only_engineering_checked": True,
                    "probe_subject_ids": chosen, "raw_sources": {sid: sources[sid] for sid in chosen},
                    "engineering_optimizer_updates": 0, "engineering_backward_calls": 2,
                    "gradient_checks": checks, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "baseline_allocated_bytes": peak_start, "seconds": time.perf_counter() - started,
                    "initial_state_sha256": initial, "resume_probe": S.desc(probe)}
        R.lock_condition(condition, contract)
        for seed in P.SEEDS:
            for fold in P.FOLDS:
                R.record_fit_status(condition, seed, fold, "ready")
        return contract
    finally:
        del model, optimizer, raw
        gc.collect()
        torch.cuda.empty_cache()


def fit(condition, seed, fold, *, resume=False, micro_subjects=1, window_chunk=8):
    require(condition in CONDITIONS and seed in P.SEEDS and fold in P.FOLDS, "Unregistered fit identity")
    require(micro_subjects in (1, 2) and window_chunk > 0, "Invalid engineering microbatch")
    require(torch.cuda.is_available(), "Formal fine-tuning requires CUDA")
    folder = R.fit_directory(condition, seed, fold)
    if (folder / "complete.json").exists():
        evidence = S.read(folder / "complete.json")
        R._validate_complete(evidence, folder)
        if R.fit_status(condition, seed, fold)["status"] != "training_complete":
            R.record_fit_status(condition, seed, fold, "training_complete", evidence=evidence)
        return S.read(folder / "metadata.json")
    partial = (folder / "started.json").exists()
    require(not partial or resume, "Partial fit requires full-state --resume")
    # Refuse to manufacture readiness during formal fitting.
    R._condition_lock(condition)
    cohort, folds, labels = C.context()
    split = folds[f"fold_{fold}"]
    raw, raw_sources = _raw_sources(split)
    model, audit = build_model(condition, seed=seed, device="cuda", window_chunk=window_chunk)
    source_lock = _contract_matches(condition, model, micro_subjects, window_chunk)
    C.runtime(seed)
    optimizer = optimizer_for(model)
    initial = state_digest(model.state_dict())
    initial_selected = selected_state(model)
    frozen_initial = state_digest(frozen_state(model))
    history, order, best, epoch, extended, updates, backwards = [], [], None, 0, False, 0, 0
    R.record_fit_status(condition, seed, fold, "training")
    if not partial:
        S.save(folder / "started.json", {"condition": condition, "seed": seed, "fold": fold, "started_at": S.now(),
               "train_ids": split["train"], "val_ids": split["val"], "test_forward_count": 0,
               "condition_source_lock": source_lock, "raw_sources": raw_sources,
               "fresh_formal_pretrained": True, "fresh_head_seed": seed, "frozen_best_head_inherited": False}, immutable=True)

    def payload():
        return {"condition": condition, "seed": seed, "fold": fold, "registry": R.registration_descriptor(),
                "condition_source_lock": source_lock, "raw_sources": raw_sources,
                "state_dict": cpu_state(model.state_dict()), "optimizer_state_dict": cpu_state(optimizer.state_dict()),
                "scheduler_state_dict": {"policy": "finetune", "completed_epoch": epoch, "extended": extended},
                "rng": rng_state(model.device), "sampler_position": {"epoch_completed": epoch, "next_batch": 0},
                "epoch": epoch, "history": history, "subject_order": order, "best_selected_state_dict": best,
                "best_selected_state_sha256": state_digest(best) if best else None,
                "initial_state_sha256": initial, "frozen_initial_sha256": frozen_initial,
                "optimizer_updates": updates, "backward_calls": backwards}

    durable_epoch = 0
    started, start_updates = time.perf_counter(), 0
    torch.cuda.reset_peak_memory_stats()
    try:
        if partial:
            target = folder / "latest.pt" if (folder / "latest.pt").exists() else folder / "initial.pt"
            if not target.exists():
                started_record = S.read(folder / "started.json")
                require(all(started_record[key] == expected for key, expected in {
                    "condition": condition, "seed": seed, "fold": fold,
                    "condition_source_lock": source_lock, "raw_sources": raw_sources,
                    "train_ids": split["train"], "val_ids": split["val"],
                    "fresh_head_seed": seed}.items()), "Epoch-zero start identity/source differs")
                _recover_epoch_zero(folder, payload())
            saved = torch.load(target, map_location="cpu", weights_only=False)
            require(all(saved[key] == expected for key, expected in {"condition": condition, "seed": seed, "fold": fold,
                    "registry": R.registration_descriptor(), "condition_source_lock": source_lock, "raw_sources": raw_sources,
                    "initial_state_sha256": initial, "frozen_initial_sha256": frozen_initial}.items()), "Resume identity/source/initial state differs")
            epoch, history, order, best = saved["epoch"], saved["history"], saved["subject_order"], saved["best_selected_state_dict"]
            updates, backwards = saved["optimizer_updates"], saved["backward_calls"]
            extended = saved["scheduler_state_dict"]["extended"]
            require(0 <= epoch <= POLICY.final_epochs and len(history) == len(order) == epoch
                    and saved["sampler_position"] == {"epoch_completed": epoch, "next_batch": 0}
                    and saved["scheduler_state_dict"]["completed_epoch"] == epoch
                    and saved["scheduler_state_dict"]["policy"] == "finetune", "Resume epoch/scheduler/sampler differs")
            expected_updates = math.ceil(len(split["train"]) / 8) * epoch
            updates_per_epoch = math.ceil(len(split["train"]) / 8)
            backwards_per_epoch = (len(split["train"]) // 8 * math.ceil(8 / micro_subjects)
                                   + math.ceil(len(split["train"]) % 8 / micro_subjects))
            require(updates == expected_updates and backwards == backwards_per_epoch * epoch
                    and all(sorted(people) == sorted(split["train"]) for people in order), "Resume actual updates/backward calls/subject coverage differs")
            for index, row in enumerate(history, 1):
                require(row["epoch"] == index and row["optimizer_updates"] == updates_per_epoch * index
                        and row["backward_calls"] == backwards_per_epoch * index
                        and row["window_indices"] == bag_indices(index).tolist()
                        and row["unfrozen"] is (index > 10), "Resume epoch/window schedule/phase history differs")
            require(all(torch.isfinite(value).all() for value in saved["state_dict"].values()), "Resume model contains nonfinite values")
            require((best is None and epoch == 0) or (best is not None and state_digest(best) == saved["best_selected_state_sha256"]), "Resume best state differs")
            model.load_state_dict(saved["state_dict"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            group_names = [group.get("group_name") for group in optimizer.param_groups]
            require(group_names == ["head", "backbone"], "Resume optimizer group identity differs")
            for group, expected_steps in zip(optimizer.param_groups, (updates, math.ceil(len(split["train"]) / 8) * max(0, epoch - 10))):
                for parameter in group["params"]:
                    state = optimizer.state.get(parameter, {})
                    require((expected_steps == 0 and not state) or (state and float(state["step"]) == expected_steps), "Resume optimizer step count differs")
            restore_rng(saved["rng"])
            require(rng_equal(saved["rng"], rng_state(model.device)) and state_digest(frozen_state(model)) == frozen_initial, "Resume RNG/frozen state differs")
        else:
            S.save_torch(folder / "initial.pt", payload(), immutable=True)
        durable_epoch, start_updates = epoch, updates
        S.save(folder / "train_history.json", history)
        S.save(folder / "validation_history.json", [{"epoch": row["epoch"], "val_ce": row["val_eval_ce"]} for row in history])
        decision = P.training_decision("finetune", [r["val_eval_ce"] for r in history], extended=extended) if history else {"stop": False}
        while not decision["stop"]:
            epoch += 1
            modes = set_train_phase(model, epoch)
            set_learning_rates(optimizer, epoch)
            people = [split["train"][i] for i in torch.randperm(len(split["train"])).tolist()]
            windows = bag_indices(epoch)
            epoch_ce, epoch_people = 0., 0
            for start in range(0, len(people), 8):
                logical = people[start:start + 8]
                S.save(folder / "progress.json", {"epoch": epoch, "optimizer_updates_started": updates + 1,
                       "optimizer_updates_completed": updates, "batch_subject_ids": logical,
                       "window_indices": windows.tolist(), "last_durable_epoch": durable_epoch})
                result = subject_update(model, optimizer, logical, raw, labels, windows, micro_subjects=micro_subjects)
                updates += 1
                backwards += result["backward_calls"]
                epoch_ce += result["subject_ce_sum"]
                epoch_people += result["subjects"]
                S.save(folder / "progress.json", {"epoch": epoch, "optimizer_updates_completed": updates,
                       "backward_calls_completed": backwards, "last_durable_epoch": durable_epoch})
            require(epoch_people == len(split["train"]) and state_digest(frozen_state(model)) == frozen_initial, "Epoch coverage/frozen state changed")
            order.append(people)
            history.append({"epoch": epoch, "train_bag_ce": epoch_ce / epoch_people,
                            "optimizer_updates": updates, "backward_calls": backwards, "window_indices": windows.tolist(),
                            "cycle": (epoch - 1) // 8 + 1, "cycle_epoch": (epoch - 1) % 8 + 1,
                            "repeated_padding_windows_this_epoch": [0, 1, 2, 3] if epoch % 8 == 0 else [],
                            "head_lr": optimizer.param_groups[0]["lr"], "backbone_lr": optimizer.param_groups[1]["lr"],
                            "unfrozen": epoch > 10, "training_module_subtrees": modes,
                            **diagnostics(model, raw, split, labels)})
            decision = P.training_decision("finetune", [r["val_eval_ce"] for r in history], extended=extended)
            if decision["best_epoch"] == epoch:
                best = selected_state(model)
            if decision["extend"]:
                extended = True
            S.save_torch(folder / "latest.pt", payload())
            durable_epoch = epoch
            S.save(folder / "train_history.json", history)
            S.save(folder / "validation_history.json", [{"epoch": row["epoch"], "val_ce": row["val_eval_ce"]} for row in history])
            if epoch % 20 == 0:
                print(f"FINETUNE {condition} seed={seed} fold={fold} epoch={epoch} best={decision['best_epoch']}", flush=True)
        require(best is not None and epoch > POLICY.warmup_epochs, "No validation best state or actual backbone training")
        latest_selected = selected_state(model)
        head_names = [n for n in latest_selected if n.startswith("head.")]
        encoder_names = [n for n in latest_selected if n.startswith("encoder.")]
        changed = {n: not torch.equal(latest_selected[n], initial_selected[n]) for n in latest_selected}
        require(all(changed[n] for n in head_names) and all(changed[n] for n in encoder_names), "At least one intended head/last-block parameter never updated")
        load_selected_state(model, best)
        require(state_digest(frozen_state(model)) == frozen_initial, "Best checkpoint altered frozen state")
        model.eval()
        best_path = folder / "best.pt"
        best_payload = {"state_dict": cpu_state(model.state_dict()), "best_epoch": decision["best_epoch"],
                        "registry": R.registration_descriptor(), "condition_source_lock": source_lock,
                        "condition": condition, "seed": seed, "fold": fold}
        S.save_torch(best_path, best_payload, immutable=True)
        probe_people = split["train"][:1]
        with torch.inference_mode():
            reference = model(np.stack([raw[sid] for sid in probe_people])).cpu()
            reloaded_state = torch.load(best_path, map_location="cpu", weights_only=False)["state_dict"]
            model.load_state_dict(reloaded_state, strict=True)
            require(torch.equal(reference, model(np.stack([raw[sid] for sid in probe_people])).cpu()), "Reloaded all-window subject prediction differs")
        metadata = {"condition": condition, "seed": seed, "fold": fold, "epochs": epoch, "optimizer_updates": updates,
                    "backward_calls": backwards, "head_optimizer_updates": updates,
                    "backbone_optimizer_updates": math.ceil(len(split["train"]) / 8) * max(0, epoch - 10),
                    "best_epoch": decision["best_epoch"], "stop_reason": decision["reason"], "extended": extended,
                    "train_ids": split["train"], "val_ids": split["val"], "test_ids": split["test"], "test_forward_count": 0,
                    "raw_sources": raw_sources, "adapter_audit": audit, "condition_source_lock": source_lock,
                    "initial_state_sha256": initial, "selected_state_sha256": state_digest(model.state_dict()),
                    "frozen_initial_sha256": frozen_initial, "frozen_final_sha256": state_digest(frozen_state(model)),
                    "parameter_changed_after_training": changed,
                    "micro_subjects": micro_subjects, "window_chunk": window_chunk, "effective_subject_batch": 8,
                    "window_bag_size": 8, "eval_window_count": 60, "full_cycle_unique_windows": 60,
                    "full_cycle_padding_windows": [0, 1, 2, 3], "precision": "float32",
                    "fresh_official_pretrained": True, "fresh_head": True, "frozen_best_head_inherited": False,
                    "reload_probe_subject_ids": probe_people, "seconds_this_attempt": time.perf_counter() - started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        metadata_path = folder / "metadata.json"
        if metadata_path.exists():
            previous = S.read(metadata_path)
            require(all(previous[k] == metadata[k] for k in ("condition", "seed", "fold", "epochs", "optimizer_updates", "best_epoch", "stop_reason", "selected_state_sha256", "frozen_final_sha256")), "Existing complete metadata differs on recovery")
            metadata = previous
        else:
            S.save(metadata_path, metadata, immutable=True)
        evidence = {"passed": True, "checkpoint_reload_verified": True, "parameter_update_verified": True,
                    "frozen_parameters_verified": True, "epochs": epoch, "optimizer_updates": updates,
                    "stop_reason": decision["reason"], "registry": R.registration_descriptor(),
                    "checkpoint": S.desc(best_path), "train_history": S.desc(folder / "train_history.json"),
                    "validation_history": S.desc(folder / "validation_history.json"), "metadata": S.desc(metadata_path)}
        S.save(folder / "complete.json", evidence, immutable=True)
        R.record_fit_status(condition, seed, fold, "training_complete", evidence=evidence)
        print(f"COMPLETE {condition} seed={seed} fold={fold} epochs={epoch} updates={updates}", flush=True)
        return metadata
    except BaseException as error:
        S.save(folder / ("failure_" + str(time.time_ns()) + ".json"), {"error": str(error), "error_type": type(error).__name__,
               "epoch_started": epoch, "last_durable_epoch": durable_epoch, "optimizer_updates": updates,
               "updates_completed_this_process": updates - start_updates, "time": S.now()})
        R.record_fit_status(condition, seed, fold, "failed", reason=str(error))
        raise
    finally:
        del model, optimizer, raw
        gc.collect()
        torch.cuda.empty_cache()


def run(conditions=CONDITIONS, *, resume=False, limit=None, micro_subjects=1, window_chunk=8, prepare=False):
    require(limit is None or limit > 0, "Limit must be positive")
    completed = 0
    for condition in conditions:
        require(condition in CONDITIONS, "Unregistered finetune condition")
        if prepare:
            prepare_condition(condition, micro_subjects=micro_subjects, window_chunk=window_chunk)
            continue
        for seed in P.SEEDS:
            for fold in P.FOLDS:
                fit(condition, seed, fold, resume=resume, micro_subjects=micro_subjects, window_chunk=window_chunk)
                completed += 1
                if limit is not None and completed >= limit:
                    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--prepare", action="store_true", help="Explicit zero-update training-person engineering checks only")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--micro-subjects", type=int, choices=(1, 2), default=1)
    parser.add_argument("--window-chunk", type=int, default=8)
    args = parser.parse_args()
    run(args.conditions, resume=args.resume, limit=args.limit, micro_subjects=args.micro_subjects,
        window_chunk=args.window_chunk, prepare=args.prepare)

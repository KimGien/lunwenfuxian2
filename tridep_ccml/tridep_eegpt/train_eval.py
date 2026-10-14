"""Paired original CNN fitting and fixed-weight evaluation on matched4s inputs.

The caller owns top-level ``fit_all``/``evaluate`` action markers. A fit start
is persisted before constructing its model, and any partial fit is preserved.
No validation/test head forward is allowed until all30 fits have completed.
"""
from __future__ import annotations

from collections import Counter
import gc
import hashlib
import math
from pathlib import Path
from statistics import mean
import time

from . import common as C

np, torch = C.np, C.torch
CONDITIONS = tuple(C.CONDITIONS)
SEEDS = tuple(C.SEEDS)
TRAINING = dict(C.TRAINING)
VIEWS = dict(C.VIEWS)
KEYS = tuple(C.KEYS)
FEATURE_KEYS = dict(zip(CONDITIONS, ("labram", "eegpt")))
FEATURE_ARRAY_SHAPES = {
    "eegpt": (60, 200), "labram": (60, 200),
    "eegpt_input": (60, 19, 1024), "labram_input": (60, 19, 800),
    "eegpt_tokens": (60, 16, 4, 512), "labram_tokens": (60, 76, 200),
    "eegpt_native512": (60, 512),
    "eegpt_repeat_tokens": (8, 16, 4, 512), "labram_repeat_tokens": (8, 76, 200),
    "eegpt_repeat_native512": (8, 512), "eegpt_repeat": (8, 200), "labram_repeat": (8, 200),
}


def original_model():
    from tridep_ccml.tridep_stage1.run import original_model as construct
    return construct()


def require(value, message):
    if not value:
        raise ValueError(message)


def _reg_desc():
    return C.desc(C.OUT / "registration.json")


def validate_context(lock, folds, labels):
    ids = lock["subject_ids"]
    require(len(ids) == 37 and len(set(ids)) == 37, "Requires original37 unique people")
    require(set(labels) == set(ids) and set(labels.values()) == {0, 1}, "Original binary label coverage differs")
    require(set(folds) == {f"fold_{f}" for f in range(1, 6)}, "Requires original five folds")
    held_out = []
    for split in folds.values():
        require(set(split) == {"train", "val", "test"}, "Fold roles differ")
        require(len(split["train"]) in (26, 27) and len(split["val"]) == 3
                and len(split["test"]) in (7, 8), "Original role counts differ")
        flat = sum((split[part] for part in ("train", "val", "test")), [])
        require(len(flat) == 37 and len(set(flat)) == 37 and set(flat) == set(ids),
                "Train/val/test person isolation failure")
        require(all(set(labels[s] for s in split[part]) == {0, 1} for part in split),
                "Both classes required in each original role")
        held_out.extend(split["test"])
    require(Counter(held_out) == Counter(ids), "Each original person must test exactly once")


def _entry_guard(action):
    reg = C.registration()
    require(reg["training"] == {c: TRAINING for c in CONDITIONS}
            and reg["formal_fits"] == 30 and reg["formal_updates"] == 3600
            and reg["training_gradient_passes"] == 3600 and reg["fusion_selections"] == 0,
            "Registered fitting or no-selection contract differs")
    marker = C.read(C.OUT / "actions" / f"{action}.json")
    require(marker["status"] == "started" and marker["registration"] == _reg_desc(),
            "Root must persist the current action start before calling this entry")
    inputs = C.read(C.REPORT / "input_independent_verification.json")
    require(inputs["passed"] and inputs["registration"] == _reg_desc(),
            "Current independent input verification required before fitting/scoring")
    return reg


def load_features(lock):
    path = C.OUT / "features_manifest.json"
    manifest = C.read(path)
    require(manifest.get("passed", True), "Feature producer did not pass")
    if "registration" in manifest:
        require(manifest["registration"] == _reg_desc(), "Feature registration differs")
    records = manifest["records"]
    ids = lock["subject_ids"]
    require(len(records) == 37 and len({r["subject_id"] for r in records}) == 37
            and {r["subject_id"] for r in records} == set(ids), "Feature person coverage differs")
    by_id = {r["subject_id"]: r for r in records}
    arrays = {name: [] for name in FEATURE_KEYS.values()}
    for sid in ids:
        descriptor = by_id[sid]["features"]
        if "bundle" in by_id[sid]:
            require(by_id[sid]["bundle"] == descriptor, "Feature bundle alias differs")
        with np.load(C.checked(descriptor), allow_pickle=False) as data:
            require(set(data.files) == set(FEATURE_ARRAY_SHAPES), "Feature archive audit-key allowlist differs")
            for name, shape in FEATURE_ARRAY_SHAPES.items():
                x = data[name]
                require(x.shape == shape and x.dtype == np.float32 and np.isfinite(x).all(),
                        "Invalid frozen feature/audit array: " + name)
                if name in arrays:
                    arrays[name].append(x.copy())
    return {name: np.stack(values) for name, values in arrays.items()}, C.desc(path)


def batch_order_digest(order):
    return hashlib.sha256(np.asarray(order, dtype="<i8").tobytes()).hexdigest()


def validate_batch_order(order, indices):
    order = np.asarray(order)
    require(order.dtype == np.int64 and order.shape == (60, len(indices)), "Batch-order shape/dtype differs")
    expected = Counter(indices)
    require(all(Counter(row.tolist()) == expected for row in order),
            "Each epoch must use only original train people exactly once")


def training_dataset(ids, train_ids, labels):
    """Create only original train indices/labels; never access held-out labels."""
    indices = [ids.index(s) for s in train_ids]
    return indices, torch.utils.data.TensorDataset(torch.tensor(indices, dtype=torch.long),
                   torch.tensor([labels[s] for s in train_ids], dtype=torch.long))


def _save_array(path, values, compressed=False):
    path = Path(path)
    require(not path.exists(), "Refusing to replace array artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        if compressed:
            np.savez_compressed(stream, **values)
        else:
            np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    return value


def _save_torch(path, value):
    require(not path.exists(), "Refusing to replace checkpoint")
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(_cpu(value), temporary)
    temporary.replace(path)


def _folder(condition, seed, fold):
    return C.OUT / f"runs/{condition}/seed_{seed}/fold_{fold}"


def fit_all():
    _entry_guard("fit_all")
    lock, folds, labels = C.context()
    validate_context(lock, folds, labels)
    for condition in CONDITIONS:
        for seed in SEEDS:
            for fold in range(1, 6):
                require(not _folder(condition, seed, fold).exists(),
                        "Existing complete/partial fit requires explicit review; no automatic retry")
    arrays, feature_manifest = load_features(lock)
    tensors = {key: torch.as_tensor(x, device="cuda") for key, x in arrays.items()}
    ids = lock["subject_ids"]
    completed = []
    for seed in SEEDS:
        for fold in range(1, 6):
            split = folds[f"fold_{fold}"]
            pair = None
            for condition in CONDITIONS:
                folder = _folder(condition, seed, fold)
                require(not folder.exists(), "Existing complete/partial fit requires explicit review; no automatic retry")
                folder.mkdir(parents=True, exist_ok=False)
                C.save(folder / "started.json", {"condition": condition, "seed": seed, "fold": fold,
                       "started_at": C.now(), "train_ids": split["train"], "registration": _reg_desc(),
                       "planned_updates": 120, "feature_manifest": feature_manifest}, immutable=True)
                # The durable fit start precedes runtime/model construction.
                C.runtime(seed)
                model = original_model().to("cuda")
                require(sum(p.numel() for p in model.parameters()) == 498434
                        and all(p.requires_grad for p in model.parameters()), "Original CNN parameter contract differs")
                initial = C.state_digest(model.state_dict())
                _save_torch(folder / "initial.pt", {"state_dict": model.state_dict(), "registration": _reg_desc(),
                                                   "condition": condition, "seed": seed, "fold": fold})
                indices, dataset = training_dataset(ids, split["train"], labels)
                loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
                optimizer = torch.optim.Adamax(model.parameters(), lr=.005, weight_decay=0.)
                updates = gradients = 0
                history, order = [], []
                started = time.perf_counter()
                for epoch in range(60):
                    model.train()
                    epoch_order, losses, count = [], [], 0
                    for ii, yy in loader:
                        require(all(int(i) in indices for i in ii), "Nontrain person reached a fit forward")
                        require(yy.tolist() == [labels[ids[int(i)]] for i in ii], "Training label/source mismatch")
                        C.save(folder / "progress.json", {"epoch": epoch + 1, "updates_started": updates + 1,
                               "updates_completed": updates, "gradient_passes_completed": gradients,
                               "batch_subject_ids": [ids[int(i)] for i in ii], "registration": _reg_desc()})
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(tensors[FEATURE_KEYS[condition]][ii.to("cuda")])
                        loss = torch.nn.functional.cross_entropy(logits, yy.to("cuda"))
                        require(torch.isfinite(loss), "Nonfinite train CE")
                        loss.backward()
                        gradients += 1
                        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),
                                "Nonfinite or disconnected head gradient")
                        optimizer.step()
                        require(all(torch.isfinite(p).all() for p in model.parameters()), "Nonfinite updated head")
                        updates += 1
                        epoch_order.extend(ii.tolist())
                        losses.append({"batch_size": len(ii), "train_ce": float(loss.detach().cpu())})
                        count += len(ii)
                        C.save(folder / "progress.json", {"epoch": epoch + 1, "updates_started": updates,
                               "updates_completed": updates, "gradient_passes_completed": gradients,
                               "registration": _reg_desc()})
                    require(count == len(indices), "Training epoch person count differs")
                    order.append(epoch_order)
                    history.append({"epoch": epoch + 1, "updates_completed": updates,
                                    "gradient_passes_completed": gradients,
                                    "train_ce": sum(v["train_ce"] * v["batch_size"] for v in losses) / count,
                                    "batches": losses})
                    C.save(folder / "train_history.json", history)
                    if (epoch + 1) % 20 == 0:
                        print(f"TRAIN {condition} seed={seed} fold={fold} epoch={epoch+1}/60", flush=True)
                require(updates == gradients == 120, "Fit update/gradient budget differs")
                order_array = np.asarray(order, dtype=np.int64)
                validate_batch_order(order_array, indices)
                _save_array(folder / "batch_order.npy", order_array)
                model.eval()
                meta = {"condition": condition, "seed": seed, "fold": fold, "cohort_id": lock["cohort_id"],
                        "folds_sha256": lock["folds_sha256"], "train_ids": split["train"], "val_ids": split["val"],
                        "test_ids": split["test"], "training": TRAINING, "registration": _reg_desc(),
                        "initial_state_sha256": initial, "final_state_sha256": C.state_digest(model.state_dict()),
                        "batch_order_sha256": batch_order_digest(order_array), "updates": updates,
                        "gradient_passes": gradients, "epochs": 60, "parameters": 498434,
                        "input_key": FEATURE_KEYS[condition], "feature_manifest": feature_manifest,
                        "input_independent_verification": C.desc(C.REPORT / "input_independent_verification.json"),
                        "no_val_test_forward_during_training": True, "checkpoint_selection": "fixed_final_epoch",
                        "seconds": time.perf_counter() - started}
                current_pair = (initial, meta["batch_order_sha256"])
                require(pair is None or pair == current_pair, "Conditions are not paired in initialization and order")
                pair = current_pair
                _save_torch(folder / "checkpoint.pt", {"state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(), "metadata": meta})
                C.save(folder / "metadata.json", meta, immutable=True)
                artifacts = [C.desc(folder / name) for name in ("started.json", "initial.pt", "checkpoint.pt",
                             "metadata.json", "train_history.json", "batch_order.npy", "progress.json")]
                C.save(folder / "complete.json", {"passed": True, "registration": _reg_desc(),
                       "updates": 120, "artifacts": artifacts}, immutable=True)
                completed.append(meta)
                C.save(C.OUT / "training_ledger.json", {"fits_completed": len(completed),
                       "updates_completed": sum(m["updates"] for m in completed),
                       "gradient_passes_completed": sum(m["gradient_passes"] for m in completed),
                       "planned_fits": 30, "planned_updates": 3600, "fits": completed, "metrics_released": False,
                       "registration": _reg_desc()})
                C.save(C.REPORT / "status.json", {"status": "training" if len(completed) < 30 else
                       "all_fits_complete_pending_uniform_evaluation", "fits_completed": len(completed),
                       "updates_completed": sum(m["updates"] for m in completed), "metrics_released": False})
                del model, optimizer
                gc.collect()
    require(len(completed) == 30 and sum(m["updates"] for m in completed) == 3600,
            "All30 fits must finish before evaluation")
    C.registration()
    return C.read(C.OUT / "training_ledger.json")


def completed_fits(lock, folds):
    """Verify every fit/artifact/pair without constructing or forwarding a model."""
    ledger = C.read(C.OUT / "training_ledger.json")
    require(ledger["fits_completed"] == 30 and ledger["updates_completed"] == 3600
            and ledger["gradient_passes_completed"] == 3600 and len(ledger["fits"]) == 30
            and ledger["registration"] == _reg_desc(), "Uniform release requires all30 complete fits")
    indexed = {(m["condition"], m["seed"], m["fold"]): m for m in ledger["fits"]}
    require(set(indexed) == {(c, s, f) for c in CONDITIONS for s in SEEDS for f in range(1, 6)},
            "Fit identities omitted/duplicated")
    result = {}
    for seed in SEEDS:
        for fold in range(1, 6):
            pair = None
            split = folds[f"fold_{fold}"]
            for condition in CONDITIONS:
                folder = _folder(condition, seed, fold)
                complete = C.read(folder / "complete.json")
                require(complete["passed"] and complete["registration"] == _reg_desc(), "Fit did not complete")
                artifact_paths = {r["path"] for r in complete["artifacts"]}
                expected_paths = {(folder / name).relative_to(C.WORK).as_posix() for name in
                                  ("started.json", "initial.pt", "checkpoint.pt", "metadata.json",
                                   "train_history.json", "batch_order.npy", "progress.json")}
                require(len(complete["artifacts"]) == 7 and artifact_paths == expected_paths,
                        "Fit completion must bind all seven expected artifacts")
                for record in complete["artifacts"]:
                    C.checked(record)
                meta = C.read(folder / "metadata.json")
                expected = {"condition": condition, "seed": seed, "fold": fold, "train_ids": split["train"],
                            "val_ids": split["val"], "test_ids": split["test"], "training": TRAINING,
                            "cohort_id": lock["cohort_id"], "folds_sha256": lock["folds_sha256"],
                            "registration": _reg_desc(), "updates": 120, "gradient_passes": 120,
                            "parameters": 498434, "input_key": FEATURE_KEYS[condition],
                            "no_val_test_forward_during_training": True, "checkpoint_selection": "fixed_final_epoch"}
                require(meta == indexed[condition, seed, fold] and all(meta.get(k) == v for k, v in expected.items()),
                        "Fit provenance/role/config differs from registered original fold")
                C.checked(meta["feature_manifest"]); C.checked(meta["input_independent_verification"])
                initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)
                saved = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
                require(all(initial.get(k) == v for k, v in {"condition": condition, "seed": seed,
                        "fold": fold, "registration": _reg_desc()}.items()), "Initial checkpoint identity differs")
                require(saved["metadata"] == meta and C.state_digest(initial["state_dict"]) == meta["initial_state_sha256"]
                        and C.state_digest(saved["state_dict"]) == meta["final_state_sha256"], "Checkpoint tensor/provenance differs")
                order = np.load(folder / "batch_order.npy", allow_pickle=False)
                validate_batch_order(order, [lock["subject_ids"].index(s) for s in split["train"]])
                require(batch_order_digest(order) == meta["batch_order_sha256"], "Saved batch order differs")
                history = C.read(folder / "train_history.json")
                require(len(history) == 60 and [r["epoch"] for r in history] == list(range(1, 61))
                        and all(r["updates_completed"] == r["gradient_passes_completed"] == 2*r["epoch"]
                                and len(r["batches"]) == 2
                                and [b["batch_size"] for b in r["batches"]] == [16, len(split["train"])-16]
                                and all(math.isfinite(b["train_ce"]) for b in r["batches"])
                                and math.isfinite(r["train_ce"]) for r in history), "Training trajectory incomplete")
                current = (meta["initial_state_sha256"], meta["batch_order_sha256"])
                require(pair is None or pair == current, "All paired initial/order tensors must match before release")
                pair = current
                result[condition, seed, fold] = (meta, saved)
    return result


def metric(labels, probabilities):
    from tridep_ccml.stage0 import binary_metrics
    require(len(labels) == len(probabilities) and set(labels) == {0, 1}
            and all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities), "Invalid original metric inputs")
    return binary_metrics(labels, probabilities)


def summarize(records, folds):
    result = {}
    for condition in CONDITIONS:
        selected = [r for r in records if r["condition"] == condition]
        require(len(selected) == 555, "Each condition requires555 original-role rows")
        result[condition] = {}
        for view, key in VIEWS.items():
            fm, seed_means, pooled = [], {}, {}
            for seed in SEEDS:
                for fold in range(1, 6):
                    for part in ("train", "val", "test"):
                        group = [r for r in selected if (r["seed"], r["outer_fold"], r["split"]) == (seed, fold, part)]
                        expected = folds[f"fold_{fold}"][part]
                        require(len(group) == len(expected) and {r["subject_id"] for r in group} == set(expected),
                                "Metric role person coverage differs")
                        fm.append({"seed": seed, "fold": fold, "split": part,
                                   **metric([r["label"] for r in group], [r[key] for r in group])})
                seed_means[str(seed)] = {part: {k: mean(r[k] for r in fm if r["seed"] == seed and r["split"] == part)
                                         for k in KEYS} for part in ("train", "val", "test")}
                group = [r for r in selected if r["seed"] == seed and r["split"] == "test"]
                require(len(group) == 37 and len({r["subject_id"] for r in group}) == 37, "Pooled OOF coverage differs")
                pooled[str(seed)] = metric([r["label"] for r in group], [r[key] for r in group])
            result[condition][view] = {"main_test": {k: mean(v["test"][k] for v in seed_means.values()) for k in KEYS},
                                       "seed_means": seed_means, "fold_metrics": fm, "pooled_oof": pooled}
    return result


def gate(candidate, control, candidate_seeds, control_seeds):
    delta = {k: candidate[k] - control[k] for k in KEYS}
    improving = sum(candidate_seeds[str(s)]["test"]["macro_f1"] > control_seeds[str(s)]["test"]["macro_f1"] for s in SEEDS)
    checks = {"F1_delta_at_least_0_01": delta["macro_f1"] >= .01, "at_least2_improving_seeds": improving >= 2,
              "BA_not_lower": delta["balanced_accuracy"] >= 0., "MDD_recall_drop_at_most0_05": delta["mdd_recall"] >= -.05}
    return {"passed": all(checks.values()), "checks": checks, "improving_seeds": improving,
            "delta": delta, "probability_improvement": delta["nll"] < 0 and delta["brier"] < 0}


def _legacy(lock, folds, labels):
    source = {(r["seed"], r["outer_fold"], r["subject_id"]): r for r in C.rows(C.STAGE0 / "branch_probabilities.jsonl")}
    controls = {(r["condition"], r["seed"], r["outer_fold"], r["subject_id"]): r
                for r in C.rows(C.STAGE0 / "control_predictions.jsonl")}
    require(len(source) == 555, "Original S/T source coverage differs")
    fixed = {}
    for seed in SEEDS:
        for fold in range(1, 6):
            split = folds[f"fold_{fold}"]
            weights = None
            for sid in lock["subject_ids"]:
                row = source[seed, fold, sid]
                role = next(part for part in split if sid in split[part])
                expected = {"label": labels[sid], "split": role, "cohort_id": lock["cohort_id"],
                            "folds_sha256": lock["folds_sha256"]}
                require(all(row[k] == v for k, v in expected.items()), "Original S/T role/label provenance differs")
                for name in ("ST-val-weighted", "EST-original-val-weighted"):
                    old = controls[name, seed, fold, sid]
                    require(all(old[k] == v for k, v in expected.items()), "Original control provenance differs")
                    w = old["fusion_weights"]
                    require(len(w) == 3 and all(math.isfinite(x) and 0 <= x <= 1 for x in w)
                            and math.isclose(sum(w), 1., abs_tol=1e-12), "Original fixed fusion weights invalid")
                    require(old["p_final"] == sum(wi * row["p_" + m] for wi, m in zip(w, ("eeg", "speech", "text"))),
                            "Original control does not replay its saved fixed weights")
                    if name == "ST-val-weighted":
                        require(w[0] == 0, "Fixed ST contains EEG")
                    else:
                        require(weights is None or weights == w, "Original EST weights changed within fold")
                        weights = w
            fixed[seed, fold] = weights
    return source, controls, fixed


@torch.inference_mode()
def _predict(model, x):
    model.eval().requires_grad_(False)
    before = C.state_digest(model.state_dict())
    logits_out, hidden, probabilities_out = [], [], []
    captured = []
    hook = model.classifier[-1].register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach()))
    try:
        for i in range(len(x)):
            logits = model(x[i:i+1])
            h = captured.pop()
            require(logits.shape == (1, 2) and h.shape == (1, 128) and torch.isfinite(logits).all()
                    and torch.isfinite(h).all(), "Original head output interface differs")
            logits_out.append(logits.cpu().numpy()[0]); hidden.append(h.cpu().numpy()[0])
            probabilities_out.append(torch.softmax(logits, dim=-1).cpu().numpy()[0])
    finally:
        hook.remove()
    require(before == C.state_digest(model.state_dict()) and all(not m.training for m in model.modules())
            and all(not p.requires_grad and p.grad is None for p in model.parameters()), "Frozen evaluation state changed")
    logits_array = np.stack(logits_out)
    probabilities = np.stack(probabilities_out)
    return {"logits": logits_array, "probabilities": probabilities, "features": np.stack(hidden),
            "p_eeg": probabilities[:, 1].astype(np.float64)}, before


def evaluate():
    reg = _entry_guard("evaluate")
    trained = C.read(C.OUT / "actions/fit_all.json")
    require(trained["status"] == "complete" and trained["registration"] == _reg_desc(), "Training action must fully complete")
    lock, folds, labels = C.context()
    validate_context(lock, folds, labels)
    fits = completed_fits(lock, folds)  # all30 BEFORE the first model/val/test forward
    arrays, feature_manifest = load_features(lock)
    source, controls, weights = _legacy(lock, folds, labels)
    require(not (C.OUT / "evaluation").exists() and not (C.REPORT / "prediction_rows.jsonl").exists(),
            "Partial or already released evaluation requires explicit review; no automatic retry")
    C.OUT.joinpath("evaluation").mkdir(parents=True, exist_ok=False)
    C.save(C.OUT / "evaluation/started.json", {"registration": _reg_desc(), "all_fits_verified": 30,
           "started_at": C.now(), "expected_producer_head_person_forward_calls": 2220,
           "feature_manifest": feature_manifest}, immutable=True)
    tensors = {key: torch.as_tensor(x, device="cuda") for key, x in arrays.items()}
    predictions, reload_checks, fixed_bindings = [], [], []
    calls = 0
    for seed in SEEDS:
        for fold in range(1, 6):
            split = folds[f"fold_{fold}"]
            for condition in CONDITIONS:
                meta, saved = fits[condition, seed, fold]
                folder = _folder(condition, seed, fold)
                outputs = []
                for replay in range(2):
                    C.runtime(seed)
                    model = original_model().to("cuda")
                    model.load_state_dict(saved["state_dict"], strict=True)
                    require(C.state_digest(model.state_dict()) == meta["final_state_sha256"], "Reloaded head state differs")
                    result, state = _predict(model, tensors[FEATURE_KEYS[condition]])
                    calls += len(lock["subject_ids"])
                    outputs.append(result)
                    del model
                require(all(np.array_equal(outputs[0][k], outputs[1][k]) for k in outputs[0]),
                        "Two original head reloads must replay logits/probabilities/representations bitwise")
                output = outputs[0]
                ids = lock["subject_ids"]
                roles = [next(part for part in split if sid in split[part]) for sid in ids]
                checkpoint = C.desc(folder / "checkpoint.pt")
                bundle = C.OUT / f"evaluation/{condition}/seed_{seed}/fold_{fold}.npz"
                _save_array(bundle, {**output, "subject_ids": np.array(ids), "labels": np.array([labels[s] for s in ids]),
                                    "splits": np.array(roles)}, compressed=True)
                C.save(bundle.with_suffix(".json"), {"condition": condition, "seed": seed, "fold": fold,
                       "bundle": C.desc(bundle), "checkpoint": checkpoint, "registration": _reg_desc(),
                       "feature_manifest": feature_manifest, "fixed_ST_source": C.desc(C.STAGE0 / "branch_probabilities.jsonl"),
                       "original_EST_weights_source": C.desc(C.STAGE0 / "control_predictions.jsonl"),
                       "state_before": state, "state_after": state, "producer_head_person_forward_calls": 74}, immutable=True)
                w = weights[seed, fold]
                fixed_bindings.append({"condition": condition, "seed": seed, "fold": fold, "weights": w,
                                       "w_E_zero": w[0] == 0, "selection_count": 0, "original_val_ids": split["val"]})
                for i, sid in enumerate(ids):
                    old = source[seed, fold, sid]
                    p = float(output["p_eeg"][i])
                    fixed_est = sum(wi * pi for wi, pi in zip(w, (p, old["p_speech"], old["p_text"])))
                    st = controls["ST-val-weighted", seed, fold, sid]["p_final"]
                    est_old = controls["EST-original-val-weighted", seed, fold, sid]["p_final"]
                    require(w[0] != 0 or fixed_est == est_old, "Zero EEG weight must produce unchanged original EST")
                    predictions.append({"condition": condition, "seed": seed, "outer_fold": fold,
                        "subject_id": sid, "label": labels[sid], "split": roles[i], "cohort_id": lock["cohort_id"],
                        "folds_sha256": lock["folds_sha256"], "p_eeg": p, "p_speech": old["p_speech"],
                        "p_text": old["p_text"], "p_fixed_EST": fixed_est, "fusion_weights": w,
                        "p_fixed_ST": st, "p_original_EST": est_old, "prediction_eeg": int(p > .5),
                        "prediction_fixed_EST": int(fixed_est > .5), "eeg_checkpoint": checkpoint,
                        "fixed_ST_checkpoints": {m: old["parent_checkpoints"][m] for m in ("speech", "text")},
                        "registration": _reg_desc()})
                reload_checks.append({"condition": condition, "seed": seed, "fold": fold, "checkpoint": checkpoint,
                                      "state_sha256": state, "exact_logits": True, "exact_probabilities": True,
                                      "exact_representations": True, "frozen_eval_state_unchanged": True})
                C.save(C.OUT / "evaluation/progress.json", {"producer_head_person_forward_calls": calls,
                       "models_reloaded_twice": len(reload_checks), "registration": _reg_desc()})
                print(f"RELOAD VERIFIED {condition} seed={seed} fold={fold}", flush=True)
    require(calls == reg["budget"]["producer_head_person_forward_calls"] == 2220 and len(predictions) == 1110,
            "Uniform original head evaluation budget differs")
    summaries = summarize(predictions, folds)
    old_metrics = C.read(C.STAGE0 / "metrics.json")
    accepted = {}
    matched = CONDITIONS[0]
    for condition in CONDITIONS:
        eeg = summaries[condition]["eeg"]
        est = summaries[condition]["fixed_EST"]
        eeg_control = summaries[matched]["eeg"]
        est_control = summaries[matched]["fixed_EST"]
        accepted[condition] = {"is_candidate": condition != matched, "direct_control": matched,
            "eeg_vs_matched_control": gate(eeg["main_test"], eeg_control["main_test"], eeg["seed_means"], eeg_control["seed_means"]),
            "fixed_EST_vs_matched_control": gate(est["main_test"], est_control["main_test"], est["seed_means"], est_control["seed_means"]),
            "fixed_EST_vs_original_EST": gate(est["main_test"], old_metrics["EST-original-val-weighted"]["main_test"],
                    est["seed_means"], old_metrics["EST-original-val-weighted"]["train_val_test_seed_means"]),
            "fixed_EST_vs_fixed_ST": gate(est["main_test"], old_metrics["ST-val-weighted"]["main_test"],
                    est["seed_means"], old_metrics["ST-val-weighted"]["train_val_test_seed_means"]),
            "w_E_zero_model_groups": sum(r["condition"] == condition and r["w_E_zero"] for r in fixed_bindings),
            "fusion_selections": 0, "new_calibration_or_threshold_selection": False}
        accepted[condition]["eeg_keep"] = condition != matched and accepted[condition]["eeg_vs_matched_control"]["passed"]
        accepted[condition]["fusion_keep"] = condition != matched and all(accepted[condition][key]["passed"] for key in
                 ("fixed_EST_vs_matched_control", "fixed_EST_vs_original_EST", "fixed_EST_vs_fixed_ST"))
    lookup = {(r["condition"], r["seed"], r["outer_fold"], r["subject_id"]): r for r in predictions}
    paired = []
    for row in predictions:
        if row["split"] != "test":
            continue
        for view, key in VIEWS.items():
            references = {"matched-control": lookup[matched, row["seed"], row["outer_fold"], row["subject_id"]][key]}
            if view == "fixed_EST":
                references.update({"fixed-ST": row["p_fixed_ST"], "original-EST": row["p_original_EST"]})
            for reference, p_ref in references.items():
                pc, pr = int(row[key] > .5), int(p_ref > .5)
                cc, cr = pc == row["label"], pr == row["label"]
                change = "add_correct" if cc and not cr else "add_error" if cr and not cc else "unchanged_correct" if cc else "unchanged_error"
                paired.append({k: row[k] for k in ("condition", "seed", "outer_fold", "subject_id", "label")} |
                              {"view": view, "reference": reference, "p_candidate": row[key], "p_reference": p_ref,
                               "prediction_candidate": pc, "prediction_reference": pr, "change": change})
    C.save_rows(C.REPORT / "prediction_rows.jsonl", predictions)
    C.save_rows(C.REPORT / "paired_changes.jsonl", paired)
    C.save(C.REPORT / "metrics.json", summaries, immutable=True)
    C.save(C.REPORT / "acceptance.json", accepted, immutable=True)
    C.save(C.REPORT / "fusion_weight_bindings.json", {"selection_count": 0, "records": fixed_bindings,
           "source": C.desc(C.STAGE0 / "control_predictions.jsonl")}, immutable=True)
    C.save(C.REPORT / "fixed_controls.json", {"stage0": C.desc(C.STAGE0 / "metrics.json"),
           "ST": old_metrics["ST-val-weighted"], "EST": old_metrics["EST-original-val-weighted"],
           "LaBraM5s_diagnostic_source": C.desc(C.WORK / "output/analysis/tridep_stage1_20261005/metrics.json"),
           "LaBraM5s_diagnostic_only": True, "different_window_length_is_not_direct_backbone_control": True}, immutable=True)
    C.save(C.REPORT / "paired_summary.json", {condition: {view: {reference:
           dict(Counter(r["change"] for r in paired if r["condition"] == condition and r["view"] == view and r["reference"] == reference))
           for reference in (("matched-control",) if view == "eeg" else ("matched-control", "fixed-ST", "original-EST"))}
           for view in VIEWS} for condition in CONDITIONS}, immutable=True)
    C.save(C.REPORT / "reload_validation.json", {"passed": True, "registration": _reg_desc(),
           "producer_head_person_forward_calls": calls, "checks": reload_checks}, immutable=True)
    C.save(C.OUT / "evaluation/complete.json", {"passed": True, "registration": _reg_desc(),
           "producer_head_person_forward_calls": calls, "models": 30, "prediction_records": 1110,
           "metric_groups": 180, "fusion_selections": 0}, immutable=True)
    C.registration()
    C.save(C.REPORT / "status.json", {"status": "uniform_evaluation_complete_pending_independent_verification",
           "fits_completed": 30, "updates_completed": 3600, "prediction_records": 1110, "metric_groups": 180,
           "producer_head_person_forward_calls": calls, "reload_verified_models": 30, "metrics_released": True,
           "registration": _reg_desc(), "next_phase_started": False})
    del tensors, arrays
    gc.collect()
    return summaries

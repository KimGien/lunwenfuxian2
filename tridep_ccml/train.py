"""Fixed first-round TC head training. Default is validation only; --execute trains.

The frozen features contain all splits, but optimization indexes train subjects
only. All heads start from the same seed-specific weights and use an independent
batch-order generator. TC03 only reads TC02 weights. TC06 is deferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from .ccml import (ThreeViewHeads, ccml_loss, ce_heads_loss, dirichlet_from_evidence,
                   evidence_dc, single_evidential_loss)
from .records import (ROOT, MODALITIES, STAGES, RecordContext, digest, read_json,
                      require, validate_prediction_records)

DEFAULT_EXPERIMENTS = [name for name in STAGES if not name.startswith("TC-00-")]


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def state_digest(state):
    hasher = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        hasher.update(name.encode())
        hasher.update(str(array.dtype).encode())
        hasher.update(str(array.shape).encode())
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def make_initial_state(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ThreeViewHeads(in_features=128, num_classes=2, modalities=MODALITIES)
        require(sum(p.numel() for p in model.parameters()) == 774, "Expected 774 head parameters")
        return {k: v.detach().clone() for k, v in model.state_dict().items()}


def loss_parameters(experiment, config):
    parameters = {k: config[k] for k in ("rho", "delta", "gamma", "anneal_epochs")}
    if experiment == "TC-05a-37-fulltext":
        parameters["rho"] = 1.
    if experiment == "TC-05b-37-fulltext":
        parameters["delta"] = 0.
    if experiment == "TC-05c-37-fulltext":
        parameters["gamma"] = 0.
    return parameters


def validate_config(context):
    config = context.config
    fixed = {"epochs": 60, "batch_size": 8, "learning_rate": .001,
             "weight_decay": 0, "gradient_clip": 1, "rho": 1.25, "delta": 1,
             "gamma": 1, "anneal_epochs": 50, "threshold": .5, "tie_class": 0,
             "seeds": [11, 22, 33], "folds": [1, 2, 3, 4, 5],
             "modalities": list(MODALITIES), "cohort_id": context.lock["cohort_id"],
             "cohort_sha256": context.cohort_sha256, "folds_sha256": context.folds_sha256}
    for key, expected in fixed.items():
        require(config.get(key) == expected, f"First-round configuration changed: {key}")
    context.check_file({"path": config["plan_path"], "sha256": config["plan_sha256"]})
    return config


def load_features(context, seed, fold):
    manifest, manifest_hash = context.manifest(seed, fold)
    path = context.root / "features" / f"seed_{seed}" / f"fold_{fold}.npz"
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {"features", "subject_ids", "labels", "splits"},
                "Unexpected feature array fields")
        features, ids, labels, splits = [archive[k].copy() for k in
                                            ("features", "subject_ids", "labels", "splits")]
    require(features.shape == (37, 3, 128) and features.dtype == np.float32
            and np.isfinite(features).all(), "Invalid frozen feature tensor")
    require(ids.shape == (37,) and ids.dtype.kind in "US" and ids.tolist() == context.ids,
            "Feature subjects/order differ from locked cohort")
    require(labels.shape == (37,) and labels.dtype == np.int64
            and labels.tolist() == [context.labels[sid] for sid in context.ids], "Feature diagnoses differ")
    assignment = context.folds[f"fold_{fold}"]
    expected_splits = [next(p for p in ("train", "val", "test") if sid in assignment[p]) for sid in context.ids]
    require(splits.shape == (37,) and splits.dtype.kind in "US" and splits.tolist() == expected_splits,
            "Feature splits differ from locked fold")
    return {"features": features, "subject_ids": ids.tolist(), "labels": labels,
            "splits": splits.tolist(), "manifest": manifest, "manifest_hash": manifest_hash}


def run_directory(root, experiment, seed, fold):
    return Path(root) / "runs" / experiment / f"seed_{seed}" / f"fold_{fold}"


def _checkpoint_metadata(context, data, experiment, seed, fold, initial_sha, batch_sha, device):
    return {"experiment": experiment, "seed": seed, "fold": fold,
            "cohort_id": context.lock["cohort_id"], "cohort_sha256": context.cohort_sha256,
            "folds_sha256": context.folds_sha256,
            "train_ids": context.folds[f"fold_{fold}"]["train"],
            "feature_manifest_sha256": data["manifest_hash"],
            "feature_npz_sha256": data["manifest"]["npz_sha256"],
            "parent_checkpoints": data["manifest"]["parent_checkpoints"],
            "training_config_sha256": context.config_sha256,
            "initialization_sha256": initial_sha, "batch_order_sha256": batch_sha,
            "effective_loss_parameters": loss_parameters(experiment, context.config),
            "completed_epochs": context.config["epochs"], "selection": "last_epoch",
            "trainable_parameters": 774, "upstream_mode": "frozen_eval_cached_features",
            "optimizer": "Adamax", "class_weights": None, "resampling": False,
            "device": str(device), "torch_version": str(torch.__version__),
            "runner_sha256": digest(Path(__file__)),
            "ccml_sha256": digest(Path(__file__).with_name("ccml.py")),
            "records_sha256": digest(Path(__file__).with_name("records.py"))}


def load_tc02(context, data, seed, fold, device):
    source = run_directory(context.root, "TC-02-37-fulltext", seed, fold) / "checkpoint.pt"
    require(source.is_file(), f"TC03 requires completed TC02 checkpoint: {source}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    meta = checkpoint["metadata"]
    for k, expected in {"experiment": "TC-02-37-fulltext", "seed": seed, "fold": fold,
                        "cohort_id": context.lock["cohort_id"], "cohort_sha256": context.cohort_sha256,
                        "folds_sha256": context.folds_sha256, "training_config_sha256": context.config_sha256,
                        "train_ids": context.folds[f"fold_{fold}"]["train"],
                        "feature_manifest_sha256": data["manifest_hash"],
                        "feature_npz_sha256": data["manifest"]["npz_sha256"],
                        "parent_checkpoints": data["manifest"]["parent_checkpoints"],
                        "completed_epochs": context.config["epochs"], "selection": "last_epoch",
                        "runner_sha256": digest(Path(__file__)),
                        "ccml_sha256": digest(Path(__file__).with_name("ccml.py")),
                        "records_sha256": digest(Path(__file__).with_name("records.py")),
                        "initialization_sha256": state_digest(make_initial_state(seed))}.items():
        require(meta.get(k) == expected, f"TC02 checkpoint provenance mismatch: {k}")
    sidecar = read_json(source.with_suffix(".json"))
    require(sidecar["checkpoint_sha256"] == digest(source)
            and sidecar["metadata"] == meta, "TC02 checkpoint sidecar mismatch")
    model = ThreeViewHeads(in_features=128, num_classes=2, modalities=MODALITIES).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, source, meta


def fit_heads(context, data, experiment, seed, fold, directory, device):
    config = context.config
    initial = make_initial_state(seed)
    model = ThreeViewHeads(in_features=128, num_classes=2, modalities=MODALITIES).to(device)
    model.load_state_dict(initial, strict=True)
    model.train()
    optimizer = torch.optim.Adamax(model.parameters(), lr=config["learning_rate"],
                                   weight_decay=config["weight_decay"])
    # Preserve the locked training-list order before applying the shared shuffle.
    lookup = {sid: i for i, sid in enumerate(data["subject_ids"])}
    train_indices = [lookup[sid] for sid in context.folds[f"fold_{fold}"]["train"]]
    x_train = torch.from_numpy(data["features"][train_indices]).to(device)
    y_train = torch.from_numpy(data["labels"][train_indices]).to(device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_hasher = hashlib.sha256()
    history = []
    parameters = loss_parameters(experiment, config)
    for epoch in range(1, config["epochs"] + 1):
        order = torch.randperm(len(train_indices), generator=generator)
        batch_hasher.update(order.numpy().astype("<i8").tobytes())
        total = 0.
        for start in range(0, len(order), config["batch_size"]):
            selected = order[start:start + config["batch_size"]].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train[selected])
            if experiment == "TC-01-37-fulltext":
                loss = ce_heads_loss(logits, y_train[selected])
            elif experiment == "TC-02-37-fulltext":
                loss = single_evidential_loss(F.softplus(logits), y_train[selected], epoch,
                                               anneal_epochs=config["anneal_epochs"])
            else:
                loss = ccml_loss(F.softplus(logits), y_train[selected], epoch, **parameters)
            require(torch.isfinite(loss).item(), f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"],
                                                  error_if_nonfinite=True)
            require(torch.isfinite(norm).item(), f"Nonfinite gradient at epoch {epoch}")
            optimizer.step()
            require(all(torch.isfinite(p).all().item() for p in model.parameters()),
                    f"Nonfinite parameter at epoch {epoch}")
            total += loss.item() * len(selected)
        history.append({"epoch": epoch, "train_loss": total / len(train_indices)})
    model.eval()
    meta = _checkpoint_metadata(context, data, experiment, seed, fold, state_digest(initial),
                                batch_hasher.hexdigest(), device)
    meta["optimizer_updates"] = config["epochs"] * ((len(train_indices) + config["batch_size"] - 1)
                                                   // config["batch_size"])
    path = directory / "checkpoint.pt"
    with path.open("xb") as f:
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "metadata": meta}, f)
    write_json_new(path.with_suffix(".json"), {"checkpoint_sha256": digest(path), "metadata": meta})
    write_json_new(directory / "train_history.json", history)
    # Check save/reload equivalence without additional training.
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    require(state_digest(reloaded["state_dict"]) == state_digest(model.state_dict()),
            "Saved head state failed exact reload check")
    return model, path, meta


def predict_records(context, data, model, experiment, seed, fold, checkpoint_path, metadata, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(data["features"]).to(device))
        if experiment == "TC-01-37-fulltext":
            evidence, per_view = None, {"p": logits.softmax(dim=-1)}
        else:
            evidence = F.softplus(logits)
            per_view = dirichlet_from_evidence(evidence)
        method = STAGES[experiment]
        parameters = loss_parameters(experiment, context.config)
        decomposition = None
        if method == "mean_probability":
            final = {"p": per_view["p"].mean(dim=1)}
        elif method in ("mean_evidence", "sum_evidence"):
            final_e = evidence.mean(dim=1) if method == "mean_evidence" else evidence.sum(dim=1)
            final = dirichlet_from_evidence(final_e)
            final["e_final"] = final_e
        else:
            decomposition = evidence_dc(evidence, rho=parameters["rho"])
            final = decomposition
        rows = []
        joint = experiment.startswith(("TC-04-", "TC-05"))
        checkpoint_hash = digest(checkpoint_path)
        for i, sid in enumerate(data["subject_ids"]):
            views = {}
            for v, modality in enumerate(MODALITIES):
                views[modality] = {"e": evidence[i, v].cpu().tolist() if evidence is not None else None,
                    "alpha": per_view["alpha"][i, v].cpu().tolist() if evidence is not None else None,
                    "p": per_view["p"][i, v].cpu().tolist(),
                    "u": per_view["u"][i, v].item() if evidence is not None else None}
            p = final["p"][i].cpu().tolist()
            rows.append({"experiment": experiment, "cohort_id": context.lock["cohort_id"],
                "cohort_sha256": context.cohort_sha256, "folds_sha256": context.folds_sha256,
                "subject_id": sid, "seed": seed, "outer_fold": fold, "split": data["splits"][i],
                "label": int(data["labels"][i]), "train_ids": context.folds[f"fold_{fold}"]["train"],
                "parent_checkpoints": data["manifest"]["parent_checkpoints"],
                "source_features": {m: data["manifest"]["source_features"][m][sid] for m in MODALITIES},
                "feature_manifest_sha256": data["manifest_hash"],
                "feature_npz_sha256": data["manifest"]["npz_sha256"],
                "training_config_sha256": context.config_sha256,
                "checkpoint_sha256": checkpoint_hash,
                "initialization_sha256": metadata["initialization_sha256"],
                "fusion_method": method, "fusion_weights": [1 / 3] * 3 if method == "mean_probability" else None,
                "views": views,
                **{k: decomposition[k][i].cpu().tolist() if decomposition else None for k in ("m", "e_con", "e_cmp")},
                "e_final": final["e_final"][i].cpu().tolist() if "e_final" in final else None,
                "p_final": p, "u_final": final["u"][i].item() if "u" in final else None,
                "prediction": int(p[1] > .5), "rho": parameters["rho"] if method == "ccml" else None,
                "delta": parameters["delta"] if joint else None, "gamma": parameters["gamma"] if joint else None})
    return rows


def run_one(context, data, experiment, seed, fold, device):
    directory = run_directory(context.root, experiment, seed, fold)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        if experiment.startswith("TC-03-"):
            model, checkpoint, metadata = load_tc02(context, data, seed, fold, device)
        else:
            model, checkpoint, metadata = fit_heads(context, data, experiment, seed, fold, directory, device)
        rows = predict_records(context, data, model, experiment, seed, fold, checkpoint, metadata, device)
        validation = validate_prediction_records(rows, experiment, require_complete=False, context=context)
        with (directory / "predictions.jsonl").open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        report = {"status": "complete", "experiment": experiment, "seed": seed, "fold": fold,
                  "training_performed": not experiment.startswith("TC-03-"),
                  "checkpoint_path": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                  "source_metadata": metadata, "validation": validation}
        write_json_new(directory / "run.json", report)
        return report
    except Exception as exc:
        write_json_new(directory / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                       "error": str(exc), "experiment": experiment, "seed": seed, "fold": fold})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--experiment", action="append", choices=DEFAULT_EXPERIMENTS)
    parser.add_argument("--seed", action="append", type=int, choices=[11, 22, 33])
    parser.add_argument("--fold", action="append", type=int, choices=[1, 2, 3, 4, 5])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="Actually train new heads / run TC03 inference")
    modes.add_argument("--dry-run", action="store_true", help="Default: validate provenance and planned work")
    parser.add_argument("--device", default="cpu", help="cpu (default), cuda or cuda:0")
    parser.add_argument("--report", type=Path, help="Write validation report without overwriting")
    args = parser.parse_args()
    context = RecordContext(args.root)
    config = validate_config(context)
    requested = set(args.experiment or DEFAULT_EXPERIMENTS)
    experiments = [name for name in DEFAULT_EXPERIMENTS if name in requested]
    seeds, folds = sorted(set(args.seed or config["seeds"])), sorted(set(args.fold or config["folds"]))
    features, jobs, blockers = {}, [], []
    for seed in seeds:
        for fold in folds:
            try:
                features[(seed, fold)] = load_features(context, seed, fold)
            except (OSError, ValueError, KeyError) as exc:
                blockers.append({"seed": seed, "fold": fold, "kind": "feature_provenance", "error": str(exc)})
            for experiment in experiments:
                output = run_directory(context.root, experiment, seed, fold)
                if output.exists():
                    blockers.append({"experiment": experiment, "seed": seed, "fold": fold,
                                     "kind": "output_exists", "error": f"Refusing to overwrite {output}"})
                if experiment.startswith("TC-03-") and "TC-02-37-fulltext" not in requested:
                    source = run_directory(context.root, "TC-02-37-fulltext", seed, fold) / "checkpoint.pt"
                    if not source.is_file():
                        blockers.append({"experiment": experiment, "seed": seed, "fold": fold,
                                         "kind": "dependency", "error": "TC03 requires completed TC02"})
                    elif (seed, fold) in features:
                        try:
                            load_tc02(context, features[(seed, fold)], seed, fold, torch.device("cpu"))
                        except (OSError, ValueError, KeyError, RuntimeError) as exc:
                            blockers.append({"experiment": experiment, "seed": seed, "fold": fold,
                                             "kind": "dependency_provenance", "error": str(exc)})
                jobs.append({"experiment": experiment, "seed": seed, "fold": fold,
                             "operation": "reuse_TC02_inference" if experiment.startswith("TC-03-") else "train_new_heads",
                             "output": str(output)})
    report = {"mode": "execute" if args.execute else "dry_run", "passed": not blockers,
              "training_performed": False, "config_sha256": context.config_sha256,
              "cohort_sha256": context.cohort_sha256, "folds_sha256": context.folds_sha256,
              "validated_feature_folds": len(features), "jobs": jobs, "blockers": blockers,
              "deferred": ["TC06 joint representation training is outside first-round preparation"],
              "paired_controls": {"initialization": "same seed-specific state across every trained stage",
                    "batch_order": "independent CPU generator with identical seed and train-list order",
                    "budget": "60 epochs, batch 8, last epoch; train split only"}}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json_new(args.report, report)
    if blockers or not args.execute:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        if blockers:
            raise SystemExit(2)
        return
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    for experiment in experiments:
        for seed in seeds:
            for fold in folds:
                result = run_one(context, features[(seed, fold)], experiment, seed, fold, device)
                print(json.dumps({k: result[k] for k in ("status", "experiment", "seed", "fold", "training_performed")}), flush=True)


if __name__ == "__main__":
    main()

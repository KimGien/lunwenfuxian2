"""Decision-level fusion of unimodal predictions.

Implements the 12 multimodal configurations from the paper table:
  - Weighted Averaging (4 configs)
  - Bayesian Fusion (4 configs)
  - Majority Voting (4 configs)

Usage
-----
    python scripts/fusion.py --config configs/training/fusion.yaml
    python scripts/fusion.py --config configs/training/fusion.yaml data.predictions_dir=predictions/
"""

import argparse
import csv
import json
import sys
from math import exp, log
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "split_dataset_june"


# ── Data loading ─────────────────────────────────────────────────────────


def load_predictions(csv_path: "str | Path") -> dict:
    """Load a prediction CSV into a dict keyed by subject_id.

    Returns:
        Mapping ``{subject_id: {"fold": int, "gt": int, "pred": int,
        "prob": float}}``.
    """
    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["subject_id"]
            data[sid] = {
                "fold": int(row["fold"]),
                "gt": int(row["ground_truth"]),
                "pred": int(row["predicted_label"]),
                "prob": float(row["probability"]),
            }
    return data


# ── Fusion methods ───────────────────────────────────────────────────────


def weighted_avg_fusion(
    subject_ids: list[str],
    modality_data: dict,
    weights: dict[str, float],
) -> dict[str, int]:
    """Weighted average of probabilities; unanimous votes are taken directly.

    If all modalities predict the same class, that class is used without
    weighting.  Otherwise the weighted sum of depressed-class probabilities
    is thresholded at 0.5.

    Args:
        subject_ids: Ordered list of subject identifiers to fuse.
        modality_data: Nested dict ``{modality: {subject_id: {pred, prob, ...}}}``.
        weights: Per-modality weights that sum to 1.

    Returns:
        Dict ``{subject_id: predicted_label}`` for every subject in
        *subject_ids*.
    """
    fused = {}
    modalities = list(weights.keys())
    for sid in subject_ids:
        preds = [modality_data[mod][sid]["pred"] for mod in modalities]
        if all(p == preds[0] for p in preds):
            fused[sid] = preds[0]
        else:
            weighted_prob = sum(
                weights[mod] * modality_data[mod][sid]["prob"] for mod in modalities
            )
            fused[sid] = 1 if weighted_prob > 0.5 else 0
    return fused


def bayesian_fusion(
    subject_ids: list[str],
    modality_data: dict,
    weights: dict[str, float],
    prior: float,
) -> dict[str, int]:
    """Bayesian fusion using weighted log-likelihood ratios.

    Combines per-modality likelihood ratios (raised to their respective
    weights) with a Bernoulli prior to compute a posterior probability of
    depression.  Unanimous predictions bypass the fusion step.

    Args:
        subject_ids: Ordered list of subject identifiers to fuse.
        modality_data: Nested dict ``{modality: {subject_id: {pred, prob, ...}}}``.
        weights: Per-modality weights applied as exponents to each LR.
        prior: Prior probability of depression, *P(depressed)*.

    Returns:
        Dict ``{subject_id: predicted_label}`` for every subject in
        *subject_ids*.
    """
    fused = {}
    modalities = list(weights.keys())
    epsilon = 1e-10
    for sid in subject_ids:
        preds = [modality_data[mod][sid]["pred"] for mod in modalities]
        if all(p == preds[0] for p in preds):
            fused[sid] = preds[0]
        else:
            combined_lr = 1.0
            for mod in modalities:
                p = modality_data[mod][sid]["prob"]
                lr = (p + epsilon) / ((1 - p) + epsilon)
                combined_lr *= exp(weights[mod] * log(lr))
            posterior = (prior * combined_lr) / (prior * combined_lr + (1 - prior))
            fused[sid] = 1 if posterior >= 0.5 else 0
    return fused


def majority_vote_fusion(
    subject_ids: list[str],
    modality_data: dict,
    modalities: list[str],
) -> dict[str, int]:
    """Majority voting; ties broken by the mean depressed-class probability.

    Args:
        subject_ids: Ordered list of subject identifiers to fuse.
        modality_data: Nested dict ``{modality: {subject_id: {pred, prob, ...}}}``.
        modalities: List of modality keys to include in the vote.

    Returns:
        Dict ``{subject_id: predicted_label}`` for every subject in
        *subject_ids*.
    """
    fused = {}
    for sid in subject_ids:
        preds = [modality_data[mod][sid]["pred"] for mod in modalities]
        count_1 = sum(preds)
        count_0 = len(preds) - count_1
        if count_1 > count_0:
            fused[sid] = 1
        elif count_0 > count_1:
            fused[sid] = 0
        else:
            avg_prob = np.mean([modality_data[mod][sid]["prob"] for mod in modalities])
            fused[sid] = 1 if avg_prob > 0.5 else 0
    return fused


# ── Evaluation ───────────────────────────────────────────────────────────


def evaluate_fold(
    fused_preds: dict[str, int],
    modality_data: dict,
    ref_modality: str,
) -> float:
    """Compute macro-averaged F1 for a set of fused predictions.

    Args:
        fused_preds: Dict ``{subject_id: predicted_label}``.
        modality_data: Nested dict used to look up ground-truth labels via
            ``modality_data[ref_modality][subject_id]["gt"]``.
        ref_modality: Modality key used as the ground-truth source.

    Returns:
        Macro-averaged F1 score in ``[0, 1]``.
    """
    y_true = [modality_data[ref_modality][sid]["gt"] for sid in fused_preds]
    y_pred = [fused_preds[sid] for sid in fused_preds]
    return f1_score(y_true, y_pred, average="macro")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Decision-level fusion")
    parser.add_argument(
        "--config",
        default="configs/training/fusion.yaml",
        help="Path to fusion YAML config (default: configs/training/fusion.yaml)",
    )
    args, overrides = parser.parse_known_args()

    yaml_cfg = OmegaConf.load(PROJECT_ROOT / args.config)
    if overrides:
        yaml_cfg = OmegaConf.merge(yaml_cfg, OmegaConf.from_dotlist(overrides))

    pred_dir = PROJECT_ROOT / yaml_cfg.data.predictions_dir
    fold_file = PROJECT_ROOT / "data" / "split_dataset_june" / "fold_assignments.json"
    MODALITY_FILES = OmegaConf.to_container(yaml_cfg.modality_files, resolve=True)
    PRIOR_DEPRESSED = float(yaml_cfg.prior_depressed)
    ALL_CONFIGS = OmegaConf.to_container(yaml_cfg.configs, resolve=True)

    # Load fold assignments
    with open(fold_file) as f:
        folds = json.load(f)

    # Load unimodal predictions
    modality_data = {}
    for mod, filename in MODALITY_FILES.items():
        csv_path = pred_dir / filename
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found — skipping configs that use {mod}")
            continue
        modality_data[mod] = load_predictions(csv_path)

    print(f"\n{'=' * 75}")
    print(f"{'Config':<45s}  {'F1 (mean +/- std)':>20s}")
    print(f"{'=' * 75}")

    for cfg in ALL_CONFIGS:
        # Check all modalities are available
        if not all(mod in modality_data for mod in cfg["modalities"]):
            missing = [m for m in cfg["modalities"] if m not in modality_data]
            print(f"  {cfg['name']:<45s}  SKIPPED (missing: {', '.join(missing)})")
            continue

        fold_f1s = []
        all_predictions = []

        for fold_idx in range(5):
            fold_name = f"fold_{fold_idx + 1}"
            test_ids = folds[fold_name]["test"]

            # Only subjects present in all modalities
            subject_ids = [
                sid
                for sid in test_ids
                if all(sid in modality_data[mod] for mod in cfg["modalities"])
            ]

            ref_mod = cfg["modalities"][0]

            if cfg["method"] == "weighted_avg":
                fused = weighted_avg_fusion(subject_ids, modality_data, cfg["weights"])
            elif cfg["method"] == "bayesian":
                fused = bayesian_fusion(
                    subject_ids, modality_data, cfg["weights"], PRIOR_DEPRESSED
                )
            elif cfg["method"] == "majority_vote":
                fused = majority_vote_fusion(
                    subject_ids, modality_data, cfg["modalities"]
                )
            else:
                raise ValueError(f"Unknown method: {cfg['method']}")

            f1 = evaluate_fold(fused, modality_data, ref_mod)
            fold_f1s.append(f1)

            # Collect per-subject predictions for CSV output
            for sid in fused:
                gt = modality_data[ref_mod][sid]["gt"]
                # Average probability across modalities for the fused prediction
                avg_prob = np.mean(
                    [modality_data[mod][sid]["prob"] for mod in cfg["modalities"]]
                )
                all_predictions.append(
                    {
                        "subject_id": sid,
                        "fold": fold_idx + 1,
                        "ground_truth": gt,
                        "predicted_label": fused[sid],
                        "probability": avg_prob,
                    }
                )

        mean_f1 = np.mean(fold_f1s)
        std_f1 = np.std(fold_f1s, ddof=1)
        print(f"  {cfg['name']:<45s}  {mean_f1:.3f} +/- {std_f1:.3f}")

        # Save fused predictions CSV
        # Generate filename: e.g. "wa_eeg_speech_text.csv"
        method_prefix = {"weighted_avg": "wa", "bayesian": "bf", "majority_vote": "mv"}
        fname = (
            method_prefix[cfg["method"]] + "_" + "_".join(cfg["modalities"]) + ".csv"
        )
        out_path = pred_dir / fname
        with open(out_path, "w") as f:
            f.write("subject_id,fold,ground_truth,predicted_label,probability\n")
            for row in sorted(
                all_predictions, key=lambda r: (r["fold"], r["subject_id"])
            ):
                f.write(
                    f"{row['subject_id']},{row['fold']},{row['ground_truth']},"
                    f"{row['predicted_label']},{row['probability']:.6f}\n"
                )

    print(f"{'=' * 75}")
    print(f"\nFused prediction CSVs saved to: {pred_dir}/")


if __name__ == "__main__":
    main()

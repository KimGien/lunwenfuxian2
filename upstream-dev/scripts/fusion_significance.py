"""Statistical significance analysis for fusion results.

Computes per-fold F1 scores for key configurations, then runs:
  - Paired Wilcoxon signed-rank tests
  - Paired t-tests (one-sided)
  - 95% confidence intervals
  - CI overlap analysis

Usage
-----
    python scripts/fusion_significance.py --config configs/training/fusion.yaml
"""

import argparse
import csv
import json
import sys
from math import exp, log
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from scipy import stats
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Parse config ──────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--config", default="configs/training/fusion.yaml")
_args, _ = _parser.parse_known_args()
_yaml_cfg = OmegaConf.load(PROJECT_ROOT / _args.config)

MODALITY_FILES = OmegaConf.to_container(_yaml_cfg.modality_files, resolve=True)
PRIOR_DEPRESSED = float(_yaml_cfg.prior_depressed)

DATA_DIR = PROJECT_ROOT / "data" / "split_dataset_june"
FOLD_FILE = DATA_DIR / "fold_assignments.json"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# ── Data loading ─────────────────────────────────────────────────────────

with open(FOLD_FILE) as f:
    folds = json.load(f)


def load_predictions(csv_path):
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


mod_data = {}
for mod, filename in MODALITY_FILES.items():
    mod_data[mod] = load_predictions(PREDICTIONS_DIR / filename)

PRIOR = PRIOR_DEPRESSED

# ── Fusion methods ───────────────────────────────────────────────────────


def weighted_avg_fusion(sids, md, weights):
    fused = {}
    mods = list(weights.keys())
    for sid in sids:
        preds = [md[m][sid]["pred"] for m in mods]
        if all(p == preds[0] for p in preds):
            fused[sid] = preds[0]
        else:
            wp = sum(weights[m] * md[m][sid]["prob"] for m in mods)
            fused[sid] = 1 if wp > 0.5 else 0
    return fused


def bayesian_fusion(sids, md, weights, prior):
    fused = {}
    mods = list(weights.keys())
    eps = 1e-10
    for sid in sids:
        preds = [md[m][sid]["pred"] for m in mods]
        if all(p == preds[0] for p in preds):
            fused[sid] = preds[0]
        else:
            clr = 1.0
            for m in mods:
                p = md[m][sid]["prob"]
                lr = (p + eps) / ((1 - p) + eps)
                clr *= exp(weights[m] * log(lr))
            post = (prior * clr) / (prior * clr + (1 - prior))
            fused[sid] = 1 if post >= 0.5 else 0
    return fused


def majority_vote_fusion(sids, md, mods):
    fused = {}
    for sid in sids:
        preds = [md[m][sid]["pred"] for m in mods]
        c1 = sum(preds)
        c0 = len(preds) - c1
        if c1 > c0:
            fused[sid] = 1
        elif c0 > c1:
            fused[sid] = 0
        else:
            avg_p = np.mean([md[m][sid]["prob"] for m in mods])
            fused[sid] = 1 if avg_p > 0.5 else 0
    return fused


# ── Per-fold evaluation ──────────────────────────────────────────────────


def get_fold_f1s(method, mods, weights=None):
    fold_f1s = []
    for fi in range(5):
        fn = f"fold_{fi + 1}"
        test_ids = folds[fn]["test"]
        sids = [s for s in test_ids if all(s in mod_data[m] for m in mods)]
        ref = mods[0]
        if method == "wa":
            fused = weighted_avg_fusion(sids, mod_data, weights)
        elif method == "bf":
            fused = bayesian_fusion(sids, mod_data, weights, PRIOR)
        elif method == "mv":
            fused = majority_vote_fusion(sids, mod_data, mods)
        yt = [mod_data[ref][s]["gt"] for s in fused]
        yp = [fused[s] for s in fused]
        fold_f1s.append(f1_score(yt, yp, average="macro"))
    return np.array(fold_f1s)


def get_unimodal_fold_f1s(mod_key):
    fold_f1s = []
    for fi in range(5):
        fn = f"fold_{fi + 1}"
        test_ids = folds[fn]["test"]
        sids = [s for s in test_ids if s in mod_data[mod_key]]
        yt = [mod_data[mod_key][s]["gt"] for s in sids]
        yp = [mod_data[mod_key][s]["pred"] for s in sids]
        fold_f1s.append(f1_score(yt, yp, average="macro"))
    return np.array(fold_f1s)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    # Configurations to analyse
    configs = {
        "EEG (unimodal)": get_unimodal_fold_f1s("eeg"),
        "Speech (unimodal)": get_unimodal_fold_f1s("speech"),
        "Text (unimodal)": get_unimodal_fold_f1s("text"),
        "WA: Speech+Text": get_fold_f1s(
            "wa", ["speech", "text"], {"speech": 0.40, "text": 0.60}
        ),
        "WA: EEG+Sp+Txt (trimodal)": get_fold_f1s(
            "wa",
            ["eeg", "speech", "text"],
            {"eeg": 0.05, "speech": 0.25, "text": 0.70},
        ),
        "BF: EEG+Sp+Txt (trimodal)": get_fold_f1s(
            "bf",
            ["eeg", "speech", "text"],
            {"eeg": 0.05, "speech": 0.20, "text": 0.75},
        ),
        "MV: Speech+Text": get_fold_f1s("mv", ["speech", "text"]),
    }

    # ── Per-fold F1 scores ───────────────────────────────────────────────
    print("Per-fold F1 scores:")
    print(
        f"{'Config':<30s}  {'F1_1':>6s}  {'F1_2':>6s}  {'F1_3':>6s}  "
        f"{'F1_4':>6s}  {'F1_5':>6s}  {'Mean':>7s}  {'Std':>6s}"
    )
    print("-" * 95)
    for name, f1s in configs.items():
        print(
            f"{name:<30s}  {f1s[0]:6.3f}  {f1s[1]:6.3f}  {f1s[2]:6.3f}  "
            f"{f1s[3]:6.3f}  {f1s[4]:6.3f}  {np.mean(f1s):7.4f}  {np.std(f1s, ddof=1):6.4f}"
        )

    # ── Pairwise comparisons ─────────────────────────────────────────────
    comparisons = [
        ("WA: EEG+Sp+Txt (trimodal)", "Text (unimodal)"),
        ("WA: EEG+Sp+Txt (trimodal)", "Speech (unimodal)"),
        ("WA: EEG+Sp+Txt (trimodal)", "WA: Speech+Text"),
        ("BF: EEG+Sp+Txt (trimodal)", "Text (unimodal)"),
        ("BF: EEG+Sp+Txt (trimodal)", "WA: Speech+Text"),
        ("WA: Speech+Text", "Text (unimodal)"),
        ("WA: Speech+Text", "Speech (unimodal)"),
    ]

    # Wilcoxon signed-rank tests
    print(f"\n{'=' * 90}")
    print("PAIRED WILCOXON SIGNED-RANK TESTS (5 folds)")
    print("Note: with n=5 paired observations, minimum achievable p-value = 0.0625")
    print("=" * 90)
    print(
        f"\n{'Comparison':<55s}  {'Delta':>7s}  {'W-stat':>7s}  "
        f"{'p-val':>8s}  {'Sig?':>5s}"
    )
    print("-" * 90)
    for a_name, b_name in comparisons:
        a = configs[a_name]
        b = configs[b_name]
        mean_diff = np.mean(a - b)
        try:
            stat, p = stats.wilcoxon(a, b, alternative="greater")
        except ValueError:
            stat, p = np.nan, np.nan
        sig = "yes" if p < 0.1 else "no"
        print(
            f"  {a_name} > {b_name:<25s}  {mean_diff:+7.4f}  "
            f"{stat:7.1f}  {p:8.4f}  {sig:>5s}"
        )

    # Paired t-tests (one-sided)
    print(f"\n{'=' * 90}")
    print("PAIRED T-TESTS (5 folds)")
    print("=" * 90)
    print(
        f"\n{'Comparison':<55s}  {'Delta':>7s}  {'t-stat':>7s}  "
        f"{'p-val':>8s}  {'Sig?':>5s}"
    )
    print("-" * 90)
    for a_name, b_name in comparisons:
        a = configs[a_name]
        b = configs[b_name]
        mean_diff = np.mean(a - b)
        t_stat, p = stats.ttest_rel(a, b)
        p_one = p / 2 if t_stat > 0 else 1 - p / 2
        sig = "yes" if p_one < 0.05 else ("~" if p_one < 0.1 else "no")
        print(
            f"  {a_name} > {b_name:<25s}  {mean_diff:+7.4f}  "
            f"{t_stat:7.3f}  {p_one:8.4f}  {sig:>5s}"
        )

    # 95% confidence intervals
    print(f"\n{'=' * 90}")
    print("95% CONFIDENCE INTERVALS (t-distribution, n=5)")
    print("=" * 90)
    for name, f1s in configs.items():
        mean = np.mean(f1s)
        se = np.std(f1s, ddof=1) / np.sqrt(5)
        ci = stats.t.interval(0.95, df=4, loc=mean, scale=se)
        print(f"  {name:<35s}  {mean:.4f}  [{ci[0]:.4f}, {ci[1]:.4f}]")

    # CI overlap analysis
    print(f"\n{'=' * 90}")
    print("OVERLAP ANALYSIS")
    print("=" * 90)
    print("\nDo 95% CIs overlap?")
    for a_name, b_name in comparisons:
        a = configs[a_name]
        b = configs[b_name]
        a_mean = np.mean(a)
        a_se = np.std(a, ddof=1) / np.sqrt(5)
        b_mean = np.mean(b)
        b_se = np.std(b, ddof=1) / np.sqrt(5)
        a_ci = stats.t.interval(0.95, df=4, loc=a_mean, scale=a_se)
        b_ci = stats.t.interval(0.95, df=4, loc=b_mean, scale=b_se)
        overlap = a_ci[0] < b_ci[1] and b_ci[0] < a_ci[1]
        print(
            f"  {a_name:<30s} vs {b_name:<25s}  "
            f"{'OVERLAP' if overlap else 'NO OVERLAP'}"
        )


if __name__ == "__main__":
    main()

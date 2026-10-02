"""Grid search over fusion weights for all modality combinations.

Searches Weighted Averaging and Bayesian Fusion weights in 0.05 steps,
and reports Majority Voting (no weights) for all combinations.

Usage
-----
    python scripts/fusion_grid_search.py --config configs/training/fusion.yaml

Reads prediction CSVs from the directory specified in the config and fold
assignments from ``data/split_dataset_june/fold_assignments.json``.
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
FOLD_FILE = DATA_DIR / "fold_assignments.json"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# ── Parse config ──────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--config", default="configs/training/fusion.yaml")
_args, _ = _parser.parse_known_args()
_yaml_cfg = OmegaConf.load(PROJECT_ROOT / _args.config)

MODALITY_FILES = OmegaConf.to_container(_yaml_cfg.modality_files, resolve=True)
PRIOR_DEPRESSED = float(_yaml_cfg.prior_depressed)
_pred_dir = PROJECT_ROOT / _yaml_cfg.data.predictions_dir

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
    mod_data[mod] = load_predictions(_pred_dir / filename)

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


# ── Evaluation ───────────────────────────────────────────────────────────


def eval_config(method, mods, weights=None):
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
    return np.mean(fold_f1s), np.std(fold_f1s, ddof=1)


# ── Grid search ──────────────────────────────────────────────────────────

STEP = 0.05
weight_vals = [round(i * STEP, 2) for i in range(1, int(1 / STEP))]  # 0.05 to 0.95


def main():
    # ── Trimodal: EEG + Speech + Text ────────────────────────────────────
    print("=" * 90)
    print("TRIMODAL: EEG + Speech + Text (step=0.05)")
    print("=" * 90)

    tri_wa = []
    tri_bf = []
    for we in weight_vals:
        for ws in weight_vals:
            wt = round(1.0 - we - ws, 2)
            if wt < 0.05 or wt > 0.95:
                continue
            w = {"eeg": we, "speech": ws, "text": wt}
            m, s = eval_config("wa", ["eeg", "speech", "text"], w)
            tri_wa.append((m, s, we, ws, wt))
            m, s = eval_config("bf", ["eeg", "speech", "text"], w)
            tri_bf.append((m, s, we, ws, wt))

    tri_wa.sort(key=lambda x: (-x[0], x[1]))
    tri_bf.sort(key=lambda x: (-x[0], x[1]))

    print(f"\nTop 15 Weighted Averaging:")
    print(f"  {'w_eeg':>5s}  {'w_sp':>5s}  {'w_txt':>5s}  {'F1':>7s}  {'std':>7s}")
    for m, s, we, ws, wt in tri_wa[:15]:
        print(f"  {we:5.2f}  {ws:5.2f}  {wt:5.2f}  {m:7.4f}  {s:7.4f}")

    print(f"\nTop 15 Bayesian Fusion:")
    print(f"  {'w_eeg':>5s}  {'w_sp':>5s}  {'w_txt':>5s}  {'F1':>7s}  {'std':>7s}")
    for m, s, we, ws, wt in tri_bf[:15]:
        print(f"  {we:5.2f}  {ws:5.2f}  {wt:5.2f}  {m:7.4f}  {s:7.4f}")

    m, s = eval_config("mv", ["eeg", "speech", "text"])
    print(f"\nMajority Vote (trimodal): F1={m:.4f} +/- {s:.4f}")

    # ── Bimodal: all 3 pairs ─────────────────────────────────────────────
    bimodal = [("eeg", "speech"), ("eeg", "text"), ("speech", "text")]

    for m1, m2 in bimodal:
        print(f"\n{'=' * 90}")
        print(f"BIMODAL: {m1.upper()} + {m2.upper()} (step=0.05)")
        print(f"{'=' * 90}")

        bi_wa = []
        bi_bf = []
        for w1 in weight_vals:
            w2 = round(1.0 - w1, 2)
            if w2 < 0.05:
                continue
            w = {m1: w1, m2: w2}
            m, s = eval_config("wa", [m1, m2], w)
            bi_wa.append((m, s, w1, w2))
            m, s = eval_config("bf", [m1, m2], w)
            bi_bf.append((m, s, w1, w2))

        bi_wa.sort(key=lambda x: (-x[0], x[1]))
        bi_bf.sort(key=lambda x: (-x[0], x[1]))

        print(f"\nTop 5 Weighted Averaging:")
        print(f"  {'w_' + m1:>8s}  {'w_' + m2:>8s}  {'F1':>7s}  {'std':>7s}")
        for m, s, w1, w2 in bi_wa[:5]:
            print(f"  {w1:8.2f}  {w2:8.2f}  {m:7.4f}  {s:7.4f}")

        print(f"\nTop 5 Bayesian Fusion:")
        print(f"  {'w_' + m1:>8s}  {'w_' + m2:>8s}  {'F1':>7s}  {'std':>7s}")
        for m, s, w1, w2 in bi_bf[:5]:
            print(f"  {w1:8.2f}  {w2:8.2f}  {m:7.4f}  {s:7.4f}")

        m, s = eval_config("mv", [m1, m2])
        print(f"\n  Majority Vote: F1={m:.4f} +/- {s:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("OVERALL BEST PER CATEGORY")
    print(f"{'=' * 90}")
    print(f"  {'Category':<40s}  {'Method':<5s}  {'F1':>7s}  {'std':>7s}  Weights")
    print(f"  {'-' * 85}")

    # Unimodal
    for mod_label, fname in [
        ("Text (MacBERT)", MODALITY_FILES["text"]),
        ("Speech (XLSR)", MODALITY_FILES["speech"]),
        ("EEG (CBraMod)", MODALITY_FILES["eeg"]),
    ]:
        d = {}
        with open(PREDICTIONS_DIR / fname) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                fold = int(row["fold"])
                if fold not in d:
                    d[fold] = {"gt": [], "pred": []}
                d[fold]["gt"].append(int(row["ground_truth"]))
                d[fold]["pred"].append(int(row["predicted_label"]))
        f1s = [f1_score(d[k]["gt"], d[k]["pred"], average="macro") for k in sorted(d)]
        print(
            f"  {mod_label:<40s}  {'--':<5s}  {np.mean(f1s):7.4f}  {np.std(f1s, ddof=1):7.4f}  --"
        )

    # Best bimodal
    for m1, m2 in bimodal:
        label = f"{m1.upper()}+{m2.upper()}"
        best_all = []
        for w1 in weight_vals:
            w2 = round(1.0 - w1, 2)
            if w2 < 0.05:
                continue
            w = {m1: w1, m2: w2}
            for method in ["wa", "bf"]:
                m, s = eval_config(method, [m1, m2], w)
                best_all.append((m, s, method, w1, w2))
        m, s = eval_config("mv", [m1, m2])
        best_all.append((m, s, "mv", None, None))
        best_all.sort(key=lambda x: (-x[0], x[1]))
        b = best_all[0]
        wstr = f"{m1}={b[3]:.2f}, {m2}={b[4]:.2f}" if b[3] is not None else "--"
        print(f"  {label:<40s}  {b[2].upper():<5s}  {b[0]:7.4f}  {b[1]:7.4f}  {wstr}")

    # Best trimodal
    best_tri = []
    for r in tri_wa:
        best_tri.append((r[0], r[1], "wa", r[2], r[3], r[4]))
    for r in tri_bf:
        best_tri.append((r[0], r[1], "bf", r[2], r[3], r[4]))
    m, s = eval_config("mv", ["eeg", "speech", "text"])
    best_tri.append((m, s, "mv", None, None, None))
    best_tri.sort(key=lambda x: (-x[0], x[1]))
    b = best_tri[0]
    wstr = (
        f"eeg={b[3]:.2f}, sp={b[4]:.2f}, txt={b[5]:.2f}" if b[3] is not None else "--"
    )
    print(
        f"  {'EEG+SPEECH+TEXT':<40s}  {b[2].upper():<5s}  {b[0]:7.4f}  {b[1]:7.4f}  {wstr}"
    )


if __name__ == "__main__":
    main()

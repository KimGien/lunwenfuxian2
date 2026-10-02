"""Recheck local fusion selection and evaluate the paper's fixed WA weights.

Read existing predictions only. No fitting, new weight search on test labels,
or changes to the archived experiment protocol.
"""
from collections import Counter
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .audit_results import WORK, OUT, SEEDS, FOLDS, verify_descriptor, read, demand
from tridep_ccml.metrics import binary_metrics

DEST = OUT / "fusion_recheck"
CLEAN = WORK / "tridep_ccml/experiments/TC-00-E-37-fulltext"
PAPER_WEIGHTS = (0.05, 0.35, 0.60)
MODALITIES = ("eeg", "speech", "text")
spec = importlib.util.spec_from_file_location("s00_fusion_metrics", WORK / "S00/scripts/metrics.py")
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)


def save(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        demand(path.read_text(encoding="utf-8") == content, "Existing recheck changed")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def evaluate(rows, weights):
    probs = [sum(w * row["p_" + modality] for w, modality in zip(weights, MODALITIES)) for row in rows]
    return local.metrics([r["label"] for r in rows], probs)["macro_f1"]


def ties(rows, saved_selection):
    val = [row for row in rows if row["split"] == "val"]
    demand(len(val) == 3, "Unexpected validation count")
    selected = local.select_weights(val)
    demand(selected == saved_selection, "Local selection cannot be reproduced")
    candidates = []
    for i in range(21):
        for j in range(21-i):
            weights = (i/20, j/20, (20-i-j)/20)
            score = evaluate(val, weights)
            candidates.append((score, weights))
    best = max(score for score, weights in candidates)
    best_weights = [w for score, w in candidates if score == best]
    return {"val_subjects": len(val), "selected": selected, "candidate_count": len(candidates),
            "best_tie_count": len(best_weights),
            "best_with_positive_eeg_count": sum(w[0] > 0 for w in best_weights),
            "best_with_all_modalities_positive_count": sum(min(w) >= .05 for w in best_weights),
            "min_eeg_weight_among_best": min(w[0] for w in best_weights),
            "max_eeg_weight_among_best": max(w[0] for w in best_weights),
            "paper_weights_validation_f1": evaluate(val, PAPER_WEIGHTS)}


def summarize(rows, key):
    keys = ("macro_f1", "mdd_recall", "balanced_accuracy", "nll", "brier")
    by_seed = {}
    for seed in SEEDS:
        folds = {}
        for fold in FOLDS:
            subset = [r for r in rows if r["seed"] == seed and r["outer_fold"] == fold and r["split"] == "test"]
            labels, probs = [r["label"] for r in subset], [r[key] for r in subset]
            metrics = binary_metrics(labels, probs)
            independent = f1_score(labels, np.asarray(probs) > .5, labels=[0, 1], average="macro", zero_division=0)
            demand(abs(metrics["macro_f1"] - independent) < 1e-12, "Independent F1 mismatch")
            folds[str(fold)] = metrics
        by_seed[str(seed)] = {"folds": folds, "mean": {
            k: float(np.mean([folds[str(f)][k] for f in FOLDS])) for k in keys}}
    return {"primary": {k: float(np.mean([by_seed[str(s)]["mean"][k] for s in SEEDS])) for k in keys},
            "by_seed": by_seed}


def main():
    fold_audits, predictions = [], []
    for seed in SEEDS:
        for fold in FOLDS:
            clean_folder = CLEAN / f"runs/seed_{seed}/fold_{fold}"
            aug_folder = OUT / f"tridep/runs/seed_{seed}/fold_{fold}"
            for folder in (clean_folder, aug_folder):
                for descriptor in read(folder / "complete.json").values():
                    verify_descriptor(descriptor)
            clean = read(clean_folder / "predictions.json")
            augmented = read(aug_folder / "predictions.json")
            demand(len(clean) == len(augmented) == 37, "Incomplete predictions")
            fold_audits.append({"seed": seed, "fold": fold,
                "control": ties(clean, read(clean_folder / "fusion_selection.json")),
                "augmented": ties(augmented, read(aug_folder / "fusion_selection.json"))})
            for old, new in zip(clean, augmented):
                demand(all(old[k] == new[k] for k in ("subject_id", "seed", "outer_fold", "split", "label", "p_speech", "p_text")),
                       "Subject identity or unchanged modality differs")
                row = {k: old[k] for k in ("subject_id", "seed", "outer_fold", "split", "label")}
                row["weights"] = PAPER_WEIGHTS
                for name, source in (("control", old), ("augmented", new)):
                    row["p_" + name] = sum(w * source["p_" + m] for w, m in zip(PAPER_WEIGHTS, MODALITIES))
                    row["prediction_" + name] = int(row["p_" + name] > .5)
                predictions.append(row)
    for seed in SEEDS:
        test = [r for r in predictions if r["seed"] == seed and r["split"] == "test"]
        demand(len(test) == len({r["subject_id"] for r in test}) == 37, "OOF subject coverage differs")
    stats = {}
    for condition in ("control", "augmented"):
        stats[condition] = {
            "selected_eeg_zero_runs": sum(f[condition]["selected"]["weights"][0] == 0 for f in fold_audits),
            "runs_with_positive_eeg_among_best": sum(f[condition]["best_with_positive_eeg_count"] > 0 for f in fold_audits),
            "runs_with_all_positive_among_best": sum(f[condition]["best_with_all_modalities_positive_count"] > 0 for f in fold_audits),
            "best_tie_count_range": [min(f[condition]["best_tie_count"] for f in fold_audits), max(f[condition]["best_tie_count"] for f in fold_audits)],
        }
    control, augmented = summarize(predictions, "p_control"), summarize(predictions, "p_augmented")
    changed = [r for r in predictions if r["split"] == "test" and r["prediction_control"] != r["prediction_augmented"]]
    report = {"status": "passed", "paper": "https://arxiv.org/html/2510.14922v2#S4.T4",
              "analysis": "Post-hoc protocol recheck, no training and no test-label weight selection",
              "paper_weighted_average_weights": PAPER_WEIGHTS,
              "local_grid_includes_zero": True, "upstream_trimodal_grid_min_weight": .05,
              "local_tie_break": "First maximum in ascending EEG then speech grid order",
              "local_selection_summary": stats, "local_selection_by_fold": fold_audits,
              "fixed_paper_weight_control": control, "fixed_paper_weight_augmented": augmented,
              "fixed_paper_weight_delta": {k: augmented["primary"][k]-control["primary"][k] for k in control["primary"]},
              "fixed_paper_weight_test_flips": {"total": len(changed),
                  "corrected": sum(r["prediction_augmented"] == r["label"] for r in changed),
                  "regressed": sum(r["prediction_augmented"] != r["label"] for r in changed)},
              "independent_subjects": 37, "test_records_per_condition": 111,
              "scope": "Same local subject predictions, not an exact reproduction of the paper's 38-subject result"}
    save(DEST / "paper_weight_predictions.json", predictions)
    save(DEST / "recheck.json", report)
    print(json.dumps({"selection": stats, "paper_weight_f1_control": control["primary"]["macro_f1"],
                      "paper_weight_f1_augmented": augmented["primary"]["macro_f1"],
                      "test_flips": report["fixed_paper_weight_test_flips"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Descriptive TC metrics, keeping fold means separate from pooled OOF."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .records import ROOT, validate_prediction_records


def binary_metrics(labels, probabilities):
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Nonempty labels and probabilities must have equal length")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("Expected 0=HC, 1=MDD")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid MDD probabilities")
    predicted = [int(p > .5) for p in probabilities]
    recalls, f1s = [], []
    for c in (0, 1):
        tp = sum(y == c and p == c for y, p in zip(labels, predicted))
        fp = sum(y != c and p == c for y, p in zip(labels, predicted))
        fn = sum(y == c and p != c for y, p in zip(labels, predicted))
        recalls.append(tp / (tp + fn) if tp + fn else None)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    eps = 1e-15
    nll = -sum(math.log(max(eps, min(1 - eps, p if y else 1 - p)))
               for y, p in zip(labels, probabilities)) / len(labels)
    return {"n_subjects": len(labels), "macro_f1": sum(f1s) / 2,
            "mdd_recall": recalls[1],
            "balanced_accuracy": sum(recalls) / 2 if None not in recalls else None,
            "nll": nll,
            "brier": sum((p - y) ** 2 for y, p in zip(labels, probabilities)) / len(labels)}


def reliability_ece5(labels, probabilities):
    """Five equal-width bins of MDD probability; ECE compares P(MDD) to its rate.

    This is binary class-probability calibration (not top-label confidence ECE).
    Intervals are [lower,upper), with p=1 included in the last interval.
    """
    binary_metrics(labels, probabilities)
    bins = []
    ece = 0.
    for b in range(5):
        indices = [i for i, p in enumerate(probabilities) if min(int(p * 5), 4) == b]
        mean_p = sum(probabilities[i] for i in indices) / len(indices) if indices else None
        rate = sum(labels[i] for i in indices) / len(indices) if indices else None
        if indices:
            ece += len(indices) / len(labels) * abs(mean_p - rate)
        bins.append({"lower": b / 5, "upper": (b + 1) / 5, "n": len(indices),
                     "mean_p_mdd": mean_p, "observed_mdd_rate": rate})
    return {"ece5": ece, "definition": "MDD probability vs MDD event rate; 5 equal-width bins",
            "bins": bins}


def uncertainty_diagnostics(rows):
    if any(r["u_final"] is None for r in rows):
        return None
    correct, incorrect = [], []
    for row in rows:
        (correct if row["prediction"] == row["label"] else incorrect).append(row["u_final"])
    ordered = sorted(rows, key=lambda r: (r["u_final"], r["subject_id"]))
    # Add all tied-u subjects together, avoiding arbitrary selective ranking.
    curve, errors, accepted = [], 0, 0
    for i, row in enumerate(ordered):
        errors += row["prediction"] != row["label"]
        accepted += 1
        if i == len(ordered) - 1 or ordered[i + 1]["u_final"] != row["u_final"]:
            curve.append({"max_u": row["u_final"], "accepted": accepted,
                          "coverage": accepted / len(rows), "risk": errors / accepted})
    return {"correct_u": correct, "incorrect_u": incorrect,
            "risk_definition": "misclassification rate among subjects retained by u <= max_u",
            "ties": "all equal-u subjects retained together", "risk_coverage": curve}


def _mean_metrics(metrics):
    names = ("macro_f1", "mdd_recall", "balanced_accuracy", "nll", "brier")
    return {key: sum(row[key] for row in metrics) / len(metrics)
            if all(row[key] is not None for row in metrics) else None for key in names}


def summarize(rows):
    """Call validate_prediction_records first for a complete formal report."""
    test = [row for row in rows if row["split"] == "test"]
    if not test:
        raise ValueError("No test predictions")
    if len({(r["experiment"], r["seed"], r["subject_id"]) for r in test}) != len(test):
        raise ValueError("A subject must appear once per experiment and seed")
    if len({r["experiment"] for r in test}) != 1:
        raise ValueError("Summarize one experiment at a time")
    groups = defaultdict(list)
    for row in test:
        groups[row["seed"]].append(row)
    seed_reports = {}
    for seed, group in sorted(groups.items()):
        fold_reports = {}
        for fold in sorted({r["outer_fold"] for r in group}):
            subset = [r for r in group if r["outer_fold"] == fold]
            fold_reports[str(fold)] = binary_metrics([r["label"] for r in subset],
                                                     [r["p_final"][1] for r in subset])
        pooled = binary_metrics([r["label"] for r in group], [r["p_final"][1] for r in group])
        pooled.update(reliability_ece5([r["label"] for r in group], [r["p_final"][1] for r in group]))
        seed_reports[str(seed)] = {"fold_metrics": fold_reports,
                                   "unweighted_fold_mean": _mean_metrics(list(fold_reports.values())),
                                   "pooled_oof": pooled,
                                   "uncertainty": uncertainty_diagnostics(group)}
    return {"experiment": test[0]["experiment"],
            "evaluation_scope": "fixed-split developmental comparison; previously viewed test cohort",
            "independent_subjects": len({r["subject_id"] for r in test}),
            "seed_reports": seed_reports,
            "primary_seed_mean_of_fold_means": _mean_metrics(
                [r["unweighted_fold_mean"] for r in seed_reports.values()]),
            "separate_seed_mean_of_pooled_oof": _mean_metrics(
                [r["pooled_oof"] for r in seed_reports.values()]),
            "note": "Seeds/folds are not additional independent subjects. u is evidence insufficiency."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    validation = validate_prediction_records(rows, args.experiment, root=args.root)
    report = summarize(rows)
    report["validation"] = validation
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"passed": True, "report": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

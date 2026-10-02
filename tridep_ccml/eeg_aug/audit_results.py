"""Independent, read-only audit of the three EEG augmentation comparisons.

The script reads predictions and checkpoints already saved by the experiment
runners. It never retrains a model or selects a result using test labels.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


WORK = Path(__file__).resolve().parents[2]
OUT = WORK / "tridep_ccml/experiments/EEG-AUG-JITTER-37-fulltext"
SEEDS = (11, 22, 33)
FOLDS = (1, 2, 3, 4, 5)
METRICS = ("macro_f1", "mdd_recall", "balanced_accuracy", "nll", "brier")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def demand(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_descriptor(descriptor):
    path = WORK / descriptor["path"]
    demand(path.is_file(), f"Missing file: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    demand(digest == descriptor["sha256"], f"Changed file: {path}")
    return path


def identity(row):
    return row["seed"], row["outer_fold"], row["subject_id"]


def test_map(rows):
    demand(len(rows) == 555, "Expected 15 x 37 subject predictions")
    grouped = Counter((row["seed"], row["outer_fold"]) for row in rows)
    demand(set(grouped) == {(s, f) for s in SEEDS for f in FOLDS}, "Missing seed/fold")
    demand(all(count == 37 for count in grouped.values()), "Missing within-fold subject")
    out = {identity(row): row for row in rows if row["split"] == "test"}
    demand(len(out) == 111, "Expected one outer-test row per subject and seed")
    for seed in SEEDS:
        ids = [key[2] for key in out if key[0] == seed]
        demand(len(ids) == len(set(ids)) == 37, "OOF identity duplication")
    return out


def probability(row, route):
    if route == "tridep_weighted_new":
        return float(row["p_final"])
    if route == "tridep_mean_new":
        return float(row["p_mean"])
    if route == "tridep_eeg_new":
        return float(row["p_eeg"])
    if route in ("tridep_weighted_old", "tridep_mean_old", "tridep_eeg_old"):
        return float(row["p_final"][1])
    if route in ("tc_fusion", "v2_fusion"):
        return float(row["p_final"][1])
    if route in ("tc_eeg", "v2_eeg"):
        return float(row["views"]["eeg"]["p"][1])
    raise ValueError(route)


def scores(rows, route):
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    probs = np.asarray([probability(row, route) for row in rows], dtype=np.float64)
    demand(len(probs) > 0 and np.isfinite(probs).all() and
           np.logical_and(probs >= 0, probs <= 1).all(), "Invalid probability")
    predicts = (probs > 0.5).astype(np.int64)
    clipped = np.clip(probs, 1e-12, 1 - 1e-12)
    return {
        "macro_f1": float(f1_score(labels, predicts, labels=[0, 1], average="macro", zero_division=0)),
        "mdd_recall": float(recall_score(labels, predicts, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicts)),
        "nll": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
        "brier": float(np.mean((labels - probs) ** 2)),
    }


def fold_mean(test, route):
    by_seed = {}
    for seed in SEEDS:
        folds = {}
        for fold in FOLDS:
            rows = [row for key, row in test.items() if key[:2] == (seed, fold)]
            demand(len(rows) in (7, 8), "Incorrect test fold size")
            folds[str(fold)] = scores(rows, route)
        by_seed[str(seed)] = {metric: float(np.mean([folds[str(f)][metric] for f in FOLDS]))
                              for metric in METRICS}
    main = {metric: float(np.mean([by_seed[str(s)][metric] for s in SEEDS]))
            for metric in METRICS}
    return {"primary": main, "by_seed": by_seed}


def compare(old, new, old_route, new_route):
    demand(set(old) == set(new), "Paired test identities differ")
    for key in old:
        demand(old[key]["label"] == new[key]["label"], "Paired labels differ")
    old_scores = fold_mean(old, old_route)
    new_scores = fold_mean(new, new_route)
    flips = Counter()
    changes = []
    for key in sorted(old):
        a = probability(old[key], old_route)
        b = probability(new[key], new_route)
        ay, by = int(a > .5), int(b > .5)
        if ay != by:
            flips["total"] += 1
            flips["corrected"] += int(ay != old[key]["label"] and by == old[key]["label"])
            flips["regressed"] += int(ay == old[key]["label"] and by != old[key]["label"])
        changes.append(abs(b-a))
    return {
        "control": old_scores, "augmented": new_scores,
        "delta": {m: new_scores["primary"][m] - old_scores["primary"][m] for m in METRICS},
        "f1_delta_by_seed": {str(s): new_scores["by_seed"][str(s)]["macro_f1"] -
                             old_scores["by_seed"][str(s)]["macro_f1"] for s in SEEDS},
        "test_hard_prediction_flips": dict(flips),
        "test_probability_mean_absolute_change": float(np.mean(changes)),
        "test_probability_max_absolute_change": float(np.max(changes)),
    }


def check_summary(result, summary, branch=None):
    if branch:
        summary = summary["branch_summaries"][branch]
    expected = summary["primary_seed_mean_of_fold_means"]
    for key in METRICS:
        demand(np.isclose(result["primary"][key], expected[key], atol=1e-8),
               f"Stored summary mismatch: {branch or 'fusion'} {key}")


def check_runs():
    for family in ("tridep", "v2", "tc_clean", "tc_aug"):
        for seed in SEEDS:
            for fold in FOLDS:
                folder = (OUT / "tridep/runs" if family == "tridep" else
                          OUT / "runs" / family) / f"seed_{seed}/fold_{fold}"
                manifest = read(folder / "complete.json")
                for descriptor in manifest.values():
                    verify_descriptor(descriptor)
                side = read(folder / "checkpoint.json")
                verify_descriptor(side["checkpoint"])


def check_tc_pair():
    changed_eeg = 0
    for seed in SEEDS:
        for fold in FOLDS:
            clean = OUT / f"runs/tc_clean/seed_{seed}/fold_{fold}"
            aug = OUT / f"runs/tc_aug/seed_{seed}/fold_{fold}"
            cm = read(clean / "checkpoint.json")["metadata"]
            am = read(aug / "checkpoint.json")["metadata"]
            demand(cm["initialization_sha256"] == am["initialization_sha256"] and
                   cm["batch_order_sha256"] == am["batch_order_sha256"],
                   "TC initial state or batch order differs")
            demand(cm["optimizer_updates"] == am["optimizer_updates"] == 240,
                   "TC update count differs")
            demand(cm["parent_checkpoints"]["speech"] == am["parent_checkpoints"]["speech"] and
                   cm["parent_checkpoints"]["text"] == am["parent_checkpoints"]["text"],
                   "TC unchanged parent checkpoint differs")
            cp = OUT / f"features/tc_clean/seed_{seed}/fold_{fold}.npz"
            ap = OUT / f"features/tc_aug/seed_{seed}/fold_{fold}.npz"
            with np.load(cp, allow_pickle=False) as c, np.load(ap, allow_pickle=False) as a:
                demand(np.array_equal(c["features"][:, 1:], a["features"][:, 1:]),
                       "TC speech/text input representations changed")
                changed_eeg += int(not np.array_equal(c["features"][:, 0], a["features"][:, 0]))
    demand(changed_eeg == 15, "TC EEG input did not change in every fold")
    return {"paired_runs": 15, "eeg_feature_exports_changed": changed_eeg,
            "speech_text_feature_exports_identical": True,
            "initialization_and_batch_order_identical": True}


def main():
    check_runs()
    old_all = read(WORK / "tridep_ccml/experiments/TC-00-E-37-fulltext/predictions_all.json")
    new_all = []
    for seed in SEEDS:
        for fold in FOLDS:
            new_all.extend(read(OUT / f"tridep/runs/seed_{seed}/fold_{fold}/predictions.json"))
    new_test = test_map(new_all)
    tridep = {}
    for name, old_route, new_route in (
        ("eeg", "tridep_eeg_old", "tridep_eeg_new"),
        ("weighted", "tridep_weighted_old", "tridep_weighted_new"),
        ("mean", "tridep_mean_old", "tridep_mean_new"),
    ):
        tridep[name] = compare(test_map(old_all[name]), new_test, old_route, new_route)
        actual = tridep[name]["augmented"]
        expected = read(OUT / "tridep/summary.json")["routes"][name]
        check_summary(actual, expected)
    old_weighted = {identity(r): r for r in old_all["weighted"]}
    for row in new_all:
        control = old_weighted[identity(row)]
        demand(row["p_speech"] == control["p_speech"] and row["p_text"] == control["p_text"],
               "TRI-DEP unchanged modality probability differs")

    clean_all = read(OUT / "tc_clean_predictions_all.json")
    aug_all = read(OUT / "tc_aug_predictions_all.json")
    ct, at = test_map(clean_all), test_map(aug_all)
    tc = {"fusion": compare(ct, at, "tc_fusion", "tc_fusion"),
          "eeg": compare(ct, at, "tc_eeg", "tc_eeg"),
          "paired_inputs": check_tc_pair()}
    clean_summary, aug_summary = read(OUT / "tc_clean_summary.json"), read(OUT / "tc_aug_summary.json")
    for route in ("fusion", "eeg"):
        check_summary(tc[route]["control"], clean_summary, None if route == "fusion" else "eeg")
        check_summary(tc[route]["augmented"], aug_summary, None if route == "fusion" else "eeg")

    base_v2 = read(WORK / "tridep_ccml/experiments/CCML-v2-S-37-fulltext/outer_predictions_control.json")
    new_v2 = read(OUT / "v2_predictions.json")
    bv, nv = test_map(base_v2), test_map(new_v2)
    v2 = {"fusion": compare(bv, nv, "v2_fusion", "v2_fusion"),
          "eeg": compare(bv, nv, "v2_eeg", "v2_eeg")}
    vr = read(OUT / "v2_results.json")
    for route in ("fusion", "eeg"):
        if route == "fusion":
            check_summary(v2[route]["augmented"], vr["summary"])
        for condition, saved in (("control", vr["comparison"][route]["control"]),
                                 ("augmented", vr["comparison"][route]["augmented"])):
            for metric in METRICS:
                demand(np.isclose(v2[route][condition]["primary"][metric], saved[metric], atol=1e-8),
                       f"Stored v2 {route} {condition} metric mismatch: {metric}")

    report = {"status": "passed", "cohort_independent_subjects": 37,
              "test_rows_per_condition": 111, "completed_supervised_runs": 60,
              "completed_optimizer_updates": 12600,
              "tridep": tridep, "tc04_corrected_pair": tc, "v2_s_control": v2,
              "interpretation": "Fixed split developmental comparison on previously viewed test subjects; "
                                "three seeds do not create 111 independent subjects."}
    path = OUT / "independent_audit.json"
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        demand(path.read_text(encoding="utf-8") == content, "Existing audit differs")
    else:
        path.write_text(content, encoding="utf-8")
    print(json.dumps({"status": "passed", "runs": 60, "updates": 12600,
                      "macro_f1": {k: {route: round(v["delta"]["macro_f1"], 6)
                                      for route, v in section.items() if "delta" in v}
                                   for k, section in (("tridep", tridep),
                                                      ("tc04", tc), ("v2", v2))}}, ensure_ascii=False))


if __name__ == "__main__":
    main()

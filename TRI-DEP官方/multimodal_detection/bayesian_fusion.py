import json
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from math import log, exp

def load_fold_assignments(filename):
    with open(filename, 'r') as f:
        nested_folds = json.load(f)
    # Returns dict: subject_id -> list of folds (0-based)
    subject_to_folds = {}
    for fold_name, fold_data in nested_folds.items():
        fold_idx = int(fold_name.split('_')[-1]) - 1
        for subject_id in fold_data.get("test", []):
            if subject_id not in subject_to_folds:
                subject_to_folds[subject_id] = []
            subject_to_folds[subject_id].append(fold_idx)
    return subject_to_folds

def load_predictions_and_probabilities(filename):
    predictions = {}
    probabilities = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    identifier, prediction, probability = parts
                    identifier = identifier.strip()
                    try:
                        predictions[identifier] = int(float(prediction))
                        probabilities[identifier] = float(probability)
                    except ValueError as e:
                        print(f"Warning: Skipping line due to parse error in {filename}: {line.strip()} ({e})")
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return {}, {}
    return predictions, probabilities

def bayesian_fusion_weighted_probs(probs, weights, prior_depressed):
    epsilon = 1e-10
    combined_lr = 1.0
    for p, w in zip(probs, weights):
        lr = (p + epsilon) / ((1 - p) + epsilon)
        combined_lr *= exp(w * log(lr))
    posterior_prob = (prior_depressed * combined_lr) / ((prior_depressed * combined_lr) + (1 - prior_depressed))
    return posterior_prob

def decision_level_bayesian_fusion(ids, modality_preds, modality_probs, weights, prior_depressed):
    fused_predictions = {}
    modalities = list(weights.keys())

    for identifier in ids:
        preds = []
        probs = []
        missing_mods = []
        for mod in modalities:
            p = modality_preds[mod].get(identifier)
            pr = modality_probs[mod].get(identifier)
            if p is None or pr is None:
                missing_mods.append(mod)
            else:
                preds.append(p)
                probs.append(pr)
        if missing_mods:
            print(f"Skipping '{identifier}' due to missing data in: {', '.join(missing_mods)}")
            continue

        if all(pred == preds[0] for pred in preds):
            fused_predictions[identifier] = preds[0]
            continue

        wts = [weights[mod] for mod in modalities]
        fused_prob = bayesian_fusion_weighted_probs(probs, wts, prior_depressed)
        fused_predictions[identifier] = 1 if fused_prob >= 0.5 else 0

    return fused_predictions

def evaluate_predictions_from_dict(fused_predictions):
    y_true, y_pred = [], []
    for identifier, pred in fused_predictions.items():
        # Adjust this if your true label logic is different
        true_label = 1 if identifier.startswith('0201') else 0
        y_true.append(true_label)
        y_pred.append(pred)
    if not y_true or not y_pred:
        print("No predictions or labels to evaluate.")
        return None
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return accuracy, precision, recall, f1

def save_all_folds_and_metrics(all_fold_preds, modalities, avg_metrics, std_metrics):
    filename = f"results/bayesian_5fold_{'_'.join(modalities)}.txt"
    with open(filename, 'w') as f:
        for fold_idx, fused_preds in enumerate(all_fold_preds):
            f.write(f"Fold {fold_idx + 1} predictions:\n")
            for identifier, pred in fused_preds.items():
                f.write(f"{identifier},{pred}\n")
            f.write("\n")

        f.write("Average 5-Fold Metrics (mean ± std):\n")
        for metric in avg_metrics:
            f.write(f"{metric.capitalize()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}\n")

    print(f"Saved all folds and average metrics to {filename}")

def get_user_config():
    available_modalities = ['eeg', 'speech', 'text']
    print(f"Available modalities: {', '.join(available_modalities)}")
    while True:
        selected = input("Enter modalities separated by commas (e.g., eeg,speech): ").lower().replace(" ", "").split(',')
        if all(mod in available_modalities for mod in selected) and 1 <= len(selected) <= 3:
            break
        print("Invalid selection. Please choose from eeg, speech, text (1-3 modalities).")

    weights = {}
    print("Enter weights for the selected modalities. Weights must be between 0 and 1 and sum to 1.")
    while True:
        for mod in selected:
            while True:
                try:
                    w = float(input(f"Enter weight for {mod}: "))
                    if 0 <= w <= 1:
                        weights[mod] = w
                        break
                    else:
                        print("Weight must be between 0 and 1.")
                except ValueError:
                    print("Invalid input. Enter a number.")
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) < 1e-6:
            break
        print(f"Weights sum to {total_weight:.4f}, but must sum to 1. Please re-enter weights.")

    return selected, weights

if __name__ == '__main__':
    modalities, weights = get_user_config()

    # File mapping
    filename_map = {
        'eeg': 'results/eeg_predictions.txt',
        'speech': 'results/speech_predictions.txt',
        'text': 'results/text_predictions.txt'
    }

    prior_depressed = 17 / 21  # Adjust based on your dataset

    # Load fold assignments
    fold_assignments = load_fold_assignments('split_dataset/fold_assignments.json')

    # Load all modality predictions + probs once
    modality_preds = {}
    modality_probs = {}
    for mod in modalities:
        preds, probs = load_predictions_and_probabilities(filename_map[mod])
        modality_preds[mod] = preds
        modality_probs[mod] = probs

    # Prepare fold to subject IDs
    folds = {i: [] for i in range(5)}
    # Only subjects present in ALL modalities
    common_subjects = set.intersection(*(set(modality_preds[mod].keys()) for mod in modalities))
    for subject_id in common_subjects:
        if subject_id in fold_assignments:
            for fold_idx in fold_assignments[subject_id]:
                folds[fold_idx].append(subject_id)

    all_fold_preds = []
    fold_metrics = []

    for fold_idx in range(5):
        print(f"\nProcessing Fold {fold_idx + 1}...")
        fold_subjects = list(set(folds[fold_idx]))  # unique subjects in fold
        print(f"Subjects in fold: {len(fold_subjects)}")
        fused_preds = decision_level_bayesian_fusion(fold_subjects, modality_preds, modality_probs, weights, prior_depressed)
        all_fold_preds.append(fused_preds)
        metrics = evaluate_predictions_from_dict(fused_preds)
        if metrics:
            acc, prec, rec, f1 = metrics
            print(f"Fold {fold_idx + 1} metrics: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
            fold_metrics.append(metrics)
        else:
            fold_metrics.append((np.nan, np.nan, np.nan, np.nan))

    # Calculate average and std deviation for each metric
    fold_metrics = np.array(fold_metrics, dtype=np.float64)
    avg_metrics = dict(zip(['accuracy', 'precision', 'recall', 'f1'], np.nanmean(fold_metrics, axis=0)))
    std_metrics = dict(zip(['accuracy', 'precision', 'recall', 'f1'], np.nanstd(fold_metrics, axis=0)))

    print("\nFinal 5-Fold Average Metrics (mean ± std):")
    for metric in avg_metrics:
        print(f"{metric.capitalize()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    save_all_folds_and_metrics(all_fold_preds, modalities, avg_metrics, std_metrics)

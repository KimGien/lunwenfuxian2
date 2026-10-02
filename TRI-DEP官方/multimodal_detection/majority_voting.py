import json
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import numpy as np

def load_nested_fold_assignments(filename='split_dataset/fold_assignments.json'):
    with open(filename, 'r') as f:
        nested_folds = json.load(f)

    subject_to_fold = {}
    fold_subjects = {}
    for fold_name, fold_data in nested_folds.items():
        fold_idx = int(fold_name.split('_')[-1]) - 1
        subjects = fold_data.get("test", [])
        fold_subjects[fold_idx] = subjects
        for subject_id in subjects:
            subject_to_fold[subject_id] = fold_idx
    return subject_to_fold, fold_subjects

def load_predictions_and_probabilities(filename):
    predictions = {}
    probabilities = {}
    with open(filename, 'r') as file:
        next(file)  # Skip header
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 3:
                identifier, prediction, probability = parts
                predictions[identifier] = int(float(prediction))
                probabilities[identifier] = float(probability)
    return predictions, probabilities

def decision_level_fusion(selected_modalities, preds_dict, probs_dict, test_ids):
    fused_predictions = {}
    for identifier in test_ids:
        if not all(identifier in preds_dict[mod] for mod in selected_modalities):
            continue
        preds = [preds_dict[mod][identifier] for mod in selected_modalities]
        probs = [probs_dict[mod][identifier] for mod in selected_modalities]
        count_0 = preds.count(0)
        count_1 = preds.count(1)
        if count_0 > count_1:
            fused_predictions[identifier] = 0
        elif count_1 > count_0:
            fused_predictions[identifier] = 1
        else:
            avg_prob = sum(probs) / len(probs)
            fused_predictions[identifier] = 1 if avg_prob > 0.5 else 0
    return fused_predictions

def evaluate_predictions(fused_predictions):
    y_true = []
    y_pred = []
    for identifier, prediction in fused_predictions.items():
        true_label = 1 if identifier.startswith('0201') else 0
        y_true.append(true_label)
        y_pred.append(prediction)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def save_results(fold_metrics, fold_predictions, fold_subjects, filename):
    with open(filename, 'w') as f:
        all_acc, all_prec, all_rec, all_f1 = [], [], [], []
        for fold_idx, predictions in fold_predictions.items():
            f.write(f"# --- Fold {fold_idx + 1} ---\n")
            f.write(f"# Subjects: {', '.join(fold_subjects[fold_idx])}\n")
            for identifier, prediction in predictions.items():
                f.write(f"{identifier},{prediction}\n")
            acc, prec, rec, f1_ = fold_metrics[fold_idx]
            all_acc.append(acc)
            all_prec.append(prec)
            all_rec.append(rec)
            all_f1.append(f1_)
            f.write(f"# Accuracy: {acc:.4f}\n")
            f.write(f"# Precision: {prec:.4f}\n")
            f.write(f"# Recall: {rec:.4f}\n")
            f.write(f"# F1 Score: {f1_:.4f}\n\n")

        f.write("# === Overall Metrics ===\n")
        f.write(f"# Mean Accuracy: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}\n")
        f.write(f"# Mean Precision: {np.mean(all_prec):.4f} ± {np.std(all_prec):.4f}\n")
        f.write(f"# Mean Recall: {np.mean(all_rec):.4f} ± {np.std(all_rec):.4f}\n")
        f.write(f"# Mean F1 Score: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}\n")

def run_mean_fusion_5fold():
    selected_modalities = input("Enter modalities separated by commas (e.g., eeg,speech,text): ").lower().replace(" ", "").split(',')
    filenames = {
        'speech': 'results/speech_predictions.txt',
        'text': 'results/text_predictions.txt',
        'eeg': 'results/eeg_predictions.txt'
    }

    preds_dict = {mod: load_predictions_and_probabilities(filenames[mod])[0] for mod in selected_modalities}
    probs_dict = {mod: load_predictions_and_probabilities(filenames[mod])[1] for mod in selected_modalities}

    subject_to_fold, fold_subjects = load_nested_fold_assignments()

    fold_predictions = {}
    fold_metrics = {}

    for fold_idx, test_ids in fold_subjects.items():
        fused = decision_level_fusion(selected_modalities, preds_dict, probs_dict, test_ids)
        metrics = evaluate_predictions(fused)
        fold_predictions[fold_idx] = fused
        fold_metrics[fold_idx] = metrics

    result_filename = f"results/mean_fusion_5fold_{'_'.join(selected_modalities)}.txt"
    save_results(fold_metrics, fold_predictions, fold_subjects, result_filename)
    print(f"\n✅ All results saved to {result_filename}")

# Run the script
run_mean_fusion_5fold()

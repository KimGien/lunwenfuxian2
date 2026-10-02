import json
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score

def load_nested_fold_assignments(filename):
    with open(filename, 'r') as f:
        nested_folds = json.load(f)
    
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
            next(file)  # skip header if present
            for line in file:
                parts = line.strip().split(',')
                if len(parts) == 3:
                    identifier, prediction, probability = parts
                    predictions[identifier] = int(float(prediction))
                    probabilities[identifier] = float(probability)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return {}, {}
    return predictions, probabilities

def decision_level_fusion(ids, modality_preds, modality_probs, weights):
    fused_predictions = {}
    for identifier in ids:
        try:
            preds = [modality_preds[mod][identifier] for mod in weights.keys()]
            # If all modalities agree, just pick that class
            if all(p == preds[0] for p in preds):
                fused_predictions[identifier] = preds[0]
                continue
            # Otherwise, do weighted average of probabilities
            weighted_prob = sum(weights[mod] * modality_probs[mod][identifier] for mod in weights)
            fused_predictions[identifier] = 1 if weighted_prob > 0.5 else 0
        except KeyError:
            print(f"Warning: Missing prediction/probability for subject {identifier} in one of the modalities")
    return fused_predictions

def evaluate_fold_predictions(fused_predictions):
    y_true = []
    y_pred = []
    for identifier, pred in fused_predictions.items():
        # Determine true label by ID pattern or other logic
        true_label = 1 if identifier.startswith('0201') or 'PM' in identifier or 'PF' in identifier else 0
        y_true.append(true_label)
        y_pred.append(pred)

    if not y_true or not y_pred:
        print("Warning: Empty predictions or labels in fold!")
        return {'accuracy': np.nan, 'precision': 0, 'recall': 0, 'f1': 0}

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

if __name__ == '__main__':
    try:
        # User input: choose modalities
        valid_modalities = ['eeg', 'speech', 'text']
        selected_modalities = input("Enter modalities separated by commas (eeg,speech,text): ").strip().lower().split(',')
        selected_modalities = [mod.strip() for mod in selected_modalities if mod.strip() in valid_modalities]
        if len(selected_modalities) < 2 or len(selected_modalities) > 3:
            raise ValueError("Select 2 or 3 valid modalities from eeg, speech, text.")

        # User input: weights per modality
        weights = {}
        total_weight = 0.0
        for mod in selected_modalities:
            w = float(input(f"Enter weight for {mod} (0 to 1): "))
            if not (0 <= w <= 1):
                raise ValueError("Weight must be between 0 and 1.")
            weights[mod] = w
            total_weight += w
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.")

        # Load fold assignments (subject -> list of folds)
        subject_to_folds = load_nested_fold_assignments('split_dataset/fold_assignments.json')
        print(f"Number of subjects assigned to folds: {len(subject_to_folds)}")

        # Show subjects assigned to fold 1 (index 0)
        fold1_subjects = [sid for sid, folds in subject_to_folds.items() if 0 in folds]
        print(f"Fold 1 test subjects ({len(fold1_subjects)}): {fold1_subjects}")

        # Load predictions and probabilities for selected modalities
        file_paths = {
            'eeg': 'results/eeg_predictions.txt',
            'speech': 'results/speech_predictions.txt',
            'text': 'results/text_predictions.txt'
        }

        modality_preds = {}
        modality_probs = {}

        for mod in selected_modalities:
            preds, probs = load_predictions_and_probabilities(file_paths[mod])
            if not preds or not probs:
                raise RuntimeError(f"Failed to load predictions/probabilities for {mod}.")
            modality_preds[mod] = preds
            modality_probs[mod] = probs

        # Build fold-to-subjects mapping (subjects can appear in multiple folds)
        folds = {i: [] for i in range(5)}
        common_ids = set.intersection(*[set(modality_preds[mod].keys()) for mod in selected_modalities])
        for subject_id in common_ids:
            if subject_id in subject_to_folds:
                for fold_idx in subject_to_folds[subject_id]:
                    folds[fold_idx].append(subject_id)

        # Evaluate each fold and store metrics
        fold_metrics = []
        for fold_idx in range(5):
            ids_in_fold = folds[fold_idx]
            unique_ids = list(set(ids_in_fold))  # remove duplicates if any
            print(f"\nFold {fold_idx + 1} - {len(unique_ids)} subjects")
            fused_preds = decision_level_fusion(unique_ids, modality_preds, modality_probs, weights)
            metrics = evaluate_fold_predictions(fused_preds)
            fold_metrics.append(metrics)
            print(f"  Metrics: {metrics}")

            # Optional: you can still save per fold predictions individually if needed
            # with open(f"fused_fold_{fold_idx+1}.txt", 'w') as f:
            #     for identifier, pred in fused_preds.items():
            #         f.write(f"{identifier},{pred}\n")

        # Aggregate mean and std metrics over folds
        metrics_per_key = {
            key: [fold_metrics[i][key] for i in range(5) if not np.isnan(fold_metrics[i][key])]
            for key in fold_metrics[0]
        }
        avg_metrics = {key: np.mean(values) if values else np.nan for key, values in metrics_per_key.items()}
        std_metrics = {key: np.std(values) if values else np.nan for key, values in metrics_per_key.items()}

        print("\nAverage 5-Fold Metrics (mean ± std):")
        for metric in avg_metrics:
            print(f"{metric.capitalize()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

        # Save ALL folds + metrics in ONE file
        combined_filename = f"results/WA_{'_'.join(selected_modalities)}.txt"
        with open(combined_filename, 'w') as f:
            for fold_idx in range(5):
                f.write(f"Fold {fold_idx + 1} predictions:\n")
                ids_in_fold = folds[fold_idx]
                unique_ids = list(set(ids_in_fold))
                fused_preds = decision_level_fusion(unique_ids, modality_preds, modality_probs, weights)
                for identifier, pred in fused_preds.items():
                    f.write(f"{identifier},{pred}\n")
                f.write("\n")

            f.write("Average 5-Fold Metrics (mean ± std):\n")
            for metric in avg_metrics:
                f.write(f"{metric.capitalize()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}\n")

        print(f"\nSaved all folds and average metrics to: {combined_filename}")

    except Exception as e:
        print(f"Error: {e}")

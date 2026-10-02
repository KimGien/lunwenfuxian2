import os
import json
import torch
import numpy as np
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from utils_training.dataloader import UnimodalDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from utils_training.set_seed import set_seed
from utils_training.train_model import train_model
from utils_training.collate_pad import collate_fn_padd  # default import

# ---- Load Config ----
with open("training_module/config.json", "r") as f:
    cfg = json.load(f)

# Dynamic import for dataset class
embedding_filename = cfg.get("embedding_filename", "audio_embedding_gru.npy")

# Dynamic dataset class selection
dataset_class_name = cfg["dataset_class"]
dataset_cls = globals().get(dataset_class_name)
if dataset_cls is None:
    raise ValueError(f"Dataset class {dataset_class_name} not found")

# Dynamic import for saaving the predictions
save_prediction = cfg["save_pred"]

# Dynamic import for classifier
classifier_name = cfg["classifier"]
ClassifierClass = None

if classifier_name.lower() == "convpoolclassifier":
    from detection_module.conv_fc import ConvPoolClassifier
    ClassifierClass = ConvPoolClassifier
elif classifier_name.lower() == "bigruattention":
    from detection_module.bigruattention import BiGRUAttentionClassifier
    ClassifierClass = BiGRUAttentionClassifier
elif classifier_name.lower() == "bilstm_fc":
    from detection_module.bilstm_fc import LSTMClassifier
    ClassifierClass = LSTMClassifier
elif classifier_name.lower() == "lstm_fc":
    from detection_module.lstm_fc import LSTM1Classifier
    ClassifierClass = LSTM1Classifier
elif classifier_name.lower() == "gruattention":
    from detection_module.gruattention import GRUAttentionClassifier
    ClassifierClass = GRUAttentionClassifier
else:
    raise ValueError(f"Unknown classifier: {classifier_name}")

#dynamic input size 
input_size = cfg["input_size"]

# ---- Settings ----
set_seed(42)
device = torch.device(cfg["device"])
base_dir = cfg["base_dir"]

epochs = cfg["hyperparameters"]["epochs"]
lr = cfg["hyperparameters"]["learning_rate"]
hidden_dim = cfg["hyperparameters"]["hidden_dim"]
batch_size = cfg["hyperparameters"]["batch_size"]

# ---- Load folds ----
with open(os.path.join(base_dir, "fold_assignments.json"), "r") as f:
    folds = json.load(f)

results, fold_reports, fold_conf_matrices, all_subject_preds = [], [], [], []

# ---- Cross-validation loop ----
for fold_idx in range(5):
    fold_name = f"fold_{fold_idx + 1}"
    train_ids = folds[fold_name]["train"]
    val_ids = folds[fold_name]["val"]
    test_ids = folds[fold_name]["test"]

    train_subjs = [os.path.join(base_dir, sid) for sid in train_ids]
    val_subjs = [os.path.join(base_dir, sid) for sid in val_ids]
    test_subjs = [os.path.join(base_dir, sid) for sid in test_ids]

    train_set = dataset_cls(train_subjs, embedding_filename=embedding_filename)
    val_set = dataset_cls(val_subjs, embedding_filename=embedding_filename)
    test_set = dataset_cls(test_subjs, embedding_filename=embedding_filename)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_padd)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn_padd)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn_padd)

    # Special handling for ConvPoolClassifier (no hidden_dim needed)
    if classifier_name.lower() == "convpoolclassifier":
        model = ClassifierClass(input_dim=input_size).to(device)
    else:
        model = ClassifierClass(input_dim=input_size, hidden_dim=hidden_dim).to(device)

    optimizer = optim.Adamax(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=epochs)

    # ---- Evaluation ----
    trained_model.eval()
    correct, total = 0, 0
    fold_preds, fold_labels = [], []

    with torch.no_grad():
        for x, y, subject_ids in test_loader:
            x, y = x.to(device), y.to(device)
            output = trained_model(x)
            probs = torch.softmax(output, dim=1)
            preds = output.argmax(dim=1)

            fold_preds.extend(preds.cpu().numpy())
            fold_labels.extend(y.cpu().numpy())
            correct += (preds == y).sum().item()
            total += y.size(0)

            for sid, pred, prob in zip(subject_ids, preds.cpu().numpy(), probs.cpu().numpy()):
                all_subject_preds.append({
                    "subject_id": sid,
                    "prediction": int(pred),
                    "probabilities": prob.tolist()
                })

    acc = correct / total
    precision = precision_score(fold_labels, fold_preds, average='macro')
    recall = recall_score(fold_labels, fold_preds, average='macro')
    f1 = f1_score(fold_labels, fold_preds, average='macro')

    print(f"Fold {fold_idx + 1} Test Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    results.append((acc, precision, recall, f1))

    fold_reports.append(classification_report(fold_labels, fold_preds, output_dict=True, target_names=["Healthy", "Depressed"]))
    fold_conf_matrices.append(confusion_matrix(fold_labels, fold_preds))

# ---- Final results ----
results = np.array(results)
print(f"\n5-Fold CV Results:")
print(f"Mean Accuracy  = {results[:,0].mean():.4f} ± {results[:,0].std():.4f}")
print(f"Mean Precision = {results[:,1].mean():.4f} ± {results[:,1].std():.4f}")
print(f"Mean Recall    = {results[:,2].mean():.4f} ± {results[:,2].std():.4f}")
print(f"Mean F1-Score  = {results[:,3].mean():.4f} ± {results[:,3].std():.4f}")

with open(save_prediction, "w") as f:
    json.dump(all_subject_preds, f, indent=4)
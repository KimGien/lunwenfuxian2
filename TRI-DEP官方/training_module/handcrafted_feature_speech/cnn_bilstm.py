import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
from glob import glob
from tqdm import tqdm

# Fix random seed for reproducibility
def set_seed(seed=42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)


import torch
import torch.nn as nn



class AudioTemporalBiLSTMEncoder(nn.Module):
    def __init__(self, input_dim=46, cnn_dim=128, lstm_dim=256):
        super(AudioTemporalBiLSTMEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(cnn_dim, lstm_dim, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_dim)

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Mean pooling over valid lengths
        out = out.sum(dim=1) / lengths.unsqueeze(1).to(out.device)  # (batch, 2 * lstm_dim)
        return out  # (batch, 2*lstm_dim)


class AudioFCClassifier(nn.Module):
    def __init__(self, input_dim=256, num_classes=2, dropout=0.3):
        super(AudioFCClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: (batch, input_dim)
        return self.classifier(x)


class SubjectAudioDataset(Dataset):
    def __init__(self, subject_dirs):
        self.subject_dirs = subject_dirs
        self.labels = [1 if os.path.basename(d).startswith("0201") else 0 for d in subject_dirs]

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        path = os.path.join(self.subject_dirs[idx], "raw_audio_features.npy")
        arr = np.load(path, allow_pickle=True)  # list of 29 arrays [N_i, 46]
        tensors = [torch.tensor(a, dtype=torch.float32) for a in arr]
        label = self.labels[idx]
        return tensors, label

def collate_fn(batch):
    """
    batch: list of tuples (list of tensors per recording, label)
    Return:
        padded_segments: (total_segments_in_batch, max_seq_len, feature_dim=46)
        segment_lengths: lengths of each segment sequence (total_segments_in_batch,)
        segment_to_subject: tensor mapping each segment index to which subject idx in batch it belongs
        subject_labels: tensor of labels for each subject in batch
    """
    sequences, labels = zip(*batch)  # sequences: list of lists of tensors
    batch_size = len(sequences)
    num_recordings = len(sequences[0])  # 29

    all_segments = []
    segment_lengths = []
    segment_to_subject = []

    for subj_idx, subj in enumerate(sequences):
        for rec in subj:
            all_segments.append(rec)
            segment_lengths.append(rec.shape[0])
            segment_to_subject.append(subj_idx)

    padded_segments = nn.utils.rnn.pad_sequence(all_segments, batch_first=True)  # (total_segments, max_len, 46)
    segment_lengths = torch.tensor(segment_lengths)
    segment_to_subject = torch.tensor(segment_to_subject)
    subject_labels = torch.tensor(labels)


    return padded_segments, segment_lengths, segment_to_subject, subject_labels

import json

set_seed(42)
class AudioModel(nn.Module):
    def __init__(self, encoder, classifier):
        super(AudioModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x, lengths):
        x = self.encoder(x, lengths)  # (total_segments, gru_dim)
        logits = self.classifier(x)   # (total_segments, num_classes)
        return logits


# Utility to extract all segments from dataset to fit scaler
def extract_all_segments(dataset):
    all_segments = []
    for i in range(len(dataset)):
        tensors, _ = dataset[i]
        for seg in tensors:
            all_segments.append(seg.numpy())
    all_data = np.concatenate(all_segments, axis=0)  # (total_segments, 46)
    return all_data


# Normalize batch data with fitted scaler
def normalize_batch(x, scaler):
    # x: (batch, seq_len, features)
    b, s, f = x.shape
    x_reshaped = x.reshape(-1, f).cpu().numpy()
    x_norm = scaler.transform(x_reshaped)
    x_norm = torch.tensor(x_norm, dtype=torch.float32).to(x.device)
    x_norm = x_norm.reshape(b, s, f)
    return x_norm


base_dir = "split_dataset"

# Main training loop with cross-validation
set_seed(42)

def run_training():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 110
    lr = 0.0005

    results = []
    fold_reports = []
    fold_conf_matrices = []

    with open("split_dataset/fold_assignments.json", "r") as f:
        folds = json.load(f)

    results = []
    fold_reports = []
    fold_conf_matrices = []
    all_subject_preds = []

    for fold_idx in range(5):
        fold_name = f"fold_{fold_idx + 1}"
        train_ids = folds[fold_name]["train"]
        val_ids = folds[fold_name]["val"]
        test_ids = folds[fold_name]["test"]

        train_subjs = [os.path.join(base_dir, sid) for sid in train_ids]
        val_subjs   = [os.path.join(base_dir, sid) for sid in val_ids]
        test_subjs  = [os.path.join(base_dir, sid) for sid in test_ids]

        train_set = SubjectAudioDataset(train_subjs)
        val_set = SubjectAudioDataset(val_subjs)
        test_set = SubjectAudioDataset(test_subjs)

        # Fit scaler on training data ONLY
        print("Fitting scaler on train data...")
        train_data_for_scaler = extract_all_segments(train_set)
        scaler = StandardScaler()
        scaler.fit(train_data_for_scaler)

        train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=32, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=32, collate_fn=collate_fn)

        encoder = AudioTemporalBiLSTMEncoder(cnn_dim=128, lstm_dim=128).to(device)
        classifier = AudioFCClassifier(input_dim = 256).to(device)
        model = AudioModel(encoder, classifier).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for x, lengths, segment_to_subject, y in train_loader:
                x = normalize_batch(x, scaler)
                x, lengths = x.to(device), lengths.to(device)
                segment_to_subject = segment_to_subject.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                logits = model(x, lengths)  # (total_segments, num_classes)

                # Aggregate logits per subject (mean pooling)
                # segment_to_subject maps each segment to subject idx in batch
                num_subjects = y.size(0)
                # Prepare tensor to store subject logits sum and count
                subject_logits_sum = torch.zeros(num_subjects, logits.size(1), device=device)
                subject_counts = torch.zeros(num_subjects, device=device)

                subject_logits_sum.index_add_(0, segment_to_subject, logits)
                subject_counts.index_add_(0, segment_to_subject, torch.ones_like(segment_to_subject, dtype=torch.float))

                # Avoid division by zero (just in case)
                subject_counts = subject_counts.unsqueeze(1).clamp(min=1.0)
                subject_logits_avg = subject_logits_sum / subject_counts  # (num_subjects, num_classes)

                loss = criterion(subject_logits_avg, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_loss:.4f}")

        # Evaluation on test set
        model.eval()
        correct = 0
        total = 0
        fold_preds = []
        fold_labels = []

        with torch.no_grad():
            for x, lengths, segment_to_subject, y in test_loader:
                x = normalize_batch(x, scaler)
                x, lengths = x.to(device), lengths.to(device)
                segment_to_subject = segment_to_subject.to(device)
                y = y.to(device)

                logits = model(x, lengths)

                num_subjects = y.size(0)
                subject_logits_sum = torch.zeros(num_subjects, logits.size(1), device=device)
                subject_counts = torch.zeros(num_subjects, device=device)

                subject_logits_sum.index_add_(0, segment_to_subject, logits)
                subject_counts.index_add_(0, segment_to_subject, torch.ones_like(segment_to_subject, dtype=torch.float))

                subject_counts = subject_counts.unsqueeze(1).clamp(min=1.0)
                subject_logits_avg = subject_logits_sum / subject_counts

                preds = torch.argmax(subject_logits_avg, dim=1)

                fold_preds.extend(preds.cpu().numpy())
                fold_labels.extend(y.cpu().numpy())

                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        precision = precision_score(fold_labels, fold_preds, average='macro')
        recall = recall_score(fold_labels, fold_preds, average='macro')
        f1 = f1_score(fold_labels, fold_preds, average='macro')

        print(f"Fold {fold_idx + 1} Test Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        results.append((acc, precision, recall, f1))

        report = classification_report(fold_labels, fold_preds, output_dict=True, target_names=["Healthy", "Depressed"])
        conf_matrix = confusion_matrix(fold_labels, fold_preds)

        fold_reports.append(report)
        fold_conf_matrices.append(conf_matrix)

    # Final aggregated results
    results_arr = np.array(results)
    mean_acc, mean_prec, mean_rec, mean_f1 = results_arr.mean(axis=0)
    std_acc = results_arr[:, 0].std()
    std_prec = results_arr[:, 1].std()
    std_rec = results_arr[:, 2].std()
    std_f1 = results_arr[:, 3].std()

    print(f"\n5-Fold CV Results:")
    print(f"Mean Accuracy  = {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Mean Precision = {mean_prec:.4f} ± {std_prec:.4f}")
    print(f"Mean Recall    = {mean_rec:.4f} ± {std_rec:.4f}")
    print(f"Mean F1-Score  = {mean_f1:.4f} ± {std_f1:.4f}")

    # Optionally return reports/conf matrices for detailed analysis

if __name__ == "__main__":
    run_training()

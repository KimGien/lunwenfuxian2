import os
from glob import glob
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

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
# === Dataset ===
class SubjectAudioDataset(Dataset):
    def __init__(self, subject_dirs):
        self.subject_dirs = subject_dirs
        self.labels = [1 if os.path.basename(d).startswith("0201") else 0 for d in subject_dirs]

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        path = os.path.join(self.subject_dirs[idx], "raw_audio_features.npy")
        arr = np.load(path, allow_pickle=True)  # list of 29 arrays, each shape (N_i, 46)
        tensors = [torch.tensor(a, dtype=torch.float32) for a in arr]
        label = self.labels[idx]
        return tensors, label  # tensors: list of 29 tensors (N_i, 46)

# === Model components ===

class BiGRUAttentionEncoder(nn.Module):
    def __init__(self, input_dim=46, cnn_dim=128, gru_hidden=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(cnn_dim, gru_hidden, batch_first=True, bidirectional=True)

        # Attention layer parameters
        self.attention_fc = nn.Linear(gru_hidden * 2, 128)
        self.attention_score = nn.Linear(128, 1, bias=False)

    def forward(self, x, lengths):
        # x: (batch=29, seq_len, input_dim=46)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.cnn(x)        # (batch, cnn_dim, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_dim)

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # (batch, seq_len, gru_hidden*2)

        # Attention mechanism
        attn_weights = torch.tanh(self.attention_fc(out))  # (batch, seq_len, 128)
        attn_weights = self.attention_score(attn_weights).squeeze(-1)  # (batch, seq_len)

        # Mask padded positions (based on lengths)
        mask = torch.arange(out.size(1), device=lengths.device)[None, :] < lengths[:, None]  # (batch, seq_len)
        attn_weights[~mask] = float('-inf')  # mask padding positions

        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len)

        # Weighted sum of GRU outputs
        attended = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)  # (batch, gru_hidden*2)

        return attended  # embedding of size (batch, 512)

class DetectionLSTM(nn.Module):
    def __init__(self, input_dim=256, lstm_hidden= 1024, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len=29, input_dim=256)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last output
        logits = self.fc(out)
        return logits

class AudioModel(nn.Module):
    def __init__(self, encoder, detector):
        super().__init__()
        self.encoder = encoder
        self.detector = detector

    def forward(self, batch):
        """
        batch: list of length batch_size
            each element: list of 29 tensors (seq_len_i, 46)
        """
        device = next(self.parameters()).device
        batch_embeddings = []

        for subject_seqs in batch:
            lengths = torch.tensor([seq.shape[0] for seq in subject_seqs], device=device)
            padded_seqs = nn.utils.rnn.pad_sequence(subject_seqs, batch_first=True).to(device)  # (29, max_len, 46)
            emb = self.encoder(padded_seqs, lengths)  # (29, 256)
            batch_embeddings.append(emb)

        # Stack to (batch_size, 29, 256)
        batch_embeddings = torch.stack(batch_embeddings, dim=0)
        logits = self.detector(batch_embeddings)  # (batch_size, num_classes)
        return logits

# === Utilities ===

def collate_fn(batch):
    """
    batch: list of tuples (list_of_29_tensors, label)
    returns:
        batch_sequences: list of list_of_29_tensors  (no padding here)
        labels: tensor of shape (batch_size,)
    """
    sequences, labels = zip(*batch)
    return list(sequences), torch.tensor(labels)

def extract_all_segments(dataset):
    all_segments = []
    for i in range(len(dataset)):
        tensors, _ = dataset[i]
        for seg in tensors:
            all_segments.append(seg.numpy())
    all_data = np.concatenate(all_segments, axis=0)  # (total_segments, 46)
    return all_data

def normalize_batch(sequences, scaler):
    # sequences: list of list of tensors (batch_size, 29, seq_len_i, 46)
    norm_sequences = []
    for subject_seqs in sequences:
        norm_subject = []
        for seq in subject_seqs:
            arr = seq.numpy()
            arr_norm = scaler.transform(arr)
            norm_subject.append(torch.tensor(arr_norm, dtype=torch.float32))
        norm_sequences.append(norm_subject)
    return norm_sequences

# === Training loop ===
import json
base_dir = "split_dataset"

def run_training():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 60
    lr = 0.001
    batch_size = 8  # subjects per batch

    results = []

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

        print("Fitting scaler on training data...")
        train_data = extract_all_segments(train_set)
        scaler = StandardScaler()
        scaler.fit(train_data)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=collate_fn)

        encoder = BiGRUAttentionEncoder().to(device)
        detector = DetectionLSTM(input_dim = 512).to(device)
        model = AudioModel(encoder, detector).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for sequences, labels_batch in train_loader:
                # Normalize sequences
                sequences = normalize_batch(sequences, scaler)

                labels_batch = labels_batch.to(device)
                optimizer.zero_grad()

                logits = model(sequences)  # (batch, num_classes)

                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for sequences, labels_batch in test_loader:
                sequences = normalize_batch(sequences, scaler)
                labels_batch = labels_batch.to(device)

                logits = model(sequences)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        print(f"Fold {fold_idx+1} Test Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        results.append((acc, precision, recall, f1))

    # Print average results
    results_arr = np.array(results)
    print("\n5-Fold CV results (mean ± std):")
    print(f"Accuracy : {results_arr[:,0].mean():.4f} ± {results_arr[:,0].std():.4f}")
    print(f"Precision: {results_arr[:,1].mean():.4f} ± {results_arr[:,1].std():.4f}")
    print(f"Recall   : {results_arr[:,2].mean():.4f} ± {results_arr[:,2].std():.4f}")
    print(f"F1 Score : {results_arr[:,3].mean():.4f} ± {results_arr[:,3].std():.4f}")

if __name__ == "__main__":
    run_training()
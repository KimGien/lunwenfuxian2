"""DenseNet-121 baseline for multimodal depression detection.

Two-phase pipeline:
  1. Extract DenseNet-121 features (1024-dim) from EEG and audio spectrogram
     images and save as .npy files per subject.
  2. Train :class:`~lib.models.models.ConvPoolReLUClassifier` on the combined
     embeddings using 5-fold subject-level cross-validation.

Usage
-----
    # Phase 1 — feature extraction
    python scripts/baselines/baseline_densenet_121.py extract \\
        --source_dir data/split_dataset_june \\
        --target_dir data/split_dataset_densenet

    # Phase 2 — training & evaluation
    python scripts/baselines/baseline_densenet_121.py train \\
        --feat_dir data/split_dataset_densenet \\
        --fold_file data/split_dataset_june/fold_assignments.json
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from lib.baselines.datasets import SpectrogramEmbeddingDataset, collate_spectrogram
from lib.baselines.models import ConvPoolReLUClassifier


# ── Reproducibility ───────────────────────────────────────────────────────────


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: Integer seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Feature extraction ────────────────────────────────────────────────────────


def build_densenet(device: torch.device) -> tuple[nn.Module, transforms.Compose]:
    """Load DenseNet-121 pretrained on ImageNet (1024-dim features).

    Args:
        device: Torch device for inference.

    Returns:
        Tuple of (feature_extractor, preprocess_transform).
    """
    densenet = models.densenet121(pretrained=True).to(device)
    densenet.eval()
    feature_extractor = nn.Sequential(
        *list(densenet.features.children()),
        nn.AdaptiveAvgPool2d((1, 1)),
    ).to(device)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return feature_extractor, preprocess


def extract_image_feature(
    image_path: str,
    feature_extractor: nn.Module,
    preprocess: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    """Extract a 1024-dim DenseNet feature vector from a spectrogram image.

    Args:
        image_path: Path to a PNG spectrogram image.
        feature_extractor: DenseNet feature extractor (without classifier).
        preprocess: ImageNet preprocessing transform.
        device: Torch device for inference.

    Returns:
        1-D CPU tensor of shape ``(1024,)``.
    """
    image = Image.open(image_path).convert("RGB")
    inp = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_extractor(inp).view(-1)
    return feat.cpu()


def extract_all_features(
    source_dir: str,
    target_dir: str,
    device: torch.device,
) -> None:
    """Extract DenseNet-121 features for all subjects and save as .npy.

    Reads EEG and audio spectrogram images from ``eeg_stft_spectrogram2/``
    and ``audio_spectrogram/`` inside each subject directory.

    Args:
        source_dir: Root directory containing ``train/``, ``val/``, ``test/``.
        target_dir: Root directory where per-subject ``.npy`` files are written.
        device: Torch device for inference.
    """
    feature_extractor, preprocess = build_densenet(device)

    for split in ["train", "val", "test"]:
        split_path = os.path.join(source_dir, split)
        for subject_id in tqdm(sorted(os.listdir(split_path)), desc=f"Extracting {split}"):
            subject_path = os.path.join(split_path, subject_id)
            if not os.path.isdir(subject_path):
                continue

            eeg_dir = os.path.join(subject_path, "eeg_stft_spectrogram2")
            audio_dir = os.path.join(subject_path, "audio_spectrogram")

            eeg_files = sorted([os.path.join(eeg_dir, f) for f in os.listdir(eeg_dir) if f.endswith(".png")])
            audio_files = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(".png")])

            eeg_embs: list[torch.Tensor] = []
            for p in eeg_files:
                try:
                    eeg_embs.append(extract_image_feature(p, feature_extractor, preprocess, device))
                except Exception as e:
                    print(f"Skipping EEG image {p}: {e}")

            audio_embs: list[torch.Tensor] = []
            for p in audio_files:
                try:
                    audio_embs.append(extract_image_feature(p, feature_extractor, preprocess, device))
                except Exception as e:
                    print(f"Skipping audio image {p}: {e}")

            if not eeg_embs or not audio_embs:
                print(f"Skipping {subject_id} — insufficient embeddings.")
                continue

            eeg_stack = torch.stack(eeg_embs)
            audio_stack = torch.stack(audio_embs)

            eeg_len, audio_len = eeg_stack.size(0), audio_stack.size(0)
            if audio_len < eeg_len:
                audio_stack = torch.cat([audio_stack, torch.zeros(eeg_len - audio_len, audio_stack.size(1))], dim=0)
            elif eeg_len < audio_len:
                eeg_stack = torch.cat([eeg_stack, torch.zeros(audio_len - eeg_len, eeg_stack.size(1))], dim=0)

            save_dir = os.path.join(target_dir, split, subject_id)
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "eeg_embedding.npy"), eeg_stack.numpy())
            np.save(os.path.join(save_dir, "audio_embedding.npy"), audio_stack.numpy())


# ── Training ──────────────────────────────────────────────────────────────────


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 30,
) -> nn.Module:
    """Train with early-stopping on validation accuracy.

    Args:
        model: Classifier to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Parameter optimiser.
        device: Torch device.
        epochs: Maximum training epochs. Default 30.

    Returns:
        Model loaded with best-validation-accuracy weights.
    """
    best_state, best_val_acc = None, 0.0
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(dim=1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_cv(feat_dir: str, fold_file: str, device: torch.device) -> None:
    """Run 5-fold cross-validation and print results.

    Args:
        feat_dir: Root directory containing pre-extracted per-subject .npy files.
        fold_file: Path to ``fold_assignments.json``.
        device: Torch device for training.
    """
    with open(fold_file) as f:
        folds = json.load(f)

    subject_path_map: dict[str, str] = {}
    for split in ["train", "val", "test"]:
        split_path = os.path.join(feat_dir, split)
        if not os.path.isdir(split_path):
            continue
        for subj in os.listdir(split_path):
            subject_path_map[subj] = os.path.join(split_path, subj)

    results: list[tuple[float, float, float, float]] = []
    for fold_idx in range(5):
        fn = f"fold_{fold_idx + 1}"
        train_subjs = [subject_path_map[s] for s in folds[fn]["train"] if s in subject_path_map]
        val_subjs   = [subject_path_map[s] for s in folds[fn]["val"]   if s in subject_path_map]
        test_subjs  = [subject_path_map[s] for s in folds[fn]["test"]  if s in subject_path_map]

        train_loader = DataLoader(
            SpectrogramEmbeddingDataset(train_subjs),
            batch_size=16, shuffle=True, collate_fn=collate_spectrogram,
        )
        val_loader = DataLoader(
            SpectrogramEmbeddingDataset(val_subjs),
            batch_size=16, collate_fn=collate_spectrogram,
        )
        test_loader = DataLoader(
            SpectrogramEmbeddingDataset(test_subjs),
            batch_size=16, collate_fn=collate_spectrogram,
        )

        model = ConvPoolReLUClassifier(input_dim=2048).to(device)
        optimizer = optim.Adamax(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        trained = train_model(model, train_loader, val_loader, criterion, optimizer, device)
        trained.eval()

        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = trained(x).argmax(dim=1)
                fold_preds.extend(preds.cpu().numpy())
                fold_labels.extend(y.cpu().numpy())

        acc  = sum(p == l for p, l in zip(fold_preds, fold_labels)) / len(fold_labels)
        prec = precision_score(fold_labels, fold_preds, average="macro")
        rec  = recall_score(fold_labels, fold_preds, average="macro")
        f1   = f1_score(fold_labels, fold_preds, average="macro")
        print(f"Fold {fold_idx+1}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
        results.append((acc, prec, rec, f1))

    arr = np.array(results)
    print("\n5-Fold CV Summary:")
    for i, name in enumerate(["Accuracy", "Precision", "Recall", "F1"]):
        print(f"  {name}: {arr[:, i].mean():.4f} ± {arr[:, i].std():.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments and dispatch to extract or train phase."""
    parser = argparse.ArgumentParser(description="DenseNet-121 baseline for depression detection")
    sub = parser.add_subparsers(dest="phase", required=True)

    ext = sub.add_parser("extract", help="Extract DenseNet-121 features from spectrograms")
    ext.add_argument("--source_dir", default="data/split_dataset_june")
    ext.add_argument("--target_dir", default="data/split_dataset_densenet")
    ext.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    trn = sub.add_parser("train", help="Train classifier on pre-extracted features")
    trn.add_argument("--feat_dir", default="data/split_dataset_densenet")
    trn.add_argument("--fold_file", default="data/split_dataset_june/fold_assignments.json")
    trn.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    set_seed(42)
    device = torch.device(args.device)

    if args.phase == "extract":
        extract_all_features(args.source_dir, args.target_dir, device)
    else:
        run_cv(args.feat_dir, args.fold_file, device)


if __name__ == "__main__":
    main()

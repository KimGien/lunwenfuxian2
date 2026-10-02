"""Baseline datasets for the ViT and DenseNet-121 baselines."""

import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class SpectrogramEmbeddingDataset(Dataset):
    """Load pre-extracted spectrogram embeddings (ViT or DenseNet-121).

    Each subject directory must contain ``eeg_embedding.npy`` and
    ``audio_embedding.npy``. An optional ``eeg1d_embedding.npy`` (raw EEG
    channel encodings) is concatenated when present.

    Args:
        subject_dirs: Absolute paths to per-subject directories.
        include_eeg1d: Whether to also load ``eeg1d_embedding.npy``. Default False.
    """

    def __init__(
        self,
        subject_dirs: list[str],
        include_eeg1d: bool = False,
    ) -> None:
        self.subject_dirs = subject_dirs
        self.include_eeg1d = include_eeg1d

    def __len__(self) -> int:
        return len(self.subject_dirs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        subject_path = self.subject_dirs[idx]
        subject_id = os.path.basename(subject_path)

        eeg = np.load(os.path.join(subject_path, "eeg_embedding.npy"))
        audio = np.load(os.path.join(subject_path, "audio_embedding.npy"))
        parts = [eeg, audio]

        if self.include_eeg1d:
            eeg1d_path = os.path.join(subject_path, "eeg1d_embedding.npy")
            if os.path.exists(eeg1d_path):
                parts.append(np.load(eeg1d_path))

        combined = np.concatenate(parts, axis=1)
        label = 1 if subject_id.startswith("0201") else 0
        return (
            torch.tensor(combined, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def collate_spectrogram(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length spectrogram embedding sequences.

    Args:
        batch: List of (features, label) tuples.

    Returns:
        Tuple of (padded_features, labels) with shapes
        ``(batch, max_seq_len, feat_dim)`` and ``(batch,)``.
    """
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True, padding_value=0), labels

import os

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Load pre-extracted text embeddings for each subject.

    Parameters
    ----------
    subject_dirs : list[str]
        Absolute paths to subject directories.
    embedding_filename : str
        Name of the ``.npy`` file to load inside each subject directory
        (e.g. ``"text_embedding_macbert.npy"``).
    """

    def __init__(self, subject_dirs: list[str], embedding_filename: str):
        self.subject_dirs = subject_dirs
        self.embedding_filename = embedding_filename

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_path = self.subject_dirs[idx]
        subject_id = os.path.basename(subject_path)

        embedding = np.load(os.path.join(subject_path, self.embedding_filename))
        # MDD subjects start with '0201', healthy controls with '0202'/'0203'
        label = 1 if subject_id.startswith("0201") else 0

        embedding_t = torch.tensor(embedding, dtype=torch.float32)
        label_t = torch.tensor(label, dtype=torch.long)

        return embedding_t, label_t, subject_id


def collate_fn_with_ids(batch):
    """Pad variable-length embedding sequences and pass through subject IDs.

    Parameters
    ----------
    batch : list of (embedding, label, subject_id)

    Returns
    -------
    padded_seqs : Tensor (batch, max_seq_len, feat_dim)
    labels : Tensor (batch,)
    subject_ids : list[str]
    """
    sequences = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    subject_ids = [item[2] for item in batch]

    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)

    return padded_seqs, labels, subject_ids


class TrimodalDataset(Dataset):
    """Load pre-extracted embeddings for all three modalities per subject.

    Parameters
    ----------
    subject_dirs : list[str]
        Absolute paths to subject directories.
    eeg_file : str
        EEG embedding filename (e.g. ``"cbramod_embeddings_mumtaz.npy"``).
    speech_file : str
        Speech embedding filename (e.g. ``"audio_embedding_hubert_lstm.npy"``).
    text_file : str
        Text embedding filename (e.g. ``"text_embedding_macbert.npy"``).
    """

    def __init__(
        self,
        subject_dirs: list[str],
        eeg_file: str,
        speech_file: str,
        text_file: str,
    ):
        self.subject_dirs = subject_dirs
        self.eeg_file = eeg_file
        self.speech_file = speech_file
        self.text_file = text_file

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_path = self.subject_dirs[idx]
        subject_id = os.path.basename(subject_path)

        eeg = np.load(os.path.join(subject_path, self.eeg_file))
        speech = np.load(os.path.join(subject_path, self.speech_file))
        text = np.load(os.path.join(subject_path, self.text_file))
        label = 1 if subject_id.startswith("0201") else 0

        return (
            torch.tensor(eeg, dtype=torch.float32),
            torch.tensor(speech, dtype=torch.float32),
            torch.tensor(text, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            subject_id,
        )


def collate_trimodal(batch):
    """Collate for :class:`TrimodalDataset`.

    Returns
    -------
    eeg : Tensor (batch, eeg_seq, eeg_dim)
    speech : Tensor (batch, 29, speech_dim)
    text : Tensor (batch, 29, text_dim)
    labels : Tensor (batch,)
    subject_ids : list[str]
    """
    eeg = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
    speech = pad_sequence(
        [item[1] for item in batch], batch_first=True, padding_value=0
    )
    text = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([item[3] for item in batch], dtype=torch.long)
    subject_ids = [item[4] for item in batch]
    return eeg, speech, text, labels, subject_ids


class RawAudioDataset(Dataset):
    """Load raw audio features (pickle object array of variable-length segments).

    Each subject's ``.npy`` file is an object array of shape ``(29,)`` where
    each element is a 2-D array ``(N_i, 46)`` — one per recording.

    Parameters
    ----------
    subject_dirs : list[str]
        Absolute paths to subject directories.
    embedding_filename : str
        Name of the ``.npy`` file (default ``"raw_audio_features.npy"``).
    """

    def __init__(
        self,
        subject_dirs: list[str],
        embedding_filename: str = "raw_audio_features.npy",
    ):
        self.subject_dirs = subject_dirs
        self.embedding_filename = embedding_filename

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_path = self.subject_dirs[idx]
        subject_id = os.path.basename(subject_path)

        arr = np.load(
            os.path.join(subject_path, self.embedding_filename), allow_pickle=True
        )
        # arr: object array of shape (29,), each element (N_i, 46)
        segments = [torch.tensor(a, dtype=torch.float32) for a in arr]
        label = 1 if subject_id.startswith("0201") else 0

        return segments, torch.tensor(label, dtype=torch.long), subject_id


def collate_raw_audio(batch):
    """Collate for :class:`RawAudioDataset` — keeps segment lists intact.

    Returns
    -------
    segments_list : list[list[Tensor]]
        One list of 29 tensors per subject.
    labels : Tensor (batch,)
    subject_ids : list[str]
    """
    segments_list = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    subject_ids = [item[2] for item in batch]
    return segments_list, labels, subject_ids

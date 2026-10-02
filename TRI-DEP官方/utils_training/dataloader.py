# utils_training/dataloader.py
import os
import torch
from torch.utils.data import Dataset
import numpy as np

class UnimodalDataset(Dataset):
    def __init__(self, subject_dirs, embedding_filename):
        """
        Args:
            subject_dirs (list[str]): Paths to subject directories
            embedding_filename (str): Name of the embedding file (e.g., 'audio_embedding_gru.npy')
        """
        self.subject_dirs = subject_dirs
        self.embedding_filename = embedding_filename
        self.labels = [
            1 if os.path.basename(d).startswith("0201") else 0
            for d in subject_dirs
        ]

    def __len__(self):
        return len(self.subject_dirs)

    def __getitem__(self, idx):
        subject_dir = self.subject_dirs[idx]
        file_path = os.path.join(subject_dir, self.embedding_filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embedding file not found: {file_path}")

        file_embeddings = np.load(file_path)  # (29, dim)
        file_tensor = torch.tensor(file_embeddings, dtype=torch.float)
        label = self.labels[idx]
        return file_tensor, label, subject_dir

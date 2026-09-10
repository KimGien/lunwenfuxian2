"""Chinese BERT-base text encoder for transcription embeddings.

Encodes per-subject transcription CSVs using ``bert-base-chinese`` and
saves CLS-token embeddings as ``text_embedding_bert.npy`` inside each
subject directory.
"""

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def encode_texts_bert(base_dir: str, save_dir: str) -> None:
    """Encode all subjects' transcriptions with Chinese BERT-base.

    For each subject directory the function reads
    ``<subject_id>/<subject_id>_transcription.csv``, encodes the text
    column (index 1) in batches of 32 using the CLS token, and saves the
    result as ``<save_dir>/<subject_id>/text_embedding_bert.npy`` with
    shape ``(n_texts, 768)``.

    Args:
        base_dir: Root directory whose sub-directories are per-subject
            folders containing ``<subject_id>_transcription.csv``.
        save_dir: Root directory where per-subject ``.npy`` files are
            written.  May be the same as *base_dir*.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese").to(device)
    model.eval()

    for subject_id in tqdm(sorted(os.listdir(base_dir)), desc="Processing subjects"):
        subject_path = os.path.join(base_dir, subject_id)
        if not os.path.isdir(subject_path):
            continue

        csv_path = os.path.join(subject_path, f"{subject_id}_transcription.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        texts: list[str] = df.iloc[:, 1].astype(str).tolist()

        embeddings: list[np.ndarray] = []
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            inputs = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # (B, 768)
                embeddings.append(cls_emb.cpu().numpy())

        subject_save_dir = os.path.join(save_dir, subject_id)
        os.makedirs(subject_save_dir, exist_ok=True)
        np.save(
            os.path.join(subject_save_dir, "text_embedding_bert.npy"),
            np.vstack(embeddings),
        )

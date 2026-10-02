"""MacBERT text encoder for Chinese transcription embeddings.

Encodes per-subject transcription CSVs using
``hfl/chinese-macbert-base`` and saves CLS-token embeddings as
``text_embedding_macbert.npy`` inside each subject directory.
"""

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


def encode_texts_macbert(base_dir: str, save_dir: str) -> None:
    """Encode all subjects' transcriptions with Chinese MacBERT.

    For each subject directory in *base_dir* the function:

    1. Finds the first ``.csv`` file and reads text from column 2.
    2. Encodes text in batches of 32 using the MacBERT CLS token.
    3. Saves the result as
       ``<save_dir>/<subject_id>/text_embedding_macbert.npy``
       with shape ``(n_texts, 768)``.

    Args:
        base_dir: Root directory whose sub-directories are per-subject
            folders containing a transcription CSV.
        save_dir: Root directory where per-subject ``.npy`` files are
            written.  May be the same as *base_dir*.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "hfl/chinese-macbert-base"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    model.eval()

    for subject_id in tqdm(sorted(os.listdir(base_dir)), desc="Processing subjects"):
        subject_path = os.path.join(base_dir, subject_id)
        if not os.path.isdir(subject_path):
            continue

        csv_files = [f for f in os.listdir(subject_path) if f.endswith(".csv")]
        if not csv_files:
            continue

        df = pd.read_csv(os.path.join(subject_path, csv_files[0]))
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
            os.path.join(subject_save_dir, "text_embedding_macbert.npy"),
            np.vstack(embeddings),
        )

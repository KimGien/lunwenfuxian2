"""Chinese XLNet text encoder for transcription embeddings.

Encodes per-subject transcription CSVs using ``hfl/chinese-xlnet-base``
and saves the last-token hidden state as ``text_embedding_xlnet.npy``
inside each subject directory.

XLNet uses permutation language modelling; the last non-padding token's
hidden state is used as the sentence representation.
"""

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import XLNetModel, XLNetTokenizer


def encode_texts_xlnet(base_dir: str, save_dir: str) -> None:
    """Encode all subjects' transcriptions with Chinese XLNet.

    For each subject directory in *base_dir* the function finds the first
    ``.csv`` file, reads text from column 2, encodes it in batches of 16
    using the last non-padding token's hidden state, and saves the result
    as ``<save_dir>/<subject_id>/text_embedding_xlnet.npy`` with shape
    ``(n_texts, hidden_dim)``.

    Args:
        base_dir: Root directory whose sub-directories are per-subject
            folders containing a transcription CSV.
        save_dir: Root directory where per-subject ``.npy`` files are
            written.  May be the same as *base_dir*.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "hfl/chinese-xlnet-base"
    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    model = XLNetModel.from_pretrained(model_name).to(device)
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
        for i in range(0, len(texts), 16):
            batch = texts[i : i + 16]
            inputs = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                # Use the last non-padding token as the sentence representation
                last_token_indices = inputs["attention_mask"].sum(dim=1) - 1
                last_hidden = outputs.last_hidden_state  # (B, seq_len, hidden_dim)
                batch_emb = last_hidden[
                    range(last_hidden.size(0)), last_token_indices
                ]  # (B, hidden_dim)
                embeddings.append(batch_emb.cpu().numpy())

        subject_save_dir = os.path.join(save_dir, subject_id)
        os.makedirs(subject_save_dir, exist_ok=True)
        np.save(
            os.path.join(subject_save_dir, "text_embedding_xlnet.npy"),
            np.vstack(embeddings),
        )

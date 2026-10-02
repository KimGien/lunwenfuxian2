"""Multilingual MPNet text encoder for transcription embeddings.

Encodes per-subject transcription CSVs using
``paraphrase-multilingual-mpnet-base-v2`` (SentenceTransformers) and
saves sentence embeddings as ``text_embedding_mpnet.npy`` inside each
subject directory.
"""

import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def encode_texts_mpnet(base_dir: str, save_dir: str) -> None:
    """Encode all subjects' transcriptions with multilingual MPNet.

    For each subject directory in *base_dir* the function finds the first
    ``.csv`` file, reads text from column 2, encodes it with
    ``paraphrase-multilingual-mpnet-base-v2``, and saves the result as
    ``<save_dir>/<subject_id>/text_embedding_mpnet.npy`` with shape
    ``(n_texts, 768)``.

    Args:
        base_dir: Root directory whose sub-directories are per-subject
            folders containing a transcription CSV.
        save_dir: Root directory where per-subject ``.npy`` files are
            written.  May be the same as *base_dir*.
    """
    text_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    for subject_id in tqdm(sorted(os.listdir(base_dir)), desc="Processing subjects"):
        subject_path = os.path.join(base_dir, subject_id)
        if not os.path.isdir(subject_path):
            continue

        csv_files = [f for f in os.listdir(subject_path) if f.endswith(".csv")]
        if not csv_files:
            continue

        df = pd.read_csv(os.path.join(subject_path, csv_files[0]))
        texts: list[str] = df.iloc[:, 1].astype(str).tolist()
        embeddings: np.ndarray = text_model.encode(texts)  # (n_texts, 768)

        subject_save_dir = os.path.join(save_dir, subject_id)
        os.makedirs(subject_save_dir, exist_ok=True)
        np.save(
            os.path.join(subject_save_dir, "text_embedding_mpnet.npy"),
            embeddings,
        )

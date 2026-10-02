import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def encode_texts_mpnet(base_dir, save_dir):
    text_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    subject_ids = os.listdir(base_dir)
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        subject_path = os.path.join(base_dir, subject_id)
        if not os.path.isdir(subject_path):
            continue

        csv_files = [f for f in os.listdir(subject_path) if f.endswith(".csv")]
        if not csv_files:
            continue

        csv_path = os.path.join(subject_path, csv_files[0])
        df = pd.read_csv(csv_path)

        # Use second column as Chinese text
        texts = df.iloc[:, 1].astype(str).tolist()

        embeddings = text_model.encode(texts)  # (n_texts, 768)

        save_subject_dir = os.path.join(save_dir, subject_id)
        os.makedirs(save_subject_dir, exist_ok=True)
        np.save(os.path.join(save_subject_dir, "text_embedding_mpnet.npy"), embeddings)

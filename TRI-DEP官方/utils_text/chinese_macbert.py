import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel

def encode_texts_macbert(base_dir, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "hfl/chinese-macbert-base"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    model.eval()

    subject_ids = os.listdir(base_dir)
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        subject_path = os.path.join(base_dir, subject_id)
        if not os.path.isdir(subject_path):
            continue

        # Find the first CSV file under the subject folder
        csv_files = [f for f in os.listdir(subject_path) if f.endswith(".csv")]
        if not csv_files:
            continue

        csv_path = os.path.join(subject_path, csv_files[0])
        df = pd.read_csv(csv_path)
        texts = df.iloc[:, 1].astype(str).tolist()

        batch_size = 32
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
                embeddings.append(cls_embeddings.cpu().numpy())

        embeddings = np.vstack(embeddings)  # shape (num_texts, 768)

        save_subject_dir = os.path.join(save_dir, subject_id)
        os.makedirs(save_subject_dir, exist_ok=True)
        save_path = os.path.join(save_subject_dir, "text_embedding_macbert.npy")
        np.save(save_path, embeddings)

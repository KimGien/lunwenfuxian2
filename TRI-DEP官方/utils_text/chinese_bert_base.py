import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel

def encode_texts_bert(base_dir, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertModel.from_pretrained("bert-base-chinese").to(device)
    model.eval()

    subject_ids = os.listdir(base_dir)
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        subject_path = os.path.join(base_dir, subject_id)
        if not os.path.isdir(subject_path):
            continue

        csv_path = os.path.join(subject_path, f"{subject_id}_transcription.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        texts = df.iloc[:, 1].astype(str).tolist()

        batch_size = 32
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.cpu().numpy())

        embeddings = np.vstack(embeddings)

        save_subject_dir = os.path.join(save_dir, subject_id)
        os.makedirs(save_subject_dir, exist_ok=True)
        np.save(os.path.join(save_subject_dir, "text_embedding_bert.npy"), embeddings)

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import XLNetTokenizer, XLNetModel

def encode_texts_xlnet(base_dir, save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "hfl/chinese-xlnet-base"
    tokenizer = XLNetTokenizer.from_pretrained(model_name)
    model = XLNetModel.from_pretrained(model_name).to(device)
    model.eval()

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
        texts = df.iloc[:, 1].astype(str).tolist()

        batch_size = 16
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                last_token_indices = inputs['attention_mask'].sum(dim=1) - 1
                last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
                batch_embeddings = last_hidden[range(last_hidden.size(0)), last_token_indices]  # (batch_size, hidden_dim)
                embeddings.append(batch_embeddings.cpu().numpy())

        embeddings = np.vstack(embeddings)  # (num_texts, hidden_dim)

        save_subject_dir = os.path.join(save_dir, subject_id)
        os.makedirs(save_subject_dir, exist_ok=True)
        np.save(os.path.join(save_subject_dir, "text_embedding_xlnet.npy"), embeddings)

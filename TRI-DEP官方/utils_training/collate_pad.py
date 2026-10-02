import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn_padd(batch):
    """
    batch: list of tuples (x, y, subject_id)
    x: tensor with shape (seq_len, feat_dim)
    y: int label
    subject_id: string
    """
    sequences = [item[0] for item in batch]  # list of (seq_len, feat_dim) tensors
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    subject_ids = [item[2] for item in batch]

    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)

    return padded_seqs, labels, subject_ids

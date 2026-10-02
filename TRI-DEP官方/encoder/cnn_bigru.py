import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRUAttentionEncoder(nn.Module):
    def __init__(self, input_dim=40, cnn_dim=128, rnn_hidden=256):  # increased to 256
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim),
            nn.ReLU()
        )
        self.bigru = nn.GRU(
            input_size=cnn_dim,
            hidden_size=rnn_hidden,  # 256
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention = nn.Linear(2 * rnn_hidden, 1)  # 512 -> 1

    def forward(self, x):
        # x: (batch, seq_len, input_dim) or (N_segments, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        x = x.transpose(1, 2)        # (batch, input_dim, seq_len)
        x = self.cnn(x)              # (batch, cnn_dim, seq_len)
        x = x.transpose(1, 2)        # (batch, seq_len, cnn_dim)

        rnn_out, _ = self.bigru(x)   # (batch, seq_len, 512)

        attn_weights = torch.softmax(self.attention(rnn_out), dim=1)  # (batch, seq_len, 1)
        attended = torch.sum(attn_weights * rnn_out, dim=1)           # (batch, 512)

        return attended  # (batch, 512)

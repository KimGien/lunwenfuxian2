import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUAttentionClassifier(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=128, num_layers=2, num_classes=2, dropout=0.4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

        # Attention parameters: learnable query vector for attention
        self.attention_vector = nn.Parameter(torch.randn(hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim)

        # Compute attention scores:
        # Shape of attention_vector: (hidden_dim,)
        # We expand it to (batch, seq_len, hidden_dim) for dot product
        # Compute scores by dot product between hidden states and attention_vector
        # scores shape: (batch, seq_len)
        scores = torch.tanh(gru_out) @ self.attention_vector
        attention_weights = F.softmax(scores, dim=1).unsqueeze(2)  # (batch, seq_len, 1)

        # Compute weighted sum of gru_out by attention weights
        context = torch.sum(gru_out * attention_weights, dim=1)  # (batch, hidden_dim)

        context = self.dropout(context)
        out = self.classifier(context)  # (batch, num_classes)
        return out

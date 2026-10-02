import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: (batch, seq_len=29, input_dim=768)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        # Use last time step output (or mean over time)
        out = lstm_out[:, -1, :]  # (batch, hidden_dim*2)
        return self.classifier(out)

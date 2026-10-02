import torch
import torch.nn as nn

class LSTM1Classifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, num_classes=2, dropout=0.3):
        super(LSTM1Classifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False  # Normal LSTM
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),  # No *2 here
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        out = lstm_out[:, -1, :]    # Last time step output
        return self.classifier(out)

import torch
import torch.nn as nn

class CNNGRUEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super(CNNGRUEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        """
        x: Tensor of shape (N_segments, input_dim), e.g. (29, 768)
        returns: Tensor of shape (1, hidden_dim), e.g. (1, 256)
        """
        # Reshape for Conv1d: (batch=1, channels=input_dim, sequence_length=N_segments)
        x = x.transpose(0, 1).unsqueeze(0)  # shape: (1, input_dim, N_segments)
        x = self.cnn(x)                     # shape: (1, hidden_dim, N_segments)
        x = x.transpose(1, 2)               # shape: (1, N_segments, hidden_dim)

        # GRU
        x, _ = self.gru(x)                  # shape: (1, N_segments, hidden_dim)

        # Temporal average pooling
        x = x.mean(dim=1)                   # shape: (1, hidden_dim)

        return x

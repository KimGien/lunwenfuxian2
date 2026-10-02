import torch.nn as nn

class AudioCNNEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super(AudioCNNEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Reduces over time dimension
        )
        self.output_dim = hidden_dim

    def forward(self, x):  # x: (N_segments, input_dim)
        self.eval()
        x = x.transpose(1, 0).unsqueeze(0)  # (1, input_dim, N_segments)
        x = self.cnn(x)                     # (1, hidden_dim, 1)
        return x.squeeze(-1)                # (1, hidden_dim)

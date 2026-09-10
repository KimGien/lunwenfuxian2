"""CNN + LSTM encoder for audio segment sequences.

Treats all segments of one recording as a single 1-D sequence, applies
a CNN for local feature extraction, feeds the result into an LSTM, and
returns a time-averaged embedding.
"""

import torch
import torch.nn as nn


class CNNLSTMEncoder(nn.Module):
    """CNN → LSTM → temporal mean pooling encoder.

    Architecture:

    1. ``Conv1d(input_dim, hidden_dim) → BatchNorm1d → ReLU``
    2. ``LSTM(hidden_dim, hidden_dim)``
    3. Mean pooling across the time dimension.

    Args:
        input_dim: Feature dimensionality of each input segment.
            Default 1024 (XLSR-53 output).
        hidden_dim: CNN output channels and LSTM hidden size.  Default 256.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of segment embeddings into one vector.

        Args:
            x: Tensor of shape ``(N_segments, input_dim)``.

        Returns:
            Tensor of shape ``(1, hidden_dim)``.
        """
        self.eval()
        x = x.transpose(1, 0).unsqueeze(0)   # (1, input_dim, N_segments)
        x = self.cnn(x)                       # (1, hidden_dim, N_segments)
        x = x.squeeze(0).transpose(0, 1)      # (N_segments, hidden_dim)
        x, _ = self.lstm(x.unsqueeze(0))      # (1, N_segments, hidden_dim)
        return x.mean(dim=1)                  # (1, hidden_dim)

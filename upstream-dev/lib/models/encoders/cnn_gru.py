"""CNN + GRU encoder for audio segment sequences.

Treats all segments of one recording as a single sequence, applies a
1-D CNN for local feature extraction, passes the result through a GRU,
and returns a time-averaged embedding.
"""

import torch
import torch.nn as nn


class CNNGRUEncoder(nn.Module):
    """CNN → GRU → temporal mean pooling encoder.

    Architecture:

    1. ``Conv1d(input_dim, hidden_dim) → BatchNorm1d → ReLU``
    2. ``GRU(hidden_dim, hidden_dim)``
    3. Mean pooling across the time dimension.

    Args:
        input_dim: Feature dimensionality of each input segment.
            Default 768 (Chinese BERT / MPNet).
        hidden_dim: CNN output channels and GRU hidden size.  Default 256.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of segment embeddings into one vector.

        Args:
            x: Tensor of shape ``(N_segments, input_dim)``, e.g.
                ``(29, 768)`` for one recording.

        Returns:
            Tensor of shape ``(1, hidden_dim)``, e.g. ``(1, 256)``.
        """
        x = x.transpose(0, 1).unsqueeze(0)  # (1, input_dim, N_segments)
        x = self.cnn(x)                     # (1, hidden_dim, N_segments)
        x = x.transpose(1, 2)               # (1, N_segments, hidden_dim)
        x, _ = self.gru(x)                  # (1, N_segments, hidden_dim)
        x = x.mean(dim=1)                   # (1, hidden_dim)
        return x

"""CNN encoder for audio segment embeddings.

Treats all segments of one recording as a single batch, applies a
1-D convolution over the time (segment) axis, and returns a single
fixed-size embedding via adaptive average pooling.
"""

import torch
import torch.nn as nn


class AudioCNNEncoder(nn.Module):
    """Single-layer CNN that collapses a variable-length segment sequence.

    Architecture: ``Conv1d → BatchNorm1d → ReLU → AdaptiveAvgPool1d(1)``.

    The encoder treats the sequence of segment embeddings as a 1-D signal
    and pools it to a fixed-size representation regardless of how many
    segments are present.

    Args:
        input_dim: Dimensionality of each input segment embedding.
            Default 1024 (XLSR-53 output).
        hidden_dim: Number of CNN output channels / encoder output
            dimensionality.  Default 256.

    Attributes:
        output_dim: Equal to *hidden_dim*; exposed for downstream use.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.output_dim: int = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a sequence of segment embeddings into one vector.

        Args:
            x: Tensor of shape ``(N_segments, input_dim)``.

        Returns:
            Tensor of shape ``(1, hidden_dim)``.
        """
        self.eval()
        x = x.transpose(1, 0).unsqueeze(0)  # (1, input_dim, N_segments)
        x = self.cnn(x)                     # (1, hidden_dim, 1)
        return x.squeeze(-1)                # (1, hidden_dim)

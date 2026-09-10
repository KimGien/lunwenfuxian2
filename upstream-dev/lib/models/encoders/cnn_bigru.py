"""CNN + Bidirectional GRU encoder with attention for audio segments.

Applies a 1-D CNN to extract local features from each segment, feeds the
result through a bidirectional GRU to capture temporal context, and
produces a single embedding via soft attention pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGRUAttentionEncoder(nn.Module):
    """CNN → BiGRU → attention pooling encoder.

    Architecture:

    1. ``Conv1d(input_dim, cnn_dim) → BatchNorm1d → ReLU``
    2. ``Bidirectional GRU(cnn_dim, rnn_hidden)``
    3. Soft-attention pooling over time steps (learned linear scorer).

    Args:
        input_dim: Feature dimensionality of each input segment.
            Default 40.
        cnn_dim: Number of CNN output channels.  Default 128.
        rnn_hidden: Hidden size of each GRU direction; the final output
            has dimension ``2 * rnn_hidden``.  Default 256.
    """

    def __init__(
        self,
        input_dim: int = 40,
        cnn_dim: int = 128,
        rnn_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim),
            nn.ReLU(),
        )
        self.bigru = nn.GRU(
            input_size=cnn_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(2 * rnn_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of segment sequences into fixed-size embeddings.

        Args:
            x: Tensor of shape ``(batch, seq_len, input_dim)`` or
                ``(N_segments, input_dim)`` for a single recording.

        Returns:
            Tensor of shape ``(batch, 2 * rnn_hidden)`` — one attended
            embedding per sequence.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)            # (batch=N, 1, input_dim)

        x = x.transpose(1, 2)            # (batch, input_dim, seq_len)
        x = self.cnn(x)                  # (batch, cnn_dim, seq_len)
        x = x.transpose(1, 2)            # (batch, seq_len, cnn_dim)

        rnn_out, _ = self.bigru(x)       # (batch, seq_len, 2*rnn_hidden)

        attn_weights = torch.softmax(
            self.attention(rnn_out), dim=1
        )                                # (batch, seq_len, 1)
        attended = torch.sum(attn_weights * rnn_out, dim=1)  # (batch, 2*rnn_hidden)

        return attended

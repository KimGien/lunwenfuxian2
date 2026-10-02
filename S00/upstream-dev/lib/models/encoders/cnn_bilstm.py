"""CNN + Bidirectional LSTM encoder for variable-length audio segments.

Applies a 1-D CNN for local feature extraction, then a BiLSTM with
pack/pad handling for variable-length sequences, and mean-pools over
the valid time steps to produce a fixed-size embedding.
"""

import torch
import torch.nn as nn


class AudioTemporalBiLSTMEncoder(nn.Module):
    """CNN → BiLSTM → length-masked mean pooling encoder.

    Architecture:

    1. ``Conv1d(input_dim, cnn_dim) → BatchNorm1d → ReLU``
    2. ``Bidirectional LSTM(cnn_dim, lstm_dim)`` with
       :func:`~torch.nn.utils.rnn.pack_padded_sequence` support.
    3. Mean pooling over valid (unpadded) time steps.

    Args:
        input_dim: Feature dimensionality of each input segment.
            Default 46 (handcrafted speech features).
        cnn_dim: Number of CNN output channels.  Default 128.
        lstm_dim: Hidden size of each LSTM direction; the output has
            dimension ``2 * lstm_dim``.  Default 256.
    """

    def __init__(
        self,
        input_dim: int = 46,
        cnn_dim: int = 128,
        lstm_dim: int = 256,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(cnn_dim, lstm_dim, batch_first=True, bidirectional=True)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode a padded batch of segment sequences.

        Args:
            x: Padded tensor of shape ``(batch, seq_len, input_dim)``.
            lengths: 1-D integer tensor of valid sequence lengths per
                sample, shape ``(batch,)``.

        Returns:
            Tensor of shape ``(batch, 2 * lstm_dim)`` — one embedding per
            sequence, computed by averaging over the valid time steps.
        """
        x = x.transpose(1, 2)   # (batch, input_dim, seq_len)
        x = self.cnn(x)
        x = x.transpose(1, 2)   # (batch, seq_len, cnn_dim)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Mean pooling over valid lengths
        out = out.sum(dim=1) / lengths.unsqueeze(1).to(out.device)  # (batch, 2*lstm_dim)
        return out

"""Baseline model architectures for the ViT and DenseNet-121 baselines."""

import torch
import torch.nn as nn


class EEG1DEncoder(nn.Module):
    """Lightweight CNN-BiLSTM encoder for raw single-channel EEG time series.

    Encodes a single EEG channel ``(1, 1, T)`` into a fixed-size embedding
    via two Conv1d + MaxPool1d blocks followed by a bidirectional LSTM and a
    linear projection.

    Args:
        hidden_dim: LSTM hidden size (each direction). Default 128.
        embed_dim: Output embedding dimensionality. Default 512.
    """

    def __init__(self, hidden_dim: int = 128, embed_dim: int = 512) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single EEG channel.

        Args:
            x: Tensor of shape ``(1, 1, T)`` — one channel time series.

        Returns:
            Tensor of shape ``(embed_dim,)``.
        """
        x = self.cnn(x)                      # (1, 128, T')
        x = x.permute(0, 2, 1)              # (1, T', 128)
        lstm_out, _ = self.lstm(x)           # (1, T', hidden*2)
        pooled = lstm_out.mean(dim=1)        # (1, hidden*2)
        return self.proj(pooled).squeeze(0)  # (embed_dim,)


class ConvPoolReLUClassifier(nn.Module):
    """1-D convolutional classifier with global average pooling.

    Applies a single Conv1d layer across the time/sequence axis, global
    average pools to a fixed-size vector, then passes through two FC layers.

    Args:
        input_dim: Feature dimensionality of the input sequence. Default 2048.
        hidden_dim: FC hidden layer size. Default 1024.
        num_classes: Number of output classes. Default 2.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 1024,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(batch, seq_len, input_dim)``.

        Returns:
            Logits tensor of shape ``(batch, num_classes)``.
        """
        x = x.transpose(1, 2)          # (batch, input_dim, seq_len)
        x = self.relu(self.conv1(x))   # (batch, 256, seq_len)
        x = torch.mean(x, dim=2)       # (batch, 256) — global avg pool
        x = self.relu(self.fc1(x))
        return self.fc2(x)

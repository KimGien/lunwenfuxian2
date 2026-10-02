import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioTemporalBiLSTMEncoder(nn.Module):
    def __init__(self, input_dim=46, cnn_dim=128, lstm_dim=256):
        super(AudioTemporalBiLSTMEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(cnn_dim, lstm_dim, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_dim)

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bilstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Mean pooling over valid lengths
        out = out.sum(dim=1) / lengths.unsqueeze(1).to(out.device)  # (batch, 2 * lstm_dim)
        return out  # (batch, 2*lstm_dim)

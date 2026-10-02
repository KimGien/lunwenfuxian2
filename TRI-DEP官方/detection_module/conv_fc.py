import torch
import torch.nn as nn

class ConvPoolClassifier(nn.Module):
    def __init__(self, input_dim=768, num_classes=2, dropout=0.3):
        super(ConvPoolClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # reduces seq_len roughly by half
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # Calculate flattened feature size after conv+pool layers
        # Original seq_len=29; after two maxpools with kernel 2 -> floor division by 2 twice
        conv_out_len = 29 // 2 // 2  # = 7 (approximate)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * conv_out_len, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len) for Conv1d
        x = self.conv(x)       # (batch, 128, conv_out_len)
        x = self.classifier(x)
        return x

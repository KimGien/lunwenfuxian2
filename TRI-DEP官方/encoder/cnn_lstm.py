import torch.nn as nn

class CNNLSTMEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256):
        super(CNNLSTMEncoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        self.eval()
        x = x.transpose(1, 0).unsqueeze(0)
        x = self.cnn(x)
        x = x.squeeze(0).transpose(0, 1)
        x, _ = self.lstm(x.unsqueeze(0))
        return x.mean(dim=1)

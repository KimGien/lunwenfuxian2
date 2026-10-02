import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRUAttentionClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, num_classes=2, dropout=0.3):
        super(BiGRUAttentionClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = True
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=self.bidirectional
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)  
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):  
        gru_out, _ = self.gru(x) 

        # Attention mechanism
        attn_weights = self.attention(gru_out)  
        attn_weights = F.softmax(attn_weights, dim=1)  
        context = torch.sum(attn_weights * gru_out, dim=1) 

        return self.classifier(context)

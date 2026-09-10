import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM1Classifier(nn.Module):
    """Unidirectional LSTM followed by a two-layer MLP classifier.

    Takes the last time-step hidden state as the sequence representation.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        out = lstm_out[:, -1, :]  # last time step
        return self.classifier(out)


class ConvPoolClassifier(nn.Module):
    """Two-layer 1-D CNN with max-pooling, followed by an MLP classifier.

    Expects a fixed maximum sequence length of 29 (the number of interview
    recordings per subject).  Two ``MaxPool1d(kernel_size=2)`` layers reduce
    ``29 -> 14 -> 7``, so the flattened feature size is ``128 * 7 = 896``.
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_dim, out_channels=256, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        conv_out_len = 29 // 2 // 2  # = 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * conv_out_len, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len) for Conv1d
        x = self.conv(x)  # (batch, 128, conv_out_len)
        return self.classifier(x)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM with LayerNorm and mean+max pooling classifier.

    Used for speech modality experiments (MFCC, HuBERT, XLSR).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.layernorm = nn.LayerNorm(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        lstm_out = self.layernorm(lstm_out)
        avg_pool = torch.mean(lstm_out, dim=1)  # (batch, hidden_dim*2)
        max_pool, _ = torch.max(lstm_out, dim=1)  # (batch, hidden_dim*2)
        pooled = torch.cat((avg_pool, max_pool), dim=1)  # (batch, hidden_dim*4)
        return self.classifier(pooled)


class SpeechConvPoolClassifier(nn.Module):
    """Two-layer 1-D CNN with max-pooling for speech embeddings.

    Unlike the text ``ConvPoolClassifier``, the first conv layer output
    channels are controlled by ``hidden_dim`` and the second layer uses
    ``hidden_dim // 2``.
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 2,
        dropout: float = 0.4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        conv_out_len = 29 // 2 // 2  # = 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear((hidden_dim // 2) * conv_out_len, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        return self.classifier(self.conv(x))


# ── EEG pretrained embedding models ─────────────────────────────────────────


class EEGLSTMClassifier(nn.Module):
    """Unidirectional LSTM for pretrained EEG embeddings (CBraMod / LaBraM).

    Takes the last time-step hidden state as the sequence representation.
    """

    def __init__(
        self,
        input_dim: int = 200,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :])


class EEGConvPoolClassifier(nn.Module):
    """Two-layer 1-D CNN with max-pooling for pretrained EEG embeddings.

    Unlike the text ``ConvPoolClassifier``, ``seq_len`` is parameterizable
    because different EEG encoders produce different segment counts
    (CBraMod: 75, LaBraM: 30, Mumtaz: 60).
    """

    def __init__(
        self,
        input_dim: int = 200,
        seq_len: int = 75,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        conv_out_len = seq_len // 2 // 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * conv_out_len, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        return self.classifier(self.conv(x))


# ── EEG modality models ─────────────────────────────────────────────────────


class GRUAttentionClassifier(nn.Module):
    """Two-layer GRU with learnable attention pooling."""

    def __init__(
        self,
        input_dim: int = 200,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.attention_vector = nn.Parameter(torch.randn(hidden_dim))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden_dim)
        scores = torch.tanh(gru_out) @ self.attention_vector  # (batch, seq_len)
        attention_weights = F.softmax(scores, dim=1).unsqueeze(2)  # (batch, seq_len, 1)
        context = torch.sum(gru_out * attention_weights, dim=1)  # (batch, hidden_dim)
        context = self.dropout(context)
        return self.classifier(context)


# ── Speech prosody+MFCC end-to-end models ────────────────────────────────


class BiGRUAttentionEncoder(nn.Module):
    """CNN + Bidirectional GRU + Attention encoder for variable-length audio segments.

    Encodes a batch of segments ``(batch, seq_len, 46)`` into fixed-size
    embeddings of dimension ``gru_hidden * 2``.
    """

    def __init__(self, input_dim: int = 46, cnn_dim: int = 128, gru_hidden: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(cnn_dim, gru_hidden, batch_first=True, bidirectional=True)
        self.attention_fc = nn.Linear(gru_hidden * 2, 128)
        self.attention_score = nn.Linear(128, 1, bias=False)

    def forward(self, x, lengths):
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.cnn(x)  # (batch, cnn_dim, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, cnn_dim)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        attn_weights = torch.tanh(self.attention_fc(out))
        attn_weights = self.attention_score(attn_weights).squeeze(-1)

        mask = (
            torch.arange(out.size(1), device=lengths.device)[None, :] < lengths[:, None]
        )
        attn_weights[~mask] = float("-inf")
        attn_weights = F.softmax(attn_weights, dim=1)

        attended = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)
        return attended  # (batch, gru_hidden*2)


class DetectionLSTM(nn.Module):
    """LSTM classifier that takes stacked per-recording embeddings."""

    def __init__(
        self, input_dim: int = 512, hidden_dim: int = 1024, num_classes: int = 2
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, 29, input_dim)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class ProsodyMFCCModel(nn.Module):
    """End-to-end prosody+MFCC model: BiGRUAttentionEncoder + DetectionLSTM.

    Accepts a list of segment lists (one per subject), encodes each subject's
    29 recordings independently, stacks them, and classifies.
    """

    def __init__(
        self,
        input_dim: int = 46,
        cnn_dim: int = 128,
        gru_hidden: int = 256,
        lstm_hidden: int = 1024,
        num_classes: int = 2,
    ):
        super().__init__()
        self.encoder = BiGRUAttentionEncoder(input_dim, cnn_dim, gru_hidden)
        self.detector = DetectionLSTM(gru_hidden * 2, lstm_hidden, num_classes)

    def forward(self, batch_segments):
        """
        Parameters
        ----------
        batch_segments : list[list[Tensor]]
            Outer list: subjects in batch. Inner list: 29 tensors of shape
            ``(N_i, 46)`` (variable-length recordings).

        Returns
        -------
        logits : Tensor (batch, num_classes)
        """
        device = next(self.parameters()).device
        batch_embeddings = []

        for subject_seqs in batch_segments:
            lengths = torch.tensor(
                [seq.shape[0] for seq in subject_seqs], device=device
            )
            padded = nn.utils.rnn.pad_sequence(subject_seqs, batch_first=True).to(
                device
            )  # (29, max_len, 46)
            emb = self.encoder(padded, lengths)  # (29, gru_hidden*2)
            batch_embeddings.append(emb)

        batch_embeddings = torch.stack(batch_embeddings, dim=0)  # (batch, 29, emb_dim)
        return self.detector(batch_embeddings)


# ── EEG handcrafted features model ──────────────────────────────────────


# ── Intermediate Fusion models ───────────────────────────────────────────
#
# Load pretrained unimodal models, freeze their encoders, extract 128-dim
# representations from each, and train only a lightweight fusion head.


class IntermediateFusionConcat(nn.Module):
    """Intermediate fusion: frozen pretrained encoders + trainable fusion head.

    Loads the three best unimodal models (EEGConvPoolClassifier,
    SpeechConvPoolClassifier, LSTM1Classifier), freezes all their weights,
    extracts the 128-dim representation from each (output of penultimate
    layer in their classifiers), and trains a small MLP on the concatenated
    384-dim vector.
    """

    def __init__(
        self,
        eeg_model: nn.Module,
        speech_model: nn.Module,
        text_model: nn.Module,
        proj_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.eeg_model = eeg_model
        self.speech_model = speech_model
        self.text_model = text_model

        # Freeze all pretrained weights
        for model in [self.eeg_model, self.speech_model, self.text_model]:
            for param in model.parameters():
                param.requires_grad = False

        # Fusion classifier (only trainable part)
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def _extract_eeg(self, x):
        """Run EEGConvPoolClassifier up to 128-dim representation."""
        x = x.transpose(1, 2)
        x = self.eeg_model.conv(x)
        # classifier: Flatten, Dropout, Linear(1920,128), ReLU, Dropout, Linear(128,2)
        x = self.eeg_model.classifier[0](x)  # Flatten
        x = self.eeg_model.classifier[1](x)  # Dropout
        x = self.eeg_model.classifier[2](x)  # Linear -> 128
        x = self.eeg_model.classifier[3](x)  # ReLU
        return x  # (B, 128) — stop before final Dropout + Linear(128,2)

    def _extract_speech(self, x):
        """Run SpeechConvPoolClassifier up to 128-dim representation."""
        x = x.transpose(1, 2)
        x = self.speech_model.conv(x)
        x = self.speech_model.classifier[0](x)  # Flatten
        x = self.speech_model.classifier[1](x)  # Dropout
        x = self.speech_model.classifier[2](x)  # Linear -> 128
        x = self.speech_model.classifier[3](x)  # ReLU
        return x  # (B, 128)

    def _extract_text(self, x):
        """Run LSTM1Classifier up to 128-dim representation."""
        lstm_out, _ = self.text_model.lstm(x)
        out = lstm_out[:, -1, :]
        # classifier: Linear(1024,128), ReLU, Dropout, Linear(128,2)
        x = self.text_model.classifier[0](out)  # Linear -> 128
        x = self.text_model.classifier[1](x)  # ReLU
        return x  # (B, 128) — stop before Dropout + Linear(128,2)

    def forward(self, eeg, speech, text):
        with torch.no_grad():
            eeg_h = self._extract_eeg(eeg)
            speech_h = self._extract_speech(speech)
            text_h = self._extract_text(text)
        fused = torch.cat([eeg_h, speech_h, text_h], dim=1)  # (B, 384)
        return self.classifier(fused)


class IntermediateFusionGated(nn.Module):
    """Intermediate fusion with learned gating over frozen encoder outputs.

    Same frozen encoders as IntermediateFusionConcat, but uses a learned
    gating mechanism that weights each modality's contribution dynamically
    per sample, before the final classifier.
    """

    def __init__(
        self,
        eeg_model: nn.Module,
        speech_model: nn.Module,
        text_model: nn.Module,
        proj_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.eeg_model = eeg_model
        self.speech_model = speech_model
        self.text_model = text_model

        for model in [self.eeg_model, self.speech_model, self.text_model]:
            for param in model.parameters():
                param.requires_grad = False

        # Gating network: takes concat of 3 modalities, outputs 3 weights
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 3),
        )
        # Classifier on the gated representation
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def _extract_eeg(self, x):
        x = x.transpose(1, 2)
        x = self.eeg_model.conv(x)
        x = self.eeg_model.classifier[0](x)
        x = self.eeg_model.classifier[1](x)
        x = self.eeg_model.classifier[2](x)
        x = self.eeg_model.classifier[3](x)
        return x

    def _extract_speech(self, x):
        x = x.transpose(1, 2)
        x = self.speech_model.conv(x)
        x = self.speech_model.classifier[0](x)
        x = self.speech_model.classifier[1](x)
        x = self.speech_model.classifier[2](x)
        x = self.speech_model.classifier[3](x)
        return x

    def _extract_text(self, x):
        lstm_out, _ = self.text_model.lstm(x)
        out = lstm_out[:, -1, :]
        x = self.text_model.classifier[0](out)
        x = self.text_model.classifier[1](x)
        return x

    def forward(self, eeg, speech, text):
        with torch.no_grad():
            eeg_h = self._extract_eeg(eeg)  # (B, 128)
            speech_h = self._extract_speech(speech)  # (B, 128)
            text_h = self._extract_text(text)  # (B, 128)

        concat = torch.cat([eeg_h, speech_h, text_h], dim=1)  # (B, 384)
        weights = F.softmax(self.gate(concat), dim=1)  # (B, 3)

        # Weighted sum of modality representations
        stacked = torch.stack([eeg_h, speech_h, text_h], dim=1)  # (B, 3, 128)
        gated = (stacked * weights.unsqueeze(2)).sum(dim=1)  # (B, 128)
        return self.classifier(gated)


# ── Early Fusion models ──────────────────────────────────────────────────
#
# These reuse the exact per-modality encoder architectures that achieved
# the best unimodal results, but replace the three independent classification
# heads with a single shared fusion classifier.
#
# Best unimodal encoders:
#   EEG   : EEGConvPoolClassifier  (Conv1d→MaxPool→Conv1d→MaxPool→Flatten)
#           Input (B, 60, 200) → representation 128*15 = 1920-dim
#   Speech: SpeechConvPoolClassifier (Conv1d→MaxPool→Conv1d→MaxPool→Flatten)
#           Input (B, 29, 512) → representation (1024//2)*7 = 3584-dim
#   Text  : LSTM1Classifier (2-layer LSTM, last time-step)
#           Input (B, 29, 768) → representation 1024-dim
#
# Each encoder's internal representation is projected to 128-dim (matching
# the original classifiers' first FC layer), then the three 128-dim vectors
# are concatenated (384-dim) and fed to a shared fusion head.


class EarlyFusionConcat(nn.Module):
    """Early fusion with proven per-modality encoders + concat + MLP.

    Reuses the exact encoder architectures from the best unimodal models:
    - EEG: 2-layer Conv1d + MaxPool (from EEGConvPoolClassifier)
    - Speech: 2-layer Conv1d + MaxPool (from SpeechConvPoolClassifier)
    - Text: 2-layer LSTM, last time-step (from LSTM1Classifier)

    Each encoder produces a 128-dim representation; the three are
    concatenated (384-dim) and classified by a shared MLP head.
    """

    def __init__(
        self,
        eeg_dim: int = 200,
        eeg_seq_len: int = 60,
        speech_dim: int = 512,
        speech_hidden: int = 1024,
        text_dim: int = 768,
        text_hidden: int = 1024,
        proj_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── EEG encoder (same as EEGConvPoolClassifier) ──
        self.eeg_conv = nn.Sequential(
            nn.Conv1d(eeg_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        eeg_flat_dim = 128 * (eeg_seq_len // 2 // 2)  # 128 * 15 = 1920
        self.eeg_proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(eeg_flat_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Speech encoder (same as SpeechConvPoolClassifier) ──
        self.speech_conv = nn.Sequential(
            nn.Conv1d(speech_dim, speech_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(speech_hidden, speech_hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        speech_flat_dim = (speech_hidden // 2) * (29 // 2 // 2)  # 512 * 7 = 3584
        self.speech_proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(speech_flat_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Text encoder (same as LSTM1Classifier) ──
        self.text_lstm = nn.LSTM(
            input_size=text_dim,
            hidden_size=text_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Shared fusion classifier ──
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 3, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_classes),
        )

    def forward(self, eeg, speech, text):
        # EEG: (B, 60, 200)
        eeg_h = self.eeg_proj(self.eeg_conv(eeg.transpose(1, 2)))  # (B, 128)
        # Speech: (B, 29, 512)
        speech_h = self.speech_proj(
            self.speech_conv(speech.transpose(1, 2))
        )  # (B, 128)
        # Text: (B, 29, 768)
        text_out, _ = self.text_lstm(text)
        text_h = self.text_proj(text_out[:, -1, :])  # (B, 128)
        # Fuse and classify
        fused = torch.cat([eeg_h, speech_h, text_h], dim=1)  # (B, 384)
        return self.classifier(fused)


class EarlyFusionBottleneck(nn.Module):
    """Bottleneck fusion: proven encoders + cross-modal Transformer.

    Same per-modality encoders as ``EarlyFusionConcat``, but instead of
    simple concatenation, the three 128-dim modality tokens are processed
    by a small Transformer encoder (cross-modal self-attention) before
    mean-pooling and classification.  This allows the model to learn
    cross-modal interactions.
    """

    def __init__(
        self,
        eeg_dim: int = 200,
        eeg_seq_len: int = 60,
        speech_dim: int = 512,
        speech_hidden: int = 1024,
        text_dim: int = 768,
        text_hidden: int = 1024,
        proj_dim: int = 128,
        nhead: int = 4,
        num_tf_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── EEG encoder (same as EEGConvPoolClassifier) ──
        self.eeg_conv = nn.Sequential(
            nn.Conv1d(eeg_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        eeg_flat_dim = 128 * (eeg_seq_len // 2 // 2)
        self.eeg_proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(eeg_flat_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Speech encoder (same as SpeechConvPoolClassifier) ──
        self.speech_conv = nn.Sequential(
            nn.Conv1d(speech_dim, speech_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(speech_hidden, speech_hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        speech_flat_dim = (speech_hidden // 2) * (29 // 2 // 2)
        self.speech_proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(speech_flat_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Text encoder (same as LSTM1Classifier) ──
        self.text_lstm = nn.LSTM(
            input_size=text_dim,
            hidden_size=text_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Cross-modal Transformer ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=nhead,
            dim_feedforward=proj_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_tf_layers
        )

        # ── Shared fusion classifier ──
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, eeg, speech, text):
        # EEG: (B, 60, 200)
        eeg_h = self.eeg_proj(self.eeg_conv(eeg.transpose(1, 2)))  # (B, proj_dim)
        # Speech: (B, 29, 512)
        speech_h = self.speech_proj(
            self.speech_conv(speech.transpose(1, 2))
        )  # (B, proj_dim)
        # Text: (B, 29, 768)
        text_out, _ = self.text_lstm(text)
        text_h = self.text_proj(text_out[:, -1, :])  # (B, proj_dim)
        # Cross-modal attention over 3 modality tokens
        tokens = torch.stack([eeg_h, speech_h, text_h], dim=1)  # (B, 3, proj_dim)
        tokens = self.transformer(tokens)  # (B, 3, proj_dim)
        pooled = tokens.mean(dim=1)  # (B, proj_dim)
        return self.classifier(pooled)


class EEGCNNLSTMClassifier(nn.Module):
    """CNN + unidirectional LSTM for EEG handcrafted features.

    Input shape: (batch, num_channels, num_features) — e.g. (B, 29, 10).
    Conv1d operates on the channel dimension, then LSTM on the resulting
    temporal sequence, taking the last time-step output.
    """

    def __init__(self, num_channels: int = 29, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=32, hidden_size=64, batch_first=True, bidirectional=False
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, num_channels, num_features) e.g. (B, 29, 10)
        x = self.relu(self.bn1(self.conv1(x)))  # (B, 32, num_features)
        x = x.permute(0, 2, 1)  # (B, num_features, 32)
        lstm_out, _ = self.lstm(x)  # (B, num_features, 64)
        x = lstm_out[:, -1, :]  # last time-step
        return self.fc(x)


class EEGCNNFCClassifier(nn.Module):
    """CNN + AdaptiveAvgPool + FC for EEG handcrafted features (Conv-based).

    Input shape: (batch, num_channels, num_features) — e.g. (B, 29, 10).
    """

    def __init__(self, num_channels: int = 29, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, 29, 10)
        x = self.relu(self.bn1(self.conv1(x)))  # (B, 64, 10)
        x = self.pool(x).squeeze(-1)  # (B, 64)
        return self.fc(x)


class EEGCNNGRUAttentionClassifier(nn.Module):
    """CNN + GRU + Attention for EEG handcrafted features (GRU-based).

    Input shape: (batch, num_channels, num_features) — e.g. (B, 29, 10).
    """

    def __init__(self, num_channels: int = 29, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(
            input_size=32, hidden_size=64, batch_first=True, bidirectional=False
        )
        self.attention = nn.Linear(64, 1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, 29, 10)
        x = self.relu(self.bn1(self.conv1(x)))  # (B, 32, 10)
        x = x.permute(0, 2, 1)  # (B, 10, 32)
        gru_out, _ = self.gru(x)  # (B, 10, 64)
        weights = F.softmax(self.attention(gru_out).squeeze(-1), dim=1)  # (B, 10)
        context = (gru_out * weights.unsqueeze(-1)).sum(dim=1)  # (B, 64)
        return self.fc(context)

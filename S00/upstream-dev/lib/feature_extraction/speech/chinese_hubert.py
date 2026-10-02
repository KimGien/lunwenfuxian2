"""Chinese HuBERT feature extractor for speech embeddings.

Wraps ``xmj2002/hubert-base-ch-speech-emotion-recognition`` to produce
frame-level hidden-state embeddings from raw audio files.
"""

import torch
import torchaudio
import numpy as np
from transformers import (
    AutoConfig,
    HubertModel,
    HubertPreTrainedModel,
    Wav2Vec2FeatureExtractor,
)


class HubertChineseExtractor:
    """Frame-level speech embedder using Chinese HuBERT.

    Loads the ``xmj2002/hubert-base-ch-speech-emotion-recognition`` model
    and returns the last hidden states of the HuBERT encoder as a NumPy
    array of shape ``(T, hidden_size)``.

    Args:
        model_name: HuggingFace model identifier.  Defaults to the
            Chinese speech-emotion HuBERT checkpoint.
        device: Torch device for inference.  Defaults to CUDA if available,
            otherwise CPU.
    """

    def __init__(
        self,
        model_name: str = "xmj2002/hubert-base-ch-speech-emotion-recognition",
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.config = AutoConfig.from_pretrained(model_name)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        class _HubertEmbedder(HubertPreTrainedModel):
            """Thin wrapper exposing HuBERT last hidden states."""

            def __init__(self, config):
                super().__init__(config)
                self.hubert = HubertModel(config)
                self.init_weights()

            def forward(self, input_values: torch.Tensor) -> torch.Tensor:
                outputs = self.hubert(input_values)
                return outputs.last_hidden_state  # (B, T, hidden_size)

        self.model = _HubertEmbedder.from_pretrained(model_name, config=self.config)
        self.model.to(self.device).eval()

    def extract_embedding(
        self,
        wav_path: str,
        max_len_sec: int = 5,
    ) -> np.ndarray:
        """Extract frame-level HuBERT embeddings from an audio file.

        Loads the waveform, resamples to 16 kHz if necessary, converts
        to mono, truncates to *max_len_sec* seconds, and returns the
        last hidden states.

        Args:
            wav_path: Path to a ``.wav`` audio file.
            max_len_sec: Maximum audio length in seconds before truncation.
                Default 5.

        Returns:
            NumPy array of shape ``(T, hidden_size)`` where *T* is the
            number of HuBERT output frames.
        """
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.shape[1] < 1000:
            print(f"[Warning] {wav_path} seems too short.")

        input_values = self.processor(
            waveform.squeeze(0),
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=16000 * max_len_sec,
        ).input_values.to(self.device)

        with torch.no_grad():
            emb = self.model(input_values)  # (1, T, hidden_size)

        return emb.squeeze(0).cpu().numpy()  # (T, hidden_size)

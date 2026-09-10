"""XLSR-53 feature extractor for Chinese speech embeddings.

Wraps ``jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`` to produce
frame-level hidden-state embeddings of shape ``(T, 1024)`` from raw
audio files.
"""

import os

import numpy as np
import torch
import torchaudio
from transformers import AutoModel, AutoProcessor


class XSLR53FeatureExtractor:
    """Frame-level speech embedder using XLSR-53 (Chinese fine-tune).

    Loads ``jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`` and
    returns the last hidden states of the wav2vec2 encoder.

    Args:
        device: Torch device for inference.  Defaults to CUDA if
            available, otherwise CPU.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.processor = AutoProcessor.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
        )
        self.model = AutoModel.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
        ).to(self.device)
        self.model.eval()

    def extract_embedding(
        self,
        wav_path: str,
        max_len_frames: int | None = None,
    ) -> torch.Tensor:
        """Extract frame-level XLSR-53 embeddings from an audio file.

        Args:
            wav_path: Path to a ``.wav`` audio file.
            max_len_frames: If set, truncate the output to this many
                frames along the time axis.

        Returns:
            CPU tensor of shape ``(T, 1024)`` (or ``(max_len_frames, 1024)``
            if truncation is applied).
        """
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=16000
            )
            waveform = resampler(waveform)

        input_values = self.processor(
            waveform.squeeze(0),
            sampling_rate=16000,
            return_tensors="pt",
        ).input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            embeddings = outputs.last_hidden_state.squeeze(0)  # (T, 1024)

        if max_len_frames is not None:
            embeddings = embeddings[:max_len_frames, :]
        return embeddings.cpu()


def extract_embeddings_from_dir(
    audio_dir: str,
    subject_id: str | None = None,
    encoder: torch.nn.Module | None = None,
) -> np.ndarray:
    """Extract XLSR-53 embeddings for all ``.wav`` files in a directory.

    Uses a module-level :class:`XSLR53FeatureExtractor` instance.

    Args:
        audio_dir: Directory containing ``.wav`` files.
        subject_id: Unused; present for interface compatibility.
        encoder: Optional encoder module applied after extraction; expected
            to accept a batched tensor and return a tensor.

    Returns:
        Concatenated embeddings of shape ``(sum_T, 1024)``, or an empty
        array of shape ``(0, 1024)`` if no ``.wav`` files are found.
    """
    all_embeddings: list[np.ndarray] = []
    for fname in sorted(os.listdir(audio_dir)):
        if fname.endswith(".wav"):
            wav_path = os.path.join(audio_dir, fname)
            emb = _extractor.extract_embedding(wav_path)  # (T, 1024)
            if encoder is not None:
                with torch.no_grad():
                    emb = encoder(emb.unsqueeze(0)).squeeze(0)
            all_embeddings.append(emb.cpu().numpy())

    if not all_embeddings:
        return np.empty((0, 1024))
    return np.concatenate(all_embeddings, axis=0)


# Module-level extractor instance reused by extract_embeddings_from_dir
_extractor = XSLR53FeatureExtractor()

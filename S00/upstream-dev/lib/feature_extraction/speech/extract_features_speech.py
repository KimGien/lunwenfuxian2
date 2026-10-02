"""Speech feature extraction pipeline for the TRI-DEP dataset.

Extracts per-subject speech embeddings by:

1. Loading a pretrained feature extractor (XLSR-53, MFCC, or Chinese HuBERT)
   to obtain frame-level features from each audio segment.
2. Mean-pooling across time to produce a fixed-size segment embedding.
3. Passing all segment embeddings through a trainable encoder (CNN-BiGRU,
   CNN-LSTM, etc.) and mean-pooling across segments to produce one
   ``encoder_output_dim``-dimensional vector per recording.
4. Concatenating per-recording embeddings into a single subject-level array
   and saving it as a ``.npy`` file.

Usage
-----
    python lib/feature_extraction/speech/extract_features_speech.py \\
        --base_dir data/split_dataset_june \\
        --feature_extractor xslr53 \\
        --encoder cnn_bigru \\
        --input_dim 1024 \\
        --encoder_output_dim 256
"""

import os
import argparse
import importlib

import numpy as np
import torch
from tqdm import tqdm


# ── Feature-extractor factory ────────────────────────────────────────────────


def get_feature_extractor(name: str):
    """Instantiate a speech feature extractor by name.

    Args:
        name: One of ``"xslr53"``, ``"mfcc"``, or ``"chinese_hubert"``.

    Returns:
        A feature extractor instance with an ``extract_embedding(wav_path)``
        method that returns an array of shape ``(T, feat_dim)``.

    Raises:
        ValueError: If *name* is not one of the supported extractors.
    """
    if name == "xslr53":
        from lib.feature_extraction.speech.xslr_53 import XSLR53FeatureExtractor
        return XSLR53FeatureExtractor()
    elif name == "mfcc":
        from lib.feature_extraction.speech.mfccs import MFCCFeatureExtractor
        return MFCCFeatureExtractor()
    elif name == "chinese_hubert":
        from lib.feature_extraction.speech.chinese_hubert import HubertChineseExtractor
        return HubertChineseExtractor()
    else:
        raise ValueError(f"Unknown feature extractor: {name}")


# ── Encoder factory ──────────────────────────────────────────────────────────


def get_encoder(name: str, input_dim: int) -> torch.nn.Module:
    """Instantiate an audio encoder by name.

    Encoder class names are resolved via ``lib.models.encoders.<name>``.

    Args:
        name: One of ``"cnn_bigru"``, ``"cnn_lstm"``, ``"cnn_gru"``,
            ``"cnn"``, or ``"cnn_bilstm"``.
        input_dim: Feature dimensionality passed to the encoder constructor.

    Returns:
        An uninitialized (random-weight) encoder ``torch.nn.Module``.

    Raises:
        ValueError: If *name* is not a recognised encoder identifier.
    """
    class_map: dict[str, str] = {
        "cnn_bigru": "BiGRUAttentionEncoder",
        "cnn_lstm": "CNNLSTMEncoder",
        "cnn_gru": "CNNGRUEncoder",
        "cnn": "AudioCNNEncoder",
        "cnn_bilstm": "AudioTemporalBiLSTMEncoder",
    }
    if name not in class_map:
        raise ValueError(f"Unknown encoder: {name}")

    module = importlib.import_module(f"lib.models.encoders.{name}")
    EncoderClass = getattr(module, class_map[name])
    return EncoderClass(input_dim=input_dim)


# ── Per-recording processing ─────────────────────────────────────────────────


def process_recording(
    segment_dir: str,
    pattern: str,
    extractor,
    encoder: torch.nn.Module,
    device: torch.device,
    encoder_output_dim: int,
) -> torch.Tensor:
    """Extract a single-recording embedding from its audio segments.

    For each ``.wav`` segment whose filename starts with *pattern*, the
    function extracts frame-level features, mean-pools across time, stacks
    the results, runs the encoder (no gradient), and mean-pools across
    segments.

    Args:
        segment_dir: Directory containing ``.wav`` segment files.
        pattern: Filename prefix used to identify segments for this
            recording (e.g. ``"01_part"``).
        extractor: Feature extractor instance (e.g. XLSR-53).
        encoder: Trained or randomly initialised encoder module.
        device: Torch device on which encoder inference is performed.
        encoder_output_dim: Dimensionality of the encoder output; used to
            build a zero-vector fallback when no segments are found.

    Returns:
        Tensor of shape ``(1, encoder_output_dim)`` on CPU.
    """
    segment_files = sorted([
        f for f in os.listdir(segment_dir)
        if f.startswith(pattern) and f.endswith(".wav")
    ])
    segment_embeddings: list[torch.Tensor] = []

    for fname in segment_files:
        try:
            wav_path = os.path.join(segment_dir, fname)
            feat = extractor.extract_embedding(wav_path)          # (T, feat_dim)
            feat_tensor = torch.tensor(feat, dtype=torch.float32)
            feat_mean = torch.mean(feat_tensor, dim=0)            # (feat_dim,)
            segment_embeddings.append(feat_mean)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    if not segment_embeddings:
        return torch.zeros(1, encoder_output_dim)

    segments_tensor = torch.stack(segment_embeddings, dim=0).to(device)

    with torch.no_grad():
        encoded_segments = encoder(segments_tensor)               # (N, encoder_output_dim)

    audio_embedding = torch.mean(encoded_segments, dim=0, keepdim=True)  # (1, encoder_output_dim)
    return audio_embedding.cpu()


# ── Per-subject processing ────────────────────────────────────────────────────


def process_subject(
    subject_path: str,
    extractor,
    encoder: torch.nn.Module,
    device: torch.device,
    encoder_output_dim: int,
) -> np.ndarray:
    """Extract embeddings for all 29 recordings of one subject.

    Recordings are identified by the prefixes ``01`` through ``29``; their
    segments live in ``<subject_path>/segmented_audio/``.

    Args:
        subject_path: Root directory of a single subject.
        extractor: Initialised feature extractor.
        encoder: Encoder module (eval mode expected by caller).
        device: Torch device for encoder inference.
        encoder_output_dim: Encoder output dimensionality.

    Returns:
        NumPy array of shape ``(29, encoder_output_dim)``.
    """
    segment_dir = os.path.join(subject_path, "segmented_audio")
    audio_file_embeddings: list[torch.Tensor] = []

    for i in range(1, 30):
        prefix = f"{i:02d}"
        pattern = f"{prefix}_part"
        audio_embedding = process_recording(
            segment_dir, pattern, extractor, encoder, device, encoder_output_dim
        )
        audio_file_embeddings.append(audio_embedding)

    subject_embedding = torch.cat(audio_file_embeddings, dim=0)
    return subject_embedding.numpy()


# ── Batch processing ─────────────────────────────────────────────────────────


def process_all_subjects(
    base_dir: str,
    extractor,
    encoder: torch.nn.Module,
    device: torch.device,
    encoder_output_dim: int,
) -> None:
    """Extract and save speech embeddings for every subject in *base_dir*.

    For each subject directory the result is saved as::

        <base_dir>/<subject_id>/audio_<feature_extractor_name>_encoded_<encoder_name>.npy

    The file name is constructed from the module-level globals
    ``feature_extractor_name`` and ``encoder_name`` set in ``__main__``.

    Args:
        base_dir: Root directory whose immediate sub-directories are
            per-subject folders.
        extractor: Initialised feature extractor.
        encoder: Encoder module (must be in eval mode).
        device: Torch device for encoder inference.
        encoder_output_dim: Encoder output dimensionality.
    """
    for subject_id in tqdm(sorted(os.listdir(base_dir)), desc="Processing subjects"):
        subject_path = os.path.join(base_dir, subject_id)
        if os.path.isdir(subject_path):
            try:
                feats = process_subject(
                    subject_path, extractor, encoder, device, encoder_output_dim
                )
                save_path = os.path.join(
                    subject_path,
                    f"audio_{feature_extractor_name}_encoded_{encoder_name}.npy",
                )
                np.save(save_path, feats)
                print(f"Succeeded on {subject_id}, features shape: {feats.shape}")
            except Exception as e:
                print(f"Failed on {subject_id}: {e}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio embedding extraction")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="split_dataset",
        help="Base directory containing subject folders",
    )
    parser.add_argument(
        "--feature_extractor",
        type=str,
        default="xslr53",
        choices=["xslr53", "mfcc", "chinese_hubert"],
        help="Feature extractor to use",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="cnn_bigru",
        choices=["cnn_bigru", "cnn_lstm", "cnn_gru", "cnn_bilstm", "cnn"],
        help="Encoder architecture",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        default=1024,
        help="Input dimension for encoder",
    )
    parser.add_argument(
        "--rnn_hidden",
        type=int,
        default=256,
        help=(
            "Hidden size passed to the encoder constructor. "
            "Note: bidirectional encoders (e.g. cnn_bigru, cnn_bilstm) produce "
            "2 * rnn_hidden output dimensions. The actual output dim is inferred "
            "automatically."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use ('cuda' or 'cpu')",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # Module-level names used when constructing the output filename
    feature_extractor_name: str = args.feature_extractor
    encoder_name: str = args.encoder

    print(f"Loading feature extractor: {feature_extractor_name}")
    extractor = get_feature_extractor(feature_extractor_name)

    print(f"Loading encoder: {encoder_name} with input_dim={args.input_dim}")
    encoder = get_encoder(encoder_name, input_dim=args.input_dim).to(device)
    encoder.eval()

    # Infer actual output dim with a dummy forward pass — handles bidirectional
    # encoders (e.g. cnn_bigru outputs 2 * rnn_hidden, not rnn_hidden)
    with torch.no_grad():
        dummy = torch.zeros(1, args.input_dim).to(device)
        actual_output_dim: int = encoder(dummy).shape[-1]
    print(f"Encoder actual output dim: {actual_output_dim}")

    process_all_subjects(args.base_dir, extractor, encoder, device, actual_output_dim)

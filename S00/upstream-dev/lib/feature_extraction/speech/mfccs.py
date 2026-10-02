"""MFCC feature extractor for speech segments.

Provides :class:`MFCCFeatureExtractor` for frame-level MFCC extraction
compatible with the speech feature extraction pipeline, and helper
functions for per-recording and per-subject processing.
"""

import os

import librosa
import numpy as np
from tqdm import tqdm


class MFCCFeatureExtractor:
    """Extract MFCC features from audio files.

    Produces a 2-D array of shape ``(T, n_mfcc)`` from a single ``.wav``
    file, compatible with the segment-embedding interface expected by
    :mod:`lib.feature_extraction.speech.extract_features_speech`.

    Args:
        sr_target: Target sample rate in Hz.  Default 16 000.
        n_mfcc: Number of MFCC coefficients to compute.  Default 40.
    """

    def __init__(self, sr_target: int = 16000, n_mfcc: int = 40) -> None:
        self.sr_target = sr_target
        self.n_mfcc = n_mfcc

    def extract_embedding(self, wav_path: str) -> np.ndarray:
        """Extract MFCC features from an audio file.

        Args:
            wav_path: Path to a ``.wav`` audio segment.

        Returns:
            Array of shape ``(T, n_mfcc)`` where *T* is the number of
            MFCC time frames.
        """
        y, sr = librosa.load(wav_path, sr=self.sr_target)
        # librosa returns (n_mfcc, T); transpose to (T, n_mfcc)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc).T
        return mfcc


def process_recording(
    segment_dir: str,
    pattern: str,
    extractor: MFCCFeatureExtractor,
) -> np.ndarray:
    """Extract MFCCs for all segments of one recording.

    Args:
        segment_dir: Directory containing ``.wav`` segment files.
        pattern: Substring used to filter segment filenames.
        extractor: Initialised :class:`MFCCFeatureExtractor`.

    Returns:
        Array of shape ``(N_segments, n_mfcc)``, or ``(1, n_mfcc)``
        of zeros if no segments are found.
    """
    segment_files = sorted(
        [f for f in os.listdir(segment_dir) if pattern in f]
    )
    feats: list[np.ndarray] = []
    for fname in segment_files:
        try:
            feat = extractor.extract_embedding(os.path.join(segment_dir, fname))
            feats.append(feat)
        except Exception as e:
            print(f"Error in {fname}: {e}")

    if not feats:
        return np.zeros((1, extractor.n_mfcc))
    return np.stack(feats)  # (N_segments, n_mfcc)


def process_subject(
    subject_path: str,
    extractor: MFCCFeatureExtractor,
) -> list[np.ndarray]:
    """Extract MFCCs for all 29 recordings of one subject.

    Args:
        subject_path: Root directory of a single subject.
        extractor: Initialised :class:`MFCCFeatureExtractor`.

    Returns:
        List of 29 arrays, where element *i* has shape
        ``(N_segments_i, n_mfcc)``.
    """
    segment_dir: str = os.path.join(subject_path, "segmented_audio")
    subject_id: str = os.path.basename(subject_path)
    subject_feats: list[np.ndarray] = []
    for i in range(1, 30):
        recording_id = f"{i:02d}"
        pattern = f"{subject_id}_{recording_id}_segment"
        rec_feat = process_recording(segment_dir, pattern, extractor)
        subject_feats.append(rec_feat)
    return subject_feats


def process_all_subjects(base_dir: str) -> None:
    """Extract and save MFCC features for every subject in *base_dir*.

    Expects a ``train/``, ``val/``, and ``test/`` split structure under
    *base_dir*.  Results are saved as ``raw_audio_mfccs.npy`` inside each
    subject directory.

    Args:
        base_dir: Root directory containing ``train/``, ``val/``,
            and ``test/`` subdirectories.
    """
    extractor = MFCCFeatureExtractor()
    for split in ["train", "val", "test"]:
        split_path = os.path.join(base_dir, split)
        for subject_id in tqdm(
            sorted(os.listdir(split_path)), desc=f"Processing {split}"
        ):
            subject_path = os.path.join(split_path, subject_id)
            try:
                feats = process_subject(subject_path, extractor)
                np.save(
                    os.path.join(subject_path, "raw_audio_mfccs.npy"),
                    np.array(feats, dtype=object),
                    allow_pickle=True,
                )
            except Exception as e:
                print(f"Failed on {subject_id}: {e}")


def extract_features(subject_path: str, **kwargs) -> list[np.ndarray]:
    """Convenience wrapper for per-subject MFCC extraction.

    Args:
        subject_path: Root directory of a single subject.
        **kwargs: Ignored; present for interface compatibility.

    Returns:
        List of 29 per-recording feature arrays as returned by
        :func:`process_subject`.
    """
    extractor = MFCCFeatureExtractor()
    return process_subject(subject_path, extractor)

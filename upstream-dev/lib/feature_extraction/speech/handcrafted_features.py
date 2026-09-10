"""Handcrafted speech feature utilities (legacy helper module).

Provides standalone functions for extracting 46 prosodic/MFCC features
from audio segments.  These functions mirror the logic in
:mod:`lib.feature_extraction.speech.extract_handcrafted_features_speech`
and are kept here for backwards compatibility.

Features extracted per segment (46 total):

* Energy, F0 mean, RMS, pause rate, phonation time, speech rate (6)
* MFCC coefficients 1–40, mean across frames (40)
"""

import os

import librosa
import numpy as np
from tqdm import tqdm


def extract_handcrafted_features(
    wav_path: str,
    sr_target: int = 16000,
) -> np.ndarray:
    """Extract 46 handcrafted features from a single audio segment.

    Args:
        wav_path: Path to a ``.wav`` audio segment file.
        sr_target: Target sample rate for loading.  Default 16 000 Hz.

    Returns:
        Feature vector of shape ``(46,)``.
    """
    y, sr = librosa.load(wav_path, sr=sr_target)

    energy: float = float(np.sum(y ** 2) / len(y))
    f0: np.ndarray = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0_mean: float = float(np.mean(f0[f0 > 0])) if np.any(f0 > 0) else 0.0
    rms: float = float(np.mean(librosa.feature.rms(y=y)))

    intervals: np.ndarray = librosa.effects.split(y, top_db=30)
    total_duration: float = len(y) / sr
    speech_duration: float = float(np.sum((intervals[:, 1] - intervals[:, 0]) / sr))
    pause_duration: float = total_duration - speech_duration
    pause_rate: float = pause_duration / total_duration if total_duration > 0 else 0.0
    phonation_time: float = (
        speech_duration / total_duration if total_duration > 0 else 0.0
    )
    speech_rate: float = (
        len(intervals) / total_duration if total_duration > 0 else 0.0
    )

    mfcc: np.ndarray = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean: np.ndarray = np.mean(mfcc, axis=1)  # (40,)

    prosodic = np.array([energy, f0_mean, rms, pause_rate, phonation_time, speech_rate])
    return np.concatenate([prosodic, mfcc_mean])  # (46,)


def process_recording(segment_dir: str, pattern: str) -> np.ndarray:
    """Extract features for all segments of one recording.

    Args:
        segment_dir: Directory containing ``.wav`` segment files.
        pattern: Substring used to filter segment filenames.

    Returns:
        Array of shape ``(N_segments, 46)``, or ``(1, 46)`` of zeros if
        no segments are found.
    """
    segment_files: list[str] = sorted(
        [f for f in os.listdir(segment_dir) if pattern in f]
    )
    feats: list[np.ndarray] = []
    for fname in segment_files:
        try:
            feature = extract_handcrafted_features(os.path.join(segment_dir, fname))
            feats.append(feature)
        except Exception as e:
            print(f"Error in {fname}: {e}")
    if not feats:
        return np.zeros((1, 46))
    return np.stack(feats)  # (N_segments, 46)


def process_subject(subject_path: str) -> list[np.ndarray]:
    """Extract features for all 29 recordings of one subject.

    Args:
        subject_path: Root directory of a single subject.

    Returns:
        List of 29 arrays, where element *i* has shape
        ``(N_segments_i, 46)``.
    """
    segment_dir: str = os.path.join(subject_path, "segmented_audio")
    subject_id: str = os.path.basename(subject_path)
    subject_feats: list[np.ndarray] = []
    for i in range(1, 30):
        recording_id = f"{i:02d}"
        pattern = f"{subject_id}_{recording_id}_segment"
        rec_feat = process_recording(segment_dir, pattern)
        subject_feats.append(rec_feat)
    return subject_feats


def process_all_subjects(base_dir: str) -> None:
    """Extract and save handcrafted speech features for all subjects.

    Expects ``train/``, ``val/``, and ``test/`` split subdirectories.
    Results are saved as ``raw_audio_features.npy`` inside each subject
    directory.

    Args:
        base_dir: Root directory containing split subdirectories.
    """
    for split in ["train", "val", "test"]:
        split_path = os.path.join(base_dir, split)
        for subject_id in tqdm(
            sorted(os.listdir(split_path)), desc=f"Processing {split}"
        ):
            subject_path = os.path.join(split_path, subject_id)
            try:
                feats = process_subject(subject_path)
                np.save(
                    os.path.join(subject_path, "raw_audio_features.npy"),
                    np.array(feats, dtype=object),
                    allow_pickle=True,
                )
            except Exception as e:
                print(f"Failed on {subject_id}: {e}")


def extract_features(subject_path: str, **kwargs) -> list[np.ndarray]:
    """Convenience wrapper matching the signature of other extractor modules.

    Args:
        subject_path: Root directory of a single subject.
        **kwargs: Ignored.

    Returns:
        Same as :func:`process_subject`.
    """
    return process_subject(subject_path)

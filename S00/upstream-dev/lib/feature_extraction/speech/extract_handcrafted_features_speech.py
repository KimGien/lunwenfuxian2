"""Handcrafted speech feature extraction for the TRI-DEP dataset.

Extracts 46 features per audio segment:

* 6 prosodic/acoustic features: energy, F0 mean, RMS, pause rate,
  phonation time ratio, speech rate
* 40 MFCC coefficients (mean across frames)

Results are saved as object-arrays of shape ``(29,)`` where element *i* is
an array of shape ``(N_segments_i, 46)`` for recording *i*.

Usage
-----
    python lib/feature_extraction/speech/extract_handcrafted_features_speech.py \\
        --base_dir data/split_dataset_june
"""

import os
import argparse

import librosa
import numpy as np
from tqdm import tqdm


# ── Per-segment feature extraction ───────────────────────────────────────────


def extract_handcrafted_features(
    wav_path: str,
    sr_target: int = 16000,
) -> np.ndarray:
    """Extract 46 handcrafted features from a single audio segment.

    Features (in order):

    * Energy — mean squared amplitude.
    * F0 mean — mean fundamental frequency over voiced frames (Hz); 0 if
      unvoiced throughout.
    * RMS energy — mean root-mean-square energy.
    * Pause rate — fraction of total duration without speech.
    * Phonation time — fraction of total duration with speech.
    * Speech rate — number of speech intervals per second.
    * MFCCs 1–40 — mean MFCC coefficients across frames.

    Args:
        wav_path: Path to a ``.wav`` audio segment file.
        sr_target: Target sample rate for loading (default 16 000 Hz).

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
    speech_duration: float = float(
        np.sum((intervals[:, 1] - intervals[:, 0]) / sr)
    )
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
    return np.concatenate([prosodic, mfcc_mean])   # (46,)


# ── Per-recording processing ──────────────────────────────────────────────────


def process_recording(segment_dir: str, pattern: str) -> np.ndarray:
    """Extract features for all segments belonging to one recording.

    Args:
        segment_dir: Directory containing ``.wav`` segment files.
        pattern: Filename prefix identifying segments for this recording
            (e.g. ``"01_part"``).

    Returns:
        Array of shape ``(N_segments, 46)``, or ``(1, 46)`` of zeros if no
        segments are found or all fail.
    """
    segment_files: list[str] = sorted(
        [f for f in os.listdir(segment_dir) if f.startswith(pattern)]
    )
    feats: list[np.ndarray] = []
    for fname in segment_files:
        try:
            feature = extract_handcrafted_features(
                os.path.join(segment_dir, fname)
            )
            feats.append(feature)
        except Exception as e:
            print(f"Error in {fname}: {e}")

    if not feats:
        return np.zeros((1, 46))
    return np.stack(feats)  # (N_segments, 46)


# ── Per-subject processing ────────────────────────────────────────────────────


def process_subject(subject_path: str) -> list[np.ndarray]:
    """Extract features for all 29 recordings of one subject.

    Args:
        subject_path: Root directory of a single subject; must contain a
            ``segmented_audio/`` sub-directory.

    Returns:
        List of 29 arrays, where element *i* has shape
        ``(N_segments_i, 46)``.
    """
    segment_dir: str = os.path.join(subject_path, "segmented_audio")
    subject_feats: list[np.ndarray] = []

    for i in range(1, 30):
        prefix = f"{i:02d}"
        pattern = f"{prefix}_part"
        rec_feat = process_recording(segment_dir, pattern)
        subject_feats.append(rec_feat)

    return subject_feats


# ── Batch processing ──────────────────────────────────────────────────────────


def process_all_subjects(base_dir: str) -> None:
    """Extract and save handcrafted speech features for every subject.

    For each subject directory the result is saved as
    ``<subject_id>/raw_audio_features.npy`` using ``allow_pickle=True``
    (object array of per-recording feature arrays).

    Args:
        base_dir: Root directory whose immediate sub-directories are
            per-subject folders.
    """
    for subject_id in tqdm(sorted(os.listdir(base_dir)), desc="Processing subjects"):
        subject_path = os.path.join(base_dir, subject_id)
        if not os.path.isdir(subject_path):
            continue
        try:
            feats: list[np.ndarray] = process_subject(subject_path)
            save_path: str = os.path.join(subject_path, "raw_audio_features.npy")
            np.save(save_path, np.array(feats, dtype=object), allow_pickle=True)
        except Exception as e:
            print(f"Failed on {subject_id}: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract handcrafted audio features."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="split_dataset",
        help="Base dataset directory containing subject folders",
    )
    args = parser.parse_args()
    process_all_subjects(args.base_dir)

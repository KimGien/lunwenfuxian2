"""Speech preprocessing pipeline for the TRI-DEP dataset.

Provides:
    - preprocess_audio: normalise amplitude, trim silence, median-filter
    - segment_audio: slice a waveform into overlapping fixed-length segments
    - process_and_segment: full per-file preprocessing pipeline
    - preprocess_all_speech: batch-process all subjects in the split_dataset tree
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard-library / third-party imports
# ──────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import scipy.signal

# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────────

ROOT_DIR: Path = Path("split_dataset")
RAW_SUBDIR: str = "audio"
PROC_SUBDIR: str = "processed_audio"
SEG_SUBDIR: str = "segmented_audio"

SEGMENT_DURATION: float = 5.0   # seconds per segment
OVERLAP_DURATION: float = 2.5   # seconds of overlap between consecutive segments
SR: int = 16000                  # target sample rate in Hz


# ──────────────────────────────────────────────────────────────────────────────
# Per-file helpers
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_audio(y: np.ndarray) -> np.ndarray:
    """Normalise amplitude, trim leading/trailing silence, and median-filter.

    Processing steps applied in order:

    1. Peak-normalise to the range ``[-1, 1]`` (skipped for silent signals).
    2. Trim silence below –20 dB using :func:`librosa.effects.trim`.
    3. Apply a length-3 median filter via :func:`scipy.signal.medfilt` to
       reduce transient noise.

    Args:
        y: 1-D float32 waveform array with shape ``(n_samples,)``.

    Returns:
        Preprocessed waveform with the same dtype and approximate shape as
        *y* (trimming may shorten it).

    Example:
        >>> import librosa
        >>> y, _ = librosa.load("speech.wav", sr=16000, mono=True)
        >>> y_clean = preprocess_audio(y)
    """
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    y, _ = librosa.effects.trim(y, top_db=20)
    y = scipy.signal.medfilt(y, kernel_size=3)
    return y


def segment_audio(
    y: np.ndarray,
    sr: int,
    file_basename: str,
    seg_dir: Path,
) -> None:
    """Slice a waveform into overlapping fixed-length segments and write to disk.

    Segments are saved as ``<file_basename>_part<N>.wav`` files inside
    *seg_dir*.  A stride of ``SEGMENT_DURATION - OVERLAP_DURATION`` seconds
    is used.  If the tail of the signal after the last full stride is longer
    than *OVERLAP_DURATION* seconds, an additional segment anchored at the
    very end of the signal is written.

    Files shorter than *SEGMENT_DURATION* seconds are skipped entirely.

    Args:
        y: 1-D float32 waveform array with shape ``(n_samples,)``.
        sr: Sample rate in Hz (used only to determine segment boundaries).
        file_basename: Base name (without extension) prepended to each
            output file name.
        seg_dir: Directory where segment ``.wav`` files are written.
    """
    step_size: int = int((SEGMENT_DURATION - OVERLAP_DURATION) * sr)
    segment_size: int = int(SEGMENT_DURATION * sr)

    # Skip if audio is shorter than 5 seconds
    if len(y) < segment_size:
        print(f"Skipping {file_basename}: shorter than {SEGMENT_DURATION} seconds")
        return

    part_num: int = 1
    start: int = 0
    for start in range(0, len(y) - segment_size + 1, step_size):
        segment: np.ndarray = y[start : start + segment_size]
        seg_name: str = f"{file_basename}_part{part_num}.wav"
        sf.write(seg_dir / seg_name, segment, sr)
        part_num += 1

    # Handle last partial segment
    if (len(y) - start) > OVERLAP_DURATION * sr:
        segment = y[-segment_size:]
        seg_name = f"{file_basename}_part{part_num}.wav"
        sf.write(seg_dir / seg_name, segment, sr)


def process_and_segment(
    file_path: str | Path,
    proc_dir: Path,
    seg_dir: Path,
) -> None:
    """Full preprocessing pipeline for a single audio file.

    Loads the audio at ``SR`` Hz (mono), applies :func:`preprocess_audio`,
    writes the processed waveform to *proc_dir*, then calls
    :func:`segment_audio` to write overlapping segments to *seg_dir*.

    Args:
        file_path: Path to the source ``.wav`` audio file.
        proc_dir: Directory where the fully processed (non-segmented) file is
            saved.  Created if absent.
        seg_dir: Directory where segmented ``.wav`` parts are saved.  Created
            if absent.

    Raises:
        Exception: Any exception is caught, printed, and skipped so that
            processing continues for other files.
    """
    try:
        file_name: str = Path(file_path).name
        base: str = Path(file_path).stem

        print(f"Processing {file_path}")
        y: np.ndarray
        y, _ = librosa.load(file_path, sr=SR, mono=True)
        y = preprocess_audio(y)

        # Save processed file
        proc_dir.mkdir(parents=True, exist_ok=True)
        sf.write(proc_dir / file_name, y, SR)

        # Save segmented parts
        seg_dir.mkdir(parents=True, exist_ok=True)
        segment_audio(y, SR, base, seg_dir)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Batch entry point
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_all_speech() -> None:
    """Batch-process audio for every subject in the split_dataset directory.

    Expects the directory tree::

        split_dataset/
        └── <subject_id>/
            └── audio/
                └── *.wav

    Processed files are written to ``<subject_id>/processed_audio/`` and
    overlapping segments to ``<subject_id>/segmented_audio/``.

    Subjects without an ``audio/`` sub-directory are silently skipped.
    """
    for subject_dir in sorted(ROOT_DIR.iterdir()):
        audio_dir: Path = subject_dir / RAW_SUBDIR
        if not audio_dir.exists():
            continue  # Skip if no audio folder

        for file in sorted(audio_dir.glob("*.wav")):
            proc_dir: Path = subject_dir / PROC_SUBDIR
            seg_dir: Path = subject_dir / SEG_SUBDIR
            process_and_segment(file, proc_dir, seg_dir)

    print("All speech files processed.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preprocess_all_speech()

"""EEG preprocessing pipeline for the TRI-DEP dataset.

Provides:
    - process_and_segment_eeg: load raw ``.mat`` EEG files, bandpass-filter,
      average-reference, and save fixed-length segments as ``.npy`` arrays
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard-library / third-party imports
# ──────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import scipy.io
from scipy import signal
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────────

# 0-based channel indices selected from the 128-channel EEG cap
channels_to_extract: list[int] = [
    9, 22, 11, 33, 24, 124, 122, 39, 29, 6,
    111, 115, 36, 104, 45, 42, 55, 93, 108, 50,
    52, 62, 92, 101, 58, 96, 70, 75, 83,
]

segment_length: int = 10   # seconds per EEG segment
fixed_num_segments: int = 30  # number of segments retained per subject


# ──────────────────────────────────────────────────────────────────────────────
# Core preprocessing function
# ──────────────────────────────────────────────────────────────────────────────

def process_and_segment_eeg(
    base_path: str | Path,
    save_base_path: str | Path,
    segment_length: int = 10,
) -> None:
    """Load, filter, reference, segment, and save EEG data for all subjects.

    For every subject directory found under *base_path* the function:

    1. Reads the raw ``.mat`` file from ``<subject_id>/eeg/``.
    2. Selects the predefined subset of EEG channels
       (``channels_to_extract``).
    3. Applies a 4th-order Butterworth bandpass filter (0.5 – 50 Hz).
    4. Applies average reference (subtract mean across selected channels).
    5. Splits the signal into non-overlapping *segment_length*-second epochs
       and retains exactly ``fixed_num_segments`` epochs.
    6. Saves the result as ``processed_segmented_eeg.npy`` with shape
       ``(fixed_num_segments, n_channels, samples_per_segment)``.

    Args:
        base_path: Root directory whose immediate sub-directories are
            per-subject folders, each containing an ``eeg/`` sub-folder with
            a single ``.mat`` file.
        save_base_path: Root directory where per-subject output folders are
            created.  May be the same as *base_path*.
        segment_length: Duration of each EEG segment in seconds.  Defaults
            to 10.

    Raises:
        Exception: Any per-file exception is caught, printed, and skipped so
            processing continues for the remaining subjects.

    Example:
        >>> process_and_segment_eeg("split_dataset", "split_dataset")
    """
    base_path = Path(base_path)
    save_base_path = Path(save_base_path)

    for subject_dir in base_path.iterdir():
        if not subject_dir.is_dir():
            continue

        subject_id: str = subject_dir.name
        eeg_dir: Path = subject_dir / "eeg"
        mat_files: list[Path] = list(eeg_dir.glob("*.mat"))
        if not mat_files:
            print(f"No .mat EEG files found for subject {subject_id}")
            continue

        for mat_file in mat_files:
            try:
                data: dict = scipy.io.loadmat(mat_file)
                eeg_key: str = [key for key in data.keys() if "mat" in key][0]
                eeg_data: np.ndarray = data[eeg_key]  # (129, timepoints)
                sampling_rate: int = int(
                    data.get("samplingRate", [[250]])[0][0]
                )  # fallback 250 Hz

                # Select channels → (n_channels, timepoints)
                eeg_selected: np.ndarray = eeg_data[channels_to_extract, :]

                # Bandpass filter 0.5 – 50 Hz
                nyquist: float = 0.5 * sampling_rate
                b, a = signal.butter(
                    4,
                    [0.5 / nyquist, 50.0 / nyquist],
                    btype="band",
                )
                filtered: np.ndarray = np.array(
                    [signal.filtfilt(b, a, ch) for ch in eeg_selected]
                )  # (n_channels, timepoints)

                # Average reference
                avg_reference: np.ndarray = np.mean(filtered, axis=0)  # (timepoints,)
                referred: np.ndarray = filtered - avg_reference  # (n_channels, timepoints)

                samples_per_segment: int = segment_length * sampling_rate
                total_segments: int = referred.shape[1] // samples_per_segment

                if total_segments < fixed_num_segments:
                    print(
                        f"Skipping {subject_id} — not enough data "
                        f"({total_segments} segments)."
                    )
                    continue

                # Truncate or keep exactly fixed_num_segments segments
                segments: np.ndarray = np.array([
                    referred[:, i * samples_per_segment : (i + 1) * samples_per_segment]
                    for i in range(fixed_num_segments)
                ])  # (fixed_num_segments, n_channels, samples_per_segment)

                subject_save_dir: Path = save_base_path / subject_id
                subject_save_dir.mkdir(parents=True, exist_ok=True)

                save_file: Path = subject_save_dir / "processed_segmented_eeg.npy"
                np.save(save_file, segments)
                print(f"Saved {save_file}, shape: {segments.shape}")

            except Exception as e:
                print(f"Error processing {mat_file}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process and segment EEG data from split_dataset."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="split_dataset",
        help="Path to the base EEG dataset folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="split_dataset",
        help="Path to save processed EEG segments",
    )
    args = parser.parse_args()

    process_and_segment_eeg(args.input_dir, args.output_dir, segment_length=segment_length)

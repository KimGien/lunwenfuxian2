"""Dataset creation utilities for the TRI-DEP pipeline.

Provides:
    - collect_aligned_subject_ids: find subjects with both EEG and audio data
    - copy_subject_data: copy a single subject's EEG and audio files to the output directory
    - create_split_dataset: orchestrate full dataset assembly from raw source folders
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard-library imports
# ──────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import shutil
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────────

def collect_aligned_subject_ids(eeg_dir: Path, audio_dir: Path) -> list[str]:
    """Return sorted subject IDs that have both EEG and audio data.

    A subject is considered *aligned* when a ``.mat`` file whose stem starts
    with the subject ID exists inside *eeg_dir* **and** a directory with the
    same subject ID exists inside *audio_dir*.

    Args:
        eeg_dir: Directory containing per-subject ``*.mat`` EEG files.  Each
            file's stem is expected to start with an 8-character subject ID
            (e.g. ``20150101_eeg.mat`` → ID ``20150101``).
        audio_dir: Directory whose immediate sub-directories are named after
            subject IDs.

    Returns:
        A sorted list of subject ID strings present in both directories.

    Example:
        >>> aligned = collect_aligned_subject_ids(
        ...     Path("dataset/EEG_128channels_resting_lanzhou_2015"),
        ...     Path("dataset/audio_lanzhou_2015-2"),
        ... )
        >>> print(aligned[:3])
        ['20150101', '20150103', '20150107']
    """
    eeg_subjects: set[str] = {f.stem[:8] for f in eeg_dir.glob("*.mat")}
    audio_subjects: set[str] = {d.name for d in audio_dir.iterdir() if d.is_dir()}

    aligned_ids: list[str] = sorted(eeg_subjects & audio_subjects)

    print(f"EEG subjects: {len(eeg_subjects)}")
    print(f"Audio subjects: {len(audio_subjects)}")
    print(f"Aligned subjects: {len(aligned_ids)}")

    return aligned_ids


def copy_subject_data(
    subject_id: str,
    eeg_dir: Path,
    audio_dir: Path,
    output_dir: Path,
) -> None:
    """Copy a single subject's EEG and audio data into the output directory.

    Creates the following layout under *output_dir*::

        output_dir/
        └── <subject_id>/
            ├── eeg/
            │   └── <subject_id>*.mat
            └── audio/
                └── <original audio files>

    Args:
        subject_id: 8-character subject identifier used to locate source files.
        eeg_dir: Source directory containing ``*.mat`` EEG files.
        audio_dir: Source directory containing per-subject audio sub-directories.
        output_dir: Root destination directory; created if absent.

    Raises:
        StopIteration: Handled internally — prints a warning and returns early
            when no ``.mat`` file matching *subject_id* is found in *eeg_dir*.
    """
    subj_output_dir: Path = output_dir / subject_id
    subj_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy EEG
    # Find the specific EEG file that starts with the subject_id
    try:
        eeg_src: Path = next(eeg_dir.glob(f"{subject_id}*.mat"))
    except StopIteration:
        print(f"Warning: No EEG file found for subject {subject_id}. Skipping.")
        return

    eeg_dst_dir: Path = subj_output_dir / "eeg"
    eeg_dst_dir.mkdir(exist_ok=True)
    shutil.copy(eeg_src, eeg_dst_dir / eeg_src.name)

    # Copy audio files
    audio_src_dir: Path = audio_dir / subject_id
    audio_dst_dir: Path = subj_output_dir / "audio"
    shutil.copytree(audio_src_dir, audio_dst_dir, dirs_exist_ok=True)


def create_split_dataset(
    eeg_dir: Path,
    audio_dir: Path,
    output_dir: Path,
) -> list[str]:
    """Assemble a split dataset by copying aligned subject data to *output_dir*.

    Collects all subject IDs present in both *eeg_dir* and *audio_dir*, then
    copies each subject's data into the unified *output_dir* structure.

    Args:
        eeg_dir: Directory containing raw EEG ``*.mat`` files.
        audio_dir: Directory containing raw per-subject audio sub-directories.
        output_dir: Destination root; created if it does not already exist.

    Returns:
        A sorted list of aligned subject ID strings that were successfully
        discovered (individual copy failures are logged but do not prevent
        other subjects from being processed).

    Example:
        >>> aligned = create_split_dataset(
        ...     eeg_dir=Path("dataset/EEG_128channels_resting_lanzhou_2015"),
        ...     audio_dir=Path("dataset/audio_lanzhou_2015-2"),
        ...     output_dir=Path("split_dataset"),
        ... )
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    aligned_ids: list[str] = collect_aligned_subject_ids(eeg_dir, audio_dir)

    for sid in aligned_ids:
        copy_subject_data(sid, eeg_dir, audio_dir, output_dir)

    print(f"Split dataset created at: {output_dir}")
    return aligned_ids


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_raw: Path = Path("dataset")
    output_dir: Path = Path("data/split_dataset_june")

    eeg_dir: Path = base_raw / "EEG_128channels_resting_lanzhou_2015"
    audio_dir: Path = base_raw / "audio_lanzhou_2015-2"

    aligned_ids: list[str] = create_split_dataset(eeg_dir, audio_dir, output_dir)

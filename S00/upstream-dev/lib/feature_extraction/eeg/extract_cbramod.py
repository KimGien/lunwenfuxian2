#!/usr/bin/env python3
"""CBraMod EEG embedding extraction for the TRI-DEP dataset.

Provides:
    - extract_cbramod_embeddings: load raw EEG ``.mat`` files, preprocess with
      MNE, patch the signal, and extract patch-level embeddings with a
      pretrained CBraMod transformer model
"""

# ──────────────────────────────────────────────────────────────────────────────
# Standard-library / third-party imports
# ──────────────────────────────────────────────────────────────────────────────

import os
import numpy as np
import torch
from pathlib import Path
from typing import Optional
import scipy.io
import mne
from lib.models.cbramod.cbramod_model import Model, Params

# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────────

# 0-based channel indices selected from the 128-channel EEG cap
channels_to_extract: list[int] = [
    9, 22, 33, 24, 124, 122, 6, 36, 104, 45,
    55, 108, 52, 62, 92, 58, 96, 70, 83,
]

channel_names: list[str] = [
    "FP2", "FP1", "F7", "F3", "F4", "F8", "FCz", "C3", "C4", "T3",
    "CPz", "T4", "P3", "Pz", "P4", "T5", "T6", "O1", "O2",
]

sfreq_new: int = 200    # target sampling frequency after resampling (Hz)
patch_size: int = 200   # number of time-samples per patch fed to CBraMod


# ──────────────────────────────────────────────────────────────────────────────
# Extraction function
# ──────────────────────────────────────────────────────────────────────────────

def extract_cbramod_embeddings(
    base_path: str | Path,
    pretrained_weights_path: Optional[str | Path] = None,
) -> None:
    """Extract CBraMod patch embeddings from EEG ``.mat`` files.

    For each subject sub-directory inside *base_path* the function:

    1. Loads the raw ``.mat`` EEG file from ``<subject_id>/eeg/``.
    2. Selects 19 canonical EEG channels, creates an MNE
       :class:`~mne.io.RawArray`, and applies:

       * Resampling to ``sfreq_new`` (200 Hz).
       * Band-pass filtering 0.3 – 75 Hz.
       * Notch filtering at 50 Hz.

    3. Splits the recording into 5-second fixed-length epochs.
    4. Divides each epoch into non-overlapping patches of size
       ``patch_size`` samples.
    5. Feeds the patched tensor through the CBraMod model (no gradient),
       averages over the segment and patch dimensions, and saves the
       resulting embedding array.

    Output shape per subject: ``(n_segments, 200)`` — one 200-dimensional
    vector per 5-second epoch — saved as
    ``<subject_dir>/cbramod_embeddings_mumtaz.npy``.

    Args:
        base_path: Root directory whose immediate sub-directories are
            per-subject folders (each containing an ``eeg/`` sub-folder with
            one ``.mat`` file).
        pretrained_weights_path: Optional path to a pretrained CBraMod
            checkpoint (``.pth``).  When provided, overrides
            ``Params.foundation_dir``.

    Raises:
        Exception: Per-subject exceptions are caught, printed, and skipped so
            that processing continues for remaining subjects.

    Example:
        >>> extract_cbramod_embeddings(
        ...     "split_dataset",
        ...     "cbramod_pretrained_weights/pretrained_weights2.pth",
        ... )
    """
    base_path = Path(base_path)
    param: Params = Params()

    if pretrained_weights_path is not None:
        param.foundation_dir = pretrained_weights_path

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: Model = Model(param).to(device)
    model.eval()

    for subject_folder in base_path.iterdir():
        if not subject_folder.is_dir():
            continue

        mat_files: list[Path] = list((subject_folder / "eeg").glob("*.mat"))

        if len(mat_files) == 0:
            print(f"No .mat file found in {subject_folder}")
            continue
        eeg_mat_path: Path = mat_files[0]
        save_eeg_path: Path = subject_folder / "cbramod_embeddings_mumtaz.npy"

        if not eeg_mat_path.exists():
            print(f"Missing EEG file for {subject_folder.name}")
            continue

        try:
            # Load EEG .mat
            data: dict = scipy.io.loadmat(eeg_mat_path)
            eeg_key: Optional[str] = next(
                (k for k in data if isinstance(data[k], np.ndarray) and data[k].ndim == 2),
                None,
            )
            if eeg_key is None:
                print(f"No valid EEG matrix in {eeg_mat_path.name}")
                continue

            # Select channels & create MNE Raw
            eeg_selected: np.ndarray = data[eeg_key][channels_to_extract, :]  # (19, T)
            info: mne.Info = mne.create_info(
                ch_names=channel_names, sfreq=250, ch_types="eeg"
            )
            raw: mne.io.RawArray = mne.io.RawArray(eeg_selected, info)

            # Preprocess
            raw = raw.resample(sfreq_new)
            raw.filter(0.3, 75)
            raw.notch_filter(50)

            # Epoch into 5-second chunks
            epochs: mne.Epochs = mne.make_fixed_length_epochs(
                raw, duration=5.0, preload=True
            )
            eeg_data: np.ndarray = epochs.get_data()  # (n_segments, n_channels, n_timestamps)
            n_segments: int
            n_channels: int
            n_timestamps: int
            n_segments, n_channels, n_timestamps = eeg_data.shape
            n_patches: int = n_timestamps // patch_size

            # Reshape into patches: (n_segments, n_channels, n_patches, patch_size)
            eeg_patched: np.ndarray = eeg_data[
                :, :, : n_patches * patch_size
            ].reshape(n_segments, n_channels, n_patches, patch_size)

            input_tensor: torch.Tensor = torch.tensor(
                eeg_patched, dtype=torch.float32
            ).to(device)  # (n_segments, n_channels, n_patches, patch_size)

            # Extract embeddings
            with torch.no_grad():
                embeddings: torch.Tensor = model(input_tensor)
            embeddings_mean: torch.Tensor = embeddings.mean(
                dim=[1, 2]
            )  # (n_segments, 200)

            np.save(save_eeg_path, embeddings_mean)
            print(
                f"{subject_folder.name} | saved: {save_eeg_path.name} "
                f"| shape: {embeddings_mean.shape}"
            )

        except Exception as e:
            print(f"{subject_folder.name}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ==== USER SETTINGS ====
    BASE_PATH: str = "split_dataset"  # folder containing subject folders
    PRETRAINED_WEIGHTS: str = "cbramod_pretrained_weights/pretrained_weights2.pth"  # or None

    extract_cbramod_embeddings(BASE_PATH, PRETRAINED_WEIGHTS)

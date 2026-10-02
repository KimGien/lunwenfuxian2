#!/usr/bin/env python3

import os
import numpy as np
import torch
from pathlib import Path
import scipy.io
import mne
from cbramod.cbramod_model import Model, Params

# EEG channels to extract (1-based MATLAB indexing converted to 0-based Python indexing)
channels_to_extract = [9, 22, 33, 24, 124, 122, 6, 36, 104, 45, 55, 108, 52, 62, 92, 58, 96, 70, 83]
channel_names = ['FP2', 'FP1', 'F7', 'F3', 'F4', 'F8', 'FCz', 'C3', 'C4', 'T3', 'CPz', 'T4',
                 'P3', 'Pz', 'P4', 'T5', 'T6', 'O1', 'O2']

sfreq_new = 200
patch_size = 200

def extract_cbramod_embeddings(base_path, pretrained_weights_path=None):
    """
    Extract CBraMod embeddings from EEG .mat files.

    Args:
        base_path (str | Path): Path to folder containing subject subfolders.
        pretrained_weights_path (str | Path, optional): Path to pretrained CBraMod weights.
    """

    base_path = Path(base_path)
    param = Params()

    if pretrained_weights_path is not None:
        param.foundation_dir = pretrained_weights_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(param).to(device)
    model.eval()

    for subject_folder in base_path.iterdir():
        if not subject_folder.is_dir():
            continue

        mat_files = list((subject_folder / "eeg").glob("*.mat"))

        if len(mat_files) == 0:
            print(f"No .mat file found in {subject_folder}")
            continue
        eeg_mat_path = mat_files[0]
        save_eeg_path = subject_folder / "cbramod_embeddings_mumtaz.npy"

        if not eeg_mat_path.exists():
            print(f"Missing EEG file for {subject_folder.name}")
            continue

        try:
            # Load EEG .mat
            data = scipy.io.loadmat(eeg_mat_path)
            eeg_key = next((k for k in data if isinstance(data[k], np.ndarray) and data[k].ndim == 2), None)
            if eeg_key is None:
                print(f"No valid EEG matrix in {eeg_mat_path.name}")
                continue

            # Select channels & create MNE Raw
            eeg_selected = data[eeg_key][channels_to_extract, :]
            info = mne.create_info(ch_names=channel_names, sfreq=250, ch_types='eeg')
            raw = mne.io.RawArray(eeg_selected, info)

            # Preprocess
            raw = raw.resample(sfreq_new)
            raw.filter(0.3, 75)
            raw.notch_filter(50)

            # Epoch into 5-second chunks
            epochs = mne.make_fixed_length_epochs(raw, duration=5.0, preload=True)
            eeg_data = epochs.get_data()
            n_segments, n_channels, n_timestamps = eeg_data.shape
            n_patches = n_timestamps // patch_size

            eeg_patched = eeg_data[:, :, :n_patches * patch_size].reshape(n_segments, n_channels, n_patches, patch_size)
            input_tensor = torch.tensor(eeg_patched, dtype=torch.float32).to(device)

            # Extract embeddings
            with torch.no_grad():
                embeddings = model(input_tensor)
            embeddings_mean = embeddings.mean(dim=[1, 2])  # shape: (segments, 200)

            np.save(save_eeg_path, embeddings_mean)
            print(f"{subject_folder.name} | saved: {save_eeg_path.name} | shape: {embeddings_mean.shape}")

        except Exception as e:
            print(f"{subject_folder.name}: {e}")


if __name__ == "__main__":
    # ==== USER SETTINGS ====
    BASE_PATH = "split_dataset"  # folder containing subject folders
    PRETRAINED_WEIGHTS = "cbramod_pretrained_weights/pretrained_weights2.pth"  # or None

    extract_cbramod_embeddings(BASE_PATH, PRETRAINED_WEIGHTS)

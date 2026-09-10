"""EEG feature extraction using the LaBraM pretrained model.

Loads a ``braindecode`` LaBraM model, runs forward_features on each
subject's preprocessed EEG segments, and saves the resulting embeddings
as ``.npy`` files.

Usage
-----
    python lib/feature_extraction/eeg/extract_labram.py
"""

import os
import glob

import numpy as np
import torch
from braindecode.models import Labram


# ── Model loading ─────────────────────────────────────────────────────────────


def load_labram_model(device: torch.device) -> torch.nn.Module:
    """Instantiate and return a LaBraM model in eval mode.

    The model is configured for the TRI-DEP EEG setup: 29 channels,
    2 500 time points per segment (10 s at 250 Hz), and a patch size of 200.

    Args:
        device: Torch device on which the model parameters are placed.

    Returns:
        LaBraM model in evaluation mode on *device*.
    """
    model = Labram(
        n_times=2500,          # 10 s × 250 Hz
        n_outputs=2,
        n_chans=29,
        sfreq=250,
        patch_size=200,
        use_mean_pooling=True,
    )
    model = model.to(device)
    model.eval()
    return model


# ── Embedding extraction ──────────────────────────────────────────────────────


def extract_labram_embeddings(
    source_base_path: str,
    target_base_path: str,
    device: torch.device,
) -> None:
    """Extract LaBraM embeddings for all subjects and save to disk.

    For each file matching ``<source_base_path>/*/*processed_segmented_eeg.npy``
    the function loads the array, runs :meth:`Labram.forward_features`, and
    saves the output as ``labram_embeddings.npy`` in the corresponding
    subject folder under *target_base_path*.

    Args:
        source_base_path: Root directory whose immediate sub-directories are
            per-subject folders containing ``processed_segmented_eeg.npy``.
        target_base_path: Root directory where per-subject output folders are
            created.  May be the same as *source_base_path*.
        device: Torch device used for model inference.
    """
    model = load_labram_model(device)

    eeg_files: list[str] = glob.glob(
        os.path.join(source_base_path, "*", "processed_segmented_eeg.npy")
    )
    if not eeg_files:
        print(f"No EEG files found in: {source_base_path}")
        return

    print(f"Found {len(eeg_files)} EEG files to process.")

    for i, eeg_file_path in enumerate(eeg_files):
        print(f"\n[{i + 1}/{len(eeg_files)}] Processing: {eeg_file_path}")
        try:
            eeg_data: np.ndarray = np.load(eeg_file_path)
            eeg_tensor = torch.from_numpy(eeg_data).float().to(device)

            with torch.no_grad():
                embeddings = model.forward_features(eeg_tensor)

            embeddings_np: np.ndarray = embeddings.cpu().numpy()

            subject_id: str = os.path.basename(os.path.dirname(eeg_file_path))
            output_folder: str = os.path.join(target_base_path, subject_id)
            os.makedirs(output_folder, exist_ok=True)

            output_path: str = os.path.join(output_folder, "labram_embeddings.npy")
            np.save(output_path, embeddings_np)

            print(
                f"Saved embeddings to: {output_path} | shape: {embeddings_np.shape}"
            )

        except Exception as e:
            print(f"Error processing {eeg_file_path}: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract LaBraM EEG embeddings.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/split_dataset_june",
        help="Root directory containing per-subject EEG folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/split_dataset_june",
        help="Root directory where per-subject embeddings are saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device ('cuda' or 'cpu').",
    )
    _args = parser.parse_args()
    extract_labram_embeddings(
        _args.base_dir, _args.output_dir, torch.device(_args.device)
    )

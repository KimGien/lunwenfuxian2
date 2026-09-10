"""Handcrafted EEG feature extraction for the TRI-DEP dataset.

Computes 10 statistical and spectral features per EEG channel per segment:

* Mean, standard deviation, kurtosis, skewness (statistical)
* Alpha (8–13 Hz), beta (13–30 Hz), theta (4–8 Hz), delta (0.5–4 Hz)
  band power (spectral)
* Spectral entropy and Shannon entropy

Output shape per subject: ``(n_segments, n_channels, 10)``.

Usage
-----
    python lib/feature_extraction/eeg/extract_handcrafted_features.py
"""

import os

import numpy as np
from scipy.signal import welch
from scipy.stats import entropy, kurtosis, skew


# ── Module-level constant ─────────────────────────────────────────────────────

sfreq: int = 250  # EEG sampling rate in Hz


# ── Feature functions ─────────────────────────────────────────────────────────


def shannon_entropy(signal: np.ndarray) -> float:
    """Compute the Shannon entropy of a signal using a 100-bin histogram.

    Args:
        signal: 1-D array of EEG amplitude values.

    Returns:
        Shannon entropy in bits (base-2 logarithm).
    """
    hist, _ = np.histogram(signal, bins=100, density=True)
    hist = hist + 1e-12  # avoid log(0)
    return float(-np.sum(hist * np.log2(hist)))


def spectral_entropy(signal: np.ndarray, fs: int, nperseg: int = 256) -> float:
    """Compute the normalised spectral entropy of a signal.

    Args:
        signal: 1-D array of EEG amplitude values.
        fs: Sampling frequency in Hz.
        nperseg: Length of each Welch segment (default 256).

    Returns:
        Spectral entropy in bits (base-2 logarithm).
    """
    freqs, psd = welch(signal, fs, nperseg=nperseg)
    psd_norm = psd / np.sum(psd)
    return float(entropy(psd_norm, base=2))


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    """Sum PSD values within a given frequency band.

    Args:
        freqs: 1-D array of frequency values from :func:`scipy.signal.welch`.
        psd: 1-D power spectral density array corresponding to *freqs*.
        band: ``(low_hz, high_hz)`` frequency band boundaries (inclusive).

    Returns:
        Total power within the band.
    """
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return float(np.sum(psd[idx_band]))


# ── Per-segment feature extraction ───────────────────────────────────────────


def extract_features(eeg_segment: np.ndarray) -> np.ndarray:
    """Extract 10 features from a single-channel EEG segment.

    Features (in order):

    1. Mean
    2. Kurtosis
    3. Skewness
    4. Standard deviation
    5. Alpha band power (8–13 Hz)
    6. Beta band power (13–30 Hz)
    7. Theta band power (4–8 Hz)
    8. Delta band power (0.5–4 Hz)
    9. Spectral entropy
    10. Shannon entropy

    Args:
        eeg_segment: 1-D array of shape ``(n_times,)`` for a single channel
            and segment.

    Returns:
        Feature vector of shape ``(10,)``.
    """
    features: list[float] = [
        float(np.mean(eeg_segment)),
        float(kurtosis(eeg_segment)),
        float(skew(eeg_segment)),
        float(np.std(eeg_segment)),
    ]

    freqs, psd = welch(eeg_segment, sfreq, nperseg=512)
    features.append(band_power(freqs, psd, (8, 13)))    # alpha
    features.append(band_power(freqs, psd, (13, 30)))   # beta
    features.append(band_power(freqs, psd, (4, 8)))     # theta
    features.append(band_power(freqs, psd, (0.5, 4)))   # delta
    features.append(spectral_entropy(eeg_segment, sfreq))
    features.append(shannon_entropy(eeg_segment))

    return np.array(features)


def extract_all_features(eeg_data: np.ndarray) -> np.ndarray:
    """Extract features for all segments and channels of one subject.

    Args:
        eeg_data: Array of shape ``(n_segments, n_channels, n_times)``.

    Returns:
        Feature array of shape ``(n_segments, n_channels, 10)``.
    """
    n_segments, n_channels, _ = eeg_data.shape
    n_features = 10
    all_features = np.zeros((n_segments, n_channels, n_features))
    for seg_i in range(n_segments):
        for ch_i in range(n_channels):
            all_features[seg_i, ch_i, :] = extract_features(eeg_data[seg_i, ch_i, :])
    return all_features


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Extract handcrafted EEG features for all subjects."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract handcrafted EEG features."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/split_dataset_june",
        help="Root directory containing per-subject EEG folders.",
    )
    args = parser.parse_args()
    base_dir: str = args.base_dir

    subject_folders: list[str] = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not subject_folders:
        print("No subject folders found.")
        return

    for subject_folder in subject_folders:
        subject_id: str = os.path.basename(subject_folder)
        eeg_path: str = os.path.join(subject_folder, "processed_segmented_eeg.npy")
        if not os.path.isfile(eeg_path):
            print(f"Missing file: {eeg_path}")
            continue

        print(f"Processing subject: {subject_id}")
        eeg_segmented: np.ndarray = np.load(eeg_path)  # (n_segments, n_channels, n_times)
        features: np.ndarray = extract_all_features(eeg_segmented)

        feature_path: str = os.path.join(subject_folder, "eeg_handcrafted_features.npy")
        np.save(feature_path, features)
        print(f"Saved features for {subject_id} in {feature_path}")


if __name__ == "__main__":
    main()

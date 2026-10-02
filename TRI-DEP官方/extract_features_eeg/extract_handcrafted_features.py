import os
import numpy as np
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import welch

sfreq = 250  # adjust if your EEG sampling rate is different

def shannon_entropy(signal):
    hist, _ = np.histogram(signal, bins=100, density=True)
    hist = hist + 1e-12  # avoid log(0)
    return -np.sum(hist * np.log2(hist))

def spectral_entropy(signal, sfreq, nperseg=256):
    freqs, psd = welch(signal, sfreq, nperseg=nperseg)
    psd_norm = psd / np.sum(psd)
    return entropy(psd_norm, base=2)

def band_power(freqs, psd, band):
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.sum(psd[idx_band])

def extract_features(eeg_segment):
    features = []
    features.append(np.mean(eeg_segment))
    features.append(kurtosis(eeg_segment))
    features.append(skew(eeg_segment))
    features.append(np.std(eeg_segment))

    freqs, psd = welch(eeg_segment, sfreq, nperseg=512)
    features.append(band_power(freqs, psd, (8,13)))   # alpha
    features.append(band_power(freqs, psd, (13,30)))  # beta
    features.append(band_power(freqs, psd, (4,8)))    # theta
    features.append(band_power(freqs, psd, (0.5,4)))  # delta

    features.append(spectral_entropy(eeg_segment, sfreq))
    features.append(shannon_entropy(eeg_segment))
    return np.array(features)

def extract_all_features(eeg_data):
    # eeg_data shape: (n_segments, n_channels, n_times)
    n_segments, n_channels, _ = eeg_data.shape
    n_features = 10
    all_features = np.zeros((n_segments, n_channels, n_features))
    for seg_i in range(n_segments):
        for ch_i in range(n_channels):
            all_features[seg_i, ch_i, :] = extract_features(eeg_data[seg_i, ch_i, :])
    return all_features

def main():
    base_dir = "split_dataset"  # adjust if different

    subject_folders = [
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not subject_folders:
        print("No subject folders found.")
        return

    for subject_folder in subject_folders:
        subject_id = os.path.basename(subject_folder)
        eeg_path = os.path.join(subject_folder, "processed_segmented_eeg.npy")
        if not os.path.isfile(eeg_path):
            print(f"⚠️ Missing file: {eeg_path}")
            continue

        print(f"Processing subject: {subject_id}")
        eeg_segmented = np.load(eeg_path)  # shape: (n_segments, n_channels, n_times)
        features = extract_all_features(eeg_segmented)

        feature_path = os.path.join(subject_folder, "eeg_handcrafted_features.npy")
        np.save(feature_path, features)
        print(f"Saved features for {subject_id} in {feature_path}")

if __name__ == "__main__":
    main()

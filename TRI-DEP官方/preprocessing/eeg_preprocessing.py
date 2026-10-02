# preprocessing/eeg_preprocessing.py

import os
import numpy as np
import scipy.io
from scipy import signal
from pathlib import Path

# EEG configuration
channels_to_extract = [9, 22, 11, 33, 24, 124, 122, 39, 29, 6, 111, 115, 36, 104, 45, 42, 55, 93, 108, 50, 52, 62, 92, 101, 58, 96, 70, 75, 83]
segment_length = 10  # seconds
fixed_num_segments = 30

def process_and_segment_eeg(base_path, save_base_path, segment_length=10):
    base_path = Path(base_path)
    save_base_path = Path(save_base_path)

    for subject_dir in base_path.iterdir():
        if not subject_dir.is_dir():
            continue
        
        subject_id = subject_dir.name
        eeg_dir = subject_dir / "eeg"
        mat_files = list(eeg_dir.glob("*.mat"))
        if not mat_files:
            print(f"No .mat EEG files found for subject {subject_id}")
            continue

        for mat_file in mat_files:
            try:
                data = scipy.io.loadmat(mat_file)
                eeg_key = [key for key in data.keys() if 'mat' in key][0]
                eeg_data = data[eeg_key]  # shape (129, timepoints)
                sampling_rate = int(data.get('samplingRate', [[250]])[0][0])  # fallback 250Hz

                eeg_selected = eeg_data[channels_to_extract, :]

                # Bandpass filter
                nyquist = 0.5 * sampling_rate
                b, a = signal.butter(4, [0.5 / nyquist, 50.0 / nyquist], btype='band')
                filtered = np.array([signal.filtfilt(b, a, ch) for ch in eeg_selected])

                # Average reference
                avg_reference = np.mean(filtered, axis=0)
                referred = filtered - avg_reference

                samples_per_segment = segment_length * sampling_rate
                total_segments = referred.shape[1] // samples_per_segment

                if total_segments < fixed_num_segments:
                    print(f"Skipping {subject_id} — not enough data ({total_segments} segments).")
                    continue

                # Truncate or keep exactly fixed_num_segments segments
                segments = np.array([
                    referred[:, i * samples_per_segment : (i + 1) * samples_per_segment]
                    for i in range(fixed_num_segments)
                ])  # shape: (fixed_num_segments, channels, timesteps)

                subject_save_dir = save_base_path / subject_id
                subject_save_dir.mkdir(parents=True, exist_ok=True)

                save_file = subject_save_dir / "processed_segmented_eeg.npy"
                np.save(save_file, segments)
                print(f"Saved {save_file}, shape: {segments.shape}")

            except Exception as e:
                print(f"Error processing {mat_file}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process and segment EEG data from split_dataset.")
    parser.add_argument("--input_dir", type=str, default="split_dataset", help="Path to the base EEG dataset folder")
    parser.add_argument("--output_dir", type=str, default="split_dataset", help="Path to save processed EEG segments")
    args = parser.parse_args()

    process_and_segment_eeg(args.input_dir, args.output_dir, segment_length=segment_length)

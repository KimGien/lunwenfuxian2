# extract_labram.py
import os
import glob
import numpy as np
import torch
from braindecode.models import Labram

def load_labram_model(device):
    model = Labram(
        n_times=2500,   # adjust if your EEG segments have a different time dimension
        n_outputs=2,
        n_chans=29,     # adjust to match your EEG channels
        sfreq=250,
        patch_size=200,
        use_mean_pooling=True
    )
    model = model.to(device)
    model.eval()
    return model

def extract_labram_embeddings(source_base_path, target_base_path, device):
    model = load_labram_model(device)

    # Look for EEG files in the expected folder structure
    eeg_files = glob.glob(os.path.join(source_base_path, "*", "processed_segmented_eeg.npy"))
    if not eeg_files:
        print(f"No EEG files found in: {source_base_path}")
        return

    print(f"Found {len(eeg_files)} EEG files to process.")

    for i, eeg_file_path in enumerate(eeg_files):
        print(f"\n[{i+1}/{len(eeg_files)}] Processing: {eeg_file_path}")
        try:
            eeg_data = np.load(eeg_file_path)
            eeg_tensor = torch.from_numpy(eeg_data).float().to(device)

            # Forward through Labram feature extractor
            with torch.no_grad():
                embeddings = model.forward_features(eeg_tensor)

            embeddings_np = embeddings.cpu().numpy()

            # Save under the same subject folder
            subject_id = os.path.basename(os.path.dirname(eeg_file_path))
            output_folder = os.path.join(target_base_path, subject_id)
            os.makedirs(output_folder, exist_ok=True)

            output_path = os.path.join(output_folder, "labram_embeddings.npy")
            np.save(output_path, embeddings_np)

            print(f"Saved embeddings to: {output_path} | shape: {embeddings_np.shape}")

        except Exception as e:
            print(f"Error processing {eeg_file_path}: {e}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_base_path = "split_dataset"   # adjust if different
    target_base_path = "split_dataset"   # save embeddings under same folder
    extract_labram_embeddings(source_base_path, target_base_path, device)

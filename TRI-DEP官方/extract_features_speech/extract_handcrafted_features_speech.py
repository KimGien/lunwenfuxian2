
import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm

def extract_handcrafted_features(wav_path, sr_target=16000):
    y, sr = librosa.load(wav_path, sr=sr_target)

    # Energy
    energy = np.sum(y ** 2) / len(y)

    # Fundamental frequency (pitch)
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0_mean = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0

    # RMS energy
    rms = np.mean(librosa.feature.rms(y=y))

    # Speech / silence durations
    intervals = librosa.effects.split(y, top_db=30)
    speech_duration = np.sum((intervals[:, 1] - intervals[:, 0]) / sr)
    total_duration = len(y) / sr
    pause_duration = total_duration - speech_duration
    pause_rate = pause_duration / total_duration if total_duration > 0 else 0
    phonation_time = speech_duration / total_duration if total_duration > 0 else 0
    speech_rate = len(intervals) / total_duration if total_duration > 0 else 0

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)  # (40,)

    feats = np.array([energy, f0_mean, rms, pause_rate, phonation_time, speech_rate])
    return np.concatenate([feats, mfcc_mean])  # (46,)

def process_recording(segment_dir, pattern):
    segment_files = sorted([f for f in os.listdir(segment_dir) if f.startswith(pattern)])
    feats = []
    for fname in segment_files:
        try:
            feature = extract_handcrafted_features(os.path.join(segment_dir, fname))
            feats.append(feature)
        except Exception as e:
            print(f"Error in {fname}: {e}")
    if not feats:
        return np.zeros((1, 46))
    return np.stack(feats)  # (N_segments, 46)

def process_subject(subject_path):
    segment_dir = os.path.join(subject_path, "segmented_audio")
    subject_feats = []
    for i in range(1, 30):  # recordings named 01_partX, 02_partX, ..., 29_partX
        prefix = f"{i:02d}"
        pattern = f"{prefix}_part"
        rec_feat = process_recording(segment_dir, pattern)
        subject_feats.append(rec_feat)  # (N_segments_i, 46)
    return subject_feats  # list of 29 arrays

def process_all_subjects(base_dir):
    for subject_id in tqdm(sorted(os.listdir(base_dir)), desc="Processing subjects"):
        subject_path = os.path.join(base_dir, subject_id)
        if not os.path.isdir(subject_path):
            continue
        try:
            feats = process_subject(subject_path)  # list of 29 arrays
            save_path = os.path.join(subject_path, "raw_audio_features.npy")
            np.save(save_path, np.array(feats, dtype=object), allow_pickle=True)
        except Exception as e:
            print(f"Failed on {subject_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract handcrafted audio features.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="split_dataset",  # default path
        help="Base dataset directory containing subject folders (default: split_dataset)"
    )
    args = parser.parse_args()

    process_all_subjects(args.base_dir)
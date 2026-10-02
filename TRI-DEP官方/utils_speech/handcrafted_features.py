import os
import numpy as np
import librosa
from tqdm import tqdm

def extract_handcrafted_features(wav_path, sr_target=16000):
    y, sr = librosa.load(wav_path, sr=sr_target)

    energy = np.sum(y ** 2) / len(y)
    f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
    f0_mean = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0
    rms = np.mean(librosa.feature.rms(y=y))
    intervals = librosa.effects.split(y, top_db=30)
    speech_duration = np.sum((intervals[:, 1] - intervals[:, 0]) / sr)
    total_duration = len(y) / sr
    pause_duration = total_duration - speech_duration
    pause_rate = pause_duration / total_duration if total_duration > 0 else 0
    phonation_time = speech_duration / total_duration if total_duration > 0 else 0
    speech_rate = len(intervals) / total_duration if total_duration > 0 else 0
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)  # (40,)
    feats = np.array([energy, f0_mean, rms, pause_rate, phonation_time, speech_rate])
    return np.concatenate([feats, mfcc_mean])  # (46,)

def process_recording(segment_dir, pattern):
    segment_files = sorted([f for f in os.listdir(segment_dir) if pattern in f])
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
    subject_id = os.path.basename(subject_path)
    subject_feats = []
    for i in range(1, 30):  # 29 recordings
        recording_id = f"{i:02d}"
        pattern = f"{subject_id}_{recording_id}_segment"
        rec_feat = process_recording(segment_dir, pattern)
        subject_feats.append(rec_feat)  # (N_segments_i, 46)
    return subject_feats  # list of 29 arrays

def process_all_subjects(base_dir):
    for split in ["train", "val", "test"]:
        split_path = os.path.join(base_dir, split)
        for subject_id in tqdm(sorted(os.listdir(split_path)), desc=f"Processing {split}"):
            subject_path = os.path.join(split_path, subject_id)
            try:
                feats = process_subject(subject_path)  # list of 29 arrays
                np.save(os.path.join(subject_path, "raw_audio_features.npy"), np.array(feats, dtype=object), allow_pickle=True)
            except Exception as e:
                print(f"Failed on {subject_id}: {e}")

def extract_features(subject_path, **kwargs):
    """
    Wrapper to match the signature of other extractors.
    """
    return process_subject(subject_path)
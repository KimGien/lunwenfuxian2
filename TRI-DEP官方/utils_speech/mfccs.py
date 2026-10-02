# mfccs.py
import os
import numpy as np
import librosa
from tqdm import tqdm

class MFCCFeatureExtractor:
    def __init__(self, sr_target=16000, n_mfcc=40):
        self.sr_target = sr_target
        self.n_mfcc = n_mfcc

    def extract_embedding(self, wav_path):
        """
        Extracts MFCC features from an audio file.
        Returns a 2D array of shape (T, n_mfcc), where T is the number of time steps.
        This ensures compatibility with the main feature extraction script.
        """
        y, sr = librosa.load(wav_path, sr=self.sr_target)
        # librosa.feature.mfcc returns shape (n_mfcc, T). We transpose it to (T, n_mfcc).
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc).T
        return mfcc

def process_recording(segment_dir, pattern, extractor):
    segment_files = sorted([f for f in os.listdir(segment_dir) if pattern in f])
    feats = []
    for fname in segment_files:
        try:
            feat = extractor.extract_embedding(os.path.join(segment_dir, fname))
            feats.append(feat)
        except Exception as e:
            print(f"Error in {fname}: {e}")
    if not feats:
        return np.zeros((1, extractor.n_mfcc))
    return np.stack(feats)  # (N_segments, n_mfcc)

def process_subject(subject_path, extractor):
    segment_dir = os.path.join(subject_path, "segmented_audio")
    subject_id = os.path.basename(subject_path)
    subject_feats = []
    for i in range(1, 30):  # 29 recordings
        recording_id = f"{i:02d}"
        pattern = f"{subject_id}_{recording_id}_segment"
        rec_feat = process_recording(segment_dir, pattern, extractor)
        subject_feats.append(rec_feat)  # list of arrays (N_segments_i, n_mfcc)
    return subject_feats

def process_all_subjects(base_dir):
    extractor = MFCCFeatureExtractor()
    for split in ["train", "val", "test"]:
        split_path = os.path.join(base_dir, split)
        for subject_id in tqdm(sorted(os.listdir(split_path)), desc=f"Processing {split}"):
            subject_path = os.path.join(split_path, subject_id)
            try:
                feats = process_subject(subject_path, extractor)  # list of arrays
                np.save(os.path.join(subject_path, "raw_audio_mfccs.npy"), np.array(feats, dtype=object), allow_pickle=True)
            except Exception as e:
                print(f"Failed on {subject_id}: {e}")

def extract_features(subject_path, **kwargs):
    """
    Wrapper to match the signature of other extractors.
    """
    extractor = MFCCFeatureExtractor()
    return process_subject(subject_path, extractor)

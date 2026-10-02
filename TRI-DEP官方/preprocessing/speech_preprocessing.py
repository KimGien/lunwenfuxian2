# preprocessing/speech_preprocessing.py

import os
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import scipy.signal

# ===============================
ROOT_DIR = Path("split_dataset")
RAW_SUBDIR = "audio"
PROC_SUBDIR = "processed_audio"
SEG_SUBDIR = "segmented_audio"

SEGMENT_DURATION = 5.0      # seconds
OVERLAP_DURATION = 2.5      # seconds
SR = 16000                  # sample rate
# ===============================

def preprocess_audio(y):
    """Normalize, trim silence, and apply median filtering."""
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    y, _ = librosa.effects.trim(y, top_db=20)
    y = scipy.signal.medfilt(y, kernel_size=3)
    return y

def segment_audio(y, sr, file_basename, seg_dir):
    step_size = int((SEGMENT_DURATION - OVERLAP_DURATION) * sr)
    segment_size = int(SEGMENT_DURATION * sr)

    # Skip if audio is shorter than 5 seconds
    if len(y) < segment_size:
        print(f"Skipping {file_basename}: shorter than {SEGMENT_DURATION} seconds")
        return

    part_num = 1
    for start in range(0, len(y) - segment_size + 1, step_size):
        segment = y[start:start + segment_size]
        seg_name = f"{file_basename}_part{part_num}.wav"
        sf.write(seg_dir / seg_name, segment, sr)
        part_num += 1

    # Handle last partial segment
    if (len(y) - start) > OVERLAP_DURATION * sr:
        segment = y[-segment_size:]
        seg_name = f"{file_basename}_part{part_num}.wav"
        sf.write(seg_dir / seg_name, segment, sr)

def process_and_segment(file_path, proc_dir, seg_dir):
    """Full processing pipeline for a single file."""
    try:
        file_name = Path(file_path).name
        base = Path(file_path).stem

        print(f"Processing {file_path}")
        y, _ = librosa.load(file_path, sr=SR, mono=True)
        y = preprocess_audio(y)

        # Save processed file
        proc_dir.mkdir(parents=True, exist_ok=True)
        sf.write(proc_dir / file_name, y, SR)

        # Save segmented parts
        seg_dir.mkdir(parents=True, exist_ok=True)
        segment_audio(y, SR, base, seg_dir)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def preprocess_all_speech():
    """
    Processes audio for all subjects in split_dataset/[subject_id]/audio/
    Stores results in split_dataset/[subject_id]/processed_audio/ and segmented_audio/
    """
    for subject_dir in sorted(ROOT_DIR.iterdir()):
        audio_dir = subject_dir / RAW_SUBDIR
        if not audio_dir.exists():
            continue  # Skip if no audio folder

        for file in sorted(audio_dir.glob("*.wav")):
            proc_dir = subject_dir / PROC_SUBDIR
            seg_dir = subject_dir / SEG_SUBDIR
            process_and_segment(file, proc_dir, seg_dir)

    print("All speech files processed.")

if __name__ == "__main__":
    preprocess_all_speech()

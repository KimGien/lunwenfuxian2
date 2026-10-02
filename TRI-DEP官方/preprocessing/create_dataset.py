# scripts/create_split_dataset.py

from pathlib import Path
import shutil

def collect_aligned_subject_ids(eeg_dir, audio_dir):
    eeg_subjects = {f.stem[:8] for f in eeg_dir.glob("*.mat")}
    audio_subjects = {d.name for d in audio_dir.iterdir() if d.is_dir()}

    aligned_ids = sorted(eeg_subjects & audio_subjects)

    print(f"EEG subjects: {len(eeg_subjects)}")
    print(f"Audio subjects: {len(audio_subjects)}")
    print(f"✅ Aligned subjects: {len(aligned_ids)}")

    return aligned_ids

def copy_subject_data(subject_id, eeg_dir, audio_dir, output_dir):
    subj_output_dir = output_dir / subject_id
    subj_output_dir.mkdir(parents=True, exist_ok=True)

    # Copy EEG
    # Find the specific EEG file that starts with the subject_id
    try:
        eeg_src = next(eeg_dir.glob(f"{subject_id}*.mat"))
    except StopIteration:
        print(f"⚠️ Warning: No EEG file found for subject {subject_id}. Skipping.")
        return

    eeg_dst_dir = subj_output_dir / "eeg"
    eeg_dst_dir.mkdir(exist_ok=True)
    shutil.copy(eeg_src, eeg_dst_dir / eeg_src.name)

    # Copy audio files
    audio_src_dir = audio_dir / subject_id
    audio_dst_dir = subj_output_dir / "audio"
    shutil.copytree(audio_src_dir, audio_dst_dir, dirs_exist_ok=True)


def create_split_dataset(eeg_dir, audio_dir, output_dir):
    output_dir.mkdir(exist_ok=True, parents=True)

    aligned_ids = collect_aligned_subject_ids(eeg_dir, audio_dir)

    for sid in aligned_ids:
        copy_subject_data(sid, eeg_dir, audio_dir, output_dir)

    print(f"Split dataset created at: {output_dir}")
    return aligned_ids

if __name__ == "__main__":
    base_raw = Path("dataset")
    output_dir = Path("split_dataset")

    eeg_dir = base_raw / "EEG_128channels_resting_lanzhou_2015"
    audio_dir = base_raw / "audio_lanzhou_2015-2"

    aligned_ids = create_split_dataset(eeg_dir, audio_dir, output_dir)



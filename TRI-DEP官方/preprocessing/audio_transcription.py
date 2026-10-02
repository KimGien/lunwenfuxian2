import whisperx
import pandas as pd
from pathlib import Path

# ===============================
BASE_PATH = Path("split_dataset")
DEVICE = "cpu"          # or "cuda"
MODEL_SIZE = "medium"   # e.g. tiny, base, small, medium, large
# ===============================

def transcribe_subject_folder(subject_folder, output_csv, device=DEVICE, model_size=MODEL_SIZE):
    print(f"Loading WhisperX model: {model_size} on {device} ...")
    model = whisperx.load_model(model_size, device=device, compute_type="float32")

    audio_folder = subject_folder / "audio"
    if not audio_folder.exists():
        print(f"Audio folder missing: {audio_folder}")
        return

    audio_files = sorted([f for f in audio_folder.iterdir() if f.suffix == ".wav"])
    results = []

    for audio_path in audio_files:
        print(f"Transcribing: {audio_path.name}")
        try:
            result = model.transcribe(str(audio_path), batch_size=1, language="zh", task="transcribe")
            chinese_text = result['segments'][0]['text'] if result['segments'] else ""
        except Exception as e:
            chinese_text = "[ERROR]"
            print(f"Failed: {audio_path.name}, error: {e}")

        results.append([audio_path.name, chinese_text])

    # Save transcription CSV in the subject folder
    df = pd.DataFrame(results, columns=["audio file", "Chinese text"])
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Transcriptions saved to: {output_csv}")

def transcribe_all_subjects():
    for subject_dir in BASE_PATH.iterdir():
        if subject_dir.is_dir():
            subject_id = subject_dir.name
            output_csv = subject_dir / f"transcriptions_{subject_id}.csv"
            print(f"\n🗂️ Transcribing subject: {subject_id}")
            transcribe_subject_folder(subject_dir, output_csv)

if __name__ == "__main__":
    transcribe_all_subjects()

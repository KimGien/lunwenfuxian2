"""Audio transcription utilities for the TRI-DEP dataset using WhisperX.

Provides:
    - transcribe_subject_folder: transcribe all ``.wav`` files for one subject
      and save results to a per-subject CSV
    - transcribe_all_subjects: batch-transcribe every subject in the dataset
"""

# ──────────────────────────────────────────────────────────────────────────────
# Third-party imports
# ──────────────────────────────────────────────────────────────────────────────

import whisperx
import pandas as pd
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────────

BASE_PATH: Path = Path("split_dataset")
DEVICE: str = "cpu"        # or "cuda"
MODEL_SIZE: str = "medium" # e.g. tiny, base, small, medium, large


# ──────────────────────────────────────────────────────────────────────────────
# Transcription functions
# ──────────────────────────────────────────────────────────────────────────────

def transcribe_subject_folder(
    subject_folder: Path,
    output_csv: Path,
    device: str = DEVICE,
    model_size: str = MODEL_SIZE,
) -> None:
    """Transcribe all ``.wav`` files for a single subject and write results to CSV.

    Loads a WhisperX model, iterates over every ``.wav`` file inside
    ``<subject_folder>/audio/``, transcribes each file in Mandarin Chinese
    (``language="zh"``), and stores the filename together with the
    transcribed text in *output_csv*.

    The CSV has two columns:

    * ``audio file`` — the file name (not the full path).
    * ``Chinese text`` — the first transcription segment; ``"[ERROR]"`` if
      transcription failed.

    Args:
        subject_folder: Root directory for one subject (must contain an
            ``audio/`` sub-directory with ``.wav`` files).
        output_csv: Destination path for the transcription CSV file.
        device: PyTorch device string passed to
            :func:`whisperx.load_model` (``"cpu"`` or ``"cuda"``).
        model_size: WhisperX model size string (e.g. ``"tiny"``,
            ``"base"``, ``"medium"``, ``"large"``).

    Raises:
        FileNotFoundError: Handled internally — prints a warning and returns
            early when ``<subject_folder>/audio/`` does not exist.
        Exception: Per-file transcription exceptions are caught, logged as
            ``"[ERROR]"``, and processing continues for remaining files.
    """
    print(f"Loading WhisperX model: {model_size} on {device} ...")
    model = whisperx.load_model(model_size, device=device, compute_type="float32")

    audio_folder: Path = subject_folder / "audio"
    if not audio_folder.exists():
        print(f"Audio folder missing: {audio_folder}")
        return

    audio_files: list[Path] = sorted(
        [f for f in audio_folder.iterdir() if f.suffix == ".wav"]
    )
    results: list[list[str]] = []

    for audio_path in audio_files:
        print(f"Transcribing: {audio_path.name}")
        chinese_text: str
        try:
            result: dict = model.transcribe(
                str(audio_path), batch_size=1, language="zh", task="transcribe"
            )
            chinese_text = result["segments"][0]["text"] if result["segments"] else ""
        except Exception as e:
            chinese_text = "[ERROR]"
            print(f"Failed: {audio_path.name}, error: {e}")

        results.append([audio_path.name, chinese_text])

    # Save transcription CSV in the subject folder
    df: pd.DataFrame = pd.DataFrame(results, columns=["audio file", "Chinese text"])
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Transcriptions saved to: {output_csv}")


def transcribe_all_subjects() -> None:
    """Batch-transcribe every subject directory found under ``BASE_PATH``.

    Iterates over all immediate sub-directories of ``BASE_PATH``, treating
    each as a subject folder.  For each subject the transcription CSV is
    saved as::

        <BASE_PATH>/<subject_id>/transcriptions_<subject_id>.csv

    Non-directory entries at the top level of ``BASE_PATH`` are silently
    skipped.
    """
    for subject_dir in BASE_PATH.iterdir():
        if subject_dir.is_dir():
            subject_id: str = subject_dir.name
            output_csv: Path = subject_dir / f"transcriptions_{subject_id}.csv"
            print(f"\nTranscribing subject: {subject_id}")
            transcribe_subject_folder(subject_dir, output_csv)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transcribe_all_subjects()

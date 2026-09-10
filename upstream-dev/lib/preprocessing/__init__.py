"""Preprocessing pipeline: dataset creation, EEG/speech processing, transcription."""

from lib.preprocessing.create_dataset import create_split_dataset
from lib.preprocessing.eeg_preprocessing import process_and_segment_eeg
from lib.preprocessing.speech_preprocessing import preprocess_all_speech
from lib.preprocessing.audio_transcription import (
    transcribe_all_subjects,
    transcribe_subject_folder,
)

__all__ = [
    "create_split_dataset",
    "process_and_segment_eeg",
    "preprocess_all_speech",
    "transcribe_all_subjects",
    "transcribe_subject_folder",
]

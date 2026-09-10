"""Baseline model architectures and datasets (ViT and DenseNet-121 baselines)."""

from lib.baselines.datasets import SpectrogramEmbeddingDataset, collate_spectrogram
from lib.baselines.models import ConvPoolReLUClassifier, EEG1DEncoder

__all__ = [
    "SpectrogramEmbeddingDataset",
    "collate_spectrogram",
    "ConvPoolReLUClassifier",
    "EEG1DEncoder",
]

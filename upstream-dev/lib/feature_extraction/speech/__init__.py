"""Speech feature extraction: XLSR-53, HuBERT, MFCC, and handcrafted features."""

from lib.feature_extraction.speech.chinese_hubert import HubertChineseExtractor
from lib.feature_extraction.speech.extract_features_speech import (
    get_encoder,
    get_feature_extractor,
    process_all_subjects,
    process_subject,
)
from lib.feature_extraction.speech.extract_handcrafted_features_speech import (
    extract_handcrafted_features,
)
from lib.feature_extraction.speech.mfccs import MFCCFeatureExtractor
from lib.feature_extraction.speech.xslr_53 import XSLR53FeatureExtractor

__all__ = [
    "HubertChineseExtractor",
    "get_encoder",
    "get_feature_extractor",
    "process_all_subjects",
    "process_subject",
    "extract_handcrafted_features",
    "MFCCFeatureExtractor",
    "XSLR53FeatureExtractor",
]

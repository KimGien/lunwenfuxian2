"""Text feature extraction: MacBERT, BERT-base, MPNet, and XLNet encoders."""

from lib.feature_extraction.text.chinese_bert_base import encode_texts_bert
from lib.feature_extraction.text.chinese_macbert import encode_texts_macbert
from lib.feature_extraction.text.chinese_mpnet import encode_texts_mpnet
from lib.feature_extraction.text.chinese_xlnet import encode_texts_xlnet
from lib.feature_extraction.text.extract_features_text import run_encoding

__all__ = [
    "encode_texts_bert",
    "encode_texts_macbert",
    "encode_texts_mpnet",
    "encode_texts_xlnet",
    "run_encoding",
]

"""Text feature extraction pipeline for the TRI-DEP dataset.

Runs one or more pretrained Chinese language models over per-subject
transcription files and saves sentence embeddings as ``.npy`` arrays.

Supported models
----------------
* ``mpnet``   — Chinese MPNet (``lib.feature_extraction.text.chinese_mpnet``)
* ``macbert`` — MacBERT (``lib.feature_extraction.text.chinese_macbert``)
* ``bert``    — Chinese BERT-base (``lib.feature_extraction.text.chinese_bert_base``)
* ``xlnet``   — Chinese XLNet (``lib.feature_extraction.text.chinese_xlnet``)

Usage
-----
    python lib/feature_extraction/text/extract_features_text.py \\
        --models macbert bert \\
        --base_dir data/split_dataset_june \\
        --save_dir data/split_dataset_june
"""

import argparse

from lib.feature_extraction.text.chinese_mpnet import encode_texts_mpnet
from lib.feature_extraction.text.chinese_macbert import encode_texts_macbert
from lib.feature_extraction.text.chinese_bert_base import encode_texts_bert
from lib.feature_extraction.text.chinese_xlnet import encode_texts_xlnet


# ── Encoding helper ──────────────────────────────────────────────────────────


def run_encoding(model_fn, base_dir: str, save_dir: str, model_name: str) -> None:
    """Run a single encoding function and print progress messages.

    Args:
        model_fn: Callable with signature ``(base_dir, save_dir) -> None``
            that encodes all subjects and saves ``.npy`` files.
        base_dir: Root directory containing per-subject transcription files.
        save_dir: Root directory where embedding ``.npy`` files are written.
        model_name: Human-readable model name used only for console output.
    """
    print(f"Starting encoding with {model_name}...")
    model_fn(base_dir, save_dir)
    print(f"Finished encoding with {model_name}.\n")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract text embeddings for each subject"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mpnet", "macbert", "bert", "xlnet"],
        help="List of models to use: mpnet, macbert, bert, xlnet",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="split_dataset",
        help="Path to the split_dataset directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="split_dataset",
        help="Directory where .npy embeddings are saved",
    )
    args = parser.parse_args()

    model_map: dict[str, object] = {
        "mpnet": encode_texts_mpnet,
        "macbert": encode_texts_macbert,
        "bert": encode_texts_bert,
        "xlnet": encode_texts_xlnet,
    }

    for model_name in args.models:
        if model_name not in model_map:
            print(f"Unknown model: {model_name}")
            continue
        run_encoding(model_map[model_name], args.base_dir, args.save_dir, model_name)

import argparse
from utils_text.chinese_mpnet import encode_texts_mpnet
from utils_text.chinese_macbert import encode_texts_macbert
from utils_text.chinese_bert_base import encode_texts_bert
from utils_text.chinese_xlnet import encode_texts_xlnet

def run_encoding(model_fn, base_dir, save_dir, model_name):
    print(f"Starting encoding with {model_name}...")
    model_fn(base_dir, save_dir)
    print(f"Finished encoding with {model_name}.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text embeddings for each subject")
    parser.add_argument(
        "--models", nargs="+", default=["mpnet", "macbert", "bert", "xlnet"],
        help="List of models to use: mpnet, macbert, bert, xlnet"
    )
    parser.add_argument("--base_dir", type=str, default="split_dataset", help="Path to split_dataset/")
    parser.add_argument("--save_dir", type=str, default="split_dataset", help="Where to save .npy embeddings")
    args = parser.parse_args()

    model_map = {
        "mpnet": encode_texts_mpnet,
        "macbert": encode_texts_macbert,
        "bert": encode_texts_bert,
        "xlnet": encode_texts_xlnet,
    }

    for model_name in args.models:
        if model_name not in model_map:
            print(f"Unknown model: {model_name}")
            continue
        model_fn = model_map[model_name]
        run_encoding(model_fn, args.base_dir, args.save_dir, model_name)

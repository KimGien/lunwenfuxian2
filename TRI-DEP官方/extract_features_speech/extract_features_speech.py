import os
import torch
import numpy as np
from tqdm import tqdm
import argparse
import importlib

def get_feature_extractor(name):
    if name == "xslr53":
        from utils_speech.xslr_53 import XSLR53FeatureExtractor
        return XSLR53FeatureExtractor()
    elif name == "mfcc":
        from utils_speech.mfccs import MFCCFeatureExtractor
        return MFCCFeatureExtractor()
    elif name == "chinese_hubert":
        from utils_speech.chinese_hubert import HubertChineseExtractor
        return HubertChineseExtractor()
    else:
        raise ValueError(f"Unknown feature extractor: {name}")

def get_encoder(name, input_dim):
    module_name = f"encoder.{name}"
    class_name = None
    # Determine class name by convention
    if name == "cnn_bigru":
        class_name = "BiGRUAttentionEncoder"
    elif name == "cnn_lstm":
        class_name = "CNNLSTMEncoder"
    elif name == "cnn_gru":
        class_name = "CNNGRUEncoder"
    elif name == "cnn":
        class_name = "AudioCNNEncoder"
    elif name == "cnn_bilstm":
        class_name = "AudioTemporalBiLSTMEncoder"
    else:
        raise ValueError(f"Unknown encoder: {name}")

    module = importlib.import_module(module_name)
    EncoderClass = getattr(module, class_name)
    return EncoderClass(input_dim=input_dim)

def process_recording(segment_dir, pattern, extractor, encoder, device, encoder_output_dim):
    segment_files = sorted([f for f in os.listdir(segment_dir) if f.startswith(pattern) and f.endswith(".wav")])
    segment_embeddings = []

    for fname in segment_files:
        try:
            wav_path = os.path.join(segment_dir, fname)
            feat = extractor.extract_embedding(wav_path)  # shape (T, feat_dim)
            feat_tensor = torch.tensor(feat, dtype=torch.float32)
            feat_mean = torch.mean(feat_tensor, dim=0)  # (feat_dim,)
            segment_embeddings.append(feat_mean)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    if not segment_embeddings:
        return torch.zeros(1, encoder_output_dim)

    segments_tensor = torch.stack(segment_embeddings, dim=0).to(device)

    with torch.no_grad():
        encoded_segments = encoder(segments_tensor)  # (N_segments, encoder_output_dim)

    audio_embedding = torch.mean(encoded_segments, dim=0, keepdim=True)  # (1, encoder_output_dim)
    return audio_embedding.cpu()

def process_subject(subject_path, extractor, encoder, device, encoder_output_dim):
    segment_dir = os.path.join(subject_path, "segmented_audio")
    audio_file_embeddings = []

    for i in range(1, 30):
        prefix = f"{i:02d}"
        pattern = f"{prefix}_part"
        audio_embedding = process_recording(segment_dir, pattern, extractor, encoder, device, encoder_output_dim)
        audio_file_embeddings.append(audio_embedding)

    subject_embedding = torch.cat(audio_file_embeddings, dim=0)
    return subject_embedding.numpy()

def process_all_subjects(base_dir, extractor, encoder, device, encoder_output_dim):
    for subject_id in tqdm(sorted(os.listdir(base_dir)), desc="Processing subjects"):
        subject_path = os.path.join(base_dir, subject_id)
        if os.path.isdir(subject_path):
            try:
                feats = process_subject(subject_path, extractor, encoder, device, encoder_output_dim)
                save_path = os.path.join(subject_path, f"audio_{feature_extractor_name}_encoded_{encoder_name}.npy")
                np.save(save_path, feats)
                print(f"Succeeded on {subject_id}, features shape: {feats.shape}")
            except Exception as e:
                print(f"Failed on {subject_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio embedding extraction")
    parser.add_argument("--base_dir", type=str, default="split_dataset", help="Base directory containing subjects")
    parser.add_argument("--feature_extractor", type=str, default="xslr53",
                        choices=["xslr53", "mfcc", "chinese_hubert"], help="Feature extractor to use")
    parser.add_argument("--encoder", type=str, default="cnn_bigru",
                        choices=["cnn_bigru", "cnn_lstm", "cnn_gru", "cnn_bilstm", "cnn"], help="Encoder architecture")
    parser.add_argument("--input_dim", type=int, default=1024, help="Input dimension for encoder")
    parser.add_argument("--encoder_output_dim", type=int, default=256, help="Output dimension of encoder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Global names for saving and print
    feature_extractor_name = args.feature_extractor
    encoder_name = args.encoder

    print(f"Loading feature extractor: {feature_extractor_name}")
    extractor = get_feature_extractor(feature_extractor_name)

    print(f"Loading encoder: {encoder_name} with input_dim={args.input_dim}")
    encoder = get_encoder(encoder_name, input_dim=args.input_dim).to(device)
    encoder.eval()

    process_all_subjects(args.base_dir, extractor, encoder, device, args.encoder_output_dim)

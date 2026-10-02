import os
import numpy as np
import torch
import torchaudio
from transformers import AutoProcessor, AutoModel

class XSLR53FeatureExtractor:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
        self.model = AutoModel.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn").to(self.device)
        self.model.eval()

    def extract_embedding(self, wav_path, max_len_frames=None):
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)

        input_values = self.processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            outputs = self.model(input_values)
            embeddings = outputs.last_hidden_state.squeeze(0)  # (T, 1024)

        if max_len_frames is not None:
            embeddings = embeddings[:max_len_frames, :]
        return embeddings.cpu()

# Instantiate extractor once at module level
extractor = XSLR53FeatureExtractor()

def extract_embeddings_from_dir(audio_dir, subject_id=None, encoder=None):
    """
    Extract embeddings for all wav files in audio_dir.
    
    Args:
        audio_dir (str): folder containing wav files.
        subject_id (str): unused, just for compatibility.
        encoder (torch.nn.Module): optional encoder to further process embeddings.
        
    Returns:
        np.ndarray: concatenated embeddings from all wav files, shape (sum_T, embedding_dim).
    """
    all_embeddings = []
    for fname in sorted(os.listdir(audio_dir)):
        if fname.endswith(".wav"):
            wav_path = os.path.join(audio_dir, fname)
            emb = extractor.extract_embedding(wav_path)  # (T, 1024)
            if encoder is not None:
                # Add batch dim and encode; encoder expected to handle batch
                with torch.no_grad():
                    emb = encoder(emb.unsqueeze(0)).squeeze(0)  # shape depends on encoder output
            emb_np = emb.cpu().numpy()
            all_embeddings.append(emb_np)

    if not all_embeddings:
        # Return empty array if no wav files
        return np.empty((0, 1024))

    # Concatenate along time dimension (first dim)
    return np.concatenate(all_embeddings, axis=0)
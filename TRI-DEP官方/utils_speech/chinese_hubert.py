# chinese_hubert.py
import torch
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor, HubertPreTrainedModel, HubertModel

class HubertChineseExtractor:
    def __init__(self, model_name="xmj2002/hubert-base-ch-speech-emotion-recognition", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = AutoConfig.from_pretrained(model_name)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # Define a small wrapper model to get hidden states
        class HubertForSpeechEmbedding(HubertPreTrainedModel):
            def __init__(self, config):
                super().__init__(config)
                self.hubert = HubertModel(config)
                self.init_weights()

            def forward(self, input_values):
                outputs = self.hubert(input_values)
                hidden_states = outputs.last_hidden_state  # (B, T, hidden_size)
                return hidden_states
        
        self.model = HubertForSpeechEmbedding.from_pretrained(model_name, config=self.config)
        self.model.to(self.device).eval()

    def extract_embedding(self, wav_path, max_len_sec=5):
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if waveform.shape[1] < 1000:
            print(f"[Warning] {wav_path} seems too short.")

        input_values = self.processor(
            waveform.squeeze(0),
            sampling_rate=16000,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=16000 * max_len_sec
        ).input_values.to(self.device)

        with torch.no_grad():
            emb = self.model(input_values)  # (1, T, hidden_size)
        
        return emb.squeeze(0).cpu().numpy() # Return shape (T, hidden_size)

from encoder.cnn import AudioCNNEncoder
from encoder.cnn_lstm import CNNLSTMEncoder
from encoder.cnn_gru import CNNGRUEncoder
from encoder.cnn_bigru import BiGRUAttentionEncoder
from encoder.cnn_bilstm import AudioTemporalBiLSTMEncoder

def get_encoder(name):
    if name is None:
        return None

    name = name.lower()
    if name == "cnn":
        return AudioCNNEncoder().eval()
    elif name == "cnn_lstm":
        return CNNLSTMEncoder().eval()
    elif name == "cnn_gru":
        return CNNGRUEncoder().eval()
    elif name == "cnn_bigru":
        return BiGRUAttentionEncoder().eval()
    elif name == "cnn_bilstm":
        return AudioTemporalBiLSTMEncoder().eval()
    else:
        raise ValueError(f"Unknown encoder: {name}")

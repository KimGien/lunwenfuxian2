"""Pinned BIOT PREST-16 encoder on author-ordered bipolar raw voltages."""
from __future__ import annotations

import numpy as np
import torch

from ..protocol import BIOT_BIPOLAR, CHANNELS, CHANNEL_ALIASES
from ._author_support import author_module, parameter_names, raw_contract, require, select_trainability, verified_resources

FEATURE_DIM = 256
CANONICAL = tuple(CHANNEL_ALIASES.get(c.upper(), c.upper()) for c in CHANNELS)
PAIR_INDICES = tuple((CANONICAL.index(pair.split('-')[0]), CANONICAL.index(pair.split('-')[1])) for pair in BIOT_BIPOLAR)
INPUT_CONTRACT = {'raw': [19, 1000], 'sample_rate_hz': 200, 'physical_window_seconds': 5,
                  'model_input': [16, 1000], 'bipolar_order': list(BIOT_BIPOLAR),
                  'amplitude': 'first difference common-reference raw, then each bipolar channel / (95th percentile abs + 1e-8)',
                  'reference': 'author PREST-16 bipolar longitudinal montage',
                  'pretrained_window_seconds': 10, 'evaluation_window_difference': 'local full5s windows',
                  'pooling': 'author BIOTEncoder mean transformer sequence',
                  'calibrated_microvolts_confirmed': False}


def adapt_raw(raw):
    raw_contract(raw)
    values = np.stack([raw[:, a, :] - raw[:, b, :] for a, b in PAIR_INDICES], axis=1)
    denominator = np.quantile(np.abs(values), q=0.95, method='linear', axis=-1, keepdims=True) + 1e-8
    return np.ascontiguousarray(values / denominator, dtype=np.float32)


def exact_finetune_parameter_names(model):
    layers = model.transformer.layers.layers
    return parameter_names(model, [f'transformer.layers.layers.{len(layers)-1}'])


def load_encoder(device='cuda', trainable=False):
    checkpoint, audit = verified_resources('biot', 'EEG-PREST-16-channels.ckpt')
    module = author_module('biot', 'model', 'biot')
    config = dict(emb_size=256, heads=8, depth=4, n_channels=16, n_fft=200, hop_length=100)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = module.BIOTEncoder(**config)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    loaded = model.load_state_dict(state, strict=True)
    names = exact_finetune_parameter_names(model)
    model = select_trainability(model, trainable, names).to(device=device, dtype=torch.float32)
    audit.update({'configuration': config, 'strict': True, 'loaded_keys': len(state),
                  'missing_keys': loaded.missing_keys, 'unexpected_keys': loaded.unexpected_keys,
                  'feature_dim': FEATURE_DIM, 'parameters': sum(p.numel() for p in model.parameters()),
                  'input_contract': INPUT_CONTRACT, 'finetune_parameter_names': names,
                  'normalization_location': 'after bipolar raw difference, before encoder',
                  'neural_forward_calls_during_load': 0})
    return model, audit


def encode_windows(model, tensor):
    require(tensor.ndim == 3 and tuple(tensor.shape[1:]) == (16, 1000), 'BIOT expects [windows,16,1000]')
    output = model(tensor, n_channel_offset=0, perturb=False)
    require(tuple(output.shape) == (len(tensor), FEATURE_DIM) and bool(torch.isfinite(output).all()), 'Invalid BIOT features')
    return output

"""Pinned LUNA Base; physical resampling, author normalization and latent tokens."""
from __future__ import annotations

import numpy as np
import torch

from ..protocol import CHANNELS
from ._author_support import author_module, parameter_names, raw_contract, require, select_trainability, verified_resources

FEATURE_DIM = 256
INPUT_CONTRACT = {'raw': [19, 1000], 'raw_sample_rate_hz': 200, 'sample_rate_hz': 256,
                  'physical_window_seconds': 5, 'model_input': [19, 1280], 'patch_points': 40,
                  'amplitude': 'after physical resampling: author per-window per-channel z score; unbiased std + 1e-8',
                  'reference': 'preserve accepted E2v2 common reference',
                  'coordinates': 'author helper MNE standard_1005 nominal coordinates in meters; physical head coordinates, then author min-max internally',
                  'pooling': 'mean 32 temporal tokens after norm, preserve Q*D=4*64=256',
                  'calibrated_microvolts_confirmed': False}


def adapt_raw(raw):
    raw_contract(raw)
    from scipy.signal import resample_poly
    values = np.ascontiguousarray(resample_poly(raw.astype(np.float64), up=32, down=25, axis=-1,
                                               window=('kaiser', 5.), padtype='constant'), dtype=np.float32)
    # Match author ChannelWiseNormalize(torch.std default correction=1).
    value = torch.from_numpy(values)
    value = (value - value.mean(dim=-1, keepdim=True)) / (value.std(dim=-1, keepdim=True) + 1e-8)
    result = value.numpy()
    require(result.shape == (len(raw), 19, 1280) and bool(np.isfinite(result).all()), 'Invalid LUNA adapted inputs')
    return np.ascontiguousarray(result, dtype=np.float32)


def exact_finetune_parameter_names(model):
    # Pretrained query-unification queries are present; downstream classifier
    # learned_agg is absent from the formal pretrained checkpoint.
    return parameter_names(model, [f'blocks.{len(model.blocks)-1}', 'norm', 'cross_attn.query_embed'])


def load_encoder(device='cuda', trainable=False):
    checkpoint, audit = verified_resources('luna', 'LUNA_base.safetensors')
    module = author_module('luna', 'models', 'LUNA', torcheeg_constants=True)
    from safetensors.torch import load_file
    config = dict(patch_size=40, num_queries=4, embed_dim=64, depth=8, num_heads=2, drop_path=0., num_classes=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = module.LUNA(**config)
    state = load_file(str(checkpoint), device='cpu')
    loaded = model.load_state_dict(state, strict=True)
    # Use the author's coordinate helper on actual electrode names. It invokes
    # standard_1005 via MNE and preserves FCz/CPz identity.
    locations = module.ChannelEmbeddings.__init__.__globals__['get_channel_locations'](list(CHANNELS))
    coordinates = torch.tensor(np.stack(locations), dtype=torch.float32)
    require(tuple(coordinates.shape) == (19, 3) and bool(torch.isfinite(coordinates).all()), 'Invalid LUNA nominal coordinates')
    model.register_buffer('_eegall_coordinates', coordinates, persistent=False)
    names = exact_finetune_parameter_names(model)
    model = select_trainability(model, trainable, names).to(device=device, dtype=torch.float32)
    audit.update({'configuration': config, 'strict': True, 'loaded_keys': len(state),
                  'missing_keys': loaded.missing_keys, 'unexpected_keys': loaded.unexpected_keys,
                  'feature_dim': FEATURE_DIM, 'parameters': sum(p.numel() for p in model.parameters()),
                  'input_contract': INPUT_CONTRACT, 'coordinates': coordinates.tolist(), 'channel_names': list(CHANNELS),
                  'finetune_parameter_names': names, 'excluded_forward_heads': ['decoder_head', 'channel_emb'],
                  'neural_forward_calls_during_load': 0})
    return model, audit


def encode_windows(model, tensor):
    require(tensor.ndim == 3 and tuple(tensor.shape[1:]) == (19, 1280), 'LUNA expects [windows,19,1280]')
    coordinates = model._eegall_coordinates.unsqueeze(0).expand(len(tensor), -1, -1)
    x, _ = model.prepare_tokens(tensor, coordinates, mask=None)
    x, _ = model.cross_attn(x)
    x = x.reshape(len(tensor), 32, FEATURE_DIM)
    for block in model.blocks:
        x = block(x)
    output = model.norm(x).mean(dim=1)
    require(tuple(output.shape) == (len(tensor), FEATURE_DIM) and bool(torch.isfinite(output).all()), 'Invalid LUNA features')
    return output

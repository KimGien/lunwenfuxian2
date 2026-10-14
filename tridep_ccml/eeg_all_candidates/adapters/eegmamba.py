"""Official EEGMamba/Mamba2 adapter; requires the qualified Linux runtime."""
from __future__ import annotations

import importlib.metadata
import sys

import numpy as np
import torch

from ._author_support import author_module, parameter_names, raw_contract, require, select_trainability, verified_resources

FEATURE_DIM = 200
INPUT_CONTRACT = {'raw': [19, 1000], 'sample_rate_hz': 200, 'physical_window_seconds': 5,
                  'model_input': [19, 5, 200], 'amplitude': 'native accepted MNE-reader numeric /100 as author MODMA dataset',
                  'reference': 'preserve accepted E2v2 common reference',
                  'calibrated_microvolts_confirmed': False,
                  'pooling': 'author proj_out=Identity; mean channel and one-second token axes after encoder.norm_f',
                  'direction': 'author flips hidden_states and residual after every Mamba2 layer;12 alternating directions',
                  'full_paper_reproduction': False, 'experiment_label': 'author-public-implementation adaptation'}


def adapt_raw(raw):
    raw_contract(raw)
    return np.ascontiguousarray((raw / np.float32(100.)).reshape(len(raw), 19, 5, 200), dtype=np.float32)


def exact_finetune_parameter_names(model):
    return parameter_names(model, [f'encoder.layers.{len(model.encoder.layers)-1}', 'encoder.norm_f'])


def load_encoder(device='cuda', trainable=False):
    require(sys.platform == 'linux', 'EEGMamba requires the separately qualified official Linux Mamba2 runtime')
    checkpoint, audit = verified_resources('eegmamba', 'pretrained_EEGMamba.pth')
    module = author_module('eegmamba', 'models', 'eegmamba', sibling_packages=('modules',))
    config = dict(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = module.EEGMamba(**config)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    loaded = model.load_state_dict(state, strict=True)
    require(len(model.encoder.layers) == 12 and all(type(b.mixer).__name__ == 'Mamba2' for b in model.encoder.layers),
            'Formal EEGMamba must execute12 genuine official Mamba2 blocks')
    # Exact author downstream feature switch; the reconstruction linear was
    # first loaded strictly so its absence is explicit and never a random residue.
    model.proj_out = torch.nn.Identity()
    names = exact_finetune_parameter_names(model)
    model = select_trainability(model, trainable, names).to(device=device, dtype=torch.float32)
    audit.update({'configuration': config, 'ssm_configuration': {'layer': 'Mamba2', 'headdim': 50, 'd_state': 64},
                  'strict': True, 'loaded_keys_before_feature_switch': len(state),
                  'missing_keys': loaded.missing_keys, 'unexpected_keys': loaded.unexpected_keys,
                  'feature_dim': FEATURE_DIM, 'parameters': sum(p.numel() for p in model.parameters()),
                  'input_contract': INPUT_CONTRACT, 'finetune_parameter_names': names,
                  'official_mamba2_classes': [type(b.mixer).__module__ + '.' + type(b.mixer).__name__ for b in model.encoder.layers],
                  'runtime': {'python': sys.version, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                              'mamba_ssm': importlib.metadata.version('mamba_ssm'),
                              'causal_conv1d': importlib.metadata.version('causal_conv1d'),
                              'triton': importlib.metadata.version('triton')},
                  'excluded_feature_module': 'proj_out, strictly loaded reconstruction linear then Identity as author downstream',
                  'neural_forward_calls_during_load': 0})
    return model, audit


def encode_windows(model, tensor):
    require(tensor.is_cuda and tensor.ndim == 4 and tuple(tensor.shape[1:]) == (19, 5, 200),
            'Official EEGMamba Triton/Mamba2 forward requires CUDA [windows,19,5,200]')
    output = model(tensor).mean(dim=(1, 2))
    require(tuple(output.shape) == (len(tensor), FEATURE_DIM) and bool(torch.isfinite(output).all()), 'Invalid EEGMamba features')
    return output

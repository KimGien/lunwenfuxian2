"""Pinned EEGPT target encoder; the formal new readout keeps native D512."""
from __future__ import annotations

import numpy as np
import torch
from scipy.signal import resample_poly
from tridep_ccml.tridep_eegpt import assets
from .. import storage as S


def adapt_raw(raw):
    if raw.ndim != 3 or raw.shape[1:] != (19, 1000) or raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('EEGPT requires accepted common-reference raw [W,19,1000]')
    sampled = resample_poly(raw.astype(np.float64), 32, 25, axis=-1, window=('kaiser', 5.), padtype='constant')
    return np.ascontiguousarray(sampled[..., :1024] / 1000., dtype=np.float32)


def exact_finetune_parameter_names(model):
    prefix = f'blocks.{len(model.blocks)-1}.'
    return [name for name, parameter in model.named_parameters() if name.startswith(prefix)]


def load_encoder(device='cuda', trainable=False):
    model, audit = assets.load_frozen_eegpt(device)
    selected = exact_finetune_parameter_names(model)
    if trainable:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in selected)
    audit.update(native_dimension=512, lastblock_parameter_names=selected,
                 reference='unchanged E2v2 common reference; differs from average-reference pretraining',
                 amplitude='reader numeric microvolt convention /1000; physical calibration unconfirmed',
                 pooling='mean16 time tokens and4 summary tokens; no512-to200 projection',
                 input_shape=[19, 1024], frequency_hz=256, physical_seconds=4)
    audit['implementation_sources'] = [S.desc(assets.__file__), S.desc(assets.SOURCE)]
    audit['weight_sources'] = [S.desc(assets.CHECKPOINT)]
    return model, audit


def encode_windows(model, tensor):
    chan_ids = torch.as_tensor(assets.channel_ids(), device=tensor.device, dtype=torch.long)
    tokens = model(tensor, chan_ids=chan_ids)
    if tokens.ndim != 4 or tokens.shape[1:] != (16, 4, 512):
        raise ValueError('EEGPT official token output differs')
    return tokens.mean(dim=(1, 2))

"""Pinned original LaBraM base encoder, with matched four/five-second input."""
from __future__ import annotations

import numpy as np
from tridep_ccml.tridep_stage1 import labram_assets as A
from tridep_ccml.tridep_stage1 import inputs
from tridep_ccml.tridep_stage1.inputs import CHANNEL_NAMES
from .. import storage as S


def adapt_raw(raw, seconds=5):
    if seconds not in (4, 5) or raw.ndim != 3 or raw.shape[1:] != (19, 1000) or raw.dtype != np.float32:
        raise ValueError('LaBraM requires accepted19-channel raw and4/5 seconds')
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite LaBraM input')
    return np.ascontiguousarray(raw[..., :seconds*200] / np.float32(100.), dtype=np.float32)


def exact_finetune_parameter_names(model):
    prefix = f'blocks.{len(model.blocks)-1}.'
    return [name for name, parameter in model.named_parameters() if name.startswith(prefix) or name.startswith('norm.')]


def load_encoder(device='cuda', trainable=False):
    model, audit = A.load_frozen_labram(device)
    selected = exact_finetune_parameter_names(model)
    if trainable:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in selected)
    audit.update(native_dimension=200, lastblock_parameter_names=selected,
                 reference='unchanged accepted E2v2 common reference',
                 amplitude='reader numeric microvolt convention /100; physical calibration unconfirmed',
                 pooling='mean patch tokens excluding CLS; pretrained output norm retained')
    # The legacy loader reports bare hashes. The new fitting registry needs
    # actual hash-bound files so source/weight changes are detectable on resume.
    manifest = A.verify_assets()
    audit['weight_sources'] = [S.desc(A.CHECKPOINT)]
    audit['implementation_sources'] = [S.desc(A.__file__), S.desc(inputs.__file__), S.desc(A.MANIFEST)] + [
        S.desc(A.ROOT / row['path']) for row in manifest['files'] if row['path'].endswith('.py')]
    return model, audit


def encode_windows(model, tensor):
    if tensor.ndim != 3 or tensor.shape[1] != 19 or tensor.shape[2] not in (800, 1000):
        raise ValueError('LaBraM adapted shape differs')
    seconds = tensor.shape[-1] // 200
    tokens = model.forward_features(tensor.reshape(len(tensor), 19, seconds, 200),
             input_chans=A.channel_indices(CHANNEL_NAMES), return_patch_tokens=True, return_all_tokens=False)
    if tuple(tokens.shape[1:]) != (19*seconds, 200):
        raise ValueError('LaBraM official token output differs')
    return tokens.mean(dim=1)

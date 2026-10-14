"""Pinned author CNN/attention backbone, with registered 19-channel task adaptation."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from einops.layers.torch import Rearrange

ROOT = Path(__file__).resolve().parents[1] / 'vendor/eegconformer'
SOURCE = ROOT / 'conformer.py'
SOURCE_SHA256 = '866ea540a20bf158dc22450ff393c136def016bcf75ee3d3d646d24161f6954c'
FEATURE_DIM = 40
DEFAULT_CONFIG = {'emb_size': 40, 'depth': 6, 'num_heads': 10, 'dropout': .5,
                  'temporal_kernel': 25, 'spatial_channels': 19, 'pool_kernel': 75,
                  'pool_stride': 15, 'tokens_per_window': 61}
INPUT_CONTRACT = {'raw': [19, 1000], 'sample_rate_hz': 200, 'physical_window_seconds': 5,
                  'model_input': [1, 19, 1000], 'native_dimension': 40,
                  'reference': 'preserve accepted E2v2 common reference',
                  'normalization': 'one scalar mean/std fitted only on this outer fold train windows',
                  'pooling': 'mean of the 61 author convolution/attention tokens; retain subject window axis',
                  'pretrained_checkpoint': None, 'training': 'entire author CNN and six Transformer blocks from scratch'}
_CLASSES = None


def require(value, message):
    if not value:
        raise ValueError(message)


def author_classes():
    """Extract only hash-bound class definitions; never execute the author script."""
    global _CLASSES
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256, 'Conformer author source changed')
    if _CLASSES is not None:
        return _CLASSES
    allow = ('PatchEmbedding', 'MultiHeadAttention', 'ResidualAdd', 'FeedForwardBlock',
             'GELU', 'TransformerEncoderBlock', 'TransformerEncoder')
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in allow]
    require(tuple(node.name for node in classes) == allow, 'Conformer author class allowlist differs')
    changed = 0
    for node in ast.walk(classes[0]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'Conv2d':
            if len(node.args) >= 3 and isinstance(node.args[2], ast.Tuple):
                values = node.args[2].elts
                if len(values) == 2 and all(isinstance(v, ast.Constant) for v in values) and [v.value for v in values] == [22, 1]:
                    values[0].value = 19
                    changed += 1
    require(changed == 1, 'Conformer adaptation must change exactly one spatial kernel')
    namespace = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional, 'Tensor': torch.Tensor,
                 'rearrange': rearrange, 'Rearrange': Rearrange, 'math': math}
    program = ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[]))
    exec(compile(program, str(SOURCE) + ':19-channel-classes', 'exec'), namespace)
    _CLASSES = {name: namespace[name] for name in allow}
    return _CLASSES


def fit_train_statistics(raw, subject_ids, train_ids):
    """The caller must pass only the named original outer-train subject arrays."""
    require(isinstance(raw, np.ndarray) and raw.dtype == np.float32 and raw.ndim == 4
            and raw.shape[1:] == (60, 19, 1000) and np.isfinite(raw).all(), 'Expected outer-train [people,60,19,1000]')
    require(list(subject_ids) == list(train_ids) and len(subject_ids) == len(raw)
            and len(set(subject_ids)) == len(subject_ids), 'Normalizer input must exactly match declared train people')
    mean, std = float(np.mean(raw, dtype=np.float64)), float(np.std(raw, dtype=np.float64))
    require(math.isfinite(mean) and math.isfinite(std) and std > 0, 'Invalid train-only normalization statistics')
    return {'mean': mean, 'std': std, 'ddof': 0, 'fit_subject_ids': list(train_ids), 'scope': 'outer-train scalar across channels/windows/time'}


def adapt_raw(raw, statistics):
    require(isinstance(raw, np.ndarray) and raw.dtype == np.float32 and raw.ndim == 3
            and raw.shape[1:] == (19, 1000) and len(raw) > 0 and np.isfinite(raw).all(), 'Expected finite E2v2 float32 [W,19,1000]')
    require(statistics.get('fit_subject_ids') and statistics.get('ddof') == 0
            and math.isfinite(statistics['mean']) and math.isfinite(statistics['std']) and statistics['std'] > 0,
            'Conformer requires frozen outer-train normalization statistics')
    return np.ascontiguousarray(((raw.astype(np.float64) - statistics['mean']) / statistics['std'])[:, None], dtype=np.float32)


class ConformerEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        classes = author_classes()
        self.patch = classes['PatchEmbedding'](40)
        self.transformer = classes['TransformerEncoder'](6, 40)

    def forward(self, tensor):
        require(tensor.ndim == 4 and tuple(tensor.shape[1:]) == (1, 19, 1000), 'Conformer expects [W,1,19,1000]')
        tokens = self.transformer(self.patch(tensor))
        require(tuple(tokens.shape) == (len(tensor), 61, 40), 'Conformer author token shape differs')
        return tokens.mean(dim=1)


def load_encoder(device='cuda', trainable=True):
    require(trainable, 'Conformer requires full supervised training; a frozen random backbone is not a completed route')
    model = ConformerEncoder().to(device=device, dtype=torch.float32)
    audit = json.loads((ROOT / 'author_source_binding.json').read_text(encoding='utf-8'))
    return model, {'source': {'repository': audit['repository'], 'revision': audit['revision'], 'sha256': SOURCE_SHA256},
                   'configuration': DEFAULT_CONFIG, 'input_contract': INPUT_CONTRACT,
                   'pretrained_checkpoint': None, 'feature_dim': FEATURE_DIM,
                   'trainable_parameter_names': [n for n, _ in model.named_parameters()],
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'runtime_source_adaptations': ['one author spatial Conv2d kernel 22→19 channels',
                                                 'author classification head replaced by registered token-mean subject readout'],
                   'neural_forward_calls_during_load': 0}


def encode_windows(model, tensor):
    return model(tensor)

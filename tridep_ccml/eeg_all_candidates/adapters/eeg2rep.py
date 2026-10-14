"""Author EEG2Rep cores with explicit SSP, independent target EMA and native readout.

This adapter has no external pretrained weight. `build_model` starts each fold's
registered self-supervised run; the caller must keep learned states fold-local.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import types

import numpy as np
import torch
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1] / 'vendor/eeg2rep'
SOURCE_SHA256 = {'Models/Attention.py': '8d639ea0b03b0d1fd4fe5cd36d3e67b23345e853aef8ca0c939bda8c868c2074',
                 'Models/model.py': 'aa0a3e28e167f5bf16b8c26458ccfcc2d269e44066d06e1d73efc34d89ed2f88',
                 'trainer.py': 'c14a65487a362e133689bd7fa4ce258d7a1270ead9600b60f247800eca816185'}
FEATURE_DIM = 16
DEFAULT_CONFIG = {'Data_shape': [0, 19, 1000], 'num_labels': 2, 'emb_size': 16,
                  'num_heads': 8, 'dim_ff': 256, 'layers': 4, 'pre_layers': 2,
                  'dropout': .1, 'mask_ratio': .5, 'momentum': .99, 'pooling_size': 2,
                  'device': 'cpu'}
SSL_CONTRACT = {'logical_window_batch': 64, 'token_count': 500, 'ssp_chunk_count': 2,
                'ssp_visible_ratio': .5, 'ssp_chunk_length': 125,
                'ssp_overlap': 'author two independently sampled chunks may overlap; duplicate visible indices retained',
                'target_ema_momentum': .99, 'ema_updates': 'once per actual logical optimizer step; never in validation',
                'target_initialization': 'copy online values once; independent storage; target always requires_grad=False',
                'target_train_dropout': 'author target_encoder follows train/eval module mode; target forward under no_grad',
                'loss_coefficients': {'alignment_mse': 1., 'variance_hinge': 1., 'offdiagonal_covariance': 1.},
                'variance_epsilon': 1e-4, 'variance_estimator': 'unbiased=True across logical batch',
                'variance_target': 1., 'covariance_denominator': 'logical_batch_size-1',
                'covariance_dimension_normalizer': FEATURE_DIM,
                'variance_covariance_representation': 'masked prediction token mean per window, after differentiable logical-batch assembly',
                'minimum_logical_batch': 2,
                'validation': 'fixed explicit SSP masks, eval, no optimizer or EMA; restore all training RNG',
                'input_normalization': 'author main default Norm=False; retain accepted native common-reference raw',
                'architecture': 'InputEmbedding→Norm→sinusoidal positions→Norm2→4 context Transformer blocks; predictor 2 cross-attention blocks'}
AUTHOR_DEFECT_REPAIRS = [
    'copy_weight author .data assignment aliases online/target storage; copy_ once replaces aliasing and per-epoch reset',
    'linear_prob author no_grad disables supervision; native representation is recomputed with gradients in supervised stage',
    'linear_prob/loss unconditional squeeze removes batch-one axis; explicit mean(dim=1) preserves window batch axis',
    'author trainer probes test labels every five epochs; adapter exposes no author trainer/main or external test evaluation',
]
_AUTHOR = None


def require(value, message):
    if not value:
        raise ValueError(message)


def author_definitions():
    global _AUTHOR
    for name, expected in SOURCE_SHA256.items():
        require(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, 'EEG2Rep author source changed: ' + name)
    if _AUTHOR is not None:
        return _AUTHOR
    namespace = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional, 'np': np,
                 'math': math, 'random': random, 'copy': copy, 'rearrange': rearrange}
    for filename in ('Models/Attention.py', 'Models/model.py'):
        tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
        definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(ROOT / filename) + ':definitions-only', 'exec'), namespace)
    _AUTHOR = namespace
    return namespace


def adapt_raw(raw):
    require(isinstance(raw, np.ndarray) and raw.dtype == np.float32 and raw.ndim == 3
            and raw.shape[1:] == (19, 1000) and len(raw) > 0 and np.isfinite(raw).all(), 'EEG2Rep expects finite accepted E2v2 [W,19,1000]')
    return np.ascontiguousarray(raw, dtype=np.float32)


def make_ssp_mask(token_count=500, *, rng=None):
    """Run the unchanged author SSP function against an optional independent RNG."""
    require(token_count == SSL_CONTRACT['token_count'], 'Registered EEG2Rep token count differs')
    author = author_definitions()['Semantic_Subsequence_Preserving']
    local_globals = dict(author.__globals__)
    local_globals['random'] = random if rng is None else rng
    ssp = types.FunctionType(author.__code__, local_globals)
    chunks = ssp(np.arange(token_count), 2, .5)
    visible = np.ravel(chunks).astype(np.int64)
    masked = np.setdiff1d(np.arange(token_count), visible).astype(np.int64)
    return {'token_count': token_count, 'visible_indices': visible.tolist(), 'masked_indices': masked.tolist(),
            'chunk_indices': [np.asarray(c, dtype=np.int64).tolist() for c in chunks]}


class EEG2RepEncoder(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = copy.deepcopy(DEFAULT_CONFIG if config is None else config)
        # All non-device author hyperparameters are fixed before formal fitting.
        require({k: v for k, v in self.config.items() if k != 'device'} == {k: v for k, v in DEFAULT_CONFIG.items() if k != 'device'},
                'EEG2Rep formal non-optimization hyperparameters differ')
        self.core = author_definitions()['EEG2Rep'](self.config, num_classes=2)
        self.initialize_target()
        self.set_stage('ssl')

    @torch.no_grad()
    def initialize_target(self):
        self.core.target_encoder.load_state_dict(self.core.contex_encoder.state_dict(), strict=True)
        self.core.target_encoder.requires_grad_(False)
        require(all(t.data_ptr() != p.data_ptr() for t, p in zip(self.core.target_encoder.parameters(), self.core.contex_encoder.parameters())),
                'EEG2Rep target and online parameters share storage')

    def set_stage(self, stage):
        require(stage in ('ssl', 'supervised'), 'Unknown EEG2Rep training stage')
        self.stage = stage
        self.core.requires_grad_(False)
        for module in (self.core.InputEmbedding, self.core.Norm, self.core.Norm2, self.core.contex_encoder):
            module.requires_grad_(True)
        if stage == 'ssl':
            self.core.Predictor.requires_grad_(True)
            self.core.mask_token.requires_grad_(True)
        # The original window classification layer is replaced by the subject head.
        self.core.predict_head.requires_grad_(False)
        self.core.target_encoder.requires_grad_(False)

    def embed(self, tensor):
        require(tensor.ndim == 3 and tuple(tensor.shape[1:]) == (19, 1000), 'EEG2Rep expects [W,19,1000]')
        patches = self.core.Norm(self.core.InputEmbedding(tensor))
        return self.core.Norm2(patches + self.core.PositionalEncoding(patches))

    def forward(self, tensor):
        return self.core.contex_encoder(self.embed(tensor)).mean(dim=1)

    def pretrain_forward(self, tensor, *, mask=None, mask_rng=None):
        require(self.stage == 'ssl', 'Pretraining called outside EEG2Rep SSL stage')
        patches = self.embed(tensor)
        mask = make_ssp_mask(patches.shape[1], rng=mask_rng) if mask is None else mask
        require(mask['token_count'] == patches.shape[1], 'SSP mask token count differs')
        visible = np.asarray(mask['visible_indices'], dtype=np.int64)
        masked = np.asarray(mask['masked_indices'], dtype=np.int64)
        require(len(visible) == 250 and len(masked) > 0 and np.all((visible >= 0) & (visible < 500))
                and np.array_equal(masked, np.setdiff1d(np.arange(500), visible)), 'Invalid registered SSP visible/masked indices')
        vi = torch.as_tensor(visible, dtype=torch.long, device=tensor.device)
        mi = torch.as_tensor(masked, dtype=torch.long, device=tensor.device)
        tokens = self.core.mask_token.repeat(patches.shape[0], patches.shape[1], 1)
        tokens = tokens + self.core.PositionalEncoding(tokens)
        context = self.core.contex_encoder(patches[:, vi])
        with torch.no_grad():
            target = self.core.target_encoder(patches)
            target_masked = target[:, mi]
        prediction = self.core.Predictor(context, tokens[:, mi])
        return {'target_masked': target_masked, 'prediction_masked': prediction,
                'context': context, 'target_full': target, 'mask': mask}

    @torch.no_grad()
    def momentum_update(self):
        require(self.stage == 'ssl', 'EEG2Rep EMA must only run after an SSL optimizer step')
        for target, online in zip(self.core.target_encoder.parameters(), self.core.contex_encoder.parameters()):
            require(target.data_ptr() != online.data_ptr(), 'EMA target aliases online storage')
            target.mul_(.99).add_(online, alpha=.01)


def ssl_loss(output):
    """Author MSE + predicted-token variance + covariance on one logical batch."""
    target, predicted = output['target_masked'], output['prediction_masked']
    require(predicted.ndim == 3 and predicted.shape == target.shape and len(predicted) >= 2
            and predicted.shape[-1] == FEATURE_DIM, 'SSL loss needs a complete logical batch of at least two windows')
    # Retain a float32 loss/statistics path even if the caller uses bfloat16 features.
    predicted, target = predicted.float(), target.detach().float()
    alignment = torch.nn.functional.mse_loss(predicted, target)
    centered = predicted.mean(dim=1)
    centered = centered - centered.mean(dim=0)
    deviation = torch.sqrt(centered.var(dim=0, unbiased=True) + 1e-4)
    variance = torch.nn.functional.relu(1. - deviation).mean()
    covariance = centered.T @ centered / (len(centered) - 1)
    size = covariance.shape[0]
    off_diagonal = covariance.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()
    cov_loss = off_diagonal.square().sum() / size
    total = alignment + variance + cov_loss
    require(bool(torch.isfinite(total)), 'Nonfinite EEG2Rep self-supervised loss')
    return {'loss': total, 'alignment_mse': alignment, 'variance_hinge': variance,
            'offdiagonal_covariance': cov_loss, 'prediction_std_mean': deviation.mean()}


def build_model(device='cpu', *, stage='ssl'):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config['device'] = str(device)
    model = EEG2RepEncoder(config).to(device=device, dtype=torch.float32)
    model.set_stage(stage)
    binding = json.loads((ROOT / 'author_source_binding.json').read_text(encoding='utf-8'))
    audit = {'source': {'repository': binding['repository'], 'revision': binding['revision'], 'files_sha256': SOURCE_SHA256},
             'configuration': config, 'ssl_contract': SSL_CONTRACT, 'author_defect_repairs': AUTHOR_DEFECT_REPAIRS,
             'stage': stage, 'feature_dim': FEATURE_DIM, 'pretrained_checkpoint': None,
             'requires_fold_local_ssl_training': True, 'parameters': sum(p.numel() for p in model.parameters()),
             'trainable_parameter_names': [n for n, p in model.named_parameters() if p.requires_grad],
             'frozen_parameter_names': [n for n, p in model.named_parameters() if not p.requires_grad],
             'neural_forward_calls_during_build': 0}
    return model, audit


def load_encoder(device='cuda', trainable=True):
    require(trainable, 'EEG2Rep random backbone cannot be used as a frozen pretrained baseline; load a fold-local learned checkpoint explicitly')
    return build_model(device, stage='supervised')


def encode_windows(model, tensor):
    output = model(tensor)
    require(tuple(output.shape) == (len(tensor), FEATURE_DIM) and bool(torch.isfinite(output).all()), 'Invalid native EEG2Rep representation')
    return output

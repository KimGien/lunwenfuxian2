"""Pinned CodeBrain EEGSSM feature backbone with device-portable author SWA."""
from __future__ import annotations

import ast
import inspect
import textwrap
import types

import numpy as np
import torch

from ._author_support import author_module, parameter_names, raw_contract, require, select_trainability, verified_resources

FEATURE_DIM = 200
INPUT_CONTRACT = {'raw': [19, 1000], 'sample_rate_hz': 200, 'physical_window_seconds': 5,
                  'model_input': [19, 5, 200], 'amplitude': 'native MNE-reader numeric / 100',
                  'reference': 'preserve accepted E2v2 common reference',
                  'calibrated_microvolts_confirmed': False,
                  'pooling': 'mean channel and one-second patch axes after author output LayerNorm',
                  'architecture': 'SSSM EEGSSM; not tokenizer or NeuralTransformer'}


def adapt_raw(raw):
    raw_contract(raw)
    return np.ascontiguousarray((raw / np.float32(100.)).reshape(len(raw), 19, 5, 200), dtype=np.float32)


def _portable_author_forward(module, block):
    """The sole runtime rewrite replaces `.cuda()` by `.to(h_s.device)`.

    This does not change the mask, attention, convolution, or EEGSSM equations.
    It avoids the author's forced default CUDA device during CPU audits.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(module.Residual_block.forward)))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'cuda':
            require(not node.args and not node.keywords, 'Unexpected author CUDA call')
            node.func.attr = 'to'
            node.args = [ast.Attribute(value=ast.Name(id='h_s', ctx=ast.Load()), attr='device', ctx=ast.Load())]
            changed += 1
    require(changed == 1, 'Author portable-device rewrite must affect exactly one CUDA call')
    context = dict(vars(module))
    exec(compile(ast.fix_missing_locations(tree), str(module.__file__) + ':device-adapter', 'exec'), context)
    block.forward = types.MethodType(context['forward'], block)


def exact_finetune_parameter_names(model):
    last = len(model.residual_layer.residual_blocks) - 1
    return parameter_names(model, [f'residual_layer.residual_blocks.{last}', 'norm'])


def load_encoder(device='cuda', trainable=False):
    checkpoint, audit = verified_resources('codebrain', 'CodeBrain.pth')
    module = author_module('codebrain', 'Models', 'SSSM')
    config = dict(in_channels=200, res_channels=200, skip_channels=200, out_channels=200,
                  num_res_layers=8, diffusion_step_embed_dim_in=200,
                  diffusion_step_embed_dim_mid=200, diffusion_step_embed_dim_out=200,
                  s4_lmax=570, s4_d_state=64, s4_dropout=0.1, s4_bidirectional=True,
                  s4_layernorm=True, codebook_size_t=4096, codebook_size_f=4096, if_codebook=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = module.SSSM(**config)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = {name.removeprefix('module.'): tensor for name, tensor in payload.items()}
    loaded = model.load_state_dict(state, strict=True)
    for block in model.residual_layer.residual_blocks:
        _portable_author_forward(module, block)
    names = exact_finetune_parameter_names(model)
    model = select_trainability(model, trainable, names).to(device=device, dtype=torch.float32)
    audit.update({'configuration': config, 'strict': True, 'loaded_keys': len(state),
                  'missing_keys': loaded.missing_keys, 'unexpected_keys': loaded.unexpected_keys,
                  'feature_dim': FEATURE_DIM, 'parameters': sum(p.numel() for p in model.parameters()),
                  'input_contract': INPUT_CONTRACT, 'finetune_parameter_names': names,
                  'runtime_source_adaptation': 'Residual_block.forward: one mask.cuda() -> mask.to(h_s.device) per block; source bytes unchanged',
                  'excluded_forward_heads': ['lm_head_t', 'lm_head_f'], 'neural_forward_calls_during_load': 0})
    return model, audit


def encode_windows(model, tensor):
    require(tensor.ndim == 4 and tuple(tensor.shape[1:]) == (19, 5, 200), 'CodeBrain expects [windows,19,5,200]')
    # Author SSSM.squeeze removes batch axis when W=1; restore it explicitly.
    tokens = model(tensor).reshape(len(tensor), 19, 5, FEATURE_DIM)
    output = tokens.mean(dim=(1, 2))
    require(tuple(output.shape) == (len(tensor), FEATURE_DIM) and bool(torch.isfinite(output).all()), 'Invalid CodeBrain features')
    return output

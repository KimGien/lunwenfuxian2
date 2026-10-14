"""Explicit CodeBrain repair: initialize author weight-only caches before freeze.

The locked v1 adapter and author snapshots remain unchanged. The author GConv
normalizes its full constructed kernel before cropping to the signal length,
so this initialization depends solely on formal pretrained weights.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
from pathlib import Path
import textwrap

import torch

from . import codebrain as V1
from ._author_support import WORK, digest, require, select_trainability

FEATURE_DIM = V1.FEATURE_DIM
INPUT_CONTRACT = copy.deepcopy(V1.INPUT_CONTRACT)
INPUT_CONTRACT['adapter_version'] = 2
INPUT_CONTRACT['kernel_normalization_cache'] = 'author full-kernel norm computed solely from official weights before first forward, before sequence cropping'
DEAD_SUFFIXES = ('sn.scale', 'res_conv.bias', 'res_conv.weight_g', 'res_conv.weight_v')


def adapt_raw(raw):
    return V1.adapt_raw(raw)


def _author_weight_kernel(gconv):
    """Compile the unchanged kernel-construction statements from author forward.

    It excludes data handling, buffer mutation, FFT, and all neural layers.
    """
    original = gconv.__class__.forward
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    body = tree.body[0].body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'kernel_list' for target in node.targets))
    stop = next(i for i, node in enumerate(body) if isinstance(node, ast.If)
                and isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Attribute) and node.test.operand.attr == 'kernel_norm_initialized')
    require(start < stop, 'Author kernel/cache boundary differs')
    function = ast.FunctionDef(name='_author_weight_kernel_formula',
                              args=ast.arguments(posonlyargs=[], args=[ast.arg(arg='self')], vararg=None,
                                                 kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                              body=copy.deepcopy(body[start:stop]) + [ast.Return(value=ast.Name(id='k', ctx=ast.Load()))],
                              decorator_list=[])
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 original.__code__.co_filename + ':weight-only-kernel-formula', 'exec'), namespace)
    return namespace['_author_weight_kernel_formula'](gconv)


def exact_finetune_parameter_names(model):
    last = len(model.residual_layer.residual_blocks) - 1
    excluded = {f'residual_layer.residual_blocks.{last}.{suffix}' for suffix in DEAD_SUFFIXES}
    original = V1.exact_finetune_parameter_names(model)
    require(excluded.issubset(set(original)), 'Author dead last-residual branch names changed')
    return [name for name in original if name not in excluded]


def load_encoder(device='cuda', trainable=False):
    model, audit = V1.load_encoder(device=device, trainable=False)
    initialized = []
    with torch.no_grad():
        for index, block in enumerate(model.residual_layer.residual_blocks):
            gconv = block.S41
            old_flag = bool(gconv.kernel_norm_initialized)
            kernel = _author_weight_kernel(gconv)
            norm = kernel.norm(dim=-1, keepdim=True).detach()
            require(norm.shape == gconv.kernel_norm.shape and bool(torch.isfinite(norm).all())
                    and bool((norm > 0).all()), 'Invalid author weight-only GConv norm')
            if old_flag:
                require(torch.equal(gconv.kernel_norm, norm), 'Already initialized official cache differs from author formula')
            else:
                gconv.kernel_norm = norm
                gconv.kernel_norm_initialized = torch.tensor(1, dtype=torch.bool, device=kernel.device)
            initialized.append({'block': index, 'official_checkpoint_cache_was_initialized': old_flag,
                                'kernel_shape': list(kernel.shape), 'norm_shape': list(norm.shape),
                                'norm_sha256': hashlib.sha256(norm.cpu().contiguous().numpy().tobytes()).hexdigest()})
    names = exact_finetune_parameter_names(model)
    select_trainability(model, trainable, names)
    last = len(model.residual_layer.residual_blocks) - 1
    audit.update({'adapter_version': 2, 'input_contract': INPUT_CONTRACT,
                  'repair_reason': 'author checkpoint has8 uninitialized normalization caches; last residual branch is unused by skip-only output',
                  'finetune_parameter_names': names,
                  'excluded_dead_lastblock_parameters': [f'residual_layer.residual_blocks.{last}.{suffix}' for suffix in DEAD_SUFFIXES],
                  'fixed5s_zero_signal_gradient_kernel_parameters': [f'residual_layer.residual_blocks.{last}.S41.kernel_list.{i}.kernel' for i in (7, 8, 9)],
                  'weight_only_cache_initialization': initialized,
                  'weight_only_kernel_formula_calls_during_load': len(initialized),
                  'neural_forward_calls_during_load': 0,
                  'initialization_uses_cohort_data': False,
                  'v1_adapter_source_preserved': str(Path(V1.__file__)),
                  'implementation_sources': [{'path': Path(path).resolve().relative_to(WORK).as_posix(), 'sha256': digest(path)}
                                             for path in (V1.__file__, __file__, Path(__file__).parent / '_author_support.py')],
                  'source_modified': False})
    return model, audit


def encode_windows(model, tensor):
    return V1.encode_windows(model, tensor)

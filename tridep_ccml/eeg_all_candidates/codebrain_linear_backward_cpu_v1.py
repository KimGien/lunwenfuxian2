"""Unintegrated CodeBrain repair prototype; native forward, CPU native derivative.

Only SGConv's FP32 linear interpolation with positive power-of-two scaling is
supported. The author forward remains torch.nn.functional.interpolate on the
input device. Its derivative uses the same native ATen linear1d backward on CPU
and copies the result back. No determinism flags are changed. Host transfers
and synchronization are a real cost, not a performance optimization.

Importing this module does not install it into any model or running queue.
Actual GPU/model qualification and a new reviewed execution lineage are required.
"""
from types import SimpleNamespace
import math

import torch
import torch.nn.functional as F

_NATIVE_INTERPOLATE = F.interpolate


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _factor(value):
    require(type(value) in (int, float) and math.isfinite(value)
            and value >= 1 and float(value).is_integer(), 'Integer upscale required')
    number = int(value)
    require(number & (number - 1) == 0, 'SGConv power-of-two scaling required')
    return float(number)


class NativeForwardCpuBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        require(value.ndim == 3 and value.dtype == torch.float32,
                'Only original 3D FP32 SGConv kernel supported')
        factor = _factor(scale)
        result = _NATIVE_INTERPOLATE(value, scale_factor=factor,
                                     mode='linear', align_corners=False)
        ctx.input_shape = tuple(value.shape)
        ctx.output_width = result.shape[-1]
        ctx.factor = factor
        ctx.input_device = value.device
        return result

    @staticmethod
    def backward(ctx, gradient):
        require(gradient.ndim == 3 and gradient.dtype == torch.float32,
                'Native FP32 gradient required')
        # Never invoke the CUDA atomic-add backward or disable strict algorithms.
        host = gradient.detach().to(device='cpu')
        result = torch.ops.aten.upsample_linear1d_backward.default(
            host, [ctx.output_width], list(ctx.input_shape), False, ctx.factor)
        return result.to(device=ctx.input_device), None


def interpolate(value, size=None, scale_factor=None, mode='nearest',
                align_corners=None, recompute_scale_factor=None, antialias=False):
    if mode != 'linear':
        return _NATIVE_INTERPOLATE(value, size=size, scale_factor=scale_factor,
                                  mode=mode, align_corners=align_corners,
                                  recompute_scale_factor=recompute_scale_factor,
                                  antialias=antialias)
    require(size is None and align_corners in (None, False)
            and recompute_scale_factor is None and antialias is False,
            'Only the exact author SGConv interpolation call is supported')
    return NativeForwardCpuBackward.apply(value, _factor(scale_factor))


def functional_namespace():
    """Return a private namespace; caller must not mutate global functional."""
    return SimpleNamespace(**{**vars(F), 'interpolate': interpolate})

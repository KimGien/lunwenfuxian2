"""Private last-axis FP32 prefix sums for SSD backward, pending GPU qualification.

The installed Triton backward uses CUDA cumsum(out=the same storage), which
fails under strict deterministic algorithms. This prototype copies that
intermediate to CPU, adds in a fixed left-to-right FP32 order, and writes the
same view back. The core reverse-scan call also supports out=None and returns
new storage on the input device. This changes accumulation order; it is NOT bitwise equality
evidence against the unavailable deterministic CUDA implementation.

No installed package/global torch function is changed on import. Private
backward-function wiring, real Linux GPU/full-model qualification, costs and
new formal execution/audit/evaluation/final source binding remain mandatory.
"""
from types import SimpleNamespace
import torch

_NATIVE_CUMSUM=torch.cumsum

def require(ok,message):
    if not ok: raise ValueError(message)

@torch.no_grad()
def cumsum(value,dim,*,dtype=None,out=None):
    if out is None:
        require(isinstance(value,torch.Tensor) and value.dtype==torch.float32
                and dtype in (None,torch.float32), 'New-result scan keeps FP32 operands and output')
        require(value.ndim>0 and type(dim) is int and dim in (-1,value.ndim-1)
                and value.shape[-1]>0 and value.numel()>0, 'Nonempty last-axis scan required')
        require(not value.requires_grad, 'Backward intermediate only; higher-order autograd unsupported')
        host=value.detach().to(device='cpu').contiguous().clone()
        result=torch.empty_like(host)
        running=torch.zeros_like(host[...,0])
        for i in range(host.shape[-1]):
            running=running+host[...,i]
            result[...,i].copy_(running)
        return result.to(device=value.device)
    require(isinstance(value,torch.Tensor) and isinstance(out,torch.Tensor),
            'Only explicit same-storage out= backward intermediate supported')
    require(value.dtype==out.dtype==torch.float32 and dtype in (None,torch.float32),
            'Keep FP32 operands and output')
    require(value.ndim>0 and type(dim) is int and dim in (-1,value.ndim-1)
            and value.shape[-1]>0 and value.numel()>0,'Nonempty last-axis scan required')
    require(not value.requires_grad and not out.requires_grad,
            'Backward intermediate only; higher-order autograd unsupported')
    require(value.device==out.device and value.shape==out.shape and value.stride()==out.stride()
            and value.storage_offset()==out.storage_offset() and value.data_ptr()==out.data_ptr(),
            'The original input/output view must alias exactly')
    host=value.detach().to(device='cpu').contiguous().clone()
    result=torch.empty_like(host)
    running=torch.zeros_like(host[...,0])
    for i in range(host.shape[-1]):
        # Both operands/results remain FP32; no FP64/promoted accumulator.
        running=running+host[...,i]
        result[...,i].copy_(running)
    out.copy_(result.to(device=out.device))
    return out

def torch_namespace():
    return SimpleNamespace(**{**vars(torch),'cumsum':cumsum})

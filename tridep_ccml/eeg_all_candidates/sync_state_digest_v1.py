"""Qualification-only prototype for the exact existing state-digest byte stream.

This module does not import torch, load a checkpoint, or change a training
function on import. Homogeneous CUDA states can be packed as uint8 bytes on
their existing device and copied to the CPU once. The original sorted name,
dtype, shape, and logical C-order bytes are then hashed unchanged.

The prototype is NOT a production qualification. A caller must compare it
with the sealed original on the actual device/state, check state/RNG/modes,
and measure wall time plus peak memory before source-bound integration.
Packing allocates an additional byte buffer; allocation/copy failures are
propagated, never silently retried as an unrecorded fallback.
"""
from __future__ import annotations

import hashlib

DEFAULT_MAX_PACK_BYTES = 256 * 1024 * 1024
NUMPY_DTYPES = ("bool", "uint8", "int8", "int16", "int32", "int64",
                "float16", "float32", "float64", "complex64", "complex128")


class UnsupportedState(ValueError):
    """Preflight rejected the state before a packing/copy operation occurred."""


def _emit(observer, event, **values):
    if observer is not None:
        observer({"component": "sync_state_digest_v1", "event": event, **values})


def _torch(torch_module):
    if torch_module is None:
        import torch
        return torch
    return torch_module


def _plan(state, torch, *, allow_cpu, max_pack_bytes):
    """Metadata-only preflight: no tensor copy, arithmetic, or scalar read."""
    if type(max_pack_bytes) is not int or max_pack_bytes < 0:
        raise ValueError("max_pack_bytes must be a nonnegative integer")
    if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
        raise UnsupportedState("Expected a string-keyed state dictionary")
    if not state:
        return [], 0, None
    allowed = {getattr(torch, name) for name in NUMPY_DTYPES if hasattr(torch, name)}
    entries, total, device = [], 0, None
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise UnsupportedState("Only dense strided tensor values are eligible")
        if tensor.dtype not in allowed or tensor.is_quantized:
            raise UnsupportedState("Dtype is not in the original NumPy-compatible subset")
        if tensor.is_conj() or tensor.is_neg():
            raise UnsupportedState("Unresolved conjugate/negative view bits require original behavior")
        kind = tensor.device.type
        if kind != "cuda" and not (allow_cpu and kind == "cpu"):
            raise UnsupportedState("CUDA state required; CPU packing is synthetic-test-only")
        if device is not None and tensor.device != device:
            raise UnsupportedState("All tensors must reside on one device")
        device = tensor.device
        size = tensor.numel() * tensor.element_size()
        entries.append({"name": name, "dtype": str(tensor.dtype),
                        "shape": tuple(tensor.shape), "offset": total,
                        "nbytes": size, "tensor": tensor})
        total += size
        if total > max_pack_bytes:
            raise UnsupportedState("Byte-packing budget exceeded")
    return entries, total, device


def hash_packed_bytes(entries, data):
    """Pure-stdlib reconstruction of the sealed C.state_digest byte stream."""
    view = memoryview(data).cast("B")
    h, expected_offset = hashlib.sha256(), 0
    for row in entries:
        offset, size = row["offset"], row["nbytes"]
        if type(offset) is not int or type(size) is not int or size < 0 or offset != expected_offset:
            raise ValueError("Noncontiguous byte plan")
        if offset + size > len(view):
            raise ValueError("Truncated packed byte buffer")
        h.update(row["name"].encode())
        h.update(row["dtype"].encode())
        h.update(str(tuple(row["shape"])).encode())
        h.update(view[offset:offset + size])
        expected_offset += size
    if expected_offset != len(view):
        raise ValueError("Trailing packed bytes")
    return h.hexdigest()


def _execute(entries, total, device, torch, observer):
    fields = {"tensor_count": len(entries), "packed_bytes": total,
              "device": str(device) if device is not None else None}
    _emit(observer, "pack_started", **fields)
    if total == 0:
        # Empty tensors still contribute headers, but require no device copy.
        _emit(observer, "pack_returned", **fields)
        result = hash_packed_bytes(entries, b"")
        _emit(observer, "digest_returned", **fields, host_copy_calls=0,
              cuda_to_cpu_copy_calls=0, digest=result)
        return result
    # Detach before view/cat: no graph is constructed and no values are cast.
    pieces = [row["tensor"].detach().contiguous().reshape(-1).view(torch.uint8)
              for row in entries]
    packed = torch.cat(pieces, dim=0)
    _emit(observer, "pack_returned", **fields)
    _emit(observer, "host_copy_started", **fields)
    host = packed.cpu()  # Exactly one blocking host-copy API call.
    _emit(observer, "host_copy_returned", **fields)
    result = hash_packed_bytes(entries, memoryview(host.numpy()))
    _emit(observer, "digest_returned", **fields, host_copy_calls=1,
          cuda_to_cpu_copy_calls=int(device.type == "cuda"), digest=result)
    return result


def packed_state_digest(state, *, torch_module=None, allow_cpu_for_synthetic_check=False,
                        max_pack_bytes=DEFAULT_MAX_PACK_BYTES, observer=None):
    """Explicit candidate only; unsupported states fail before tensor work.

    ``allow_cpu_for_synthetic_check`` exercises identical byte packing with
    artificial CPU tensors. Such tests do not qualify the CUDA implementation.
    The caller must exclude concurrent updates while either digest runs.
    """
    if type(allow_cpu_for_synthetic_check) is not bool:
        raise ValueError("Synthetic CPU flag must be boolean")
    torch = _torch(torch_module)
    entries, total, device = _plan(state, torch, allow_cpu=allow_cpu_for_synthetic_check,
                                  max_pack_bytes=max_pack_bytes)
    return _execute(entries, total, device, torch, observer)


def make_state_digest(original, *, torch_module=None,
                      max_pack_bytes=DEFAULT_MAX_PACK_BYTES, observer=None):
    """Return an isolated candidate callable; never install it globally.

    CPU, mixed-device, unusual-dtype/layout and oversized states use the
    original function. Only a metadata preflight rejection falls back; a
    failure after packing starts propagates to the qualification cost trace.
    A callback can persist started/returned events before tensor operations.
    """
    if not callable(original):
        raise TypeError("An explicit original state-digest callable is required")
    if type(max_pack_bytes) is not int or max_pack_bytes < 0:
        raise ValueError("max_pack_bytes must be a nonnegative integer")

    def digest(state):
        torch = _torch(torch_module)
        try:
            entries, total, device = _plan(state, torch, allow_cpu=False,
                                          max_pack_bytes=max_pack_bytes)
        except UnsupportedState as error:
            _emit(observer, "original_fallback_started", reason=str(error))
            result = original(state)
            _emit(observer, "original_fallback_returned", reason=str(error), digest=result)
            return result
        return _execute(entries, total, device, torch, observer)

    return digest

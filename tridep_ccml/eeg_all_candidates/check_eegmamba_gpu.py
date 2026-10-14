"""Explicit synthetic official-CUDA engineering check; never called on import."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .adapters import eegmamba
from .protocol import REPORT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', required=True)
    args = parser.parse_args()
    if not args.run or not torch.cuda.is_available():
        raise RuntimeError('Explicit run flag and qualified CUDA runtime required')
    torch.set_num_threads(2)
    torch.manual_seed(20261008)
    torch.cuda.manual_seed_all(20261008)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raw = np.random.default_rng(20261008).normal(0, 20, (2, 19, 1000)).astype(np.float32)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model, audit = eegmamba.load_encoder('cuda', trainable=False)
    value = torch.from_numpy(eegmamba.adapt_raw(raw)).cuda()
    with torch.no_grad():
        features = eegmamba.encode_windows(model, value)
        repeated = eegmamba.encode_windows(model, value)
        single = eegmamba.encode_windows(model, value[:1])
    assert features.shape == (2, 200) and single.shape == (1, 200)
    assert torch.allclose(features, repeated, atol=1e-6, rtol=1e-6)
    names = eegmamba.exact_finetune_parameter_names(model)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    output = eegmamba.encode_windows(model, value)
    readout = torch.randn(200, 2, device='cuda') * .05
    loss = torch.nn.functional.cross_entropy(output @ readout, torch.tensor([0, 1], device='cuda'))
    loss.backward()
    gradients = {n: float(p.grad.norm()) for n, p in model.named_parameters() if n in names and p.grad is not None}
    assert gradients and all(np.isfinite(x) for x in gradients.values()) and any(x > 0 for x in gradients.values())
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001)
    optimizer.step()
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())]
    assert changed and set(changed).issubset(set(names))
    torch.cuda.synchronize()
    result = {'passed': True, 'load_audit': audit, 'source_data': 'synthetic CPU Gaussian RNG20261008; no cohort data or labels',
              'feature_shape': list(features.shape), 'single_shape': list(single.shape),
              'repeat_max_abs_error': float((features-repeated).abs().max()),
              'finetune_gradient_norms': gradients, 'changed_parameter_names': changed,
              'frozen_parameters_unchanged_after_step': True, 'synthetic_loss': float(loss.detach()),
              'official_mamba2_block_count': 12, 'precision': 'float32, TF32disabled, no autocast',
              'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(), 'seconds': time.monotonic()-started,
              'engineering_optimizer_steps': 1, 'formal_fits': 0, 'gpu_encode_calls': 4}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'eegmamba_official_gpu_check.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'parameters_changed': len(changed), 'seconds': result['seconds'], 'peak_bytes': result['peak_gpu_allocated_bytes']}), flush=True)


if __name__ == '__main__':
    main()

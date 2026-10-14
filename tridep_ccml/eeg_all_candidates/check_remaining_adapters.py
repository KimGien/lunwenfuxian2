"""CPU-only synthetic source/load/feature/gradient checks; no cohort access."""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .adapters import biot, codebrain, luna
from .adapters._author_support import DEPENDENCIES, digest
from .protocol import REPORT


def check(module):
    torch.manual_seed(20261008)
    generator = np.random.default_rng(20261008)
    raw = generator.normal(0, 20, (2, 19, 1000)).astype(np.float32)
    started = time.monotonic()
    model, audit = module.load_encoder('cpu', trainable=False)
    adapted = module.adapt_raw(raw)
    value = torch.from_numpy(adapted)
    with torch.no_grad():
        feature = module.encode_windows(model, value)
        one = module.encode_windows(model, value[:1])
        repeated = module.encode_windows(model, value)
    assert torch.equal(feature, repeated)
    assert feature.shape == (2, module.FEATURE_DIM) and one.shape == (1, module.FEATURE_DIM)
    assert not any(p.requires_grad for p in model.parameters())
    names = module.exact_finetune_parameter_names(model)
    model.requires_grad_(False)
    selected = {n: p for n, p in model.named_parameters() if n in names}
    for p in selected.values():
        p.requires_grad_(True)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    features = module.encode_windows(model, value)
    readout = torch.randn(module.FEATURE_DIM, 2, dtype=torch.float32) * .05
    loss = torch.nn.functional.cross_entropy(features @ readout, torch.tensor([0, 1]))
    loss.backward()
    gradients = {n: float(p.grad.norm()) for n, p in selected.items() if p.grad is not None}
    assert gradients and all(np.isfinite(x) for x in gradients.values()) and any(x > 0 for x in gradients.values())
    optimizer = torch.optim.SGD(selected.values(), lr=.001)
    optimizer.step()
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())]
    assert changed and set(changed).issubset(set(names))
    return {'passed': True, 'load_audit': audit, 'adapted_shape': list(adapted.shape),
            'features_shape': list(feature.shape), 'single_window_shape': list(one.shape),
            'repeat_exact': True, 'synthetic_loss': float(loss.detach()),
            'finetune_gradient_norms': gradients, 'changed_parameter_names': changed,
            'frozen_parameters_unchanged_after_step': True, 'cpu_seconds': time.monotonic()-started,
            'engineering_optimizer_steps': 1, 'formal_fits': 0, 'cuda_calls': 0,
            'data_source': 'synthetic Gaussian CPU RNG20261008; no cohort arrays or labels read'}


def main():
    torch.set_num_threads(2)
    report = {'data_source': 'synthetic only', 'cuda_calls': 0, 'formal_fits': 0, 'models': {}, 'failures': {}}
    for module in (codebrain, luna, biot):
        name = module.__name__.split('.')[-1]
        try:
            report['models'][name] = check(module)
            print('DONE CPU adapter check ' + name, flush=True)
        except Exception as exc:
            report['failures'][name] = type(exc).__name__ + ': ' + str(exc)
            traceback.print_exc()
    report['dependency_metadata_bindings'] = [{'path': str(p), 'sha256': digest(p)}
                                             for p in sorted(DEPENDENCIES.glob('*.dist-info/METADATA'))]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'remaining_adapters_cpu_check.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'passed': list(report['models']), 'failures': report['failures']}), flush=True)
    if report['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

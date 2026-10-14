"""CPU regression comparing original-first-forward and formula-initialized v2."""
from __future__ import annotations

import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))

import numpy as np
import torch
from tridep_ccml.eeg_all_candidates import finetune as F
from tridep_ccml.eeg_all_candidates.adapters import codebrain as V1, codebrain_v2 as V2


def clone_buffers(model):
    return {n: v.detach().clone() for n, v in model.named_buffers()}


def differences(before, model):
    return [n for n, v in model.named_buffers() if not torch.equal(before[n], v)]


def main():
    torch.set_num_threads(2)
    raw = np.random.default_rng(20261008).normal(0, 20, (1, 8, 19, 1000)).astype(np.float32)
    tensor = torch.from_numpy(V1.adapt_raw(raw[0]))
    v1, original = V1.load_encoder('cpu', trainable=False)
    first_buffers = clone_buffers(v1)
    with torch.no_grad():
        first_features = V1.encode_windows(v1, tensor)
        changed_first = differences(first_buffers, v1)
        second_buffers = clone_buffers(v1)
        repeated = V1.encode_windows(v1, tensor)
        changed_second = differences(second_buffers, v1)
    assert len(changed_first) == 16 and not changed_second and torch.equal(first_features, repeated)
    v2, repaired = V2.load_encoder('cpu', trainable=False)
    before = clone_buffers(v2)
    with torch.no_grad():
        features = V2.encode_windows(v2, tensor)
    assert not differences(before, v2) and torch.equal(first_features, features)
    names = V2.exact_finetune_parameter_names(v2)
    head = F.SubjectBagHead(200)
    model = F.RawSubjectModel(v2, head, V2, names, 5, 8)
    F.set_train_phase(model, 11)
    frozen = F.state_digest(F.frozen_state(model))
    optimizer = F.optimizer_for(model)
    F.set_learning_rates(optimizer, 11)
    result = F.subject_update(model, optimizer, ['synthetic_0'], {'synthetic_0': np.tile(raw[0], (8, 1, 1))[:60]},
                              {'synthetic_0': 1}, F.bag_indices(11), micro_subjects=1)
    assert F.state_digest(F.frozen_state(model)) == frozen
    gradients = {n: float(p.grad.norm()) for n, p in v2.named_parameters() if p.requires_grad}
    assert len(names) == 25 and all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                    for p in v2.parameters() if p.requires_grad)
    output = {'passed': True, 'data_source': 'synthetic Gaussian; no cohort data or labels', 'cuda_calls': 0,
              'formal_fits': 0, 'engineering_optimizer_updates': 1, 'v1_first_forward_changed_buffers': changed_first,
              'v1_second_forward_changed_buffers': changed_second, 'v2_before_after_buffers_equal': True,
              'v1_v2_frozen_features_exact': True, 'v1_v2_max_abs_error': float((first_features-features).abs().max()),
              'v2_feature_dim': 200, 'selected_encoder_parameters': names, 'selected_gradient_norms': gradients,
              'subject_update': result, 'v2_loading_audit': repaired}
    Path(__file__).with_suffix('.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'v1_first_buffer_changes': len(changed_first), 'v2_names': len(names), 'features_equal': True}), flush=True)


if __name__ == '__main__':
    main()

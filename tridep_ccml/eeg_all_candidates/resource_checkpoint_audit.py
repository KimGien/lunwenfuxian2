"""CPU-only metadata and strict loading audit for newly downloaded assets."""
from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import traceback

WORK = Path(__file__).resolve().parents[2]
MODULE = Path(__file__).resolve().parent
ASSETS = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets'
OUTPUT = WORK / 'output/analysis/eeg_all_candidates_v1_20261008/checkpoint_resource_audit.json'


def tensor_inventory(path):
    import torch
    if path.suffix == '.safetensors':
        from safetensors import safe_open
        with safe_open(str(path), framework='pt', device='cpu') as f:
            tensors = {k: {'shape': list(f.get_slice(k).get_shape()),
                           'dtype': f.get_slice(k).get_dtype()} for k in f.keys()}
        wrapper = 'safetensors'
    else:
        state = torch.load(path, map_location='cpu', weights_only=True)
        wrapper = list(state)[:12] if isinstance(state, dict) else str(type(state))
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        elif isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
            state = state['model']
        if not isinstance(state, dict):
            raise TypeError('Checkpoint is not a tensor dictionary')
        tensors = {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'finite': bool(torch.isfinite(v).all())}
                   for k, v in state.items() if isinstance(v, torch.Tensor)}
    return {'file': path.relative_to(WORK).as_posix(), 'wrapper': wrapper,
            'tensor_count': len(tensors), 'scalar_entries': sum(math.prod(v['shape']) for v in tensors.values()),
            'tensor_metadata': tensors, 'no_neural_forward': True, 'device': 'cpu'}


def strict_csbrain():
    import torch
    path = MODULE / 'vendor/csbrain'
    sys.path.insert(0, str(path))
    cls = importlib.import_module('models.CSBrain').CSBrain
    state = torch.load(ASSETS / 'csbrain/CSBrain.pth', map_location='cpu', weights_only=True)
    mapped = {k.removeprefix('module.'): v for k, v in state.items()}
    regions = [0, 0, 0, 0, 4, 4, 1, 1, 3, 3, 0, 0, 2, 2, 2, 2, 0, 4, 1]
    order = [0, 10, 2, 16, 3, 11, 1, 6, 18, 7, 14, 12, 13, 15, 8, 9, 4, 17, 5]
    model = cls(brain_regions=regions, sorted_indices=order)
    result = model.load_state_dict(mapped, strict=True)
    return {'strict': True, 'encoder_keys': len(model.state_dict()), 'missing': result.missing_keys,
            'unexpected': result.unexpected_keys, 'parameters': sum(p.numel() for p in model.parameters()),
            'configuration': {'in_dim': 200, 'out_dim': 200, 'd_model': 200, 'n_layer': 12, 'nhead': 8},
            'no_neural_forward': True, 'note': 'Author Mumtaz topology used only to check tensor compatibility; local FCz/CPz topology still requires registration.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strict-csbrain', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    result = {'cuda_calls': 0, 'neural_forward_calls': 0, 'assets': {}, 'failures': {}}
    manifest = json.loads((ASSETS / 'download_manifest.json').read_text(encoding='utf-8'))
    for name, record in manifest['weights'].items():
        for binding in record['files']:
            path = WORK / binding['path']
            if path.suffix not in ('.safetensors', '.pth', '.pt', '.ckpt'):
                continue
            try:
                result['assets'][name] = tensor_inventory(path)
                print('DONE tensor metadata ' + name, flush=True)
            except Exception as e:
                result['failures'][name] = type(e).__name__ + ': ' + str(e)
    if args.strict_csbrain:
        try:
            result['csbrain_strict_load'] = strict_csbrain()
        except Exception as e:
            result['failures']['csbrain_strict_load'] = type(e).__name__ + ': ' + str(e)
            traceback.print_exc()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'assets': list(result['assets']), 'failures': result['failures'],
                     'strict_csbrain': result.get('csbrain_strict_load')}), flush=True)


if __name__ == '__main__':
    main()

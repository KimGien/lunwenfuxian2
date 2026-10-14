"""Export external frozen candidates without fitting on any cohort person."""
from __future__ import annotations

import argparse
import gc
import importlib
from pathlib import Path
import sys
import time

from tridep_ccml.tridep_stage1 import common as C
from . import storage as S, registry as R

DEPS = S.OUT / 'assets/python_dependencies'
sys.path.insert(0, str(DEPS))
np, torch = C.np, C.torch


def export(method, *, batch_size=4):
    R.load_registration()
    condition = method + '-frozen-smallhead'
    root = S.OUT / 'frozen_features' / condition
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        existing = S.read(manifest_path)
        for source in existing['sources']:
            S.checked(source)
        for row in existing['records']:
            S.checked(row['features'])
        return existing
    adapter = importlib.import_module('.adapters.' + method.lower(), __package__)
    from tridep_ccml.tridep_stage1.inputs import CHANNEL_NAMES
    rawmanifest = C.OUT / 'input_manifest.json'
    rawrows = S.read(rawmanifest)['records']
    lock, folds, labels = C.context()
    require([row['subject_id'] for row in rawrows] == lock['subject_ids'], 'Raw cohort order differs')
    registration_path = root / 'export_registration.json'
    if not registration_path.exists():
        S.save(registration_path, {'condition': condition, 'raw_manifest': S.desc(rawmanifest),
               'producer': S.desc(Path(__file__)), 'adapter': S.desc(Path(adapter.__file__)),
               'channel_names': CHANNEL_NAMES, 'batch_size': batch_size, 'dtype': 'float32',
               'encoder_fit_subject_ids': [], 'optimizer_updates': 0,
               'registry': R.registration_descriptor(), 'registered_at': S.now()}, immutable=True)
    reg = S.read(registration_path)
    require(reg['adapter'] == S.desc(Path(adapter.__file__)) and reg['producer'] == S.desc(Path(__file__)),
            'Partial export implementation changed')
    require(reg['batch_size'] == batch_size, 'Partial export batch size changed')
    C.runtime(11)
    model, audit = adapter.load_encoder(device='cuda', trainable=False)
    require(not model.training and not any(p.requires_grad for p in model.parameters()), 'External encoder not frozen/eval')
    before = C.state_digest(model.state_dict())
    records, source_descriptors = [], [S.desc(registration_path), S.desc(rawmanifest)]
    counts = {'new_windows': 0, 'repeat_windows': 0, 'encoder_forward_batches': 0}
    ledger_path = root / 'export_attempts' / (str(time.time_ns()) + '.json')
    ledger = {'started_at': S.now(), 'started_windows': 0, 'returned_windows': 0,
              'started_batches': 0, 'returned_batches': 0, 'status': 'running', 'optimizer_updates': 0}
    S.save(ledger_path, ledger, immutable=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for row in rawrows:
        sid = row['subject_id']
        raw = np.load(S.checked(row['raw']), allow_pickle=False)
        source_descriptors.append(row['raw'])
        target = root / 'windows' / (sid + '.npy')
        sidecar = target.with_suffix('.json')
        if sidecar.exists():
            recorded = S.read(sidecar)
            require(recorded['raw'] == row['raw'] and recorded['export_registration'] == S.desc(registration_path), 'Partial exported source differs')
            S.checked(recorded['features'])
            records.append({'subject_id': sid, 'features': recorded['features'], 'sidecar': S.desc(sidecar)})
            continue
        require(not target.exists(), 'Unbound partial array requires inspection before repeating encoder work')
        values = adapter.adapt_raw(raw)
        blocks = []
        with torch.inference_mode():
            for start in range(0, 60, batch_size):
                ledger['started_batches'] += 1
                ledger['started_windows'] += min(batch_size, 60-start)
                S.save(ledger_path, ledger)
                output = adapter.encode_windows(model, torch.as_tensor(values[start:start+batch_size], device='cuda'))
                ledger['returned_batches'] += 1
                ledger['returned_windows'] += len(output)
                S.save(ledger_path, ledger)
                require(output.ndim == 2 and bool(torch.isfinite(output).all()), 'Nonfinite or invalid frozen native output')
                blocks.append(output.float().cpu().numpy())
                counts['new_windows'] += len(output)
                counts['encoder_forward_batches'] += 1
            ledger['started_batches'] += 1
            ledger['started_windows'] += batch_size
            S.save(ledger_path, ledger)
            repeat = adapter.encode_windows(model, torch.as_tensor(values[:batch_size], device='cuda')).float().cpu().numpy()
            ledger['returned_batches'] += 1
            ledger['returned_windows'] += len(repeat)
            S.save(ledger_path, ledger)
            counts['repeat_windows'] += len(repeat)
            counts['encoder_forward_batches'] += 1
        array = np.ascontiguousarray(np.concatenate(blocks), dtype=np.float32)
        require(np.array_equal(array[:batch_size], repeat), 'Identical frozen same-batch readout differs')
        require(array.shape[0] == 60, 'Frozen output must retain all60 windows')
        S.write(target, lambda stream: np.save(stream, array, allow_pickle=False), immutable=True)
        S.save(sidecar, {'subject_id': sid, 'raw': row['raw'], 'features': S.desc(target),
               'export_registration': S.desc(registration_path), 'repeat_exact': True,
               'shape': list(array.shape), 'encoder_fit_subject_ids': []}, immutable=True)
        records.append({'subject_id': sid, 'features': S.desc(target), 'sidecar': S.desc(sidecar)})
        print(f'FROZEN_EXPORT {method} {sid} shape={list(array.shape)}', flush=True)
    require(before == C.state_digest(model.state_dict()) and not model.training
            and not any(p.requires_grad or p.grad is not None for p in model.parameters()), 'Frozen encoder changed')
    for source in source_descriptors:
        S.checked(source)
    dimension = np.load(S.checked(records[0]['features']), allow_pickle=False).shape[-1]
    input_contract = audit.get('input_contract', {})
    ledger.update(status='complete', finished_at=S.now())
    S.save(ledger_path, ledger)
    # Preserve all prior attempts rather than reporting only a successful rerun.
    execution_attempts = [S.desc(p) for p in sorted((root / 'export_attempts').glob('*.json'))]
    for source in audit.get('implementation_sources', []) + audit.get('weight_sources', []):
        S.checked(source)
        if source not in source_descriptors:
            source_descriptors.append(source)
    if isinstance(audit.get('checkpoint'), dict) and 'path' in audit['checkpoint']:
        source_descriptors.append({k: audit['checkpoint'][k] for k in ('path', 'sha256')})
    result = {'passed': True, 'condition': condition, 'subject_ids': lock['subject_ids'], 'records': records,
              'sources': source_descriptors, 'encoder_fit_subject_ids': [], 'dimension': dimension,
              'native_dimension_retained': True, 'pooling': input_contract.get('pooling', audit.get('pooling', 'registered author adapter')),
              'reference': input_contract.get('reference', audit.get('reference')),
              'amplitude': input_contract.get('amplitude', input_contract.get('scale', audit.get('amplitude'))),
              'input_contract': input_contract, 'execution_attempts': execution_attempts,
              'input_seconds': 5, 'loading_audit': audit,
              'frozen_state_before': before, 'frozen_state_after': before, 'frozen_state_exact': True,
              'new_encoder_forwards': counts['encoder_forward_batches'], 'current_attempt_counts': counts,
              'seconds_this_attempt': time.perf_counter()-started,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'registry': R.registration_descriptor()}
    S.save(manifest_path, result, immutable=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def require(value, message):
    if not value:
        raise ValueError(message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('methods', nargs='+')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    for method in args.methods:
        export(method, batch_size=args.batch_size)

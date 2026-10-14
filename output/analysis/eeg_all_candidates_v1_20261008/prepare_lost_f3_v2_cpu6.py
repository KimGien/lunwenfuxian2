"""One read-only real loss-input preflight; no checkpoint/tensor reload or launch."""
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import lost_f3_closure_v2 as L


def main():
    sources = [L.descriptor(p) for p in (SOURCE, L.SOURCE,
        SOURCE.with_name('audit_completed_finetune_checkpoints_v6.py'),
        SOURCE.with_name('test_lost_f3_closure_v2_cpu6.py'))]
    original_open = Path.open
    hash_only_binary_paths = set()

    def guarded_open(path, mode='r', *args, **kwargs):
        resolved = Path(path).resolve()
        if not any(flag in mode for flag in ('w', 'a', 'x', '+')):
            if resolved.suffix.lower() in ('.pt', '.pth', '.bin', '.safetensors', '.npy', '.npz'):
                relative = resolved.relative_to(WORK).as_posix()
                if '/runs/' in relative or '/snapshot_' in relative or resolved.suffix.lower() in ('.npy', '.npz'):
                    raise AssertionError('No experiment tensor/raw input may be reopened by this preflight')
                hash_only_binary_paths.add(relative)
        return original_open(path, mode, *args, **kwargs)

    if 'torch' in sys.modules:
        raise AssertionError('Torch must not be imported')
    started = time.perf_counter()
    with patch.object(Path, 'open', guarded_open):
        context = L.verify_recovery_inputs(L.LOSS_BINDING)
    if 'torch' in sys.modules:
        raise AssertionError('Torch unexpectedly imported')
    for binding in sources:
        if L.descriptor(WORK / binding['path']) != binding:
            raise AssertionError('Preparation source changed')
    value = {'schema': 'lost_f3_v2_cpu6_actual_input_preparation_v1', 'passed': True,
             'source': sources[0], 'sources': sources, 'loss_evidence': L.LOSS_BINDING,
             'actual_input_preflight': context, 'seconds_this_readonly_preflight': time.perf_counter() - started,
             'official_binary_hash_only_paths_observed': sorted(hash_only_binary_paths),
             'scope': 'Actual saved loss/CPU JSON and source/provenance checks. Official weight bytes may be hashed; no tensor deserialization.',
             'checkpoint_loads': 0, 'model_constructions': 0, 'GPU_operations': 0,
             'process_queries': 0, 'process_controls': 0, 'test_probability_reads': 0,
             'training_launches': 0, 'new_recovery_registration_created': False,
             'new_loss_closure_created': False, 'full_plan_certificate': False}
    path = SOURCE.parent / ('lost_f3_v2_cpu6_actual_input_preparation_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'artifact': L.descriptor(path), 'passed': True, 'durable': context['durable'],
        'known_extra_lower_bounds': context['known_extra_lower_bounds'],
        'official_binary_hash_only_paths_observed': value['official_binary_hash_only_paths_observed'], 'checkpoint_loads': 0}))


if __name__ == '__main__':
    main()

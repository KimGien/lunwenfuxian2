"""CPU v6 driver: unchanged CPU v5 math/three-load audit with loss verifier v2.

Only the delegated auditor's training_complete branch can load checkpoint
tensors. This wrapper does not run that audit on import. The historical X/L
provenance checks may hash official weight bytes; this is not JSON-only I/O.
There are no model constructors, raw/test arrays, test gate, or GPU operations.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import lost_f3_closure_v2 as L, storage as S

DELEGATE = SOURCE.with_name('audit_completed_finetune_checkpoints_v5.py')
DELEGATE_SHA = '58b101d301f92da4283890e0a4fa80bef40cdb38eeace1001d6fa18e583a3eba'


def require(value, message):
    if not value:
        raise ValueError(message)


def load_delegate():
    require(hashlib.sha256(DELEGATE.read_bytes()).hexdigest() == DELEGATE_SHA,
            'Sealed CPU v5 source changed')
    spec = importlib.util.spec_from_file_location('completed_ft_cpu_v6_private_v5', DELEGATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # A private namespace only. The imported global v5 module and its file are
    # untouched; every tensor/math/boundary function is original compiled v5.
    module.L = L
    module.SOURCE = SOURCE
    previous_sources = module.delegated_sources

    def sources():
        require(hashlib.sha256(DELEGATE.read_bytes()).hexdigest() == DELEGATE_SHA,
                'Sealed CPU v5 source changed')
        L.verify_delegated_source()
        return {**previous_sources(), 'previous_cpu_auditor_v5': S.desc(DELEGATE),
                'actual_cpu_auditor_v6': S.desc(SOURCE),
                'previous_lost_f3_closure_verifier_v1': S.desc(Path(L.L1.__file__)),
                'lost_f3_closure_verifier_v2': S.desc(Path(L.__file__))}

    module.delegated_sources = sources
    return module


B = load_delegate()


def audit():
    before = S.desc(SOURCE)
    result = B.audit()
    require(S.desc(SOURCE) == before, 'CPU v6 source changed during audit')
    require(result['schema_version'] == 5 and result['auditor_source'] == before,
            'Delegated CPU audit shape/provenance differs')
    result.update(schema_version=6, auditor_source=before,
                  delegated_cpu_auditor_source=S.desc(DELEGATE),
                  loss_closure_verifier_source=S.desc(Path(L.__file__)),
                  original_cpu_math_and_three_checkpoint_loads_unchanged=True,
                  historical_weight_bytes_may_be_hashed_for_source_binding=True,
                  process_loss_does_not_imply_natural_exit_or_zero_unflushed_tail=True)
    return result


def main():
    result = audit()
    path = S.REPORT / ('completed_finetune_checkpoint_audit_v6_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact': S.desc(path), 'status': result['status'],
                      'completed_finetune_observed': result['training_complete_finetune_observed'],
                      'verified': result['verified_completed_finetune_fits'],
                      'expected': result['registered_finetune_fits'],
                      'checkpoint_loads': result['checkpoint_loads'], 'errors': result['errors']}, ensure_ascii=True))
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())

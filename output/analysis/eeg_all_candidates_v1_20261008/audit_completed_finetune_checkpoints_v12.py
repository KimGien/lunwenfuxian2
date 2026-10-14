"""CPU12: unchanged CPU8 three-load checks plus actual repair observer8.

Default/import do not run an audit. --run audits only actual training_complete
FT. No tensor/model/GPU/raw/test is used by the added source/observer wrapper.
Actual8 provenance is kept; old2 arithmetic is only a private in-memory view.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S
from tridep_ccml.eeg_all_candidates import repair_observation_v5 as V
DELEGATE = SOURCE.with_name('audit_completed_finetune_checkpoints_v8.py')
DELEGATE_SHA = '43ab48da0bd862399d13aed718ad87500a718e795e9742663d78ba97c958252f'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def load_delegate():
    require(S.digest(DELEGATE) == DELEGATE_SHA, 'Original CPU8 source changed')
    spec = importlib.util.spec_from_file_location('_CPU12_private_unchanged_CPU8', DELEGATE)
    private = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(private)
    base = private.B
    base.SOURCE = SOURCE
    old_boundary = base.trace_boundary
    old_sources = base.delegated_sources

    def boundary(fit, metadata, inputs):
        versions = {x['record'].get('implementation', {}).get('version') for x in inputs}
        if 8 not in versions:
            require('actual_deterministic_repair_execution' not in metadata, 'Actual8 metadata lacks actual8 physical trace')
            return old_boundary(fit, metadata, inputs)
        require(versions == {8} and fit['condition'] in V.D.REPAIR_CONDITIONS, 'Unknown/mixed physical observer in repaired trajectory')
        return V.trace_boundary(base.legacy_trace_boundary, fit, metadata, inputs)

    def sources():
        require(S.digest(DELEGATE) == DELEGATE_SHA, 'Original CPU8 source changed')
        V.source_check()
        return {**old_sources(), 'previous_CPU8': S.desc(DELEGATE), 'actual_CPU12': S.desc(SOURCE), 'actual_repair_observer_verifier': S.desc(V.SOURCE), 'actual_F8_source': S.desc(V.F.SOURCE), 'repair_execution_verifier': S.desc(V.D.SOURCE)}
    base.trace_boundary = boundary
    base.delegated_sources = sources
    return base
B = load_delegate()

def audit():
    before = S.desc(SOURCE)
    result = B.audit()
    require(result['auditor_source'] == before and result['schema_version'] == 5, 'Original private three-load audit shape differs')
    require(S.desc(SOURCE) == before, 'CPU12 source changed during audit')
    result.update(schema_version=12, auditor_source=before, delegated_CPU8_source=S.desc(DELEGATE), repair_observer_verifier_source=S.desc(V.SOURCE), old_actual2_3_loss_and_checkpoint_math_unchanged=True, actual_observer8_sources_verified_before_counter_arithmetic=True, actual_observer8_never_relabelled_as_actual3=True, new_CPU_checkpoint_loads_per_completed_fit=3, historical_weight_or_engineering_probe_bytes_may_be_hashed=True)
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(argv)
    if not args.run:
        print(json.dumps({'status': 'preparation_only', 'actual_CPU_audit_started': False, 'model_calls': 0, 'GPU_calls': 0, 'checkpoint_tensor_loads': 0, 'new_report_writes': 0}))
        return 0
    result = audit()
    target = S.REPORT / ('completed_finetune_checkpoint_audit_v12_' + str(time.time_ns()) + '.json')
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'artifact': S.desc(target), 'status': result['status'], 'completed_finetune_observed': result['training_complete_finetune_observed'], 'verified': result['verified_completed_finetune_fits'], 'expected': 135, 'checkpoint_loads': result['checkpoint_loads'], 'errors': result['errors']}))
    return 1 if result['errors'] else 0
if __name__ == '__main__':
    raise SystemExit(main())

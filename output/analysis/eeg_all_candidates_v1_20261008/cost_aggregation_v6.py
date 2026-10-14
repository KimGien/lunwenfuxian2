"""Read-only cost V6: immutable process-loss closures, never a fabricated finish.

V5 retains all inventory, counter and resource math. A lost F3 record remains
running in its original bytes; the closure only explains why it is historical.
Known recovery deltas are already inside the persisted physical observations.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve()
WORK = SOURCE.parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import lost_f3_closure_v1 as L

DELEGATE = SOURCE.with_name('cost_aggregation_v5.py')
DELEGATE_SHA = 'b354f26ae0e79ebb49d0597d18ca052b3c3211501a7c7de53e884b9b7f4f1fc9'


def load_v5():
    if hashlib.sha256(DELEGATE.read_bytes()).hexdigest() != DELEGATE_SHA:
        raise ValueError('Sealed V5 cost delegate changed')
    spec = importlib.util.spec_from_file_location('_cost_v6_sealed_v5', DELEGATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_disclosure(proof):
    for key in ('passed', 'persisted_counters_are_lower_bounds', 'unflushed_tail_unknown', 'extra_work_not_added_again'):
        if proof.get(key) is not True:
            raise ValueError('Lost F3 closure disclosure differs: ' + key)
    if proof.get('original_status') != 'running' or proof.get('graceful_terminal_flush') is not False:
        raise ValueError('Original running/unguarded terminal state must remain truthful')
    for key in ('unflushed_tail_counters', 'final_elapsed_seconds'):
        if key not in proof or proof[key] is not None:
            raise ValueError('Unobserved tail/time must remain explicit null: ' + key)
    for key, expected in [('known_extra_optimizer_returns_outside_checkpoint', 4),
                          ('known_extra_backward_returns_outside_checkpoint', 27)]:
        if type(proof.get(key)) is not int or proof[key] != expected:
            raise ValueError('Recovery lower-bound delta differs: ' + key)


def augment(result, *, verifier=None, reader=None):
    """Add independently verified closures; no counters or logical fits are added."""
    verifier = verifier or L.verify_lost_f3_attempt
    reader = reader or L.read_json
    result['schema_version'] = 6
    result['previous_cost_source_v5'] = result['source']
    result['source'] = L.descriptor(SOURCE)
    result['lost_f3_closure_verifier_source'] = L.descriptor(Path(L.__file__))
    closed, issues, affected = [], [], set()
    # No duplicate closure is permitted to introduce another physical invocation.
    primary_closures = set()
    for row in result['cost_records']:
        if '/finetune_v2_attempts/' not in row['artifact']['path']:
            continue
        try:
            record = reader(row['artifact'])
            if record.get('implementation', {}).get('version') != 3 or record.get('status') != 'running':
                continue
            # Endpoint inference must never charge suspension/overnight time.
            # This applies even when closure validation fails: an error is not a
            # measured process endpoint and will block a final certificate.
            row['seconds_observed'] = None
            row['seconds_scope'] = 'Lost or nonterminal F3 process final elapsed is unknown; pause/overnight excluded'
            affected.add(row['category'])
            fit = {key: record[key] for key in ('condition', 'seed', 'fold')}
            fit['fit_id'] = fit['condition'] + '/seed_' + str(fit['seed']) + '/fold_' + str(fit['fold'])
            proof = verifier(row['artifact'], record, fit)
            if proof is None:
                continue
            validate_disclosure(proof)
            if row.get('status') != record['status']:
                raise ValueError('Delegated row changed the original running status')
            closure_id = (proof['closure']['path'], proof['closure']['sha256'])
            if row['aggregation_role'] == 'primary_observed_invocation':
                if closure_id in primary_closures:
                    raise ValueError('The same lost F3 closure was counted as two primary invocations')
                primary_closures.add(closure_id)
            row['source_bound_process_loss_closure'] = proof
            row['original_trace_status_preserved'] = 'running'
            row['counter_scope'] = 'persisted buffered lower bounds; process later observed absent; unflushed tail unknown'
            row['known_counts_do_not_cover_unflushed_tail'] = True
            row['unflushed_tail_counters'] = None
            row['recovery_delta_is_subset_of_existing_counters'] = True
            row['additional_optimizer_returns_added_for_recovery'] = 0
            row['additional_backward_returns_added_for_recovery'] = 0
            closed.append({'artifact': row['artifact'], 'aggregation_role': row['aggregation_role'], **proof})
            for key in ('closure', 'recovery_registration', 'closure_source', 'checkpoint_cpu_resume_proof'):
                if proof[key] not in result['inputs']:
                    result['inputs'].append(proof[key])
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
            issues.append({'artifact': row['artifact'], 'error_type': type(error).__name__, 'error': str(error)})
            affected.add(row['category'])
    for category in affected:
        totals = result['category_observations'][category]
        primary = [row for row in result['cost_records'] if row['category'] == category
                   and row['aggregation_role'] == 'primary_observed_invocation']
        seconds = [row['seconds_observed'] for row in primary if row['seconds_observed'] is not None]
        totals['known_recorded_scope_seconds_subtotal'] = sum(seconds) if seconds else None
        totals['seconds_unknown_records'] = len(primary) - len(seconds)
        totals['full_category_wall_or_gpu_busy_time'] = None
        totals['buffered_or_unverified_tail_coverage_unknown'] = True
        for value in totals['counter_totals'].values():
            value['exact_category_total'] = None
            value['known_observed_subtotal_excludes_unknown_buffered_tails'] = True
    result['source_bound_lost_f3_attempts'] = closed
    result['lost_f3_closure_verification'] = {
        'source': result['lost_f3_closure_verifier_source'], 'observations': closed, 'errors': issues,
        'passed_for_observed_closures': not issues,
        'original_attempt_bytes_and_status_preserved': True,
        'recovery_extra_work_already_in_primary_counters_not_added_again': True,
        'all_unflushed_tails_and_final_elapsed_remain_null': True}
    result['input_read_errors'].extend(issues)
    result['all_plan_physical_optimizer_returns_exact'] = None
    result['all_plan_wall_seconds_exact'] = None
    result['rules'].extend([
        'Oct10 lost F3 closure is separate from the Oct8 legacy4/27 and Oct9 F2 handoff0/2.',
        'Oct10 known4 optimizer returns and27 backward returns outside checkpoint are lower bounds already included in persisted36/243; never add them again.',
        'Original running trace remains byte-identical; later absence does not establish an exit time/code, graceful flush, elapsed duration or zero tail.',
        'Pause and overnight intervals are not training time; unknown original process elapsed remains null.'])
    return result


def collect():
    return augment(load_v5().collect())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = collect()
    target = (args.output or SOURCE.parent / ('actual_cost_v6_' + str(time.time_ns()) + '.json')).resolve()
    if not target.is_relative_to(SOURCE.parent):
        raise ValueError('Write only a new standalone report')
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'artifact': L.descriptor(target), 'lost_f3_closures': len(result['source_bound_lost_f3_attempts']),
                      'read_errors': result['input_read_errors'], 'draft': True}))


if __name__ == '__main__': main()

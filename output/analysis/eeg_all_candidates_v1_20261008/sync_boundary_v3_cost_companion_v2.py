"""Bundle-only compatibility for original G3 control evidence and actual cost17/wide7."""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
SOURCE=Path(__file__).resolve(); REPORT=SOURCE.parent; WORK=SOURCE.parents[3]
FINALIZER_SHA='ab5d4d892f3d6a0e75c2c82db5bd3bbcfdffd086b4be4f92858234612982a712'
WIDE_SHA='ad8ddc3b50fd8c4537d5aaffe02451196762f27a6d3ffaa33fdf282fee65e795'
OLD_G3_SOURCE={'path': 'output/analysis/eeg_all_candidates_v1_20261008/sync_boundary_v3_cost_companion_v1.py', 'sha256': 'e2eb75c9b436c7d083a6446a92699d85c7261ce84db743141f1c15cfea0d3744'}
PAUSE_SOURCE={'path': 'output/analysis/eeg_all_candidates_v1_20261008/pause_memory_resume_cost_companion_v2.py', 'sha256': 'fe5a5d0102090aaac9480770035ce45de1ac86468b7412cc1c9be2a57d3ea460'}

def require(value, message):
    if not value: raise ValueError(message)

def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0

def integer(value): return type(value) is int and value >= 0

class Reader:
    """Only source/JSON/Markdown bytes, with a final same-byte recheck."""
    def __init__(self, work=WORK): self.work, self.inputs = Path(work).resolve(), {}
    def descriptor(self, path):
        path = Path(path).resolve()
        require(path.is_relative_to(self.work) and path.suffix in ('.json', '.py', '.md'), 'Only workspace source/JSON/Markdown allowed')
        binding = {'path': path.relative_to(self.work).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        previous = self.inputs.get(binding['path'])
        require(previous is None or previous == binding, 'Artifact changed within this review')
        self.inputs[binding['path']] = binding
        return binding
    def check(self, binding):
        require(type(binding) is dict and set(binding) == {'path', 'sha256'}, 'Exact source descriptor required')
        path = (self.work / binding['path']).resolve()
        require(self.descriptor(path) == binding, 'Source-bound artifact changed')
        return path
    def read(self, binding):
        path = self.check(binding); require(path.suffix == '.json', 'JSON input required')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == binding['sha256'], 'JSON changed during same-byte read')
        return json.loads(raw.decode('utf-8-sig'))
    def recheck(self):
        for binding in list(self.inputs.values()): self.check(binding)

def verify_final_shape(cost, acceptance, registration):
    """Only validate existing report structure; do not repeat scientific audit."""
    registered = registration.get('fits', [])
    expected = {fit['fit_id'] for fit in registered}
    require(len(registered) == len(expected) == 360, 'Original unique360 registration required')
    require(type(cost.get('schema_version')) is int and cost['schema_version'] == 17
            and type(cost.get('formal_fits_expected')) is int and cost['formal_fits_expected'] == 360
            and cost.get('formal_status_counts') == {'training_complete': 360}
            and cost.get('input_read_errors') == [], 'Final cost has partial coverage or read errors')
    cost_rows, audit_rows = cost.get('formal_fits', []), acceptance.get('training_rows', [])
    require(len(cost_rows) == len(audit_rows) == 360
            and {r['fit_id'] for r in cost_rows} == {r['fit_id'] for r in audit_rows} == expected,
            'Cost/acceptance must cover each original identity exactly once')
    updates = {row['fit_id']: row.get('updates') for row in audit_rows}
    require(all(integer(value) and value > 0 for value in updates.values())
            and all(integer(row.get('formal_durable_updates')) and row['formal_durable_updates'] == updates[row['fit_id']] for row in cost_rows)
            and cost.get('formal_durable_updates_known_subtotal') == sum(updates.values()), 'Final durable update binding differs')
    require(type(acceptance.get('schema_version')) is int and acceptance['schema_version'] == 7
            and all(type(acceptance.get(k)) is int and acceptance[k] == 360 for k in
                    ('formal_identity_count', 'completed_training_json_verified', 'deep_cpu_checkpoint_scope_verified'))
            and acceptance.get('status_counts') == {'training_complete': 360}
            and acceptance.get('verification_errors') == acceptance.get('incomplete_fits') == []
            and len(acceptance.get('cpu_checkpoint_proofs', [])) == 360, 'Actual acceptance is partial or failed')
    requirements = acceptance.get('requirements', [])
    require(len(requirements) == 30 and {row['id'] for row in requirements} == {'R%02d' % i for i in range(1, 31)}
            and all(row.get('status') == 'verified_complete' for row in requirements), 'One or more full-plan requirements remain pending')
    require(all(acceptance.get(k, {}).get('passed') is True for k in ('evaluation', 'semantic_review', 'final_cost', 'actual_training_lifecycle')),
            'Actual evaluation/semantic/cost/queue proof absent')
    cpu = cost.get('source_bound_cpu_checkpoint_verification', {})
    require(cpu.get('passed') is True and type(cpu.get('verified_fits')) is int and cpu['verified_fits'] == 135
            and type(cpu.get('prior_actual_cpu_checkpoint_loads')) is int and cpu['prior_actual_cpu_checkpoint_loads'] == 405
            and type(cpu.get('new_cpu_checkpoint_loads')) is int and cpu['new_cpu_checkpoint_loads'] == 0,
            'Actual135/405 existing CPU proof summary absent')
    require(cost.get('actual_completed_report_verifier_called') is True and cost.get('existing_outer_prediction_artifacts_read') is True
            and all(type(cost.get(k)) is int and cost[k] == 0 for k in
                    ('gpu_calls_by_finalizer', 'model_constructions_by_finalizer', 'optimizer_updates_by_finalizer', 'new_predictions_computed')),
            'Existing evaluation verification provenance differs')
    require(cost.get('all_plan_physical_optimizer_returns_exact') is None and cost.get('all_plan_wall_seconds_exact') is None,
            'Unknown historical global totals must remain unknown')
    return {'registered_and_verified_fits': 360, 'existing_finetune_CPU_fit_proof': 135,
            'existing_finetune_CPU_load_proof': 405, 'scientific_checks_reexecuted': False}

def final_parent_timer(reader, cost, companion):
    original = companion['parent_attempt_timer']['original_attempt']
    rows = [r for r in cost['cost_records'] if r['artifact']['path'] == original['path']]
    require(len(rows) == 1 and rows[0]['category'] == 'dispatcher_elapsed'
            and rows[0]['aggregation_role'] == 'supplemental_nested_or_unmeasured', 'Original Q5 parent timer must remain supplemental once')
    row = rows[0]; current = reader.read(row['artifact'])
    retained = reader.read(companion['retained_pre_resume_control_snapshots']['queue_attempt_snapshot'])
    require(current.get('status') == 'complete' and type(current.get('returncode')) is int and current['returncode'] == 0
            and all(current.get(k) == retained.get(k) for k in ('job', 'command', 'started_at', 'child_pid', 'registration',
                'execution_registration', 'execution_version', 'source', 'recovery_registration', 'recovery_execution_source', 'recovery_closure')),
            'Terminal parent attempt is not the original completed invocation')
    seconds = current.get('seconds')
    require(seconds is None or number(seconds), 'Invalid final parent seconds')
    require(row['seconds_observed'] == seconds, 'Final parent timer scope differs from producer')
    return {'artifact': row['artifact'], 'seconds_observed_including_hold_and_probe': seconds,
            'seconds_added_to_formal_or_primary_probe': False, 'pure_training_seconds': None,
            'hold_subtracted_to_infer_training_time': False}

def build_bundle(companion_path, final_cost_path, acceptance_path):
    reader = Reader()
    companion_binding, cost_binding, acceptance_binding = [reader.descriptor(p) for p in (companion_path, final_cost_path, acceptance_path)]
    companion, cost, acceptance = [reader.read(x) for x in (companion_binding, cost_binding, acceptance_binding)]
    require(companion.get('source') == OLD_G3_SOURCE and companion.get('passed') is True
            and companion.get('status') == 'verified_terminal_control_cost', 'Verified actual companion required')
    for binding in companion['inputs']: reader.check(binding)
    human = companion['human_readable_control_cost_report']; reader.check(human)
    finalizer = reader.descriptor(REPORT / 'finalize_full_plan_cost_v7.py')
    wide = reader.descriptor(REPORT / 'audit_full_plan_acceptance_v7.py')
    require(finalizer['sha256'] == FINALIZER_SHA and wide['sha256'] == WIDE_SHA, 'Sealed final delivery sources differ')
    require(cost.get('finalizer_source') == finalizer and cost.get('status') == 'verified_final_cost'
            and cost.get('final_snapshot') is True and cost.get('draft_not_completion_certificate') is False
            and cost.get('all_available_artifacts_accounted') is True and cost.get('coverage_gaps_disclosed') is True,
            'Actual final cost artifact required; no finalizer is run here')
    require(acceptance.get('source') == wide and acceptance.get('full_plan_completed') is True
            and acceptance.get('objective_satisfied') is True and acceptance.get('final_cost', {}).get('snapshot') == cost_binding,
            'Actual full acceptance must bind this exact final cost')
    registration = reader.read(cost['registration'])
    require(acceptance.get('registration') == cost['registration'], 'Final artifacts use different original training registrations')
    final_shape = verify_final_shape(cost, acceptance, registration)
    helper=load_pause_helper(reader)
    helper['verify_final_pair'](lambda d: read_text_or_json(reader,d),cost,acceptance,cost_binding,acceptance_binding)
    require(cost.get('completion_gate', {}).get('recovery_execution_proof', {}).get('original_recovery_registration') == companion['recovery_registration']
            and cost['completion_gate']['recovery_execution_proof'].get('original_recovery_closure') == companion['recovery_closure'],
            'Final cost and control companion recovery lineage differ')
    reader.check(cost['human_readable_cost_report'])
    for primary in companion['primary_engineering_ledgers']:
        rows = [r for r in cost['cost_records'] if r['artifact'] == primary['artifact'] and r['aggregation_role'] == 'primary_observed_invocation']
        require(len(rows) == 1 and rows[0]['observed_counters'] == primary['observed_counters']
                and rows[0]['seconds_observed'] == primary['seconds_observed'] and rows[0]['peak_allocated_bytes'] == primary['peak_allocated_bytes'],
                'Actual probe is absent, duplicated or changed in final cost')
    parent_timer = final_parent_timer(reader, cost, companion)
    reader.recheck()
    return {'schema': 'eeg_final_cost_with_sync_boundary_v3_bundle_v2', 'source': reader.descriptor(SOURCE), 'passed': True,
            'final_cost': cost_binding, 'final_cost_human_report': cost['human_readable_cost_report'],
            'full_plan_acceptance': acceptance_binding, 'sync_control_cost_companion': companion_binding,
            'sync_control_human_report': human, 'all_probe_primaries_already_in_final_cost_once': True,
            'existing_final_artifact_shape_verified': final_shape, 'parent_attempt_timer': parent_timer,
            'companion_costs_added_again': False, 'full_plan_verification_rerun': False,
            'GPU_or_evaluation_calls': 0, 'inputs': list(reader.inputs.values())}

def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')

def read_text_or_json(reader,d):
    path=reader.check(d)
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.suffix.lower()=='.json' else path.read_text(encoding='utf-8-sig')

def load_pause_helper(reader):
    path=reader.check(PAUSE_SOURCE); ns={'__file__':str(path),'__name__':'_text_only_pause_compat'}
    exec(compile(path.read_text(encoding='utf8'),str(path),'exec'),ns); return ns

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true')
    for k in ('companion','final-cost','final-acceptance'):p.add_argument('--'+k,type=Path)
    a=p.parse_args(argv)
    if not a.run: print(json.dumps({'status':'prepared_only','operations':0}));return 0
    require(a.companion and a.final_cost and a.final_acceptance,'Three actual existing final inputs required')
    value=build_bundle(a.companion,a.final_cost,a.final_acceptance)
    target=REPORT/('final_cost_sync_control_bundle_v2_'+str(time.time_ns())+'.json');write_new(target,value)
    print(json.dumps(Reader().descriptor(target)));return 0

if __name__=='__main__':raise SystemExit(main())

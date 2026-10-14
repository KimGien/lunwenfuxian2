"""Preserve candidate draft bytes, then correct only the unregistered new family."""
from pathlib import Path
import hashlib, json, time
BASE = Path(__file__).resolve().parent
WORK = BASE.parents[2]
stamp = str(time.time_ns())
archive = BASE / 'new_repair_source_preparation_history' / ('native_tree_draft_' + stamp)
archive.mkdir(parents=True)
names = ('audit_completed_finetune_checkpoints_v12.py', 'cost_aggregation_v13.py', 'repair_postprocessing_gate_v5.py')
raw = {name: (BASE / name).read_bytes() for name in names}
for name, data in raw.items():
    (archive / name).write_bytes(data)
sources = {name: data.decode('utf-8') for name, data in raw.items()}
sources[names[0]] = sources[names[0]].replace('completed_finetune_checkpoint_audit_v11_', 'completed_finetune_checkpoint_audit_v12_')
closure = '''
FAILURE_CLOSURE = {'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/deterministic_repair_qualification_v5/failure_closures/pre_model_linux_permission_1791778566044163800/closure.json', 'sha256': '98bd04eb5256ef3d895eb750cb1a7ef9c4ac3f69ccce97b5dd37414739a21e37'}

def add_preserved_failure_closure(result):
    """Bind a zero-model failure without converting unknown old timing to zero."""
    value = S.read(S.checked(FAILURE_CLOSURE))
    require(value.get('schema') == 'actual_pre_model_qualification5_failure_closure_v1'
            and value.get('source_qualification') == S.desc(PREVIOUS_SUCCESS_Q.SOURCE), 'Exact original Qual5 failure closure required')
    require(all(type(value.get(k)) is int and value[k] == 0 for k in
                ('formal_fits', 'model_GPU_calls', 'forward_calls', 'backward_calls', 'optimizer_updates', 'checkpoint_tensor_loads'))
            and value.get('model_construction_started') is False and value.get('cleanup_verified') is True,
            'Failure closure must preserve actual pre-model zero-call scope')
    require(value.get('original_controller_status_preserved') == 'worker_starting'
            and value.get('original_controller_exit_code') is None and value.get('original_controller_elapsed') is None
            and value.get('actual_exit_time_unknown') is True, 'Original failure endpoint uncertainty changed')
    controller = S.read(S.checked(value['original_controller']))
    require(controller.get('status') == 'worker_starting'
            and controller.get('returncode') is None and controller.get('inclusive_controller_seconds') is None,
            'Original unreturned controller bytes/status must remain unchanged')
    artifacts = [FAILURE_CLOSURE] + [value[k] for k in
                 ('source_qualification', 'archived_owner', 'original_controller', 'worker_claim',
                  'failure_log', 'native_absence', 'root_linux_absence', 'permission_diagnosis')]
    existing = {x['path']: x for x in result['inputs']}
    for artifact in artifacts:
        S.checked(artifact)
        if artifact['path'] in existing:
            require(existing[artifact['path']] == artifact, 'Failure closure input byte binding differs')
        else:
            result['inputs'].append(artifact)
            existing[artifact['path']] = artifact
    result['repair_qualification_failure_closures'] = [{
        'artifact': FAILURE_CLOSURE, 'original_controller': value['original_controller'],
        'original_status_preserved': 'worker_starting', 'original_elapsed': None,
        'actual_exit_time_unknown': True, 'extra_primary_counters_added': False,
        'extra_primary_seconds_added': False, 'zero_model_scope': True}]
    return result

'''
marker = '\ndef collect():'
assert sources[names[1]].count(marker) == 1
sources[names[1]] = sources[names[1]].replace(marker, '\n' + closure + '\ndef collect():')
sources[names[1]] = sources[names[1]].replace('result = add_repair_ledgers(result, old.load_v4().normalized)', 'result = add_repair_ledgers(result, old.load_v4().normalized)\n    result = add_preserved_failure_closure(result)')
for name in names[:2]:
    before = hashlib.sha256(raw[name]).hexdigest()
    after = hashlib.sha256(sources[name].encode()).hexdigest()
    assert sources[names[2]].count(before) == 1
    sources[names[2]] = sources[names[2]].replace(before, after)
for name, source in sources.items():
    compile(source, name, 'exec')
    temp = BASE / (name + '.' + stamp + '.tmp')
    temp.write_bytes(source.encode())
    temp.replace(BASE / name)
print(json.dumps({'archive': str(archive), 'sources': {name: hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in names}, 'model_calls': 0, 'process_queries': 0}))

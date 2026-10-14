"""Create only new unregistered sources; never launch runtime/process/GPU work."""
from pathlib import Path
import ast
import hashlib
import json
import re
import time

WORK = Path(__file__).resolve().parents[3]
MOD = WORK / 'tridep_ccml/eeg_all_candidates'
A = Path(__file__).resolve().parent
PRIOR = A / 'independent_linux_identity_reuse_chain_source_review_1791780532160912600.json'
PRIOR_SHA = 'dc64317ceb61f985815a91f90a06c0e6e2636337c2286e287b4ca62e7c060dd7'
MAP = {
    'qualify_deterministic_repairs_v6': 'qualify_deterministic_repairs_v7',
    'deterministic_repair_execution_v4': 'deterministic_repair_execution_v5',
    'finetune_v7': 'finetune_v8',
    'repair_observation_v4': 'repair_observation_v5',
    'repair_runtime_owner_v4': 'repair_runtime_owner_v5',
    'repair_owned_worker_v4': 'repair_owned_worker_v5',
    'continuation_queue_v9': 'continuation_queue_v10',
    'evaluate_v7': 'evaluate_v8',
    'audit_completed_finetune_checkpoints_v11': 'audit_completed_finetune_checkpoints_v12',
    'cost_aggregation_v12': 'cost_aggregation_v13',
    'review_actual_repair_engineering_v4': 'review_actual_repair_engineering_v5',
    'repair_postprocessing_gate_v4': 'repair_postprocessing_gate_v5',
}
LABELS = {'observer7': 'observer8', 'Observer7': 'Observer8', 'observer_version7': 'observer_version8',
          'Actual7': 'Actual8', 'actual7': 'actual8', 'actual_F7': 'actual_F8',
          'repair7': 'repair8', 'F7': 'F8', 'Q9': 'Q10', 'E7': 'E8',
          'CPU11': 'CPU12', 'Cost12': 'Cost13', 'cost12': 'cost13',
          'evaluation_v7_control': 'evaluation_v8_control',
          'deterministic_repair_qualification_v6': 'deterministic_repair_qualification_v7',
          'eeg_finetune_deterministic_repair_execution_v4': 'eeg_finetune_deterministic_repair_execution_v5',
          'eeg_deterministic_repair_queue_v9': 'eeg_deterministic_repair_queue_v10'}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def desc(path): return {'path': Path(path).relative_to(WORK).as_posix(), 'sha256': sha(path)}
def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:120], text.count(old))
    return text.replace(old, new)
def migrate(text):
    mappings = {**MAP, **LABELS}
    pattern = '|'.join(re.escape(x) for x in sorted(mappings, key=len, reverse=True))
    return re.sub(pattern, lambda m: mappings[m.group()], text)
def write_new(path, text):
    compile(text, str(path), 'exec')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
def pin(text, name, value):
    pattern = rf"(?m)^({re.escape(name)} = )'[^']*'"
    result, count = re.subn(pattern, lambda m: m[1] + repr(value), text)
    assert count == 1, name
    return result

def main():
    assert sha(PRIOR) == PRIOR_SHA
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    assert prior['passed'] is True and len(prior['reviewed_sources']) == 127
    original = {x['path']: sha(WORK / x['path']) for x in prior['reviewed_sources']}
    assert all(original[x['path']] == x['sha256'] for x in prior['reviewed_sources'])
    texts = {}
    locations = {}
    for old, new in MAP.items():
        folder = MOD if (MOD / (old + '.py')).exists() else A
        path = folder / (new + '.py')
        assert not path.exists(), 'Never overwrite prepared or sealed source: ' + str(path)
        locations[new] = path
        texts[new] = migrate((folder / (old + '.py')).read_text(encoding='utf-8'))

    q = texts['qualify_deterministic_repairs_v7']
    q = q.replace('from . import linux_worker_identity_v2 as L', 'from . import linux_worker_identity_v2 as L\nfrom . import native_launcher_tree_v1 as N')
    tree = ast.parse(q)
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PINS' for t in n.targets))
    segment = ast.get_source_segment(q, node)
    q = replace_once(q, segment, segment + "\nPINS[REL + 'native_launcher_tree_v1.py'] = " + repr(sha(MOD / 'native_launcher_tree_v1.py')))
    q = q.replace("require(not (OUT / 'deterministic_repair_qualification_v5/owner.json').exists(), 'Previous failed qualifier lease not cleared')", "require(all(not (OUT / f'deterministic_repair_qualification_v{version}/owner.json').exists() for version in (5, 6)), 'Previous qualifier lease not cleared')")
    q = q.replace("while not owner.get('linux_worker_identity'):", "while not (owner.get('linux_worker_identity') and owner.get('child_native_roles') and owner.get('child_pid')):")
    q = replace_once(q,
        "        worker_hosts = [x for x in items if 'wsl.exe' in x['command'].lower() and '--worker' in x['command'] and ('--admission-sha256' in x['command']) and ('qualify_deterministic_repairs_v7' in x['command'])]\n        require(len(worker_hosts) == 1 and worker_hosts[0]['parent'] == owner['pid'], 'Exact WSL worker launcher absent')\n        allowed.extend(worker_hosts)",
        "        require(sys.platform == 'linux' and admission_sha256, 'Exact Linux admission required')\n        tree = N.verify(items, owner['child_native_roles'], launcher_pid=owner['child_pid'], controller_pid=owner['pid'], module='tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v7', nonce=admission_sha256)\n        require('wsl.exe' in tree[0]['command'].lower(), 'Exact WSL Popen launcher required')\n        allowed.extend(tree)")
    q = replace_once(q,
        "                current['linux_worker_identity'] = L.wait_capture(module='tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v7', flags={'--condition': condition, '--admission-sha256': sha(folder / 'admission.json')}, is_alive=lambda: child.poll() is None)\n                save(ROOT / 'owner.json', current)",
        "                linux_identity = L.wait_capture(module='tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v7', flags={'--condition': condition, '--admission-sha256': sha(folder / 'admission.json')}, is_alive=lambda: child.poll() is None)\n                items = host_inventory()\n                native_tree = N.capture(items, launcher_pid=child.pid, controller_pid=os.getpid(), module='tridep_ccml.eeg_all_candidates.qualify_deterministic_repairs_v7', nonce=sha(folder / 'admission.json'))\n                admit_host(items, current['native_roles'] + native_tree)\n                current.update(linux_worker_identity=linux_identity, child_pid=child.pid, child_native_roles=native_tree)\n                save(ROOT / 'owner.json', current)")
    texts['qualify_deterministic_repairs_v7'] = q

    o = texts['repair_runtime_owner_v5']
    o = o.replace('from . import linux_worker_identity_v2 as L', 'from . import linux_worker_identity_v2 as L\nfrom . import native_launcher_tree_v1 as N')
    o = o.replace("owner.get('execution_version') == 9", "owner.get('execution_version') == 10").replace("registration['version'] == 9", "registration['version'] == 10")
    o = replace_once(o,
        "        require(admission_binding is not None and owner.get('linux_worker_identity') is not None, 'Controller-captured Linux identity required')",
        "        require(admission_binding is not None and owner.get('linux_worker_identity') is not None and owner.get('child_native_roles'), 'Controller-captured Linux/native identities required')\n        native_tree = N.verify(items, owner['child_native_roles'], launcher_pid=owner['child_pid'], controller_pid=owner['pid'], module=WORKER_MODULE, nonce=admission['token'])\n        require(native_tree[0] == launch, 'Popen primary differs from bound native tree')\n        roles.extend(native_tree[1:])")
    o = o.replace("(sys.platform != 'linux' or owner.get('linux_worker_identity'))", "(sys.platform != 'linux' or (owner.get('linux_worker_identity') and owner.get('child_native_roles')))")
    texts['repair_runtime_owner_v5'] = o

    queue = texts['continuation_queue_v10']
    queue = queue.replace('from . import linux_worker_identity_v2 as L', 'from . import linux_worker_identity_v2 as L\nfrom . import native_launcher_tree_v1 as N')
    queue = queue.replace("'version': 4", "'version': 5").replace("'version': 9", "'version': 10").replace("['version'] == 9", "['version'] == 10").replace("'execution_version': 9", "'execution_version': 10").replace("'new_actual_observer_version': 7", "'new_actual_observer_version': 8").replace("'version': 11", "'version': 12")
    queue = queue.replace('child_native_identity=None, linux_worker_identity=None', 'child_native_identity=None, linux_worker_identity=None, child_native_roles=None')
    queue = replace_once(queue,
        "            owner['linux_worker_identity'] = L.wait_capture(module=O.WORKER_MODULE, flags={'--token': token, '--condition': condition, '--worker-admission-sha256': binding['sha256'], '--repair-registration-sha256': repair_registration['sha256']}, is_alive=lambda: process.poll() is None)\n            S.save(ROOT / 'owner.json', owner)",
        "            linux_identity = L.wait_capture(module=O.WORKER_MODULE, flags={'--token': token, '--condition': condition, '--worker-admission-sha256': binding['sha256'], '--repair-registration-sha256': repair_registration['sha256']}, is_alive=lambda: process.poll() is None)\n            items = Q.host_inventory()\n            native_tree = N.capture(items, launcher_pid=process.pid, controller_pid=owner['pid'], module=O.WORKER_MODULE, nonce=token)\n            require(native_tree[0] == owner['child_native_identity'], 'Native primary changed during Linux handshake')\n            Q.admit_host(items, owner['native_roles'] + native_tree)\n            owner.update(linux_worker_identity=linux_identity, child_native_roles=native_tree)\n            S.save(ROOT / 'owner.json', owner)\n            record.update(child_native_roles=native_tree, linux_worker_identity=linux_identity)\n            S.save(folder / 'status.json', record)")
    texts['continuation_queue_v10'] = queue

    e = texts['evaluate_v8']
    e = e.replace('from . import linux_worker_identity_v2 as L', 'from . import linux_worker_identity_v2 as L\nfrom . import native_launcher_tree_v1 as N')
    e = e.replace("'version': 7", "'version': 8").replace("get('schema_version') == 11", "get('schema_version') == 12").replace("queue['version'] == 9", "queue['version'] == 10")
    e = replace_once(e,
        "        require(owner.get('linux_worker_identity') is not None, 'Controller-captured evaluator Linux identity required')",
        "        require(owner.get('linux_worker_identity') is not None and owner.get('child_native_roles'), 'Controller-captured evaluator Linux/native identities required')\n        native_tree = N.verify(items, owner['child_native_roles'], launcher_pid=owner['child_pid'], controller_pid=owner['pid'], module=MODULE, nonce=token)\n        require(native_tree[0] == launch, 'Evaluator Popen primary differs from bound native tree')\n        roles.extend(native_tree[1:])")
    e = e.replace("(sys.platform != 'linux' or owner.get('linux_worker_identity'))", "(sys.platform != 'linux' or (owner.get('linux_worker_identity') and owner.get('child_native_roles')))")
    e = e.replace('child_identity=None, linux_worker_identity=None', 'child_identity=None, linux_worker_identity=None, child_native_roles=None')
    e = replace_once(e,
        "                    owner['linux_worker_identity'] = L.wait_capture(module=MODULE, flags={'--token': token, '--mode': mode, '--admission-sha256': binding['sha256'], '--cpu-gate-sha256': gate_binding['sha256']}, is_alive=lambda: process.poll() is None)\n                    S.save(CONTROL / 'owner.json', owner)",
        "                    linux_identity = L.wait_capture(module=MODULE, flags={'--token': token, '--mode': mode, '--admission-sha256': binding['sha256'], '--cpu-gate-sha256': gate_binding['sha256']}, is_alive=lambda: process.poll() is None)\n                    items = Q.host_inventory()\n                    native_tree = N.capture(items, launcher_pid=process.pid, controller_pid=owner['pid'], module=MODULE, nonce=token)\n                    require(native_tree[0] == owner['child_identity'], 'Evaluator native primary changed during Linux handshake')\n                    Q.admit_host(items, owner['native_roles'] + native_tree)\n                    owner.update(linux_worker_identity=linux_identity, child_native_roles=native_tree)\n                    S.save(CONTROL / 'owner.json', owner)")
    texts['evaluate_v8'] = e

    f = texts['finetune_v8'].replace("'version': 7", "'version': 8").replace("['observer_version'] == 7", "['observer_version'] == 8").replace("'actual_observer_version': 7", "'actual_observer_version': 8")
    texts['finetune_v8'] = f
    v = texts['repair_observation_v5'].replace("== 7", "== 8").replace("[7]", "[8]").replace("'observer_version': 7", "'observer_version': 8")
    texts['repair_observation_v5'] = v
    cpu = texts['audit_completed_finetune_checkpoints_v12'].replace('if 7 not in versions:', 'if 8 not in versions:').replace('versions == {7}', 'versions == {8}').replace('schema_version=11', 'schema_version=12')
    texts['audit_completed_finetune_checkpoints_v12'] = cpu
    d = texts['deterministic_repair_execution_v5']
    start = d.index('    current_prior = read_bound(', d.index('def implementation_sources():'))
    end = d.index('    for name in NEW_ENTRY_NAMES:', start)
    d = d[:start] + "    prior_binding = " + repr(desc(PRIOR)) + "\n    prior = read_bound(prior_binding)\n    require(prior.get('passed') is True and len(prior['reviewed_sources']) == 127, 'Previous127 reviewed source lineage required')\n    records = {x['path']: x for x in prior['reviewed_sources']}\n    records[prior_binding['path']] = prior_binding\n" + d[end:]
    d = d.replace("value.get('version') == 4", "value.get('version') == 5").replace("'version': 7", "'version': 8").replace("'observer_version': 7", "'observer_version': 8").replace("'new_actual_observer_version': 7", "'new_actual_observer_version': 8")
    texts['deterministic_repair_execution_v5'] = d

    cost = texts['cost_aggregation_v13']
    cost = cost.replace('from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v5 as PREVIOUS_SUCCESS_Q', 'from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v5 as PREVIOUS_SUCCESS_Q\nfrom tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v6 as PREVIOUS_IDENTITY_Q')
    cost = cost.replace("!= 7", "!= 8").replace('schema_version=12', 'schema_version=13').replace('actual_cost_v12_', 'actual_cost_v13_')
    cost = cost.replace("else Q\n    require(any(('/deterministic_repair_qualification_v' + str(v) + '/' in artifact['path'] for v in (4, 5, 6)))", "else PREVIOUS_IDENTITY_Q if '/deterministic_repair_qualification_v6/' in artifact['path'] else Q\n    require(any(('/deterministic_repair_qualification_v' + str(v) + '/' in artifact['path'] for v in (4, 5, 6, 7)))")
    cost = cost.replace("patterns = (", "patterns = ('deterministic_repair_qualification_v6/*/*/attempt.json', 'deterministic_repair_qualification_v6/*/*/controller.json', ", 1)
    texts['cost_aggregation_v13'] = cost
    post = texts['repair_postprocessing_gate_v5'].replace("get('schema_version') == 11", "get('schema_version') == 12")
    texts['repair_postprocessing_gate_v5'] = post

    # Source hash graph is acyclic: runtime -> qualifier -> contract -> observer -> post gate.
    order = ['qualify_deterministic_repairs_v7', 'finetune_v8', 'repair_runtime_owner_v5', 'repair_owned_worker_v5', 'continuation_queue_v10', 'evaluate_v8', 'deterministic_repair_execution_v5', 'repair_observation_v5', 'audit_completed_finetune_checkpoints_v12', 'cost_aggregation_v13', 'review_actual_repair_engineering_v5', 'repair_postprocessing_gate_v5']
    for name in order:
        text = texts[name]
        if name == 'deterministic_repair_execution_v5':
            text = pin(text, 'QUALIFIER_SHA', sha(locations['qualify_deterministic_repairs_v7']))
        if name == 'repair_observation_v5':
            text = pin(text, 'F4_SHA', sha(locations['finetune_v8']))
            text = pin(text, 'D_SHA', sha(locations['deterministic_repair_execution_v5']))
        if name == 'repair_postprocessing_gate_v5':
            tree = ast.parse(text)
            node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PINS' for t in n.targets))
            pins = {str(locations[n].relative_to(WORK).as_posix()): sha(locations[n]) for n in ('continuation_queue_v10', 'evaluate_v8', 'audit_completed_finetune_checkpoints_v12', 'cost_aggregation_v13')}
            literal = 'PINS = {' + ', '.join("S.WORK / " + repr(k) + ': ' + repr(v) for k, v in pins.items()) + '}'
            text = replace_once(text, ast.get_source_segment(text, node), literal)
        write_new(locations[name], text)
    assert all(sha(WORK / path) == digest for path, digest in original.items()), 'Historical127 source changed'
    proof = {'schema': 'native_launcher_tree_source_preparation_v1', 'previous127_review': desc(PRIOR), 'new_sources': [desc(locations[n]) for n in order] + [desc(MOD / 'native_launcher_tree_v1.py')], 'old_sources_preserved': True, 'numeric_model_algorithm_changed': False, 'actual_GPU_or_process_queries': 0, 'model_calls': 0, 'actual_qualification_started': False, 'actual_registration_created': False, 'independent_review_passed': False, 'captured_ns': time.time_ns()}
    path = A / ('native_launcher_tree_source_preparation_' + str(proof['captured_ns']) + '.json')
    with path.open('x', encoding='utf-8') as stream: json.dump(proof, stream, indent=2)
    print(json.dumps({'proof': desc(path), 'new_sources': proof['new_sources']}))

if __name__ == '__main__': main()

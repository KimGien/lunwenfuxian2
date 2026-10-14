"""Independent recovery accounting; saved evidence only, never neural forwards.

The original verifier remains source-frozen and audits the canonical30 models.
This additive verifier also checks the discarded82 updates, immutable23 reuse,
identical replacement prefix, and the total actual31 starts/3682 updates.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import verify as V

OUT, REPORT, WORK = V.OUT, V.REPORT, V.WORK
RECOVERY = OUT / 'recovery'
FAILED_ID = (V.CONDITIONS[1], 33, 2)
PARTIAL_NAMES = {'started.json', 'initial.pt', 'progress.json', 'progress.json.part', 'train_history.json'}
FIT_NAMES = PARTIAL_NAMES - {'progress.json.part'} | {'checkpoint.pt', 'metadata.json', 'batch_order.npy', 'complete.json'}


def function_node(source, name):
    tree = ast.parse(source)
    matches = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    V.require(len(matches) == 1, 'Expected one source function: ' + name)
    return matches[0]


def epoch_node(node):
    matches = [item for item in ast.walk(node) if isinstance(item, ast.For) and
               isinstance(item.target, ast.Name) and item.target.id == 'epoch']
    V.require(len(matches) == 1, 'Exactly one unchanged training epoch loop required')
    return matches[0]


def node_hash(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def audit_preforward_boundary(source, save_source):
    """Prove an uncaught failure at the pending save cannot enter the next NN call."""
    epoch = epoch_node(function_node(source, 'fit_all'))
    loops = [node for node in epoch.body if isinstance(node, ast.For)]
    V.require(len(loops) == 1, 'Expected one minibatch loop')
    body = loops[0].body
    calls = [(index, node.value) for index, node in enumerate(body) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
    saves = [(i, call) for i, call in calls if isinstance(call.func, ast.Attribute) and
             isinstance(call.func.value, ast.Name) and call.func.value.id == 'C' and call.func.attr == 'save']
    V.require(len(saves) == 2 and isinstance(saves[0][1].args[0], ast.BinOp) and
              isinstance(saves[0][1].args[0].right, ast.Constant) and saves[0][1].args[0].right.value == 'progress.json',
              'Pending pre-forward progress save missing')
    before = saves[0][0]
    names = []
    for index, node in enumerate(body):
        for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
            if isinstance(call.func, ast.Name) and call.func.id == 'model':
                names.append((index, 'forward'))
            elif isinstance(call.func, ast.Attribute) and call.func.attr in ('zero_grad', 'backward', 'step'):
                names.append((index, call.func.attr))
    V.require([name for _, name in names] == ['zero_grad', 'forward', 'backward', 'step'] and
              all(index > before for index, _ in names), 'Pending save does not precede every neural/gradient/update operation')
    save = function_node(save_source, 'save')
    V.require(not any(isinstance(n, (ast.Try, ast.TryStar)) for n in ast.walk(save)), 'Original save must propagate replace failure')
    final = save.body[-1]
    V.require(isinstance(final, ast.Expr) and isinstance(final.value, ast.Call) and
              isinstance(final.value.func, ast.Attribute) and final.value.func.attr == 'replace', 'Original save no longer ends in atomic replace')
    return {'pre_forward_save_precedes_zero_grad_forward_backward_step': True,
            'original_replace_failure_propagates': True, 'pending83_forward_gradient_optimizer_calls': 0,
            'epoch_AST_sha256': node_hash(epoch)}


def validate_partial(progress, pending, history, train_ids, method_registration):
    V.require(progress == {'epoch': 41, 'updates_started': 82, 'updates_completed': 82,
                          'gradient_passes_completed': 82, 'registration': method_registration}, 'Discarded completed progress differs')
    V.require(set(pending) == {'epoch', 'updates_started', 'updates_completed', 'gradient_passes_completed', 'batch_subject_ids', 'registration'} and
              pending['epoch'] == 42 and pending['updates_started'] == 83 and pending['updates_completed'] ==
              pending['gradient_passes_completed'] == 82 and pending['registration'] == method_registration and
              len(pending['batch_subject_ids']) == len(set(pending['batch_subject_ids'])) == 16 and
              set(pending['batch_subject_ids']).issubset(train_ids), 'Pending83 pre-forward progress differs')
    V.require(len(train_ids) == 26 and len(history) == 41, 'Discarded partial must contain41 whole26-person epochs')
    for epoch, row in enumerate(history, 1):
        batches = row['batches']
        V.require(row['epoch'] == epoch and row['updates_completed'] == row['gradient_passes_completed'] == 2 * epoch and
                  [b['batch_size'] for b in batches] == [16, 10] and
                  all(math.isfinite(b['train_ce']) and b['train_ce'] >= 0 for b in batches) and
                  row['train_ce'] == sum(b['batch_size'] * b['train_ce'] for b in batches) / 26,
                  'Discarded history loss/gradient/update/example count differs')
    return {'discarded_completed_updates': 82, 'discarded_training_gradient_passes': 82,
            'discarded_train_person_examples': 1066, 'discarded_complete_epochs': 41,
            'pending_pre_forward_update': 83, 'pending_pre_forward_people': pending['batch_subject_ids']}


def validate_replacement_prefix(old_history, new_history, order, ids, pending_ids):
    V.require(len(old_history) == 41 and len(new_history) == 60 and new_history[:41] == old_history,
              'Replacement first41 epochs including each minibatch CE differ bitwise from discarded partial')
    V.require(order.dtype == np.int64 and order.shape == (60, 26) and
              [ids[int(i)] for i in order[41, :16]] == pending_ids,
              'Replacement pending83 ordered people differ from old pre-forward evidence')
    return {'replacement_first41_epochs_and82_minibatch_CE_exact': True,
            'replacement_pending83_ordered16_people_exact': True}


def validate_actual_counts(value):
    expected = {'canonical_completed_fits': 30, 'canonical_optimizer_updates': 3600,
                'canonical_training_gradient_passes': 3600, 'canonical_train_person_examples': 47880,
                'discarded_fit_starts': 1, 'discarded_optimizer_updates': 82,
                'discarded_training_gradient_passes': 82, 'discarded_train_person_examples': 1066,
                'actual_fit_starts': 31, 'actual_optimizer_updates': 3682,
                'actual_training_gradient_passes': 3682, 'actual_train_person_examples': 48946}
    V.require(all(value.get(key) == count for key, count in expected.items()), 'Actual recovery accounting omits discarded work or differs')
    return expected


def validate_recovery_scope(reg):
    expected = {'kind': 'registered_IO_failure_recovery', 'client_date': '2026-10-07',
                'user_resume_instruction_verbatim': '继续', 'model_execution_allowed': True,
                'model_math_or_hyperparameters_changed': False, 'restart_from_saved_optimizer_state': False,
                'old_failed_action_must_remain_failed': True, 'new_test_selection_or_threshold_searches': 0}
    V.require(all(type(reg.get(key)) is type(value) and reg.get(key) == value for key, value in expected.items()),
              'Recovery authorization/scope changes user instruction, training mathematics, selection or failure preservation')
    V.equal(reg['IO_only_retry'], {'max_seconds': 5., 'max_attempts': 40, 'max_sleep_seconds': .1,
                                 'same_already_serialized_temporary_file': True, 'model_computation_retried': False},
            'Registered I/O-only retry scope')
    return {'user_resume_instruction_verified': '继续', 'training_math_hyperparameters_unchanged': True,
            'saved_optimizer_resume_claimed': False, 'old_failure_preservation_required': True,
            'new_test_selection_or_threshold_searches': 0, 'only_serialized_replace_retried': True}


def validate_snapshot_links(snapshot, reg):
    V.require(snapshot['old_registration'] == reg['original_registration'] and
              snapshot['failed_action'] == reg['failed_action'] and snapshot['prior_training_ledger'] == reg['prior_training_ledger'] and
              snapshot['preserved_completed_fits'] == reg['preserved_completed_fits'], 'Failure snapshot original registration/failure/prior23 differs')
    V.require(snapshot['completed_models'] == 23 and snapshot['completed_updates'] == 2760 and
              snapshot['partial_updates'] == 82 and snapshot['actual_updates_before_recovery'] == 2842 and
              snapshot['next_step_not_forwarded'] == 83 and snapshot['no_evaluation_or_scores_released'] is True and
              snapshot['checkpoint_and_optimizer_at_step82_missing'] is True, 'Captured failure accounting or pre-evaluation boundary differs')
    before, archived = snapshot['failed_partial'], reg['failed_partial']
    V.require(identity(before) == identity(archived) == FAILED_ID and before['original_folder'] == archived['original_folder'] and
              before['archive_folder'] == archived['archive_folder'], 'Before/archive partial model identity or folder differs')
    before_index = {record['name']: record for record in before['artifacts']}
    archived_index = {record['name']: record for record in archived['artifacts']}
    V.require(len(before['artifacts']) == len(archived['artifacts']) == 5 and set(before_index) == set(archived_index) == PARTIAL_NAMES,
              'Before/archive partial inventory duplicates or omits a file')
    for name, record in archived_index.items():
        original = before_index[name]
        V.require(original['original_path'] == record['original_path'] and
                  original['expected_archive_path'] == record['archived']['path'] and
                  original['sha256_before_archive'] == record['archived']['sha256'],
                  'Failure file before/after archival SHA256 or path differs: ' + name)
    return {'captured_original_registration_failure_and_prior23_equal': True,
            'archived_files_with_exact_before_after_SHA256': 5, 'pending_part_before_after_SHA256_exact': True,
            'step82_model_optimizer_missing_disclosed': True, 'scores_unreleased_at_capture': True}


def identity(record):
    return record['condition'], record['seed'], record['fold']


def canonical_identities():
    return [(condition, seed, fold) for seed in V.SEEDS for fold in range(1, 6) for condition in V.CONDITIONS]


def independent_setup_ast(node):
    selected = []
    for item in ast.walk(node):
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
            function = item.value.func
            if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name) and function.value.id == 'C' and function.attr == 'runtime':
                selected.append(item)
        elif isinstance(item, ast.Assign):
            target = item.targets[0]
            if isinstance(target, ast.Name) and target.id in {'model', 'indices', 'loader', 'optimizer'} or \
                    isinstance(target, ast.Tuple) and any(isinstance(n, ast.Name) and n.id == 'indices' for n in target.elts):
                selected.append(item)
    V.require(len(selected) == 5, 'Expected the original five RNG/model/dataset/loader/optimizer statements')
    return ast.Module(body=selected, type_ignores=[])


def audit_math_ast(reg):
    # The source-frozen transformer is called only to return ASTs. No compiled
    # fitting entry, model constructor, optimizer or neural call is executed.
    from . import recovery_fit as F
    original, generated = F.build_fit_ast()
    independent = function_node((WORK / 'tridep_ccml/tridep_eegpt/train_eval.py').read_text(encoding='utf-8'), 'fit_all')
    V.require(node_hash(original) == node_hash(independent), 'Recovery AST does not start from the frozen original fit')
    original_epoch, generated_epoch = node_hash(epoch_node(independent)), node_hash(epoch_node(generated))
    original_setup, generated_setup = node_hash(independent_setup_ast(independent)), node_hash(independent_setup_ast(generated))
    V.require(original_epoch == generated_epoch and original_setup == generated_setup,
              'Recovery changes numerical epoch loop or RNG/model/dataset/optimizer setup')
    # Remove precisely the four authorized control/provenance changes and
    # require the entire remaining function to be the original function.
    restored = copy.deepcopy(generated)
    restored.name = 'fit_all'
    V.require(ast.unparse(restored.body[0]) == '_recovery.guard_training()' and
              ast.unparse(restored.body[6]) == 'completed = _preload_completed(_recovery)' and
              ast.unparse(restored.body[7]) == "_completed_by_id = {(m['condition'], m['seed'], m['fold']): m for m in completed}",
              'Recovery added an unregistered top-level control change')
    restored.body[0] = copy.deepcopy(independent.body[0])
    restored.body[6] = copy.deepcopy(independent.body[7])
    del restored.body[7]
    restored.body.insert(3, copy.deepcopy(independent.body[3]))
    condition_loops = [n for n in ast.walk(restored) if isinstance(n, ast.For) and
                       isinstance(n.target, ast.Name) and n.target.id == 'condition']
    # The restored initial precheck also iterates condition.
    live = [n for n in condition_loops if n.body and isinstance(n.body[0], ast.Assign)]
    V.require(len(live) == 1, 'Recovery live condition loop differs')
    body = live[0].body
    expected_skip = ast.parse("""if (condition, seed, fold) in _completed_by_id:
    pair = _reuse_completed_pair(_completed_by_id[condition, seed, fold], folder, pair)
    continue""").body[0]
    V.require(node_hash(body[1]) == node_hash(expected_skip), 'Recovery skip changes training or model computation')
    del body[1]
    meta_updates = [i for i, n in enumerate(body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
                    isinstance(n.value.func, ast.Attribute) and isinstance(n.value.func.value, ast.Name) and
                    n.value.func.value.id == 'meta' and n.value.func.attr == 'update']
    expected_meta = ast.parse('meta.update(execution_registration=C.desc(C.OUT / "recovery/registration.json"), execution_source=C.desc(Path(_recovery_fit_source)))').body[0]
    V.require(len(meta_updates) == 1 and node_hash(body[meta_updates[0]]) == node_hash(expected_meta), 'Recovery metadata change contains unregistered computation')
    del body[meta_updates[0]]
    V.require(node_hash(restored) == node_hash(independent), 'Recovery changes the original training function beyond four allowed controls/provenance additions')
    V.equal(reg['math_ast_evidence'], F.ast_evidence(), 'Registered recovery transformation proof')
    return {'epoch_AST_sha256': original_epoch, 'setup_AST_sha256': original_setup,
            'original_and_recovery_training_math_identical': True, 'full_function_identical_after_four_authorized_changes_removed': True,
            'new_model_execution_in_AST_audit': 0}


def audit_io_source():
    source = WORK / 'tridep_ccml/tridep_eegpt/recovery_io.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    limits = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and
              isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith('MAX_')}
    V.require(limits == {'MAX_REPLACE_ATTEMPTS': 40, 'MAX_REPLACE_SECONDS': 5.0, 'MAX_SLEEP_SECONDS': .1}, 'Bounded replacement retry limits differ')
    bridge = function_node(source.read_text(encoding='utf-8'), 'installed_io')
    functions = {n.name: n for n in bridge.body if isinstance(n, ast.FunctionDef)}
    V.require({'event', 'paths', 'replace', 'save', 'save_torch', 'save_array'}.issubset(functions), 'Expected bounded serialization bridge functions missing')
    replace = functions['replace']
    allowed = {'time.monotonic', 'temporary.replace', 'isinstance', 'getattr', 'temporary.exists', '_sha256',
               'event', 'str', 'type', 'time.sleep', 'min', 'range'}
    calls = {ast.unparse(n.func) for n in ast.walk(replace) if isinstance(n, ast.Call)}
    V.require(calls.issubset(allowed) and 'temporary.replace' in calls and
              not any(isinstance(n, (ast.With, ast.AsyncWith)) for n in ast.walk(replace)), 'Replacement retry contains serialization or model work')
    loops = [n for n in ast.walk(replace) if isinstance(n, (ast.For, ast.While))]
    V.require(len(loops) == 1 and isinstance(loops[0], ast.For) and
              ast.unparse(loops[0].iter) == 'range(1, MAX_REPLACE_ATTEMPTS + 1)', 'Replacement attempts are not bounded')
    handlers = [n for n in ast.walk(replace) if isinstance(n, ast.ExceptHandler)]
    V.require(len(handlers) == 1 and ast.unparse(handlers[0].type) == 'OSError', 'Replace retry error class differs')
    assignment = next(n for n in handlers[0].body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'eligible')
    V.require(ast.unparse(assignment.value) == "isinstance(error, PermissionError) or getattr(error, 'winerror', None) == 5", 'Retry eligibility includes an unregistered failure')
    V.require(not any(isinstance(n, ast.Call) and ast.unparse(n.func) in {'C.save', 'save', 'replace'} for n in ast.walk(functions['event'])),
              'I/O evidence append recursively calls bridged save')
    return {'source': V.desc(source), 'replacement_max_attempts': 40, 'replacement_max_seconds': 5.0,
            'serialization_outside_retry': True, 'new_neural_calls_per_IO_retry': 0}


def audit_registration_only():
    original, ids, folds, _, original_audit = V.audit_registration()
    path = RECOVERY / 'registration.json'
    reg = V.read(path)
    V.require(V.read(RECOVERY / 'registration_anchor.json')['registration'] == V.desc(path), 'Recovery registration anchor differs')
    V.require(reg['original_registration'] == V.desc(OUT / 'registration.json') and
              reg['original_registration_anchor'] == V.desc(OUT / 'registration_anchor.json'), 'Recovery replaced original method registration')
    scope = validate_recovery_scope(reg)
    V.require(reg['sources'].get(V.desc(__file__)['path']) == V.digest(__file__), 'Recovery verifier not frozen in recovery registration')
    for name, sha in reg['sources'].items():
        V.checked({'path': name, 'sha256': sha})
    V.require(all(reg['sources'].get(name) == sha for name, sha in original['sources'].items()), 'Recovery omits/changes a frozen original source')
    budget = {'prior_completed_fits': 23, 'prior_completed_updates': 2760,
              'discarded_partial_fit_starts': 1, 'discarded_partial_updates': 82,
              'discarded_partial_gradient_passes': 82, 'discarded_partial_person_examples': 1066,
              'new_complete_fits': 7, 'new_updates': 840, 'new_gradient_passes': 840,
              'canonical_completed_models': 30, 'canonical_optimizer_updates': 3600,
              'canonical_gradient_passes': 3600, 'canonical_person_examples': 47880,
              'actual_fit_starts': 31, 'actual_optimizer_updates': 3682,
              'actual_training_gradient_passes': 3682, 'actual_person_examples': 48946,
              'producer_head_person_forward_calls': 2220, 'independent_replay_head_person_forward_calls': 1110}
    V.require(all(reg['budget'].get(k) == v for k, v in budget.items()), 'Recovery budget omits prior/discarded work or changes planned fitting/evaluation')
    prior = V.read(V.checked(reg['prior_training_ledger']))
    V.require(prior['fits_completed'] == 23 and prior['updates_completed'] == prior['gradient_passes_completed'] == 2760 and
              prior['registration'] == reg['original_registration'] and len(prior['fits']) == 23 and
              [identity(r) for r in prior['fits']] == canonical_identities()[:23], 'Preserved prior ledger is not the original23 complete fits')
    protected = reg['preserved_completed_fits']
    V.require([identity(r) for r in protected] == canonical_identities()[:23], 'Immutable23 reuse identities/order differ')
    for record, meta in zip(protected, prior['fits']):
        folder = OUT / f'runs/{record["condition"]}/seed_{record["seed"]}/fold_{record["fold"]}'
        complete_path = V.checked(record['complete'])
        complete = V.read(complete_path)
        V.require(complete_path == folder / 'complete.json' and complete['passed'] and complete['updates'] == 120 and
                  complete['registration'] == reg['original_registration'] and complete['artifacts'] == record['artifacts'] and
                  len(record['artifacts']) == 7, 'Preserved completion proof/artifact inventory differs')
        for descriptor in record['artifacts']:
            V.require(V.checked(descriptor).parent == folder and reg['sources'].get(descriptor['path']) == descriptor['sha256'], 'Prior fit artifact omitted/changed')
        V.require(reg['sources'].get(record['complete']['path']) == record['complete']['sha256'] and
                  {p.name for p in folder.iterdir() if p.is_file()} == FIT_NAMES and
                  V.read(folder / 'metadata.json') == meta, 'Immutable complete fit changed or acquired an extra artifact')
    V.require([identity(r) for r in reg['replacement_fits']] == canonical_identities()[23:], 'Registered remaining seven identities/order differ')
    failed_path = V.checked(reg['failed_action'])
    failed = V.read(failed_path)
    V.require(failed_path == OUT / 'actions/fit_all.json' and failed['status'] == 'failed' and
              failed['error_type'] == 'PermissionError' and '[WinError 5]' in failed['error'] and
              'progress.json.part' in failed['error'] and 'progress.json' in failed['error'], 'Original pre-forward failure marker changed')
    pause = V.read(V.checked(reg['pause_record']))
    V.require(pause['status'] == 'paused_by_user' and pause['actual_fit_starts'] == 24 and
              pause['actual_optimizer_updates_to_pause'] == pause['actual_training_gradient_passes_to_pause'] == 2842 and
              pause['partial_model_updates'] == 82 and pause['completed_models'] == 23 and
              pause['classification_evaluation_started'] is False, 'Preserved pause accounting/release barrier differs')
    snapshot = V.read(V.checked(reg['failure_snapshot']))
    snapshot_audit = validate_snapshot_links(snapshot, reg)
    archive_record = V.read(RECOVERY / 'archive_record.json')
    V.require(archive_record['failed_partial'] == reg['failed_partial'] and
              archive_record['all_five_files_byte_identical'] is True and archive_record['original_failed_action_unchanged'] is True and
              reg['sources'].get(V.desc(RECOVERY / 'archive_record.json')['path']) == V.digest(RECOVERY / 'archive_record.json'),
              'Archived movement receipt differs or was not source-bound')
    partial = reg['failed_partial']
    V.require(identity(partial) == FAILED_ID and len(partial['artifacts']) == 5 and
              {r['name'] for r in partial['artifacts']} == PARTIAL_NAMES, 'Partial failure archive identity or five-file inventory differs')
    original_folder = OUT / f'runs/{FAILED_ID[0]}/seed_33/fold_2'
    archive = RECOVERY / 'failed_EEGPT_seed33_fold2'
    V.require((WORK / partial['original_folder']).resolve() == original_folder and
              (WORK / partial['archive_folder']).resolve() == archive and
              {p.name for p in archive.iterdir() if p.is_file()} == PARTIAL_NAMES, 'Failure archive paths/files differ')
    for record in partial['artifacts']:
        V.require((WORK / record['original_path']).resolve() == original_folder / record['name'] and
                  V.checked(record['archived']) == archive / record['name'] and
                  reg['sources'].get(record['archived']['path']) == record['archived']['sha256'], 'Archived partial bytes/source binding differs')
    split = folds['fold_2']
    started = V.read(archive / 'started.json')
    V.require(identity(started) == FAILED_ID and started['train_ids'] == split['train'] and
              started['planned_updates'] == 120 and started['registration'] == reg['original_registration'], 'Discarded model fit-start/people binding differs')
    partial_audit = validate_partial(V.read(archive / 'progress.json'), V.read(archive / 'progress.json.part'),
                                     V.read(archive / 'train_history.json'), split['train'], reg['original_registration'])
    old_source = WORK / 'tridep_ccml/tridep_eegpt/train_eval.py'
    save_source = WORK / 'tridep_ccml/tridep_stage1/common.py'
    boundary = audit_preforward_boundary(old_source.read_text(encoding='utf-8'), save_source.read_text(encoding='utf-8'))
    import torch
    old_initial = torch.load(archive / 'initial.pt', map_location='cpu', weights_only=True)
    expected = V.expected_initial_state(33)
    V.require(identity(old_initial) == FAILED_ID and old_initial['registration'] == reg['original_registration'] and
              set(old_initial['state_dict']) == set(expected) and all(torch.equal(old_initial['state_dict'][k], expected[k]) for k in expected),
              'Discarded initial checkpoint does not match independent original CNN seed recipe')
    paired = torch.load(OUT / f'runs/{V.CONDITIONS[0]}/seed_33/fold_2/initial.pt', map_location='cpu', weights_only=True)
    V.require(all(torch.equal(old_initial['state_dict'][k], paired['state_dict'][k]) for k in expected), 'Discarded initial differs from immutable pairedLaBraM initial')
    return reg, original, ids, folds, {'original_source_audit': original_audit,
        'recovery_sources_verified': len(reg['sources']), 'preserved_complete_models_verified': 23,
        'preserved_complete_updates': 2760, 'preserved_complete_artifacts_verified': 184,
        'archived_failure_files_verified': 5, 'discarded_initial_matches_independent_and_paired_seed': True,
        'recovery_scope': scope, 'before_after_failure_snapshot': snapshot_audit,
        'partial_accounting': partial_audit, 'pending83_boundary': boundary,
        'math_AST': audit_math_ast(reg), 'IO_source': audit_io_source()}


def audit_io_events():
    path = RECOVERY / 'io_events.jsonl'
    events = V.rows(path)
    V.require(events and all(isinstance(r.get('time'), str) for r in events), 'Recovery I/O event evidence missing')
    active, installed, removed, retries, recovered = 0, 0, 0, 0, 0
    for event in events:
        kind = event['kind']
        if kind == 'bridge_installed':
            V.require(active == 0 and event['max_attempts'] == 40 and event['max_seconds'] == 5.0 and
                      event['max_sleep_seconds'] == .1 and event['retries_cover_model_computation'] is False, 'I/O bridge reentrant/unbounded/model retry claim')
            active = 1; installed += 1
        elif kind == 'bridge_removed':
            V.require(active == 1, 'I/O bridge removed without current installation')
            active = 0; removed += 1
        else:
            V.require(active == 1 and kind in {'immutable_reused', 'replace_retry', 'replace_recovered'}, 'Unresolved/unregistered I/O event in completed recovery')
            path_value = Path(event['path']).resolve()
            V.require(path_value.is_relative_to(OUT) or path_value.is_relative_to(REPORT), 'Recovery I/O event escaped output tree')
            if kind.startswith('replace_'):
                V.require(Path(event['temporary']).resolve() == path_value.with_suffix(path_value.suffix + '.part') and
                          isinstance(event['attempt'], int) and 1 <= event['attempt'] <= 40 and
                          math.isfinite(event['elapsed_seconds']) and event['elapsed_seconds'] >= 0 and
                          isinstance(event['temporary_sha256'], str) and len(event['temporary_sha256']) == 64, 'Replace event path/count/time/SHA differs')
            if kind == 'replace_retry':
                V.require(event['attempt'] < 40 and event['elapsed_seconds'] < 5.0 and
                          (event['error_type'] == 'PermissionError' or event['winerror'] == 5), 'I/O retry exceeds registered bounds/eligible errors')
                retries += 1
            elif kind == 'replace_recovered':
                V.require(event['attempt'] > 1, 'Recovered event without prior replacement failure')
                recovered += 1
    V.require(active == 0 and installed == removed and installed >= 1, 'Completed recovery left I/O bridge open')
    return {'events': V.desc(path), 'bridge_installations': installed, 'bridge_removals': removed,
            'replace_retry_events': retries, 'replace_recovered_events': recovered,
            'new_neural_calls_from_IO_retries': 0}


def audit_recovery(reg, original, ids, folds):
    registration = V.desc(RECOVERY / 'registration.json')
    source = V.desc(WORK / 'tridep_ccml/tridep_eegpt/recovery_fit.py')
    completion = V.read(RECOVERY / 'training_completion.json')
    V.require(completion['passed'] and completion['method_registration'] == reg['original_registration'] and
              completion['execution_registration'] == registration and completion['preserved_completed_models'] == 23 and
              completion['new_completed_models'] == 7 and completion['canonical_completed_models'] == 30 and
              completion['canonical_optimizer_updates'] == 3600 and completion['actual_fit_starts'] == 31 and
              completion['actual_optimizer_updates'] == completion['actual_training_gradient_passes'] == 3682 and
              completion['actual_person_examples'] == 48946 and completion['training_ledger'] == V.desc(OUT / 'training_ledger.json'), 'Recovery completion omits canonical/actual work or differs')
    bindings = completion['new_fit_bindings']
    V.require([identity(r) for r in bindings] == canonical_identities()[23:], 'Seven new completion bindings differ')
    for record in bindings:
        folder = OUT / f'runs/{record["condition"]}/seed_{record["seed"]}/fold_{record["fold"]}'
        V.require(V.checked(record['metadata']) == folder / 'metadata.json' and V.checked(record['complete']) == folder / 'complete.json', 'New fit binding points to another fit')
        meta = V.read(folder / 'metadata.json')
        V.require(meta['execution_registration'] == registration and meta['execution_source'] == source and
                  meta['registration'] == reg['original_registration'] and set(meta) - set(V.read(V.checked(reg['prior_training_ledger']))['fits'][0]) ==
                  {'execution_registration', 'execution_source'}, 'New fit execution provenance differs or other algorithm metadata was added')
    import torch
    archive = RECOVERY / 'failed_EEGPT_seed33_fold2'
    replacement = OUT / f'runs/{FAILED_ID[0]}/seed_33/fold_2'
    initial = torch.load(replacement / 'initial.pt', map_location='cpu', weights_only=True)
    discarded = torch.load(archive / 'initial.pt', map_location='cpu', weights_only=True)
    V.require(initial.keys() == discarded.keys() and all(initial[k] == discarded[k] for k in initial if k != 'state_dict') and
              initial['state_dict'].keys() == discarded['state_dict'].keys() and
              all(torch.equal(initial['state_dict'][k], discarded['state_dict'][k]) for k in initial['state_dict']), 'Replacement initial checkpoint differs from archived original')
    pending = V.read(archive / 'progress.json.part')
    prefix = validate_replacement_prefix(V.read(archive / 'train_history.json'), V.read(replacement / 'train_history.json'),
        np.load(replacement / 'batch_order.npy', allow_pickle=False), ids, pending['batch_subject_ids'])
    canonical = V.read(OUT / 'training_ledger.json')
    canonical_examples = sum(len(row['train_ids']) * row['epochs'] for row in canonical['fits'])
    V.require(canonical_examples == 47880, 'Canonical actual train-person example count differs')
    counts = validate_actual_counts({'canonical_completed_fits': canonical['fits_completed'],
        'canonical_optimizer_updates': canonical['updates_completed'], 'canonical_training_gradient_passes': canonical['gradient_passes_completed'],
        'canonical_train_person_examples': canonical_examples, 'discarded_fit_starts': 1, 'discarded_optimizer_updates': 82,
        'discarded_training_gradient_passes': 82, 'discarded_train_person_examples': 1066,
        'actual_fit_starts': len(canonical['fits']) + 1, 'actual_optimizer_updates': canonical['updates_completed'] + 82,
        'actual_training_gradient_passes': canonical['gradient_passes_completed'] + 82, 'actual_train_person_examples': canonical_examples + 1066})
    marker = V.read(OUT / 'actions/fit_all_recovery.json')
    V.require(marker['status'] == 'complete' and marker['registration'] == reg['original_registration'] and marker['execution_registration'] == registration,
              'Real recovery completion action not complete or source bound')
    redirect = V.read(RECOVERY / 'evaluation_dependency_redirect.json')
    V.require(redirect['original_failed_action'] == reg['failed_action'] and redirect['new_completed_action'] == V.desc(OUT / 'actions/fit_all_recovery.json') and
              redirect['execution_registration'] == registration and redirect['old_failed_file_unchanged'] is True and redirect['only_this_dependency_redirected'] is True,
              'Uniform evaluation did not use the single explicit real recovery dependency view')
    V.require(V.read(V.checked(reg['failed_action']))['status'] == 'failed', 'Recovery rewrote old failure evidence')
    return {'training_completion': V.desc(RECOVERY / 'training_completion.json'), 'counts': counts,
            'new_execution_bound_complete_models': 7, 'replacement_prefix': prefix,
            'evaluation_dependency_redirect': V.desc(RECOVERY / 'evaluation_dependency_redirect.json'),
            'original_failed_action_preserved': True, 'all30_paired_initial_and_orders_checked_by_canonical_verifier': True,
            'IO_events': audit_io_events()}


def audit_closeout(reg, original):
    status = V.read(REPORT / 'status.json')
    V.require(status['status'] == 'complete' and status['passed'] and status['formal_fits'] == 31 and
              status['formal_updates'] == status['training_gradient_passes'] == 3682 and
              status['canonical_completed_models'] == 30 and status['canonical_optimizer_updates'] == 3600 and
              status['registration'] == reg['original_registration'] and status['execution_registration'] == V.desc(RECOVERY / 'registration.json') and
              status['budget'] == original['budget'] and status['recovery_budget'] == reg['budget'] and
              status['independent_verification'] == V.desc(REPORT / 'independent_verification.json') and
              status['recovery_independent_verification'] == V.desc(REPORT / 'recovery_independent_verification.json') and
              status['new_independent_people'] == 0 and status['stage6_14_people_reused_for_selection'] is False and
              status['original_stage4_outputs_modified'] is False and status['next_phase_started'] is False,
              'Final recovery status omits/changes actual31/3682 versus canonical30/3600 or evaluation roles')
    V.checked(status['report']); V.checked(status['acceptance'])
    inventory_path = REPORT / 'deliverable_hashes.json'
    inventory = V.read(inventory_path)
    actual = {p.relative_to(WORK).as_posix() for root in (OUT, REPORT) for p in root.rglob('*') if p.is_file() and p != inventory_path}
    V.require(set(inventory) == actual, 'Recovery final inventory omits/adds an output, archive or .part artifact')
    for name, sha in inventory.items():
        V.checked({'path': name, 'sha256': sha})
    V.require((RECOVERY / 'failed_EEGPT_seed33_fold2/progress.json.part').relative_to(WORK).as_posix() in inventory,
              'Final inventory omitted failed pre-forward temporary evidence')
    return {'passed': True, 'read_only': True, 'deliverable_hashes_verified': len(inventory),
            'includes_five_archived_partial_files_and_pending_part': True, 'actual31_fits3682_updates_disclosed': True}


def verify(post_closeout=False, registration_only=False, save_result=True):
    V.require(not (post_closeout and registration_only), 'Registration gate is not completed-experiment closeout')
    reg, original, ids, folds, registered = audit_registration_only()
    if registration_only:
        V.require(not (OUT / 'actions/fit_all_recovery.json').exists() and
                  all(not (OUT / f'runs/{r["condition"]}/seed_{r["seed"]}/fold_{r["fold"]}').exists() for r in reg['replacement_fits']),
                  'Initial recovery registration acceptance must precede every recovery fit start')
        result = {'passed': True, 'mode': 'registration_only', 'execution_registration': V.desc(RECOVERY / 'registration.json'),
                  'source': V.desc(__file__), 'registration_audit': registered, 'new_model_forwards_in_verifier': 0,
                  'new_fits_updates_in_verifier': 0, 'read_only': True}
        print(json.dumps({'passed': True, 'mode': 'registration_only', 'recovery_sources_verified': registered['recovery_sources_verified'],
                          'preserved_completed_models_verified': 23, 'new_model_forwards_in_verifier': 0}))
        return result
    canonical = V.verify(save_result=False if post_closeout else save_result, post_closeout=False)
    result = {'passed': True, 'mode': 'full', 'registration': reg['original_registration'],
              'execution_registration': V.desc(RECOVERY / 'registration.json'), 'source': V.desc(__file__),
              'canonical_independent_verification': V.desc(REPORT / 'independent_verification.json'),
              'registration_audit': registered, 'recovery_audit': audit_recovery(reg, original, ids, folds),
              'canonical_fits_verified': canonical['fit_audit']['fits_verified'],
              'new_model_forwards_in_verifier': 0, 'new_fits_updates_in_verifier': 0}
    target = REPORT / 'recovery_independent_verification.json'
    if target.exists():
        V.require(V.read(target) == result, 'Read-only recovery reverification differs from saved independent evidence')
    elif post_closeout:
        V.require(False, 'Missing saved recovery verification at post-closeout')
    elif save_result:
        V.save(target, result)
    if post_closeout:
        result = {**result, 'post_closeout': audit_closeout(reg, original)}
    print(json.dumps({'passed': True, 'mode': 'full', 'actual_fit_starts': 31, 'actual_optimizer_updates': 3682,
                      'canonical_completed_models': 30, 'canonical_optimizer_updates': 3600,
                      'new_model_forwards_in_verifier': 0, 'saved': save_result and not post_closeout,
                      'deliverable_hashes_verified': result.get('post_closeout', {}).get('deliverable_hashes_verified')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration-only', action='store_true')
    parser.add_argument('--post-closeout', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    options = parser.parse_args()
    verify(options.post_closeout, options.registration_only, not options.no_save)

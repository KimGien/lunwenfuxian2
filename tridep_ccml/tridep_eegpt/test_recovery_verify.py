"""Recovery audit rejection tests; no training or neural forward occurs."""
from __future__ import annotations

import ast
import copy
import unittest

import numpy as np

from . import recovery_verify as R
from . import recovery_fit as F


class RecoveryVerificationTests(unittest.TestCase):
    def setUp(self):
        self.registration = {'path': 'method.json', 'sha256': 'a' * 64}
        self.ids = [f'person{i:02d}' for i in range(37)]
        self.train = self.ids[:26]
        self.progress = {'epoch': 41, 'updates_started': 82, 'updates_completed': 82,
                         'gradient_passes_completed': 82, 'registration': self.registration}
        self.pending = {'epoch': 42, 'updates_started': 83, 'updates_completed': 82,
                        'gradient_passes_completed': 82, 'batch_subject_ids': self.train[:16],
                        'registration': self.registration}
        self.history = []
        for epoch in range(1, 61):
            batches = [{'batch_size': 16, 'train_ce': 1 / epoch}, {'batch_size': 10, 'train_ce': .5 / epoch}]
            self.history.append({'epoch': epoch, 'updates_completed': 2 * epoch,
                'gradient_passes_completed': 2 * epoch, 'batches': batches,
                'train_ce': sum(b['batch_size'] * b['train_ce'] for b in batches) / 26})
        self.order = np.tile(np.arange(26, dtype=np.int64), (60, 1))

    def test_partial_whole_epoch_and_exact_pending_boundaries(self):
        result = R.validate_partial(self.progress, self.pending, self.history[:41], self.train, self.registration)
        self.assertEqual(result['discarded_train_person_examples'], 1066)
        self.assertEqual(result['discarded_completed_updates'], 82)
        self.assertEqual(result['pending_pre_forward_update'], 83)

    def test_pending_started_is_not_counted_as_update_or_gradient(self):
        for field in ('updates_completed', 'gradient_passes_completed'):
            pending = copy.deepcopy(self.pending); pending[field] = 83
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Pending83'):
                R.validate_partial(self.progress, pending, self.history[:41], self.train, self.registration)

    def test_partial_rejects_nontrain_or_duplicate_pending_people(self):
        for last in (self.ids[30], self.train[0]):
            pending = copy.deepcopy(self.pending); pending['batch_subject_ids'][-1] = last
            with self.subTest(last=last), self.assertRaisesRegex(ValueError, 'Pending83'):
                R.validate_partial(self.progress, pending, self.history[:41], self.train, self.registration)

    def test_partial_history_checks_full_82_batches_and_weighted_CE(self):
        for mutation in ('gradient', 'loss', 'batchsize', 'missingepoch'):
            history = copy.deepcopy(self.history[:41])
            if mutation == 'gradient': history[20]['gradient_passes_completed'] -= 1
            if mutation == 'loss': history[20]['train_ce'] += 1e-9
            if mutation == 'batchsize': history[20]['batches'][1]['batch_size'] = 9
            if mutation == 'missingepoch': history.pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                R.validate_partial(self.progress, self.pending, history, self.train, self.registration)

    def test_replacement_prefix_checks_each_minibatch_not_only_epoch_mean(self):
        R.validate_replacement_prefix(self.history[:41], self.history, self.order, self.ids, self.train[:16])
        replacement = copy.deepcopy(self.history)
        # Preserve the weighted epoch mean while changing both batch CEs.
        replacement[20]['batches'][0]['train_ce'] += .001
        replacement[20]['batches'][1]['train_ce'] -= .0016
        with self.assertRaisesRegex(ValueError, 'first41'):
            R.validate_replacement_prefix(self.history[:41], replacement, self.order, self.ids, self.train[:16])

    def test_replacement_rejects_pending83_order_change(self):
        order = self.order.copy(); order[41, 0], order[41, 1] = order[41, 1], order[41, 0]
        with self.assertRaisesRegex(ValueError, 'pending83'):
            R.validate_replacement_prefix(self.history[:41], self.history, order, self.ids, self.train[:16])

    def test_actual_accounting_cannot_hide_discarded_fit_or_updates(self):
        expected = {'canonical_completed_fits': 30, 'canonical_optimizer_updates': 3600,
            'canonical_training_gradient_passes': 3600, 'canonical_train_person_examples': 47880,
            'discarded_fit_starts': 1, 'discarded_optimizer_updates': 82, 'discarded_training_gradient_passes': 82,
            'discarded_train_person_examples': 1066, 'actual_fit_starts': 31, 'actual_optimizer_updates': 3682,
            'actual_training_gradient_passes': 3682, 'actual_train_person_examples': 48946}
        self.assertEqual(R.validate_actual_counts(expected), expected)
        for field, omitted in (('actual_fit_starts', 30), ('actual_optimizer_updates', 3600),
                               ('actual_training_gradient_passes', 3600), ('actual_train_person_examples', 47880)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'accounting'):
                R.validate_actual_counts({**expected, field: omitted})

    def test_recovery_scope_cannot_add_model_retries_selection_or_change_user_authorization(self):
        scope = {'kind': 'registered_IO_failure_recovery', 'client_date': '2026-10-07',
            'user_resume_instruction_verbatim': '继续', 'model_execution_allowed': True,
            'model_math_or_hyperparameters_changed': False, 'restart_from_saved_optimizer_state': False,
            'old_failed_action_must_remain_failed': True, 'new_test_selection_or_threshold_searches': 0,
            'IO_only_retry': {'max_seconds': 5., 'max_attempts': 40, 'max_sleep_seconds': .1,
                'same_already_serialized_temporary_file': True, 'model_computation_retried': False}}
        self.assertTrue(R.validate_recovery_scope(scope)['only_serialized_replace_retried'])
        for field, changed in (('user_resume_instruction_verbatim', '暂停'), ('model_math_or_hyperparameters_changed', True),
                               ('new_test_selection_or_threshold_searches', 1), ('old_failed_action_must_remain_failed', False)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'authorization/scope'):
                R.validate_recovery_scope({**scope, field: changed})
        retry = copy.deepcopy(scope); retry['IO_only_retry']['model_computation_retried'] = True
        with self.assertRaisesRegex(ValueError, 'I/O-only'):
            R.validate_recovery_scope(retry)

    def snapshot_fixture(self):
        descriptors = [{'name': name, 'original_path': 'old/' + name,
                        'archived': {'path': 'archive/' + name, 'sha256': str(i) * 64}}
                       for i, name in enumerate(sorted(R.PARTIAL_NAMES), 1)]
        archived = {'condition': R.FAILED_ID[0], 'seed': 33, 'fold': 2,
                    'original_folder': 'old', 'archive_folder': 'archive', 'artifacts': descriptors}
        before = {**archived, 'artifacts': [{'name': d['name'], 'original_path': d['original_path'],
                   'expected_archive_path': d['archived']['path'], 'sha256_before_archive': d['archived']['sha256']}
                  for d in descriptors]}
        reg = {'original_registration': self.registration, 'failed_action': {'path': 'failed', 'sha256': 'b' * 64},
               'prior_training_ledger': {'path': 'prior', 'sha256': 'c' * 64}, 'preserved_completed_fits': ['23-immutable'],
               'failed_partial': archived}
        snapshot = {'old_registration': self.registration, 'failed_action': reg['failed_action'],
                    'prior_training_ledger': reg['prior_training_ledger'], 'preserved_completed_fits': reg['preserved_completed_fits'],
                    'failed_partial': before, 'completed_models': 23, 'completed_updates': 2760,
                    'partial_updates': 82, 'actual_updates_before_recovery': 2842, 'next_step_not_forwarded': 83,
                    'no_evaluation_or_scores_released': True, 'checkpoint_and_optimizer_at_step82_missing': True}
        return snapshot, reg

    def test_every_archived_file_SHA_matches_before_move_including_pending_part(self):
        snapshot, reg = self.snapshot_fixture()
        result = R.validate_snapshot_links(snapshot, reg)
        self.assertEqual(result['archived_files_with_exact_before_after_SHA256'], 5)
        for index in range(5):
            moved = copy.deepcopy(reg); moved['failed_partial']['artifacts'][index]['archived']['sha256'] = 'f' * 64
            with self.subTest(file=index), self.assertRaisesRegex(ValueError, 'before/after archival SHA256'):
                R.validate_snapshot_links(snapshot, moved)

    def test_failure_snapshot_cannot_substitute_prior_model_list_or_registration(self):
        snapshot, reg = self.snapshot_fixture()
        for field in ('old_registration', 'failed_action', 'prior_training_ledger', 'preserved_completed_fits'):
            changed = copy.deepcopy(snapshot); changed[field] = ['substituted']
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'original registration/failure/prior23'):
                R.validate_snapshot_links(changed, reg)

    def test_source_proves_original_failure_save_before_every_neural_operation(self):
        source = (R.WORK / 'tridep_ccml/tridep_eegpt/train_eval.py').read_text(encoding='utf-8')
        save = (R.WORK / 'tridep_ccml/tridep_stage1/common.py').read_text(encoding='utf-8')
        result = R.audit_preforward_boundary(source, save)
        self.assertEqual(result['pending83_forward_gradient_optimizer_calls'], 0)
        tree = ast.parse(source)
        fit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fit_all')
        epoch = R.epoch_node(fit)
        batch = next(n for n in epoch.body if isinstance(n, ast.For))
        index = next(i for i, n in enumerate(batch.body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and
                     ast.unparse(n.value.func) == 'C.save')
        pending_save = batch.body.pop(index)
        forward_index = next(i for i, n in enumerate(batch.body) if isinstance(n, ast.Assign) and
                             isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'logits')
        batch.body.insert(forward_index + 1, pending_save)
        with self.assertRaisesRegex(ValueError, 'precede'):
            R.audit_preforward_boundary(ast.unparse(tree), save)

    def test_original_save_must_not_swallow_atomic_replace_failure(self):
        source = (R.WORK / 'tridep_ccml/tridep_eegpt/train_eval.py').read_text(encoding='utf-8')
        save = '''def save(path, data, immutable=False):
    try:
        temporary.replace(path)
    except PermissionError:
        pass
'''
        with self.assertRaisesRegex(ValueError, 'propagate'):
            R.audit_preforward_boundary(source, save)

    def test_entire_fit_identical_after_only_four_registered_changes_removed(self):
        result = R.audit_math_ast({'math_ast_evidence': F.ast_evidence()})
        self.assertTrue(result['full_function_identical_after_four_authorized_changes_removed'])
        self.assertEqual(result['new_model_execution_in_AST_audit'], 0)

    def test_IO_retry_source_has_no_model_gradient_optimizer_or_serialization_call(self):
        result = R.audit_io_source()
        self.assertEqual(result['replacement_max_attempts'], 40)
        self.assertTrue(result['serialization_outside_retry'])
        self.assertEqual(result['new_neural_calls_per_IO_retry'], 0)


if __name__ == '__main__':
    unittest.main()

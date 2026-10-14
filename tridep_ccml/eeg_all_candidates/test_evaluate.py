"""Synthetic CPU evaluation checks; never load or forward outer-test data."""
from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import evaluate as E


def row(person, label, probability, role='test'):
    gap = math.log(probability/(1-probability))
    return {'subject_id': person, 'label': label, 'p_mdd': probability, 'p_hc': 1-probability,
            'prediction': E.predicted_class(probability), 'role': role, 'logits': [0., gap]}


class EvaluatorCPU(unittest.TestCase):
    def test_00_blocked_gate_precedes_every_constructor_or_input_plan(self):
        with patch.object(E.R, 'assert_test_unlocked', side_effect=ValueError('360 incomplete')) as gate, \
             patch.object(E, '_materialize_predictor') as model, patch.object(E, '_input_plan') as data, \
             patch.object(E.S, 'save') as save:
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                E.run()
            gate.assert_called_once_with()
            model.assert_not_called()
            data.assert_not_called()
            save.assert_not_called()

    def test_tie_is_hc_and_metrics_include_both_classes(self):
        records = [row('synthetic_a', 0, .5), row('synthetic_b', 1, .5)]
        value = E.metrics(records)
        self.assertEqual(E.predicted_class(.5), 0)
        self.assertEqual(E.predicted_class(.500000001), 1)
        self.assertEqual(value['confusion_matrix'], [[1, 0], [1, 0]])
        self.assertAlmostEqual(value['macro_f1'], 1/3)
        self.assertEqual(value['balanced_accuracy'], .5)
        self.assertEqual(value['mdd_recall'], 0.)
        self.assertEqual(value['auc'], .5)
        self.assertAlmostEqual(value['nll'], math.log(2))
        self.assertEqual(value['brier'], .25)

    def test_public_postconditions_also_gate_before_reading_predictions(self):
        for function, args in ((E.completed_condition_verified, ('LUNA-frozen-smallhead',)), (E.completed_report_verified, ())):
            with patch.object(E.R, 'assert_test_unlocked', side_effect=ValueError('360 incomplete')) as gate, patch.object(E.S, 'read') as read:
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    function(*args)
                gate.assert_called_once_with()
                read.assert_not_called()

    def test_wrong_gate_or_old_evaluator_source_cannot_reuse_completed_bundle(self):
        current = {'test_gate': {'path': 'synthetic_gate', 'sha256': 'a'*64},
                   'evaluator_source': {'path': 'synthetic_evaluate', 'sha256': 'b'*64}}
        for old in ({**current, 'test_gate': {'path': 'old_gate', 'sha256': 'c'*64}},
                    {**current, 'evaluator_source': {'path': 'synthetic_evaluate', 'sha256': 'd'*64}}):
            with patch.object(E.S, 'read', return_value={'passed': True, 'binding': old}), patch.object(E, '_check_sources') as sources:
                with self.assertRaisesRegex(ValueError, 'source binding'):
                    E._verified_completed_fit('synthetic_only', current, {}, {})
                sources.assert_not_called()

    def test_probability_binding_rejects_wrong_person_role_label_or_logit(self):
        value = [row('synthetic_a', 1, .75)]
        E._validate_rows(value, ['synthetic_a'], {'synthetic_a': 1}, 'test')
        with self.assertRaises(ValueError):
            E._validate_rows(value, ['synthetic_other'], {'synthetic_other': 1}, 'test')
        with self.assertRaises(ValueError):
            E._validate_rows(value, ['synthetic_a'], {'synthetic_a': 0}, 'test')
        corrupt = [{**value[0], 'p_mdd': .76}]
        with self.assertRaises(ValueError):
            E._validate_rows(corrupt, ['synthetic_a'], {'synthetic_a': 1}, 'test')

    def test_dirty_or_unverified_cached_batch_cannot_become_complete(self):
        binding = {'condition': 'LUNA-frozen-smallhead', 'selected_model_state_sha256': 'a'*64}
        dirty = {'binding': binding, 'frozen_state_verified': False, 'no_gradients_verified': True,
                 'model_state_before': 'a'*64, 'model_state_after': 'b'*64}
        with self.assertRaisesRegex(ValueError, 'unchanged-state'):
            E._validate_batch_value(dirty, binding)
        falsely_marked = {**dirty, 'frozen_state_verified': True}
        with self.assertRaises(ValueError):
            E._validate_batch_value(falsely_marked, binding)
        good = {**falsely_marked, 'model_state_after': 'a'*64}
        E._validate_batch_value(good, binding)
        for invalid in (None, 'same_unbound_word'):
            with self.assertRaises(ValueError):
                E._validate_batch_value({**good, 'model_state_before': invalid, 'model_state_after': invalid}, binding)

    def test_linux_mamba_bundle_runtime_is_checked_independently(self):
        binding = {'condition': 'EEGMamba-lastblock-finetune', 'selected_model_state_sha256': 'a'*64}
        value = {'binding': binding, 'frozen_state_verified': True, 'no_gradients_verified': True,
                 'model_state_before': 'a'*64, 'model_state_after': 'a'*64,
                 'runtime': {'platform': 'linux', 'torch': '2.5.1+cu124', 'mamba_ssm': '2.2.4',
                             'causal_conv1d': '1.5.0.post8', 'triton': '3.1.0'}}
        E._validate_batch_value(value, binding)
        value['runtime']['platform'] = 'win32'
        with self.assertRaisesRegex(ValueError, 'qualified official Linux'):
            E._validate_batch_value(value, binding)

    def test_all_cached_linux_mamba_recovers_missing_terminal_without_model(self):
        E.S.REPORT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='evaluate_cpu_', dir=E.S.REPORT) as temporary:
            destination = Path(temporary).resolve()
            self.assertTrue(destination.is_relative_to(E.S.REPORT.resolve()))
            split = {role: ['synthetic_' + role] for role in E.ROLES}
            labels = {person: index%2 for index, person in enumerate(sum(split.values(), []))}
            binding = {'condition': 'EEGMamba-lastblock-finetune', 'selected_model_state_sha256': 'a'*64,
                       'roles': split, 'checkpoint': {'path': 'synthetic_not_loaded', 'sha256': 'b'*64}}
            runtime = {'platform': 'linux', 'torch': '2.5.1+cu124', 'mamba_ssm': '2.2.4',
                       'causal_conv1d': '1.5.0.post8', 'triton': '3.1.0'}
            for role in E.ROLES:
                person = split[role][0]
                E.S.save(destination / f'{role}_batch_0.json', {'binding': binding,
                         'rows': [row(person, labels[person], .75, role)], 'runtime': runtime,
                         'model_state_before': 'a'*64, 'model_state_after': 'a'*64,
                         'frozen_state_verified': True, 'no_gradients_verified': True}, immutable=True)
            from . import linux_entry
            with patch.object(E, '_fit_context', return_value=({}, {'kind': 'raw'}, binding, destination)), \
                 patch.object(E.sys, 'platform', 'linux'), patch.object(linux_entry, '_require_runtime') as runtime_guard, \
                 patch.object(E, '_materialize_predictor') as model:
                completed, records = E._evaluate_fit(binding['condition'], 11, 1, 'best', {}, {}, split, labels, {})
                model.assert_not_called()
                runtime_guard.assert_called_once_with()
            self.assertTrue(completed['passed'])
            self.assertEqual(len(records), 3)
            ledger = E.S.read(E.S.checked(completed['attempts'][0]))
            self.assertEqual(ledger['status'], 'complete')
            self.assertTrue(ledger['cache_finalize_only'])
            self.assertEqual(ledger['terminal_verification_basis'], 'all_verified_cached_batches')
            self.assertEqual(ledger['outer_forward_started'], 0)
            self.assertEqual(ledger['encoder_forward_started'], 0)

    def test_primary_fold_mean_is_not_pooled_oof_score(self):
        ids = [f'synthetic_{i}' for i in range(37)]
        folds = [ids[:8], ids[8:16], ids[16:23], ids[23:30], ids[30:]]
        entries = []
        for seed in E.P.SEEDS:
            for fold, people in enumerate(folds, 1):
                records = [row(person, index%2, .9 if (index%2 and fold != 1) else .1) for index, person in enumerate(people)]
                value = E.metrics(records)
                entries.append({'seed': seed, 'fold': fold, 'rows': records,
                                'result': {'metrics': {role: value for role in E.ROLES}}})
        result = E.condition_summary(entries, ids)
        expected = sum(E.metrics(e['rows'])['macro_f1'] for e in entries if e['seed'] == 11)/5
        self.assertAlmostEqual(result['primary']['macro_f1'], expected)
        self.assertEqual(result['oof_predictions'], 111)
        self.assertEqual(result['distinct_subjects'], 37)
        self.assertFalse(result['repeated_seed_predictions_are_independent_people'])
        self.assertEqual(result['seeds']['11']['oof_metrics']['subjects'], 37)

    def test_predefined_retention_and_matched_controls(self):
        def summary(score, ba, recall, nll, brier):
            value = {'macro_f1': score, 'balanced_accuracy': ba, 'mdd_recall': recall, 'auc': .6, 'nll': nll, 'brier': brier}
            return {'primary': value, 'seeds': {str(s): {'fold_mean': value} for s in E.P.SEEDS}}
        baseline = summary(.4, .5, .7, .7, .25)
        self.assertTrue(E.retention(summary(.42, .5, .66, .6, .2), baseline)['passed'])
        self.assertFalse(E.retention(summary(.42, .5, .64, .6, .2), baseline)['passed'])
        self.assertEqual(E.matched_control('EEGPT-frozen-smallhead'), 'LaBraM-native4s-frozen-smallhead')
        self.assertEqual(E.matched_control('BIOT-lastblock-finetune'), 'LaBraM-native5s-lastblock-finetune')


if __name__ == '__main__':
    unittest.main()

"""Checks for the user-requested post-test retention revision.

Synthetic tests cover decision boundaries and rejection paths.  Integration
checks use existing sealed results, when present, without altering them.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import retention_revision as revision


ORIGINAL_BASELINE = 0.4494011544011543
EXPECTED_RETAINED = {
    'LUNA-frozen-smallhead',
    'CodeBrain-frozen-smallhead',
    'CodeBrain-lastblock-finetune',
    'BIOT-lastblock-finetune',
    'LaBraM-native4s-originalCNN-300ep',
    'LaBraM-native5s-frozen-smallhead',
    'LaBraM-native4s-frozen-smallhead',
    'LaBraM-native5s-lastblock-finetune',
    'LaBraM-native4s-lastblock-finetune',
}


def synthetic_comparison():
    rows, conditions = [], {}
    for index in range(23):
        condition = f'condition_{index}'
        primary = {metric: 0.4 for metric in revision.METRICS}
        row = {'condition': condition, 'method': 'CSBrain',
               'primary_snapshot': 'best', 'matched_control': 'historical',
               'retention_passed': 'False', **primary}
        rows.append(row)
        conditions[condition] = {
            'primary_snapshot': 'best',
            'variants': {'best': {'primary': primary},
                         'later_higher_test_score': {'primary': {
                             **primary, 'macro_f1': 0.99}}},
        }
    return rows, {'conditions': conditions}, {
        'condition': 'original_S00', 'macro_f1': ORIGINAL_BASELINE}


class DecisionRuleTests(unittest.TestCase):
    def test_strict_unrounded_boundary(self):
        self.assertFalse(revision.keep(ORIGINAL_BASELINE, ORIGINAL_BASELINE))
        self.assertTrue(revision.keep(math.nextafter(ORIGINAL_BASELINE, 1), ORIGINAL_BASELINE))
        self.assertFalse(revision.keep(math.nextafter(ORIGINAL_BASELINE, 0), ORIGINAL_BASELINE))

    def test_invalid_scores_and_baselines_are_rejected(self):
        for invalid in (math.nan, math.inf, -math.inf, -0.01, 1.01):
            for score, baseline in ((invalid, ORIGINAL_BASELINE), (0.5, invalid)):
                with self.subTest(score=score, baseline=baseline):
                    with self.assertRaises(ValueError):
                        revision.keep(score, baseline)

    def test_primary_snapshot_is_preserved_despite_better_test_alternative(self):
        rows, evaluation, baseline = synthetic_comparison()
        results = revision.classify(rows, evaluation, baseline)
        self.assertTrue(all(row['primary_snapshot'] == 'best' for row in results))
        self.assertTrue(all(not row['retention_passed'] for row in results))
        self.assertEqual(results[0]['historical_retention_passed'], 'False')
        self.assertTrue(all(row['post_test_development_decision'] for row in results))

    def test_metric_disagreement_is_rejected(self):
        rows, evaluation, baseline = synthetic_comparison()
        rows[0]['macro_f1'] = 0.6
        with self.assertRaisesRegex(ValueError, 'metric disagreement'):
            revision.classify(rows, evaluation, baseline)

    def test_nonfinite_saved_metric_is_rejected(self):
        rows, evaluation, baseline = synthetic_comparison()
        rows[0]['macro_f1'] = math.nan
        with self.assertRaises(ValueError):
            revision.classify(rows, evaluation, baseline)

    def test_changed_primary_snapshot_is_rejected(self):
        rows, evaluation, baseline = synthetic_comparison()
        rows[0]['primary_snapshot'] = 'later_higher_test_score'
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            revision.classify(rows, evaluation, baseline)

    def test_duplicate_condition_is_rejected(self):
        rows, evaluation, baseline = synthetic_comparison()
        rows[-1] = copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError, 'unique conditions'):
            revision.classify(rows, evaluation, baseline)

    def test_unregistered_condition_is_rejected(self):
        rows, evaluation, baseline = synthetic_comparison()
        rows[-1]['condition'] = 'unregistered'
        with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
            revision.classify(rows, evaluation, baseline)


class SourceIntegrityTests(unittest.TestCase):
    def test_changed_source_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=revision.WORK, prefix='.retention-test-') as temporary:
            root = Path(temporary).resolve()
            source = root / 'source.json'
            source.write_text('{}', encoding='utf-8')
            descriptor = revision.descriptor(source, root)
            self.assertEqual(revision.checked(descriptor, root), source)
            source.write_text('{"changed":true}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                revision.checked(descriptor, root)

    def test_source_outside_workspace_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=revision.WORK, prefix='.retention-test-') as temporary:
            root = Path(temporary).resolve() / 'workspace'
            root.mkdir()
            outside = root.parent / 'outside.json'
            outside.write_text('{}', encoding='utf-8')
            descriptor = {'path': '../outside.json',
                          'sha256': hashlib.sha256(outside.read_bytes()).hexdigest()}
            with self.assertRaisesRegex(ValueError, 'outside workspace'):
                revision.checked(descriptor, root)


@unittest.skipUnless((revision.WORK / revision.BASELINE).is_file()
                     and (revision.WORK / revision.EVALUATION / 'complete.json').is_file(),
                     'Historical local experiment is not available')
class SealedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = revision.WORK
        cls.baseline, cls.test_ids = revision.baseline_evidence(cls.root)
        cls.evaluation = revision.read_json(cls.root / revision.EVALUATION / 'evaluation_report.json')
        cls.rows = list(csv.DictReader(
            (cls.root / revision.REPORT / 'unified_comparison.csv').read_text(
                encoding='utf-8-sig').splitlines()))
        cls.revised = revision.classify(cls.rows, cls.evaluation, cls.baseline)

    def test_baseline_is_original_s00_and_recalculated_from_predictions(self):
        self.assertAlmostEqual(self.baseline['macro_f1'], ORIGINAL_BASELINE, places=15)
        self.assertNotAlmostEqual(self.baseline['macro_f1'], 0.4826, places=4)
        self.assertNotAlmostEqual(self.baseline['macro_f1'], 0.4404, places=4)
        self.assertEqual(self.baseline['experiment'], 'B0-37')
        self.assertTrue(self.baseline['prediction_recalculation_passed'])

    def test_substituted_baseline_summary_is_rejected(self):
        original_reader = revision.read_json
        for incorrect_baseline in (0.4826, 0.4404):
            def substituted_reader(path):
                value = original_reader(path)
                if path == self.root / revision.BASELINE:
                    value['aggregate']['eeg']['macro_f1']['mean_of_seed_fold_means'] = incorrect_baseline
                return value
            with self.subTest(incorrect_baseline=incorrect_baseline):
                with patch.object(revision, 'read_json', side_effect=substituted_reader):
                    with self.assertRaisesRegex(ValueError, 'Baseline mean mismatch'):
                        revision.baseline_evidence(self.root)

    def test_expected_nine_routes_are_retained(self):
        self.assertEqual({row['condition'] for row in self.revised if row['retention_passed']},
                         EXPECTED_RETAINED)

    def test_duplicate_retained_model_completion_is_rejected(self):
        evaluation = copy.deepcopy(self.evaluation)
        descriptor = next(desc for desc in evaluation['probability_batch_artifacts']
                          if '/LUNA-frozen-smallhead/' in desc['path'])
        evaluation['probability_batch_artifacts'].append(descriptor)
        with self.assertRaisesRegex(ValueError, 'duplicate retained completion'):
            revision.retained_artifacts(self.revised, evaluation, self.test_ids, self.root)

    def test_mismatched_retained_snapshot_binding_is_rejected(self):
        original_reader = revision.read_json
        def changed_snapshot_reader(path):
            result = original_reader(path)
            if '/LUNA-frozen-smallhead/' in path.as_posix():
                result['binding']['snapshot'] = 'unregistered_snapshot'
            return result
        with patch.object(revision, 'read_json', side_effect=changed_snapshot_reader):
            with self.assertRaisesRegex(ValueError, 'snapshot mismatch'):
                revision.retained_artifacts(self.revised, self.evaluation, self.test_ids, self.root)

    def test_readonly_build_preserves_sources_and_retains_all_folds(self):
        complete_path = self.root / revision.EVALUATION / 'complete.json'
        complete = revision.read_json(complete_path)
        sources = [complete_path, *(self.root / complete[key]['path'] for key in (
            'evaluation_report', 'comparison_table', 'oof_probabilities',
            'nine_paper_report', 'evaluator_source')),
            *(self.root / desc['path'] for desc in self.baseline['sources'])]
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        with patch.object(Path, 'write_bytes', side_effect=AssertionError('build must be read-only')), \
             patch.object(Path, 'write_text', side_effect=AssertionError('build must be read-only')), \
             patch.object(Path, 'replace', side_effect=AssertionError('build must be read-only')):
            products, current = revision.build(self.root)
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
        policy = json.loads(products[revision.POLICY_FILE])
        manifest = json.loads(products[revision.MODELS_FILE])
        self.assertEqual(set(policy['retained_conditions']), EXPECTED_RETAINED)
        self.assertEqual(current['retained_candidate_family_count'], 3)
        self.assertEqual(manifest['model_count'], 135)
        self.assertEqual({(model['condition'], model['seed'], model['fold'])
                          for model in manifest['models']},
                         {(condition, seed, fold) for condition in EXPECTED_RETAINED
                          for seed in (11, 22, 33) for fold in (1, 2, 3, 4, 5)})
        self.assertTrue(policy['post_test_development_reclassification'])
        self.assertFalse(policy['checkpoint_selection_changed'])
        self.assertFalse(policy['fusion_or_deployment_changed'])
        for name, descriptor in current['artifacts'].items():
            self.assertEqual(hashlib.sha256(products[name]).hexdigest(), descriptor['sha256'])


if __name__ == '__main__':
    unittest.main()

"""Pure evidence-shape regressions; no full review or completion certificate."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time
import unittest

REPORT = Path(__file__).resolve().parent
WORK = REPORT.parents[2]
SOURCE = REPORT / 'review_csbrain_full_finetune_15_v2.py'
REGISTRATION = WORK / 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/registration.json'
registry_bytes = REGISTRATION.read_bytes()
assert hashlib.sha256(registry_bytes).hexdigest() == '27bf2972c08f77457959cf3cba5bc6056c320f223c2426af5da3ae88d55ce090'
REGISTRY = json.loads(registry_bytes)
spec = importlib.util.spec_from_file_location('csbrain_full15_v2_shape_only', SOURCE)
MODULE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MODULE)
FT = [f for f in REGISTRY['fits'] if f['stage'] == '3']
BASE = [f for f in FT if f['condition'] in ('CSBrain-lastblock-finetune', 'EEGPT-lastblock-finetune')]
EXTRA = [f for f in FT if f not in BASE]


def identity_row(fit):
    return {**{k: fit[k] for k in ('fit_id', 'condition', 'seed', 'fold')},
            'passed': True, 'training_json_verified': True}


def shape_fixture(count):
    # Registered identities are genuine; these minimal shape fixtures deliberately
    # have no checkpoint descriptors, auditor/source proof or completed metadata.
    # They cannot be used by review() and are not evidence of actual completion.
    fits = BASE + EXTRA[:count - 30]
    ids = {f['fit_id'] for f in fits}
    return {'registered_total_fits': 360, 'registered_finetune_fits': 135,
            'training_complete_finetune_observed': count, 'verified_completed_finetune_fits': count,
            'logical_completed_fits_counted': count, 'checkpoint_loads': count * 3,
            'completed_rows': [identity_row(f) for f in fits],
            'pending_fits': [{'fit_id': f['fit_id']} for f in FT if f['fit_id'] not in ids],
            'errors': [], 'deferred_completed_fits': []}


def sync_pending(audit):
    ids = {r['fit_id'] for r in audit['completed_rows']}
    audit['pending_fits'] = [{'fit_id': f['fit_id']} for f in FT if f['fit_id'] not in ids]


class EvidenceShapeTests(unittest.TestCase):
    def reject(self, audit):
        with self.assertRaises(ValueError):
            MODULE.verify_cpu_evidence_shape(audit, REGISTRY)

    def test_registered_30_31_135_supersets_have_dynamic_counts(self):
        for count in (30, 31, 135):
            with self.subTest(count=count):
                out = MODULE.verify_cpu_evidence_shape(shape_fixture(count), REGISTRY)
                self.assertEqual(out['verified_global_fits'], count)
                self.assertEqual(out['observed_global_fits'], count)
                self.assertEqual(out['checkpoint_loads'], count * 3)
                self.assertEqual(out['pending_global_fits'], 135 - count)
                self.assertEqual({r['fit_id'] for r in out['CSBrain_rows']}, MODULE.EXPECTED_IDS)
                self.assertEqual(out['EEGPT_fits_covered'], 15)

    def test_missing_csbrain_replaced_with_registered_extra_rejected(self):
        a = shape_fixture(30)
        i = next(i for i, r in enumerate(a['completed_rows']) if r['condition'] == MODULE.CONDITION)
        a['completed_rows'][i] = identity_row(EXTRA[0])
        sync_pending(a)
        self.reject(a)

    def test_missing_eegpt_replaced_with_registered_extra_rejected(self):
        a = shape_fixture(30)
        i = next(i for i, r in enumerate(a['completed_rows']) if r['condition'] == 'EEGPT-lastblock-finetune')
        a['completed_rows'][i] = identity_row(EXTRA[0])
        sync_pending(a)
        self.reject(a)

    def test_duplicate_completed_identity_rejected(self):
        a = shape_fixture(31)
        a['completed_rows'][-1] = deepcopy(a['completed_rows'][0])
        self.reject(a)

    def test_wrong_loads_rejected(self):
        for loads in (90, 92, 94, 93.0):
            a = shape_fixture(31)
            a['checkpoint_loads'] = loads
            self.reject(a)

    def test_each_global_count_mismatch_rejected(self):
        for key in ('training_complete_finetune_observed', 'verified_completed_finetune_fits', 'logical_completed_fits_counted'):
            for value in (30, 32, 31.0, True):
                with self.subTest(key=key, value=value):
                    a = shape_fixture(31)
                    a[key] = value
                    self.reject(a)

    def test_errors_rejected(self):
        a = shape_fixture(30)
        a['errors'] = ['actual CPU proof failed']
        self.reject(a)

    def test_deferred_completed_fit_rejected(self):
        a = shape_fixture(30)
        a['deferred_completed_fits'] = [{'fit_id': EXTRA[0]['fit_id']}]
        self.reject(a)

    def test_bad_row_identity_or_result_rejected(self):
        for field, value in [('seed', 99), ('condition', 'EEGPT-lastblock-finetune'), ('passed', False), ('training_json_verified', False)]:
            a = shape_fixture(31)
            a['completed_rows'][-1][field] = value
            self.reject(a)

    def test_unregistered_identity_rejected(self):
        a = shape_fixture(31)
        a['completed_rows'][-1]['fit_id'] = 'REVE-lastblock-finetune/seed_99/fold_1'
        self.reject(a)

    def test_pending_missing_duplicate_or_completed_identity_rejected(self):
        for mutate in ('missing', 'duplicate', 'completed'):
            a = shape_fixture(31)
            if mutate == 'missing':
                a['pending_fits'].pop()
            elif mutate == 'duplicate':
                a['pending_fits'][-1] = deepcopy(a['pending_fits'][0])
            else:
                a['pending_fits'][-1] = {'fit_id': a['completed_rows'][0]['fit_id']}
            self.reject(a)

    def test_under30_rejected(self):
        a = shape_fixture(30)
        a['completed_rows'].pop()
        a.update(training_complete_finetune_observed=29, verified_completed_finetune_fits=29,
                 logical_completed_fits_counted=29, checkpoint_loads=87)
        sync_pending(a)
        self.reject(a)


def desc(path):
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EvidenceShapeTests))
    report = {'kind': 'csbrain_full15_v2_pure_stdlib_evidence_shape_tests', 'passed': result.wasSuccessful(),
              'created_at': datetime.now(timezone.utc).isoformat(), 'helper_source': desc(SOURCE), 'test_source': desc(Path(__file__)),
              'registration': desc(REGISTRATION), 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'test_output': stream.getvalue(), 'fixture_scope': 'Minimal evidence-shape fixtures use actual registered IDs, not actual completed checkpoint evidence.',
              'accepted_shape_counts': [30, 31, 135], 'actual_completion_or_full_stage_certificate_generated': False,
              'operations': {'full_review_calls': 0, 'full_stage_gate_calls': 0, 'actual_cpu_report_reads': 0,
                             'checkpoint_reads': 0, 'models_constructed': 0, 'GPU': 0, 'raw_or_features_read': 0,
                             'test_probability_reads': 0, 'training_actions': 0}}
    path = REPORT / ('csbrain_full15_v2_shape_stdlib_tests_' + str(time.time_ns()) + '.json')
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'artifact': desc(path), 'passed': report['passed'], 'tests_run': result.testsRun,
                      'failures': len(result.failures), 'errors': len(result.errors)}))
    if not result.wasSuccessful():
        print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())

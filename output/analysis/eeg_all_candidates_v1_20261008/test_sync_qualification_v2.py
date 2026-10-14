"""Stdlib compatibility/provenance tests; no actual guardian/device/data."""
import ast
import copy
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates import sync_qualification_gate_v2 as G
spec = importlib.util.spec_from_file_location('diagnostic_sync_v2_subject', Path(__file__).with_name('qualify_sync_reduction_v2.py'))
Q = importlib.util.module_from_spec(spec); spec.loader.exec_module(Q)


def shape_fixture():
    split = {'train': ['a'], 'val': ['b'], 'test': ['c']}
    original, q4 = {'path': 'original.json', 'sha256': 'a'}, {'path': 'q4.json', 'sha256': 'b'}
    rows = [{'fit_id': f'CSBrain-lastblock-finetune/seed_{seed}/fold_{fold}', 'condition': 'CSBrain-lastblock-finetune'}
            for seed in (11, 22, 33) for fold in range(1, 6)]
    proof = {'schema': 'sync_qualification_safe_boundary_v2', 'passed': True, 'queue': 'continuation_queue_v4',
             'registration': q4, 'original_training_registration': original, 'gpu_children_exit_verified': True,
             'held_dispatcher_running_suspended': True, 'gpu_idle': True, 'zero_optimizer_probe_only': True,
             'completed_CSBrain_fits_verified': 15, 'condition': 'CSBrain-lastblock-finetune', 'seed': 22, 'fold': 5,
             'registered_fold_ids': split}
    return proof, {'fits': rows}, q4, original, split


class CompatibilityTests(unittest.TestCase):
    def test_gate_ast_changes_exactly_two_literals(self):
        data = G.DELEGATE.read_bytes(); original = ast.parse(data); derived = G.compatibility_tree(data)
        inverse = {value: key for key, value in G.REPLACEMENTS.items()}
        for node in ast.walk(derived):
            if isinstance(node, ast.Constant) and type(node.value) is str and node.value in inverse:
                node.value = inverse[node.value]
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(derived, include_attributes=False))

    def test_diagnostic_ast_changes_only_gate_import(self):
        data = Q.DELEGATE.read_bytes(); original = ast.parse(data); derived = Q.compatibility_tree(data)
        for node in derived.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == 'sync_qualification_gate_v2': alias.name = 'sync_qualification_gate_v1'
        self.assertEqual(ast.dump(original, include_attributes=False), ast.dump(derived, include_attributes=False))

    def test_all_original_measurement_functions_unchanged(self):
        old = ast.parse(Q.DELEGATE.read_bytes()); new = Q.compatibility_tree(Q.DELEGATE.read_bytes())
        funcs = lambda tree: {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(funcs(old), funcs(new))
        self.assertEqual(Q.qualify.__code__.co_filename, str(Q.DELEGATE))
        self.assertIs(Q.D.G, G)
        self.assertEqual(Q.D.SOURCE, Q.SOURCE)
        self.assertEqual(G.GUARDIAN.name, 'sync_qualification_boundary_guard_v2.py')
        self.assertEqual(G.OWNER, Q.OWNER)

    def test_missing_or_duplicate_compatibility_anchor_rejected(self):
        with self.assertRaises(ValueError): G.compatibility_tree('pass')
        with self.assertRaises(ValueError): Q.compatibility_tree('pass')
        with self.assertRaises(ValueError):
            Q.compatibility_tree(Q.DELEGATE.read_text(encoding='utf-8') + '\nfrom tridep_ccml.eeg_all_candidates import sync_qualification_gate_v1 as G\n')

    def test_exact_ce_keeps_signed_zero_and_nan_rejection(self):
        self.assertTrue(Q.exact_ce({'ce': 1.25}, {'ce': 1.25}))
        self.assertFalse(Q.exact_ce({'ce': 0.0}, {'ce': -0.0}))
        self.assertFalse(Q.exact_ce({'ce': float('nan')}, {'ce': float('nan')}))


class BoundaryTests(unittest.TestCase):
    def test_v2_shape_requires_original_full_condition_and_held_q4(self):
        fixture = shape_fixture()
        self.assertEqual(len(G.verify_shape(*fixture)), 15)
        for key, value in [('schema', 'sync_qualification_safe_boundary_v1'), ('schema', 'deferred'),
                           ('completed_CSBrain_fits_verified', 14), ('completed_CSBrain_fits_verified', True),
                           ('gpu_children_exit_verified', False), ('held_dispatcher_running_suspended', False),
                           ('gpu_idle', False), ('zero_optimizer_probe_only', False), ('seed', 11), ('fold', True),
                           ('queue', 'continuation_queue_v5'), ('registered_fold_ids', {'train': ['test-person']})]:
            changed = {**fixture[0], key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): G.verify_shape(changed, *fixture[1:])

    def test_preflight_preserves_original_proof_and_adds_actual_delegated_sources(self):
        proof = shape_fixture()[0]
        result = (proof, {'fits': []}, {'seed': 22, 'fold': 5}, [{'epoch': 1}], {'guardian_live_verification': {'passed': True}})
        with patch.object(G.D, 'preflight', return_value=result) as check:
            actual = G.preflight({'path': 'fake.json', 'sha256': 'x'})
        self.assertIs(actual[0], proof)
        self.assertEqual(actual[:4], result[:4])
        self.assertEqual(actual[4]['actual_shared_gate_source'], G.descriptor(G.SOURCE))
        self.assertEqual(actual[4]['delegated_shared_gate_source']['sha256'], G.DELEGATE_SHA)
        check.assert_called_once()

    def test_strict_device_failure_is_not_silenced(self):
        with patch.object(G.D, 'preflight', side_effect=ValueError('GPU busy')):
            with self.assertRaisesRegex(ValueError, 'GPU busy'): G.preflight({})

    def test_all_graphics_compute_and_combined_contexts_refuse_profile(self):
        for kind in ('G', 'C', 'C+G'):
            observed = {'idle': False, 'contexts': [{'pid': 123, 'type': kind}]}
            with patch.object(G, 'load_gpu_observer', return_value=types.SimpleNamespace(gpu_observation=lambda: observed)):
                with self.subTest(kind=kind), self.assertRaises(ValueError): G.gpu_idle()
        with patch.object(G, 'load_gpu_observer', return_value=types.SimpleNamespace(gpu_observation=lambda: {'idle': True, 'contexts': []})):
            self.assertTrue(G.gpu_idle()['idle'])
        self.assertIs(G.D.gpu_idle, G.gpu_idle)

    def test_GPU_observer_source_is_pinned_and_import_is_stdlib_only(self):
        observer = G.load_gpu_observer()
        self.assertEqual(G.descriptor(Path(observer.__file__))['sha256'], G.GPU_OBSERVER_SHA)
        self.assertNotIn('torch', sys.modules)
        with patch.object(G, 'descriptor', return_value={'sha256': 'changed'}):
            with self.assertRaises(ValueError): G.load_gpu_observer()


class ProvenanceTests(unittest.TestCase):
    def test_source_and_schema_are_actual_v2(self):
        for suffix in ('attempt', 'pending', None):
            schema = 'diagnostic_sync_qualification' + ('_' + suffix if suffix else '') + '_v1'
            value = {'schema': schema, 'source_bindings': {'other': {'path': 'keep', 'sha256': 'a'}}}
            actual = Q.provenance(value)
            self.assertEqual(actual['schema'], schema[:-2] + 'v2')
            self.assertEqual(actual['source'], Q.D.desc(Q.SOURCE))
            self.assertEqual(actual['delegated_qualification_source']['sha256'], Q.DELEGATE_SHA)
            self.assertEqual(actual['shared_boundary_gate_source'], G.descriptor(G.SOURCE))
            self.assertEqual(actual['delegated_boundary_gate_source']['sha256'], G.DELEGATE_SHA)
            self.assertFalse(actual['formal_activation_authorized'])
            self.assertEqual(actual['source_bindings']['other'], value['source_bindings']['other'])
            self.assertEqual(value['schema'], schema)

    def test_lease_identity_record_is_not_relabelled(self):
        value = {'id': 'one', 'pid': 3, 'attempt': 'fake.json'}
        self.assertEqual(Q.provenance(value), value)

    def test_default_pending_never_checks_actual_boundary(self):
        records = []
        with patch.object(Q, 'ORIGINAL_SAVE', side_effect=lambda p, v, **kw: records.append(copy.deepcopy(v))), \
             patch.object(Q.D, 'desc', return_value={'path': 'in_memory', 'sha256': 'x'}), \
             patch.object(Q.D, 'qualify', side_effect=AssertionError('Cannot run actual qualifier')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(Q.main([]), 0)
        self.assertEqual(records[0]['schema'], 'diagnostic_sync_qualification_pending_v2')
        self.assertEqual(records[0]['model_constructions'], 0)
        self.assertNotIn('torch', sys.modules)

    def test_failed_boundary_persists_zero_work_before_tensor_import(self):
        records = []
        with patch.object(Q, 'ORIGINAL_SAVE', side_effect=lambda p, v, **kw: records.append(copy.deepcopy(v))), \
             patch.object(Q.D, 'preflight', side_effect=ValueError('not ready')):
            with self.assertRaisesRegex(ValueError, 'not ready'): Q.qualify({}, {})
        self.assertEqual([v['status'] for v in records], ['running', 'failed'])
        self.assertFalse(records[-1]['gpu_stage_entered'])
        self.assertEqual(records[-1]['latest_checkpoint_load_started'], 0)
        self.assertEqual(records[-1]['model_constructions_started'], 0)
        self.assertNotIn('torch', sys.modules)

    def test_second_boundary_failure_releases_exact_owned_lease(self):
        owner, records = {}, []
        class MemoryOwner:
            def exists(self): return bool(owner)
            def unlink(self): owner.clear()
        lease = MemoryOwner()
        def save(path, value, **kw):
            if path is lease: owner.update(value)
            records.append(copy.deepcopy(value))
        first = ({}, {}, {'condition': 'CSBrain-lastblock-finetune', 'seed': 22, 'fold': 5}, [], {})
        with patch.object(Q, 'ORIGINAL_SAVE', side_effect=save), patch.object(Q.D, 'OWNER', lease), \
             patch.object(Q.D, 'read', side_effect=lambda path: dict(owner)), \
             patch.object(Q.D, 'preflight', side_effect=[first, ValueError('late boundary failure')]):
            with self.assertRaisesRegex(ValueError, 'late boundary failure'): Q.qualify({}, {})
        self.assertFalse(owner)
        self.assertEqual(records[-1]['status'], 'failed')
        self.assertEqual(records[-1]['latest_checkpoint_load_started'], 0)
        self.assertFalse(records[-1]['gpu_stage_entered'])
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Independent math contracts; no encoder/head forward or fitting occurs."""
from __future__ import annotations
import hashlib
import math
import unittest
from pathlib import Path
import numpy as np
from . import verify as V


class IndependentMathTests(unittest.TestCase):
    def test_head_initial_state_matches_protected_historical_initials(self):
        for seed in V.SEEDS:
            state = V.expected_initial_state(seed)
            path = V.WORK / f'tridep_ccml/experiments/TriDep-Stage1-EEG-37-fulltext/runs/LaBraM-originalCNN/seed_{seed}/fold_1/metadata.json'
            self.assertEqual(V.state_digest(state), V.read(path)['initial_state_sha256'])
            self.assertEqual(sum(value.numel() for value in state.values()), 498434)

    def test_projection_independent_exact_seed_sign_recipe(self):
        from .assets import projection
        producer, audit = projection()
        oracle, q64 = V.fixed_projection()
        self.assertTrue(np.array_equal(oracle, producer))
        self.assertLess(np.abs(q64.T @ q64 - np.eye(200)).max(), 1e-12)
        self.assertEqual(hashlib.sha256(q64.tobytes()).hexdigest(), audit['float64_array_sha256'])
        row = np.linspace(-2, 2, 512)
        self.assertLessEqual(np.linalg.norm(row @ q64), np.linalg.norm(row) + 1e-12)

    def test_resampling_physical_duration_and_native_crop(self):
        from .extract import paired_inputs
        time = np.arange(1000, dtype=np.float64) / 200.
        raw = np.broadcast_to(np.sin(2 * np.pi * 12 * time).astype(np.float32), (60, 19, 1000)).copy()
        oracle_eegpt, oracle_labram = V.input_math(raw)
        result = paired_inputs(raw)
        self.assertTrue(np.array_equal(oracle_eegpt, result['eegpt']))
        self.assertTrue(np.array_equal(oracle_labram, result['labram']))
        self.assertEqual(oracle_eegpt.shape, (60, 19, 1024))
        self.assertTrue(np.array_equal(oracle_labram, raw[..., :800] / np.float32(100.)))
        expected = np.sin(2 * np.pi * 12 * np.arange(1024) / 256.)
        self.assertLess(np.abs(oracle_eegpt[0, 0, 30:990] * 1000 - expected[30:990]).max(), .003)
        changed = raw.copy(); changed[..., 800:] += 100.
        _, unchanged_labram = V.input_math(changed)
        self.assertTrue(np.array_equal(oracle_labram, unchanged_labram))
        self.assertFalse(np.array_equal(oracle_eegpt, V.input_math(changed)[0]))

    def test_block8_pooling_projection_same_first_repeat(self):
        from .extract import eegpt_interface, labram_interface
        generator = np.random.default_rng(9)
        tokens = generator.standard_normal((60, 16, 4, 512)).astype(np.float32)
        matrix, _ = V.fixed_projection()
        native, projected = V.pool_and_project(tokens, matrix)
        expected_native, expected = eegpt_interface(tokens, matrix)
        repeat_native, repeat = eegpt_interface(tokens[:8], matrix)
        self.assertTrue(np.array_equal(native, expected_native))
        self.assertTrue(np.array_equal(projected, expected))
        self.assertTrue(np.array_equal(native[:8], repeat_native))
        self.assertTrue(np.array_equal(projected[:8], repeat))
        constant = np.full((60, 16, 4, 512), np.float32(3.5))
        self.assertTrue(np.array_equal(V.pool_and_project(constant, matrix)[0], np.full((60, 512), np.float32(3.5))))
        labram = generator.standard_normal((60, 76, 200)).astype(np.float32)
        oracle = np.concatenate([labram[i:i+8].mean(axis=1, dtype=np.float32) for i in range(0, 60, 8)])
        self.assertTrue(np.array_equal(labram_interface(labram), oracle))

    def test_threshold_tie_hc_and_trueclass_saturated_nll(self):
        rows = [{'label': 0, 'p': .5}, {'label': 1, 'p': .5}]
        value = V.metric(rows, 'p')
        self.assertEqual(value['confusion_hc_mdd'], [[1, 0], [1, 0]])
        self.assertEqual(value['mdd_recall'], 0.)
        self.assertAlmostEqual(value['macro_f1'], 1 / 3)
        saturated = V.metric([{'label': 0, 'p': 0.}, {'label': 1, 'p': 0.}], 'p')
        self.assertAlmostEqual(saturated['nll'], (-math.log(1-1e-15)-math.log(1e-15))/2)
        self.assertEqual(saturated['brier'], .5)

    def test_probability_gain_cannot_bypass_four_classification_gates(self):
        control = dict.fromkeys(V.KEYS, .5); candidate = {**control, 'macro_f1': .509, 'nll': .4, 'brier': .4}
        a = {str(s): {'test': {**control, 'macro_f1': .6}} for s in V.SEEDS}
        b = {str(s): {'test': control} for s in V.SEEDS}
        result = V.gate(candidate, control, a, b)
        self.assertTrue(result['probability_improvement'])
        self.assertFalse(result['passed'])
        candidate.update(macro_f1=.52, mdd_recall=.44)
        self.assertFalse(V.gate(candidate, control, a, b)['passed'])


if __name__ == '__main__':
    unittest.main()

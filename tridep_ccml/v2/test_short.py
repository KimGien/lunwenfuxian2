import ast
from pathlib import Path
import unittest

import numpy as np

from .short_audio import short_window, pcm16_roundtrip, waveform_hash, replace_short_means


class ShortCoverageTests(unittest.TestCase):
    def test_real_length_is_preserved_without_padding_or_duplication(self):
        for length in [400, 14848, 79872, 79999]:
            y = np.linspace(-.8, .8, length, dtype=np.float32)
            self.assertIs(short_window(y), y)
            quantized = pcm16_roundtrip(y)
            self.assertEqual(quantized.shape, y.shape)
            self.assertLessEqual(float(np.abs(y-quantized).max()), 1/32768)
            self.assertEqual(waveform_hash(quantized), waveform_hash(pcm16_roundtrip(y)))

    def test_out_of_scope_length_and_nonfinite_input_fail(self):
        for length in [0, 399, 80000, 90000]:
            with self.assertRaises(ValueError):
                short_window(np.zeros(length, np.float32))
        with self.assertRaises(ValueError):
            short_window(np.full(14848, np.nan, np.float32))

    def test_only_empty_questions_change_and_unaffected_subjects_remain_exact(self):
        old = np.ones((37, 29, 768), dtype=np.float32)
        counts = np.ones((37, 29), dtype=np.int64)
        old[0, 0] = 0
        counts[0, 0] = 0
        replacement = {(0, 0): np.full(768, 3, dtype=np.float32)}
        new, new_counts = replace_short_means(old, counts, replacement)
        self.assertTrue(np.array_equal(new[1:], old[1:]))
        self.assertTrue(np.array_equal(new[0, 1:], old[0, 1:]))
        self.assertTrue(np.array_equal(new[0, 0], replacement[0, 0]))
        self.assertEqual(int(new_counts[0, 0]), 1)
        self.assertEqual(int(counts[0, 0]), 0)
        self.assertTrue(np.array_equal(old[0, 0], np.zeros(768, np.float32)))

    def test_missing_or_extra_replacement_cannot_silently_change_scope(self):
        old = np.zeros((37, 29, 768), dtype=np.float32)
        counts = np.ones((37, 29), dtype=np.int64)
        counts[0, 0] = 0
        for replacement in [{}, {(1, 0): np.ones(768, np.float32)},
                            {(0, 0): np.ones(768, np.float32), (1, 0): np.ones(768, np.float32)}]:
            with self.assertRaises(ValueError):
                replace_short_means(old, counts, replacement)

    def test_training_functions_identical_to_A_except_display_labels(self):
        root = Path(__file__).parent
        def functions(name):
            text = (root/name).read_text(encoding='utf-8')
            text = text.replace('Completed AS-{stage}', 'Completed A-{stage}').replace('Short-audio stage complete:', 'Audio stage complete:')
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
        old, new = functions('audio.py'), functions('short_train.py')
        for name in ['load_data', 'predictions', 'fit', 'train']:
            self.assertEqual(old[name], new[name])


if __name__ == '__main__':
    unittest.main()

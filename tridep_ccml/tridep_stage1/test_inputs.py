"""CPU-only contract tests; these never load a backbone or export EEG."""
import unittest

import numpy as np

from .inputs import (CHANNEL_NAMES, MONTAGE_NAMES, RAW_SHAPE, TOKEN_SHAPE,
                     _validate_partitions, build_adjacency, validate_raw)


class InputContractTests(unittest.TestCase):
    def test_graph_has_registered_nearest_neighbors_union_and_row_normalization(self):
        xyz = np.stack([np.arange(19), np.zeros(19), np.zeros(19)], axis=1)
        graph = build_adjacency(xyz)
        self.assertEqual(graph.shape, (19, 19))
        np.testing.assert_allclose(graph.sum(1), 1, rtol=0, atol=1e-7)
        self.assertTrue(np.all(np.diag(graph) > 0))
        np.testing.assert_array_equal(graph > 0, graph.T > 0)
        self.assertEqual(set(np.flatnonzero(graph[0])), {0, 1, 2, 3})
        # Stable tie ordering selects index 7 rather than 11 for the third
        # outgoing neighbor of index 9; union still preserves all reverse edges.
        self.assertGreater(graph[9, 7], 0)
        self.assertEqual(graph[0, 18], 0)
        self.assertTrue(np.all((graph > 0).sum(1) >= 4))

    def test_bad_coordinates_are_rejected(self):
        with self.assertRaises(ValueError):
            build_adjacency(np.zeros((18, 3)))
        xyz = np.zeros((19, 3)); xyz[0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_adjacency(xyz)

    def test_window_patch_axis_reconstruction_is_exact(self):
        raw = np.arange(np.prod(RAW_SHAPE), dtype=np.float32).reshape(RAW_SHAPE)
        patches = validate_raw(raw)
        self.assertEqual(patches.shape, TOKEN_SHAPE)
        np.testing.assert_array_equal(patches[4, 6, 2], raw[4, 6, 400:600])
        np.testing.assert_array_equal(patches.reshape(RAW_SHAPE), raw)

    def test_wrong_dtype_shape_and_nonfinite_raw_are_rejected(self):
        for array in (np.zeros(RAW_SHAPE, dtype=np.float64),
                      np.zeros((60, 19, 800), dtype=np.float32)):
            with self.assertRaises(ValueError):
                validate_raw(array)
        raw = np.zeros(RAW_SHAPE, dtype=np.float32); raw[7, 8, 9] = np.inf
        with self.assertRaises(ValueError):
            validate_raw(raw)

    def test_aliases_preserve_registered_order(self):
        self.assertEqual(len(CHANNEL_NAMES), len(set(MONTAGE_NAMES)))
        aliases = dict(zip(CHANNEL_NAMES, MONTAGE_NAMES))
        self.assertEqual({k: aliases[k] for k in ('FP2', 'FP1', 'T3', 'T4', 'T5', 'T6')},
                         {'FP2': 'Fp2', 'FP1': 'Fp1', 'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'})

    def test_leaking_split_is_rejected_before_feature_access(self):
        ids = [str(i) for i in range(37)]
        labels = {s: int(i < 16) for i, s in enumerate(ids)}
        # Duplicate one person across train and val: no source access is needed
        # to reject this otherwise plausible list of split names.
        folds = {f'fold_{i}': {'train': ids[:27], 'val': ids[26:29], 'test': ids[29:]}
                 for i in range(1, 6)}
        with self.assertRaisesRegex(ValueError, 'Person overlap'):
            _validate_partitions(folds, ids, labels)


if __name__ == '__main__':
    unittest.main()

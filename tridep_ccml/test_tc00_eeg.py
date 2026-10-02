"""Check original architecture, unchanged sampling, and validation-only fusion."""
import unittest
import torch
from torch import nn
from .tc00_eeg import TrackingDataset, models, original_seed, fuse, select_weights


class OriginalEEGTests(unittest.TestCase):
    def test_original_model_and_dropout(self):
        model = models.EEGConvPoolClassifier(input_dim=200, seq_len=60)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 498434)
        self.assertEqual(model.classifier[-1].out_features, 2)
        self.assertEqual([m.p for m in model.modules() if isinstance(m, nn.Dropout)], [.4, .4])
        self.assertFalse(any(isinstance(m, nn.LayerNorm) for m in model.modules()))

    def test_tracking_keeps_sampling_and_rng_identical(self):
        observations = []
        for dataset_type in [torch.utils.data.TensorDataset, TrackingDataset]:
            original_seed(11)
            loader = torch.utils.data.DataLoader(dataset_type(torch.arange(27), torch.arange(27)),
                                                  batch_size=16, shuffle=True, num_workers=0)
            order = []
            for _ in range(3):
                for x, y in loader:
                    order.extend(x.tolist())
            observations.append((order, torch.get_rng_state()))
            if isinstance(loader.dataset, TrackingDataset):
                self.assertEqual(loader.dataset.seen, order)
        self.assertEqual(observations[0][0], observations[1][0])
        self.assertTrue(torch.equal(observations[0][1], observations[1][1]))

    def test_fusion_test_labels_cannot_select_weights(self):
        rows = [dict(subject_id=str(i), seed=11, outer_fold=1, split=part, label=y,
                     p_eeg=p, p_speech=.4, p_text=.6) for i, (part, y, p) in enumerate([
                         ('val', 0, .1), ('val', 1, .9), ('test', 0, .8)])]
        before = [dict(r) for r in rows]
        corrected = {'0': .1, '1': .9, '2': .8}
        a, weights = fuse(rows, corrected)
        self.assertEqual(rows, before)
        rows[2]['label'] = 1
        b, second = fuse(rows, corrected)
        self.assertEqual(weights, second)
        self.assertEqual([r['p_final'] for r in a], [r['p_final'] for r in b])
        with self.assertRaises(ValueError):
            select_weights(rows)


if __name__ == '__main__':
    unittest.main()

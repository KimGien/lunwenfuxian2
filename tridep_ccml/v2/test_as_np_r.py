"""Check the sole treatment, parameter coverage and paired training implementation."""
import ast
import inspect
import unittest

import torch
from torch import nn
from . import as_np_r as candidate
from . import as_projection as baseline


class RegularizationTests(unittest.TestCase):
    def test_training_loop_has_only_registered_changes(self):
        expected = inspect.getsource(baseline.fit)
        expected = expected.replace("PARENT/'runs'/CONTROL/f'seed_{seed}/fold_{fold}/checkpoint.json'",
                                    "PARENT/f'runs/seed_{seed}/fold_{fold}/checkpoint.json'")
        expected = expected.replace("'weight_decay': 0.", "'weight_decay': cfg['weight_decay']")
        expected = expected.replace("""optimizer = torch.optim.Adamax([
            {'params': model.projections.parameters(), 'lr': cfg['projection_lr']},
            {'params': model.heads.parameters(), 'lr': cfg['head_lr']}], weight_decay=0.)""",
                                    'optimizer = make_optimizer(model, cfg)')
        expected = expected.replace('AS-NP ', 'AS-NP-R ')
        self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(inspect.getsource(candidate.fit))))
        self.assertIs(candidate.NormalizedProjectionHeads, baseline.NormalizedProjectionHeads)
        self.assertIs(candidate.load_data, baseline.load_data)
        self.assertIs(candidate.ccml_loss, baseline.ccml_loss)

    def test_all_projection_and_head_parameters_receive_decay(self):
        initial = {}
        for m, size in baseline.DIMENSIONS.items():
            initial[f'{m}.0.weight'] = torch.zeros(128, size)
            initial[f'{m}.0.bias'] = torch.zeros(128)
        model = candidate.NormalizedProjectionHeads(initial, 11)
        opt = candidate.make_optimizer(model, {'projection_lr': 1e-4, 'head_lr': 1e-3, 'weight_decay': 1e-4})
        grouped = [p for g in opt.param_groups for p in g['params']]
        self.assertEqual(len({id(p) for p in grouped}), len(grouped))
        self.assertEqual({id(p) for p in grouped}, {id(p) for p in model.parameters()})
        self.assertEqual(sum(p.numel() for p in grouped), 476294)
        self.assertEqual([g['weight_decay'] for g in opt.param_groups], [1e-4, 1e-4])
        self.assertEqual([g['lr'] for g in opt.param_groups], [1e-4, 1e-3])
        self.assertTrue(all(not m.training for m in model.modules()))

    def test_decay_changes_zero_gradient_weights_and_biases(self):
        for decay in [0., 1e-4]:
            model = nn.Module()
            model.projections = nn.Linear(1, 1).double()
            model.heads = nn.Linear(1, 1).double()
            for p in model.parameters():
                p.data.fill_(2.)
                p.grad = torch.zeros_like(p)
            opt = candidate.make_optimizer(model, {'projection_lr': 1e-4, 'head_lr': 1e-3, 'weight_decay': decay})
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            for group in opt.param_groups:
                expected = 2. - group['lr']*(decay*2.)/(decay*2.+group['eps'])
                for p in group['params']:
                    torch.testing.assert_close(p, torch.full_like(p, expected), rtol=0., atol=1e-12)


if __name__ == '__main__':
    unittest.main()

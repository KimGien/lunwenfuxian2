"""Engineering gates use synthetic/train data, never outer performance."""
from __future__ import annotations
import unittest
import tempfile
from unittest.mock import patch
from .common import *
from .models import R1Model, TokenPool, SparNet, REGIONS
from .data import fit_standardizer, check_splits
from .train import orders, step


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(11)

    def test_locked_subject_splits_and_batch_pairing(self):
        context = RecordContext()
        check_splits(context)
        for fold in FOLDS:
            ids = context.folds[f'fold_{fold}']['train']
            for seed in SEEDS:
                a, b = orders(ids, seed), orders(ids, seed)
                self.assertEqual(a, b)
                self.assertEqual(sum((len(x)+7)//8 for x in a), 240)
                self.assertTrue(all(sorted(x) == sorted(ids) for x in a))

    def test_status_replace_retries_transient_windows_reader_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'status.json'
            original = os.replace
            calls = []
            def fail_once(source, target):
                calls.append(1)
                if len(calls)==1:
                    raise PermissionError('simulated Windows reader handle')
                return original(source,target)
            with patch('os.replace',side_effect=fail_once):
                replace_json(path,{'epoch':28})
            self.assertEqual(read_json(path),{'epoch':28})
            self.assertEqual(len(calls),2)

    def test_standardization_ignores_nontrain_subjects(self):
        raw = np.random.default_rng(19).normal(size=(3,60,19,1000)).astype(np.float32)
        a = fit_standardizer(raw, [0,1])
        raw[2] = 1e12
        b = fit_standardizer(raw, [0,1])
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)))
        with self.assertRaises(ValueError):
            fit_standardizer(np.zeros_like(raw), [0,1])

    def test_region_axes_are_verified_label_permutation(self):
        names = [r['target_name'] for r in read_json(REVISION)['selection']]
        self.assertEqual(sorted(sum(REGIONS.values(), [])), sorted(names))
        self.assertNotIn('Cz', names)
        self.assertNotIn('Fz', names)
        self.assertIn('FCz', REGIONS['central'])
        self.assertIn('CPz', REGIONS['central'])

    def test_paired_initialization_and_S_control(self):
        from tridep_ccml.v2.scale import ScaleModel
        for seed in SEEDS:
            base = R1Model('C0', seed)
            for condition in ['C1','C2','C3','C4']:
                self.assertEqual(state_digest(base.shared_state()), state_digest(R1Model(condition,seed).shared_state()))
            a, b = R1Model('C3',seed), R1Model('C4',seed)
            shared = {k:v for k,v in b.eeg.state_dict().items() if k != 'query'}
            self.assertEqual(state_digest(a.eeg.state_dict()), state_digest(shared))
            old = ScaleModel(seed, False)
            x = {m:torch.randn(3,d) for m,d in zip(MODALITIES,[200,768,768])}
            torch.testing.assert_close(base.eval()(x), old(x), rtol=0, atol=0)

    def test_original_U0_architecture(self):
        from lib.models.models import EEGConvPoolClassifier
        model = R1Model('U0', 11).eval()
        old = EEGConvPoolClassifier(input_dim=200,seq_len=60,dropout=.4).eval()
        old.load_state_dict(model.eeg.state_dict(), strict=True)
        x = torch.randn(3,60,200)
        torch.testing.assert_close(model({'eeg':x}), old(x), rtol=0,atol=0)

    def test_spectrum_and_uniform_attention_axes(self):
        time = torch.arange(1000)/200
        x = torch.sin(2*torch.pi*10*time)[None,None,:]
        spec = SparNet.spectrum(x)
        self.assertEqual(spec.shape[-1],196)
        self.assertEqual(int(spec.argmax()),45)
        self.assertAlmostEqual(float(spec.max()),1.,places=5)
        token = TokenPool(11,True)
        with torch.no_grad():
            token.query.zero_()
        z = torch.randn(3,60,95,32)
        torch.testing.assert_close(token.pool(z),z.mean(-2),rtol=1e-5,atol=1e-6)
        # Pooling is local to a subject/window; changing another subject cannot alter it.
        before = token.pool(z[:1]).detach()
        z[1:] *= 1000
        torch.testing.assert_close(token.pool(z)[:1],before,rtol=0,atol=0)

    def test_independent_metrics_and_threshold_tie(self):
        from .evaluate import independent_metrics, metric
        from tridep_ccml.metrics import binary_metrics
        labels = [0,1,0,1,0,1]
        for ps in [[.5,.5,.5,.5,.5,.5],[.1,.9,.7,.3,.2,.8],[0.,1.,1.,0.,.5,.50001]]:
            a,b = binary_metrics(labels,ps),independent_metrics(labels,ps)
            for k in ['macro_f1','balanced_accuracy','mdd_recall','nll','brier']:
                self.assertAlmostEqual(a[k],b[k],places=12)
        self.assertEqual(independent_metrics([0,1],[.5,.5])['predicted_MDD'],0)

    def test_bn_mode_gradient_constraint_and_eval_independence(self):
        model = R1Model('U1',11).cuda().train()
        x = {'eeg':torch.randn(2,60,19,1000,device='cuda')}
        y = torch.tensor([0,1],device='cuda')
        self.assertTrue(model.eeg.bn1.training and model.eeg.drop1.training)
        before = model.eeg.bn1.running_mean.clone()
        step(model,model.optimizer(),x,y,1)
        self.assertFalse(torch.equal(before,model.eeg.bn1.running_mean))
        self.assertLessEqual(float(model.eeg.spatial.weight.flatten(1).norm(dim=1).max()),1.000001)
        self.assertLessEqual(float(model.classifier.weight.norm(dim=1).max()),.250001)
        self.assertLess(model.counts()['eeg_including_head'],50000)
        model.eval()
        state = state_digest(model.state_dict())
        with torch.inference_mode():
            a = model(x)
            b = torch.cat([model({'eeg':x['eeg'][i:i+1]}) for i in range(2)])
        torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
        self.assertEqual(state,state_digest(model.state_dict()))
        del model,x
        torch.cuda.empty_cache()


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""Small synthetic checks for consequential selection and scoring boundaries."""
import unittest
from .run import metrics, grid_select, gate

class ProtocolTests(unittest.TestCase):
    def test_tie_is_hc_and_loss_finite(self):
        m=metrics([dict(label=0,p=.5),dict(label=1,p=0.),dict(label=1,p=1.)])
        self.assertEqual(m['confusion_matrix'],[[1,0],[1,1]])
        self.assertGreater(m['nll'],10)
    def test_selection_rejects_test_and_mixed_folds(self):
        r=dict(role='test',seed=11,fold=1,subject_id='a',label=0,p_eeg=.1,p_speech=.1,p_text=.1)
        with self.assertRaises(ValueError):grid_select([r])
        r['role']='val'
        with self.assertRaises(ValueError):grid_select([r,dict(r,fold=2,subject_id='b')])
        with self.assertRaises(ValueError):grid_select([r,r])
    def test_tied_grid_and_st(self):
        rr=[dict(role='val',seed=11,fold=1,subject_id=str(i),label=y,p_eeg=p,p_speech=p,p_text=p)
            for i,(y,p) in enumerate([(0,.1),(1,.9),(0,.2)])]
        selected=grid_select(rr)
        self.assertEqual(selected['weights'],[0.,0.,1.])
        self.assertEqual(selected['ties'],231)
        self.assertTrue(selected['nonzero_eeg_tied_best'])
        self.assertEqual(grid_select(rr,True)['ties'],21)
    def test_auc_half_credit(self):
        self.assertEqual(metrics([dict(label=0,p=.4),dict(label=1,p=.4)])['auc'],.5)
    def test_gate_needs_two_seeds(self):
        primary={k:.5 for k in ('macro_f1','balanced_accuracy','mdd_recall','auc','nll','brier')}
        b=dict(primary=primary,seeds={str(s):dict(fold_mean=dict(macro_f1=.5)) for s in (11,22,33)})
        a=dict(primary=dict(primary,macro_f1=.52),seeds={str(s):dict(fold_mean=dict(macro_f1=.56 if s==11 else .5)) for s in (11,22,33)})
        self.assertFalse(gate(a,b)['passed'])

if __name__=='__main__':unittest.main()

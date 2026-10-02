import copy
import io
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from .common import label_indices, set_seed, LOSS
from .rebuild_eeg import EEGAdaptation, EEGConvPoolClassifier, loss_parts, validate_row
from tridep_ccml.train import make_initial_state, state_digest
from tridep_ccml.ccml import ccml_loss, dirichlet_from_evidence, evidence_dc
from .resume_eeg import version_metadata_loader


class RevisedEEGTests(unittest.TestCase):
    def test_version_metadata_reloads_with_restricted_loader(self):
        stream=io.BytesIO()
        expected=torch.tensor([1.,2.])
        torch.save({'state_dict':{'weight':expected},'metadata':{'torch':torch.__version__}},stream)
        stream.seek(0)
        before=list(torch.serialization.get_safe_globals())
        with version_metadata_loader():
            recovered=torch.load(stream,map_location='cpu',weights_only=True)
        self.assertTrue(torch.equal(expected,recovered['state_dict']['weight']))
        self.assertEqual(str(recovered['metadata']['torch']),str(torch.__version__))
        self.assertEqual(before,torch.serialization.get_safe_globals())

    def test_channel_selection_tracks_labels_not_positions(self):
        labels=['E1','E2','E3','E4']
        x=np.array([[10],[20],[30],[40]])
        required=['E3','E1']
        first=x[label_indices(labels,required)]
        perm=[2,0,3,1]
        second=x[perm][label_indices([labels[i] for i in perm],required)]
        self.assertTrue(np.array_equal(first,second))
        self.assertEqual(first[:,0].tolist(),[30,10])

    def test_missing_or_ambiguous_electrode_fails(self):
        for labels,required in [(['E1','E1'],['E1']),(['E1'],['E2']),(['E1'],['E1','E1'])]:
            with self.assertRaises(ValueError):label_indices(labels,required)

    def test_ccml_updates_eeg_and_heads_preserves_other_representations(self):
        set_seed(11)
        source=EEGConvPoolClassifier(input_dim=200,seq_len=60)
        model=EEGAdaptation(source.state_dict(),make_initial_state(11))
        x=torch.randn(3,60,200);cached=torch.randn(3,3,128);y=torch.tensor([0,1,0])
        original=cached.clone();before=state_digest(model.eeg.state_dict())
        self.assertTrue(torch.equal(model(x,cached),model(x,cached)))
        self.assertFalse(any(m.training for m in model.modules() if isinstance(m,torch.nn.Dropout)))
        optimizer=torch.optim.Adamax(model.parameters(),lr=1e-4)
        loss=ccml_loss(F.softplus(model(x,cached)),y,1,**LOSS)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertNotEqual(before,state_digest(model.eeg.state_dict()))
        rep=model.representations(x,cached)
        self.assertTrue(torch.equal(rep[:,1:],original[:,1:]))
        self.assertTrue(torch.equal(cached,original))
        self.assertIsNone(cached.grad)

    def test_recorded_loss_components_match_full_ccml(self):
        set_seed(22)
        e=F.softplus(torch.randn(4,3,2));y=torch.tensor([0,1,1,0])
        for epoch in [1,50,60]:
            self.assertAlmostEqual(sum(loss_parts(e,y,epoch).values()),float(ccml_loss(e,y,epoch,**LOSS)),places=5)

    def test_independent_evidence_validator_rejects_changed_probability(self):
        e=torch.tensor([[[.2,1.],[1.4,.7],[.5,.8]]])
        views=dirichlet_from_evidence(e);f=evidence_dc(e)
        row={'views':{m:{'e':e[0,v].tolist(),'alpha':views['alpha'][0,v].tolist(),
                        'p':views['p'][0,v].tolist(),'u':float(views['u'][0,v])}
                      for v,m in enumerate(['eeg','speech','text'])},
             **{k:f[k][0].tolist() for k in ['m','e_con','e_cmp','e_final']},
             'p_final':f['p'][0].tolist(),'u_final':float(f['u'][0]),'prediction':int(f['p'][0,1]>.5)}
        validate_row(row)
        broken=copy.deepcopy(row);broken['p_final'][0]+=.1
        with self.assertRaises(ValueError):validate_row(broken)


if __name__=='__main__':unittest.main()

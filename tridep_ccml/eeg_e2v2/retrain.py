"""Fresh EEGNet training on a legal subset; never reuse supervised E2v2 features.

Example API: fit_subset(fold=1, seed=11, fit_ids=[...], output_dir=Path(...)).
No training is performed by importing this module.
"""
from torch.nn import functional as F
from .training import Data, make_model, set_seed, state_digest, orders, step, health, predict, torch
from .common import *


def fit_subset(fold,seed,fit_ids,output_dir):
    """Train exactly 60 epochs from scratch; output_dir must be a new experiment directory."""
    output_dir=__import__('pathlib').Path(output_dir).resolve()
    workspace=WORK.resolve()
    assert workspace in output_dir.parents,'Output must stay in this workspace'
    assert OUT.resolve() not in output_dir.parents and output_dir!=OUT.resolve(),'Keep frozen E2v2 outputs unchanged'
    assert not output_dir.exists(),'Use a new output directory'
    assert read(OUT/'real_gate.json')['passed']
    data=Data();data.select(fold,'new')
    assert len(fit_ids)==len(set(fit_ids)) and len(fit_ids)>=2
    assert set(fit_ids)<=set(data.train),'Subset must be contained in this outer training set'
    assert set(data.labels[s] for s in fit_ids)=={0,1},'Both classes required'
    set_seed(seed);model=make_model('U1',seed,'S').cuda();initial=state_digest(model.state_dict())
    assert health(model,data,'U1','S',fit_ids)['passed']
    opt=model.optimizer();history=[]
    epoch_orders=orders(fit_ids,seed)
    for epoch,order in enumerate(epoch_orders,1):
        model.train();loss_sum=0.
        for i in range(0,len(order),8):
            ids=order[i:i+8];x,y=data.fit_batch(ids,'U1','S',fit_ids)
            loss,_=step(model,opt,x,y,epoch);loss_sum+=loss*len(ids)
        history.append({'epoch':epoch,'loss':loss_sum/len(fit_ids)})
    model.eval();final_health=health(model,data,'U1','S',fit_ids);assert final_health['passed']
    output_dir.mkdir(parents=True)
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save({'model':state,'fit_ids':fit_ids,'initial_state_sha256':initial,'state_sha256':state_digest(state),
                'seed':seed,'fold':fold},output_dir/'checkpoint.pt')
    save(output_dir/'training.json',{'fit_ids':fit_ids,'outer_fold':fold,'seed':seed,'epochs':60,
        'orders':epoch_orders,'history':history,'final_health':final_health,'augmentation':False,
        'input_manifest':desc(OUT/'input_manifest.json'),'new_random_initialization':True,
        'feature_shape':[len(fit_ids),128],'no_heldout_labels_used':True,'source':desc(__file__)})
    return model


@torch.no_grad()
def representation(model,raw_windows):
    """raw_windows: Bx60x19x1000 from the accepted cache; output: Bx128."""
    from tridep_ccml.eeg_p3.methods import subject_standardize
    normalized,_,_=subject_standardize(raw_windows)
    device=next(model.parameters()).device
    model.eval()
    return model.representations({'eeg':torch.from_numpy(normalized).to(device)})

"""CCML-only EEG adaptation, revised representations, and fresh-head control."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

from .common import (WORK, TC, OUT, LOSS, ADAPT, CONTROL, setup, descriptor,
                     verify_descriptor, read_json, require, set_seed, write_new, digest)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tridep_ccml.ccml import (ThreeViewHeads, ccml_loss, evidence_dc,
                            dirichlet_from_evidence, _loss_terms, expected_cross_entropy)
from tridep_ccml.train import make_initial_state, state_digest, load_features
from tridep_ccml.metrics import binary_metrics

sys.path.insert(0, str(WORK/'S00/upstream-dev'))
from lib.models.models import EEGConvPoolClassifier


class EEGAdaptation(nn.Module):
    def __init__(self, eeg_state, head_state):
        super().__init__()
        self.eeg = EEGConvPoolClassifier(input_dim=200, seq_len=60)
        self.eeg.load_state_dict(eeg_state, strict=True)
        self.eeg.classifier[-1] = nn.Identity()
        self.heads = ThreeViewHeads()
        self.heads.load_state_dict(head_state, strict=True)
        self.eval()  # Dropout disabled; eval does not disable gradients.

    def representations(self, eeg, frozen):
        return torch.cat((self.eeg(eeg).unsqueeze(1), frozen[:, 1:]), dim=1)

    def forward(self, eeg, frozen):
        return self.heads(self.representations(eeg, frozen))


def loss_parts(e, y, epoch):
    v, vr = _loss_terms(e, y, epoch, 50)
    fused = evidence_dc(e, 1.25)
    c, cr = _loss_terms(fused['e_con'], y, epoch, 50)
    cmp = expected_cross_entropy(fused['e_cmp']+1, F.one_hot(y, 2).to(e.dtype))
    return {'views_data': float(v.sum(1).mean()), 'views_kl_annealed': float(vr.sum(1).mean()),
            'consistent_data': float(c.mean()), 'consistent_kl_annealed': float(cr.mean()),
            'complementary_data': float(cmp.mean())}


def forward_all(model, data, eeg, device, adaptation):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(data), 8):
            x = torch.as_tensor(data[start:start+8], device=device)
            values.append((model(torch.as_tensor(eeg[start:start+8], device=device), x)
                           if adaptation else model(x)).cpu())
    return torch.cat(values)


def validate_row(row):
    e = np.array([row['views'][m]['e'] for m in ['eeg','speech','text']], dtype=float)
    require(e.shape == (3,2) and np.isfinite(e).all() and (e>=0).all(), 'Invalid evidence')
    def same(a,b):
        require(np.allclose(a,b,rtol=2e-5,atol=2e-6), 'Evidence arithmetic mismatch')
    for v,m in enumerate(['eeg','speech','text']):
        alpha=e[v]+1
        same(row['views'][m]['alpha'],alpha)
        same(row['views'][m]['p'],alpha/alpha.sum())
        same(row['views'][m]['u'],2/alpha.sum())
    minimum=e.min(0)
    power=(minimum/minimum.max())**1.25 if minimum.max()>0 else np.zeros(2)
    con=3*minimum.sum()*power/power.sum() if power.sum()>0 else np.zeros(2)
    cmp=(e-minimum).mean(0)
    final=con+cmp
    same(row['m'],minimum);same(row['e_con'],con);same(row['e_cmp'],cmp)
    same(row['e_final'],final);same(row['p_final'],(final+1)/(final+1).sum())
    same(row['u_final'],2/(final+1).sum())
    require(row['prediction']==int(row['p_final'][1]>.5), 'Prediction threshold mismatch')


def predict_rows(logits, data, context, seed, fold, stage, checkpoint, feature_source):
    e = F.softplus(logits)
    views = dirichlet_from_evidence(e)
    final = evidence_dc(e, 1.25)
    rows=[]
    for i,sid in enumerate(context.ids):
        row={'experiment': stage, 'seed':seed,'outer_fold':fold,'subject_id':sid,
             'split':data['splits'][i], 'label':int(data['labels'][i]),
             'train_ids':context.folds[f'fold_{fold}']['train'],
             'cohort_sha256':context.cohort_sha256,'folds_sha256':context.folds_sha256,
             'checkpoint':checkpoint,'feature_source':feature_source,
             'configuration':descriptor(OUT/'configuration.json'),
             'fusion_method':'ccml','rho':1.25,'delta':1.,'gamma':1.,
             'views':{m:{'e':e[i,v].tolist(),'alpha':views['alpha'][i,v].tolist(),
                          'p':views['p'][i,v].tolist(),'u':float(views['u'][i,v])}
                      for v,m in enumerate(['eeg','speech','text'])},
             **{k:final[k][i].tolist() for k in ['m','e_con','e_cmp','e_final']},
             'p_final':final['p'][i].tolist(),'u_final':float(final['u'][i]),
             'prediction':int(final['p'][i,1]>.5)}
        validate_row(row)
        rows.append(row)
    return rows


def fit(model, data, eeg, context, cfg, seed, fold, stage, provenance, device):
    folder=OUT/'runs'/stage/f'seed_{seed}'/f'fold_{fold}'
    folder.mkdir(parents=True,exist_ok=True)
    cp=folder/'checkpoint.pt'
    adaptation=stage==ADAPT
    initial_state=state_digest(model.state_dict())
    meta={'experiment':stage,'seed':seed,'fold':fold,
          'train_ids':context.folds[f'fold_{fold}']['train'],
          'cohort_sha256':context.cohort_sha256,'folds_sha256':context.folds_sha256,
          'configuration':descriptor(OUT/'configuration.json'),
          'runner':descriptor(__file__), 'common':descriptor(WORK/'tridep_ccml/v2/common.py'),
          'ccml':descriptor(TC/'ccml.py'), 'head_helpers':descriptor(TC/'train.py'),
          'upstream_model_source':descriptor(WORK/'S00/upstream-dev/lib/models/models.py'),
          'inputs':provenance,'initial_state_sha256':initial_state,
          'head_initialization_sha256':state_digest(make_initial_state(seed)),
          'device':device,'torch':torch.__version__,
          'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'trainable_eeg_parameters':sum(p.numel() for p in model.eeg.parameters()) if adaptation else 0,
          'trainable_head_parameters':774,'selection':'fixed_last_epoch','completed_epochs':60,
          'optimizer_updates':240,'dropout_mode':'eval', 'objective':'full_CCML_only'}
    if cp.exists():
        saved=torch.load(cp,map_location='cpu',weights_only=True)
        require(all(saved['metadata'].get(k)==v for k,v in meta.items()), 'Existing checkpoint provenance mismatch')
        model.load_state_dict(saved['state_dict'],strict=True)
        require(read_json(folder/'checkpoint.json')['checkpoint']==descriptor(cp),'Checkpoint sidecar mismatch')
        return model,cp,saved['metadata'],folder
    require(not (folder/'train_history.json').exists(),'Incomplete run requires inspection')
    model.to(device).eval()
    groups=([{'params':model.eeg.parameters(),'lr':cfg['eeg_lr']},
             {'params':model.heads.parameters(),'lr':cfg['head_lr']}]
            if adaptation else [{'params':model.parameters(),'lr':cfg['head_lr']}])
    optimizer=torch.optim.Adamax(groups,weight_decay=0.)
    lookup={s:i for i,s in enumerate(context.ids)}
    selected=[lookup[s] for s in meta['train_ids']]
    require(not(set(meta['train_ids']) & set(context.folds[f'fold_{fold}']['val']+context.folds[f'fold_{fold}']['test'])),'Training leakage')
    x=torch.as_tensor(data['features'][selected],device=device)
    y=torch.as_tensor(data['labels'][selected],dtype=torch.long,device=device)
    raw=torch.as_tensor(eeg[selected],device=device) if adaptation else None
    generator=torch.Generator(device='cpu').manual_seed(seed)
    batch_hash=hashlib.sha256();history=[];updates=0
    for epoch in range(1,61):
        order=torch.randperm(len(selected),generator=generator)
        batch_hash.update(order.numpy().astype('<i8').tobytes())
        total=0.;terms={};norms=[];head_norms={m:[] for m in ['eeg','speech','text']}
        for start in range(0,len(order),8):
            batch=order[start:start+8].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=model(raw[batch],x[batch]) if adaptation else model(x[batch])
            evidence=F.softplus(logits)
            loss=ccml_loss(evidence,y[batch],epoch,**LOSS)
            loss.backward()
            heads=model.heads.heads if adaptation else model.heads
            for name,head in heads.items():
                require(all(p.grad is not None for p in head.parameters()),'Missing head gradient')
                head_norms[name].append(float(torch.sqrt(sum((p.grad**2).sum() for p in head.parameters()))))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();updates+=1
            require(all(torch.isfinite(p).all().item() for p in model.parameters()),'Nonfinite parameters')
            total+=float(loss.detach())*len(batch);norms.append(float(norm))
            with torch.no_grad():
                part=loss_parts(evidence.detach(),y[batch],epoch)
            for k,v in part.items():terms[k]=terms.get(k,0.)+v*len(batch)
        history.append({'epoch':epoch,'train_loss':total/len(selected),
                        'loss_terms':{k:v/len(selected) for k,v in terms.items()},
                        'preclip_grad_norm_mean':float(np.mean(norms)),
                        'head_grad_norm_mean':{k:float(np.mean(v)) for k,v in head_norms.items()}})
    require(updates==240,'Unexpected update count')
    require(state_digest(model.state_dict())!=initial_state,'No model update occurred')
    meta['batch_order_sha256']=batch_hash.hexdigest()
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    with cp.open('xb') as stream:torch.save({'state_dict':state,'metadata':meta},stream)
    write_new(folder/'checkpoint.json',{'checkpoint':descriptor(cp),'metadata':meta})
    write_new(folder/'train_history.json',history)
    reloaded=torch.load(cp,map_location='cpu',weights_only=True)
    require(state_digest(reloaded['state_dict'])==state_digest(state),'Checkpoint state reload mismatch')
    before=forward_all(model,data['features'],eeg,device,adaptation)
    model.load_state_dict(reloaded['state_dict'],strict=True)
    after=forward_all(model,data['features'],eeg,device,adaptation)
    require(torch.equal(before,after),'Checkpoint prediction reload mismatch')
    write_new(folder/'reload_check.json',{'exact_state':True,'exact_all_subject_logits':True,'subjects':37})
    return model,cp,meta,folder


def run_one(context,cfg,eeg,eeg_manifest,seed,fold):
    start=time.perf_counter();set_seed(seed)
    old=load_features(context,seed,fold)
    original=old['manifest']['parent_checkpoints']['eeg']
    original_path=verify_descriptor(original)
    saved=torch.load(original_path,map_location='cpu',weights_only=True)
    require(saved['seed']==seed and saved['fold']==fold and saved['train_ids']==context.folds[f'fold_{fold}']['train'],'EEG warm start fit provenance mismatch')
    require(saved['cohort_id']==context.lock['cohort_id'] and saved['folds_sha256']==context.folds_sha256,'EEG warm start cohort mismatch')
    model=EEGAdaptation(saved['state_dict'],make_initial_state(seed))
    before_eeg=state_digest(model.eeg.state_dict())
    provenance={'corrected_eeg_manifest':descriptor(eeg_manifest),
                'historical_bundle':descriptor(TC/f'features/seed_{seed}/fold_{fold}.npz'),
                'historical_bundle_manifest':descriptor(TC/f'features/seed_{seed}/fold_{fold}.json'),
                'historical_parents':old['manifest']['parent_checkpoints']}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,cp,meta,folder=fit(model,old,eeg,context,cfg,seed,fold,ADAPT,provenance,device)
    model.to(device).eval()
    require(state_digest(model.eeg.state_dict())!=before_eeg,'EEG aggregation did not adapt')
    logits=forward_all(model,old['features'],eeg,device,True)
    rows=predict_rows(logits,old,context,seed,fold,ADAPT,descriptor(cp),provenance)
    write_new(folder/'predictions.json',rows)
    rep=[]
    with torch.no_grad():
        for i in range(0,37,8):
            rep.append(model.representations(torch.as_tensor(eeg[i:i+8],device=device),
                       torch.as_tensor(old['features'][i:i+8],device=device)).cpu().numpy())
    rep=np.concatenate(rep)
    require(rep.shape==(37,3,128) and np.isfinite(rep).all(),'Invalid revised representations')
    require(np.array_equal(rep[:,1:],old['features'][:,1:]),'Frozen speech/text changed')
    cache=OUT/f'features/seed_{seed}/fold_{fold}.npz';cache.parent.mkdir(parents=True,exist_ok=True)
    arrays={'features':rep,'labels':old['labels'],'subject_ids':np.array(context.ids),'splits':np.array(old['splits'])}
    if cache.exists():
        with np.load(cache,allow_pickle=False) as previous:
            require(all(np.array_equal(previous[k],v) for k,v in arrays.items()),'Existing revised cache differs')
    else:
        with cache.open('xb') as stream:np.savez_compressed(stream,**arrays)
    cache_meta={'features':descriptor(cache),'parent_checkpoint':descriptor(cp),
                'inputs':provenance,'fit_subject_ids':meta['train_ids'],
                'cohort_sha256':context.cohort_sha256,'folds_sha256':context.folds_sha256,
                'speech_text_exactly_unchanged':True,'frozen_eval':True,
                'fitted_objective':'CCML','historical_supervised_ancestry':'same-fold CE parents reused as warm start/frozen representations'}
    write_new(cache.with_suffix('.json'),cache_meta)
    write_new(folder/'complete.json',{'checkpoint':descriptor(cp),'predictions':descriptor(folder/'predictions.json'),
                                     'representations':descriptor(cache),'history':descriptor(folder/'train_history.json')})
    del model;torch.cuda.empty_cache()
    # Fresh-head control uses the new frozen parent; it does not continue adaptation heads.
    set_seed(seed)
    heads=ThreeViewHeads();heads.load_state_dict(make_initial_state(seed))
    data={**old,'features':rep}
    control_input={'revised_features':descriptor(cache),'revised_manifest':descriptor(cache.with_suffix('.json'))}
    heads,head_cp,head_meta,head_folder=fit(heads,data,None,context,cfg,seed,fold,CONTROL,control_input,'cpu')
    require(head_meta['batch_order_sha256']==meta['batch_order_sha256'],'Paired batch order mismatch')
    require(head_meta['head_initialization_sha256']==meta['head_initialization_sha256'],'Paired initialization mismatch')
    logits=forward_all(heads,rep,None,'cpu',False)
    rows=predict_rows(logits,data,context,seed,fold,CONTROL,descriptor(head_cp),control_input)
    write_new(head_folder/'predictions.json',rows)
    write_new(head_folder/'complete.json',{'checkpoint':descriptor(head_cp),'predictions':descriptor(head_folder/'predictions.json'),
                                          'history':descriptor(head_folder/'train_history.json')})
    test=[r for r in rows if r['split']=='test']
    score=binary_metrics([r['label'] for r in test],[r['p_final'][1] for r in test])
    print(f'Completed EEG adapt + CCML control seed={seed} fold={fold} F1={score["macro_f1"]:.6f} seconds={time.perf_counter()-start:.1f}',flush=True)


def run(seed=None,fold=None):
    context,cfg=setup()
    manifest=OUT/'eeg_manifest.json'
    entries=read_json(manifest)
    require([r['provenance']['subject_id'] for r in entries]==context.ids,'EEG cohort/order mismatch')
    for relative,sha in entries[0]['provenance']['backbone_code_hashes'].items():
        require(digest(WORK/relative)==sha,'Backbone implementation changed')
    for row in entries:
        verify_descriptor(row['feature'])
        for key in ['source_mat','source_set','revision','producer','common_source','backbone']:
            verify_descriptor(row['provenance'][key])
        require(row['provenance']['fit_subject_ids']==[],'Backbone cache illegally fitted')
    eeg=np.stack([np.load(WORK/r['feature']['path'],allow_pickle=False) for r in entries])
    require(eeg.shape==(37,60,200) and np.isfinite(eeg).all(),'Invalid EEG cache')
    require(seed is None or seed in cfg['seeds'],'Invalid seed')
    require(fold is None or fold in cfg['folds'],'Invalid fold')
    for s in ([seed] if seed else cfg['seeds']):
        for f in ([fold] if fold else cfg['folds']):run_one(context,cfg,eeg,manifest,s,f)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--fold',type=int)
    a=p.parse_args();run(a.seed,a.fold)

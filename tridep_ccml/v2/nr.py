"""Fixed independent normalization/regularization candidates on the revised parent."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

from .common import (WORK, TC, OUT as PARENT, CONTROL, LOSS, RecordContext,
                     descriptor, verify_descriptor, read_json, require, write_new, set_seed, digest)
from .resume_eeg import load_checkpoint as load_parent_checkpoint
from .rebuild_eeg import loss_parts, validate_row
from tridep_ccml.ccml import ThreeViewHeads, ccml_loss, evidence_dc, dirichlet_from_evidence
from tridep_ccml.train import make_initial_state, state_digest
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT=TC/'experiments/CCML-v2-NR-37-fulltext'
VARIANTS={'N':{'experiment':'CCML-v2-N-37-fulltext','normalization':'per_subject_per_view_LayerNorm_128_no_affine','eps':1e-5,'weight_decay':0.},
          'R':{'experiment':'CCML-v2-R-37-fulltext','normalization':'none','eps':None,'weight_decay':1e-4}}


class CandidateHeads(ThreeViewHeads):
    def __init__(self, variant):
        super().__init__()
        require(variant in ('C','N','R'),'Unsupported variant')
        self.variant=variant
        self.normalizer=nn.LayerNorm(128,eps=1e-5,elementwise_affine=False) if variant=='N' else nn.Identity()

    def forward(self, features):
        return super().forward(self.normalizer(features))


def configuration(context):
    return {'family':'CCML-v2-NR-37-fulltext','version':1,'cohort_sha256':context.cohort_sha256,
            'folds_sha256':context.folds_sha256,'subject_ids':context.ids,'seeds':[11,22,33],'folds':[1,2,3,4,5],
            'parent_family':PARENT.name,'control_experiment':CONTROL,
            'parent_status':descriptor(PARENT/'status.json'),'parent_audit':descriptor(PARENT/'completion_audit.json'),
            'plan':descriptor(WORK/'output/plans/CCML_v2_性能优化与后续计划.md'),
            'variants':VARIANTS,'epochs':60,'batch_size':8,'learning_rate':.001,'optimizer':'Adamax',
            'gradient_clip':1.,'ccml':LOSS,'trainable_parameters':774,'threshold':.5,'tie_class':0,
            'selection':'fixed_last_epoch','device':'cpu','threads':1,'formal_runs':30,'updates':7200,
            'validation_search':False,'parent_updated':False,'combined_N_plus_R':False,
            'stage_gate':{'mean_f1_delta_min':.01,'positive_f1_seeds_min':2,'recall_delta_min':-.05,
                          'nll_brier':'report deterioration separately; not a significance test'},
            'sources':{n:descriptor(p) for n,p in {
                'runner':Path(__file__),'common':TC/'v2/common.py','head_initialization':TC/'train.py',
                'ccml':TC/'ccml.py','diagnostics':TC/'v2/rebuild_eeg.py','parent_loader':TC/'v2/resume_eeg.py',
                'records':TC/'records.py'}.items()}}


def protect_history():
    path=OUT/'history_before.json'
    if not path.exists():
        tracked=read_json(PARENT/'historical_files_before.json')
        for p in PARENT.rglob('*'):
            if p.is_file():tracked[p.relative_to(WORK).as_posix()]=digest(p)
        for name in ['common.py','rebuild_eeg.py','resume_eeg.py','extract_eeg.py','complete_eeg.py','test_eeg.py']:
            p=TC/'v2'/name;tracked[p.relative_to(WORK).as_posix()]=digest(p)
        write_new(path,tracked)
    tracked=read_json(path)
    require(all(digest(WORK/p)==sha for p,sha in tracked.items()),'Historical artifact changed')
    return len(tracked)


def load_data(context,seed,fold):
    path=PARENT/f'features/seed_{seed}/fold_{fold}.npz';info=read_json(path.with_suffix('.json'))
    verify_descriptor(info['features']);verify_descriptor(info['parent_checkpoint'])
    require(info['cohort_sha256']==context.cohort_sha256 and info['folds_sha256']==context.folds_sha256,
            'Parent cohort mismatch')
    require(info['fit_subject_ids']==context.folds[f'fold_{fold}']['train'],'Parent train IDs mismatch')
    with np.load(path,allow_pickle=False) as archive:
        arrays={k:archive[k].copy() for k in ['features','subject_ids','labels','splits']}
    expected=[next(part for part,ids in context.folds[f'fold_{fold}'].items() if sid in ids) for sid in context.ids]
    require(arrays['subject_ids'].tolist()==context.ids and arrays['splits'].tolist()==expected,'Parent identity/split mismatch')
    require(arrays['labels'].tolist()==[context.labels[s] for s in context.ids],'Parent labels mismatch')
    require(arrays['features'].shape==(37,3,128) and arrays['features'].dtype==np.float32 and np.isfinite(arrays['features']).all(),'Invalid representations')
    arrays['source']={'features':descriptor(path),'manifest':descriptor(path.with_suffix('.json')),
                      'parent_checkpoint':info['parent_checkpoint']}
    return arrays


def logits_all(model,x):
    model.eval()
    with torch.no_grad():return torch.cat([model(torch.from_numpy(x[i:i+8])) for i in range(0,len(x),8)])


def preflight():
    context=RecordContext();OUT.mkdir(parents=True,exist_ok=True)
    count=protect_history();cfg=configuration(context);write_new(OUT/'configuration.json',cfg)
    status=read_json(PARENT/'status.json');require(status['status']=='complete' and status['control_runs']==15,'Parent control incomplete')
    verify_descriptor(status['audit']);verify_descriptor(status['report'])
    checks=[]
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            data=load_data(context,seed,fold)
            folder=PARENT/'runs'/CONTROL/f'seed_{seed}'/f'fold_{fold}'
            complete=read_json(folder/'complete.json')
            for item in complete.values():verify_descriptor(item)
            saved=load_parent_checkpoint(folder/'checkpoint.pt');meta=saved['metadata']
            require(meta['inputs']['revised_features']==data['source']['features'],'Control used different representations')
            require(meta['head_initialization_sha256']==state_digest(make_initial_state(seed)),'Control initialization mismatch')
            model=CandidateHeads('C');model.load_state_dict(saved['state_dict'],strict=True)
            logits=logits_all(model,data['features']);fused=evidence_dc(F.softplus(logits))
            rows=read_json(folder/'predictions.json')
            require(len(rows)==37 and [r['subject_id'] for r in rows]==context.ids,'Control prediction identity mismatch')
            for i,row in enumerate(rows):
                validate_row(row)
                require(np.array_equal(fused['p'][i].numpy(),np.array(row['p_final'])),'Unchanged candidate path fails exact control replay')
            checks.append({'seed':seed,'fold':fold,'rows_exactly_reproduced':37,'control_checkpoint':descriptor(folder/'checkpoint.pt')})
    write_new(OUT/'preflight.json',{'passed':True,'protected_files':count,'parent_checks':checks,
                                   'formal_training_started_by_preflight':False,'replayed_subject_rows':555})
    return context,cfg


def predictions(model,data,context,seed,fold,variant,checkpoint):
    logits=logits_all(model,data['features']);e=F.softplus(logits)
    view=dirichlet_from_evidence(e);f=evidence_dc(e);rows=[]
    for i,sid in enumerate(context.ids):
        row={'experiment':VARIANTS[variant]['experiment'],'variant':variant,'seed':seed,'outer_fold':fold,
             'subject_id':sid,'split':str(data['splits'][i]),'label':int(data['labels'][i]),
             'train_ids':context.folds[f'fold_{fold}']['train'], 'cohort_sha256':context.cohort_sha256,
             'folds_sha256':context.folds_sha256,'checkpoint':descriptor(checkpoint),
             'feature_source':data['source'],'configuration':descriptor(OUT/'configuration.json'),
             'fusion_method':'ccml','rho':1.25,'delta':1.,'gamma':1.,
             'views':{m:{'e':e[i,v].tolist(),'alpha':view['alpha'][i,v].tolist(),
                         'p':view['p'][i,v].tolist(),'u':float(view['u'][i,v])}
                      for v,m in enumerate(['eeg','speech','text'])},
             **{k:f[k][i].tolist() for k in ['m','e_con','e_cmp','e_final']},
             'p_final':f['p'][i].tolist(),'u_final':float(f['u'][i]),'prediction':int(f['p'][i,1]>.5)}
        validate_row(row);rows.append(row)
    return rows


def fit(context,cfg,seed,fold,variant):
    start=time.perf_counter();set_seed(seed);data=load_data(context,seed,fold)
    folder=OUT/'runs'/VARIANTS[variant]['experiment']/f'seed_{seed}'/f'fold_{fold}'
    folder.mkdir(parents=True,exist_ok=True)
    model=CandidateHeads(variant);initial=make_initial_state(seed);model.load_state_dict(initial,strict=True)
    require(sum(p.numel() for p in model.parameters())==774,'Parameter count changed')
    parent_side=read_json(PARENT/'runs'/CONTROL/f'seed_{seed}'/f'fold_{fold}/checkpoint.json')
    train_ids=context.folds[f'fold_{fold}']['train'];lookup={s:i for i,s in enumerate(context.ids)}
    selected=[lookup[s] for s in train_ids]
    require(not(set(train_ids)&set(context.folds[f'fold_{fold}']['val']+context.folds[f'fold_{fold}']['test'])),'Train leakage')
    meta={'experiment':VARIANTS[variant]['experiment'],'variant':variant,'seed':seed,'fold':fold,
          'configuration':descriptor(OUT/'configuration.json'),'source':data['source'],
          'control_checkpoint':parent_side['checkpoint'],'initialization_sha256':state_digest(initial),
          'train_ids':train_ids,'cohort_sha256':context.cohort_sha256,'folds_sha256':context.folds_sha256,
          'hyperparameters':VARIANTS[variant], 'trainable_parameters':774,'epochs':60,'updates':240,
          'objective':'full_CCML_only','device':'cpu','torch':str(torch.__version__),'selection':'last_epoch'}
    cp=folder/'checkpoint.pt'
    if cp.exists():
        side=read_json(cp.with_suffix('.json'));verify_descriptor(side['checkpoint'])
        saved=torch.load(cp,map_location='cpu',weights_only=True)
        require(saved['metadata']==side['metadata'] and all(saved['metadata'].get(k)==v for k,v in meta.items()),'Existing checkpoint provenance differs')
        model.load_state_dict(saved['state_dict'],strict=True)
        meta=saved['metadata']
    else:
        x=torch.from_numpy(data['features'][selected]);y=torch.from_numpy(data['labels'][selected])
        optimizer=torch.optim.Adamax([{'params':model.parameters(),'lr':.001}],weight_decay=VARIANTS[variant]['weight_decay'])
        generator=torch.Generator().manual_seed(seed);batch_hash=hashlib.sha256();history=[]
        for epoch in range(1,61):
            order=torch.randperm(len(selected),generator=generator);batch_hash.update(order.numpy().astype('<i8').tobytes())
            total=0.;terms={};grad=[];per_head={m:[] for m in ['eeg','speech','text']}
            for offset in range(0,len(order),8):
                b=order[offset:offset+8];optimizer.zero_grad(set_to_none=True)
                e=F.softplus(model(x[b]));loss=ccml_loss(e,y[b],epoch,**LOSS);loss.backward()
                for m,head in model.heads.items():
                    require(all(p.grad is not None for p in head.parameters()),'Missing gradient')
                    per_head[m].append(float(torch.sqrt(sum((p.grad**2).sum() for p in head.parameters()))))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();require(all(torch.isfinite(p).all() for p in model.parameters()),'Nonfinite parameter')
                total+=float(loss.detach())*len(b);grad.append(float(norm))
                with torch.no_grad():parts=loss_parts(e.detach(),y[b],epoch)
                for k,v in parts.items():terms[k]=terms.get(k,0.)+v*len(b)
            history.append({'epoch':epoch,'train_loss':total/len(selected),'loss_terms':{k:v/len(selected) for k,v in terms.items()},
                            'preclip_grad_norm_mean':float(np.mean(grad)),
                            'head_grad_norm_mean':{m:float(np.mean(v)) for m,v in per_head.items()}})
        meta['batch_order_sha256']=batch_hash.hexdigest()
        require(meta['batch_order_sha256']==parent_side['metadata']['batch_order_sha256'],'Control batch order differs')
        require(meta['initialization_sha256']==parent_side['metadata']['head_initialization_sha256'],'Control initialization differs')
        with cp.open('xb') as stream:torch.save({'state_dict':model.state_dict(),'metadata':meta},stream)
        write_new(cp.with_suffix('.json'),{'checkpoint':descriptor(cp),'metadata':meta})
        write_new(folder/'train_history.json',history)
        before=logits_all(model,data['features']);saved=torch.load(cp,map_location='cpu',weights_only=True)
        require(state_digest(saved['state_dict'])==state_digest(model.state_dict()),'Reload state mismatch')
        model.load_state_dict(saved['state_dict'],strict=True)
        require(torch.equal(before,logits_all(model,data['features'])),'Reload output mismatch')
        write_new(folder/'reload_check.json',{'exact_state':True,'exact_all_37_logits':True,'restricted_loader':True})
    rows=predictions(model,data,context,seed,fold,variant,cp)
    write_new(folder/'predictions.json',rows)
    write_new(folder/'complete.json',{name:descriptor(folder/name) for name in ['checkpoint.pt','checkpoint.json','train_history.json','predictions.json','reload_check.json']})
    print(f'Completed {variant} seed={seed} fold={fold} updates=240 seconds={time.perf_counter()-start:.1f}',flush=True)


def run():
    context,cfg=preflight()
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            for variant in VARIANTS:fit(context,cfg,seed,fold,variant)
    print(f'All 30 N/R fits complete; {protect_history()} historical files unchanged.',flush=True)


if __name__=='__main__':run()

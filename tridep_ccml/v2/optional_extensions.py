"""Independent G/F CCML extensions on the locked, label-free S input version."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import time
from . import scale as s
from .common import WORK, TC, RecordContext, read_json, write_new, descriptor, verify_descriptor, digest, require, set_seed, LOSS
from .rebuild_eeg import validate_row, loss_parts
from tridep_ccml.ccml import ccml_loss, evidence_dc, dirichlet_from_evidence, expected_cross_entropy
from tridep_ccml.train import state_digest
from tridep_ccml.metrics import binary_metrics
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OUT=TC/'experiments/CCML-v2-GF-37-fulltext'
CONFIG=OUT/'configuration.json'
PLAN=WORK/'output/plans/CCML_v2_G_F_独立扩展固定计划.md'
MODALITIES=s.MODALITIES
VALUES={'G':[.01,.1], 'F':[.1,1.]}
STAGES={k:f'CCML-v2-{k}-37-fulltext' for k in VALUES}


def quality_features(raw):
    detached=raw.detach()
    strength=detached.sum(-1)+2
    return torch.stack((2/strength,(detached[...,1]-detached[...,0]).abs()/strength),-1)


class ExtensionModel(s.ScaleModel):
    def __init__(self,seed,kind):
        super().__init__(seed,learned=False)
        require(kind in VALUES,'Invalid extension')
        self.kind=kind
        if kind=='G':
            self.gate_weights=nn.Parameter(torch.zeros(3,2))
            self.gate_bias=nn.Parameter(torch.zeros(3))

    def components(self,inputs):
        raw=self.raw_evidence(inputs)
        q=quality_features(raw)
        if self.kind=='G':
            logs=math.log(2)*torch.tanh((q*self.gate_weights[None]).sum(-1)+self.gate_bias[None])
        else:
            logs=torch.zeros_like(raw[...,0])
        return raw,logs,q

    def forward(self,inputs):
        raw,logs,_=self.components(inputs)
        return raw*logs.exp()[...,None]


def objective(e,logs,y,epoch,kind,value):
    ccml=ccml_loss(e,y,epoch,**LOSS)
    if kind=='G':
        extra=value*logs.square().mean()
    else:
        alpha=evidence_dc(e)['alpha']
        extra=value*expected_cross_entropy(alpha,F.one_hot(y,2).to(e.dtype)).mean()
    return ccml+extra,extra


def base_hash(model):
    return state_digest({k:v for k,v in model.state_dict().items() if not k.startswith('gate_')})


def protect_history():
    path=OUT/'history_before.json'
    if not path.exists():
        previous=TC/'experiments/CCML-v2-AS-RP-37-fulltext'
        require(read_json(previous/'status.json')['status']=='complete','AS-R/P must finish before history snapshot')
        frozen=read_json(previous/'history_before.json')
        for p in [*previous.rglob('*'),TC/'v2/as_optional.py',TC/'v2/test_as_optional.py',
                  WORK/'output/plans/CCML_v2_AS_R_P_补充固定对照计划.md',*list((TC/'experiments').glob('CCML-v2-AS-RP-*.log'))]:
            if p.is_file():frozen[p.relative_to(WORK).as_posix()]=digest(p)
        write_new(path,frozen)
    frozen=read_json(path)
    for name,sha in frozen.items():require(digest(WORK/name)==sha,'Historical artifact changed: '+name)
    return len(frozen)


def configuration(context):
    return {'family':OUT.name,'date':'2026-09-24','plan':descriptor(PLAN),'candidates':VALUES,
            'subject_ids':context.ids,'cohort_sha256':context.cohort_sha256,'folds_sha256':context.folds_sha256,
            'seeds':[11,22,33],'folds':[1,2,3,4,5], 'parent_configuration':descriptor(s.CONFIG),
            'input_manifest':descriptor(s.OUT/'input_manifest.json'),'nested_splits':descriptor(s.OUT/'nested_splits.json'),
            'control_status':descriptor(s.OUT/'status.json'),'control':'CCML-v2-S-control-37-fulltext',
            'selection':'mean inner-val NLL; tolerance 1e-12; G prefer larger lambda; F prefer smaller beta',
            'G':{'quality_features':['u','absolute_Dirichlet_class_margin'],'quality_gradient':'stop',
                 'gate':'three independent 2-to-1 affine maps, zero initialized','scale_bounds':[.5,2.],
                 'scale_formula':'exp(log(2)*tanh(a))','penalty':'lambda*mean(log(s)^2)','parameters':223375},
            'F':{'extra_loss':'beta*expected_cross_entropy(alpha_final,y), subject mean; no extra final KL',
                 'scales':'fixed identity','parameters':223366},
            'epochs':60,'batch_size':8,'optimizer':'Adamax','projection_lr':1e-4,'head_lr':1e-3,'gate_lr':1e-3,
            'weight_decay':0.,'gradient_clip':1.,'ccml':LOSS,'dropout':'none','normalization':'LayerNorm128_no_affine_eps1e-5',
            'selection_epoch':'fixed_last','threshold':.5,'tie_class':0,'device':'cpu','threads':1,
            'budget':{'inner_runs':180,'outer_runs':30,'reused_control_runs':15,'new_training_runs':210,
                      'inner_updates':32400,'outer_updates':7200,'new_updates':39600},
            'sources':[descriptor(TC/p) for p in ['v2/optional_extensions.py','v2/complete_optional_extensions.py',
                       'v2/test_optional_extensions.py','v2/scale.py','v2/complete_scale.py','v2/common.py',
                       'v2/rebuild_eeg.py','v2/complete_eeg.py','v2/complete_p.py','v2/complete_nr.py','ccml.py','train.py','metrics.py']]}


def preflight():
    OUT.mkdir(parents=True,exist_ok=True)
    count=protect_history()
    context,parent_cfg,data=s.load_setup()
    require(read_json(s.OUT/'status.json')['status']=='complete' and read_json(OUT/'test_results.json')['passed'],'Parent/tests incomplete')
    s.build_inputs(context)
    cfg=configuration(context);write_new(CONFIG,cfg)
    checks=[]
    for seed in cfg['seeds']:
        for fold in cfg['folds']:
            set_seed(seed)
            folder=s.run_folder(seed,fold,None,'control',0.)
            for d in read_json(folder/'complete.json').values():verify_descriptor(d)
            saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
            meta=saved['metadata']
            for kind in VALUES:
                model=ExtensionModel(seed,kind)
                require(base_hash(model)==meta['initial_state_sha256'],'Base initialization not paired')
                require(sum(p.numel() for p in model.parameters())==cfg[kind]['parameters'],'Wrong parameter scope')
                state=model.state_dict();state.update(saved['state_dict']);model.load_state_dict(state,strict=True)
                e=[]
                with torch.no_grad():
                    for i in range(0,37,8):e.append(model({m:data[m][i:i+8] for m in MODALITIES}))
                    prob=evidence_dc(torch.cat(e))['p'].tolist()
                expected=read_json(folder/'predictions.json')
                require(prob==[r['p_final'] for r in expected],'Unmodified control replay not exact')
            checks.append({'seed':seed,'fold':fold,'exact_control_records':37,'control':descriptor(folder/'checkpoint.pt')})
            print(f'G/F preflight seed={seed} fold={fold}: base control exact.',flush=True)
    write_new(OUT/'preflight.json',{'passed':True,'configuration':descriptor(CONFIG),'test_results':descriptor(OUT/'test_results.json'),
                                   'checks':checks,'exact_control_records':555,'protected_files_unchanged':count})


def load_setup():
    context,_,data=s.load_setup()
    cfg=read_json(CONFIG)
    require(cfg==configuration(context) and read_json(OUT/'preflight.json')['passed'],'Locked configuration/preflight changed')
    return context,cfg,data


def run_folder(kind,seed,fold,inner,value):
    return OUT/f'runs/{kind}/seed_{seed}/fold_{fold}'/(f'inner_{inner}/value_{value:g}' if inner is not None else 'outer')


def predictions(model,data,context,meta,cp,split_map):
    ids=[i for i in context.ids if i in split_map]; raw=[];logs=[];quality=[]
    with torch.no_grad():
        for i in range(0,len(ids),8):
            idx=[context.ids.index(x) for x in ids[i:i+8]]
            r,l,q=model.components({m:data[m][idx] for m in MODALITIES})
            raw.append(r);logs.append(l);quality.append(q)
        raw,logs,quality=map(torch.cat,(raw,logs,quality))
        scales=logs.exp();e=raw*scales[...,None]
        views,final,unit=dirichlet_from_evidence(e),evidence_dc(e),evidence_dc(raw)
    rows=[]
    for i,sid in enumerate(ids):
        row={'experiment':STAGES[meta['kind']],'seed':meta['seed'],'outer_fold':meta['fold'],'inner_fold':meta['inner_fold'],
             'subject_id':sid,'label':context.labels[sid],'split':split_map[sid],'train_ids':meta['train_ids'],
             'checkpoint':descriptor(cp),'configuration':descriptor(CONFIG),'feature_source':descriptor(s.OUT/'input_manifest.json'),
             'cohort_sha256':context.cohort_sha256,'folds_sha256':context.folds_sha256,'kind':meta['kind'],'value':meta['value'],
             'fusion_method':'CCML_dynamic_modality_scales' if meta['kind']=='G' else 'CCML_extra_final_supervision',
             'rho':1.25,'delta':1.,'gamma':1.,'scales':scales[i].tolist(),'quality_features':quality[i].tolist(),
             'views':{m:{'e_raw':raw[i,j].tolist(),'e':e[i,j].tolist(),'alpha':views['alpha'][i,j].tolist(),
                         'p':views['p'][i,j].tolist(),'u':float(views['u'][i,j])} for j,m in enumerate(MODALITIES)},
             **{k:final[k][i].tolist() for k in ['m','e_con','e_cmp','e_final']},
             'min_sources':[[m for j,m in enumerate(MODALITIES) if e[i,j,k]==final['m'][i,k]] for k in [0,1]],
             'p_final':final['p'][i].tolist(),'u_final':float(final['u'][i]),'prediction':int(final['p'][i,1]>.5),
             'same_trained_model_unit_scale_diagnostic':{'p_final':unit['p'][i].tolist(),'u_final':float(unit['u'][i]),'prediction':int(unit['p'][i,1]>.5)}}
        validate_row(row);rows.append(row)
    return rows


def train_model(model,inputs,y,seed,kind,value):
    groups=[{'params':model.projections.parameters(),'lr':1e-4},{'params':model.heads.parameters(),'lr':1e-3}]
    if kind=='G':groups.append({'params':[model.gate_weights,model.gate_bias],'lr':1e-3})
    opt=torch.optim.Adamax(groups,weight_decay=0.)
    generator,batch_hash,history=torch.Generator().manual_seed(seed),hashlib.sha256(),[]
    updates=0
    for epoch in range(1,61):
        order=torch.randperm(len(y),generator=generator);batch_hash.update(order.numpy().astype('<i8').tobytes())
        total,extra_sum,terms,grad,gate_grad=0.,0.,{},[],[]
        for i in range(0,len(y),8):
            b=order[i:i+8];opt.zero_grad(set_to_none=True)
            raw,logs,_=model.components({m:inputs[m][b] for m in MODALITIES})
            e=raw*logs.exp()[...,None]
            loss,extra=objective(e,logs,y[b],epoch,kind,value)
            loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),'Missing/nonfinite gradient')
            if kind=='G':gate_grad.append(float(torch.sqrt(model.gate_weights.grad.square().sum()+model.gate_bias.grad.square().sum())))
            grad.append(float(nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)))
            opt.step();updates+=1
            require(all(torch.isfinite(p).all() for p in model.parameters()),'Nonfinite parameter')
            total+=float(loss.detach())*len(b);extra_sum+=float(extra.detach())*len(b)
            with torch.no_grad():
                for k,v in loss_parts(e.detach(),y[b],epoch).items():terms[k]=terms.get(k,0.)+v*len(b)
        with torch.no_grad():
            _,endlogs,_=model.components(inputs);end_scales=endlogs.exp()
        history.append({'epoch':epoch,'train_objective':total/len(y),'extra_loss':extra_sum/len(y),
                        'ccml_loss_terms':{k:v/len(y) for k,v in terms.items()},'gradient_mean':float(np.mean(grad)),
                        'gate_gradient_mean':float(np.mean(gate_grad)) if gate_grad else 0.,
                        'fit_scale_min':end_scales.amin(0).tolist(),'fit_scale_max':end_scales.amax(0).tolist(),
                        'fit_scale_mean':end_scales.mean(0).tolist()})
    return history,updates,batch_hash.hexdigest()


def fit_run(context,data,kind,seed,fold,inner,value,train_ids,split_map,selection=None):
    start=time.perf_counter();set_seed(seed);model=ExtensionModel(seed,kind)
    folder=run_folder(kind,seed,fold,inner,value);folder.mkdir(parents=True,exist_ok=True);cp=folder/'checkpoint.pt'
    require(set(train_ids)=={sid for sid,p in split_map.items() if p=='train'} and set(train_ids)<=set(context.folds[f'fold_{fold}']['train']),'Illegal fit membership')
    meta={'experiment':STAGES[kind],'kind':kind,'seed':seed,'fold':fold,'inner_fold':inner,'value':value,
          'train_ids':train_ids,'split_map':split_map,'configuration':descriptor(CONFIG),'inputs':descriptor(s.OUT/'input_manifest.json'),
          'selection':selection,'initial_state_sha256':state_digest(model.state_dict()),'initial_base_sha256':base_hash(model),
          'cohort_sha256':context.cohort_sha256,'folds_sha256':context.folds_sha256,'supervised_parent':None,
          'epochs':60,'updates':60*math.ceil(len(train_ids)/8),'parameters':sum(p.numel() for p in model.parameters()),
          'objective':'full_CCML_plus_dynamic_log_scale_shrinkage' if kind=='G' else 'full_CCML_plus_expected_final_CE',
          'torch':str(torch.__version__),'device':'cpu'}
    if cp.exists():
        side=read_json(folder/'checkpoint.json');saved=torch.load(verify_descriptor(side['checkpoint']),map_location='cpu',weights_only=True)
        require(side['metadata']==saved['metadata'] and all(saved['metadata'][k]==v for k,v in meta.items()),'Resume provenance changed')
        model.load_state_dict(saved['state_dict'],strict=True)
    else:
        idx=[context.ids.index(x) for x in train_ids]
        y=torch.tensor([context.labels[x] for x in train_ids])
        history,updates,batch_hash=train_model(model,{m:data[m][idx].clone() for m in MODALITIES},y,seed,kind,value)
        require(updates==meta['updates'],'Budget changed');meta['batch_order_sha256']=batch_hash
        with cp.open('xb') as f:torch.save({'state_dict':model.state_dict(),'metadata':meta},f)
        write_new(folder/'checkpoint.json',{'checkpoint':descriptor(cp),'metadata':meta});write_new(folder/'train_history.json',history)
    rows=predictions(model,data,context,meta,cp,split_map)
    write_new(folder/'predictions.json',rows)
    write_new(folder/'complete.json',{n:descriptor(folder/n) for n in ['checkpoint.pt','checkpoint.json','train_history.json','predictions.json']})
    print(f'{kind} seed={seed} outer={fold} inner={inner} value={value:g} updates={meta["updates"]} seconds={time.perf_counter()-start:.1f}',flush=True)
    return rows


def select_candidate(kind,candidate_rows,outer_train_ids):
    require(set(candidate_rows)==set(VALUES[kind]),'Candidate values differ')
    scores={}
    for value,rows in candidate_rows.items():
        require(all(r['split']=='inner_val' and r['inner_fold'] in [1,2,3] and r['subject_id'] not in r['train_ids'] and
                    set(r['train_ids'])<=set(outer_train_ids) for r in rows),'Illegal inner selection')
        require(Counter(r['subject_id'] for r in rows)==Counter(outer_train_ids),'Inner OOF coverage differs')
        metrics={str(j):binary_metrics([r['label'] for r in rows if r['inner_fold']==j],[r['p_final'][1] for r in rows if r['inner_fold']==j]) for j in [1,2,3]}
        scores[value]={'fold_metrics':metrics,'mean_nll':sum(x['nll'] for x in metrics.values())/3}
    best=min(x['mean_nll'] for x in scores.values())
    ties=[k for k,v in scores.items() if v['mean_nll']<=best+1e-12]
    chosen=max(ties) if kind=='G' else min(ties)
    return {'kind':kind,'selected_value':chosen,'criterion':'inner fold mean final NLL; G stronger shrinkage; F weaker extra supervision on tie',
            'candidate_scores':{str(k):v for k,v in scores.items()},'selection_subject_ids':outer_train_ids}


def train_inner():
    context,cfg,data=load_setup();parts=read_json(s.OUT/'nested_splits.json')['folds'];locked=[]
    for kind in VALUES:
        for seed in cfg['seeds']:
            for fold in cfg['folds']:
                candidates={};sources=[]
                for value in VALUES[kind]:
                    candidates[value]=[]
                    for part in parts[str(fold)]:
                        split_map={**dict.fromkeys(part['train_ids'],'train'),**dict.fromkeys(part['val_ids'],'inner_val')}
                        rows=fit_run(context,data,kind,seed,fold,part['inner_fold'],value,part['train_ids'],split_map)
                        candidates[value]+=[r for r in rows if r['split']=='inner_val']
                        sources.append(descriptor(run_folder(kind,seed,fold,part['inner_fold'],value)/'predictions.json'))
                choice=select_candidate(kind,candidates,context.folds[f'fold_{fold}']['train'])
                choice.update(seed=seed,outer_fold=fold,configuration=descriptor(CONFIG),sources=sources,outer_val_used=False,outer_test_used=False)
                path=OUT/f'selections/{kind}/seed_{seed}/fold_{fold}.json';write_new(path,choice);locked.append(descriptor(path))
                print(f'{kind} selection locked seed={seed} outer={fold} value={choice["selected_value"]}',flush=True)
    write_new(OUT/'all_selections_locked.json',{'configuration':descriptor(CONFIG),'selections':locked,'inner_runs':180,'inner_updates':32400,'new_outer_predictions_computed':False})


def train_outer():
    context,cfg,data=load_setup();lock=read_json(OUT/'all_selections_locked.json')
    require(len(lock['selections'])==30 and lock['configuration']==descriptor(CONFIG),'All selections must be locked before outer models')
    for d in lock['selections']:verify_descriptor(d)
    for kind in VALUES:
        for seed in cfg['seeds']:
            for fold in cfg['folds']:
                path=OUT/f'selections/{kind}/seed_{seed}/fold_{fold}.json';value=read_json(path)['selected_value']
                outer=context.folds[f'fold_{fold}'];split_map={sid:p for p,ids in outer.items() for sid in ids}
                fit_run(context,data,kind,seed,fold,None,value,outer['train'],split_map,descriptor(path))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','train'])
    if parser.parse_args().action=='preflight':preflight()
    else:train_inner();train_outer()

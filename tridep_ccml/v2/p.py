"""Independent CCML projection adaptation from legal pre-projection caches."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

from .common import (WORK, TC, OUT as PARENT, CONTROL, LOSS, RecordContext, descriptor,
                     verify_descriptor, read_json, require, write_new, set_seed, digest)
from .nr import OUT as NR, load_data as load_frozen_data
from .resume_eeg import load_checkpoint as load_eeg_checkpoint
from .rebuild_eeg import loss_parts, validate_row
from tridep_ccml.train import make_initial_state, state_digest, load_features
from tridep_ccml.ccml import ThreeViewHeads, ccml_loss, evidence_dc, dirichlet_from_evidence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0,str(WORK/'S00/upstream-dev'))
from lib.models import models

OUT=TC/'experiments/CCML-v2-P-37-fulltext'
CONFIG=OUT/'configuration_v2.json'
EXPERIMENT='CCML-v2-P-37-fulltext'
MODALITIES=['eeg','speech','text']
DIMENSIONS={'eeg':1920,'speech':3584,'text':1024}
DROPOUT={'eeg':.4,'speech':.4,'text':.3}
PROJECTION_COUNT=835968


class ProjectionHeads(nn.Module):
    def __init__(self,projection_state,seed):
        super().__init__()
        self.projections=nn.ModuleDict({m:nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Dropout(DROPOUT[m]))
                                       for m,d in DIMENSIONS.items()})
        self.projections.load_state_dict(projection_state,strict=True)
        self.heads=ThreeViewHeads();self.heads.load_state_dict(make_initial_state(seed),strict=True)
        self.eval()  # Parameters retain gradients; stochastic Dropout stays disabled.

    def representations(self,features):
        return torch.stack([self.projections[m](features[m]) for m in MODALITIES],dim=1)

    def forward(self,features):return self.heads(self.representations(features))


def configuration(context):
    return {'experiment':EXPERIMENT,'version':2,'cohort_sha256':context.cohort_sha256,
            'folds_sha256':context.folds_sha256,'subject_ids':context.ids,'seeds':[11,22,33],'folds':[1,2,3,4,5],
            'parent':PARENT.name,'direct_control':CONTROL,'nr_reference':descriptor(NR/'completion_audit.json'),
            'parent_status':descriptor(PARENT/'status.json'),'parent_audit':descriptor(PARENT/'completion_audit.json'),
            'plan':descriptor(WORK/'output/plans/CCML_v2_性能优化与后续计划.md'),
            'projection_dimensions':DIMENSIONS,'projection_parameters':PROJECTION_COUNT,'head_parameters':774,
            'trainable_parameters':PROJECTION_COUNT+774,'projection_lr':1e-4,'head_lr':1e-3,
            'epochs':60,'batch_size':8,'optimizer':'Adamax','weight_decay':0.,'gradient_clip':1.,'ccml':LOSS,
            'threshold':.5,'tie_class':0,'selection':'fixed_last_epoch','device':'cpu','threads':1,
            'dropout':'eval','frozen_modules':'CNN/LSTM and all upstream encoders; cached pre-projection outputs',
            'extraction_cudnn_allow_tf32':{'eeg':False,'speech':True,'text':True},
            'extraction_matmul_allow_tf32':False,
            'preflight_revision_reason':'Reproduce historical speech/text frozen outputs with original cuDNN TF32 mode; no tolerance change or training change',
            'N_added':False,'R_added':False,'search':False,'formal_runs':15,'updates':3600,
            'preflight_representation_rtol':2e-5,'preflight_representation_atol':2e-5,
            'control_probability_atol':2e-6,'control_decisions_must_match':True,
            'historical_ancestry':'EEG projection from same-fold CCML-v2-E adaptation; speech/text projections from same-fold historical CE parents; all new optimization CCML only',
            'sources':{n:descriptor(p) for n,p in {
                'runner':Path(__file__),'common':TC/'v2/common.py','frozen_data_loader':TC/'v2/nr.py',
                'checkpoint_loader':TC/'v2/resume_eeg.py','loss_diagnostics':TC/'v2/rebuild_eeg.py',
                'ccml':TC/'ccml.py','initialization':TC/'train.py','records':TC/'records.py',
                'upstream_models':WORK/'S00/upstream-dev/lib/models/models.py'}.items()}}


def protect_history():
    path=OUT/'history_before.json'
    if not path.exists():
        tracked=read_json(NR/'history_before.json')
        for p in NR.rglob('*'):
            if p.is_file():tracked[p.relative_to(WORK).as_posix()]=digest(p)
        for name in ['nr.py','complete_nr.py','test_nr.py']:
            p=TC/'v2'/name;tracked[p.relative_to(WORK).as_posix()]=digest(p)
        log=TC/'experiments/CCML-v2-NR-training.log'
        if log.exists():tracked[log.relative_to(WORK).as_posix()]=digest(log)
        write_new(path,tracked)
    tracked=read_json(path)
    require(all(digest(WORK/p)==h for p,h in tracked.items()),'Historical artifact changed')
    return len(tracked)


def setup():
    context=RecordContext();OUT.mkdir(parents=True,exist_ok=True)
    protect_history();cfg=configuration(context);write_new(CONFIG,cfg)
    for root in [PARENT,NR]:
        status=read_json(root/'status.json');require(status['status']=='complete','Prior stage incomplete')
        verify_descriptor(status['audit']);verify_descriptor(status['report'])
    return context,cfg


def project_all(model,features):
    model.eval();outputs=[]
    with torch.no_grad():
        for i in range(0,37,8):outputs.append(model.representations({m:torch.from_numpy(features[m][i:i+8]) for m in MODALITIES}))
    return torch.cat(outputs)


def logits_all(model,features):
    model.eval();outputs=[]
    with torch.no_grad():
        for i in range(0,37,8):outputs.append(model({m:torch.from_numpy(features[m][i:i+8]) for m in MODALITIES}))
    return torch.cat(outputs)


def export_one(context,cfg,seed,fold):
    set_seed(seed);old=load_features(context,seed,fold);frozen=load_frozen_data(context,seed,fold)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    folder=OUT/f'features/seed_{seed}/fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'preprojection.npz';init_path=folder/'projection_initial.pt'
    manifest_path=folder/'manifest.json'
    if manifest_path.exists():
        data=read_json(manifest_path);verify_descriptor(data['cache']);verify_descriptor(data['projection_initial'])
        require(data['configuration']==descriptor(CONFIG),'Stale projection cache')
        return data
    features={};projection_state={};sources={};checks={}
    eeg_entries=read_json(PARENT/'eeg_manifest.json')
    eeg_paths={r['provenance']['subject_id']:r['feature'] for r in eeg_entries}
    for v,m in enumerate(MODALITIES):
        torch.backends.cudnn.allow_tf32=cfg['extraction_cudnn_allow_tf32'][m]
        if m=='eeg':
            parent_descriptor=frozen['source']['parent_checkpoint'];cp=verify_descriptor(parent_descriptor)
            saved=load_eeg_checkpoint(cp)
            model=models.EEGConvPoolClassifier(input_dim=200,seq_len=60)
            model.classifier[-1]=nn.Identity()
            model.load_state_dict({k.removeprefix('eeg.'):t for k,t in saved['state_dict'].items() if k.startswith('eeg.')},strict=True)
            require(saved['metadata']['seed']==seed and saved['metadata']['fold']==fold and saved['metadata']['train_ids']==context.folds[f'fold_{fold}']['train'],'EEG fit provenance mismatch')
            inputs=eeg_paths;linear_index=2;batch_size=8
        else:
            parent_descriptor=old['manifest']['parent_checkpoints'][m];cp=verify_descriptor(parent_descriptor)
            saved=torch.load(cp,map_location='cpu',weights_only=True)
            require(saved['seed']==seed and saved['fold']==fold and saved['train_ids']==context.folds[f'fold_{fold}']['train'],'Historical fit provenance mismatch')
            branch=saved['configuration'];model=getattr(models,branch['model_class'])(**branch['model_kwargs'])
            model.load_state_dict(saved['state_dict'],strict=True)
            inputs=old['manifest']['source_features'][m];linear_index=2 if m=='speech' else 0;batch_size=1
        model.requires_grad_(False).to(device).eval()
        linear=model.classifier[linear_index]
        require(isinstance(linear,nn.Linear) and linear.in_features==DIMENSIONS[m] and linear.out_features==128,'Wrong projection selected')
        projection_state[f'{m}.0.weight']=linear.weight.detach().cpu().clone()
        projection_state[f'{m}.0.bias']=linear.bias.detach().cpu().clone()
        captured=[];captured_post=[]
        handle=linear.register_forward_pre_hook(lambda _module,args:captured.append(args[0].detach().cpu()))
        post_handle=model.classifier[-1].register_forward_pre_hook(lambda _module,args:captured_post.append(args[0].detach().cpu()))
        arrays=np.stack([np.load(verify_descriptor(inputs[sid]),allow_pickle=False) for sid in context.ids])
        with torch.no_grad():
            for i in range(0,37,batch_size):model(torch.as_tensor(arrays[i:i+batch_size],device=device))
        handle.remove();post_handle.remove();pre=torch.cat(captured).numpy()
        require(pre.shape==(37,DIMENSIONS[m]) and np.isfinite(pre).all(),'Bad pre-projection tensor')
        features[m]=pre
        restored=F.relu(F.linear(torch.from_numpy(pre),projection_state[f'{m}.0.weight'],projection_state[f'{m}.0.bias'])).numpy()
        reference=frozen['features'][:,v]
        native_post=torch.cat(captured_post).numpy()
        require(np.allclose(native_post,reference,rtol=cfg['preflight_representation_rtol'],atol=cfg['preflight_representation_atol']),f'{m} full parent path failed reproduction')
        require(np.allclose(restored,reference,rtol=cfg['preflight_representation_rtol'],atol=cfg['preflight_representation_atol']),f'{m} projection failed parent reproduction')
        checks[m]={'input_shape':list(pre.shape),'projection_shape':[128,DIMENSIONS[m]],
                   'parameters':int(linear.weight.numel()+linear.bias.numel()),
                   'max_abs_representation_error':float(np.max(np.abs(restored-reference))),
                   'max_abs_native_parent_error':float(np.max(np.abs(native_post-reference))),
                   'cudnn_allow_tf32':cfg['extraction_cudnn_allow_tf32'][m],
                   'frozen_upstream_state_sha256':state_digest(model.state_dict()),'extraction_batch_size':batch_size}
        sources[m]={'checkpoint':parent_descriptor,'raw_features':inputs}
        del model,saved,arrays;torch.cuda.empty_cache()
    require(sum(r['parameters'] for r in checks.values())==PROJECTION_COUNT,'Wrong projection parameter count')
    model=ProjectionHeads(projection_state,seed)
    require(sum(p.numel() for p in model.parameters())==PROJECTION_COUNT+774,'Wrong trainable count')
    control_folder=PARENT/'runs'/CONTROL/f'seed_{seed}'/f'fold_{fold}'
    control=load_eeg_checkpoint(control_folder/'checkpoint.pt')
    model.heads.load_state_dict(control['state_dict'],strict=True)
    fused=evidence_dc(F.softplus(logits_all(model,features)))
    rows=read_json(control_folder/'predictions.json')
    require([r['subject_id'] for r in rows]==context.ids,'Control identity mismatch')
    reference=np.array([r['p_final'] for r in rows]);error=float(np.max(np.abs(fused['p'].numpy()-reference)))
    require(error<=cfg['control_probability_atol'],'Projection path changes control probabilities excessively')
    require((fused['p'][:,1]>.5).int().tolist()==[r['prediction'] for r in rows],'Projection path changes control decisions')
    arrays={**features,'subject_ids':np.array(context.ids),'labels':frozen['labels'],'splits':frozen['splits']}
    if path.exists():
        with np.load(path,allow_pickle=False) as previous:require(all(np.array_equal(previous[k],v) for k,v in arrays.items()),'Existing cache differs')
    else:
        with path.open('xb') as stream:np.savez_compressed(stream,**arrays)
    if init_path.exists():require(state_digest(torch.load(init_path,weights_only=True))==state_digest(projection_state),'Existing projections differ')
    else:
        with init_path.open('xb') as stream:torch.save(projection_state,stream)
    manifest={'seed':seed,'fold':fold,'cohort_sha256':context.cohort_sha256,'folds_sha256':context.folds_sha256,
              'fit_subject_ids':context.folds[f'fold_{fold}']['train'],'cache':descriptor(path),'projection_initial':descriptor(init_path),
              'configuration':descriptor(CONFIG),'source_frozen_bundle':frozen['source'],
              'sources':sources,'checks':checks,'device':device,'torch':str(torch.__version__),
              'control_checkpoint':descriptor(control_folder/'checkpoint.pt'),'control_probability_max_abs_error':error,
              'control_all_decisions_identical':True,'new_parameters_fitted_during_export':False,
              'supervised_upstream_requires_same_outer_fold':True}
    write_new(manifest_path,manifest)
    print(f'Projection cache seed={seed} fold={fold} control_max_error={error:.3g}',flush=True)
    return manifest


def export():
    context,cfg=setup();manifests=[]
    for seed in cfg['seeds']:
        for fold in cfg['folds']:manifests.append(export_one(context,cfg,seed,fold))
    write_new(OUT/'preflight.json',{'passed':True,'cache_bundles':15,'subject_fold_records':555,
                                   'projection_parameters':PROJECTION_COUNT,'total_trainable_parameters':PROJECTION_COUNT+774,
                                   'max_control_probability_error':max(r['control_probability_max_abs_error'] for r in manifests),
                                   'control_decisions_identical':True,'manifests':[descriptor(OUT/f'features/seed_{r["seed"]}/fold_{r["fold"]}/manifest.json') for r in manifests]})


def load_data(context,seed,fold):
    path=OUT/f'features/seed_{seed}/fold_{fold}/manifest.json';info=read_json(path)
    verify_descriptor(info['configuration']);cache=verify_descriptor(info['cache']);initial=verify_descriptor(info['projection_initial'])
    require(info['seed']==seed and info['fold']==fold and info['fit_subject_ids']==context.folds[f'fold_{fold}']['train'],'Projection fitting provenance mismatch')
    require(info['cohort_sha256']==context.cohort_sha256 and info['folds_sha256']==context.folds_sha256,'Projection cohort mismatch')
    for source in info['sources'].values():
        verify_descriptor(source['checkpoint'])
        for item in source['raw_features'].values():verify_descriptor(item)
    with np.load(cache,allow_pickle=False) as archive:data={k:archive[k].copy() for k in archive.files}
    expected=[next(p for p,ids in context.folds[f'fold_{fold}'].items() if sid in ids) for sid in context.ids]
    require(data['subject_ids'].tolist()==context.ids and data['splits'].tolist()==expected and data['labels'].tolist()==[context.labels[s] for s in context.ids],'Cache labels/splits mismatch')
    for m in MODALITIES:require(data[m].shape==(37,DIMENSIONS[m]) and data[m].dtype==np.float32 and np.isfinite(data[m]).all(),'Invalid cache array')
    data['initial']=torch.load(initial,map_location='cpu',weights_only=True)
    data['source']={'manifest':descriptor(path),'cache':info['cache'],'initial_projections':info['projection_initial']}
    return data


def predictions(model,data,context,seed,fold,checkpoint):
    e=F.softplus(logits_all(model,data));view=dirichlet_from_evidence(e);f=evidence_dc(e);rows=[]
    for i,sid in enumerate(context.ids):
        row={'experiment':EXPERIMENT,'seed':seed,'outer_fold':fold,'subject_id':sid,'label':int(data['labels'][i]),
             'split':str(data['splits'][i]),'train_ids':context.folds[f'fold_{fold}']['train'],
             'cohort_sha256':context.cohort_sha256,'folds_sha256':context.folds_sha256,
             'checkpoint':descriptor(checkpoint),'feature_source':data['source'],'configuration':descriptor(CONFIG),
             'fusion_method':'ccml','rho':1.25,'delta':1.,'gamma':1.,
             'views':{m:{'e':e[i,v].tolist(),'alpha':view['alpha'][i,v].tolist(),'p':view['p'][i,v].tolist(),'u':float(view['u'][i,v])} for v,m in enumerate(MODALITIES)},
             **{k:f[k][i].tolist() for k in ['m','e_con','e_cmp','e_final']},
             'p_final':f['p'][i].tolist(),'u_final':float(f['u'][i]),'prediction':int(f['p'][i,1]>.5)}
        validate_row(row);rows.append(row)
    return rows


def fit(context,cfg,seed,fold):
    start=time.perf_counter();set_seed(seed);data=load_data(context,seed,fold)
    model=ProjectionHeads(data['initial'],seed);initial_sha=state_digest(model.state_dict())
    folder=OUT/f'runs/seed_{seed}/fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
    cp=folder/'checkpoint.pt';train_ids=context.folds[f'fold_{fold}']['train']
    parent_side=read_json(PARENT/'runs'/CONTROL/f'seed_{seed}'/f'fold_{fold}/checkpoint.json')
    meta={'experiment':EXPERIMENT,'seed':seed,'fold':fold,'train_ids':train_ids,'source':data['source'],
          'cohort_sha256':context.cohort_sha256,'folds_sha256':context.folds_sha256,'configuration':descriptor(CONFIG),
          'control_checkpoint':parent_side['checkpoint'],'initial_state_sha256':initial_sha,
          'initial_projection_sha256':state_digest(data['initial']),'head_initialization_sha256':state_digest(make_initial_state(seed)),
          'projection_parameters':PROJECTION_COUNT,'trainable_parameters':PROJECTION_COUNT+774,
          'projection_lr':1e-4,'head_lr':1e-3,'weight_decay':0.,'objective':'full_CCML_only',
          'epochs':60,'updates':240,'selection':'last_epoch','device':'cpu','torch':str(torch.__version__),'dropout':'eval'}
    if cp.exists():
        side=read_json(cp.with_suffix('.json'));verify_descriptor(side['checkpoint'])
        saved=torch.load(cp,map_location='cpu',weights_only=True)
        require(saved['metadata']==side['metadata'] and all(saved['metadata'].get(k)==v for k,v in meta.items()),'Existing P checkpoint provenance mismatch')
        model.load_state_dict(saved['state_dict'],strict=True);meta=saved['metadata']
    else:
        lookup={s:i for i,s in enumerate(context.ids)};selected=[lookup[s] for s in train_ids]
        require(not(set(train_ids)&set(context.folds[f'fold_{fold}']['val']+context.folds[f'fold_{fold}']['test'])),'Training leakage')
        x={m:torch.from_numpy(data[m][selected]) for m in MODALITIES};y=torch.from_numpy(data['labels'][selected])
        optimizer=torch.optim.Adamax([{'params':model.projections.parameters(),'lr':1e-4},{'params':model.heads.parameters(),'lr':1e-3}],weight_decay=0.)
        generator=torch.Generator().manual_seed(seed);batch_hash=hashlib.sha256();history=[]
        for epoch in range(1,61):
            order=torch.randperm(len(selected),generator=generator);batch_hash.update(order.numpy().astype('<i8').tobytes())
            total=0.;terms={};norms=[];project_grad={m:[] for m in MODALITIES};head_grad={m:[] for m in MODALITIES}
            for i in range(0,len(order),8):
                b=order[i:i+8];optimizer.zero_grad(set_to_none=True)
                e=F.softplus(model({m:x[m][b] for m in MODALITIES}));loss=ccml_loss(e,y[b],epoch,**LOSS);loss.backward()
                for modules,record in [(model.projections,project_grad),(model.heads.heads,head_grad)]:
                    for m,module in modules.items():
                        require(all(p.grad is not None for p in module.parameters()),'Missing trainable gradient')
                        record[m].append(float(torch.sqrt(sum((p.grad**2).sum() for p in module.parameters()))))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                require(all(torch.isfinite(p).all() for p in model.parameters()),'Nonfinite parameters')
                total+=float(loss.detach())*len(b);norms.append(float(norm))
                with torch.no_grad():parts=loss_parts(e.detach(),y[b],epoch)
                for k,v in parts.items():terms[k]=terms.get(k,0.)+v*len(b)
            history.append({'epoch':epoch,'train_loss':total/len(selected),'loss_terms':{k:v/len(selected) for k,v in terms.items()},
                            'preclip_grad_norm_mean':float(np.mean(norms)),
                            'projection_grad_norm_mean':{m:float(np.mean(v)) for m,v in project_grad.items()},
                            'head_grad_norm_mean':{m:float(np.mean(v)) for m,v in head_grad.items()}})
        meta['batch_order_sha256']=batch_hash.hexdigest()
        require(meta['batch_order_sha256']==parent_side['metadata']['batch_order_sha256'] and meta['head_initialization_sha256']==parent_side['metadata']['head_initialization_sha256'],'Control pairing mismatch')
        require(state_digest(model.state_dict())!=initial_sha,'No parameter updates')
        with cp.open('xb') as stream:torch.save({'state_dict':model.state_dict(),'metadata':meta},stream)
        write_new(cp.with_suffix('.json'),{'checkpoint':descriptor(cp),'metadata':meta});write_new(folder/'train_history.json',history)
        before=logits_all(model,data);saved=torch.load(cp,map_location='cpu',weights_only=True)
        require(state_digest(saved['state_dict'])==state_digest(model.state_dict()),'Checkpoint state changed')
        model.load_state_dict(saved['state_dict'],strict=True)
        require(torch.equal(before,logits_all(model,data)),'Checkpoint predictions changed')
        write_new(folder/'reload_check.json',{'exact_state':True,'exact_all_37_logits':True,'restricted_loader':True})
    rows=predictions(model,data,context,seed,fold,cp);write_new(folder/'predictions.json',rows)
    write_new(folder/'complete.json',{name:descriptor(folder/name) for name in ['checkpoint.pt','checkpoint.json','train_history.json','predictions.json','reload_check.json']})
    print(f'Completed P seed={seed} fold={fold} updates=240 seconds={time.perf_counter()-start:.1f}',flush=True)


def train():
    context,cfg=setup();require(read_json(OUT/'preflight.json')['passed'],'Projection preflight incomplete')
    for seed in cfg['seeds']:
        for fold in cfg['folds']:fit(context,cfg,seed,fold)
    print(f'P complete: 15 fits, {protect_history()} historical artifacts unchanged.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['export','train','all'])
    action=parser.parse_args().action
    if action in ['export','all']:export()
    if action in ['train','all']:train()

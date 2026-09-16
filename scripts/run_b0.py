"""Train the three official heads on audited features; default is preflight.

No raw data are silently replaced. Feature rows must carry extraction provenance.
See README.md for the feature-manifest contract and unresolved upstream differences.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from prepare import ROOT, WORK, digest, save, audit_folds
from metrics import metrics, select_weights
from cohort import validate_active_cohort, validate_prediction_records

def check_inputs(feature_dir, seed_value, folds, cfg):
    import numpy as np
    lock=validate_active_cohort(ROOT,require_data=True)
    if cfg.get('cohort_id')!=lock['cohort_id'] or cfg.get('folds_sha256')!=lock['folds_sha256']:
        raise ValueError('Training configuration belongs to another cohort')
    data_report=json.loads((ROOT/'reports/modma_audit.json').read_text())
    if not data_report['passed']:
        raise ValueError('MODMA audit failed. Resolve invalid/missing raw data before B0 training.')
    protocol=feature_dir/'protocol_resolution.json'
    if not protocol.exists():
        raise ValueError('Missing protocol_resolution.json; paper/code discrepancies must be documented before training.')
    decisions=json.loads(protocol.read_text(encoding='utf-8'))
    if cfg.get('protocol_file'):
        if decisions!=json.loads((ROOT/cfg['protocol_file']).read_text(encoding='utf-8')):
            raise ValueError('Feature protocol differs from active configuration')
        if decisions.get('cohort_id')!=lock['cohort_id'] or decisions.get('folds_sha256')!=lock['folds_sha256']:
            raise ValueError('Protocol cohort mismatch')
    for key in ('speech_encoder','hubert_identity','asr_segments','audio_segmentation','eeg_units_channels','eeg_length_policy'):
        if not isinstance(decisions.get(key),str) or not decisions[key].strip():
            raise ValueError('Unresolved protocol: '+key)
    rows=json.loads((feature_dir/'features_manifest.json').read_text(encoding='utf-8'))
    indexed={}
    for row in rows:
        if (row.get('cohort_id')!=lock['cohort_id'] or row.get('folds_sha256')!=lock['folds_sha256']
                or row['subject_id'] not in lock['subject_ids']):
            raise ValueError('Feature cohort mismatch or excluded subject')
        if row['seed'] not in (None,seed_value): continue
        if cfg.get('protocol_file'):
            if row.get('protocol_sha256')!=digest(protocol): raise ValueError('Feature protocol provenance changed')
            if row.get('producer_sha256')!=digest(ROOT/'scripts/extract_b0.py'): raise ValueError('Feature producer changed')
            if row.get('backbones_sha256')!=digest(ROOT/'manifests/backbones.json'): raise ValueError('Feature backbones changed')
            if row['modality']=='speech' and row['seed']!=seed_value:
                raise ValueError('Random speech features require matching explicit seed')
        key=(row['outer_fold'],row['subject_id'],row['modality'])
        if key in indexed: raise ValueError('Duplicate feature record: '+str(key))
        path=(feature_dir/row['path']).resolve()
        if not path.is_relative_to(feature_dir.resolve()): raise ValueError('Feature outside feature directory')
        if digest(path)!=row['sha256']: raise ValueError('Feature SHA256 mismatch: '+str(path))
        x=np.load(path,allow_pickle=False)
        expected=cfg['branches'][row['modality']]['shape']
        if list(x.shape)!=expected or not np.isfinite(x).all(): raise ValueError('Feature shape/nonfinite error: '+str(path))
        if not row.get('producer_id') or not isinstance(row.get('fit_subject_ids'),list):
            raise ValueError('Missing producer/fit provenance')
        indexed[key]=(row,path)
    result={}
    for name,f in folds.items():
        fold=int(name.split('_')[1]); train=set(f['train'])
        for sid in sum([f[p] for p in ('train','val','test')],[]):
            for modality in cfg['branches']:
                item=indexed.get((fold,sid,modality),indexed.get((None,sid,modality)))
                if item is None: raise ValueError(f'Missing feature: {fold}/{sid}/{modality}')
                row,path=item
                if not set(row['fit_subject_ids']) <= train:
                    raise ValueError(f'Feature leakage into validation/test: {fold}/{sid}/{modality}')
                if row['outer_fold'] is None and row['fit_subject_ids']:
                    raise ValueError('Fitted features must have an explicit fold')
                result[(fold,sid,modality)]=path
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--feature-dir',type=Path,default=ROOT/'features/B0-37')
    parser.add_argument('--seed',type=int,choices=[11,22,33,44,55,42],default=11)
    parser.add_argument('--train',action='store_true',help='Train only after all preflight checks pass')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/b0.json').read_text())
    lock=validate_active_cohort(require_data=True)
    folds=json.loads((ROOT/'manifests/fold_assignments.json').read_text())
    if not audit_folds(folds)['passed']: raise ValueError('Invalid folds')
    paths=check_inputs(args.feature_dir,args.seed,folds,cfg)
    if not args.train:
        print('Preflight passed; no training requested.'); return
    import numpy as np
    import torch
    sys.path.insert(0,str(ROOT/'upstream-dev'))
    from lib.models import models
    from smoke_models import seed
    subjects=json.loads((ROOT/'manifests/subjects.json').read_text())
    labels={r['subject_id']:r['label'] for r in subjects}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    out_dir=ROOT/'runs'/'B0-37'/f'seed_{args.seed}'
    if out_dir.exists(): raise ValueError('Run already exists; refusing to overwrite: '+str(out_dir))
    out_dir.mkdir(parents=True)
    config_hash=digest(ROOT/'configs/b0.json')
    source_hash=digest(ROOT/'manifests/source_sha256.json')
    run={'seed':args.seed,'device':device,'config_sha256':config_hash,'feature_manifest_sha256':digest(args.feature_dir/'features_manifest.json'),'protocol_resolution_sha256':digest(args.feature_dir/'protocol_resolution.json'),'folds':{}}
    run.update(experiment=cfg['experiment'],cohort_id=lock['cohort_id'],folds_sha256=lock['folds_sha256'])
    run.update(training_script_sha256=digest(Path(__file__)),python=sys.version,torch_version=torch.__version__,
               cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
               checkpoint_selection='fixed_epochs_final_state')
    save(out_dir/'configuration.json',cfg)
    save(out_dir/'run.json',run)
    print(f'START B0-37 seed={args.seed} device={device}',flush=True)
    all_rows=[]
    for name,f in folds.items():
        fold=int(name.split('_')[1]); pred_rows={}
        for part in ('train','val','test'):
            for sid in f[part]: pred_rows[sid]={'experiment':cfg['experiment'],'cohort_id':lock['cohort_id'],'folds_sha256':lock['folds_sha256'],'subject_id':sid,'seed':args.seed,'outer_fold':fold,'inner_fold':None,'split':part,'label':labels[sid],'source_manifest_sha256':source_hash}
        branch_stats={}
        for modality,c in cfg['branches'].items():
            seed(args.seed)  # same initial state per fold; independent of prior folds
            model=getattr(models,c['model_class'])(**c['model_kwargs']).to(device)
            opt=torch.optim.Adamax(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
            x=torch.tensor(np.stack([np.load(paths[(fold,sid,modality)],allow_pickle=False) for sid in f['train']]),dtype=torch.float32)
            y=torch.tensor([labels[sid] for sid in f['train']],dtype=torch.long)
            loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=c['batch_size'],shuffle=True,num_workers=0)
            start=time.perf_counter()
            curve=[]
            for epoch in range(c['epochs']):
                model.train()
                loss_total=0.0;sample_count=0
                for xb,yb in loader:
                    opt.zero_grad(); loss=torch.nn.functional.cross_entropy(model(xb.to(device)),yb.to(device))
                    if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                    loss.backward(); opt.step()
                    loss_total+=float(loss.detach().cpu())*len(yb);sample_count+=len(yb)
                curve.append({'epoch':epoch+1,'train_cross_entropy':loss_total/sample_count})
                if epoch==0 or (epoch+1)%10==0 or epoch+1==c['epochs']:
                    print(f'TRAIN seed={args.seed} {name} {modality} epoch={epoch+1}/{c["epochs"]} loss={curve[-1]["train_cross_entropy"]:.6f}',flush=True)
                    save(out_dir/'training_progress.json',{'status':'running','seed':args.seed,'fold':fold,
                        'modality':modality,'epoch':epoch+1,'epochs':c['epochs'],'train_cross_entropy':curve[-1]['train_cross_entropy']})
            if device=='cuda': torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            model.eval()
            torch.save({'state_dict':model.state_dict(),'experiment':cfg['experiment'],'cohort_id':lock['cohort_id'],'folds_sha256':lock['folds_sha256'],'seed':args.seed,'fold':fold,'train_ids':f['train'],'configuration':c},out_dir/f'{name}_{modality}.pt')
            with torch.no_grad():
                for part in ('train','val','test'):
                    for sid in f[part]:
                        xx=torch.tensor(np.load(paths[(fold,sid,modality)],allow_pickle=False),dtype=torch.float32,device=device).unsqueeze(0)
                        pred_rows[sid]['p_'+modality]=float(torch.softmax(model(xx),dim=-1)[0,1].cpu())
            branch_stats[modality]={'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'training_seconds':elapsed,'epochs':c['epochs'],
                'training_curve':curve,'checkpoint_sha256':digest(out_dir/f'{name}_{modality}.pt')}
            del model,opt
        fusion=select_weights([pred_rows[s] for s in f['val']])
        for row in pred_rows.values():
            row['fusion_weights']=fusion['weights']
            row['p_final']=sum(a*row['p_'+m] for a,m in zip(fusion['weights'],('eeg','speech','text')))
            row['prediction']=int(row['p_final']>0.5)
        split_metrics={}
        for part in ('val','test'):
            split_metrics[part]={m:metrics([labels[s] for s in f[part]],[pred_rows[s]['p_'+m] for s in f[part]]) for m in ('eeg','speech','text','final')}
        run['folds'][name]={'branches':branch_stats,'fusion':fusion,'metrics':split_metrics}
        print(f'B0-37 seed={args.seed} {name} complete',flush=True)
        all_rows.extend(pred_rows.values()); save(out_dir/'run.json',run)
        (out_dir/'probabilities.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows),encoding='utf-8')
    run['test_summary']={m:{metric:{'mean':float(np.mean([v['metrics']['test'][m][metric] for v in run['folds'].values()])), 'std_ddof0':float(np.std([v['metrics']['test'][m][metric] for v in run['folds'].values()]))} for metric in ('macro_f1','mdd_recall','balanced_accuracy')} for m in ('eeg','speech','text','final')}
    run['cohort_validation']=validate_prediction_records(all_rows,cfg['experiment'])
    save(out_dir/'run.json',run)
    save(out_dir/'training_progress.json',{'status':'complete','seed':args.seed,'folds_completed':5,'branches_completed':15})
    print(f'COMPLETE B0-37 seed={args.seed}; 15 branch checkpoints and complete test coverage saved',flush=True)

if __name__=='__main__':
    try: main()
    except (ValueError,FileNotFoundError) as exc:
        print('B0 preflight/run stopped: '+str(exc),file=sys.stderr); raise SystemExit(2)

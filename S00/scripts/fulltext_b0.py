"""Isolated full-text control: reuse audited ASR and unchanged EEG/speech outputs."""
import argparse
import copy
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import sys
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from prepare import ROOT,save,digest
from cohort import validate_active_cohort,validate_prediction_records
from run_b0 import check_inputs
from metrics import metrics,select_weights
from smoke_models import seed
sys.path.insert(0,str(ROOT/'upstream-dev'))
from lib.models import models

EXPERIMENT='B0-37-fulltext'
FEATURES=ROOT/'features'/EXPERIMENT
TRANSCRIPTS=ROOT/'transcripts'/EXPERIMENT

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def immutable_json(p,value):
    if p.exists():
        if read(p)!=value:raise ValueError('Existing artifact differs: '+str(p))
        return
    tmp=p.with_suffix('.json.part');save(tmp,value);tmp.replace(p)

def merge_segments(segments):
    if not segments:raise ValueError('No ASR segments')
    for s in segments:
        if not all(math.isfinite(s[k]) for k in ('start','end')) or not 0<=s['start']<s['end'] or not isinstance(s['text'],str):
            raise ValueError('Invalid ASR segment')
    ordered=sorted(segments,key=lambda s:(s['start'],s['end']))
    text=''.join(s['text'] for s in ordered)
    if not text.strip() or '[ERROR]' in text:raise ValueError('Empty/error transcript')
    return text,ordered

def source_checks():
    lock=validate_active_cohort(require_data=True)
    basecfg=read(ROOT/'configs/b0.json');cfg=read(ROOT/'configs/b0_fulltext.json')
    folds=read(ROOT/'manifests/fold_assignments.json')
    if cfg['branches']!=basecfg['branches'] or cfg['seeds']!=basecfg['seeds'] or cfg['folds_sha256']!=lock['folds_sha256']:
        raise ValueError('Fulltext comparison must retain parent settings and folds')
    for value in cfg['seeds']:check_inputs(ROOT/'features/B0-37',value,folds,basecfg)
    backbone=read(ROOT/'manifests/backbones.json')
    for name,item in backbone['macbert']['files'].items():
        if digest(ROOT/'models/macbert'/name)!=item['sha256']:raise ValueError('MacBERT asset changed')
    return lock,cfg,folds

def extract():
    from transformers import BertModel,BertTokenizer
    lock,cfg,_=source_checks();transcript_hashes=read(ROOT/'manifests/transcripts_sha256.json')
    protocol=read(ROOT/'configs/protocol_fulltext.json')
    immutable_json(FEATURES/'protocol_resolution.json',protocol)
    tokenizer=BertTokenizer.from_pretrained(ROOT/'models/macbert',local_files_only=True)
    device='cuda' if torch.cuda.is_available() else 'cpu';seed(11);torch.set_num_threads(4)
    model=BertModel.from_pretrained(ROOT/'models/macbert',local_files_only=True).to(device).eval()
    entries=[];changed=[];lengths=[]
    for sid in lock['subject_ids']:
        texts=[];source_hashes=[];output_hashes=[];counts=[]
        for q in range(1,30):
            key=f'{sid}/{q:02d}';source=ROOT/'transcripts/B0-37/first_segment_code'/sid/f'{q:02d}.json'
            source_hash=digest(source)
            if source_hash!=transcript_hashes[key]:raise ValueError('Audited ASR changed: '+key)
            record=read(source)
            if record['source_sha256']!=digest(ROOT/'staged'/sid/'audio'/f'{q:02d}.wav'):raise ValueError('Audio changed')
            text,segments=merge_segments(record['segments'])
            count=len(tokenizer.encode(text,add_special_tokens=True,truncation=False))
            if count>512:raise ValueError(f'Full text exceeds512 tokens: {key}; no silent truncation permitted')
            if text!=record['answer_text']:changed.append(key)
            record.update(experiment=EXPERIMENT,export_policy='all_segments_chronological_fulltext',
                segments=segments,answer_text=text,all_segments_text=text,empty=False,parent_transcript_sha256=source_hash)
            target=TRANSCRIPTS/sid/f'{q:02d}.json';immutable_json(target,record)
            texts.append(text);counts.append(count);source_hashes.append(source_hash);output_hashes.append(digest(target))
        inputs=tokenizer(texts,padding=True,truncation=False,return_tensors='pt').to(device)
        with torch.no_grad():x=model(**inputs).last_hidden_state[:,0,:].cpu().numpy()
        if x.shape!=(29,768) or not np.isfinite(x).all():raise ValueError('Bad fulltext feature')
        path=FEATURES/'text'/f'{sid}.npy';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if not np.array_equal(x,np.load(path,allow_pickle=False)):raise ValueError('Existing fulltext feature changed')
        else:
            tmp=path.with_suffix('.npy.part')
            with tmp.open('wb') as f:np.save(f,x,allow_pickle=False)
            tmp.replace(path)
        entries.append({'experiment':EXPERIMENT,'cohort_id':lock['cohort_id'],'folds_sha256':lock['folds_sha256'],
            'subject_id':sid,'modality':'text','outer_fold':None,'seed':None,'fit_subject_ids':[],
            'path':str(path.relative_to(FEATURES)),'sha256':digest(path),'shape':[29,768],
            'producer_id':'fulltext_b0.py:all_segments_CLS','producer_sha256':digest(Path(__file__)),
            'protocol_sha256':digest(FEATURES/'protocol_resolution.json'),'backbones_sha256':digest(ROOT/'manifests/backbones.json'),
            'parent_transcript_sha256':source_hashes,'transcript_sha256':output_hashes,'token_counts':counts})
        lengths+=counts
        print(f'FULLTEXT {sid} [29,768] max_tokens={max(counts)}',flush=True)
    immutable_json(FEATURES/'features_manifest.json',entries)
    save(ROOT/'reports/fulltext_features_audit.json',{'passed':True,'experiment':EXPERIMENT,'subjects':37,'recordings':1073,
        'changed_recordings':changed,'changed_count':len(changed),'max_tokens':max(lengths),'truncated_recordings':[],
        'feature_manifest_sha256':digest(FEATURES/'features_manifest.json'),'asr_rerun':False})

def feature_paths(lock):
    rows=read(FEATURES/'features_manifest.json')
    if len(rows)!=37 or {r['subject_id'] for r in rows}!=set(lock['subject_ids']):raise ValueError('Wrong fulltext cohort')
    if read(FEATURES/'protocol_resolution.json')!=read(ROOT/'configs/protocol_fulltext.json'):raise ValueError('Fulltext protocol changed')
    paths={}
    for r in rows:
        if r['cohort_id']!=lock['cohort_id'] or r['folds_sha256']!=lock['folds_sha256'] or r['fit_subject_ids'] or r['seed'] is not None:
            raise ValueError('Fulltext feature provenance mismatch')
        if r['producer_sha256']!=digest(Path(__file__)) or r['protocol_sha256']!=digest(FEATURES/'protocol_resolution.json') or r['backbones_sha256']!=digest(ROOT/'manifests/backbones.json'):
            raise ValueError('Fulltext producer/protocol/model changed')
        p=(FEATURES/r['path']).resolve()
        if not p.is_relative_to(FEATURES.resolve()) or digest(p)!=r['sha256']:raise ValueError('Fulltext feature changed')
        x=np.load(p,allow_pickle=False)
        if x.shape!=(29,768) or not np.isfinite(x).all():raise ValueError('Bad fulltext feature')
        for q,h in enumerate(r['transcript_sha256'],1):
            if digest(TRANSCRIPTS/r['subject_id']/f'{q:02d}.json')!=h:raise ValueError('Fulltext transcript changed')
        paths[r['subject_id']]=p
    return paths

def train():
    lock,cfg,folds=source_checks();paths=feature_paths(lock)
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    for value in cfg['seeds']:
        out=ROOT/'runs'/EXPERIMENT/f'seed_{value}'
        if out.exists():raise ValueError('Refusing to overwrite fulltext run '+str(out))
        parent=ROOT/'runs/B0-37'/f'seed_{value}';parent_run=read(parent/'run.json')
        parent_rows=[json.loads(s) for s in (parent/'probabilities.jsonl').read_text().splitlines() if s]
        validate_prediction_records(parent_rows,'B0-37')
        if len(parent_rows)!=185:raise ValueError('Incomplete parent predictions')
        out.mkdir(parents=True);save(out/'configuration.json',cfg)
        run={'experiment':EXPERIMENT,'seed':value,'device':device,'cohort_id':lock['cohort_id'],'folds_sha256':lock['folds_sha256'],
            'config_sha256':digest(ROOT/'configs/b0_fulltext.json'),'feature_manifest_sha256':digest(FEATURES/'features_manifest.json'),
            'protocol_sha256':digest(FEATURES/'protocol_resolution.json'),'training_script_sha256':digest(Path(__file__)),
            'parent_run_sha256':digest(parent/'run.json'),'parent_predictions_sha256':digest(parent/'probabilities.jsonl'),'folds':{}}
        all_rows=[]
        for name,f in folds.items():
            fold=int(name[-1]);rows={r['subject_id']:copy.deepcopy(r) for r in parent_rows if r['outer_fold']==fold}
            for modality in ('eeg','speech'):
                p=parent/f'{name}_{modality}.pt'
                if digest(p)!=parent_run['folds'][name]['branches'][modality]['checkpoint_sha256']:raise ValueError('Parent weights changed')
                for part in ('val','test'):
                    observed=metrics([rows[s]['label'] for s in f[part]],[rows[s]['p_'+modality] for s in f[part]])
                    if observed!=parent_run['folds'][name]['metrics'][part][modality]:raise ValueError('Parent predictions inconsistent')
            c=cfg['branches']['text'];seed(value)
            model=getattr(models,c['model_class'])(**c['model_kwargs']).to(device)
            opt=torch.optim.Adamax(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
            x=torch.tensor(np.stack([np.load(paths[s],allow_pickle=False) for s in f['train']]),dtype=torch.float32)
            y=torch.tensor([rows[s]['label'] for s in f['train']],dtype=torch.long)
            loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y),batch_size=c['batch_size'],shuffle=True,num_workers=0)
            curve=[]
            for epoch in range(c['epochs']):
                model.train();total=0.;count=0
                for xb,yb in loader:
                    opt.zero_grad();loss=torch.nn.functional.cross_entropy(model(xb.to(device)),yb.to(device))
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    loss.backward();opt.step();total+=float(loss.detach().cpu())*len(yb);count+=len(yb)
                curve.append(total/count)
            model.eval();checkpoint=out/f'{name}_text.pt'
            torch.save({'state_dict':model.state_dict(),'experiment':EXPERIMENT,'cohort_id':lock['cohort_id'],
                'folds_sha256':lock['folds_sha256'],'seed':value,'fold':fold,'train_ids':f['train'],'configuration':c},checkpoint)
            with torch.no_grad():
                for sid,r in rows.items():
                    xx=torch.tensor(np.load(paths[sid],allow_pickle=False),dtype=torch.float32,device=device).unsqueeze(0)
                    r.update(experiment=EXPERIMENT,parent_experiment='B0-37',p_text=float(torch.softmax(model(xx),dim=-1)[0,1].cpu()),
                        fulltext_features_sha256=run['feature_manifest_sha256'])
            fusion=select_weights([rows[s] for s in f['val']])
            for r in rows.values():
                r['fusion_weights']=fusion['weights'];r['p_final']=sum(w*r['p_'+m] for w,m in zip(fusion['weights'],('eeg','speech','text')))
                r['prediction']=int(r['p_final']>.5)
            scores={part:{m:metrics([rows[s]['label'] for s in f[part]],[rows[s]['p_'+m] for s in f[part]]) for m in ('eeg','speech','text','final')} for part in ('val','test')}
            run['folds'][name]={'metrics':scores,'fusion':fusion,'text_training_curve':curve,'text_checkpoint_sha256':digest(checkpoint),
                'inherited_eeg_speech_checkpoints':{m:str((parent/f'{name}_{m}.pt').relative_to(ROOT)) for m in ('eeg','speech')}}
            all_rows+=list(rows.values());save(out/'run.json',run)
            (out/'probabilities.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows),encoding='utf-8')
            print(f'FULLTEXT TRAIN seed={value} {name} complete',flush=True)
            del model,opt
        run['cohort_validation']=validate_prediction_records(all_rows,EXPERIMENT);save(out/'run.json',run)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--extract',action='store_true');parser.add_argument('--train',action='store_true')
    args=parser.parse_args()
    if args.extract:extract()
    if args.train:train()
    if not(args.extract or args.train):
        lock,_,_=source_checks();feature_paths(lock);print('Fulltext preflight passed.')

if __name__=='__main__':main()

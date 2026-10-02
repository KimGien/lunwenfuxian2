"""Offline resumable B0-37 extraction. Vendor code remains unchanged."""
import argparse
import io
import importlib.util
import json
import os
import sys
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from prepare import ROOT, save, digest
from cohort import validate_active_cohort
sys.path.insert(0,str(ROOT/'upstream-dev'))
from smoke_models import seed

OUT=ROOT/'features/B0-37'

def speech_preprocessing():
    # Package __init__ eagerly imports WhisperX, which lives in the separate ASR environment.
    path=ROOT/'upstream-dev/lib/preprocessing/speech_preprocessing.py'
    spec=importlib.util.spec_from_file_location('b0_vendor_speech_preprocessing',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def hubert_class():
    # speech/__init__.py instantiates an unrelated XLSR model at import time.
    path=ROOT/'upstream-dev/lib/feature_extraction/speech/chinese_hubert.py'
    spec=importlib.util.spec_from_file_location('b0_vendor_hubert',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.HubertChineseExtractor

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+'.part'); save(tmp,value); tmp.replace(path)

def atomic_array(path,value):
    if not np.isfinite(value).all(): raise ValueError('Nonfinite feature: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.npy.part')
    with tmp.open('wb') as f: np.save(f,value,allow_pickle=False)
    tmp.replace(path)

def code_windows(samples,sr=16000):
    """Reproduce upstream start/end positions, including duplicate last window."""
    size=5*sr; hop=int(2.5*sr)
    if samples<size: return []
    starts=list(range(0,samples-size+1,hop))
    windows=[(s,s+size) for s in starts]
    if samples-starts[-1]>2.5*sr: windows.append((samples-size,samples))
    return windows

class Extraction:
    def __init__(self):
        self.lock=validate_active_cohort(require_data=True)
        self.cfg=read(ROOT/'configs/b0.json')
        self.protocol=read(ROOT/'configs/protocol_resolution.json')
        if read(OUT/'protocol_resolution.json')!=self.protocol: raise ValueError('Protocol copies differ')
        self.protocol_hash=digest(OUT/'protocol_resolution.json')
        self.producer_hash=digest(Path(__file__))
        self.backbone_hash=digest(ROOT/'manifests/backbones.json')
        vendor={p.relative_to(ROOT).as_posix():digest(p) for p in sorted((ROOT/'upstream-dev/lib').rglob('*.py'))}
        vendor_path=OUT/'vendor_code_lock.json'
        if vendor_path.exists() and read(vendor_path)!=vendor: raise ValueError('Vendor code changed')
        if not vendor_path.exists(): atomic_json(vendor_path,vendor)
        self.vendor_hash=digest(vendor_path)
        self.rows=read(OUT/'features_manifest.json') if (OUT/'features_manifest.json').exists() else []
        self.staged={r['subject_id']:r for r in read(ROOT/'manifests/staged_dataset.json')}
        self.raw_hash={r['path']:r['sha256'] for r in read(ROOT/'manifests/modma_raw_files.json')}
        self.subjects={r['subject_id']:r for r in read(ROOT/'manifests/subjects.json')}
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(4)
        seed(11)
        for name,item in read(ROOT/'manifests/backbones.json').items():
            if name not in ('macbert','hubert_code'): continue
            for filename,entry in item['files'].items():
                if digest(ROOT/'models'/name/filename)!=entry['sha256']: raise ValueError('Backbone changed')

    def source_audio(self,sid,q):
        path=ROOT/'staged'/sid/'audio'/f'{q:02d}.wav'
        source=str(Path(self.subjects[sid]['audio_dir'])/f'{q:02d}.wav')
        h=digest(path)
        if h!=self.raw_hash[source]: raise ValueError('Staged audio changed: '+str(path))
        return path,h

    def existing(self,sid,modality,seed_value):
        matches=[r for r in self.rows if (r['subject_id'],r['modality'],r['seed'])==(sid,modality,seed_value)]
        if not matches: return False
        if len(matches)!=1: raise ValueError('Duplicate cached feature')
        r=matches[0]
        for k,v in self.common().items():
            if r.get(k)!=v: raise ValueError('Stale feature '+k)
        p=OUT/r['path']
        if digest(p)!=r['sha256']: raise ValueError('Cached feature hash mismatch')
        x=np.load(p,allow_pickle=False)
        if list(x.shape)!=self.cfg['branches'][modality]['shape'] or not np.isfinite(x).all(): raise ValueError('Invalid cached feature')
        return True

    def common(self):
        return {'cohort_id':self.lock['cohort_id'],'folds_sha256':self.lock['folds_sha256'],
            'protocol_sha256':self.protocol_hash,'producer_sha256':self.producer_hash,
            'backbones_sha256':self.backbone_hash,'vendor_code_sha256':self.vendor_hash,
            'producer_id':'extract_b0.py:B0-37-code-v1','fit_subject_ids':[],'outer_fold':None}

    def record(self,sid,modality,seed_value,x,details):
        expected=self.cfg['branches'][modality]['shape']
        if list(x.shape)!=expected: raise ValueError(f'Wrong shape {x.shape}, expected {expected}')
        relative=f'{modality}/'+(f'seed_{seed_value}/' if seed_value is not None else '')+sid+'.npy'
        path=OUT/relative; atomic_array(path,x.astype(np.float32))
        row={**self.common(),'subject_id':sid,'modality':modality,'seed':seed_value,'path':relative,
            'sha256':digest(path),'shape':list(x.shape),'details':details}
        self.rows.append(row); atomic_json(OUT/'features_manifest.json',self.rows)
        print(f'FEATURE {modality} seed={seed_value} {sid} {list(x.shape)}',flush=True)

    def eeg(self,ids):
        import mne
        from scipy.io import loadmat
        from lib.models.cbramod.cbramod import CBraMod
        checkpoint=ROOT/'upstream-dev/cbramod_pretrained_weights/pretrained_weights2.pth'
        checkpoint_hash=digest(checkpoint)
        source=read(ROOT/'manifests/source_sha256.json')
        entry=source.get(str(checkpoint.relative_to(ROOT.parent)))
        if entry is None or entry['sha256']!=checkpoint_hash: raise ValueError('Unpinned CBraMod checkpoint')
        model=CBraMod(in_dim=200,out_dim=200,d_model=200,dim_feedforward=800,seq_len=30,n_layer=12,nhead=8)
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True); ckpt=ckpt.get('state_dict',ckpt)
        ckpt={k.replace('backbone.',''):v for k,v in ckpt.items()}
        model.proj_out=torch.nn.Identity()
        own=model.state_dict()
        if any(k not in ckpt or v.shape!=ckpt[k].shape for k,v in own.items()): raise ValueError('Missing retained CBraMod weights')
        model.load_state_dict({k:ckpt[k] for k in own},strict=True); model.to(self.device).eval()
        c=self.cfg['preprocessing']['eeg']
        for sid in ids:
            path=ROOT/self.staged[sid]['eeg_mat']
            if digest(path)!=self.staged[sid]['mat_sha256']: raise ValueError('Staged EEG changed')
            if self.existing(sid,'eeg',None): continue
            x=loadmat(path)['eeg_mat'][c['python_indices']]
            raw=mne.io.RawArray(x,mne.create_info(c['channels'],250,'eeg'),verbose='ERROR')
            raw.resample(200,verbose='ERROR'); raw.filter(.3,75,verbose='ERROR'); raw.notch_filter(50,verbose='ERROR')
            if raw.n_times<60000: raise ValueError('EEG shorter than300s')
            cropped=raw.get_data(start=0,stop=60000)
            patches=cropped.reshape(19,60,5,200).transpose(1,0,2,3).copy()
            embeddings=[]
            with torch.no_grad():
                for i in range(0,60,4):
                    y=model(torch.tensor(patches[i:i+4],dtype=torch.float32,device=self.device))
                    embeddings.append(y.mean(dim=(1,2)).cpu().numpy())
            self.record(sid,'eeg',None,np.concatenate(embeddings),{'source_sha256':self.staged[sid]['mat_sha256'],
                'checkpoint_sha256':checkpoint_hash,'resampled_samples':int(raw.n_times),'retained_samples':60000,
                'discarded_seconds':float((raw.n_times-60000)/200),'native_scale':True})

    def speech(self,ids):
        import librosa
        import soundfile as sf
        preprocess_audio=speech_preprocessing().preprocess_audio
        HubertChineseExtractor=hubert_class()
        from lib.models.encoders.cnn_bigru import BiGRUAttentionEncoder
        extractor=HubertChineseExtractor(str(ROOT/'models/hubert_code'),torch.device(self.device))
        encoders={}; weight_hashes={}
        for value in self.cfg['seeds']:
            seed(value); encoder=BiGRUAttentionEncoder(input_dim=768).eval()
            path=OUT/'encoders'/f'seed_{value}.pt'; path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                old=torch.load(path,map_location='cpu',weights_only=True)
                if old['seed']!=value or old['protocol_sha256']!=self.protocol_hash: raise ValueError('Encoder provenance mismatch')
                if any(not torch.equal(v,old['state_dict'][k]) for k,v in encoder.state_dict().items()): raise ValueError('Encoder initialization changed')
                encoder.load_state_dict(old['state_dict'])
            else:
                torch.save({'seed':value,'protocol_sha256':self.protocol_hash,'state_dict':encoder.state_dict(),'fit_subject_ids':[]},path)
            weight_hashes[value]=digest(path); encoders[value]=encoder.to(self.device).requires_grad_(False)
        for sid in ids:
            source_hashes={q:self.source_audio(sid,q)[1] for q in range(1,30)}
            done={v:self.existing(sid,'speech',v) for v in encoders}
            if all(done.values()): continue
            outputs={v:[] for v in encoders}; audit=[]
            for q in range(1,30):
                path,_=self.source_audio(sid,q)
                cache=OUT/'hubert_segments'/sid/f'{q:02d}.npy'; meta=cache.with_suffix('.json')
                expected={'source_sha256':source_hashes[q],'protocol_sha256':self.protocol_hash,
                    'producer_sha256':self.producer_hash,'backbones_sha256':self.backbone_hash}
                expected['vendor_code_sha256']=self.vendor_hash
                if cache.exists() and meta.exists():
                    info=read(meta)
                    if any(info.get(k)!=v for k,v in expected.items()) or info['sha256']!=digest(cache): raise ValueError('Stale HuBERT cache')
                    vectors=np.load(cache,allow_pickle=False)
                else:
                    y,_=librosa.load(path,sr=16000,mono=True); y=preprocess_audio(y)
                    windows=code_windows(len(y)); features=[]
                    # Upstream sorts filenames lexicographically (part1,part10,...).
                    for n in sorted(range(len(windows)),key=lambda i:f'{q:02d}_part{i+1}.wav'):
                        start,end=windows[n]
                        buffer=io.BytesIO(); sf.write(buffer,y[start:end],16000,format='WAV',subtype='PCM_16'); buffer.seek(0)
                        segment,_=sf.read(buffer,dtype='float32')
                        values=extractor.processor(segment,sampling_rate=16000,return_tensors='pt',padding='longest',truncation=True,max_length=80000).input_values.to(self.device)
                        with torch.no_grad(): vector=extractor.model(values).squeeze(0).mean(dim=0).cpu().numpy()
                        features.append(vector)
                    vectors=np.stack(features) if features else np.empty((0,768),dtype=np.float32)
                    atomic_array(cache,vectors)
                    info={**expected,'sha256':digest(cache),'processed_samples':len(y),'windows_samples':windows,
                        'segment_count':len(windows),'zero_feature':not bool(windows),'duplicate_windows':len(windows)-len(set(windows))}
                    atomic_json(meta,info)
                if vectors.ndim!=2 or vectors.shape[1]!=768 or not np.isfinite(vectors).all(): raise ValueError('Invalid segment features')
                audit.append({'question_id':f'{q:02d}',**info})
                for value,encoder in encoders.items():
                    with torch.no_grad():
                        z=encoder(torch.tensor(vectors,device=self.device)).mean(dim=0).cpu().numpy() if len(vectors) else np.zeros(512,dtype=np.float32)
                    outputs[value].append(z)
                print(f'HUBERT {sid}/{q:02d} segments={len(vectors)}',flush=True)
            atomic_json(OUT/'speech_audits'/f'{sid}.json',audit)
            for value in encoders:
                if not done[value]: self.record(sid,'speech',value,np.stack(outputs[value]),{'encoder_sha256':weight_hashes[value],
                    'source_sha256_by_question':source_hashes,'audit_sha256':digest(OUT/'speech_audits'/f'{sid}.json')})

    def text(self,ids):
        from transformers import BertModel,BertTokenizer
        tokenizer=BertTokenizer.from_pretrained(ROOT/'models/macbert',local_files_only=True)
        model=BertModel.from_pretrained(ROOT/'models/macbert',local_files_only=True).to(self.device).eval()
        for sid in ids:
            texts=[]; hashes=[]; token_counts=[]
            for q in range(1,30):
                p=ROOT/'transcripts/B0-37/first_segment_code'/sid/f'{q:02d}.json'
                row=read(p); _,source_hash=self.source_audio(sid,q)
                if (row['source_sha256']!=source_hash or row['cohort_id']!=self.lock['cohort_id'] or
                    row['folds_sha256']!=self.lock['folds_sha256'] or row['model_lock_sha256']!=self.backbone_hash or
                    row['export_policy']!='first_segment_code'): raise ValueError('Transcript provenance mismatch')
                if row['empty'] or not row['answer_text'].strip() or '[ERROR]' in row['answer_text']: raise ValueError('Unresolved empty/error transcript '+str(p))
                texts.append(row['answer_text']); hashes.append(digest(p))
                token_counts.append(len(tokenizer.encode(row['answer_text'],truncation=False)))
            if self.existing(sid,'text',None): continue
            inputs=tokenizer(texts,padding=True,truncation=True,max_length=512,return_tensors='pt').to(self.device)
            with torch.no_grad(): x=model(**inputs).last_hidden_state[:,0,:].cpu().numpy()
            self.record(sid,'text',None,x,{'transcript_sha256':hashes,'untruncated_token_counts':token_counts,
                'truncated_questions':[i+1 for i,n in enumerate(token_counts) if n>512]})

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--modality',choices=['eeg','speech','text'],required=True)
    parser.add_argument('--subject',help='One locked subject for smoke verification')
    args=parser.parse_args(); extraction=Extraction()
    ids=[args.subject] if args.subject else extraction.lock['subject_ids']
    if not set(ids)<=set(extraction.lock['subject_ids']): raise ValueError('Subject outside cohort')
    getattr(extraction,args.modality)(ids)

if __name__=='__main__': main()

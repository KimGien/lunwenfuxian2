"""Fetch public HF model files with pinned revisions and resumable local reuse."""
import argparse
import json
from pathlib import Path
import urllib.request
import subprocess
import sys
from prepare import digest, save, ROOT
sys.stdout.reconfigure(encoding='utf-8')

MODELS={'macbert':'hfl/chinese-macbert-base',
        'hubert_code':'xmj2002/hubert-base-ch-speech-emotion-recognition'}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--weights',action='store_true'); parser.add_argument('--asr',action='store_true'); args=parser.parse_args()
    lock_path=ROOT/'manifests/backbones.json'
    lock=json.loads(lock_path.read_text()) if lock_path.exists() else {}
    selected={'whisper_medium':'Systran/faster-whisper-medium'} if args.asr else MODELS
    for key,repo in selected.items():
        try:
            if key not in lock:
                meta_path=ROOT/'sources'/f'{key}_hf_metadata.json'
                subprocess.run(['curl.exe','-sS','--fail','--retry','2','--connect-timeout','15','--max-time','60','-o',str(meta_path),'https://huggingface.co/api/models/'+repo],check=True)
                metadata=json.loads(meta_path.read_text(encoding='utf-8'))
                save(ROOT/'sources'/f'{key}_hf_metadata.json',metadata)
                lock[key]={'repo':repo,'revision':metadata['sha'],'files':{},'status':'metadata_pinned'}
                save(lock_path,lock)
            item=lock[key]
            meta=json.loads((ROOT/'sources'/f'{key}_hf_metadata.json').read_text())
            names=[s['rfilename'] for s in meta['siblings']]
            chosen=[n for n in names if '/' not in n and (n.endswith('.json') or n in ('vocab.txt','README.md'))]
            if args.weights:
                chosen += ['model.bin'] if args.asr else (['model.safetensors'] if 'model.safetensors' in names else ['pytorch_model.bin'])
            if args.asr: chosen += [n for n in names if n=='vocabulary.txt']
            for name in chosen:
                target=ROOT/'models'/key/name; target.parent.mkdir(parents=True,exist_ok=True)
                if name in item['files'] and target.exists() and digest(target)==item['files'][name]['sha256']:
                    continue
                url=f'https://huggingface.co/{repo}/resolve/{item["revision"]}/{name}'
                print(f'Downloading {key}/{name}',flush=True)
                part=target.with_suffix(target.suffix+'.part')
                # Windows certificate chain/hostname validation remains enabled;
                # unavailable CRL endpoints are treated as best-effort for this request.
                subprocess.run(['curl.exe','--ssl-revoke-best-effort','-sS','-L','--fail','--retry','2','--connect-timeout','20','--max-time','900','-C','-','-o',str(part),url],check=True)
                part.replace(target)
                item['files'][name]={'sha256':digest(target),'bytes':target.stat().st_size,'url':url}
                save(lock_path,lock)
            item['status']='weights_downloaded' if args.weights else 'metadata_pinned'
            item.pop('error',None)
        except Exception as exc:
            lock.setdefault(key,{'repo':repo})['error']=str(exc)
            print(f'{key}: {exc}',flush=True)
        save(lock_path,lock)
    return int(any('error' in item for item in lock.values()))

if __name__=='__main__': raise SystemExit(main())

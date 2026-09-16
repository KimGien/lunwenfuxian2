"""Explicit offline transcription entry; writes full ASR segments for traceability."""
import argparse
import json
import time
from asr_runtime import ROOT,load
from prepare import save,digest
from cohort import validate_active_cohort

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--export-policy',choices=['first_segment_code','all_segments_paper'],required=True)
    args=parser.parse_args()
    lock=validate_active_cohort(require_data=True)
    audit=json.loads((ROOT/'reports/modma_audit.json').read_text())
    if not audit['passed']: raise ValueError('Resolve invalid recordings and signal-quality flags before complete B0 transcription')
    if not args.run:
        print('Preflight passed; add --run to transcribe.'); return
    model=load(); import whisperx
    started=time.time(); completed=0; empty=[]
    model_hash=digest(ROOT/'manifests/backbones.json')
    subjects=json.loads((ROOT/'manifests/subjects.json').read_text())
    for subject in subjects:
        sid=subject['subject_id']
        for q in range(1,30):
            path=ROOT/'staged'/sid/'audio'/f'{q:02d}.wav'
            target=ROOT/'transcripts'/'B0-37'/args.export_policy/sid/f'{q:02d}.json'
            if target.exists():
                old=json.loads(target.read_text(encoding='utf-8'))
                if old.get('cohort_id')!=lock['cohort_id'] or old.get('folds_sha256')!=lock['folds_sha256']:
                    raise ValueError('Transcript belongs to another cohort or fold policy')
                if old['source_sha256']!=digest(path) or old['model_lock_sha256']!=digest(ROOT/'manifests/backbones.json'):
                    raise ValueError('Transcript source or model provenance changed')
                completed+=1
                if old['empty']: empty.append(f'{sid}/{q:02d}')
                continue
            audio=whisperx.load_audio(str(path))
            result=model.transcribe(audio,batch_size=1,language='zh',task='transcribe')
            segments=result['segments']
            text=(''.join(s['text'] for s in segments) if args.export_policy=='all_segments_paper' else (segments[0]['text'] if segments else ''))
            record={'cohort_id':lock['cohort_id'],'folds_sha256':lock['folds_sha256'],'subject_id':sid,'question_id':f'{q:02d}','source_sha256':digest(path),'export_policy':args.export_policy,'model_lock_sha256':model_hash,'segments':segments,'answer_text':text,'empty':not bool(text.strip()),'all_segments_text':''.join(s['text'] for s in segments),'runtime':'whisperx_cpu_float32_batch1_zh_medium'}
            temporary=target.with_suffix('.json.part'); save(temporary,record); temporary.replace(target)
            completed+=1
            if record['empty']: empty.append(f'{sid}/{q:02d}')
            save(ROOT/'reports/transcription_progress.json',{'completed':completed,'expected':1073,'empty_recordings':empty,'elapsed_seconds':time.time()-started,'status':'running','export_policy':args.export_policy})
            print(f'ASR {completed}/1073 {sid}/{q:02d} segments={len(segments)} empty={record["empty"]}',flush=True)
    save(ROOT/'reports/transcription_progress.json',{'completed':completed,'expected':1073,'empty_recordings':empty,'elapsed_seconds':time.time()-started,'status':'needs_empty_review' if empty else 'complete','export_policy':args.export_policy})

if __name__=='__main__': main()

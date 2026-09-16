"""Audit full B0-37 ASR coverage, input/model provenance and export consistency."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
from prepare import ROOT,save,digest
from cohort import validate_active_cohort

def main():
    lock=validate_active_cohort(require_data=True)
    folder=ROOT/'transcripts/B0-37/first_segment_code'
    expected={f'{s}/{q:02d}' for s in lock['subject_ids'] for q in range(1,30)}
    found={f'{p.parent.name}/{p.stem}' for p in folder.glob('*/*.json')}
    model_hash=digest(ROOT/'manifests/backbones.json')
    errors=[];empty=[];rows=[];retry=[]
    backbones=json.loads((ROOT/'manifests/backbones.json').read_text(encoding='utf-8'))
    for name,entry in backbones['whisper_medium']['files'].items():
        if digest(ROOT/'models/whisper_medium'/name)!=entry['sha256']:
            errors.append('Whisper model asset changed: '+name)
    for key in sorted(expected&found):
        sid,q=key.split('/');path=folder/sid/(q+'.json');r=json.loads(path.read_text(encoding='utf-8'))
        source=ROOT/'staged'/sid/'audio'/(q+'.wav')
        checks={'subject_id':sid,'question_id':q,'cohort_id':lock['cohort_id'],
            'folds_sha256':lock['folds_sha256'],'model_lock_sha256':model_hash,
            'source_sha256':digest(source),'export_policy':'first_segment_code'}
        for name,value in checks.items():
            if r.get(name)!=value:errors.append(key+': mismatched '+name)
        text=r.get('answer_text','');segments=r.get('segments',[])
        if text!=(segments[0]['text'] if segments else ''):errors.append(key+': first-segment export mismatch')
        if r.get('all_segments_text')!=''.join(s['text'] for s in segments):errors.append(key+': full segment export mismatch')
        if r.get('empty')!=(not bool(text.strip())):errors.append(key+': empty flag mismatch')
        if not text.strip():empty.append(key)
        if '[ERROR]' in text:errors.append(key+': error placeholder')
        if r.get('recovery'):
            recovery=r['recovery'];evidence=ROOT/f'reports/asr_empty_retry_{sid}_{q}.json'
            original=ROOT/'archive/asr_original_empty'/sid/(q+'.json')
            if digest(evidence)!=recovery['evidence_sha256'] or digest(original)!=recovery['original_transcript_sha256']:
                errors.append(key+': recovery evidence changed')
            ev=json.loads(evidence.read_text(encoding='utf-8'))
            candidate=next(c for c in ev['candidates'] if c['gain']==recovery['gain'])
            if candidate['segments']!=segments or ev['source_sha256']!=checks['source_sha256']:
                errors.append(key+': recovery candidate/source mismatch')
            retry.append({'recording':key,**recovery})
        rows.append({'recording':key,'sha256':digest(path),'segments':len(segments)})
    missing=sorted(expected-found);extra=sorted(found-expected)
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'cohort_id':lock['cohort_id'],
        'folds_sha256':lock['folds_sha256'],'expected':len(expected),'completed':len(found&expected),
        'missing':missing,'extra':extra,'empty_recordings':empty,'errors':errors,
        'recovery_exceptions':retry,'files':rows,'passed':not(missing or extra or empty or errors)}
    save(ROOT/'reports/transcription_audit.json',report)
    if report['passed']:
        save(ROOT/'manifests/transcripts_sha256.json',{r['recording']:r['sha256'] for r in rows})
        progress=ROOT/'reports/transcription_progress.json'
        old=json.loads(progress.read_text(encoding='utf-8')) if progress.exists() else {}
        old.update(completed=len(expected),expected=len(expected),empty_recordings=[],status='complete',
            verified_utc=report['timestamp_utc'],audit_sha256=digest(ROOT/'reports/transcription_audit.json'))
        tmp=progress.with_suffix('.json.part');save(tmp,old);tmp.replace(progress)
    print(json.dumps({k:v for k,v in report.items() if k not in ('files','recovery_exceptions')},indent=2))
    return 0 if report['passed'] else 2

if __name__=='__main__':raise SystemExit(main())

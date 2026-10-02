"""Verify raw EEG/WAV pairing, labels, dimensions and waveform integrity."""
from collections import Counter
import json
from pathlib import Path
import wave
import hashlib
import numpy as np
from scipy.io import loadmat
from prepare import ROOT, WORK, save, digest
from cohort import validate_active_cohort

def main():
    lock=validate_active_cohort(require_subjects=False)
    base=WORK/'dataset/modma/raw'
    eeg_dir=base/'modma-dataset-2015/EEG_128channels_resting_lanzhou_2015'
    audio_dir=base/'modma-audio-group-6-dsai/MODMA_AUDIO Datatset/audio_lanzhou_2015'
    folds=json.loads((ROOT/'manifests/fold_assignments.json').read_text())
    cohort=sorted(set(s for f in folds.values() for s in f['test']))
    meta=json.loads((ROOT/'manifests/modma_labels.json').read_text())
    labels_e=meta['modma-dataset-2015.zip']['labels']
    labels_a=meta['modma-audio-group-6-dsai.zip']['labels']
    rows=[]; errors=[]; eeg_details=[]; wav_details=[]; bad_audio=[]; suspect_audio=[]
    quality_path=ROOT/'reports/audio_signal_quality.json'
    quality=json.loads(quality_path.read_text(encoding='utf-8')) if quality_path.exists() else {}
    screened={(r['subject_id'],r['question_id']):r for r in quality.get('files',[])}
    if not quality: errors.append('Missing signal quality screen; run scripts/audit_audio_signal.py')
    if quality.get('cohort_id')!=lock['cohort_id'] or quality.get('folds_sha256')!=lock['folds_sha256']:
        errors.append('Signal quality screen belongs to another cohort or fold policy')
    for sid in cohort:
        sets=list(eeg_dir.glob(sid+'*.set'))
        wavs=sorted((audio_dir/sid).glob('*.wav'))
        if len(sets)!=1 or {p.name for p in wavs}!={f'{q:02d}.wav' for q in range(1,30)}:
            errors.append(f'{sid}: missing/ambiguous raw files'); continue
        if sid not in labels_a or sid not in labels_e or labels_a[sid]['label']!=labels_e[sid]['label']:
            errors.append(f'{sid}: conflicting/missing diagnosis metadata'); continue
        eeg=loadmat(sets[0],simplify_cells=True)
        eeg=eeg.get('EEG',eeg)
        nbchan=int(eeg['nbchan']); pnts=int(eeg['pnts']); trials=int(eeg['trials']); sfreq=float(eeg['srate'])
        fdt=sets[0].with_suffix('.fdt')
        if fdt.stat().st_size!=nbchan*pnts*trials*4: errors.append(f'{sid}: FDT byte count mismatch')
        signal=np.memmap(fdt,mode='r',dtype='<f4',shape=(pnts*trials,nbchan))
        finite=bool(np.isfinite(signal).all())
        if not finite: errors.append(f'{sid}: EEG NaN/Inf')
        channels=eeg.get('chanlocs',[])
        if isinstance(channels,dict): channels=[channels]
        names=[str(c.get('labels','')) for c in channels]
        eeg_details.append({'subject_id':sid,'set_path':str(sets[0].relative_to(WORK)),'nbchan':nbchan,'samples':pnts,'trials':trials,'sample_rate':sfreq,'duration_seconds':pnts/sfreq,'finite':finite,'abs_max_native':float(np.abs(signal).max()),'channel_names':names,'eeglab_data_reference':str(eeg.get('data','')),'unit_status':'EEGLAB convention microvolts; original acquisition/export provenance still requires verification'})
        del signal
        invalid_questions=[]; suspect_questions=[]; quality_verified=True
        for wav in wavs:
            try:
                with wave.open(str(wav),'rb') as f:
                    entry={'subject_id':sid,'question_id':wav.stem,'path':str(wav.relative_to(WORK)),'sample_rate':f.getframerate(),'channels':f.getnchannels(),'sample_width':f.getsampwidth(),'frames':f.getnframes(),'duration_seconds':f.getnframes()/f.getframerate()}
                    if f.getnframes()<=0: raise ValueError('empty waveform')
                    h=hashlib.sha256(); read_bytes=0
                    while chunk:=f.readframes(65536): h.update(chunk); read_bytes+=len(chunk)
                    if read_bytes != f.getnframes()*f.getnchannels()*f.getsampwidth():
                        raise ValueError('truncated PCM payload')
                    entry['pcm_sha256']=h.hexdigest()
                    checked=screened.get((sid,wav.stem))
                    if not checked or checked['sha256']!=digest(wav):
                        quality_verified=False
                        errors.append(f'{sid}/{wav.name}: missing/stale signal quality screen')
                    elif checked['flagged_full_second_indices']:
                        suspect_questions.append(wav.stem)
                        suspect_audio.append(checked)
                        errors.append(f'{sid}/{wav.name}: severe noise flagged in signal quality screen')
                    wav_details.append(entry)
            except (wave.Error,ValueError,EOFError) as exc:
                invalid_questions.append(wav.stem)
                bad_audio.append({'subject_id':sid,'question_id':wav.stem,'path':str(wav.relative_to(WORK)),'error':str(exc),'sha256':digest(wav)})
                errors.append(f'{sid}/{wav.name}: {exc}')
        rows.append({'subject_id':sid,'label':labels_e[sid]['label'],'label_status':'verified_agree_EEG_and_audio_workbooks','eeg_set':str(sets[0].relative_to(WORK)),'eeg_fdt':str(fdt.relative_to(WORK)),'audio_dir':str((audio_dir/sid).relative_to(WORK)),'questions':29,'invalid_questions':invalid_questions,'suspect_questions':suspect_questions,'signal_quality_verified':quality_verified,'raw_complete':not invalid_questions and not suspect_questions and quality_verified})
    save(ROOT/'manifests/subjects.json',rows)
    save(ROOT/'reports/modma_eeg_audit.json',eeg_details)
    save(ROOT/'reports/modma_wav_audit.json',wav_details)
    save(ROOT/'reports/modma_bad_audio.json',bad_audio)
    save(ROOT/'reports/modma_suspect_audio.json',suspect_audio)
    pcm_groups={}
    for r in wav_details: pcm_groups.setdefault(r['pcm_sha256'],[]).append((r['subject_id'],r['question_id']))
    duplicates=[v for v in pcm_groups.values() if len(v)>1]
    cross_subject=[v for v in duplicates if len({p[0] for p in v})>1]
    if cross_subject: errors.append('Cross-subject duplicate PCM contents found')
    report={'passed':not errors and len(rows)==len(lock['subject_ids']),'cohort':len(rows),'complete_subjects':sum(r['raw_complete'] for r in rows),'diagnosis_counts':dict(Counter(r['label'] for r in rows)),'wav_count':len(wav_details),'invalid_wav_count':len(bad_audio),'eeg_count':len(eeg_details),'eeg_sample_rates':dict(Counter(r['sample_rate'] for r in eeg_details)),'wav_sample_rates':dict(Counter(r['sample_rate'] for r in wav_details)),'wav_hours':sum(r['duration_seconds'] for r in wav_details)/3600,'duplicate_pcm_groups':duplicates,'errors':errors,'original_archive_publisher_not_verified':True}
    report.update(passed=not errors and len(rows)==len(lock['subject_ids']),
                  cohort_id=lock['cohort_id'],folds_sha256=lock['folds_sha256'],
                  subject_manifest_sha256=digest(ROOT/'manifests/subjects.json'),
                  excluded_subject_ids=lock['excluded_subject_ids'],
                  suspect_wav_count=len(suspect_audio),
                  decodable_without_signal_flags=len(wav_details)-len(suspect_audio),
                  signal_screen_report='reports/audio_signal_quality.json',
                  signal_screen_sha256=digest(quality_path) if quality_path.exists() else None)
    save(ROOT/'reports/modma_audit.json',report)
    print(json.dumps(report,indent=2))
    return int(not report['passed'])

if __name__=='__main__': raise SystemExit(main())

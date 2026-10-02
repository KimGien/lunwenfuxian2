"""Screen WAV contents for severe noise; flags require review, not imputation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from cohort import validate_active_cohort

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent

def main():
    lock=validate_active_cohort()
    subjects = json.loads((ROOT/'manifests/subjects.json').read_text())
    screened = []
    suspects = []
    for subject in subjects:
        for path in sorted((WORK/subject['audio_dir']).glob('*.wav')):
            try:
                samples, sr = sf.read(path, dtype='float32', always_2d=True)
            except RuntimeError:
                continue  # Header/decode failures are already recorded by audit_modma.py.
            mono = samples.mean(axis=1)
            full_seconds = len(mono)//sr
            windows = mono[:full_seconds*sr].reshape(full_seconds, sr)
            rms = np.sqrt(np.mean(windows**2, axis=1))
            clipped = np.mean(np.abs(windows)>0.95, axis=1)
            centered = windows-windows.mean(axis=1, keepdims=True)
            corr = np.sum(centered[:,1:]*centered[:,:-1], axis=1)/np.maximum(
                np.sqrt(np.sum(centered[:,1:]**2,axis=1)*np.sum(centered[:,:-1]**2,axis=1)),1e-20)
            flagged = np.flatnonzero((rms>0.4)&(clipped>0.01)&(np.abs(corr)<0.05)).tolist()
            row = {'subject_id':subject['subject_id'], 'question_id':path.stem,
                   'path':str(path.relative_to(WORK)),
                   'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                   'sample_rate':sr, 'frames':len(mono), 'full_seconds_screened':full_seconds,
                   'flagged_full_second_indices':flagged,
                   'max_window_rms':float(rms.max()) if full_seconds else None}
            screened.append(row)
            if flagged:
                row = dict(row)
                row['windows'] = [{'start_seconds':i,'rms':float(rms[i]),
                                   'fraction_abs_gt_0_95':float(clipped[i]),
                                   'lag1_correlation':float(corr[i])} for i in range(full_seconds)]
                suspects.append(row)
    report = {'checked_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'37 active MODMA subjects; every decodable WAV; full one-second windows',
              'cohort_id':lock['cohort_id'],'folds_sha256':lock['folds_sha256'],
              'rule':'RMS > 0.4 AND fraction(|PCM| > 0.95) > 0.01 AND |lag-1 correlation| < 0.05',
              'limitations':'A conservative severe-noise screen, not a guarantee of perceptual quality. Flags are suspects; no samples are reconstructed or deleted. The final subsecond tail is not screened.',
              'files_screened':len(screened), 'suspect_count':len(suspects),
              'suspects':suspects, 'files':screened}
    out=ROOT/'reports/audio_signal_quality.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['files_screened','suspect_count','suspects']},indent=2))

if __name__=='__main__':
    main()

"""Create an upstream-compatible workspace without changing raw data.

EEGLAB channel-major MATLAB arrays are recovered from interleaved float32 FDT.
No scaling, resampling, channel selection or normalization is performed here.
"""
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.io import loadmat, savemat
from prepare import ROOT, WORK, save, digest
from cohort import validate_active_cohort

def main():
    lock=validate_active_cohort(require_data=True)
    rows=json.loads((ROOT/'manifests/subjects.json').read_text())
    sources=json.loads((ROOT/'manifests/modma_raw_files.json').read_text())
    source_hash={r['path']:r['sha256'] for r in sources}
    out=ROOT/'staged'; results=[]
    extras=[p for p in out.iterdir() if p.is_dir() and p.name not in lock['subject_ids']]
    if extras: raise ValueError('Non-cohort staged directories must be archived: '+str(extras))
    for row in rows:
        sid=row['subject_id']; folder=out/sid; (folder/'eeg').mkdir(parents=True,exist_ok=True)
        eeg=loadmat(WORK/row['eeg_set'],simplify_cells=True); eeg=eeg.get('EEG',eeg)
        shape=(int(eeg['pnts'])*int(eeg['trials']),int(eeg['nbchan']))
        x=np.memmap(WORK/row['eeg_fdt'],mode='r',dtype='<f4',shape=shape).T
        dest=folder/'eeg'/f'{sid}_native_eeg.mat'
        if not dest.exists():
            savemat(dest,{'eeg_mat':np.asarray(x),'samplingRate':np.array([[float(eeg['srate'])]]),'channel_names':np.array([c['labels'] for c in eeg['chanlocs']],dtype=object)},do_compression=False)
        check=loadmat(dest)['eeg_mat']
        if not np.array_equal(x,check): raise ValueError(f'{sid}: MATLAB conversion changed waveform values')
        del x,check
        (folder/'audio').mkdir(exist_ok=True)
        copied=[]; quarantined=[]
        for question in range(1,30):
            q=f'{question:02d}'
            if q in row['invalid_questions'] or q in row.get('suspect_questions',[]):
                old=folder/'audio'/(q+'.wav')
                if old.exists():
                    # Keep a review copy outside the directory scanned by upstream code.
                    original=WORK/row['audio_dir']/(q+'.wav')
                    expected=source_hash[str(original.relative_to(WORK))]
                    if digest(old)!=expected: raise ValueError('Unexpected staged bytes: '+str(old))
                    quarantine=ROOT/'recovery/quarantine/staged'/sid/(q+'.wav')
                    quarantine.parent.mkdir(parents=True,exist_ok=True)
                    if not old.resolve().is_relative_to(out.resolve()) or not quarantine.resolve().is_relative_to((ROOT/'recovery').resolve()):
                        raise ValueError('Quarantine path outside workspace')
                    if quarantine.exists(): raise ValueError('Quarantine already exists; refusing overwrite')
                    old.rename(quarantine)
                    quarantined.append(str(quarantine.relative_to(ROOT)))
                continue
            original=WORK/row['audio_dir']/(q+'.wav'); target=folder/'audio'/(q+'.wav')
            if not target.exists(): shutil.copyfile(original,target)
            expected=source_hash[str(original.relative_to(WORK))]
            if digest(target)!=expected: raise ValueError('Staged audio mismatch: '+str(target))
            copied.append(q)
        results.append({'subject_id':sid,'eeg_mat':str(dest.relative_to(ROOT)),'mat_sha256':digest(dest),'fdt_sha256':source_hash[row['eeg_fdt']],'conversion':'float32 interleaved FDT -> channels x time; exact numerical roundtrip; native values preserved','audio_questions':copied,'missing_questions':row['invalid_questions'],'suspect_questions':row.get('suspect_questions',[]),'quarantined_this_run':quarantined})
    shutil.copyfile(ROOT/'manifests/fold_assignments.json',out/'fold_assignments.json')
    save(ROOT/'manifests/staged_dataset.json',results)
    print(json.dumps({'subjects':len(results),'mat_roundtrips_passed':len(results),'audio_files':sum(len(r['audio_questions']) for r in results),'missing_recordings':sum(len(r['missing_questions']) for r in results),'suspect_recordings':sum(len(r['suspect_questions']) for r in results)},indent=2))

if __name__=='__main__': main()

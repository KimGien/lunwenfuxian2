"""Audit actual EEGLAB metadata and lock the disclosed B0-37 code-based policy."""
import json
import numpy as np
from scipy.io import loadmat
import mne
from prepare import ROOT, WORK, save, digest
from cohort import validate_active_cohort

def main():
    lock=validate_active_cohort(require_data=True)
    cfg=json.loads((ROOT/'configs/b0.json').read_text())
    subjects=json.loads((ROOT/'manifests/subjects.json').read_text())
    indices=cfg['preprocessing']['eeg']['python_indices']
    names=cfg['preprocessing']['eeg']['channels']
    rows=[]
    for r in subjects:
        e=loadmat(WORK/r['eeg_set'],simplify_cells=True); e=e.get('EEG',e)
        native=np.memmap(WORK/r['eeg_fdt'],mode='r',dtype='<f4',shape=(int(e['pnts']),int(e['nbchan']))).T
        raw=mne.io.read_raw_eeglab(WORK/r['eeg_set'],preload=False,verbose='ERROR')
        observed=raw.get_data(start=0,stop=1000)
        expected=np.asarray(native[:,:1000],dtype=np.float64)*1e-6
        equal=bool(np.allclose(observed,expected,rtol=1e-12,atol=1e-15))
        if not equal: raise ValueError('Unexpected EEGLAB reader calibration: '+r['subject_id'])
        channels=[]
        for i,name in zip(indices,names):
            c=e['chanlocs'][i]
            channels.append({'python_index':i,'native_label':c['labels'],'code_target_label':name,
                'native_xyz':[float(c[k]) for k in ('X','Y','Z')], 'anatomical_equivalence_verified':False})
        rows.append({'subject_id':r['subject_id'],'set_sha256':digest(WORK/r['eeg_set']),
            'layout_source':str(e.get('chaninfo',{}).get('filename','')),'reference':str(e.get('ref','')),
            'reader_native_times_1e_minus6_verified':equal,'seconds':float(e['pnts']/e['srate']),
            'channels':channels})
    save(ROOT/'reports/eeg_units_channels_resolution.json',{'subjects':rows,'reader_check_passed':True,
        'interpretation':'MNE interprets EEGLAB native values as microvolts and returns volts. This verifies reader behavior, not acquisition calibration or Mumtaz checkpoint training scale.',
        'b0_policy':'Preserve upstream native numeric scale and fixed channel indices; do not add 1e-6 scaling or montage remapping.',
        'remaining_limitations':['Mumtaz checkpoint input scale not independently established','Code target electrode equivalence not independently established; original coordinates retained']})
    backbones=json.loads((ROOT/'manifests/backbones.json').read_text())
    protocol={
        'protocol_id':'B0-37-code-v1','cohort_id':lock['cohort_id'],'folds_sha256':lock['folds_sha256'],
        'speech_encoder':'Upstream CNN-BiGRU(input_dim=768), random frozen weights saved separately for seeds 11/22/33; same weights across subjects/folds within each seed. Preserve (N,768)->(N,1,768), then mean outputs. No MODMA fitting.',
        'hubert_identity':backbones['hubert_code']['repo']+'@'+backbones['hubert_code']['revision']+'; frozen last_hidden_state, time mean, 768 dimensions. Code Base identity retained; paper Large discrepancy disclosed.',
        'asr_segments':'WhisperX medium CPU float32 zh batch1; retain all segments and timestamps, B0 uses first segment. Empty outputs require review; tokenizer CLS max_length=512 truncation=True.',
        'audio_segmentation':'Exact upstream librosa 16k mono, peak normalization, trim20dB, median3; 5s windows stride2.5s, drop <5s and retain upstream tail duplication. WAV PCM16 roundtrip retained. No segments -> zero512, flagged in audit.',
        'eeg_units_channels':'Retain native .mat numeric values and upstream 19 indices, no extra scale/reference/montage changes. MNE EEGLAB reader native*1e-6 verified separately. Checkpoint scale and target anatomical equivalence remain unverified limitations; see EEG resolution report.',
        'eeg_length_policy':'Resample entire recording250->200Hz, bandpass0.3-75, notch50, then retain first60000 samples (300s), split60x5s, patch200 samples, average channel/patch embeddings ->60x200. Reject shorter input. Explicit local adaptation, not confirmed author policy.',
        'limitations':['37-person code-based baseline, not an exact reproduction of original38-person result','Random frozen encoder and degenerate attention retained','EEG native scale and index mapping follow code; physical/checkpoint compatibility remains uncertain'],
        'eeg_audit_sha256':digest(ROOT/'reports/eeg_units_channels_resolution.json'),
        'backbones_sha256':digest(ROOT/'manifests/backbones.json')}
    for path in (ROOT/'configs/protocol_resolution.json',ROOT/'features/B0-37/protocol_resolution.json'):
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=protocol:
            raise ValueError('Existing protocol differs; explicit version migration required')
        save(path,protocol)
    cfg['status']='protocol_locked_feature_extraction_pending'
    cfg['protocol_file']='configs/protocol_resolution.json'
    cfg['unresolved']=protocol['limitations']
    cfg['preprocessing']['eeg']['unit_and_montage_verification']='code_native_indices_locked_with_disclosed_limitations'
    cfg['preprocessing']['eeg']['length_policy']='full_filter_then_first_300_seconds'
    cfg['preprocessing']['speech']['short_audio_policy']='upstream_drop_with_zero_feature_audit'
    cfg['preprocessing']['text']['asr_segment_policy']='first_segment_code_all_segments_archived'
    save(ROOT/'configs/b0.json',cfg)
    print('Protocol locked; 37 EEGLAB reader/metadata checks passed. Physical-model limitations remain.')

if __name__=='__main__': main()

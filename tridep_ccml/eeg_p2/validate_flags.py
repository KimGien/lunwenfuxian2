"""Post-QC, label-free flag interpretation; does not revise QC rules or data."""
from __future__ import annotations
import mne
from scipy.io import loadmat
from .common import *
from .quality import spectral


def stages(x,names):
    raw=mne.io.RawArray(np.asarray(x,dtype=float),mne.create_info(names,250,'eeg'),verbose='ERROR')
    arrays={'native_250':raw.get_data()[:,:75000].copy()}
    raw.resample(200,verbose='ERROR')
    arrays['resampled_200']=raw.get_data()[:,:60000].copy()
    raw.filter(.3,75,verbose='ERROR')
    arrays['bandpass_0p3_75']=raw.get_data()[:,:60000].copy()
    raw.notch_filter(50,verbose='ERROR')
    arrays['notch_50']=raw.get_data()[:,:60000].copy()
    return arrays


def energy_stats(x,fs):
    # Channel time-median removed solely for stage comparison to avoid raw DC bias.
    y=x-np.median(x,axis=-1,keepdims=True)
    w=y.reshape(len(x),60,fs*5)
    e=np.sum(w*w,axis=(0,2))
    rms=np.sqrt(np.mean(w*w,axis=(0,2)))
    return {'first5_energy_fraction':float(e[0]/e.sum()),'last5_energy_fraction':float(e[-1]/e.sum()),
            'first5_RMS':float(rms[0]),'median_middle5_RMS':float(np.median(rms[1:-1])),
            'first5_to_middle_RMS_ratio':float(rms[0]/np.median(rms[1:-1]))}


def plot_synthetic(synthetic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(11,8),layout='constrained')
    y=synthetic['notch_50'][0]
    for ax,(start,title) in zip(axes,[(0,'First 5 seconds'),(120,'Interior: 120-125 seconds'),(295,'Last 5 seconds')]):
        ax.plot(start+np.arange(1000)/200,y[start*200:(start+5)*200],lw=.8)
        ax.set(xlabel='Time (s)',ylabel='Native value',title=title)
    fig.suptitle('Stationary 50-Hz + 10-Hz input after the unchanged R1 pipeline\nInput 10-Hz amplitude = 5; plot panels use different vertical scales',fontsize=12)
    fig.savefig(OUT/'figures/synthetic_filter_boundary.png',dpi=180)
    plt.close(fig)


def main():
    protocol={'registered_before_followup_calculations_utc':now(),'reason':'Inspect initial QC flags without changing thresholds or data.',
        'boundary':'All 37: first/last 5-second energy and flagged-window count; no data removed.',
        'line':'Reexpress flagged 49-51 Hz peaks as fraction of 1-75 Hz power; compare source 50-Hz PSD with cached filtered PSD.',
        'stage_cases':'largest first-window energy fraction, largest total RMS, median total RMS, smallest total RMS; deduplicated',
        'synthetic':'Fixed 300 seconds at 250 Hz: 500*sin(2*pi*50*t)+5*sin(2*pi*10*t); same existing pipeline only.',
        'unit_invariance':'Verify S0 joint input/fit-mean/fit-std scaling cancels a fixed positive multiplier.',
        'new_classifier_training':False,'QC_thresholds_changed':False,'treatment_selected':False}
    save(OUT/'followup_flag_validation_protocol.json',protocol)
    records=read_json(OUT/'input_manifest_label_free.json')
    subjects=[read_json(OUT/f'subjects/{r["subject_id"]}.json') for r in records]
    summaries=read_json(OUT/'subject_summary.json')
    selection=read_json(REVISION)['selection'];indices=[r['expected_zero_based_index'] for r in selection]
    boundary=[];lines=[];reject=[]
    for r,s in zip(subjects,summaries):
        with np.load(WORK/r['array_file']['path']) as arr:
            wrms=arr['filtered19_window_rms']
            e=(wrms**2).sum(0)
            boundary.append({'subject_id':r['subject_id'],'first5_energy_fraction':float(e[0]/e.sum()),
                'last5_energy_fraction':float(e[-1]/e.sum()),
                'flagged_first_or_last_windows':sum(i in [0,59] for i in s['high_rms_windows_any_channel']),
                'flagged_interior_windows':sum(i not in [0,59] for i in s['high_rms_windows_any_channel'])})
            f=arr['filtered19_frequency'];p=arr['filtered19_psd'];n=arr['native128_psd'][indices]
            fraction=p[:,(f>=49)&(f<=51)].sum(-1)/p[:,(f>=1)&(f<75)].sum(-1)
            i50=int(np.flatnonzero(f==50)[0])
            for c,ch in enumerate(r['filtered_19']['channels']):
                if ch['line_noise_review']:
                    lines.append({'subject_id':r['subject_id'],'channel':ch['channel'],
                        'line_band_fraction_of_1_75':float(fraction[c]),
                        'PSD50_after_before_dB':float(10*np.log10(p[c,i50]/max(n[c,i50],1e-30)))})
        # The audit preserves nonempty reject fields, including threshold settings.
        # This interpretation distinguishes settings from actual rejection masks.
        fields=r['processing_metadata']['reject_data_fields_nonempty']
        reject.append({'subject_id':r['subject_id'],'actual_rejection_mask_fields_nonempty':[k for k in fields if k.startswith(('rej','icarej'))],
            'nonempty_threshold_or_setting_fields':[k for k in fields if not k.startswith(('rej','icarej'))]})
    ordered=sorted(subjects,key=lambda r:r['filtered_19']['subject_rms'])
    cases=list(dict.fromkeys([max(boundary,key=lambda r:r['first5_energy_fraction'])['subject_id'],ordered[-1]['subject_id'],ordered[len(ordered)//2]['subject_id'],ordered[0]['subject_id']]))
    stage_results={}
    for sid in cases:
        r=next(r for r in records if r['subject_id']==sid)
        x=loadmat(WORK/r['mat']['path'])['eeg_mat'][indices]
        values=stages(x,[e['target_name'] for e in selection])
        stage_results[sid]={k:energy_stats(v,250 if k=='native_250' else 200) for k,v in values.items()}
    t=np.arange(75000)/250
    signal=(500*np.sin(2*np.pi*50*t)+5*np.sin(2*np.pi*10*t))[None,:]
    synthetic=stages(signal,['synthetic'])
    synth_stats={k:energy_stats(v,250 if k=='native_250' else 200) for k,v in synthetic.items()}
    from .scales import fit,METHODS
    stats={r['subject_id']:{'mean':[c['mean'] for c in r['filtered_19']['channels']],
        'std':[c['centered_rms'] for c in r['filtered_19']['channels']]} for r in subjects}
    folds=read_json(WORK/'S00/manifests/fold_assignments.json')
    errors=[]
    for fold in range(1,6):
        ids=folds[f'fold_{fold}']['train']
        a,b=fit(stats,ids,ids,METHODS[0])
        scaled={s:{k:(np.array(v)*1e-6).tolist() for k,v in r.items()} for s,r in stats.items()}
        aa,bb=fit(scaled,ids,ids,METHODS[0])
        r=next(r for r in records if r['subject_id']==ids[0])
        x=np.load(WORK/r['cache']['path']).transpose(1,0,2).reshape(19,-1).astype(float)
        z=(x-a[:,None])/b[:,None];zz=(x*1e-6-aa[:,None])/bb[:,None]
        np.testing.assert_allclose(z,zz,rtol=1e-12,atol=1e-12)
        errors.append(float(np.max(np.abs(z-zz))))
    save(OUT/'flag_validation.json',{'posthoc_label_free':True,'boundary':boundary,'flagged_line_channels':lines,
        'reject_metadata_interpretation':reject,'stage_cases':stage_results,'synthetic_steady_sinusoids':synth_stats,
        'standardizer_unit_invariance_max_error_by_fold':errors,'no_caches_modified':True})
    plot_synthetic(synthetic)
    print('Flag follow-up complete; no processing changes applied.',flush=True)


if __name__=='__main__':
    main()

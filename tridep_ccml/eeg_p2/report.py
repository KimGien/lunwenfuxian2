"""Descriptive reporting of frozen QC outputs; does not choose preprocessing."""
from __future__ import annotations
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from .common import *
from .scales import METHODS


def visualizations(rows):
    root=OUT/'figures'
    root.mkdir(exist_ok=True)
    names=[c['channel'] for c in rows[0]['filtered_19']['channels']]
    rms=np.array([[c['rms'] for c in r['filtered_19']['channels']] for r in rows])
    fig,ax=plt.subplots(figsize=(10,12),layout='constrained')
    im=ax.imshow(np.log10(rms),aspect='auto',cmap='magma')
    ax.set_xticks(range(19),names,rotation=45,ha='right')
    ax.set_yticks(range(len(rows)),[r['subject_id'] for r in rows],fontsize=8)
    ax.set_title('Filtered EEG channel RMS (native numerical scale)')
    fig.colorbar(im,ax=ax,label='log10 RMS')
    fig.savefig(root/'channel_rms_overview.png',dpi=170)
    plt.close(fig)
    total=np.array([r['filtered_19']['subject_rms']**2 for r in rows]);shares=total/total.sum()
    order=np.argsort(shares)
    fig,ax=plt.subplots(figsize=(9,10),layout='constrained')
    ax.barh(range(len(rows)),shares[order]*100,color='steelblue')
    ax.set_yticks(range(len(rows)),[rows[i]['subject_id'] for i in order],fontsize=8)
    ax.set(xlabel='Share of total squared signal (%)',title='All subjects retained; no diagnostic labels used')
    fig.savefig(root/'subject_energy_share.png',dpi=170)
    plt.close(fig)
    selection=read_json(REVISION)['selection']
    for r in rows:
        sid=r['subject_id']
        with np.load(WORK/r['array_file']['path']) as z:
            wrms=z['filtered19_window_rms'].copy();freq=z['filtered19_frequency'].copy();psd=z['filtered19_psd'].copy()
        channels=np.argsort([c['rms'] for c in r['filtered_19']['channels']])[-4:][::-1]
        window=int(np.argmax(np.mean(wrms**2,0)))
        x=np.load(WORK/r['input']['cache']['path']).transpose(1,0,2).reshape(19,-1)
        raw=loadmat(WORK/r['input']['mat']['path'])['eeg_mat']
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for c in channels:
            source=selection[c]['expected_zero_based_index']
            y=raw[source,window*1250:(window+1)*1250].astype(float)
            # Remove the displayed snippet median only; the underlying audit uses original values.
            axes[0,0].plot(window*5+np.arange(1250)/250,y-np.median(y),lw=.6,label=names[c])
            axes[0,1].plot(window*5+np.arange(1000)/200,x[c,window*1000:(window+1)*1000],lw=.6,label=names[c])
            mask=(freq>=1)&(freq<=75)
            axes[1,1].semilogy(freq[mask],psd[c,mask],lw=1,label=names[c])
        axes[0,0].set(title='A  Original: display-only median removal',xlabel='Time (s)',ylabel='Native value')
        axes[0,1].set(title='B  Existing R1 filtering, same interval',xlabel='Time (s)',ylabel='Native value')
        axes[0,0].legend(fontsize=8,ncol=2);axes[0,1].legend(fontsize=8,ncol=2)
        im=axes[1,0].imshow(np.log10(np.maximum(wrms,1e-12)),aspect='auto',extent=[0,300,18.5,-.5],cmap='magma')
        axes[1,0].set_yticks(range(19),names,fontsize=7)
        axes[1,0].set(title='C  All 60 windows; log10 RMS',xlabel='Time (s)')
        fig.colorbar(im,ax=axes[1,0],label='log10 RMS',shrink=.8)
        axes[1,1].set(title='D  Full 300-second filtered PSD',xlabel='Frequency (Hz)',ylabel='Native value squared / Hz')
        axes[1,1].legend(fontsize=8,ncol=2)
        fig.suptitle(f'{sid} | Largest-energy 5-second window: {window*5}-{(window+1)*5} s | Review flags only',fontsize=12)
        fig.savefig(root/f'{sid}_quality.png',dpi=145)
        plt.close(fig)
    scales=read_json(OUT/'scale_summary.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for j,m in enumerate(METHODS):
        axes[0].plot(range(1,6),[r['methods'][m]['largest_subject_energy_share']*100 for r in scales['folds']],marker='o',label=m[:2])
        axes[1].plot(range(1,6),[r['methods'][m]['subject_rms_max_min_ratio'] for r in scales['folds']],marker='o',label=m[:2])
    axes[0].set(xlabel='Outer fold (training subjects only)',ylabel='Largest subject energy share (%)',title='Concentration after fixed scale transforms')
    axes[1].set(xlabel='Outer fold (training subjects only)',ylabel='Subject RMS max / min',title='Scale spread; no classifier trained')
    axes[0].legend();axes[1].legend()
    fig.savefig(root/'train_scale_diagnostics.png',dpi=180)
    plt.close(fig)


def summarize(rows):
    summary=read_json(OUT/'subject_summary.json')
    rms=np.array([r['filtered_19']['subject_rms'] for r in rows])
    energy=rms*rms
    index=int(energy.argmax());largest=rows[index]
    names=[c['channel'] for c in rows[0]['filtered_19']['channels']]
    channel_energy=np.array([[c['rms']**2 for c in r['filtered_19']['channels']] for r in rows])
    top_ch=np.argsort(channel_energy[index])[::-1][:5]
    stats={'subjects':37,'rms_min':float(rms.min()),'rms_median':float(np.median(rms)),'rms_max':float(rms.max()),
        'rms_max_min':float(rms.max()/rms.min()),'largest_energy_subject':largest['subject_id'],
        'largest_energy_share':float(energy[index]/energy.sum()),
        'largest_subject_top_channels':[{'channel':names[i],'rms':float(np.sqrt(channel_energy[index,i])),
            'share_of_same_channel_cohort_energy':float(channel_energy[index,i]/channel_energy[:,i].sum()),
            'share_within_subject_energy':float(channel_energy[index,i]/channel_energy[index].sum())} for i in top_ch],
        'subjects_with_native_flat_review':sum(bool(r['native_flat_channels']) for r in summary),
        'subjects_with_native_plateau_review':sum(bool(r['native_extreme_plateau_channels']) for r in summary),
        'subjects_with_filtered_scale_review':sum(bool(r['channel_scale_review']) for r in summary),
        'subjects_with_filtered_line_review':sum(bool(r['filtered_line_noise_channels']) for r in summary),
        'subjects_with_filtered_high_frequency_review':sum(bool(r['filtered_high_frequency_channels']) for r in summary),
        'subjects_with_relative_high_rms_windows':sum(bool(r['high_rms_windows_any_channel']) for r in summary),
        'reference_rows_all_zero':sum(r['reference']['last_channel_all_zero'] for r in rows),
        'subjects_with_ICA_metadata':sum(r['processing_metadata']['ica_weights_nonempty'] for r in rows),
        'subjects_with_physical_calibration_fields':sum(bool(r['unit_reader_check']['calibration_fields']) for r in rows),
        'native_active_duplicate_pairs':sum(len(r['full_record_structure']['exact_duplicate_active_channels']) for r in rows),
        'filtered_high_rms_window_count':sum(len(r['high_rms_windows_any_channel']) for r in summary),
        'quality_flag_does_not_imply_bad_or_diagnostic':True}
    save(OUT/'quality_summary.json',stats)
    return stats


def main():
    require(read_json(OUT/'verification.json')['passed'],'Numerical audit must finish first')
    records=read_json(OUT/'input_manifest_label_free.json')
    rows=[read_json(OUT/f'subjects/{r["subject_id"]}.json') for r in records]
    stats=summarize(rows)
    visualizations(rows)
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()

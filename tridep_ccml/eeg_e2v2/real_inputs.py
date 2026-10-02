import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import welch
from tridep_ccml.v2.common import label_indices
from .common import *
from .preprocessing import preprocess, extend, chain, as_windows


def plot_subject(sid,old,new):
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False})
    fig,ax=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for x,color,label in [(old,'#8893a5','旧处理'),(new,'#168f83','新处理')]:
        rms=np.sqrt(np.mean(x.reshape(19,300,200)**2,axis=(0,2)))
        ax[0,0].plot(np.arange(300),rms,color=color,label=label)
        # RMS across channels keeps all electrodes visible without selecting a favorable one.
        ax[0,1].plot(np.arange(1000)/200,np.sqrt(np.mean(x[:,:1000]**2,axis=0)),color=color,label=label,lw=.7)
        ax[1,0].plot(np.arange(59000,60000)/200,np.sqrt(np.mean(x[:,59000:]**2,axis=0)),color=color,label=label,lw=.7)
        f,p=welch(x,fs=200,nperseg=1000,axis=-1)
        ax[1,1].semilogy(f,np.maximum(p.mean(axis=0),1e-20),color=color,label=label)
    for z,title in zip(ax.flat,['全300秒：每秒跨导 RMS','首5秒：跨导瞬时 RMS','末5秒：跨导瞬时 RMS','全段平均功率谱']):
        z.set_title(title);z.grid(alpha=.15);z.legend(fontsize=9)
    for z in [ax[0,0],ax[0,1],ax[1,0]]:z.set(xlabel='时间（秒）',ylabel='原生数值尺度')
    ax[1,1].set(xlim=(0,80),xlabel='频率（Hz）',ylabel='PSD（原生数值平方/Hz）')
    fig.suptitle(f'{sid}：全窗口保留的新旧处理对照；能量降低本身不代表真实信号更准确')
    path=REPORT/'figures'/f'{sid}.png';path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=120);plt.close(fig)


def run():
    gate=read(OUT/'synthetic_gate.json');assert gate['passed']
    method=gate['accepted_method']
    reg=read(OUT/'engineering_registration.json')
    assert all(digest(WORK/p)==sha for p,sha in reg['protocol']['sources'].items())
    save(OUT/'real_registration.json',{'synthetic_gate':desc(OUT/'synthetic_gate.json'),
        'source':desc(__file__),'appendix':desc(APPENDIX),'method':method,
        'pseudo_boundary_seconds':[60,120],'required_paired_median_improvement_both_sides':True,
        'maximum_external_peak_to_observed_peak':20,'labels_used':False},immutable=True)
    records=read(P2/'input_manifest_label_free.json')
    revision=read(REVISION)['selection'];required=[r['required_source_label'] for r in revision]
    expected=[r['expected_zero_based_index'] for r in revision]
    rows=[]
    for rec in records:
        sid=rec['subject_id'];target=OUT/'subjects'/f'{sid}.json'
        if target.exists():
            row=read(target);assert digest(WORK/row['new_cache']['path'])==row['new_cache']['sha256'];rows.append(row);continue
        for key in ['set','fdt','mat','cache']:assert digest(WORK/rec[key]['path'])==rec[key]['sha256']
        obj=loadmat(WORK/rec['set']['path'],simplify_cells=True);obj=obj.get('EEG',obj)
        idx=label_indices([str(c['labels']) for c in obj['chanlocs']],required);assert idx==expected
        raw=loadmat(WORK/rec['mat']['path'])['eeg_mat'][idx].astype(np.float64)
        assert raw.shape[0]==19 and raw.shape[1]>=75000 and np.isfinite(raw).all()
        oldfull=chain(raw);old=oldfull[:,:60000]
        oldwindows=np.load(WORK/rec['cache']['path'],allow_pickle=False)
        assert np.array_equal(as_windows(old),oldwindows),'Old cache mismatch '+sid
        extended,meta=extend(raw,method)
        peak_ratios=[]
        for observed,outside in [(raw[:,:2500],extended[:,:7500]),(raw[:,-2500:],extended[:,-7500:])]:
            peak_ratios.extend((np.max(abs(outside),axis=1)/np.maximum(np.max(abs(observed),axis=1),1e-12)).tolist())
        assert max(peak_ratios)<=20,'External extrapolation explosion '+sid
        new=chain(extended)[:,6000:66000]
        windows=as_windows(new)
        cache=OUT/'cache/raw'/f'{sid}.npy';cache.parent.mkdir(parents=True,exist_ok=True)
        assert not cache.exists(),'Incomplete cache already exists '+sid
        np.save(cache,windows,allow_pickle=False)
        short=raw[:,15000:30000];reference=oldfull[:,12000:24000]
        pseudo_old,_=preprocess(short,'old',60);pseudo_new,_=preprocess(short,method,60)
        denom=np.maximum(np.sqrt(np.mean(reference**2,axis=1)),1e-12)
        pseudo={}
        for name,y in [('old',pseudo_old),('new',pseudo_new)]:
            pseudo[name]={s:(np.sqrt(np.mean((y[:,sl]-reference[:,sl])**2,axis=1))/denom).tolist()
                          for s,sl in [('first5',slice(0,1000)),('last5',slice(-1000,None))]}
        quality={}
        for name,x in [('old',old),('new',new)]:
            energies=np.mean(x.reshape(19,60,1000)**2,axis=(0,2))
            quality[name]={'global_rms':float(np.sqrt(np.mean(x*x))),
                           'first_window_energy_share':float(energies[0]/energies.sum()),
                           'last_window_energy_share':float(energies[-1]/energies.sum()),
                           'window_energy':energies.tolist()}
        row={'subject_id':sid,'source':rec,'method':method,'duration_seconds':raw.shape[1]/250,
             'indices':idx,'required_labels':required,'new_cache':desc(cache),'shape':list(windows.shape),
             'old_cache_exact':True,'recorded_samples_unchanged':bool(np.array_equal(extended[:,7500:-7500],raw)),
             'augmentation':False,'removed_windows':0,'mode_metadata':meta,'external_peak_ratios':peak_ratios,
             'pseudo_boundary':pseudo,'quality':quality,'labels_used':False}
        save(target,row,immutable=True);rows.append(row);plot_subject(sid,old,new)
        print('INPUT',sid,len(rows),'/37','harmonic_sides',sum(r['harmonic_accepted'] for r in meta),flush=True)
    summary={}
    for side in ['first5','last5']:
        old=np.array([np.median(r['pseudo_boundary']['old'][side]) for r in rows])
        new=np.array([np.median(r['pseudo_boundary']['new'][side]) for r in rows])
        summary[side]={'old_subject_median':float(np.median(old)),'new_subject_median':float(np.median(new)),
            'median_paired_difference':float(np.median(new-old)),'subjects_improved':int(np.sum(new<old)),
            'passed':bool(np.median(new)<np.median(old))}
    passed=len(rows)==37 and all(v['passed'] for v in summary.values())
    save(OUT/'input_manifest.json',{'records':rows,'method':method,'labels_used':False,'augmentation':False,
        'revision':desc(REVISION),'engineering_gate':desc(OUT/'synthetic_gate.json')},immutable=True)
    save(OUT/'real_gate.json',{'passed':passed,'subjects':len(rows),'pseudo_boundary_summary':summary,
        'new_input_manifest':desc(OUT/'input_manifest.json'),'fresh_subjects':False,
        'limitation':'Artificial internal cuts provide observed context; true recording boundary context remains unknown.'},immutable=True)
    print('REAL_GATE',passed,summary,flush=True)


if __name__=='__main__':run()

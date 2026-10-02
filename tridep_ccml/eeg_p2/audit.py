"""Read-only original signal audit and exact frozen preprocessing replay."""
from __future__ import annotations
import warnings
from scipy.io import loadmat
import mne
from .common import *
from .quality import structure,measures


def nonempty(value):
    if isinstance(value,dict):
        return any(nonempty(v) for v in value.values())
    return np.asarray(value).size>0 if value is not None else False


def scalar_text(value):
    if value is None or not nonempty(value):
        return None
    return str(value)


def audit_one(record):
    sid=record['subject_id']
    target=OUT/f'subjects/{sid}.json'
    if target.exists():
        result=read_json(target)
        require(result['input']==record,'Stale subject audit')
        print('QC reuse '+sid,flush=True)
        return result
    for item in [record[k] for k in ['set','fdt','mat','cache']]:
        require(digest(WORK/item['path'])==item['sha256'],'Original source hash mismatch')
    obj=loadmat(WORK/record['set']['path'],simplify_cells=True)
    eeg=obj.get('EEG',obj)
    fs,n,c=int(eeg['srate']),int(eeg['pnts']),int(eeg['nbchan'])
    require(fs==250 and c==129 and int(eeg['trials'])==1 and n>=75000,'Unexpected raw dimensions')
    names=[r['labels'] for r in eeg['chanlocs']]
    require(names==[f'E{i}' for i in range(1,129)]+['Cz'],'Unexpected raw channel order')
    x=np.memmap(WORK/record['fdt']['path'],mode='r',dtype='<f4',shape=(n,c)).T
    mat=loadmat(WORK/record['mat']['path'])['eeg_mat']
    require(np.array_equal(x,mat),'FDT/MAT numerical mismatch')
    layout=structure(x,names,fs)
    active=np.asarray(x[:128,:75000],dtype=float)
    native=measures(active,names[:128],fs)
    selection=read_json(REVISION)['selection']
    indices=[names.index(r['required_source_label']) for r in selection]
    selected_names=[r['target_name'] for r in selection]
    require(indices==[r['expected_zero_based_index'] for r in selection],'Verified electrode mapping mismatch')
    from tridep_ccml.v2.extract_eeg import patches
    replay=patches(mat,indices,selected_names).astype(np.float32).reshape(60,19,1000)
    cached=np.load(WORK/record['cache']['path'],allow_pickle=False)
    require(np.array_equal(replay,cached),'R1 raw preprocessing replay was not bitwise exact')
    selected=cached.transpose(1,0,2).reshape(19,-1).astype(float)
    filtered=measures(selected,selected_names,200)
    with warnings.catch_warnings(record=True) as seen:
        reader=mne.io.read_raw_eeglab(WORK/record['set']['path'],preload=False,verbose='ERROR')
        errors=[]
        for start in [0,(n-1000)//2,n-1000]:
            actual=reader.get_data(start=start,stop=start+1000)
            expected=np.asarray(x[:,start:start+1000],dtype=float)*1e-6
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
            errors.append(float(np.max(np.abs(actual-expected))))
    # Calibration fields are inspected, not assumed from the reader convention.
    calibration={k:scalar_text(eeg[k]) for k in ['units','unit','calibration','gain','resolution'] if k in eeg}
    reject=eeg.get('reject',{})
    rejected_fields=[k for k,v in reject.items() if not k.endswith(('col','color')) and nonempty(v)] if isinstance(reject,dict) else []
    array_keys=['window_rms','window_ptp','window_log_rms_z','frequency','psd','low_fraction','high_fraction','line_ratio']
    arrays={f'{prefix}_{k}':v[k] for prefix,v in [('native128',native),('filtered19',filtered)] for k in array_keys}
    array_path=OUT/f'arrays/{sid}.npz'
    array_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(array_path,**arrays)
    result={'subject_id':sid,'input':record,'labels_used':False,'duration_seconds':n/fs,
        'samples':n,'sampling_rate':fs,'channels':c,'FDT_MAT_exact':True,'R1_raw_cache_exact':True,
        'full_record_structure':layout,'unit_reader_check':{'native_to_SI_multiplier':1e-6,
            'fixed_blocks':3,'max_absolute_error':max(errors),'physical_calibration_confirmed':False,
            'calibration_fields':calibration},
        'reference':{'metadata':scalar_text(eeg.get('ref')),'last_channel':names[-1],
            'last_channel_type':scalar_text(eeg['chanlocs'][-1].get('type')),
            'last_channel_all_zero':bool(np.all(x[-1]==0)),
            'mean_active_128_rms':float(np.sqrt(np.mean(active.mean(0)**2))),
            'no_rereference_added':True},
        'processing_metadata':{'ica_weights_nonempty':nonempty(eeg.get('icaweights')),
            'ica_sphere_nonempty':nonempty(eeg.get('icasphere')),'reject_data_fields_nonempty':rejected_fields,
            'event_count':int(np.asarray(eeg.get('event',[])).size),
            'history':scalar_text(eeg.get('history')),'reader_bads':list(reader.info['bads']),
            'reader_warnings':[str(w.message) for w in seen]},
        'native_active128':{k:v for k,v in native.items() if k not in array_keys},
        'filtered_19':{k:v for k,v in filtered.items() if k not in array_keys},
        'array_file':desc(array_path),'rules':RULES}
    save(target,result)
    print(f'QC {sid}: RMS={filtered["subject_rms"]:.3f}, raw/cache replay exact',flush=True)
    return result


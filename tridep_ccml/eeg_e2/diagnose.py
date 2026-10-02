"""Label-free E0 mechanism probes. These are not classifier experiments."""
from __future__ import annotations
import inspect
import mne
from scipy.signal import freqz
from .common import *

def raw(x):
    x = np.atleast_2d(x).astype(np.float64)
    return mne.io.RawArray(x, mne.create_info([f'ch{i}' for i in range(len(x))],250,'eeg'),verbose='ERROR')

def process(x, order='old', pad='reflect_limited'):
    r=raw(x)
    result={}
    if order=='notch_native_first':
        r.notch_filter(50,pad=pad,verbose='ERROR')
        result['notch_native']=r.get_data().copy()
    r.resample(200,verbose='ERROR')
    result['resampled']=r.get_data().copy()
    if order=='notch_first':
        r.notch_filter(50,pad=pad,verbose='ERROR')
        result['notch']=r.get_data().copy()
    r.filter(.3,75,pad=pad,verbose='ERROR')
    result['bandpass']=r.get_data().copy()
    if order=='old':
        r.notch_filter(50,pad=pad,verbose='ERROR')
        result['notch']=r.get_data().copy()
    result['final']=r.get_data().copy()
    return result

def probe_metrics(y,reference,fs=200):
    n=min(y.shape[-1],reference.shape[-1],300*fs)
    d=(y[...,:n]-reference[...,:n]).reshape(-1,n)
    ref=reference[...,:n]
    denom=float(np.sqrt(np.mean(ref*ref)))
    return {label:float(np.sqrt(np.mean(d[...,sl]**2))/max(denom,1e-12))
            for label,sl in [('first5',slice(0,5*fs)),('last5',slice(n-5*fs,n)),('interior',slice(30*fs,n-30*fs))]}

def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    mne.set_log_level('ERROR')
    save(OUT/'status.json',{'state':'E0_label_free_diagnosis','updated_utc':now(),'new_models':0})
    bp=mne.filter.create_filter(None,200,.3,75,verbose='ERROR')
    nt=mne.filter.create_filter(None,200,[50.625],[49.375],l_trans_bandwidth=.5,h_trans_bandwidth=.5,verbose='ERROR')
    summary={'mne_version':mne.__version__,'plan':desc(PLAN),'time_utc':now(),
        'bandpass_taps':len(bp),'notch_taps':len(nt),'bandpass_coefficients_sha256':hashlib.sha256(bp.tobytes()).hexdigest(),
        'notch_coefficients_sha256':hashlib.sha256(nt.tobytes()).hexdigest(),
        'resample_signature':str(inspect.signature(mne.io.BaseRaw.resample)),
        'filter_signature':str(inspect.signature(mne.io.BaseRaw.filter)),
        'notch_signature':str(inspect.signature(mne.io.BaseRaw.notch_filter)),
        'probes':[],'labels_used':False,'classifier_training':False}
    np.savez(OUT/'E0_original_filter_coefficients.npz',bandpass=bp,notch=nt)
    for phase in [0.,np.pi/2]:
        t=np.arange(75000)/250
        x=500*np.sin(2*np.pi*50*t+phase)+5*np.sin(2*np.pi*10*t+phase)
        tl=np.arange(-7500,82500)/250
        xl=500*np.sin(2*np.pi*50*tl+phase)+5*np.sin(2*np.pi*10*tl+phase)
        for order,pad in [('old','reflect_limited'),('notch_first','reflect_limited'),
                          ('notch_native_first','reflect_limited'),('old','reflect'),('old','symmetric')]:
            a=process(x,order,pad)
            b=process(xl,order,pad)
            errors={k:probe_metrics(v,b[k][...,int(30*(250 if k=='notch_native' else 200)):],250 if k=='notch_native' else 200) for k,v in a.items()}
            row={'phase':phase,'order':order,'pad':pad,'errors':errors}
            summary['probes'].append(row)
            print(json.dumps(row),flush=True)
    f,h=freqz(bp,worN=np.array([1,4,10,20,35,50]),fs=200)
    _,hn=freqz(nt,worN=f,fs=200)
    summary['original_response']={'frequency_hz':f.tolist(),'bandpass_gain':abs(h).tolist(),'notch_gain':abs(hn).tolist()}
    save(OUT/'E0_mechanism_probes.json',summary)

if __name__=='__main__':main()

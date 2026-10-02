"""Structural defects and scale-free flags are review candidates, not diagnoses."""
from __future__ import annotations
import hashlib
from scipy.signal import welch
from .common import *


def robust_scale(x,axis=-1):
    center=np.median(x,axis=axis,keepdims=True)
    return np.squeeze(center,axis=axis),1.4826*np.median(np.abs(x-center),axis=axis)


def max_true_run(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,dtype=bool),False].astype(np.int8))
    lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    return int(lengths.max()) if len(lengths) else 0


def structure(x,names,fs):
    x=np.asarray(x)
    require(x.ndim==2 and len(names)==len(x),'Signal/channel axes mismatch')
    finite=np.isfinite(x)
    require(finite.all(),'Nonfinite original signal; stop numeric audit')
    rows=[]
    hashes={}
    duplicates=[]
    for name,y in zip(names,x):
        is_ref=name=='Cz'
        run=max_true_run(np.diff(y)==0)+1
        rail=max(max_true_run(y==y.min()),max_true_run(y==y.max()))
        sig=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
        if not is_ref:
            if sig in hashes:
                duplicates.append([hashes[sig],name])
            else:
                hashes[sig]=name
        rows.append({'channel':name,'reference_row':is_ref,'all_zero':bool(np.all(y==0)),
            'constant':bool(np.ptp(y)==0),'longest_exact_flat_seconds':run/fs,
            'flat_review':not is_ref and run>=fs*RULES['flat_seconds'],
            'longest_extreme_plateau_seconds':rail/fs,
            'extreme_plateau_review':not is_ref and rail>=fs*RULES['rail_plateau_seconds'],
            'minimum':float(y.min()),'maximum':float(y.max())})
    return {'channels':rows,'exact_duplicate_active_channels':duplicates,'all_samples_finite':True}


def spectral(x,fs):
    f,p=welch(np.asarray(x,dtype=float),fs=fs,nperseg=2*fs,noverlap=fs,
              detrend='constant',window='hann',axis=-1,scaling='density')
    def power(a,b):
        return p[:,(f>=a)&(f<b)].sum(-1)
    floor=RULES['scale_floor']
    low=power(1,4)/np.maximum(power(1,40),floor)
    high=power(35,75)/np.maximum(power(1,75),floor)
    line=p[:,(f>=49)&(f<=51)].mean(-1)/np.maximum(p[:,((f>=45)&(f<49))|((f>51)&(f<=55))].mean(-1),floor)
    return {'frequency':f,'psd':p,'low_fraction':low,'high_fraction':high,'line_ratio':line}


def measures(x,names,fs):
    x=np.asarray(x,dtype=float)
    require(x.ndim==2 and np.isfinite(x).all(),'Nonfinite/malformed waveform')
    require(x.shape[-1]==300*fs,'QC requires the fixed 300 seconds')
    center,scale=robust_scale(x)
    mean=x.mean(-1)
    rms=np.sqrt(np.mean(x*x,axis=-1))
    std=x.std(-1)
    w=x.reshape(len(names),60,5*fs)
    wrms=np.sqrt(np.mean(w*w,axis=-1))
    wptp=np.ptp(w,axis=-1)
    log_rms=np.log(np.maximum(wrms,RULES['scale_floor']))
    logcenter,logscale=robust_scale(log_rms)
    wz=(log_rms-logcenter[:,None])/np.maximum(logscale[:,None],RULES['scale_floor'])
    active_scale=scale[scale>RULES['scale_floor']]
    middle=float(np.median(active_scale)) if len(active_scale) else RULES['scale_floor']
    ratios=scale/middle
    sample_fraction=(np.abs(x-center[:,None])>RULES['sample_robust_z']*np.maximum(scale[:,None],RULES['scale_floor'])).mean(-1)
    diff=np.diff(x,axis=-1)
    diffcenter,diffscale=robust_scale(diff)
    jump_fraction=(np.abs(diff-diffcenter[:,None])>RULES['difference_robust_z']*np.maximum(diffscale[:,None],RULES['scale_floor'])).mean(-1)
    spec=spectral(x,fs)
    rows=[]
    for c,name in enumerate(names):
        rows.append({'channel':name,'mean':float(mean[c]),'median':float(center[c]),
            'rms':float(rms[c]),'centered_rms':float(std[c]),'sigma_mad':float(scale[c]),
            'p99_abs_centered':float(np.quantile(np.abs(x[c]-center[c]),.99)),
            'scale_relative_to_subject_median_channel':float(ratios[c]),
            'relative_scale_review':bool(ratios[c]>5 or ratios[c]<.1),
            'transient_fraction_over_8mad':float(sample_fraction[c]),
            'jump_fraction_over_20diffmad':float(jump_fraction[c]),
            'high_rms_window_indices_0based':np.flatnonzero(wz[c]>6).tolist(),
            'low_band_fraction_1_4_of_1_40':float(spec['low_fraction'][c]),
            'high_band_fraction_35_75_of_1_75':float(spec['high_fraction'][c]),
            'high_frequency_review':bool(spec['high_fraction'][c]>.5),
            'line_noise_neighbor_ratio':float(spec['line_ratio'][c]),
            'line_noise_review':bool(spec['line_ratio'][c]>10)})
    return {'channels':rows,'subject_rms':float(np.sqrt(np.mean(x*x))),
            'subject_centered_rms':float(np.sqrt(np.mean(std**2))),
            'max_channel_energy_share':float(np.max(rms*rms)/np.maximum(np.sum(rms*rms),1e-12)),
            'window_rms':wrms,'window_ptp':wptp,'window_log_rms_z':wz,**spec}

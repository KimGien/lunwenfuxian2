"""Bounded harmonic continuation outside recorded data; original FIR chain preserved."""
import numpy as np
from tridep_ccml.eeg_e2.preprocessing import _fit_edge, chain, as_windows
from .probe import residual_modes

FS=250
PAD=7500
EDGE=2500


def design(t, frequencies):
    return np.stack([v for f in frequencies for v in [np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t)]],axis=1) if frequencies else np.zeros((len(t),0))


def extend(x, method='harmonic_guarded'):
    assert method in ['harmonic_guarded','harmonic_conservative']
    x=np.atleast_2d(np.asarray(x,dtype=np.float64))
    assert x.shape[1]>=EDGE and np.isfinite(x).all()
    sides=[];metadata=[]
    for side in ['left','right']:
        edge=x[:,:EDGE] if side=='left' else x[:,-EDGE:]
        t=np.arange(EDGE)/FS
        te=np.arange(-PAD,0)/FS if side=='left' else np.arange(EDGE,EDGE+PAD)/FS
        rows=[]
        for channel,y in enumerate(edge):
            lf,lc,residual=_fit_edge(y,True)
            frequencies,info=residual_modes(residual)
            a=design(t,frequencies)
            b=np.c_[a,np.ones(EDGE),t-t.mean()]
            c=np.linalg.lstsq(b,residual,rcond=1e-6)[0][:a.shape[1]]
            fitted=a@c
            pred=design(te,frequencies)@c
            centered_scale=max(float(np.std(residual)),1e-12)
            fit_ratio=float(np.std(residual-fitted)/centered_scale)
            amplitude_bound=float(sum(np.hypot(c[::2],c[1::2])))
            # Predetermined safety fallback within the same algorithm, without labels.
            threshold=.1 if method=='harmonic_guarded' else .03
            accepted=bool(frequencies and fit_ratio<=threshold and amplitude_bound<=10*centered_scale)
            if not accepted:
                fitted=np.zeros_like(residual);pred=np.zeros(PAD)
            remaining=residual-fitted
            padded=np.pad(remaining,(PAD,PAD),mode='symmetric')
            pred+=padded[:PAD] if side=='left' else padded[-PAD:]
            pred+=design(te,[lf])@lc
            rows.append(pred)
            metadata.append({'side':side,'channel_index':channel,'line_frequency':lf,
                'line_amplitude':float(np.linalg.norm(lc)),'frequencies':frequencies,
                'mode_rank':info['rank'],'fit_ratio':fit_ratio,'harmonic_accepted':accepted,
                'amplitude_bound':amplitude_bound,'residual_scale':centered_scale})
        sides.append(np.stack(rows))
    padded=np.concatenate([sides[0],x,sides[1]],axis=1)
    assert np.array_equal(padded[:,PAD:-PAD],x) and np.isfinite(padded).all()
    return padded,metadata


def preprocess(x,method='harmonic_guarded',seconds=300):
    if method=='old': return chain(x)[:,:int(seconds*200)],[]
    padded,meta=extend(x,method)
    result=chain(padded)[:,6000:6000+int(seconds*200)]
    assert result.shape[1]==int(seconds*200) and np.isfinite(result).all()
    return result,meta

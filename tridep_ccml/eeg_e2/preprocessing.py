"""Boundary continuation only: original recorded samples and FIR settings preserved."""
from __future__ import annotations
import numpy as np
import mne
from scipy.optimize import minimize_scalar

FS_NATIVE=250
PAD=7500
EDGE=2500

def _design(t,f):
    return np.c_[np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t),np.ones(len(t)),t-t.mean()]

def _fit_edge(y,adaptive):
    t=np.arange(len(y))/FS_NATIVE
    def fit(f):
        design=_design(t,f)
        c=np.linalg.lstsq(design,y,rcond=None)[0]
        residual=y-design@c
        return float(residual@residual),c
    if adaptive:
        grid=np.linspace(49.,51.,21)
        errors=[fit(f)[0] for f in grid]
        i=int(np.argmin(errors))
        a,b=max(49.,float(grid[i]-.1)),min(51.,float(grid[i]+.1))
        result=minimize_scalar(lambda f:fit(f)[0],bounds=(a,b),method='bounded',options={'xatol':1e-9,'maxiter':100})
        assert result.success
        f=float(result.x)
    else:
        f=50.
    _,c=fit(f)
    model=_design(t,f)[:,:2]@c[:2]
    residual=y-model
    return f,c[:2],residual

def extend(x,method):
    x=np.atleast_2d(np.asarray(x,dtype=np.float64))
    assert method in ['fixed50','adaptive_line'] and x.shape[-1]>=EDGE
    assert np.isfinite(x).all()
    sides=[];metadata=[]
    for side in ['left','right']:
        edge=x[:,:EDGE] if side=='left' else x[:,-EDGE:]
        ext_t=np.arange(-PAD,0)/FS_NATIVE if side=='left' else np.arange(EDGE,EDGE+PAD)/FS_NATIVE
        rows=[]
        for channel,y in enumerate(edge):
            f,c,residual=_fit_edge(y,method=='adaptive_line')
            padded=np.pad(residual,(PAD,PAD),mode='symmetric')
            mirror=padded[:PAD] if side=='left' else padded[-PAD:]
            line=np.c_[np.sin(2*np.pi*f*ext_t),np.cos(2*np.pi*f*ext_t)]@c
            rows.append(mirror+line)
            metadata.append({'side':side,'channel_index':channel,'frequency_hz':f,
                             'fitted_line_amplitude_native':float(np.linalg.norm(c)),
                             'edge_rms_native':float(np.sqrt(np.mean(y*y)))})
        sides.append(np.stack(rows))
    output=np.concatenate([sides[0],x,sides[1]],axis=-1)
    assert np.array_equal(output[:,PAD:-PAD],x) and np.isfinite(output).all()
    return output,metadata

def chain(x):
    x=np.atleast_2d(np.asarray(x,dtype=np.float64))
    r=mne.io.RawArray(x,mne.create_info([f'ch{i}' for i in range(len(x))],FS_NATIVE,'eeg'),verbose='ERROR')
    r.resample(200,verbose='ERROR')
    r.filter(.3,75,verbose='ERROR')
    r.notch_filter(50,verbose='ERROR')
    return r.get_data()

def preprocess(x,method):
    if method=='old':
        return chain(x)[:,:60000],[]
    padded,meta=extend(x,method)
    y=chain(padded)
    return y[:,6000:66000],meta

def as_windows(x):
    assert x.ndim==2 and x.shape==(19,60000) and np.isfinite(x).all()
    return x.reshape(19,60,1000).transpose(1,0,2).astype(np.float32)

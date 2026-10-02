"""Small development probe on the known v1 failure, before v2 registration."""
import numpy as np
from scipy.linalg import svd, eigvals
from tridep_ccml.eeg_e2.preprocessing import _fit_edge, chain
from tridep_ccml.eeg_e2.engineering import cases, signal, ref, errors
from .common import OUT, save


def residual_modes(y):
    center = float(np.mean(y))
    z = y-center
    w = np.lib.stride_tricks.sliding_window_view(z, 126).T
    u,s,vh = svd(w[:-1], full_matrices=False, check_finite=False, lapack_driver='gesdd')
    rank = min(20, int(np.sum(s > s[0]*1e-3))) if s[0] > 1e-10 else 0
    if rank < 2: return [], {'rank':rank}
    a = ((u[:,:rank].T @ w[1:]) @ vh[:rank].T) / s[:rank][None,:]
    roots = eigvals(a)
    modes = sorted([(float(np.angle(r)*250/(2*np.pi)),float(abs(r))) for r in roots
                    if .98 <= abs(r) <= 1.02 and .3 <= np.angle(r)*250/(2*np.pi) <= 75.])
    fs=[]
    for f,r in modes:
        if not fs or f-fs[-1] >= .2: fs.append(f)
    return fs, {'rank':rank,'modes':modes}


def extend(x):
    x=np.atleast_2d(x).astype(float); pad=7500;edge=2500
    sides=[];meta=[]
    for side in ['left','right']:
        ys=x[:,:edge] if side=='left' else x[:,-edge:]
        t=np.arange(edge)/250
        te=np.arange(-pad,0)/250 if side=='left' else np.arange(edge,edge+pad)/250
        values=[]
        for y in ys:
            line_f, line_c, residual = _fit_edge(y, True)
            frequencies, info=residual_modes(residual)
            def design(tt,fs):
                return np.stack([a for f in fs for a in [np.sin(2*np.pi*f*tt),np.cos(2*np.pi*f*tt)]],axis=1) if fs else np.zeros((len(tt),0))
            a=design(t,frequencies)
            b=np.c_[a,np.ones(len(t)),t-t.mean()]
            c=np.linalg.lstsq(b,residual,rcond=1e-6)[0][:a.shape[1]]
            modeled=a@c
            prediction=design(te,frequencies)@c
            left=residual-modeled
            symmetric=np.pad(left,(pad,pad),mode='symmetric')
            prediction+=symmetric[:pad] if side=='left' else symmetric[-pad:]
            prediction+=np.c_[np.sin(2*np.pi*line_f*te),np.cos(2*np.pi*line_f*te)]@line_c
            values.append(prediction)
            meta.append({'side':side,'frequency_hz':frequencies,'line_hz':line_f,**info,
                         'residual_fit_ratio':float(np.linalg.norm(left)/max(np.linalg.norm(residual),1e-12))})
        sides.append(np.stack(values))
    padded=np.concatenate([sides[0],x,sides[1]],axis=1)
    assert np.array_equal(padded[:,pad:-pad],x)
    return padded,meta


def run():
    rows=[]
    for i in [104,105,4]:
        cfg=cases()[i];x=signal(cfg,np.arange(cfg['n'])/250)
        r=ref(cfg,60);denom=np.sqrt(np.mean(ref(cfg,60,False)**2))
        p,meta=extend(x);y=chain(p)[:,6000:66000]
        row={'case':i,'errors':errors(y,r,denom),'meta':meta};rows.append(row)
        print(row,flush=True)
    save(OUT/'development_probe.json',{'scope':'known development cases; not fresh validation','rows':rows})


if __name__=='__main__':run()

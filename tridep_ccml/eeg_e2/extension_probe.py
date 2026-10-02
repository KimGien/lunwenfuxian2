"""E0 input-only mechanism exploration; no EEG diagnoses/classifier scores."""
import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter
from .diagnose import process,probe_metrics
from .common import *

def line_padding(x,pad_samples=7500):
    x=np.atleast_2d(x).astype(float);n=x.shape[-1]
    output=np.pad(x,((0,0),(pad_samples,pad_samples)),mode='symmetric')
    # Fit only the fixed power-line frequency at each observed boundary.
    size=min(2500,n)
    for side in ['left','right']:
        idx=np.arange(size) if side=='left' else np.arange(n-size,n)
        ext=np.arange(-pad_samples,0) if side=='left' else np.arange(n,n+pad_samples)
        a=np.c_[np.sin(2*np.pi*50*idx/250),np.cos(2*np.pi*50*idx/250),np.ones(size)]
        coeff=np.linalg.lstsq(a,x[:,idx].T,rcond=None)[0]
        observed=(a[:,:2]@coeff[:2]).T
        residual=x[:,idx]-observed
        mirror=np.pad(residual,((0,0),(pad_samples,pad_samples)),mode='symmetric')
        e=np.c_[np.sin(2*np.pi*50*ext/250),np.cos(2*np.pi*50*ext/250)]
        pred=(e@coeff[:2]).T
        if side=='left':output[:,:pad_samples]=mirror[:,:pad_samples]+pred
        else:output[:,-pad_samples:]=mirror[:,-pad_samples:]+pred
    return output

def ar_extension(x,order=100,size=2500,pad_samples=7500):
    """Least-squares local linear recurrence, stable root projection, external only."""
    x=np.atleast_2d(x).astype(float)
    pieces=[]
    info=[]
    for side in ['left','right']:
        series=x[:,:size][:,::-1] if side=='left' else x[:,-size:]
        preds=[]
        for row in series:
            center=row.mean();scale=row.std()
            if scale<1e-12:
                pred=np.full(pad_samples,center)
                info.append({'constant':True})
            else:
                y=(row-center)/scale
                windows=np.lib.stride_tricks.sliding_window_view(y,order+1)
                a=np.linalg.lstsq(windows[:,:-1][:,::-1],windows[:,-1],rcond=1e-8)[0]
                roots=np.roots(np.r_[1.,-a])
                roots=np.where(np.abs(roots)>1,roots/np.maximum(np.abs(roots),1e-30),roots)
                a=-np.poly(roots).real[1:]
                # Extrapolated values never replace recorded samples.
                history=list(y)
                for _ in range(pad_samples):
                    value=float(np.dot(a,np.array(history[-order:][::-1])))
                    history.append(float(np.clip(value,-20,20)))
                pred=np.array(history[-pad_samples:])*scale+center
                info.append({'constant':False,'peak_relative_scale':float(np.max(np.abs((pred-center)/scale)))})
            preds.append(pred)
        piece=np.array(preds)
        pieces.append(piece[:,::-1] if side=='left' else piece)
    return np.concatenate([pieces[0],x,pieces[1]],axis=-1),info

def main():
    results=[]
    for freq,phase,line_amp in [(10,0,500),(10,np.pi/2,500),(1,0,0),(1,np.pi/2,0),(4,.4,0)]:
        t=np.arange(75000)/250
        x=5*np.sin(2*np.pi*freq*t+phase)+line_amp*np.sin(2*np.pi*50*t+phase)
        tl=np.arange(-15000,90000)/250
        reference=process(5*np.sin(2*np.pi*freq*tl+phase)+line_amp*np.sin(2*np.pi*50*tl+phase))['final'][:,12000:72000]
        for mode in ['line_symmetric','stable_ar']:
            padded=line_padding(x) if mode=='line_symmetric' else ar_extension(x)[0]
            out=process(padded)['final'][:,6000:66000]
            row={'frequency':freq,'phase':phase,'line_amplitude':line_amp,'extension':mode,'errors':probe_metrics(out,reference)}
            results.append(row);print(json.dumps(row),flush=True)
    save(OUT/'E0_continuation_probes.json',{'label_free':True,'not_acceptance_or_classifier_selection':True,'results':results})

if __name__=='__main__':main()

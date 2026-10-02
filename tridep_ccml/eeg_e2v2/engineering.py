from scipy.signal import freqz
from tridep_ccml.eeg_e2.engineering import cases as old_cases, errors, fitted_amplitudes
from tridep_ccml.eeg_e2.preprocessing import chain
from .common import *
from .preprocessing import preprocess


def cases():
    rows=[{**r,'role':'development_regression'} for r in old_cases()]
    rng=np.random.default_rng(20260930)
    for i in range(24):
        fs=(np.array([.7,2.3,6.7,11.4,19.3,33.7])+rng.uniform(-.12,.12,6)).tolist()
        rows.append({'case':108+i,'role':'fresh_synthetic_validation','n':int(rng.integers(75089,77089)),
            'frequencies':fs,'amplitudes':rng.uniform(.3,1.3,6).tolist(),'phases':rng.uniform(-np.pi,np.pi,6).tolist(),
            'phase':float(rng.uniform(-np.pi,np.pi)),'line_frequency':[49.65,50.,50.35][i%3],
            'line_amplitude':[0.,5.,500.][(i//3)%3]})
    return rows


def signal(row,t,line=True):
    phases=row.get('phases',[row['phase']]*len(row['frequencies']))
    y=sum(a*np.sin(2*np.pi*f*t+p) for f,a,p in zip(row['frequencies'],row['amplitudes'],phases))
    if line:y+=row['line_amplitude']*np.sin(2*np.pi*row['line_frequency']*t+row['phase'])
    return y[None]


def ref(row,context,line=True):
    y=chain(signal(row,np.arange(-context*250,row['n']+context*250)/250,line))
    return y[:,context*200:context*200+60000]


def register():
    sources=[PLAN,APPENDIX,CODE/'preprocessing.py',CODE/'probe.py',CODE/'common.py',CODE/'engineering.py',
             WORK/'tridep_ccml/eeg_e2/preprocessing.py',WORK/'tridep_ccml/eeg_e2/engineering.py']
    protocol={'version':'EEG-E2v2-boundary','candidates':['harmonic_guarded','harmonic_conservative'],
        'cases':cases(),'sources':{p.relative_to(WORK).as_posix():digest(p) for p in sources},
        'labels_used':False,'thresholds':{'absolute':.1,'relative':.2,'good_edge_increase':.02,
        'gain':.05,'attenuation_db':40,'reference_convergence':1e-4},'augmentation':False}
    target=OUT/'engineering_registration.json'
    if target.exists():assert read(target)['protocol']==protocol
    else:
        save(OUT/'historical_before.json',history(),immutable=True)
        save(target,{'registered_utc':now(),'protocol':protocol},immutable=True)


def stationary(method):
    coeff=np.load(WORK/'tridep_ccml/experiments/EEG-E2-37-fulltext/E0_original_filter_coefficients.npz')
    rows=[]
    for cfg in cases():
        r=ref(cfg,60);r90=ref(cfg,90);retained=ref(cfg,60,False)
        denominator=float(np.sqrt(np.mean(retained**2)))
        convergence=float(np.sqrt(np.mean((r-r90)**2))/denominator)
        x=signal(cfg,np.arange(cfg['n'])/250)
        old,_=preprocess(x,'old');new,meta=preprocess(x,method)
        oe=errors(old,r,denominator);ne=errors(new,r,denominator)
        fs=list(cfg['frequencies'])+([cfg['line_frequency']] if cfg['line_amplitude'] else [])
        amp=fitted_amplitudes(new,fs)
        _,bp=freqz(coeff['bandpass'],worN=np.array(cfg['frequencies']),fs=200)
        _,nt=freqz(coeff['notch'],worN=np.array(cfg['frequencies']),fs=200)
        targets=np.array(cfg['amplitudes'])*abs(bp*nt)
        gainerr=float(np.max(abs(amp[:len(targets)]/targets-1)))
        attenuation=float(-20*np.log10(max(amp[-1]/cfg['line_amplitude'],1e-30))) if cfg['line_amplitude'] and cfg['line_frequency']==50 else None
        failures=[]
        if convergence>1e-4:failures.append('reference_convergence')
        for s in ['first5','last5']:
            if ne[s]>.1:failures.append(s+'_absolute')
            if oe[s]>.1 and ne[s]>.2*oe[s]:failures.append(s+'_relative')
            if oe[s]<=.1 and ne[s]>oe[s]+.02:failures.append(s+'_good_edge_degradation')
        if gainerr>.05:failures.append('passband_gain')
        if attenuation is not None and attenuation<40:failures.append('line50_attenuation')
        rows.append({**cfg,'old_nrmse':oe,'new_nrmse':ne,'reference_convergence':convergence,
            'gain_error':gainerr,'attenuation50_db':attenuation,'failures':failures,'passed':not failures,
            'mode_metadata':meta})
        if (cfg['case']+1)%12==0:print(method,cfg['case']+1,'/132',flush=True)
    return rows


def transient(method,new=False,center=150):
    t=np.arange(-22500,97500)/250
    rng=np.random.default_rng(20260930 if new else 20260929)
    wave=(3*np.sin(2*np.pi*7.3*t)+2*np.sin(2*np.pi*18.1*t)) if new else 5*np.sin(2*np.pi*10*t)
    x=(wave*np.exp(-.5*((t-center)/.2)**2)+.1*rng.standard_normal(len(t)))[None]
    reference=chain(x)[:,18000:78000]
    y,_=preprocess(x[:,22500:97500],method)
    nr=float(np.linalg.norm(y-reference)/np.linalg.norm(reference))
    peak=float(abs(np.max(abs(y))/np.max(abs(reference))-1))
    return {'fresh':new,'center':center,'whole_nrmse':nr,'peak_relative_error':peak,'passed':nr<=.01 and peak<=.05}


def run():
    register();accepted=None;reports=[]
    save(OUT/'status.json',{'state':'synthetic_engineering','new_models':0})
    for method in ['harmonic_guarded','harmonic_conservative']:
        path=OUT/f'engineering_{method}.json'
        if path.exists():d=read(path)
        else:
            rows=stationary(method)
            trans=[transient(method),transient(method,True)]
            d={'method':method,'registered':desc(OUT/'engineering_registration.json'),'rows':rows,
               'central_transients':trans,'edge_transient':transient(method,True,1),
               'passed':all(r['passed'] for r in rows) and all(r['passed'] for r in trans),'finished_utc':now()}
            save(path,d,immutable=True)
        reports.append({'report':desc(path),'passed':d['passed'],'method':method})
        print('CANDIDATE',method,d['passed'],'failed',[(r['case'],r['failures']) for r in d['rows'] if not r['passed']],d['central_transients'],flush=True)
        if d['passed']:accepted=method;break
    save(OUT/'synthetic_gate.json',{'passed':accepted is not None,'accepted_method':accepted,'candidates':reports},immutable=True)
    print('SYNTHETIC_GATE',accepted,flush=True)


if __name__=='__main__':run()

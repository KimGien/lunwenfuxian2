from __future__ import annotations
import argparse
import mne
from scipy.signal import freqz
from .common import *
from .preprocessing import chain,preprocess,extend,_fit_edge

APPENDIX=WORK/'output/plans/EEG_E2_E0_E1边界修复工程执行附表.md'

def cases():
    rows=[]
    phases=[0.,np.pi/4,np.pi/2,3*np.pi/4]
    for n in [75000,75089,83876]:
        for p in phases:
            rows.append({'n':n,'phase':p,'frequencies':[10.],'amplitudes':[5.],'line_frequency':50.,'line_amplitude':0.})
            for amp in [5.,500.]:
                for f in [49.8,50.,50.2]:
                    rows.append({'n':n,'phase':p,'frequencies':[10.],'amplitudes':[5.],'line_frequency':f,'line_amplitude':amp})
    for f in [1.,4.,10.,20.,35.]:
        for p in phases:
            rows.append({'n':75000,'phase':p,'frequencies':[f],'amplitudes':[5.],'line_frequency':50.,'line_amplitude':0.})
    for p in phases:
        rows.append({'n':75000,'phase':p,'frequencies':[1.,4.,10.,20.,35.],'amplitudes':[1.,.6,.4,.3,.2],
                     'line_frequency':50.,'line_amplitude':0.})
    assert len(rows)==108
    return [{'case':i,**r} for i,r in enumerate(rows)]

def signal(row,t,line=True):
    y=sum(a*np.sin(2*np.pi*f*t+row['phase']) for f,a in zip(row['frequencies'],row['amplitudes']))
    if line:y=y+row['line_amplitude']*np.sin(2*np.pi*row['line_frequency']*t+row['phase'])
    return y[None,:]

def ref(row,context,line=True):
    t=np.arange(-context*250,row['n']+context*250)/250
    y=chain(signal(row,t,line))
    start=context*200
    return y[:,start:start+60000]

def errors(y,r,denom):
    d=y-r
    return {k:float(np.sqrt(np.mean(d[:,s]**2))/denom) for k,s in
            [('first5',slice(0,1000)),('last5',slice(59000,60000)),('interior',slice(4000,56000))]}

def fitted_amplitudes(y,freqs):
    t=np.arange(4000,56000)/200
    columns=[]
    for f in freqs:columns.extend([np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t)])
    columns.extend([np.ones(len(t)),t-t.mean()])
    co=np.linalg.lstsq(np.stack(columns,1),y[0,4000:56000],rcond=None)[0]
    return np.hypot(co[:2*len(freqs):2],co[1:2*len(freqs):2])

def register():
    target=OUT/'engineering_registration.json'
    sources=[PLAN,APPENDIX,CODE/'preprocessing.py',CODE/'engineering.py',CODE/'common.py']
    config={'version':'EEG-E2-boundary-v1','candidate_order':['fixed50','adaptive_line'],
            'fallback_only_if_primary_fails':True,'sources':{p.relative_to(WORK).as_posix():digest(p) for p in sources},
            'cases':cases(),'labels_used':False,'new_classification_results_used':False,
            'reference_context_seconds':[60,90],'thresholds':{'edge_nrmse_max':.1,'old_error_ratio_max':.2,
            'gain_error_max':.05,'line50_attenuation_db_min':40,'reference_convergence_max':1e-4},
            'processing':{'fs_native':250,'fs_model':200,'bandpass':[.3,75.],'notch_hz':50.,'padding_native_samples':7500,
                          'edge_fit_native_samples':2500,'remove_recorded_samples':False}}
    if target.exists():
        old=read(target);assert old['protocol']==config,'Registered engineering changed'
        return old
    history=freeze_history()
    save(OUT/'historical_artifacts_before.json',history,immutable=True)
    record={'registered_utc':now(),'protocol':config}
    save(target,record,immutable=True)
    return record

def stationary(method):
    coeff=np.load(OUT/'E0_original_filter_coefficients.npz')
    rows=[]
    for row in cases():
        reference=ref(row,60)
        reference90=ref(row,90)
        retained=ref(row,60,False)
        denom=float(np.sqrt(np.mean(retained**2)))
        convergence=float(np.sqrt(np.mean((reference-reference90)**2))/denom)
        x=signal(row,np.arange(row['n'])/250)
        old,_=preprocess(x,'old')
        new,_=preprocess(x,method)
        oe=errors(old,reference,denom);ne=errors(new,reference,denom)
        freqs=list(row['frequencies'])
        if row['line_amplitude']>0:freqs.append(row['line_frequency'])
        amps=fitted_amplitudes(new,freqs)
        _,bp=freqz(coeff['bandpass'],worN=np.array(row['frequencies']),fs=200)
        _,nt=freqz(coeff['notch'],worN=np.array(row['frequencies']),fs=200)
        targets=np.array(row['amplitudes'])*abs(bp*nt)
        gainerr=float(np.max(abs(amps[:len(targets)]/targets-1)))
        attenuation=None
        if row['line_amplitude']>0 and row['line_frequency']==50:
            attenuation=float(-20*np.log10(max(amps[-1]/row['line_amplitude'],1e-30)))
        failures=[]
        if convergence>1e-4:failures.append('reference_not_converged')
        for side in ['first5','last5']:
            if ne[side]>.1:failures.append(side+'_absolute')
            if oe[side]>.1 and ne[side]>.2*oe[side]:failures.append(side+'_relative')
        if gainerr>.05:failures.append('passband_gain')
        if attenuation is not None and attenuation<40:failures.append('line50_attenuation')
        rows.append({**row,'old_nrmse':oe,'new_nrmse':ne,'reference_convergence_nrmse':convergence,
                     'gain_relative_error_max':gainerr,'line50_attenuation_db':attenuation,'failures':failures,'passed':not failures})
        if (row['case']+1)%12==0:print(f'{method}: stationary cases {row["case"]+1}/108',flush=True)
    return rows

def transient(method,center):
    t=np.arange(-22500,75000+22500)/250
    rng=np.random.default_rng(20260929)
    x=(5*np.sin(2*np.pi*10*t)*np.exp(-.5*((t-center)/.2)**2)+.1*rng.standard_normal(len(t)))[None,:]
    short=x[:,22500:97500]
    reference=chain(x)[:,18000:78000]
    out,_=preprocess(short,method)
    denom=float(np.sqrt(np.mean(reference**2)))
    nr=float(np.sqrt(np.mean((out-reference)**2))/denom)
    peak=float(abs(np.max(abs(out))/np.max(abs(reference))-1))
    return {'center_seconds':center,'whole_nrmse':nr,'peak_relative_error':peak,
            'passed':nr<=.01 and peak<=.05,'role':'acceptance' if center==150 else 'edge_stress_diagnostic'}

def run():
    register()
    save(OUT/'status.json',{'state':'E1_synthetic_engineering','updated_utc':now(),'new_models':0})
    completed=[];accepted=None
    for method in ['fixed50','adaptive_line']:
        target=OUT/f'engineering_{method}.json'
        if target.exists():report=read(target)
        else:
            rows=stationary(method);tr=transient(method,150);edge=transient(method,1)
            report={'method':method,'registration':desc(OUT/'engineering_registration.json'),
                    'stationary_cases':rows,'central_transient':tr,'edge_transient':edge,
                    'passed':all(r['passed'] for r in rows) and tr['passed'],
                    'stationary_failed_cases':sum(not r['passed'] for r in rows),
                    'finished_utc':now(),'no_classifier_training':True}
            save(target,report,immutable=True)
        completed.append({'method':method,'report':desc(target),'passed':report['passed']})
        print(json.dumps({'method':method,'passed':report['passed'],'failed_cases':report['stationary_failed_cases'],
                          'transient':report['central_transient'],'edge_stress':report['edge_transient']}),flush=True)
        if report['passed']:
            accepted=method;break
    gate={'passed':accepted is not None,'accepted_method':accepted,'candidates':completed,
          'registration':desc(OUT/'engineering_registration.json'),'fixed_thresholds_unchanged':True}
    save(OUT/'synthetic_gate.json',gate,immutable=True)
    save(OUT/'status.json',{'state':'E1_synthetic_passed' if accepted else 'E1_engineering_failed_stop_before_training',
                          'updated_utc':now(),'new_models':0,'accepted_method':accepted})
    print('ENGINEERING_GATE',json.dumps(gate),flush=True)

if __name__=='__main__':run()

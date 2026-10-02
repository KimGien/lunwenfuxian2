"""Independent result checks; no candidate or threshold changes."""
from scipy.signal import freqz
from tridep_ccml.eeg_e2.audit import independent_chain, metrics_independent
from .common import *
from .preprocessing import preprocess, extend


def analytic(row,t,line=True):
    result=np.zeros_like(t)
    phases=row.get('phases',[row['phase']]*len(row['frequencies']))
    for f,a,phase in zip(row['frequencies'],row['amplitudes'],phases):
        result+=a*np.sin(2*np.pi*f*t+phase)
    if line:result+=row['line_amplitude']*np.sin(2*np.pi*row['line_frequency']*t+row['phase'])
    return result[None]


def run():
    reg=read(OUT/'engineering_registration.json')
    assert all(digest(WORK/p)==sha for p,sha in reg['protocol']['sources'].items())
    gate=read(OUT/'synthetic_gate.json');assert gate['passed']
    d=read(WORK/gate['candidates'][0]['report']['path']);method=d['method']
    assert digest(WORK/gate['candidates'][0]['report']['path'])==gate['candidates'][0]['report']['sha256']
    for row,cfg in zip(d['rows'],reg['protocol']['cases']):
        assert all(row[k]==v for k,v in cfg.items())
        assert row['reference_convergence']<=1e-4 and row['gain_error']<=.05
        for s in ['first5','last5']:
            old,new=row['old_nrmse'][s],row['new_nrmse'][s]
            assert new<=.1
            assert old<=.1 or new<=.2*old
            assert old>.1 or new-old<=.02
        assert row['attenuation50_db'] is None or row['attenuation50_db']>=40
        assert row['passed'] and not row['failures']
    checks=[]
    for i in [4,104,108,119,131]:
        row=d['rows'][i];t=np.arange(-15000,row['n']+15000)/250
        reference=independent_chain(analytic(row,t))[:,12000:72000]
        retained=independent_chain(analytic(row,t,False))[:,12000:72000]
        denominator=float(np.linalg.norm(retained)/np.sqrt(retained.size))
        x=analytic(row,np.arange(row['n'])/250)
        new,_=preprocess(x,method);extended,_=extend(x,method)
        separate=independent_chain(extended)[:,6000:66000]
        diff=float(np.max(abs(separate-new)));assert diff<1e-9
        recalculated=metrics_independent(separate,reference,denominator)
        assert all(np.isclose(recalculated[s],row['new_nrmse'][s],atol=1e-9,rtol=1e-5) for s in recalculated)
        checks.append({'case':i,'independent_fir_max_absolute_error':diff,'recomputed_nrmse':recalculated})
    subjects=read(OUT/'input_manifest.json')['records']
    assert len(subjects)==37 and len(set(r['subject_id'] for r in subjects))==37
    for row in subjects:
        assert row['old_cache_exact'] and row['recorded_samples_unchanged'] and row['removed_windows']==0
        assert row['shape']==[60,19,1000] and not row['labels_used'] and not row['augmentation']
        p=WORK/row['new_cache']['path'];assert digest(p)==row['new_cache']['sha256']
        x=np.load(p,allow_pickle=False);assert x.shape==(60,19,1000) and np.isfinite(x).all()
        assert max(row['external_peak_ratios'])<=20
    paired={}
    for side in ['first5','last5']:
        old=np.array([np.median(r['pseudo_boundary']['old'][side]) for r in subjects])
        new=np.array([np.median(r['pseudo_boundary']['new'][side]) for r in subjects])
        paired[side]={'median_old':float(np.median(old)),'median_new':float(np.median(new)),
            'median_relative_change':float(np.median(new/old-1)),
            'worse_subjects':[r['subject_id'] for j,r in enumerate(subjects) if new[j]>=old[j]]}
        assert np.median(new)<np.median(old) and np.median(new-old)<0
    save(OUT/'engineering_verification.json',{'passed':True,'registered_rows_checked':132,
        'independent_checks':checks,'new_cache_subjects_verified':37,'pseudo_boundary_recomputed':paired,
        'harmonic_accepted_recording_sides':sum(m['harmonic_accepted'] for r in subjects for m in r['mode_metadata']),
        'recording_channel_sides':37*19*2,'verification_code':desc(__file__)},immutable=True)
    print('ENGINEERING_VERIFICATION PASSED',paired,flush=True)


if __name__=='__main__':run()

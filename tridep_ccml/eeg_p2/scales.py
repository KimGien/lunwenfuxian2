"""Fixed numerical comparisons on outer-training identities only; no prediction labels."""
from __future__ import annotations
from .common import *

METHODS=['S0_R1_pooled','S1_subject_equal_robust','S2_subject_zscore','S3_subject_mad']


def check_ids(ids,allowed):
    require(len(ids)>=2 and len(ids)==len(set(ids)) and set(ids)<=set(allowed),'Scale fit outside training identities')


def fit(stats,ids,allowed,method):
    check_ids(ids,allowed)
    selected=[stats[s] for s in ids]
    if method==METHODS[0]:
        means=np.array([r['mean'] for r in selected])
        stds=np.array([r['std'] for r in selected])
        center=means.mean(0)
        scale=np.sqrt(np.mean(stds**2+(means-center)**2,0))
    elif method==METHODS[1]:
        center=np.median([r['median'] for r in selected],0)
        scale=np.median([r['mad_sigma'] for r in selected],0)
    else:
        raise ValueError('Shared fit supports S0/S1 only')
    require(np.isfinite(center).all() and np.isfinite(scale).all() and np.min(scale)>RULES['scale_floor'],'Degenerate fitted scale')
    return center,scale


def subject_transform(x,method):
    require(np.isfinite(x).all(),'Nonfinite subject input')
    if method==METHODS[2]:
        center=x.mean(-1)
        scale=x.std(-1)
    elif method==METHODS[3]:
        center=np.median(x,-1)
        scale=1.4826*np.median(np.abs(x-center[:,None]),-1)
    else:
        raise ValueError('Subject transform supports S2/S3 only')
    require(np.min(scale)>RULES['scale_floor'],'Degenerate subject scale')
    return (x-center[:,None])/scale[:,None],center,scale


def run_scale_audit(records,folds):
    lookup={r['subject_id']:r for r in records}
    stats={}
    for sid in lookup:
        r=read_json(OUT/f'subjects/{sid}.json')['filtered_19']['channels']
        stats[sid]={key:[c[field] for c in r] for key,field in [('mean','mean'),('std','centered_rms'),('median','median'),('mad_sigma','sigma_mad')]}
    reports=[]
    for fold in range(1,6):
        ids=folds[f'fold_{fold}']['train']
        shared={method:fit(stats,ids,ids,method) for method in METHODS[:2]}
        old=read_json(R1/f'cache/standardizers/fold_{fold}.json')
        require(old['fit_subject_ids']==ids,'Original standardizer identity mismatch')
        np.testing.assert_allclose(shared[METHODS[0]][0],old['mean'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(shared[METHODS[0]][1],old['std'],rtol=1e-10,atol=1e-10)
        result={'fold':fold,'fit_subject_ids':ids,'evaluation_subject_ids':ids,
                'val_test_used':False,'labels_used':False,'methods':{},'R1_standardizer_replayed':True}
        for method in METHODS:
            rows=[]
            for sid in ids:
                x=np.load(WORK/lookup[sid]['cache']['path'],allow_pickle=False).transpose(1,0,2).reshape(19,-1).astype(float)
                if method in shared:
                    center,scale=shared[method]
                    z=(x-center[:,None])/scale[:,None]
                else:
                    z,center,scale=subject_transform(x,method)
                # Check an actual affine-transform probe; fixed first 5 seconds.
                a=x[:,:1000]-x[:,:1000].mean(-1,keepdims=True)
                b=z[:,:1000]-z[:,:1000].mean(-1,keepdims=True)
                corr=np.sum(a*b,-1)/np.sqrt(np.sum(a*a,-1)*np.sum(b*b,-1))
                require(np.isfinite(z).all() and np.min(corr)>1-1e-10,'Scale transform corrupted waveform')
                rows.append({'subject_id':sid,'rms':float(np.sqrt(np.mean(z*z))),
                    'energy_sum':float(np.sum(z*z)),'fraction_abs_over_8':float(np.mean(np.abs(z)>8)),
                    'abs_max':float(np.max(np.abs(z))),'min_channel_correlation':float(corr.min()),
                    'center':center.tolist(),'scale':scale.tolist()})
            energy=np.array([r['energy_sum'] for r in rows]); rms=np.array([r['rms'] for r in rows])
            summary={'subject_rms_min':float(rms.min()),'subject_rms_median':float(np.median(rms)),
                'subject_rms_max':float(rms.max()),'subject_rms_max_min_ratio':float(rms.max()/rms.min()),
                'largest_energy_subject':rows[int(energy.argmax())]['subject_id'],
                'largest_subject_energy_share':float(energy.max()/energy.sum()),
                'mean_fraction_abs_over_8':float(np.mean([r['fraction_abs_over_8'] for r in rows]))}
            influence=[]
            if method in shared:
                for left_out in ids:
                    remaining=[s for s in ids if s!=left_out]
                    _,scale=fit(stats,remaining,ids,method)
                    ratio=shared[method][1]/scale
                    influence.append({'left_out_train_subject':left_out,
                        'max_abs_log2_scale_change':float(np.max(np.abs(np.log2(ratio)))),
                        'max_symmetric_scale_ratio':float(np.maximum(ratio,1/ratio).max())})
                summary['largest_leave_one_scale_factor']=max(r['max_symmetric_scale_ratio'] for r in influence)
            result['methods'][method]={'summary':summary,'subjects':rows,'leave_one_train_influence':influence}
        save(OUT/f'scales/fold_{fold}.json',result)
        reports.append(result)
        print(f'SCALE fold={fold} complete: train-only; S0 replay verified',flush=True)
    save(OUT/'scale_summary.json',{'methods':METHODS,'folds':[{ 'fold':r['fold'],
        'methods':{m:v['summary'] for m,v in r['methods'].items()}} for r in reports],
        'selected_method':None,'new_classification_scores':False,'labels_used':False})


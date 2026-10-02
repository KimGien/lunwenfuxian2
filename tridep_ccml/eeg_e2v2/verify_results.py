"""Recompute thresholds and aggregate metrics without using the reporting metric routine."""
import math
from .common import *


def score(rows):
    y=np.array([r['label'] for r in rows]);p=np.array([r['p'] for r in rows])
    pred=np.array([r['p']>r['threshold'] for r in rows])
    tp=int(np.sum((y==1)&pred));tn=int(np.sum((y==0)&~pred))
    fp=int(np.sum((y==0)&pred));fn=int(np.sum((y==1)&~pred))
    m=(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)+(2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0)
    pos=p[y==1];neg=p[y==0]
    auc=float(np.mean([float(a>b)+.5*float(a==b) for a in pos for b in neg]))
    return {'macro_f1':m/2,'balanced_accuracy':(tp/(tp+fn)+tn/(tn+fp))/2,
        'mdd_recall':tp/(tp+fn),'auc':auc,
        'nll':float(-sum(math.log(max(1e-15,v if label else 1-v)) for v,label in zip(p,y))/len(y)),
        'brier':float(sum((v-label)**2 for v,label in zip(p,y))/len(y))}


def run():
    assert read(OUT/'evaluation_verification.json')['passed']
    config=read(OUT/'configuration.json')
    thresholds=0
    for path in (OUT/'thresholds').rglob('fold_*.json'):
        doc=read(path);rows=doc['oof'];params=doc['parameters'];p=sorted(set(r['p'] for r in rows))
        candidates=sorted(set([0.,.5,1.]+[(a+b)/2 for a,b in zip(p[:-1],p[1:])]))
        scores=[score([{**r,'threshold':t} for r in rows])['macro_f1'] for t in candidates]
        maximum=max(scores)
        chosen=min((t for t,s in zip(candidates,scores) if abs(s-maximum)<1e-12),key=lambda t:(abs(t-.5),t))
        assert chosen==params['threshold'] and abs(maximum-params['oof_fit_macro_f1'])<1e-12
        assert set(r['subject_id'] for r in rows)==set(params['fit_subject_ids'])
        for r in rows:assert r['subject_id'] not in r['fit_ids'] and set(r['fit_ids'])<set(params['fit_subject_ids'])
        thresholds+=1
    assert thresholds==15
    allrows=read(REPORT/'predictions_long.json');results=read(REPORT/'metrics.json');metric_rows=0
    for entry in results:
        rows=[r for r in allrows if r['condition']==entry['condition'] and r['split']==entry['split']]
        recomputed=[]
        for saved in entry['per_fold']:
            group=[r for r in rows if r['seed']==saved['seed'] and r['fold']==saved['fold']]
            actual=score(group)
            for k,v in actual.items():
                if saved[k] is not None:assert abs(v-saved[k])<1e-12,(entry['condition'],k,v,saved[k])
            recomputed.append(actual);metric_rows+=1
        for k,v in entry['mean'].items():
            if v is not None:assert abs(v-np.mean([r[k] for r in recomputed]))<1e-12
    for path in (OUT/'runs').rglob('trained.json'):
        d=read(path);assert d['configuration']==desc(OUT/'configuration.json')
        assert read(path.parent/'initial_health.json')['passed'] and read(path.parent/'final_health.json')['passed']
        pairing=read(path.parent/'pairing.json')
        assert pairing['deterministic'] and not pairing['cudnn_tf32'] and not pairing['matmul_tf32']
        assert len(read(path.parent/'history.json'))==60
        assert len(read(path.parent/'batch_order.json')['orders'])==60
    assert len(list((OUT/'runs').rglob('trained.json')))==60
    # The source and every locked fitted artifact must still match after evaluation.
    for p,sha in config['sources'].items():assert digest(WORK/p)==sha
    for p,sha in read(OUT/'release_gate.json')['fitted_artifacts'].items():assert digest(OUT/p)==sha
    save(OUT/'results_verification.json',{'passed':True,'independent_threshold_checks':thresholds,
        'independent_fold_metric_rows':metric_rows,'prediction_rows':len(allrows),
        'formal_models_health_checked':60,'source':desc(__file__)},immutable=True)
    print('RESULT VERIFICATION PASSED',metric_rows,flush=True)


if __name__=='__main__':run()

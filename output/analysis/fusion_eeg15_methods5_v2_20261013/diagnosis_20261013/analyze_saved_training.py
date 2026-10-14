"""Post-hoc read-only diagnosis of sealed runs; no training or checkpoint selection."""
from pathlib import Path
import json
import hashlib
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / 'tridep_ccml/experiments/Fusion-EEG15-Methods5-v2-37-20261013'
OUT = Path(__file__).resolve().parent
METHODS = ['CCML', 'LMF', 'MMDynamics', 'QMF', 'ECML']

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def avg(xs):
    return float(np.mean(xs)) if xs else None

rows = []
bound = []
for p in sorted((EXP / 'runs').glob('*/seed_*/fold_*/trajectory.json')):
    t = read(p)
    m = read(p.with_name('metadata.json'))
    e = read(p.with_name('evaluation.json'))
    best = min(t, key=lambda x:x['val_nll'])
    assert best['epoch'] == m['best_epoch'] and len(t) == m['epochs']
    assert len(t) >= 60 and len(t) <= 120
    end10 = avg([x['val_nll'] for x in t[-10:]])
    prev10 = avg([x['val_nll'] for x in t[-20:-10]])
    r = dict(m['identity'], epochs=len(t), best_epoch=best['epoch'], parameters=m['parameters'],
             stopping_reason=m['stopping_reason'], best_val_nll=best['val_nll'],
             final_val_nll=t[-1]['val_nll'], final_minus_best_val_nll=t[-1]['val_nll']-best['val_nll'],
             end10_minus_prior10_val_nll=end10-prev10,
             first10_train_objective=avg([x['loss'] for x in t[:10]]),
             last10_train_objective=avg([x['loss'] for x in t[-10:]]),
             selected_metrics=e['metrics'], last_components=t[-1]['components'],
             updates=m['updates'])
    rows.append(r)
    bound.append({'path':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
assert len(rows) == 1200

def summarize(rs):
    b = [r['best_epoch'] for r in rs]
    capped = [r for r in rs if r['epochs']==120]
    return {
      'fits':len(rs), 'best_epoch_median':float(np.median(b)),
      'best_epoch_quartiles':np.percentile(b,[25,50,75]).tolist(),
      'best_epoch_le10':sum(x<=10 for x in b), 'best_epoch_le60':sum(x<=60 for x in b),
      'best_epoch_gt100':sum(x>100 for x in b), 'best_epoch_ge110':sum(x>=110 for x in b),
      'best_epoch_120':sum(x==120 for x in b), 'reached120':len(capped),
      'stopping_reason_counts':dict(Counter(r['stopping_reason'] for r in rs)),
      'capped_end10_val_better_than_prior10_by_1e4':sum(r['end10_minus_prior10_val_nll'] < -1e-4 for r in capped),
      'capped_best_ge110_and_tail_improving':sum(r['best_epoch']>=110 and r['end10_minus_prior10_val_nll'] < -1e-4 for r in capped),
      'median_final_minus_best_val_nll':float(np.median([r['final_minus_best_val_nll'] for r in rs])),
      'final_val_worse_than_best_by_point01':sum(r['final_minus_best_val_nll']>.01 for r in rs),
      'mean_selected_train_f1':avg([r['selected_metrics']['train']['macro_f1'] for r in rs]),
      'mean_selected_val_f1':avg([r['selected_metrics']['val']['macro_f1'] for r in rs]),
      'mean_selected_test_f1':avg([r['selected_metrics']['test']['macro_f1'] for r in rs]),
      'train_f1_equal1':sum(r['selected_metrics']['train']['macro_f1']==1 for r in rs),
      'mean_selected_train_nll':avg([r['selected_metrics']['train']['nll'] for r in rs]),
      'mean_selected_val_nll':avg([r['selected_metrics']['val']['nll'] for r in rs]),
      'mean_selected_test_nll':avg([r['selected_metrics']['test']['nll'] for r in rs]),
      'parameters_range':[min(r['parameters'] for r in rs), max(r['parameters'] for r in rs)],
      'last10_objective_median':float(np.median([r['last10_train_objective'] for r in rs])),
      'first10_objective_median':float(np.median([r['first10_train_objective'] for r in rs]))}

summary = {
    'scope':'Post-hoc diagnostic only. All saved train/validation trajectories and already published selected-checkpoint metrics. No model forward, fitting, test-based checkpoint/threshold choice, or mutation of sealed experiment.',
    'all':summarize(rows),
    'by_method':{m:summarize([r for r in rows if r['method']==m]) for m in METHODS},
    'est_by_method':{m:summarize([r for r in rows if r['method']==m and r['eeg']!='ST']) for m in METHODS},
    'st_by_method':{m:summarize([r for r in rows if r['method']==m and r['eeg']=='ST']) for m in METHODS},
    'condition_summaries':{f'{e}-{m}':summarize([r for r in rows if r['eeg']==e and r['method']==m]) for e in sorted(set(r['eeg'] for r in rows)) for m in METHODS},
    'limitations':['Folds and seeds reuse people; counts are fit counts, not independent people.',
                  'Tail validation improvement is an exploratory sign, not proof that extra epochs improve test F1.',
                  'Objective values across methods or annealing stages cannot be compared as the same loss.',
                  'Train/test gap on supervised in-fold frozen representations includes branch fitting and fusion fitting.',
                  'This analysis does not certify full method equivalence; separate source audit applies.'],
    'trajectories':bound,
    'fits':rows}
(OUT/'saved_training_diagnosis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

fig, axs = plt.subplots(1,2,figsize=(12,4.6))
colors = ['#3467a1','#85afd5','#efb850','#be4c45']
bins = [lambda x:x<=10,lambda x:10<x<=60,lambda x:60<x<110,lambda x:x>=110]
bottom = np.zeros(5)
for fn,label,color in zip(bins,['1-10','11-60','61-109','110-120'],colors):
    count=np.array([sum(fn(r['best_epoch']) for r in rows if r['method']==m) for m in METHODS])
    axs[0].bar(METHODS,count,bottom=bottom,label=label,color=color)
    bottom+=count
axs[0].set(title='Selected epoch across all 1,200 fits',ylabel='Fits per method (240)')
axs[0].legend(title='Best validation NLL epoch',fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.24),ncol=4)
xx=np.arange(5)
for i,(role,color) in enumerate([('train','#3467a1'),('val','#85afd5'),('test','#be4c45')]):
    vals=[100*summary['by_method'][m][f'mean_selected_{role}_f1'] for m in METHODS]
    axs[1].bar(xx+(i-1)*.25,vals,.25,color=color,label=role)
axs[1].set_xticks(xx,METHODS)
axs[1].set(title='Selected checkpoint: mean per-fit Macro-F1',ylabel='Macro-F1 (%)',ylim=(0,105))
axs[1].legend()
for ax in axs:
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='x',labelrotation=15)
fig.text(.01,.005,'Descriptive, not independent samples. QMF source-parity finding is reviewed separately; no retraining or test-based selection.',fontsize=8)
fig.tight_layout(rect=(0,.05,1,1))
fig.savefig(OUT/'training_diagnosis.png',dpi=180)
print(json.dumps({k:summary[k] for k in ['all','by_method']},ensure_ascii=False,indent=2))

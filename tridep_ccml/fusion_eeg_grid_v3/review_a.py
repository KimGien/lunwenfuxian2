"""Independent phase-A acceptance: recompute ledger flags/pairs and baseline."""
from __future__ import annotations
import csv
from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score
from . import common as C

def table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream: return list(csv.DictReader(stream))
def flag(value): return value == 'True'
def run():
    C.check_pause()
    out=C.REPORT/'phase_a'; complete=C.read(out/'complete.json')
    C.require(complete['passed'] and complete['counts']['new_fits']==0 and complete['counts']['new_model_forwards']==0,'A boundary differs')
    for record in [complete['script'],complete['plan'],*complete['outputs']]: C.checked(record)
    data=table(out/'oof_error_ledger.csv'); baseline=table(out/'baseline_error_ledger.csv')
    base={(int(r['seed']),int(r['fold']),r['subject_id']):r for r in baseline}
    lookup={(r['eeg'],r['method'],int(r['seed']),int(r['fold']),r['subject_id']):r for r in data}
    C.require(len(data)==len(lookup)==8880 and len(base)==111,'Ledger identities duplicate/missing')
    grouped=defaultdict(list)
    for r in data:
        C.check_pause()
        y=int(r['label']); p=float(r['p_mdd']); pred=int(p>.5); correct=pred==y
        C.require(r['role']=='test' and int(r['prediction'])==pred and flag(r['correct'])==correct,'Classification flag differs')
        C.require(flag(r['mdd_missed'])==(y==1 and pred==0) and flag(r['hc_false_alarm'])==(y==0 and pred==1),'Error direction differs')
        C.require(flag(r['boundary'])==(.4<=p<=.6) and flag(r['high_confidence_error'])==(not correct and max(p,1-p)>=.9),'Confidence category differs')
        seed,fold,sid=int(r['seed']),int(r['fold']),r['subject_id']
        b=base[(seed,fold,sid)]
        C.require(y==int(b['label']) and float(r['baseline_probability'])==float(b['p']),'Baseline row differs')
        for ref,eeg in (('baseline',None),('same_method_ST','ST'),('same_method_E00','E00')):
            q=float(b['p']) if eeg is None else float(lookup[(eeg,r['method'],seed,fold,sid)]['p_mdd'])
            was_correct=int(q>.5)==y
            C.require(flag(r[ref+'_corrected'])==(correct and not was_correct) and
                      flag(r[ref+'_new_error'])==(not correct and was_correct),'Paired correction flag differs')
        C.require(r['quality_valid_windows']=='null' and r['speech_recording_coverage']=='null','Unknown quality was fabricated')
        grouped[(r['eeg'],r['method'],seed)].append(r)
    C.require(len(grouped)==240 and all(len(v)==37 and len({r['subject_id'] for r in v})==37 for v in grouped.values()),'Condition OOF coverage differs')
    primary=float(np.mean([np.mean([f1_score([int(r['label']) for r in baseline if int(r['seed'])==s and int(r['fold'])==f],
                  [int(float(r['p'])>.5) for r in baseline if int(r['seed'])==s and int(r['fold'])==f],average='macro')
                  for f in range(1,6)]) for s in C.SEEDS]))
    C.require(abs(primary-.7255026455026455)<1e-14,'Locked baseline independent sklearn recomputation differs')
    people=table(out/'subject_summary_37.csv'); hypotheses=table(out/'mechanism_hypotheses.csv')
    C.require(len(people)==37 and len({r['subject_id'] for r in people})==37 and len(hypotheses)>=3,'A deliverable incomplete')
    report={'passed':True,'scope':'Independent phase-A ledger/paired arithmetic and source-output binding; no training',
            'reviewer':'root, independent of diagnosis_a implementation agent','phase_a':C.desc(out/'complete.json'),
            'review_script':C.desc(__file__),'reviewed_source':complete['script'],'ledger_rows_checked':len(data),
            'paired_flags_checked':len(data)*3,'baseline_sklearn_macro_f1':primary,'unique_people':37,
            'new_model_forwards':0,'new_optimizer_updates':0,'unknown_quality_preserved':True,'at':C.now()}
    C.save(out/'independent_review.json',report,immutable=True)
    print('Independent A review passed',len(data),'rows and',len(data)*3,'pairs',flush=True)
    return report
if __name__=='__main__':run()


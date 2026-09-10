"""Subject-level binary metrics and validation-only weighted late fusion."""
import math

def metrics(labels, probabilities):
    if not labels or len(labels)!=len(probabilities):
        raise ValueError('Nonempty labels and probabilities must have equal length')
    if any(y not in (0,1) for y in labels):
        raise ValueError('Labels must be 0=HC, 1=MDD')
    if any(not math.isfinite(p) or not 0<=p<=1 for p in probabilities):
        raise ValueError('Probabilities must be finite and within [0,1]')
    pred=[int(p>0.5) for p in probabilities]  # softmax/argmax tie -> HC
    recalls=[]; f1s=[]
    for c in (0,1):
        tp=sum(y==c and p==c for y,p in zip(labels,pred))
        fp=sum(y!=c and p==c for y,p in zip(labels,pred))
        fn=sum(y==c and p!=c for y,p in zip(labels,pred))
        recalls.append(tp/(tp+fn) if tp+fn else None)
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    return {'macro_f1':sum(f1s)/2,'mdd_recall':recalls[1],
            'balanced_accuracy':sum(recalls)/2 if None not in recalls else None,
            'n_subjects':len(labels)}

def select_weights(rows):
    if not rows or any(r['split']!='val' for r in rows):
        raise ValueError('Fusion selection accepts internal validation rows only')
    if len({r['subject_id'] for r in rows})!=len(rows):
        raise ValueError('Each validation subject must appear once')
    if len({(r['seed'],r['outer_fold']) for r in rows})!=1:
        raise ValueError('Do not pool folds or seeds for fusion selection')
    best=None
    for i in range(21):
        for j in range(21-i):
            w=(i/20,j/20,(20-i-j)/20)
            p=[sum(a*r[k] for a,k in zip(w,('p_eeg','p_speech','p_text'))) for r in rows]
            score=metrics([r['label'] for r in rows],p)['macro_f1']
            if best is None or score>best['validation_macro_f1']:
                best={'weights':list(w),'validation_macro_f1':score}
    return best

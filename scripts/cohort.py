"""Shared MODMA cohort contract for B0-37 and every subsequent experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path
from prepare import ROOT, audit_folds, digest

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def validate_active_cohort(root=ROOT, require_subjects=True, require_data=False):
    lock=read(root/'configs/cohort.json')
    path=root/lock['fold_file']; folds=read(path); ids=set(lock['subject_ids'])
    if len(ids)!=37 or len(lock['subject_ids'])!=37 or '02010004' in ids:
        raise ValueError('Invalid locked MODMA-37 cohort')
    if digest(path)!=lock['folds_sha256']:
        raise ValueError('Active fold hash differs from cohort lock')
    if not audit_folds(folds, expected_count=37, expected_ids=ids)['passed']:
        raise ValueError('Active folds violate MODMA-37 cohort')
    original=root/lock['official_fold_file']
    if digest(original)!=lock['official_folds_sha256']:
        raise ValueError('Archived official folds changed')
    expected={n:{p:[s for s in v if s in ids] for p,v in f.items()} for n,f in read(original).items()}
    if folds!=expected:
        raise ValueError('Active folds must retain original assignments and order')
    if require_subjects:
        subjects=read(root/'manifests/subjects.json')
        if len(subjects)!=37 or {r['subject_id'] for r in subjects}!=ids:
            raise ValueError('Subject manifest does not match locked cohort')
        labels={r['subject_id']:r['label'] for r in subjects}
        if dict(Counter(str(v) for v in labels.values()))!=lock['class_counts']:
            raise ValueError('Diagnosis counts differ from cohort lock')
        for f in folds.values():
            if any(set(labels[s] for s in v)!={0,1} for v in f.values()):
                raise ValueError('Every train/val/test split must contain both classes')
    if require_data:
        audit=read(root/'reports/modma_audit.json')
        if (not audit['passed'] or audit.get('cohort_id')!=lock['cohort_id']
                or audit.get('folds_sha256')!=lock['folds_sha256']
                or audit.get('subject_manifest_sha256')!=digest(root/'manifests/subjects.json')
                or audit['cohort']!=37 or audit['wav_count']!=1073):
            raise ValueError('Missing, failed or stale MODMA-37 data audit')
    return lock

def validate_prediction_records(rows, experiment, root=ROOT):
    lock=validate_active_cohort(root, require_data=True)
    if experiment not in lock['experiment_ids']:
        raise ValueError('Experiment is not registered for the shared cohort')
    folds=read(root/lock['fold_file'])
    labels={r['subject_id']:r['label'] for r in read(root/'manifests/subjects.json')}
    seen=set(); tests={}
    for r in rows:
        if (r.get('experiment')!=experiment or r.get('cohort_id')!=lock['cohort_id']
                or r.get('folds_sha256')!=lock['folds_sha256']):
            raise ValueError('Prediction experiment/cohort/fold provenance mismatch')
        sid=r['subject_id']; f=folds.get(f"fold_{r['outer_fold']}")
        if not f or r['split'] not in f or sid not in f[r['split']] or labels.get(sid)!=r['label']:
            raise ValueError('Prediction subject, split or diagnosis differs from locked cohort')
        if r['seed'] not in (11,22,33,44,55,42): raise ValueError('Unregistered seed')
        key=(r['seed'],r['outer_fold'],sid)
        if key in seen: raise ValueError('Duplicate subject prediction')
        seen.add(key); tests.setdefault(r['seed'],[])
        if r['split']=='test': tests[r['seed']].append(sid)
    if not tests or any(len(v)!=37 or set(v)!=set(lock['subject_ids']) for v in tests.values()):
        raise ValueError('Each seed must include exactly 37 test subjects once across five folds')
    return {'passed':True,'cohort_id':lock['cohort_id'],'experiment':experiment,'seeds':sorted(tests),'test_subjects_per_seed':37}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--experiment',default='B0-37')
    args=parser.parse_args()
    if args.predictions:
        rows=[json.loads(line) for line in args.predictions.read_text(encoding='utf-8').splitlines() if line.strip()]
        print(json.dumps(validate_prediction_records(rows,args.experiment),indent=2))
    else:
        lock=validate_active_cohort(require_data=True)
        print(json.dumps({'passed':True,'cohort_id':lock['cohort_id'],'subjects':37,'folds_sha256':lock['folds_sha256']},indent=2))

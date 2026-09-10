"""Build recovery evidence and an INACTIVE 37-subject fallback; no training."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'recovery'

def read(path):
    return json.loads((ROOT/path).read_text(encoding='utf-8'))

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if (ROOT/'configs/cohort.json').exists():
        print('MODMA-37 is active; preserving historical recovery options and evidence.')
        return
    original_path=ROOT/'manifests/fold_assignments.json'
    original_hash=digest(original_path)
    folds=read('manifests/fold_assignments.json')
    subjects=read('manifests/subjects.json')
    complete=[r for r in subjects if r['raw_complete']]
    excluded=[r['subject_id'] for r in subjects if not r['raw_complete']]
    keep={r['subject_id'] for r in complete}
    labels={r['subject_id']:r['label'] for r in subjects}
    assert len(complete)==37 and excluded==['02010004'], 'Reassess fallback if source data have changed'
    candidate={name:{part:[sid for sid in values if sid in keep] for part,values in fold.items()}
               for name,fold in folds.items()}
    checks={}; all_test=[]
    for name,fold in candidate.items():
        parts={p:set(v) for p,v in fold.items()}
        assert all(len(fold[p])==len(parts[p]) for p in parts)
        assert not(parts['train']&parts['val'] or parts['train']&parts['test'] or parts['val']&parts['test'])
        assert set.union(*parts.values())==keep
        assert all(set(labels[s] for s in v)=={0,1} for v in fold.values())
        all_test.extend(fold['test'])
        checks[name]={p:{'n':len(v),'class_counts':dict(Counter(labels[s] for s in v))} for p,v in fold.items()}
    assert len(all_test)==len(set(all_test))==37 and set(all_test)==keep
    option=OUT/'options/B0-37'
    save(option/'fold_assignments.json',candidate)
    save(option/'subjects.json',complete)
    save(option/'status.json',{'experiment':'B0-37','active':False,'training_ready':False,
        'status':'cohort_and_splits_prepared_only','excluded_subject_ids':excluded,
        'selection_basis':'raw data decode and severe-noise audit only; no prediction scores',
        'subjects':37,'class_counts':dict(Counter(r['label'] for r in complete)),
        'recordings':37*29,'source_official_folds_sha256':original_hash,
        'fold_audit':checks,'test_each_subject_once':True,'original_fold_order_retained':True,
        'scientific_limit':'Changed cohort: cannot claim exact 38-subject TRI-DEP replication; all comparisons must rerun on the same B0-37 cohort. Encoder and EEG protocol questions remain unresolved.'})
    invalid=read('reports/modma_bad_audio.json'); suspect=read('reports/modma_suspect_audio.json')
    requests=[{'subject_id':r['subject_id'],'question_id':r['question_id'],
               'current_sha256':r['sha256'],'reason':'unreadable_WAV'} for r in invalid]
    requests += [{'subject_id':r['subject_id'],'question_id':r['question_id'],
                   'current_sha256':r['sha256'],'reason':'severe_noise_in_tail_from_about_11_seconds'} for r in suspect]
    requests.sort(key=lambda r:(r['subject_id'],r['question_id']))
    save(OUT/'replacement_request.json',{'files':requests,
        'acceptance':['Original recording with matching subject and question identity from a documented source',
                      'Record original bytes and SHA-256 separately from user archives',
                      'Check full PCM decoding and rerun signal-quality screening',
                      'Compare surviving portions when possible; no header guessing, content synthesis, or other-subject substitution'],
        'recommended_source':'MODMA data custodians, or author-confirmed TRI-DEP input and handling policy'})
    save(OUT/'status.json',{'checked_utc':datetime.now(timezone.utc).isoformat(),
        'recovered_full_recordings':0,'data_problem_resolved':False,
        'original_zip_and_raw_modified':False,'original_38_subject_folds_modified':False,
        'header_invalid_files':5,'signal_suspect_files':1,'affected_subjects':['02010004'],
        'complete_subjects':37,'decodable_without_severe_noise_flags':1096,
        'mirror_matches_local_for_questions':['23','24','25','26','27','28'],
        'known_source_defect_reference':'https://pmc.ncbi.nlm.nih.gov/articles/PMC8296267/',
        'official_archive_access':'ReShare audio endpoint returned HTTP 401; registered access is required',
        'inactive_fallback':'recovery/options/B0-37/status.json',
        'baseline_cohort_changed':False,'training_run':False})
    assert digest(original_path)==original_hash
    print(json.dumps({'fallback_active':False,'complete_subjects':37,'folds':checks,'replacement_files':len(requests)},indent=2))

if __name__=='__main__':
    main()

"""Apply the user's authorized 37-subject policy; preserve prior evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from prepare import ROOT, WORK, digest, save

def main():
    if (ROOT/'configs/cohort.json').exists():
        from cohort import validate_active_cohort
        validate_active_cohort()
        print('MODMA-37 already active; no files changed.')
        return
    archive=ROOT/'archive/cohort38'
    paths=list((ROOT/'reports').glob('*.json'))+list((ROOT/'manifests').glob('*.json'))
    paths += [ROOT/'configs/b0.json',ROOT/'README.md',ROOT/'reports/差异与阻塞.md',ROOT/'recovery/status.json',ROOT/'recovery/options/B0-37/status.json']
    for path in paths:
        target=archive/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if digest(target)!=digest(path): raise ValueError('Existing archive differs: '+str(target))
        else: shutil.copyfile(path,target)
    shutil.copyfile(WORK/'output/plans/TRI-DEP_递进实验路线.md',archive/'plan_before_activation.md')
    official=ROOT/'upstream-dev/data/split_dataset_june/fold_assignments.json'
    shutil.copyfile(official,ROOT/'manifests/fold_assignments.official38.json')
    candidate=ROOT/'recovery/options/B0-37'
    subjects=json.loads((candidate/'subjects.json').read_text(encoding='utf-8'))
    if len(subjects)!=37 or any(not r['raw_complete'] or r['subject_id']=='02010004' for r in subjects):
        raise ValueError('Candidate must contain 37 complete subjects')
    excluded=ROOT/'staged/02010004'
    destination=archive/'staged/02010004'
    if not excluded.resolve().is_relative_to((ROOT/'staged').resolve()) or not destination.resolve().is_relative_to(archive.resolve()):
        raise ValueError('Invalid archive move target')
    if excluded.exists():
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists(): raise ValueError('Refusing to overwrite archived subject')
        excluded.rename(destination)
    shutil.copyfile(candidate/'fold_assignments.json',ROOT/'manifests/fold_assignments.json')
    save(ROOT/'manifests/subjects.json',subjects)
    ids=['B0-37']+[f'S{i:02d}-37' for i in range(1,13)]+['S06-T-37','S13a-37','S13b-37','S14-37','S15-37']
    lock={'cohort_id':'MODMA-37-v1','activated_utc':datetime.now(timezone.utc).isoformat(),
          'authorization':'User requested path 2 and the same 37 subjects for baseline and all subsequent experiments.',
          'baseline_experiment':'B0-37','experiment_ids':ids,'subject_count':37,
          'subject_ids':sorted(r['subject_id'] for r in subjects),'class_counts':{'0':21,'1':16},
          'excluded_subject_ids':['02010004'],'exclusion_reason':'Unreadable questions 24-28 and severe noise in question 23; source quality only, no model scores.',
          'questions_per_subject':29,'recordings':1073,
          'fold_file':'manifests/fold_assignments.json','folds_sha256':digest(ROOT/'manifests/fold_assignments.json'),
          'official_fold_file':'manifests/fold_assignments.official38.json','official_folds_sha256':digest(official),
          'split_policy':'Remove 02010004 from every original split; retain other assignments and order.',
          'comparison_policy':'All MODMA experiments and modalities use this exact cohort and folds; compare to B0-37. Original 38-subject paper scores are contextual references only.',
          'source_dataset_policy':'EATD and Mumtaz source splits are unchanged; 37 applies to target MODMA only.'}
    save(ROOT/'configs/cohort.json',lock)
    cfg=json.loads((ROOT/'configs/b0.json').read_text(encoding='utf-8'))
    cfg.update(experiment='B0-37',cohort_id=lock['cohort_id'],cohort_file='configs/cohort.json',folds_sha256=lock['folds_sha256'])
    cfg['unresolved']=[s for s in cfg['unresolved'] if not s.startswith('MODMA 02010004')]
    save(ROOT/'configs/b0.json',cfg)
    status=json.loads((candidate/'status.json').read_text(encoding='utf-8'))
    status.update(active=True,status='activated_for_baseline_and_all_followup_MODMA_experiments',cohort_id=lock['cohort_id'],active_fold_file=lock['fold_file'])
    save(candidate/'status.json',status)
    recovery=json.loads((ROOT/'recovery/status.json').read_text(encoding='utf-8'))
    recovery.update(baseline_cohort_changed=True,active_cohort_id=lock['cohort_id'],inactive_fallback=None,
                    active_cohort_policy='configs/cohort.json',active_cohort_data_blocker_resolved_by_exclusion=True)
    save(ROOT/'recovery/status.json',recovery)
    print(json.dumps({'cohort_id':lock['cohort_id'],'subjects':37,'recordings':1073,'experiments':ids},indent=2))

if __name__=='__main__': main()

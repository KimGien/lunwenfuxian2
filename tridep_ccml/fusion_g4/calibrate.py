from .common import *
from .numerical import fit_sigmoid_checked as fit_sigmoid

def main():
    cfg=config_check();lock=read(OUT/'inner_models_lock.json');assert lock['passed'] and lock['models']==90
    for p,sha in lock['artifacts'].items():assert digest(OUT/p)==sha,p
    for d in read(OUT/'postprocess_revision_lock.json')['files']:verify(d)
    count=0
    for c in CONDITIONS:
        for seed in SEEDS:
            for fold in FOLDS:
                ids=cfg['outer_folds'][f'fold_{fold}']['train'];rows=[]
                for part in cfg['inner_splits'][str(fold)]:
                    incoming=read(folder(c,seed,fold,part['inner'])/'oof.json')
                    assert [r['subject_id'] for r in incoming]==part['calibration']
                    assert all(r['fit_ids']==part['fit'] for r in incoming)
                    rows+=incoming
                validate_oof(rows,ids)
                params={'threshold':fit_threshold(rows,ids),'sigmoid':fit_sigmoid(rows,ids)}
                save(OUT/f'calibrators/{c}/seed_{seed}/fold_{fold}.json',{'condition':c,'seed':seed,'fold':fold,
                    'train_ids':ids,'oof':rows,'parameters':params,'outer_val_test_used':False,
                    'G1_outer_results_already_seen':True,'new_outer_transform_seen':False},immutable=True)
                count+=1
        print(c+': OOF threshold and sigmoid fitting completed.',flush=True)
    assert count==30
    save(OUT/'postprocessors_lock.json',{'passed':True,'locked_utc':now(),'thresholds':30,'sigmoids':30,
        'G1_outer_results_already_seen':True,'G4_transformed_outer_results_seen':False,
        'artifacts':{p.relative_to(OUT).as_posix():digest(p) for p in sorted((OUT/'calibrators').rglob('*.json'))}},immutable=True)
    save(OUT/'status.json',{'stage':'postprocessors_locked','completed_inner_models':90,'completed_updates':16200,
        'thresholds':30,'sigmoids':30,'G4_outer_transforms_released':False,'updated_utc':now()})

if __name__=='__main__':main()

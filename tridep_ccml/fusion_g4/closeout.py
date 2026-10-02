from .common import *

def main():
    config_check();ver=read(REPORT/'verification.json');assert ver['passed']
    for d in read(OUT/'postprocess_revision_lock.json')['files']:verify(d)
    for name in ['inner_models_lock.json','postprocessors_lock.json','evaluation_complete.json']:
        for p,sha in read(OUT/name)['artifacts'].items():assert digest(OUT/p)==sha,p
    assert history()==read(OUT/'historical_before.json')
    result=read(REPORT/'results.json')
    status={'stage':'G4_complete','completed_utc':now(),'new_inner_models':90,'new_outer_models':0,'reused_outer_models':30,
        'completed_updates':16200,'thresholds':30,'sigmoid_calibrators':30,'EEG_augmentation':False,
        'G1_results_already_seen_at_registration':True,'study_type':'post-release development extension',
        'G1_raw_results_unchanged':True,'best_variant_selected':False,'independent_population_validation':False,
        'comparisons':result['comparisons'],'core_plus_G4_new_neural_models':240,'core_plus_G4_linear_fits':10,
        'core_plus_G4_neural_updates':49500,'report':desc(REPORT/'FUSION-G4后处理开发实验完成报告.md'),
        'verification':desc(REPORT/'verification.json')}
    save(OUT/'completion.json',status,immutable=True);save(OUT/'status.json',status)
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='deliverable_hashes.json' and '__pycache__' not in p.parts]
    files+=list(CODE.glob('*.py'))+[p for p in REPORT.rglob('*') if p.is_file()]+[APPENDIX]
    hashes={os.path.relpath(p,OUT).replace('\\','/'):digest(p) for p in sorted(files)}
    save(OUT/'deliverable_hashes.json',hashes,immutable=True)
    for p,sha in hashes.items():assert digest(OUT/p)==sha,p
    print(f'G4 complete; {len(hashes)} deliverable hashes verified.',flush=True)

if __name__=='__main__':main()

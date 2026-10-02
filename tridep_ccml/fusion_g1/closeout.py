from .common import *

def main():
    lock_check();verification=read(REPORT/'verification.json');assert verification['passed']
    for d in read(OUT/'evaluation_lock.json')['files']:verify(d)
    for f,key in [('release_gate.json','fitted_artifacts'),('evaluation_complete.json','artifacts')]:
        for p,sha in read(OUT/f)[key].items():assert digest(OUT/p)==sha,p
    results=read(REPORT/'results.json');assert sum(r['models'] for r in results['training'])==90
    assert all(r['all_reload_exact'] for r in results['training'])
    assert history()==read(OUT/'historical_before.json')
    status={'stage':'G0_G3_complete','completed_utc':now(),'new_neural_models':90,'completed_updates':21600,
        'outer_results_released':True,'G4_postprocess_enabled':False,'complete_inner_selection':False,
        'eeg_augmentation':False,'new_DFMU_models_in_this_run':0,'decision':results['decision'],
        'EEG_plus_G_core_total_neural_models':150,'EEG_plus_G_core_total_linear_fits':10,
        'EEG_plus_G_core_total_new_fits':160,'EEG_plus_G_core_total_neural_updates':33300,
        'verification':desc(REPORT/'verification.json'),'report':desc(REPORT/'FUSION-G1整体改进主线完成报告.md')}
    save(OUT/'completion.json',status,immutable=True);save(OUT/'status.json',status)
    files=[p for p in OUT.rglob('*') if p.is_file() and p.name!='deliverable_hashes.json' and '__pycache__' not in p.parts]
    files+=list(CODE.glob('*.py'))+[p for p in REPORT.rglob('*') if p.is_file()]+[APPENDIX]
    hashes={os.path.relpath(p,OUT).replace('\\','/'):digest(p) for p in sorted(files)}
    save(OUT/'deliverable_hashes.json',hashes,immutable=True)
    for p,sha in hashes.items():assert digest(OUT/p)==sha,p
    print(f'G0-G3 complete; {len(hashes)} final artifact hashes verified.',flush=True)

if __name__=='__main__':main()

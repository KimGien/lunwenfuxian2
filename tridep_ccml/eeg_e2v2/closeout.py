"""Freeze the completed EEG-only study after all mandatory checks and report QA."""
from pathlib import Path
import os
import re
from .common import *


def run():
    for name in ['engineering_verification.json','evaluation_verification.json','results_verification.json','interface_verification.json']:
        assert read(OUT/name)['passed'],name
    handoff=read(REPORT/'handoff.json');assert handoff['eeg_mainline_completed']
    trained=[read(p) for p in (OUT/'runs').rglob('trained.json')]
    assert len(trained)==60 and sum(r['updates'] for r in trained)==11700
    for record in read(OUT/'input_manifest.json')['records']:
        for item in list(record['source'][k] for k in ['set','fdt','mat','cache'])+[record['new_cache']]:
            assert digest(WORK/item['path'])==item['sha256']
    frozen=history();assert frozen==read(OUT/'historical_before.json')
    save(OUT/'historical_after.json',frozen,immutable=True)
    report=REPORT/'EEG-E2v2主线完成报告.md'
    for p in re.findall(r'\]\(<([^>]+)>\)',report.read_text(encoding='utf-8')):
        if not p.startswith('https://'):assert Path(p).exists(),p
    completed={'state':'EEG_mainline_complete','eeg_mainline_completed':True,'updated_utc':now(),
        'input_engineering_passed':True,'new_neural_fits':60,'new_linear_fits':10,'old_models_replayed':15,
        'neural_updates':11700,'neural_improvement_passed':handoff['neural_improvement_passed'],
        'linear_improvement_passed':handoff['linear_improvement_passed'],
        'standalone_screen_passed':handoff['standalone_screen_passed'],
        'fusion_G_started':False,'augmentation':False,'source_subjects':37,'new_independent_subjects':0,
        'historical_manifest_entries_verified':sum(r['files'] for r in frozen.values()),
        'source_and_new_cache_descriptors_rechecked':185,
        'report':desc(report),'handoff':desc(REPORT/'handoff.json'),
        'evaluation_verification':desc(OUT/'evaluation_verification.json'),
        'results_verification':desc(OUT/'results_verification.json')}
    save(OUT/'status.json',completed);save(OUT/'completion.json',completed);save(REPORT/'status.json',completed)
    files=[p for root in [OUT,REPORT,CODE] for p in root.rglob('*') if p.is_file()
           and p.name!='deliverable_hashes.json' and '__pycache__' not in p.parts]
    files.extend([PLAN,APPENDIX])
    hashes={Path(os.path.relpath(p,OUT)).as_posix():digest(p) for p in sorted(set(files))}
    save(OUT/'deliverable_hashes.json',hashes)
    for p,sha in hashes.items():assert digest(OUT/p)==sha
    save(REPORT/'deliverable_hashes.json',{p.relative_to(REPORT).as_posix():digest(p) for p in REPORT.rglob('*')
        if p.is_file() and p.name!='deliverable_hashes.json'})
    print('EEG_MAINLINE_COMPLETE',len(hashes),'artifact entries; history entries',completed['historical_manifest_entries_verified'],flush=True)


if __name__=='__main__':run()

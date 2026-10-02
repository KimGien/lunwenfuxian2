"""Close the failed engineering version without marking the EEG main line complete."""
from pathlib import Path
import os
import re
from .common import *


def run():
    gate = read(OUT/'synthetic_gate.json')
    verification = read(OUT/'verification.json')
    assert not gate['passed'] and gate['accepted_method'] is None
    assert verification['passed'] and not verification['engineering_passed']
    report = REPORT/'EEG-E2输入边界修复阶段报告.md'
    assert report.exists() and (REPORT/'EEG_E2工程验收.png').exists()
    for link in re.findall(r'\]\(<([^>]+)>\)', report.read_text(encoding='utf-8')):
        if not link.startswith('https:'): assert Path(link).exists(), link
    registration = read(OUT/'engineering_registration.json')
    assert all(digest(WORK/p) == sha for p, sha in registration['protocol']['sources'].items())
    assert all(digest(WORK/c['report']['path']) == c['report']['sha256'] for c in gate['candidates'])
    state = {
        'state': 'E1_engineering_failed_stop_before_training', 'updated_utc': now(),
        'engineering_attempt_closed': True, 'eeg_mainline_completed': False,
        'engineering_passed': False, 'accepted_method': None,
        'E0': 'mechanism_complete_unit_provenance_unresolved',
        'E1_synthetic': 'failed_both_registered_candidates', 'E1_real_new_input': 'not_started',
        'E2_paired_training': 'not_started', 'E3_threshold_diagnosis': 'not_started',
        'E4_full_handoff': 'not_completed_engineering_stop_report_only', 'G_mainline': 'not_started',
        'new_models': 0, 'new_neural_models': 0, 'new_linear_fits': 0, 'new_real_caches': 0,
        'new_f1': None, 'augmentation_used': False, 'third_candidate_run': False,
        'thresholds_relaxed': False, 'next_version_required_before_training': True,
        'report': desc(report), 'gate': desc(OUT/'synthetic_gate.json'),
        'verification': desc(OUT/'verification.json'),
        'stop_reason': 'fallback case 104 last5 NRMSE 0.064962 > 0.2 * 0.240844 = 0.048169',
    }
    save(OUT/'status.json', state)
    save(OUT/'completion.json', state)
    save(REPORT/'status.json', state)
    paths = [p for root in [OUT, REPORT, CODE] for p in root.rglob('*')
             if p.is_file() and p.name != 'deliverable_hashes.json' and '__pycache__' not in p.parts]
    paths.extend([PLAN, WORK/'output/plans/EEG_E2_E0_E1边界修复工程执行附表.md'])
    manifest = {Path(os.path.relpath(p, OUT)).as_posix(): digest(p) for p in sorted(set(paths))}
    save(OUT/'deliverable_hashes.json', manifest)
    for rel, sha in read(OUT/'deliverable_hashes.json').items(): assert digest(OUT/rel) == sha
    save(REPORT/'deliverable_hashes.json', {p.name: digest(p) for p in REPORT.iterdir()
                                            if p.is_file() and p.name != 'deliverable_hashes.json'})
    print(json.dumps({'state': state['state'], 'artifact_entries': len(manifest),
                      'report_links_verified': True, 'eeg_mainline_completed': False}), flush=True)


if __name__ == '__main__': run()

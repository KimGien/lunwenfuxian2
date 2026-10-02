from tridep_ccml.eeg_e2.common import WORK, PLAN, P3, P2, R1, REVISION, read, save, digest, desc, now, np
CODE = WORK/'tridep_ccml/eeg_e2v2'
OUT = WORK/'tridep_ccml/experiments/EEG-E2v2-37-fulltext'
REPORT = WORK/'output/analysis/eeg_e2v2_20260929'
APPENDIX = WORK/'output/plans/EEG_E2v2边界修复与固定训练执行附表.md'


def history():
    from tridep_ccml.eeg_e2.common import freeze_history
    result = freeze_history()
    root = WORK/'tridep_ccml/experiments/EEG-E2-37-fulltext'
    records = read(root/'deliverable_hashes.json')
    for p, sha in records.items():
        assert digest(root/p) == sha, str(root/p)
    result[root.relative_to(WORK).as_posix()] = {'files':len(records),'manifest':desc(root/'deliverable_hashes.json')}
    return result

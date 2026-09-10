"""Offline, non-destructive S00 source/fold/data audit; no ML dependencies."""
import argparse
from collections import Counter
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def audit_folds(folds, expected_count=37, expected_ids=None):
    errors = []
    if set(folds) != {f'fold_{i}' for i in range(1, 6)}:
        errors.append('Expected exactly fold_1 through fold_5')
    universe = set(s for f in folds.values() for s in f['test'])
    test_counts = Counter(s for f in folds.values() for s in f['test'])
    if len(universe) != expected_count or any(n != 1 for n in test_counts.values()):
        errors.append(f'Test subjects must cover exactly {expected_count} unique subjects once')
    if expected_ids is not None and universe != set(expected_ids):
        errors.append('Test cohort does not match the locked subject IDs')
    details = {}
    for name, fold in folds.items():
        for part in ('train', 'val', 'test'):
            if len(fold[part]) != len(set(fold[part])):
                errors.append(f'{name}/{part}: duplicate IDs')
        for a, b in (('train','val'), ('train','test'), ('val','test')):
            overlap = sorted(set(fold[a]) & set(fold[b]))
            if overlap:
                errors.append(f'{name}/{a}-{b}: {overlap}')
        if set().union(*(set(fold[p]) for p in ('train','val','test'))) != universe:
            errors.append(f'{name}: cohort differs from test union')
        details[name] = {p: len(fold[p]) for p in ('train', 'val', 'test')}
    return {'passed': not errors, 'errors': errors, 'subjects': len(universe), 'sizes': details}

class Tables(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows=[]; self.row=None; self.cell=None
    def handle_starttag(self, tag, attrs):
        if tag=='tr': self.row=[]
        if tag in ('td','th'): self.cell=''
    def handle_data(self, data):
        if self.cell is not None: self.cell += data
    def handle_endtag(self, tag):
        if tag in ('td','th') and self.row is not None and self.cell is not None:
            self.row.append(self.cell.strip()); self.cell=None
        if tag=='tr' and self.row is not None:
            self.rows.append(self.row); self.row=None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--modma-root', type=Path)
    args=parser.parse_args()
    source=ROOT/'upstream-dev/data/split_dataset_june/fold_assignments.json'
    official=json.loads(source.read_text(encoding='utf-8'))
    from cohort import validate_active_cohort
    lock=validate_active_cohort(require_subjects=False)
    folds=json.loads((ROOT/lock['fold_file']).read_text(encoding='utf-8'))
    report={'official_json': audit_folds(official, expected_count=38),
            'active_cohort':audit_folds(folds, expected_ids=lock['subject_ids']),
            'cohort_id':lock['cohort_id'],'folds_sha256':lock['folds_sha256']}
    html=ROOT/'sources/companion.html'
    if html.exists():
        import re
        tables=Tables(); tables.feed(html.read_text(encoding='utf-8'))
        webfolds={f'fold_{i}':{} for i in range(1,6)}
        for row in tables.rows:
            if len(row)==6 and row[0].lower() in ('train','val','test'):
                for i,cell in enumerate(row[1:],1):
                    webfolds[f'fold_{i}'][row[0].lower()]=re.findall(r'\b\d{8}\b',cell)
        if all(len(f)==3 for f in webfolds.values()):
            save(ROOT/'sources/website_folds_do_not_use.json',webfolds)
            report['website_table']=audit_folds(webfolds, expected_count=38)
    save(ROOT/'reports/fold_audit.json', report)
    paths=[WORK/'baseline备选/2510.14922v2.pdf',WORK/'output/plans/TRI-DEP_递进实验路线.md',source,ROOT/'configs/b0.json']
    paths += [ROOT/'configs/cohort.json',ROOT/lock['fold_file']]
    paths += sorted((ROOT/'upstream-dev').rglob('*.py'))
    paths += sorted((ROOT/'upstream-dev').rglob('*.yaml'))
    paths += sorted((ROOT/'upstream-dev').rglob('*.pth'))
    if html.exists(): paths.append(html)
    save(ROOT/'manifests/source_sha256.json', {str(p.relative_to(WORK)): {'bytes':p.stat().st_size,'sha256':digest(p)} for p in paths})
    cohort=sorted({s for f in folds.values() for s in f['test']})
    rows=[]
    for sid in cohort:
        eeg=[]; audio=[]
        if args.modma_root:
            eeg=list((args.modma_root/'EEG_128channels_resting_lanzhou_2015').glob(sid+'*.mat'))
            audio=list((args.modma_root/'audio_lanzhou_2015-2'/sid).glob('*.wav'))
        rows.append({'subject_id':sid,'label':None,'label_status':'requires_MODMA_metadata','upstream_prefix_label':int(sid.startswith('0201')),'eeg_files':[str(p.resolve()) for p in eeg],'audio_files':[str(p.resolve()) for p in audio],'raw_files_present':len(eeg)==1 and len(audio)==29})
    save(ROOT/'manifests/expected_subjects.json',rows)
    # Final readiness also depends on waveform, environment and protocol checks.
    # Preserve their report; this source-only audit must not reset later progress.
    data_audit=ROOT/'reports/modma_audit.json'
    data=json.loads(data_audit.read_text()) if data_audit.exists() else {}
    print(json.dumps({'official_folds_valid':report['official_json']['passed'],'website_valid':report.get('website_table',{}).get('passed'),'raw_complete_subjects':data.get('complete_subjects'),'training_ready':False},indent=2))
    return 0 if report['official_json']['passed'] and report['active_cohort']['passed'] else 2

if __name__=='__main__':
    raise SystemExit(main())

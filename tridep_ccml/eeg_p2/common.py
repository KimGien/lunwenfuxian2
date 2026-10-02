from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from tridep_ccml.records import read_json, digest, require

WORK=Path(__file__).resolve().parents[2]
CODE=Path(__file__).resolve().parent
OUT=WORK/'output/analysis/eeg_quality_scale_20260928'
PLAN=WORK/'output/plans/EEG_P2无标签质量与尺度核查固定计划.md'
R1=WORK/'tridep_ccml/experiments/EEG-R1-37-fulltext'
P1=WORK/'tridep_ccml/experiments/EEG-P1-37-fulltext'
REVISION=WORK/'tridep_ccml/reports/ccml_v2_audit/eeg_channel_revision.json'
RULES={'flat_seconds':1.,'rail_plateau_seconds':.02,'window_log_rms_z':6.,
       'channel_scale_high':5.,'channel_scale_low':.1,'sample_robust_z':8.,
       'difference_robust_z':20.,'line_ratio':10.,'high_frequency_fraction':.5,
       'scale_floor':1e-12,'window_seconds':5,'duration_seconds':300}


def now():
    return datetime.now(timezone.utc).isoformat()


def desc(path):
    path=Path(path)
    return {'path':path.relative_to(WORK).as_posix(),'sha256':digest(path)}


def save(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)
    path.write_text(content,encoding='utf-8')


def history_check():
    from tridep_ccml.eeg_p1.common import verify_lock
    verify_lock()
    result={}
    for root in [R1,P1]:
        items=read_json(root/'deliverable_hashes.json')
        for name,sha in items.items():
            require(digest(root/name)==sha,'Frozen history changed: '+str(root/name))
        result[root.name]=len(items)
    return result


def inputs():
    cohort=read_json(WORK/'S00/configs/cohort.json')
    ids=cohort['subject_ids']
    source={r['subject_id']:r for r in read_json(WORK/'S00/manifests/subjects.json')}
    stage={r['subject_id']:r for r in read_json(WORK/'S00/manifests/staged_dataset.json')}
    manifest=read_json(R1/'data_manifest.json')
    cache={r['subject_id']:r for r in manifest['raw_and_tokens']}
    # Diagnostic labels and previous predictions never enter the audit interface.
    records=[]
    for sid in ids:
        records.append({'subject_id':sid,'set':desc(WORK/source[sid]['eeg_set']),
            'fdt':{'path':Path(source[sid]['eeg_fdt']).as_posix(),'sha256':stage[sid]['fdt_sha256']},
            'mat':{'path':(Path('S00')/stage[sid]['eeg_mat']).as_posix(),'sha256':stage[sid]['mat_sha256']},
            'cache':cache[sid]['raw']})
    return records,read_json(WORK/'S00'/cohort['fold_file'])

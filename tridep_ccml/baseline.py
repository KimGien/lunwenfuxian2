"""Normalize the audited TC-00 parent probabilities into the TC record contract."""
from pathlib import Path
import json

from .records import RecordContext, validate_prediction_records
from .metrics import summarize

ROOT = Path(__file__).resolve().parent


def main():
    context = RecordContext()
    rows = [json.loads(line) for line in (ROOT / 'reports/tc00_probabilities.jsonl').read_text(encoding='utf-8').splitlines() if line]
    normalized = []
    for source in rows:
        row = {key: source[key] for key in (
            'experiment', 'cohort_id', 'cohort_sha256', 'folds_sha256', 'subject_id',
            'seed', 'outer_fold', 'split', 'label', 'train_ids', 'parent_checkpoints',
            'training_config_sha256', 'fusion_method', 'm', 'e_con', 'e_cmp', 'e_final',
            'u_final', 'prediction', 'rho', 'delta', 'gamma')}
        row['source_features'] = source['parent_features']
        for key in ('feature_manifest_sha256', 'feature_npz_sha256', 'checkpoint_sha256', 'initialization_sha256'):
            row[key] = None
        row['fusion_weights'] = source['fusion_weights'] or [1 / 3] * 3
        row['p_final'] = [1 - source['p_final'], source['p_final']]
        row['views'] = {modality: {'e': None, 'alpha': None, 'p': source['p'][i], 'u': None}
                        for i, modality in enumerate(('eeg', 'speech', 'text'))}
        normalized.append(row)
    audits = []
    summaries = {}
    for experiment in ('TC-00-weighted-37-fulltext', 'TC-00-mean-37-fulltext'):
        selected = [row for row in normalized if row['experiment'] == experiment]
        audits.append(validate_prediction_records(selected, experiment, context=context))
        summaries[experiment] = summarize(selected)
    path = ROOT / 'reports/tc00_records.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in normalized), encoding='utf-8')
    (ROOT / 'reports/tc00_record_validation.json').write_text(json.dumps(audits, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'reports/tc00_metrics.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(audits, indent=2))


if __name__ == '__main__':
    main()

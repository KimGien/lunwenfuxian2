"""Read-only diagnostics of fixed TC01--04 outputs; no fitting or model selection."""
from copy import deepcopy
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .complete_tc01 import read_rows, write
from .metrics import binary_metrics, summarize
from .records import ROOT, MODALITIES, RecordContext, digest, read_json, require, validate_prediction_records

OUT = ROOT / 'reports/tc01_tc04_analysis'


def main():
    context = RecordContext()
    source_paths = {'TC-01': ROOT/'reports/tc01/predictions.jsonl',
                    'TC-02': ROOT/'reports/tc02/predictions.jsonl',
                    'TC-03-mean': ROOT/'reports/tc03/predictions_mean.jsonl',
                    'TC-03-sum': ROOT/'reports/tc03/predictions_sum.jsonl',
                    'TC-03-ccml': ROOT/'reports/tc03/predictions_ccml.jsonl',
                    'TC-04': ROOT/'reports/tc04/predictions.jsonl'}
    rows = {name: read_rows(path) for name, path in source_paths.items()}
    for name, values in rows.items():
        validate_prediction_records(values, f'{name}-37-fulltext', context=context)
    tests = {name: [r for r in values if r['split'] == 'test'] for name, values in rows.items()}
    matrices = {name: summarize(values) for name, values in rows.items()}
    derived = deepcopy(rows['TC-04'])
    for row in derived:
        p = torch.tensor([row['views'][m]['p'] for m in MODALITIES], dtype=torch.float32).mean(0).tolist()
        row.update(p_final=p, prediction=int(p[1] > .5), u_final=None)
    derived_summary = summarize(derived)
    derived_summary.update(experiment='diagnostic-TC04-fixed-weights-probability-mean',
                           posthoc=True, training_performed=False, registered_main_experiment=False)
    matrices['TC-04-fixed-probability-mean'] = derived_summary
    tests['TC-04-fixed-probability-mean'] = [r for r in derived if r['split'] == 'test']
    split_fit = {}
    for name in ('TC-01', 'TC-02', 'TC-04'):
        split_fit[name] = {}
        for split in ('train', 'test'):
            split_fit[name][split] = {}
            for output in ('fusion', *MODALITIES):
                folded = []
                for seed in (11, 22, 33):
                    for fold in range(1, 6):
                        part = [r for r in rows[name] if (r['seed'], r['outer_fold'], r['split']) == (seed, fold, split)]
                        p = [r['p_final'][1] if output == 'fusion' else r['views'][output]['p'][1] for r in part]
                        folded.append(binary_metrics([r['label'] for r in part], p))
                split_fit[name][split][output] = {k: float(np.mean([r[k] for r in folded])) for k in folded[0] if k != 'n_subjects'}
    evidence = {}
    for name in ('TC-02', 'TC-04'):
        evidence[name] = {}
        for m in MODALITIES:
            entries = tests[name]
            strength = np.array([sum(r['views'][m]['alpha']) for r in entries])
            total = np.array([sum(sum(r['views'][v]['alpha']) for v in MODALITIES) for r in entries])
            error = np.array([int(r['views'][m]['p'][1] > .5) != r['label'] for r in entries])
            u = np.array([r['views'][m]['u'] for r in entries])
            evidence[name][m] = {
                'mean_strength': float(strength.mean()), 'median_strength': float(np.median(strength)),
                'mean_strength_share': float((strength / total).mean()),
                'mean_u_correct': float(u[~error].mean()), 'mean_u_incorrect': float(u[error].mean()),
                'u_error_detection_auc_per_seed': {str(s): float(roc_auc_score(
                    [int(r['views'][m]['p'][1] > .5) != r['label'] for r in entries if r['seed'] == s],
                    [r['views'][m]['u'] for r in entries if r['seed'] == s])) for s in (11, 22, 33)}}
        strengths = np.array([[sum(r['views'][m]['alpha']) for m in MODALITIES] for r in tests[name]])
        evidence[name]['highest_strength_count_ties_fractional'] = {
            m: float(((strengths == strengths.max(1, keepdims=True)) /
                      (strengths == strengths.max(1, keepdims=True)).sum(1, keepdims=True))[:, i].sum())
            for i, m in enumerate(MODALITIES)}
        ev = np.array([[r['views'][m]['e'] for m in MODALITIES] for r in tests[name]])
        minima = ev == ev.min(1, keepdims=True)
        evidence[name]['classwise_minimum_count_ties_fractional'] = {
            str(c): {m: float((minima / minima.sum(1, keepdims=True))[:, i, c].sum()) for i, m in enumerate(MODALITIES)}
            for c in (0, 1)}
    transitions = {}
    for a, b in [('TC-02', 'TC-03-ccml'), ('TC-01', 'TC-04'), ('TC-03-ccml', 'TC-04'),
                 ('TC-04', 'TC-04-fixed-probability-mean')]:
        indexed = [{(r['seed'], r['subject_id']): r for r in tests[name]} for name in (a, b)]
        require(indexed[0].keys() == indexed[1].keys(), 'OOF pairing mismatch')
        pairs = [(indexed[0][key], indexed[1][key]) for key in sorted(indexed[0])]
        value = {}
        for seed in (11, 22, 33):
            part = [(x, y) for x, y in pairs if x['seed'] == seed]
            value[str(seed)] = {
                'correct_to_wrong': sum(x['prediction'] == x['label'] and y['prediction'] != y['label'] for x, y in part),
                'wrong_to_correct': sum(x['prediction'] != x['label'] and y['prediction'] == y['label'] for x, y in part),
                'TP_to_FN': sum(x['label'] == 1 and x['prediction'] == 1 and y['prediction'] == 0 for x, y in part),
                'FN_to_TP': sum(x['label'] == 1 and x['prediction'] == 0 and y['prediction'] == 1 for x, y in part)}
        transitions[f'{a} -> {b}'] = value
    confusion = {}
    for name, values in tests.items():
        confusion[name] = {}
        for seed in (11, 22, 33):
            part = [r for r in values if r['seed'] == seed]
            confusion[name][str(seed)] = {
                label: sum(r['label'] == y and r['prediction'] == p for r in part)
                for label, y, p in (('TN', 0, 0), ('FP', 0, 1), ('FN', 1, 0), ('TP', 1, 1))}
    result = {'purpose': 'posthoc mechanism analysis, no training, no threshold/weight search',
              'independent_subjects': 37, 'seed_subject_records': 111,
              'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in source_paths.values()},
              'analysis_script_sha256': digest(__file__), 'summaries': matrices, 'train_test_fold_means': split_fit,
              'evidence': evidence, 'paired_decision_transitions': transitions, 'confusion_per_seed': confusion,
              'note': '111 repeated seed-subject outputs are not 111 independent people; train folds overlap.'}
    write(OUT/'diagnostics.json', result)
    print(json.dumps({'primary': {n:s['primary_seed_mean_of_fold_means'] for n,s in matrices.items()},
                      'fit': split_fit, 'evidence': evidence, 'transitions': transitions,
                      'confusion': confusion}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

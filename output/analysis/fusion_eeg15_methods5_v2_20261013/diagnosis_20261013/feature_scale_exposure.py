"""Count possible exposure, not causal prediction impact, from the scale audit."""
from pathlib import Path
import hashlib
import json

SOURCE = Path(__file__).with_name('feature_scale_diagnosis.json')
DEST = Path(__file__).with_suffix('.json')
EIDS = ['ST', *[f'E{i:02d}' for i in range(14)], 'E15']
METHODS = ['CCML', 'LMF', 'MMDynamics', 'QMF', 'ECML']

def main():
    assert not DEST.exists(), 'Refuse to overwrite existing diagnostic'
    d = json.loads(SOURCE.read_text('utf-8'))
    out = {'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
           'definitions': {'exposed_fit': 'Its cached validation or test input contains at least one coordinate with abs(z) strictly above the threshold, before any learned gate or model forward. This does NOT establish a change in prediction, metric, checkpoint selection or loss.',
                           'ST': 'S and T pooled by unique (seed,fold), total 15. Each shared ST input is used by all 80 conditions at that seed/fold.',
                           'EEG': 'An EEG cache affects only its 5 methods; union with ST and across validation/test is deduplicated.',
                           'causality': 'Read-only post hoc association; no ablation, forward, refit, or threshold search.'},
           'thresholds': {}}
    for threshold in [100, 1000]:
        role_stats, combined_fits, combined_st = {}, set(), set()
        for role in ['val', 'test']:
            extreme = [r for r in d['top_examples'][role] if abs(r['normalized_value']) > threshold]
            # Ensure the saved examples are complete at this threshold.
            assert len(extreme) == d['global']['all_unique_modality_bundles']['roles'][role]['counts_strictly_above'][str(threshold)]
            st = {(r['seed'], r['fold']) for r in extreme if r['modality'] in ['S', 'T']}
            fits = set()
            for r in extreme:
                eids = EIDS if r['modality'] in ['S', 'T'] else [r['modality']]
                fits.update((e, m, r['seed'], r['fold']) for e in eids for m in METHODS)
            role_stats[role] = {
                'extreme_coordinate_count_unique_modality_bundles': len(extreme),
                'extreme_coordinate_count_from_zero_train_std': sum(r['raw_train_std'] == 0 for r in extreme),
                'ST_unique_seed_fold_count': len(st), 'ST_unique_seed_folds': sorted(st),
                'all_modalities_exposed_formal_fit_count': len(fits),
                'by_method_exposed_fit_count': {m: sum(f[1] == m for f in fits) for m in METHODS},
                'examples': extreme}
            combined_fits.update(fits); combined_st.update(st)
        out['thresholds'][str(threshold)] = {'roles': role_stats,
            'val_or_test_ST_unique_seed_fold_count': len(combined_st),
            'val_or_test_ST_unique_seed_folds': sorted(combined_st),
            'val_or_test_all_modalities_exposed_formal_fit_count': len(combined_fits),
            'val_or_test_by_method_exposed_fit_count': {m: sum(f[1] == m for f in combined_fits) for m in METHODS}}
    out['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    DEST.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({t: {**{k: v for k, v in data.items() if k != 'roles'},
                          'roles': {r: {k: v for k, v in values.items() if k != 'examples'}
                                    for r, values in data['roles'].items()}}
                      for t, data in out['thresholds'].items()}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

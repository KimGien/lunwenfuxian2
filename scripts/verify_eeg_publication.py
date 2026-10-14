"""Verify the public archive and comparison table without private participant data."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def main():
    manifest = read('docs/eeg_upload_manifest_20261014.json')
    for rel, info in manifest['files'].items():
        path = (ROOT / rel).resolve()
        assert path.is_relative_to(ROOT), rel
        assert path.is_file(), rel
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == info['published_sha256'], ('hash mismatch', rel)
    table = read('docs/eeg_tridep_comparison.json')
    eeg = read('output/analysis/eeg_all_candidates_v1_20261008/evaluation.aggregate.json')
    fusion = read('output/analysis/eeg_tridep_fusion_v1_20261013/results.json')
    baseline = read('output/analysis/tridep_stage0_20261004/metrics.aggregate.json')
    assert table['baseline_eeg'] == baseline['eeg']['main_test']
    assert len(table['conditions']) == 25
    assert len({r['condition'] for r in table['conditions']}) == 25
    for view in 'ABCD':
        assert table['baseline_fusion'][view] == fusion['baselines']['original_E_'+view]['primary']
    for row in table['conditions']:
        name, snapshot = row['condition'].split('@')
        assert row['eeg'] == eeg['conditions'][name]['variants'][snapshot]['primary']
        for view in 'ABCD':
            assert row['fusion'][view] == fusion['conditions'][row['condition']][view]['primary']
    assert eeg['formal_fits_verified'] == 360 and eeg['evaluated_checkpoints'] == 435
    assert read('output/analysis/eeg_tridep_fusion_v1_20261013/independent_verification.json')['passed']
    print(json.dumps({'passed': True, 'public_files_verified': len(manifest['files']),
                      'comparison_conditions_verified': 25, 'fusion_protocols': 4,
                      'new_model_fits': 0, 'neural_forwards': 0}))


if __name__ == '__main__':
    main()

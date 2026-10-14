"""CPU strict-loading-only regression for repaired EEGPT/LaBraM provenance."""
from __future__ import annotations

import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))

import torch
from tridep_ccml.eeg_all_candidates import finetune as F
from tridep_ccml.eeg_all_candidates import storage as S


def descriptors(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield {k: value[k] for k in ('path', 'sha256')}
        else:
            for child in value.values():
                yield from descriptors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from descriptors(child)


def main():
    torch.set_num_threads(2)
    result = {'gpu_compute_calls': 0, 'neural_forward_calls': 0, 'optimizer_updates': 0,
              'cohort_data_loaded': False, 'models': {}}
    before = S.desc(Path(F.__file__))
    for method, condition in [('LaBraM', 'LaBraM-native5s-lastblock-finetune'), ('EEGPT', 'EEGPT-lastblock-finetune')]:
        model, audit = F.build_model(condition, device='cpu')
        provenance = {item['path']: item for item in descriptors(audit)}
        assert provenance, 'Regression: empty source/weight provenance'
        for item in provenance.values():
            S.checked(item)
        files = F._implementation_sources(condition, model, [v for v in provenance.values() if v['path'].endswith('.py')])
        for item in files:
            S.checked(item)
        paths = [f['path'] for f in files]
        expected = 'tridep_ccml/tridep_stage1/labram_assets.py' if method == 'LaBraM' else 'tridep_ccml/tridep_eegpt/assets.py'
        assert expected in paths
        assert (any('vendor/labram/modeling_finetune.py' in p for p in paths) if method == 'LaBraM'
                else any(p.endswith('vendor/EEGPT_mcae.py') for p in paths))
        result['models'][method] = {'passed': True, 'provenance_descriptors': list(provenance.values()),
                                   'implementation_sources': files, 'loader_and_vendor_hashed': True,
                                   'selected_encoder_parameters': list(model.selected_names),
                                   'fresh_head': audit['fresh_head'], 'cpu_only': True,
                                   'adapter_source': S.desc(Path(model.adapter.__file__))}
        print('PASS CPU provenance ' + method, flush=True)
        del model
    assert S.desc(Path(F.__file__)) == before
    result['reviewed_finetune_source'] = before
    result['finetune_source_unchanged'] = True
    path = Path(__file__).with_name('independent_provenance_regression.json')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

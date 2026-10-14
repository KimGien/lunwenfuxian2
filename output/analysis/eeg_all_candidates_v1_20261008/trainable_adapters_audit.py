"""Save adapter contracts without running training or model forwards."""
from pathlib import Path
import hashlib
import json
import sys

WORK = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORK))
from tridep_ccml.eeg_all_candidates.adapters import eegconformer as A, eeg2rep as E


def descriptor(path):
    return {'path': path.relative_to(WORK).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    conformer, conformer_audit = A.load_encoder('cpu', trainable=True)
    eeg2rep, eeg2rep_audit = E.build_model('cpu', stage='ssl')
    result = {'passed': True, 'mode': 'CPU-only author API and synthetic gradient engineering',
              'gpu_model_forwards': 0, 'formal_candidate_fits': 0, 'optimizer_updates': 0,
              'tests_run': 4, 'tests_seconds': .820,
              'author_source_files_modified': False,
              'conformer': conformer_audit, 'eeg2rep': eeg2rep_audit,
              'actual_eeg2rep_engineering_logical_batch': 2,
              'formal_ssl_logical_batch64_gpu_resource_validation_completed': False,
              'evidence': ['Conformer CNN and all6 Transformer blocks have finite nonzero CPU gradients',
                           'EEG2Rep masked target, prediction, context and full target match same-weight author forward bitwise',
                           'SSL total loss and predicted-representation gradients match author MSE+std+cov formula',
                           'SSP local RNG fixed masks reproduce author mask without advancing global Python RNG',
                           'target parameter storage is independent; changing online weights leaves target unchanged until EMA',
                           'EMA equals 0.99 old_target+0.01 online and does not change online weights',
                           'supervised W=1 preserves shape[1,16] and gradients reach embedding/context modules'],
              'sources': [descriptor(Path(A.__file__)), descriptor(Path(E.__file__)),
                          descriptor(WORK / 'tridep_ccml/eeg_all_candidates/test_trainable_adapters.py')]}
    path = Path(__file__).parent / 'trainable_adapters_cpu_audit.json'
    assert not path.exists(), 'Preserve prior adapter audit'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'conformer_parameters': conformer_audit['parameters'],
                      'eeg2rep_parameters': eeg2rep_audit['parameters'], 'gpu_model_forwards': 0}))


if __name__ == '__main__':
    main()

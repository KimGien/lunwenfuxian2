"""Build the all-candidates resource inventory from hash-bound downloads."""
from __future__ import annotations

import json
from pathlib import Path

from .adapters._author_support import ASSETS, MODULE, WORK, digest
from .protocol import REPORT

SLUGS = {'CSBrain': 'csbrain', 'REVE': 'reve', 'EEGPT': 'eegpt', 'LUNA': 'luna',
         'CodeBrain': 'codebrain', 'EEGConformer': 'eegconformer', 'BIOT': 'biot',
         'EEG2Rep': 'eeg2rep', 'EEGMamba': 'eegmamba'}
NOTES = {
    'CSBrain': ['models.CSBrain.CSBrain; D200;12 layers;295 tensor keys strictly loaded on CPU.', 'Author Mumtaz data reader divides numeric amplitude by100.', 'Register genuine FCz/CPz region topology; do not rename as Fz/Cz.'],
    'REVE': ['Local Hugging Face wrapper + Base safetensors; D512;22 layers;140 tensor keys.', 'Official brain-bzh/reve-positions coordinate resource downloaded and pinned separately.', 'Strictly reject missing electrode names; preserve exact author coordinate numbers/scale.'],
    'EEGPT': ['Reuse previously accepted full58chs4s checkpoint and strict loader.', '512 dimensions for new small head; old random projection only for paired duration diagnosis.', '256Hz4s1024 samples; channel dictionary verified historically.'],
    'LUNA': ['Base224 tensor keys loaded strictly; Q4*D64=256 features after8 temporal layers.', '256Hz physical5s1280 samples;40point patches; author per-window/channel unbiased std z score.', 'Author MNE standard_1005 nominal head coordinates; actual19 identities retained.', 'Pretrained cross_attn query present; downstream classifier learned_agg absent from formal weights.'],
    'CodeBrain': ['Use Models.SSSM.SSSM EEGSSM, not modeling_finetune.NeuralTransformer or tokenizer.', '8 residual blocks;269 tensor keys strictly loaded; D200;5 one-second200point patches.', 'Author mask.cuda() replaced at runtime by mask.to(h_s.device); source snapshot bytes unchanged.', 'Tokenizer lm_head_t/lm_head_f exist in checkpoint but are excluded from feature forward.'],
    'EEGConformer': ['Author convolution +6 attention layers, defaultD40.', 'Change spatial convolution22 channels to19 and task readout; random initialization is required.', 'No universal pretrained checkpoint is claimed.'],
    'BIOT': ['Author PREST16 checkpoint57 tensor keys strictly loaded; D256.', 'Build exact16 bipolar differences on unstandardized common-reference raw before95%abs normalization.', 'Pretrained10s vs local5s difference disclosed; no missingFz/Cz blocker.', 'Last Transformer block contains its own pre-norms; no extra invented output normalization.'],
    'EEG2Rep': ['No verified universal pretrained checkpoint; independently train eachfold/seed on training subjects.', 'Keep SSP, maskedtarget MSE, predictor variance/covariance and EMA; author defaults recorded in source.', 'Author copy_weight uses storage alias and trainer resets it each epoch: clone initialtarget once, then independent EMA needed and must be declared.', 'SSP samples two subsequences that can overlap; do not silently change mask semantics.'],
    'EEGMamba': ['Official117 tensor keys downloaded;12 Mamba2 layers; D200; requires mamba_ssm/Triton RMSNorm kernels.', 'MixerModel flips hidden_states and residual after each layer; alternating direction, not parallel bidirectional concatenation.', 'quick_example incorrectly references CBraMod; use models.eegmamba.EEGMamba.', 'No public full text acquired; label author-public-implementation adaptation experiment.'],
}


def main():
    manifest = json.loads((ASSETS / 'download_manifest.json').read_text(encoding='utf-8'))
    checks = json.loads((REPORT / 'remaining_adapters_cpu_check.json').read_text(encoding='utf-8'))
    result = {'schema_version': 1, 'audit_date_Asia_Shanghai': '2026-10-08',
              'scope': 'sources, formal weights, metadata, adapter CPU engineering only; no GPU fitting by resource audit',
              'download_manifest': str((ASSETS / 'download_manifest.json').relative_to(WORK)),
              'dependency_directory': str((ASSETS / 'python_dependencies').relative_to(WORK)),
              'default_python': 'S00/.venv/Scripts/python.exe',
              'default_python_version': '3.13.14', 'torch_version': '2.8.0+cu126',
              'gpu_name': 'NVIDIA GeForce RTX4080 Laptop GPU',
              'eegmamba_linux_dependency_status': 'pending WSL environment audit; Windows mamba_ssm absent',
              'raw_input': {'path': 'tridep_ccml/experiments/TriDep-Stage1-EEG-37-fulltext/cache/input_exports.json',
                            'shape_per_subject': [60, 19, 1000], 'sample_rate_hz': 200,
                            'physical_calibration_confirmed': False, 'reference': 'accepted E2v2 unchanged common reference'},
              'candidates': {}}
    for name, slug in SLUGS.items():
        if slug == 'eegpt':
            source = {'repository': 'https://github.com/BINE022/EEGPT', 'revision': 'ea7282f698c406cd423ccbedbe9960853761144c',
                      'path': 'tridep_ccml/tridep_eegpt/vendor/EEGPT_mcae.py', 'sha256': 'cfe3d786e6c351e9678f9f7183cd17dab0dfbb131590ef88e3e4a9efae728a95'}
            weights = [{'path': 'work/models/eegpt_author_bound_20261006/eegpt_mcae_58chs_4s_large4E.ckpt',
                        'sha256': '9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0', 'bytes': 1020602524,
                        'hash_evidence': 'previously accepted strict load binding tridep_eegpt/assets.py'}]
        else:
            bound = manifest['sources'][slug]
            source = {'repository': bound['repository'], 'revision': bound['revision'],
                      'path': str((MODULE / 'vendor' / slug).relative_to(WORK)),
                      'binding': str((MODULE / 'vendor' / slug / 'author_source_binding.json').relative_to(WORK)),
                      'archive_sha256': bound['archive']['sha256'], 'source_files': len(bound['files']), 'source_modified': False}
            weights = [f for f in manifest['weights'].get(slug, {}).get('files', []) if Path(f['path']).suffix in ('.pth', '.pt', '.ckpt', '.safetensors')]
        result['candidates'][name] = {'source': source, 'formal_pretrained_weights': weights,
                                      'weights_expected': slug not in ('eegconformer', 'eeg2rep'),
                                      'notes': NOTES[name], 'formal_fit_status': 'not adjudicated by resource audit'}
        if slug in checks['models']:
            result['candidates'][name]['cpu_adapter_check'] = {'passed': True, 'report': str((REPORT / 'remaining_adapters_cpu_check.json').relative_to(WORK)),
                                                            'adapter_path': str((MODULE / 'adapters' / (slug + '.py')).relative_to(WORK)),
                                                            'adapter_sha256': digest(MODULE / 'adapters' / (slug + '.py'))}
    result['reve_position_resource'] = manifest['weights']['reve_positions']
    result['engineering_accounting_this_audit'] = {'gpu_forward_calls': 0, 'formal_fits': 0,
                                                  'synthetic_cpu_optimizer_steps': sum(v['engineering_optimizer_steps'] for v in checks['models'].values())}
    mamba_runtime = ASSETS / 'eegmamba_linux_runtime.json'
    mamba_cpu = REPORT / 'eegmamba_cpu_strict_load.json'
    mamba_gpu = REPORT / 'eegmamba_official_gpu_check.json'
    if mamba_runtime.exists() and mamba_cpu.exists():
        runtime = json.loads(mamba_runtime.read_text(encoding='utf-8'))
        cpu = json.loads(mamba_cpu.read_text(encoding='utf-8'))
        result['eegmamba_linux_dependency_status'] = 'official Linux runtime ready; CPU strict117keys passed; GPU engineering pending sequential slot'
        result['eegmamba_linux_runtime'] = {**cpu['runtime'], 'distribution': 'TRI-DEP-Ubuntu-24.04',
                                         'environment': runtime['environment'], 'base_environment_modified': False,
                                         'binding': str(mamba_runtime.relative_to(WORK))}
        result['candidates']['EEGMamba']['cpu_adapter_check'] = {'passed': True, 'strict_loaded_keys': 117,
                                                               'adapter_path': str((MODULE / 'adapters/eegmamba.py').relative_to(WORK)),
                                                               'adapter_sha256': digest(MODULE / 'adapters/eegmamba.py'),
                                                               'report': str(mamba_cpu.relative_to(WORK)), 'neural_forward_calls': 0}
    if mamba_gpu.exists():
        gpu = json.loads(mamba_gpu.read_text(encoding='utf-8'))
        if gpu.get('passed'):
            result['eegmamba_linux_dependency_status'] = 'official Linux Mamba2 CUDA forward and gradient qualified'
            result['candidates']['EEGMamba']['gpu_adapter_check'] = {'passed': True, 'report': str(mamba_gpu.relative_to(WORK))}
            result['engineering_accounting_this_audit']['gpu_forward_calls'] = gpu['gpu_encode_calls']
            result['engineering_accounting_this_audit']['synthetic_gpu_optimizer_steps'] = gpu['engineering_optimizer_steps']
    (MODULE / 'candidate_resources.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('Saved candidate_resources.json')


if __name__ == '__main__':
    main()

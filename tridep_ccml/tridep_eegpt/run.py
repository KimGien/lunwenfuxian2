"""Qualify, register and execute the previously deferred matched EEGPT comparison."""
from __future__ import annotations

import argparse
import gc
import inspect
import sys
from pathlib import Path

from .common import *

OLD_OUT = WORK / 'tridep_ccml/experiments/TriDep-Stage6-ST-Generalization-Resume-20261006'
OLD_REPORT = WORK / 'output/analysis/tridep_stage6_20261006'
OLD_STAGE1 = WORK / 'tridep_ccml/experiments/TriDep-Stage1-EEG-37-fulltext'


def qualify():
    require(not (OUT / 'registration.json').exists(), 'Qualification must precede registration')
    require(not (OUT / 'qualification.json').exists(), 'Qualification already saved; preserve it')
    from . import assets
    source = assets.prepare_source()
    encoder, eegpt = assets.load_frozen_eegpt('cpu')
    require(not encoder.training and not any(p.requires_grad for p in encoder.parameters()), 'Encoder not frozen')
    del encoder
    gc.collect()
    from tridep_ccml.tridep_stage1.labram_assets import load_frozen_labram, MANIFEST
    control, labram = load_frozen_labram('cpu')
    require(not control.training and not any(p.requires_grad for p in control.parameters()), 'Control not frozen')
    del control
    gc.collect()
    projection, projection_audit = assets.projection()
    require(projection.shape == (512, 200) and projection.dtype == np.float32 and np.isfinite(projection).all(),
            'Projection contract differs')
    OUT.mkdir(parents=True, exist_ok=True)
    projection_path = OUT / 'projection.npy'
    require(not projection_path.exists(), 'Do not overwrite fixed projection')
    np.save(projection_path, projection, allow_pickle=False)
    lock, folds, labels = context()
    inputs = read(OLD_STAGE1 / 'input_manifest.json')
    require(inputs['subject_ids'] == lock['subject_ids'] and len(inputs['records']) == 37, 'Accepted raw person list differs')
    raw_checks = []
    for record in inputs['records']:
        array = np.load(checked(record['raw']), allow_pickle=False)
        require(array.shape == (60, 19, 1000) and np.isfinite(array).all(), 'Accepted EEG raw array differs')
        raw_checks.append({'subject_id': record['subject_id'], 'raw': record['raw'], 'shape': list(array.shape),
                           'dtype': str(array.dtype), 'finite': True})
    save(OUT / 'qualification.json', {
        'passed': True, 'checked_at': now(), 'new_real_encoder_forward_calls': 0,
        'new_formal_fits': 0, 'new_updates': 0, 'eegpt_loading': eegpt,
        'official_encoder_source': source, 'labram_loading': labram, 'labram_asset_manifest': desc(MANIFEST),
        'projection_audit': projection_audit, 'projection': desc(projection_path),
        'raw_manifest': desc(OLD_STAGE1 / 'input_manifest.json'), 'raw_checks': raw_checks,
        'subject_ids': lock['subject_ids'], 'folds_sha256': lock['folds_sha256'],
        'previous_deferred_decision': desc(WORK / 'output/analysis/tridep_stage4_20261005/eegpt_eligibility.json'),
        'physical_calibration_confirmed': False,
    }, immutable=True)
    print('EEGPT strict loading and accepted 37-person raw inputs qualified;0 real forwards/0 fits', flush=True)


def prepare():
    require(not (OUT / 'registration.json').exists(), 'Already registered; preserve the frozen design')
    require(read(OUT / 'qualification.json')['passed'], 'Formal checkpoint strict loading qualification required')
    require(ANNEX.is_file() and (CODE / 'verify.py').is_file() and (CODE / 'train_eval.py').is_file(), 'Complete all execution sources first')
    from . import assets
    require(assets.verify_checkpoint()['author_file_info_md5'] == assets.CHECKPOINT_AUTHOR_MD5,
            'Author-bound complete checkpoint required')
    from .train_eval import TRAINING as module_training
    require(module_training == TRAINING, 'Paired training implementation and registration differ')
    old_status = read(OLD_REPORT / 'status.json')
    require(old_status['full_generalization_validation_completed'] and old_status['passed'], 'Previous stage6 must remain complete')
    old_inventory = read(OLD_REPORT / 'deliverable_hashes.json')
    actual = {p.relative_to(WORK).as_posix() for root in (OLD_OUT, OLD_REPORT) for p in root.rglob('*')
              if p.is_file() and p != OLD_REPORT / 'deliverable_hashes.json'}
    require(set(old_inventory) == actual and len(old_inventory) == 1347, 'Prior complete inventory differs')
    prior = read(OLD_OUT / 'registration.json')
    sources = dict(prior['sources'])
    for name, sha in old_inventory.items():
        require(name not in sources or sources[name] == sha, 'Conflicting prior artifact')
        sources[name] = sha
    for path in [OLD_OUT / 'registration.json', OLD_REPORT / 'deliverable_hashes.json',
                 OUT / 'qualification.json', OUT / 'projection.npy', OUT / 'pre_registration_test_results.json', ANNEX,
                 OLD_STAGE1 / 'input_manifest.json', WORK / 'output/analysis/tridep_stage4_20261005/eegpt_eligibility.json',
                 assets.CHECKPOINT, assets.CHECKPOINT.parent / 'download_ledger.json',
                 WORK / 'output/analysis/tridep_remaining_20261006/author_file_identity.json',
                 WORK / 'output/analysis/tridep_remaining_20261006/access_probes_network.json',
                 WORK / 'work/eegpt_mirror_download_20261006.py']:
        record = desc(path)
        require(record['path'] not in sources or sources[record['path']] == record['sha256'], 'Conflicting new source')
        sources[record['path']] = record['sha256']
    # Bind the actual numerical resampling implementation and compiled kernel.
    import scipy.signal._signaltools as signaltools
    import scipy.signal._upfirdn as upfirdn
    import scipy.signal._upfirdn_apply as kernel
    for path in [inspect.getfile(signaltools), inspect.getfile(upfirdn), kernel.__file__]:
        record = desc(path)
        sources[record['path']] = record['sha256']
    for path in CODE.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.part':
            record = desc(path)
            sources[record['path']] = record['sha256']
    for name, sha in sources.items():
        checked({'path': name, 'sha256': sha})
    lock, folds, labels = context()
    inputs = read(OLD_STAGE1 / 'input_manifest.json')
    reg = {
        'experiment': OUT.name, 'stage': 4, 'followup': 'previously_deferred_EEGPT_matched4s',
        'registered_at': now(), 'user_instruction_verbatim': '进行下一步',
        'sources': sources, 'cohort_id': lock['cohort_id'], 'subject_ids': lock['subject_ids'],
        'folds_sha256': lock['folds_sha256'], 'seeds': SEEDS, 'folds': [1, 2, 3, 4, 5],
        'conditions': CONDITIONS, 'training': {c: TRAINING for c in CONDITIONS},
        'formal_fits': 30, 'formal_updates': 3600, 'training_gradient_passes': 3600,
        'inner_fits': 0, 'calibrator_fits': 0, 'threshold_searches': 0,
        'fusion_selections': 0, 'early_stopping': False, 'head_selection': 'fixed_final_epoch60',
        'budget': {'features': FEATURE_BUDGET, 'formal_fits': 30, 'formal_updates': 3600,
                   'training_gradient_passes': 3600, 'producer_head_person_forward_calls': 2220,
                   'independent_replay_head_person_forward_calls': 1110,
                   'total_head_person_forward_calls': 3330, 'prediction_records': 1110,
                   'metric_groups': 180, 'new_S_T_forwards': 0, 'new_S_T_fits': 0,
                   'new_EEG_pretraining_or_finetuning': 0, 'new_ASR_calls': 0,
                   'new_fusion_selection_calls': 0, 'new_calibration_or_threshold_searches': 0},
        'features': {'raw_manifest': desc(OLD_STAGE1 / 'input_manifest.json'),
                     'projection': desc(OUT / 'projection.npy'), 'channel_names': inputs['channel_names'],
                     'batch_size': 8, 'repeat_policy': 'each person first8 windows same8 batch; exact output equality',
                     'eegpt_sampling': 'scipy.signal.resample_poly up32 down25 to1280@256Hz then first1024',
                     'labram_sampling': 'same raw5s first800@200Hz', 'eegpt_scale': 'native /1000 mV reader convention',
                     'labram_scale': 'native /100 author convention',
                     'reference': 'unchanged E2v2 Cz/common reference for both; differs from author average reference',
                     'projection_recipe': 'PCG64(20261005) normalfloat64[512,200], reducedQR, Q column sign(diagR>=0), castfloat32, scale1',
                     'pooling': 'EEGPT mean time16 and summary4 before projection; LaBraM mean19x4 patchtokens excludingCLS'},
        'fusion': {'method': 'source-stage0 fixed original EST weights on new EEG and original S/T probabilities',
                   'weight_selection': 'none; reuse each original same-seed/fold EST weights',
                   'fixed_ST': 'reuse original same-seed/fold probabilities and selected originalST weights',
                   'threshold': .5, 'tie_class': 0, 'nll_probability_clip': 1e-15},
        'keeper_rules': {'eeg': 'EEGPT EEG versus matched4s LaBraM EEG: deltaF1>=.01,>=2seeds strictly better,BA>=control,recall_drop<=.05',
                         'fusion': 'All four gates pass versus matched4s LaBraM fixedEST, originalEST and fixedST; probability improvement separately'},
        'physical_calibration_confirmed': False, 'new_independent_people': 0,
        'evaluation_role': '37-person repeatedly observed development comparison; matched4s encoder+interface comparison',
        'stage6_disclosed_14_people_used_for_fitting_or_selection': False,
        'protected_stage6_inventory': desc(OLD_REPORT / 'deliverable_hashes.json'),
        'prior_qualification': desc(WORK / 'output/analysis/tridep_stage4_20261005/eegpt_eligibility.json'),
        'asset_qualification': desc(OUT / 'qualification.json'), 'appendix': desc(ANNEX),
        'test_release': 'Only after all30 fits completed; original val3 diagnosis only; no target selection',
        'model_execution_allowed': True, 'automatic_failed_action_restart': False,
        'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                    'device': 'cuda', 'dtype': 'float32', 'AMP': False,
                    'matmul_tf32': False, 'cudnn_tf32': True, 'deterministic': True},
    }
    save(OUT / 'registration.json', reg, immutable=True)
    save(OUT / 'registration_anchor.json', {'registration': desc(OUT / 'registration.json')}, immutable=True)
    save(REPORT / 'status.json', {'stage': 4, 'followup': reg['followup'], 'status': 'registered_ready_for_features',
                                'registration': desc(OUT / 'registration.json'), 'planned_fits': 30,
                                'planned_updates': 3600, 'fits_completed': 0, 'metrics_released': False})
    print(f'Matched4s EEGPT registered: {len(sources)} sources;30fits/3600updates;no target-driven selection', flush=True)


def action(name):
    reg = registration()
    if name == 'fit_all':
        audit = read(REPORT / 'input_independent_verification.json')
        require(audit['passed'] and audit['registration'] == desc(OUT / 'registration.json'), 'Independent feature acceptance required')
        require(read(OUT / 'actions/features.json')['status'] == 'complete', 'Features action incomplete')
    if name == 'evaluate':
        require(read(OUT / 'actions/fit_all.json')['status'] == 'complete', 'All formal fitting must complete first')
    begin(name)
    try:
        if name == 'features':
            from .extract import run
            run(device='cuda')
        else:
            from . import train_eval
            getattr(train_eval, name)()
        finish(name)
    except BaseException as error:
        failed(name, error)
        raise


def complete():
    reg = registration()
    inventory = REPORT / 'deliverable_hashes.json'
    require(not inventory.exists(), 'Already sealed; use read-only verification')
    audit = read(REPORT / 'independent_verification.json')
    require(audit['passed'] and audit['registration'] == desc(OUT / 'registration.json'), 'Independent full verification required')
    metrics = read(REPORT / 'metrics.json')
    acceptance = read(REPORT / 'acceptance.json')
    matrix = []
    for condition in CONDITIONS:
        for view in ('eeg', 'fixed_EST'):
            values = metrics[condition][view]['main_test']
            matrix.append('| ' + condition + ' / ' + view + ' | ' + ' | '.join(f'{values[k]:.6f}' for k in KEYS) + ' |')
    candidate = acceptance[CONDITIONS[1]]
    eeg_gate = candidate['eeg_vs_matched_control']
    decision = (f"\nEEGPT单EEG相对匹配4秒LaBraM的Macro-F1变化为{eeg_gate['delta']['macro_f1']:+.6f}，"
                f"{eeg_gate['improving_seeds']}/3种子严格改善；"
                f"单EEG分类{'保留' if candidate['eeg_keep'] else '未通过保留门槛'}。"
                f"NLL与Brier{'同时改善' if eeg_gate['probability_improvement'] else '未同时改善'}，独立于分类结论。"
                f"固定EST分类{'保留' if candidate['fusion_keep'] else '未通过保留门槛'}；"
                f"{candidate['w_E_zero_model_groups']}/15组原固定融合权重wE为0。\n")
    if candidate['w_E_zero_model_groups'] >= 14:
        decision += ('固定融合至多一个种子受新EEG影响，无法满足“两种子改善”条件；'
                     '这是原固定权重配置的限制，应与单EEG表示的配对结果分别解释。\n')
    text = '''# TriDep EEGPT匹配4秒备选完成报告

完成：2026-10-06。作者共享页已确认正式原检查点的文件编号、路径、大小及MD5；取得完整镜像后，全文MD5与作者页面一致，完整SHA256与固定镜像版本一致，全部encoder键与形状严格加载。本次解除的是历史权重条件延期，旧阶段4延期记录保留原字节。

两条件为LaBraM-native4s-originalCNN与EEGPT-mean-projection4s-originalCNN，沿用37名反复观察的开发人员、原五折和种子11/22/33。每条件15次正式拟合；合计30次、3600次优化更新和训练梯度计算。全部使用原CNN后端498434参数，Adamax学习率0.005、batch16、weight_decay0、固定60轮；两条件初始化和样本顺序配对，仅train26/27人参与更新，val3只作统一诊断，测试7/8人在全部条件完成后评价。没有早停、内层、探针、校准、阈值搜索或S/T重新训练。

| 条件及视图 | Macro-F1 | BA | MDD召回 | AUC | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|
''' + '\n'.join(matrix) + '\n' + decision + '''

主指标先按每个种子平均5个外测试折，再平均三个种子；合并37人的指标另列。111条重复测试记录仍来自37人。固定EST直接沿用阶段0每种子每折原EST权重及原S/T概率，新增选权为0；原权重若不给EEG贡献，须按原样解释，不能替换成事后权重获得更高成绩。旧5秒LaBraM仅为历史诊断，不充当匹配4秒控制。

EEGPT每个5秒1000点窗先正确重采样为256Hz的1280点，再取前1024点；LaBraM从同一窗取原200Hz前800点，两者均为同一物理4秒。19导采用已核验映射，不补造导联。EEGPT在原固定encoder上先平均16时间与4summary token，再采用事前固定PCG64种子20261005、CPU float64 QR及列符号约定的512×200正交投影；LaBraM平均19×4预训练patch token得到原生200维。对比结论属于encoder与固定接口组合，不能单独归因于纯骨干。单位仍为reader约定，设备物理校准未证明；两条件保留E2v2参考，均与EEGPT预训练平均参考有差异。

每骨干首次处理2220窗，另对每人前8窗整批重放共296窗，合计2516窗、333批。原S/T新前向为0。分类头主运行及完整重载共2220人次，独立重放1110人次，总3330；特征及分类权重前后状态一致。逐人预测1110条，两视图180组指标，独立人数仍37。

分类保留与概率改善按登记门槛分别报告，见[验收](acceptance.json)。阴性结果、概率收益及人员配对均保留；完成登记条件后不追加投影种子、池化、单位倍数、窗长、训练轮数或阈值。本次没有使用阶段6已公布的14人结果拟合或选择；此前14人的确认评价及所有0—6阶段封存产物均保留。本次开发比较不提供新的独立受试者或跨采集域泛化证据。

源权重SHA256：`9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0`；作者页面MD5：`a165305f329f51b57691c8a3131f9b79`。作者没有公开SHA256，MD5用于官方完整文件身份绑定，本地完整SHA256另作来源锁定。镜像为第三方发布，未将其称作者授权镜像。

[登记](../../../tridep_ccml/experiments/TriDep-Stage4-EEGPT-Matched4s-37-fulltext-20261006/registration.json) · [执行附表](../../plans/TriDep_EEGPT匹配4秒备选执行附表_v1_20261006.md) · [指标](metrics.json) · [独立核验](independent_verification.json) · [状态](status.json) · [完整交付清单](deliverable_hashes.json)
'''
    report = REPORT / 'TriDep_EEGPT匹配4秒备选完成报告.md'
    require(not report.exists(), 'Do not overwrite completed report')
    report.write_text(text, encoding='utf-8')
    save(REPORT / 'status.json', {'stage': 4, 'followup': reg['followup'], 'status': 'complete', 'passed': True,
         'formal_fits': 30, 'formal_updates': 3600, 'training_gradient_passes': 3600,
         'registration': desc(OUT / 'registration.json'), 'report': desc(report),
         'independent_verification': desc(REPORT / 'independent_verification.json'),
         'budget': reg['budget'], 'acceptance': desc(REPORT / 'acceptance.json'),
         'new_independent_people': 0, 'stage6_14_people_reused_for_selection': False,
         'original_stage4_outputs_modified': False, 'next_phase_started': False})
    files = sorted(p for base in (OUT, REPORT) for p in base.rglob('*') if p.is_file() and p != inventory)
    save(inventory, {p.relative_to(WORK).as_posix(): digest(p) for p in files}, immutable=True)
    print(f'Matched4s EEGPT complete and sealed: {len(files)} files', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('qualify', 'prepare', 'features', 'fit_all', 'evaluate', 'complete'))
    args = parser.parse_args()
    if args.action in ('features', 'fit_all', 'evaluate'):
        action(args.action)
    else:
        {'qualify': qualify, 'prepare': prepare, 'complete': complete}[args.action]()

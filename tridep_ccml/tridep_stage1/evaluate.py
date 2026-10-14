"""Release all registered EEG conditions together after complete paired fitting."""
from __future__ import annotations

from collections import Counter
import gc
from pathlib import Path
import sys

from .common import *
from .run import verify_registration, load_arrays, model_for, input_key

sys.path.insert(0, str(WORK / 'S00/scripts'))
from metrics import select_weights
from tridep_ccml.stage0 import binary_metrics

VIEWS = {'eeg': 'p_eeg', 'est_weighted': 'p_est_weighted', 'est_equal': 'p_est_equal'}
KEYS = ['macro_f1', 'balanced_accuracy', 'mdd_recall', 'auc', 'nll', 'brier']


def direct_control(condition):
    return 'CB-originalCNN' if condition.endswith('originalCNN') else 'CB-mean'


def group_metrics(rr, key):
    return binary_metrics([r['label'] for r in rr], [r[key] for r in rr])


def summarize(rr, folds):
    summaries = {}
    for condition in CONDITIONS:
        selected = [r for r in rr if r['condition'] == condition]
        if len(selected) != 555:
            raise ValueError('Each condition requires 555 full split records')
        summaries[condition] = {}
        for view, key in VIEWS.items():
            fm, seed_means, pooled = [], {}, {}
            for seed in SEEDS:
                for fold in range(1, 6):
                    for part in ['train', 'val', 'test']:
                        group = [r for r in selected if (r['seed'], r['outer_fold'], r['split']) == (seed, fold, part)]
                        ids = folds[f'fold_{fold}'][part]
                        if len(group) != len(ids) or {r['subject_id'] for r in group} != set(ids):
                            raise ValueError('Prediction split isolation/coverage failure')
                        fm.append({'seed': seed, 'fold': fold, 'split': part, **group_metrics(group, key)})
                seed_means[str(seed)] = {part: {k: float(np.mean([r[k] for r in fm if r['seed'] == seed and r['split'] == part])) for k in KEYS}
                                         for part in ['train', 'val', 'test']}
                group = [r for r in selected if r['seed'] == seed and r['split'] == 'test']
                if len(group) != 37 or len({r['subject_id'] for r in group}) != 37:
                    raise ValueError('Every seed must test all37 people once')
                pooled[str(seed)] = group_metrics(group, key)
            summaries[condition][view] = {'main_test': {k: float(np.mean([v['test'][k] for v in seed_means.values()])) for k in KEYS},
                                          'seed_means': seed_means, 'fold_metrics': fm, 'pooled_oof': pooled}
    return summaries


@torch.inference_mode()
def predict(model, x, condition, ids):
    model.eval()
    before = state_digest(model.state_dict())
    ps, hs = [], []
    captured = []
    if condition.endswith('originalCNN'):
        hook = model.classifier[-1].register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
    else:
        hook = None
    for i, sid in enumerate(ids):
        if hook is None:
            logits, h = model(x[i:i+1], return_features=True)
        else:
            logits = model(x[i:i+1])
            h = captured.pop()
        p = float(torch.softmax(logits, -1)[0, 1].cpu())
        if not torch.isfinite(logits).all() or h.shape != (1, 128) or not torch.isfinite(h).all():
            raise ValueError('Invalid trained EEG output interface')
        ps.append(p)
        hs.append(h.cpu().numpy()[0])
    if hook is not None:
        hook.remove()
    if before != state_digest(model.state_dict()):
        raise ValueError('Evaluation changed trained head state/buffers')
    return np.array(ps, dtype=np.float64), np.array(hs, dtype=np.float32)


def gate(candidate, control, candidate_seeds, control_seeds):
    delta = {k: candidate[k] - control[k] for k in KEYS}
    improving = sum(candidate_seeds[str(s)]['test']['macro_f1'] > control_seeds[str(s)]['test']['macro_f1'] for s in SEEDS)
    checks = {'f1_delta_at_least_0_01': delta['macro_f1'] >= .01,
              'at_least2_seeds_strictly_improve': improving >= 2,
              'BA_not_lower': delta['balanced_accuracy'] >= 0.,
              'MDD_recall_drop_at_most0_05': delta['mdd_recall'] >= -.05}
    return {'passed': all(checks.values()), 'checks': checks, 'improving_seeds': improving,
            'delta': delta, 'probability_improvement': delta['nll'] < 0 and delta['brier'] < 0}


def evaluate():
    cfg = verify_registration()
    ledger = read(OUT / 'training_ledger.json')
    if ledger['fits_completed'] != 90 or ledger['updates_completed'] != 18000:
        raise ValueError('Do not release metrics before completing all90 fits')
    lock, folds, labels = context()
    ids = lock['subject_ids']
    # Confirm every formal fit and paired initialization/order before the first
    # validation/test forward, independently of the aggregate ledger counters.
    pairing, seen = {}, set()
    for seed in SEEDS:
        for fold in range(1, 6):
            split = folds[f'fold_{fold}']
            for condition in CONDITIONS:
                folder = OUT / f'runs/{condition}/seed_{seed}/fold_{fold}'
                for item in read(folder / 'complete.json')['artifacts']:
                    checked(item)
                meta = read(folder / 'metadata.json')
                expected = {'condition': condition, 'seed': seed, 'fold': fold, 'train_ids': split['train'],
                            'val_ids': split['val'], 'test_ids': split['test'], 'training': cfg['training'][condition],
                            'registration': desc(OUT / 'registration.json'), 'folds_sha256': lock['folds_sha256'],
                            'updates': cfg['training'][condition]['updates_per_model'],
                            'no_val_test_forward_during_training': True}
                if any(meta.get(k) != v for k, v in expected.items()):
                    raise ValueError('Formal-fit pre-release provenance mismatch')
                key = (seed, fold, 'cnn' if condition.endswith('originalCNN') else 'token')
                pair = (meta['initial_state_sha256'], meta['batch_order_sha256'])
                if key in pairing and pairing[key] != pair:
                    raise ValueError('Pre-release initialization/batch-order pairing failed')
                pairing[key] = pair
                seen.add((condition, seed, fold))
    if len(seen) != 90:
        raise ValueError('All90 registered fit identities must be complete')
    original = {(r['seed'], r['outer_fold'], r['subject_id']): r for r in rows(STAGE0 / 'branch_probabilities.jsonl')}
    baselines = {(r['condition'], r['seed'], r['outer_fold'], r['subject_id']): r for r in rows(STAGE0 / 'control_predictions.jsonl')}
    original_metrics = read(STAGE0 / 'metrics.json')
    arrays = load_arrays()
    adjacency = np.load(checked(read(OUT / 'input_manifest.json')['adjacency']), allow_pickle=False)
    predictions, selections, reload_checks = [], [], []
    for seed in SEEDS:
        for fold in range(1, 6):
            split = folds[f'fold_{fold}']
            for condition in CONDITIONS:
                runtime(seed)
                folder = OUT / f'runs/{condition}/seed_{seed}/fold_{fold}'
                complete = read(folder / 'complete.json')
                for d in complete['artifacts']:
                    checked(d)
                saved = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)
                meta = read(folder / 'metadata.json')
                expected = {'condition': condition, 'seed': seed, 'fold': fold, 'train_ids': split['train'],
                            'val_ids': split['val'], 'test_ids': split['test'], 'training': cfg['training'][condition],
                            'registration': desc(OUT / 'registration.json'), 'folds_sha256': lock['folds_sha256'],
                            'updates': cfg['training'][condition]['updates_per_model'],
                            'no_val_test_forward_during_training': True}
                if saved['metadata'] != meta or any(meta.get(k) != v for k, v in expected.items()):
                    raise ValueError('Trained checkpoint provenance differs from the registered fold')
                net = model_for(condition, adjacency).to('cuda')
                net.load_state_dict(saved['state_dict'], strict=True)
                net.requires_grad_(False).eval()
                if state_digest(net.state_dict()) != meta['final_state_sha256']:
                    raise ValueError('Trained checkpoint tensor hash differs')
                p, h = predict(net, arrays[input_key(condition)], condition, ids)
                del net
                net = model_for(condition, adjacency).to('cuda')
                net.load_state_dict(torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=True)['state_dict'], strict=True)
                net.requires_grad_(False).eval()
                repeated_p, repeated_h = predict(net, arrays[input_key(condition)], condition, ids)
                if not np.array_equal(p, repeated_p) or not np.array_equal(h, repeated_h):
                    raise ValueError('Checkpoint reload does not exactly replay all37 outputs')
                del net
                checkpoint = desc(folder / 'checkpoint.pt')
                target = OUT / f'evaluation/{condition}/seed_{seed}/fold_{fold}.npz'
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, features=h, p_eeg=p, subject_ids=np.array(ids), labels=np.array([labels[s] for s in ids]),
                                    splits=np.array([next(k for k in split if s in split[k]) for s in ids]))
                save(target.with_suffix('.json'), {'checkpoint': checkpoint, 'bundle': desc(target), 'representation_dim': 128,
                                                   'fixed_ST_source': desc(STAGE0 / 'branch_probabilities.jsonl'), 'registration': desc(OUT / 'registration.json')})
                fold_rows = []
                for i, sid in enumerate(ids):
                    source = original[(seed, fold, sid)]
                    part = next(k for k in ['train', 'val', 'test'] if sid in split[k])
                    if source['split'] != part or source['label'] != labels[sid]:
                        raise ValueError('Fixed ST subject/label/split mismatch')
                    fold_rows.append({'condition': condition, 'seed': seed, 'outer_fold': fold, 'subject_id': sid,
                                      'split': part, 'label': labels[sid], 'cohort_id': lock['cohort_id'], 'folds_sha256': lock['folds_sha256'],
                                      'p_eeg': float(p[i]), 'p_speech': source['p_speech'], 'p_text': source['p_text'],
                                      'eeg_checkpoint': checkpoint, 'fixed_ST_checkpoints': {m: source['parent_checkpoints'][m] for m in ['speech', 'text']},
                                      'fixed_ST_probability_file': desc(STAGE0 / 'branch_probabilities.jsonl'),
                                      'registration': desc(OUT / 'registration.json')})
                selection = select_weights([r for r in fold_rows if r['split'] == 'val'])
                selections.append({'condition': condition, 'seed': seed, 'fold': fold, 'val_ids': split['val'], **selection})
                for r in fold_rows:
                    r['fusion_weights'] = selection['weights']
                    r['p_est_weighted'] = sum(w * r['p_' + m] for w, m in zip(selection['weights'], MODALITIES))
                    r['p_est_equal'] = sum((1 / 3) * r['p_' + m] for m in MODALITIES)
                predictions.extend(fold_rows)
                reload_checks.append({'condition': condition, 'seed': seed, 'fold': fold, 'checkpoint': checkpoint,
                                      'exact_probabilities': True, 'exact_representations': True,
                                      'frozen_eval_state_unchanged': True})
                print(f'RELOAD VERIFIED {condition} seed={seed} fold={fold}: all37 probabilities/128d outputs exact', flush=True)
    del arrays
    gc.collect()
    torch.cuda.empty_cache()
    summaries = summarize(predictions, folds)
    paired = []
    lookup = {(r['condition'], r['seed'], r['outer_fold'], r['subject_id']): r for r in predictions}
    for r in predictions:
        if r['split'] != 'test':
            continue
        references = {'fixed-ST': baselines[('ST-val-weighted', r['seed'], r['outer_fold'], r['subject_id'])]['p_final'],
                      'original-EST': baselines[('EST-original-val-weighted', r['seed'], r['outer_fold'], r['subject_id'])]['p_final'],
                      'direct-est-weighted': lookup[(direct_control(r['condition']), r['seed'], r['outer_fold'], r['subject_id'])]['p_est_weighted']}
        candidate_pred = int(r['p_est_weighted'] > .5)
        for name, p_ref in references.items():
            reference_pred = int(p_ref > .5)
            candidate_ok, reference_ok = candidate_pred == r['label'], reference_pred == r['label']
            change = 'add_correct' if candidate_ok and not reference_ok else 'add_error' if not candidate_ok and reference_ok else 'unchanged_correct' if candidate_ok else 'unchanged_error'
            paired.append({k: r[k] for k in ['condition', 'seed', 'outer_fold', 'subject_id', 'label']} | {
                'reference': name, 'p_candidate': r['p_est_weighted'], 'p_reference': p_ref,
                'prediction_candidate': candidate_pred, 'prediction_reference': reference_pred, 'change': change})
    gates = {}
    baseline = original_metrics['EST-original-val-weighted']
    for c in CONDITIONS:
        d = direct_control(c)
        eeg_gate = gate(summaries[c]['eeg']['main_test'], summaries[d]['eeg']['main_test'], summaries[c]['eeg']['seed_means'], summaries[d]['eeg']['seed_means'])
        est_gate = gate(summaries[c]['est_weighted']['main_test'], summaries[d]['est_weighted']['main_test'], summaries[c]['est_weighted']['seed_means'], summaries[d]['est_weighted']['seed_means'])
        original_gate = gate(summaries[c]['est_weighted']['main_test'], baseline['main_test'], summaries[c]['est_weighted']['seed_means'], baseline['train_val_test_seed_means'])
        st = original_metrics['ST-val-weighted']['main_test']
        gates[c] = {'direct_control': d, 'eeg_vs_direct_control': eeg_gate, 'est_vs_direct_control': est_gate,
                    'est_vs_original_EST': original_gate,
                    'est_minus_fixed_ST': {k: summaries[c]['est_weighted']['main_test'][k] - st[k] for k in KEYS},
                    'EEG_structure_or_backbone_value_supported': eeg_gate['passed'],
                    'fusion_classification_keep': est_gate['passed'] and original_gate['passed'],
                    'accepted_combination': False, 'next_phase_started': False}
    save_rows(REPORT / 'prediction_rows.jsonl', predictions)
    save_rows(REPORT / 'paired_changes.jsonl', paired)
    save(REPORT / 'metrics.json', summaries)
    save(REPORT / 'fusion_selections.json', selections)
    save(REPORT / 'reload_validation.json', {'passed': len(reload_checks) == 90, 'checks': reload_checks})
    save(REPORT / 'acceptance.json', gates)
    save(REPORT / 'fixed_controls.json', {'stage0': desc(STAGE0 / 'metrics.json'),
                                        'ST': original_metrics['ST-val-weighted'], 'EST': baseline})
    save(REPORT / 'paired_summary.json', {c: {ref: dict(Counter(r['change'] for r in paired if r['condition'] == c and r['reference'] == ref))
                                            for ref in ['fixed-ST', 'original-EST', 'direct-est-weighted']} for c in CONDITIONS})
    verify_registration()
    save(REPORT / 'status.json', {'status': 'uniform_evaluation_complete_pending_independent_verification', 'stage': 1,
                                'fits_completed': 90, 'updates_completed': 18000, 'metrics_released': True,
                                'prediction_records': len(predictions), 'reload_verified_models': len(reload_checks),
                                'registration': desc(OUT / 'registration.json'), 'next_phase_started': False})
    print('All6 conditions evaluated uniformly; independent verification is pending', flush=True)


def complete_report():
    verification = read(REPORT / 'independent_verification.json')
    if not verification['passed']:
        raise ValueError('Independent verification must pass before closeout')
    cfg = verify_registration()
    summaries = read(REPORT / 'metrics.json')
    gates = read(REPORT / 'acceptance.json')
    paired = read(REPORT / 'paired_summary.json')
    preflight = read(OUT / 'preflight.json')
    inputs = read(OUT / 'input_manifest.json')
    labram = read(OUT / 'labram_loading_and_export.json')
    ledger = read(OUT / 'training_ledger.json')
    baseline = read(REPORT / 'fixed_controls.json')
    # Recheck original raw/S/T/cache/phase0 evidence once more after all EEG work.
    from tridep_ccml.stage0_verify import run as verify_phase0
    verify_phase0(True)
    lines = ['# TriDep 阶段1：EEG结构与骨干完成报告', '',
             '完成日期：2026-10-05。六条件×三种子×五外折全部完成：90次正式拟合、18,000次优化更新。原S/T、原ST与原EST继续使用阶段0的权重和精确输出；新增内层训练、校准、阈值搜索均为0。', '',
             '## 同版本输入与直接控制', '',
             '本轮六条件统一采用E2v2纠正导联并修复边界后的raw，19导、200Hz、60个5秒窗。重新提取CBraMod token及LaBraM表示，未将R1 token当作E2v2缓存。原S00输入、旧修订EEG控制及后期CCML/G1的成绩没有冒充本轮同输入控制。', '',
             'CB-mean/spatial/temporal/spatiotemporal共享同token来源及12,098参数骨架，仅切换原token导联/时间轴的提前均值广播。全部同初始化、批次顺序、60轮与240次更新，受试者级损失。参数数量相同，不宣称有效函数容量相同。', '',
             'CB-originalCNN与LaBraM-originalCNN共享498,434参数原EEGConvPoolClassifier、同初始化/批次顺序、原lr0.005/Adamax/batch16/60轮与120次更新。该配对用于骨干比较，不能把它与不同容量的小头之差全归因于骨干。', '',
             'LaBraM作者正式base权重221/221保留encoder参数项严格加载，冻结0可训练参数。使用预训练student.norm，无随机fc_norm；输入按作者µV/100、200点patch入口，19×5 patch均值排除CLS，保留每人60×200接口。正式权重来源及SHA见资产清单。', '',
             '公开权重与作者仓库一致，不等于预训练受试者身份与潜在人员重叠已经完全核实；这一溯源不确定性继续保留。', '',
             '空间图使用固定standard_1020近似位置的3最近邻并集和自环，不使用有冲突的GSN129坐标。原SET缺少物理单位/采集校准信息；native按EEGLAB微伏读取约定解释，无法由reader×1e-6等式证明真实物理校准或预训练尺度完全兼容。此限制保留，没有据外层成绩搜索幅值缩放。', '',
             '## 统一结果', '',
             '主口径为种子内五外折均值，再均值三种子；p>0.5为MDD，相等HC。各折/种子指标、37人OOF合并值、训练验证测试差距保存在metrics.json。原3人val用于每组231候选的原概率选权；已历史使用且人数极少，全部外层成绩仍为开发证据。全部90模型完成后才统一释放指标。', '',
             f'固定ST/原EST主Macro-F1分别为{baseline["ST"]["main_test"]["macro_f1"]:.6f}/{baseline["EST"]["main_test"]["macro_f1"]:.6f}。', '',
             '| 条件 | EEG F1 | 选权EST F1 | 等权EST F1 | 选权EST BA | MDD召回 | 对直接EST控制ΔF1 | 对固定ST ΔF1 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        m = summaries[c]['est_weighted']['main_test']
        lines.append(f'| {c} | {summaries[c]["eeg"]["main_test"]["macro_f1"]:.6f} | {m["macro_f1"]:.6f} | {summaries[c]["est_equal"]["main_test"]["macro_f1"]:.6f} | {m["balanced_accuracy"]:.6f} | {m["mdd_recall"]:.6f} | {gates[c]["est_vs_direct_control"]["delta"]["macro_f1"]:+.6f} | {gates[c]["est_minus_fixed_ST"]["macro_f1"]:+.6f} |')
    lines += ['', '| 条件 | EEG直接门槛 | EST直接门槛 | EST原基线门槛 | 概率对直接控制同时改善 | ST→新EST新增正确/错误 |', '|---|---|---|---|---|---|']
    for c in CONDITIONS:
        g = gates[c]
        counts = paired[c]['fixed-ST']
        lines.append(f'| {c} | {"通过" if g["eeg_vs_direct_control"]["passed"] else "未通过"} | {"通过" if g["est_vs_direct_control"]["passed"] else "未通过"} | {"通过" if g["est_vs_original_EST"]["passed"] else "未通过"} | {"是" if g["est_vs_direct_control"]["probability_improvement"] else "否"} | {counts.get("add_correct",0)}/{counts.get("add_error",0)} |')
    lines += ['', '各门槛严格采用ΔF1≥0.01、至少2种子严格提高、BA不下降、MDD召回下降不超过0.05。EEG单分支与融合单独判定，NLL/Brier同时下降仅作概率结论，不代替分类保留。每组111条重复种子测试记录只代表37人，不当作111名独立受试者。', '',
              '## 核验与停止', '',
              '90检查点重新实例化后，37人概率及128维表示全部逐位一致，eval期间state与统计量保持不变。全部原S/T概率与阶段0精确一致；组内初始化及批次顺序哈希一致。所有条件使用同人群分折和各自train人员，没有将窗口随机分进测试。', '',
              f'正式更新账本为{ledger["updates_completed"]}，内层拟合/校准/探针优化更新均为0；准备只对训练输入作有限梯度及显存检查。固定参数和原3人val选权不构造训练内合法OOF。原数据、S/T与阶段0产物的哈希在结束后复核通过。', '',
              '本阶段登记条件全部完成后结束，不追加搜索、阈值、种子或未登记微调，不自动替换原系统或启动后续阶段。当前37人仍为多次开发人群，没有新增独立外部验证或口音/语言能力证据。', '',
              '## 交付入口', '',
              '- [执行附表](../../plans/TriDep_阶段1_EEG结构与骨干执行附表_v1_20261005.md)',
              '- [全部指标](metrics.json)', '- [逐人概率](prediction_rows.jsonl)', '- [逐人配对变化](paired_changes.jsonl)',
              '- [各项验收](acceptance.json)', '- [独立核验](independent_verification.json)', '- [完成状态](status.json)',
              '- [固定登记](../../../tridep_ccml/experiments/TriDep-Stage1-EEG-37-fulltext/registration.json)',
              '- [训练账本](../../../tridep_ccml/experiments/TriDep-Stage1-EEG-37-fulltext/training_ledger.json)',
              '- [LaBraM正式资产来源](../../../tridep_ccml/tridep_stage1/assets/labram_asset_manifest.json)', '',
              '重核入口：`S00/.venv/Scripts/python.exe -B -m tridep_ccml.tridep_stage1.verify`。', '']
    target = REPORT / 'TriDep_阶段1_EEG结构与骨干完成报告.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    status = {'status': 'complete', 'passed': True, 'stage': 1, 'completed_at': now(),
              'formal_fits': 90, 'formal_updates': 18000, 'inner_fits': 0, 'calibrator_fits': 0,
              'threshold_searches': 0, 'new_independent_subjects': 0, 'next_phase_started': False,
              'registration': desc(OUT / 'registration.json'), 'report': desc(target),
              'independent_verification': desc(REPORT / 'independent_verification.json'),
              'reload_verified_models': 90, 'prediction_records': 3330,
              'EEG_gate_passed': [c for c in CONDITIONS if gates[c]['eeg_vs_direct_control']['passed']],
              'fusion_keep_gate_passed': [c for c in CONDITIONS if gates[c]['fusion_classification_keep']],
              'source_interpretation': 'same37 previously_observed_development; originalST fixed; EEG input version E2v2'}
    save(REPORT / 'status.json', status)
    protected = [p for p in OUT.rglob('*') if p.is_file() and p.suffix != '.part'] + [p for p in REPORT.rglob('*') if p.is_file() and p.name != 'deliverable_hashes.json']
    save(REPORT / 'deliverable_hashes.json', {desc(p)['path']: digest(p) for p in sorted(protected)})
    print(json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['evaluate', 'complete'], default='evaluate', nargs='?')
    evaluate() if parser.parse_args().action == 'evaluate' else complete_report()

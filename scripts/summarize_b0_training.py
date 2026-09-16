"""Independently audit saved predictions/checkpoints and summarize the fixed B0 run."""
from datetime import datetime,timezone
import json
import math
import sys
import numpy as np
from prepare import ROOT,save,digest
from cohort import validate_active_cohort,validate_prediction_records
from metrics import metrics,select_weights

def main():
    lock=validate_active_cohort(require_data=True)
    folds=json.loads((ROOT/'manifests/fold_assignments.json').read_text())
    cfg=json.loads((ROOT/'configs/b0.json').read_text())
    all_seeds={};checkpoint_files=[]
    measures=('macro_f1','mdd_recall','balanced_accuracy')
    for seed in (11,22,33):
        folder=ROOT/'runs/B0-37'/f'seed_{seed}'
        run=json.loads((folder/'run.json').read_text())
        rows=[json.loads(line) for line in (folder/'probabilities.jsonl').read_text().splitlines() if line]
        validate_prediction_records(rows,'B0-37')
        if len(rows)!=185 or len(run['folds'])!=5:raise ValueError('Incomplete training run')
        if run['config_sha256']!=digest(folder/'configuration.json'):raise ValueError('Saved configuration mismatch')
        if run['feature_manifest_sha256']!=digest(ROOT/'features/B0-37/features_manifest.json'):raise ValueError('Feature manifest changed')
        if run['training_script_sha256']!=digest(ROOT/'scripts/run_b0.py'):raise ValueError('Training script changed')
        fold_stats={}
        for name,f in folds.items():
            index=int(name[-1]); selected=[r for r in rows if r['outer_fold']==index]
            if len(selected)!=37:raise ValueError('Fold does not cover37 subjects')
            val=[r for r in selected if r['split']=='val'];test=[r for r in selected if r['split']=='test']
            fusion=select_weights(val)
            for r in selected:
                expected=sum(w*r['p_'+m] for w,m in zip(fusion['weights'],('eeg','speech','text')))
                if r['fusion_weights']!=fusion['weights'] or not math.isclose(r['p_final'],expected,abs_tol=1e-12):raise ValueError('Fusion validation mismatch')
                if r['prediction']!=int(expected>.5):raise ValueError('Threshold mismatch')
            fold_stats[name]={m:metrics([r['label'] for r in test],[r['p_'+m] for r in test]) for m in ('eeg','speech','text','final')}
            if fold_stats[name]!=run['folds'][name]['metrics']['test']:raise ValueError('Reported test metrics mismatch')
            for modality,c in cfg['branches'].items():
                p=folder/f'{name}_{modality}.pt';s=run['folds'][name]['branches'][modality]
                if digest(p)!=s['checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
                if s['epochs']!=c['epochs'] or len(s['training_curve'])!=c['epochs']:raise ValueError('Epoch count mismatch')
                if not all(math.isfinite(v['train_cross_entropy']) for v in s['training_curve']):raise ValueError('Nonfinite loss')
                checkpoint_files.append({'path':str(p.relative_to(ROOT)),'sha256':digest(p)})
        tests=[r for r in rows if r['split']=='test']
        all_seeds[str(seed)]={'fold_metrics':fold_stats,
            'fold_mean':{m:{k:float(np.mean([s[m][k] for s in fold_stats.values()])) for k in measures} for m in ('eeg','speech','text','final')},
            'pooled_oof':{m:metrics([r['label'] for r in tests],[r['p_'+m] for r in tests]) for m in ('eeg','speech','text','final')},
            'training_seconds':sum(b['training_seconds'] for f in run['folds'].values() for b in f['branches'].values())}
    aggregate={m:{k:{'mean_of_seed_fold_means':float(np.mean([s['fold_mean'][m][k] for s in all_seeds.values()])),
        'std_of_seed_fold_means_ddof0':float(np.std([s['fold_mean'][m][k] for s in all_seeds.values()])),
        'std_of_15_fold_scores_ddof0':float(np.std([f[m][k] for s in all_seeds.values() for f in s['fold_metrics'].values()]))}
        for k in measures} for m in ('eeg','speech','text','final')}
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'experiment':'B0-37','cohort_id':lock['cohort_id'],
        'passed':True,'seeds':[11,22,33],'folds_per_seed':5,'branch_training_runs':len(checkpoint_files),
        'unique_test_subjects_per_seed':37,'aggregate':aggregate,'per_seed':all_seeds,'checkpoint_files':checkpoint_files,
        'aggregation_note':'Mean each seed across5 folds, then mean across3 seeds. Seed SD and15-fold SD are descriptive; folds/seeds are not independent new subjects.',
        'training_scope':'Three classifiers on fixed EEG/speech/text features; no backbone pretraining or fine-tuning.',
        'limitations':json.loads((ROOT/'configs/protocol_resolution.json').read_text())['limitations']}
    save(ROOT/'reports/b0_training_summary.json',report)
    names={'eeg':'EEG','speech':'语音','text':'文本','final':'三模态融合'}
    lines=['# B0-37 三模态基线训练完成报告','',f"核验时间：{report['timestamp_utc']}。",'',
        '种子11、22、33各五折，每折训练EEG、语音、文本分类器，共45次训练完成。每种子37人测试预测完整，各人仅进入一次测试集；保存45份模型权重。',
        '', '此处训练的是冻结特征上的分类器，不是重新预训练HuBERT、MacBERT或CBraMod。', '',
        '| 分支 | Macro-F1 | MDD recall | Balanced accuracy |','|---|---:|---:|---:|']
    for m in names:
        values=[aggregate[m][k]['mean_of_seed_fold_means'] for k in measures]
        lines.append('| '+names[m]+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', '上表先对每个种子的五折取均值，再对三个种子取均值。逐折、逐种子、37人合并OOF指标及两种标准差见JSON报告；重复种子和折不当作独立新增受试者。', '',
        '验证只用于选择融合权重；固定最终epoch，未使用测试结果选择模型或修改参数。独立核验重新计算了融合权重、阈值、测试指标，检查训练轮数、有限损失及45份权重哈希。', '',
        '结果目录：runs/B0-37/seed_11、seed_22、seed_33。每个目录包含configuration.json、run.json、probabilities.jsonl、training_progress.json和15份.pt。汇总：reports/b0_training_summary.json。', '',
        '该结果建立在已披露的37人代码版协议上，不能直接将其与原论文38人成绩之差归因于方法优劣。随机语音编码器、单位置注意力、EEG尺度/通道证据限制，以及一题低音量ASR恢复例外继续保留。', '']
    (ROOT/'reports/三模态训练完成报告.md').write_bytes('\n'.join(lines).encode('utf-8'))
    print(json.dumps({'passed':True,'branch_training_runs':45,'aggregate':aggregate},indent=2))

if __name__=='__main__':main()

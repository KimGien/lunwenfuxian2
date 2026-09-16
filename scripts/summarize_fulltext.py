"""Validate the fulltext control, including checkpoint replay and parent invariance."""
import json
import math
import numpy as np
import torch
from sklearn.metrics import f1_score,recall_score,balanced_accuracy_score
from fulltext_b0 import ROOT,FEATURES,EXPERIMENT,read,source_checks,feature_paths,models,seed
from prepare import save,digest
from cohort import validate_prediction_records
from metrics import select_weights

def main():
    lock,cfg,folds=source_checks();paths=feature_paths(lock);seed(11)
    device='cuda' if torch.cuda.is_available() else 'cpu';scores={};max_error=0.;replayed=0
    for value in cfg['seeds']:
        folder=ROOT/'runs'/EXPERIMENT/f'seed_{value}';run=read(folder/'run.json')
        parent=ROOT/'runs/B0-37'/f'seed_{value}'
        if digest(parent/'run.json')!=run['parent_run_sha256'] or digest(parent/'probabilities.jsonl')!=run['parent_predictions_sha256']:
            raise ValueError('Parent results changed')
        if run['training_script_sha256']!=digest(ROOT/'scripts/fulltext_b0.py') or run['feature_manifest_sha256']!=digest(FEATURES/'features_manifest.json'):
            raise ValueError('Fulltext training/feature provenance changed')
        rows=[json.loads(s) for s in (folder/'probabilities.jsonl').read_text().splitlines() if s]
        originals={(r['outer_fold'],r['subject_id']):r for r in [json.loads(s) for s in (parent/'probabilities.jsonl').read_text().splitlines() if s]}
        validate_prediction_records(rows,EXPERIMENT)
        if len(rows)!=185:raise ValueError('Incomplete prediction coverage')
        for r in rows:
            old=originals[(r['outer_fold'],r['subject_id'])]
            if any(r[k]!=old[k] for k in ('label','split','p_eeg','p_speech')):raise ValueError('Non-text inputs/results changed')
        scores[str(value)]={}
        for name,f in folds.items():
            fold=int(name[-1]);selection=[r for r in rows if r['outer_fold']==fold]
            weights=select_weights([r for r in selection if r['split']=='val'])['weights']
            for r in selection:
                expected=sum(w*r['p_'+m] for w,m in zip(weights,('eeg','speech','text')))
                if weights!=r['fusion_weights'] or not math.isclose(expected,r['p_final'],abs_tol=1e-12) or r['prediction']!=int(expected>.5):
                    raise ValueError('Fusion mismatch')
            checkpoint=folder/f'{name}_text.pt'
            if digest(checkpoint)!=run['folds'][name]['text_checkpoint_sha256']:raise ValueError('Checkpoint hash mismatch')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if saved['train_ids']!=f['train'] or saved['seed']!=value or saved['fold']!=fold:raise ValueError('Checkpoint training provenance mismatch')
            c=cfg['branches']['text'];model=getattr(models,c['model_class'])(**c['model_kwargs']).to(device)
            model.load_state_dict(saved['state_dict'],strict=True);model.eval()
            with torch.no_grad():
                for r in selection:
                    x=torch.tensor(np.load(paths[r['subject_id']],allow_pickle=False),dtype=torch.float32,device=device).unsqueeze(0)
                    probability=float(torch.softmax(model(x),dim=-1)[0,1].cpu());error=abs(probability-r['p_text'])
                    if error>1e-6 or not math.isfinite(probability):raise ValueError('Text checkpoint replay mismatch')
                    replayed+=1;max_error=max(max_error,error)
            del model,saved
            test=[r for r in selection if r['split']=='test'];y=[r['label'] for r in test]
            scores[str(value)][name]={}
            for modality in ('eeg','speech','text','final'):
                pred=[int(r['p_'+modality]>.5) for r in test]
                result={'macro_f1':float(f1_score(y,pred,average='macro')),'mdd_recall':float(recall_score(y,pred)),
                    'balanced_accuracy':float(balanced_accuracy_score(y,pred))}
                for k,v in result.items():
                    if abs(v-run['folds'][name]['metrics']['test'][modality][k])>1e-12:raise ValueError('Metric mismatch')
                scores[str(value)][name][modality]=result
    aggregate={m:{k:float(np.mean([f[m][k] for s in scores.values() for f in s.values()])) for k in ('macro_f1','mdd_recall','balanced_accuracy')}
        for m in ('eeg','speech','text','final')}
    first=read(ROOT/'reports/b0_training_summary.json')['aggregate'];audit=read(ROOT/'reports/fulltext_features_audit.json')
    report={'passed':True,'experiment':EXPERIMENT,'parent_experiment':'B0-37','changed_recordings':audit['changed_count'],
        'max_tokens':audit['max_tokens'],'truncated_recordings':audit['truncated_recordings'],'aggregate':aggregate,
        'per_seed_fold_test_metrics':scores,'parent_macro_f1':{m:first[m]['macro_f1']['mean_of_seed_fold_means'] for m in aggregate},
        'text_checkpoints':15,'replayed_probabilities':replayed,'max_replay_absolute_error':max_error,
        'parent_results_unchanged':True,'eeg_speech_predictions_identical':True,
        'comparison_note':'Same previously observed outer folds; development comparison, not a new independent test set.'}
    save(ROOT/'reports/fulltext_comparison.json',report)
    lines=['# B0-37 全文版本对照报告','',
        '按用户要求，全文版本使用每题所有ASR分段，按(start,end)排序后原样拼接。1073题中130题增加后续内容；最长393个token，所有全文均完整输入MacBERT，无截断，也没有重新识别音频。', '',
        '实验ID为B0-37-fulltext；原B0-37首段版与已上传GitHub版本保留。37人、五折、种子11/22/33和分类器参数相同。新增37份29×768文本特征，重训15个文本分类器；复用已核验的EEG/语音预测，逐折仅在val重选融合权重。', '',
        '| 输出 | 首段版Macro-F1 | 全文版Macro-F1 |', '|---|---:|---:|']
    for m,label in (('eeg','EEG'),('speech','语音'),('text','文本'),('final','三模态融合')):
        lines.append(f'| {label} | {report["parent_macro_f1"][m]:.4f} | {aggregate[m]["macro_f1"]:.4f} |')
    lines+=['','均为3种子×5折测试分数的均值。测试划分已经在首段版中查看过，因此属于固定划分上的开发对照，不能当作新的独立测试集。', '',
        f'验收：15个文本权重重载后重现{replayed}个训练/验证/测试概率，最大绝对误差{max_error}；每种子测试37人恰好一次；全部指标与scikit-learn核对一致；EEG/语音预测和原首段版结果文件均未修改。', '',
        '新配置：configs/b0_fulltext.json、configs/protocol_fulltext.json。新转写：transcripts/B0-37-fulltext/；特征：features/B0-37-fulltext/；训练：runs/B0-37-fulltext/。', '',
        '本次数据最多393个token。将来若新增转写超过512，会停止并要求明确长文本处理方法，不静默丢弃尾部。低音量那题沿用已记录的增益重试结果；其他已披露的语音编码器及EEG证据限制继续适用。', '',
        '运行入口：scripts/fulltext_b0.py --extract --train；仅预检时不带参数。已存在结果不覆盖。该全文选择是用户要求的实现变更，不是已证实的作者原实验政策；后续采用全文的方法应与全文版比较。', '']
    (ROOT/'reports/全文版本对照报告.md').write_bytes('\n'.join(lines).encode('utf-8'))
    print(json.dumps({k:v for k,v in report.items() if k!='per_seed_fold_test_metrics'},indent=2))

if __name__=='__main__':main()

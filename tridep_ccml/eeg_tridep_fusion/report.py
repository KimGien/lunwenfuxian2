"""Publish verified results only; source plan remains an immutable registration."""
from pathlib import Path
import csv
import json
import hashlib
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tridep_ccml/experiments/EEG-TriDep-Fusion-v1-37-20261013'
REPORT=ROOT/'output/analysis/eeg_tridep_fusion_v1_20261013'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def descriptor(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
def write_json(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    active=read(OUT/'active_registration.json');assert sha(ROOT/active['path'])==active['sha256'];reg=read(ROOT/active['path']);attempt=ROOT/read(OUT/'execution_pointer.json')['attempt']
    verification=read(attempt/'independent_verification.json')
    assert verification['passed'] and verification['results_sha256']==sha(attempt/'results.json')
    r=read(attempt/'results.json');cost=read(attempt/'execution_complete.json');preflight=read(attempt/'preflight.json')
    units=read(attempt/'unit_sources.json');selections=read(attempt/'selections.json')
    main=[u for u in reg['matrix'] if u['group']=='main' and u['candidate']]
    ft=[u for u in reg['matrix'] if u['group']=='finetune' and u['candidate']]
    kept=[u['id'] for u in reg['matrix'] if u['candidate'] and r['comparisons'][u['id']]['A']['primary_retention']]
    def score(u,v='A'):return r['conditions'][u['id']][v]['primary']['macro_f1']
    def row(u):
        c=r['comparisons'][u['id']]['A']['comparisons'];ch=r['changes'][u['id']]['A']
        return f"| {u['method']} | {score(u):.6f} | {c['matched_LaBraM']['deltas']['macro_f1']:+.6f} | {c['original_E']['deltas']['macro_f1']:+.6f} | {c['ST']['deltas']['macro_f1']:+.6f} | {ch['corrected']['records']}/{ch['new_error']['records']} | {'通过' if r['comparisons'][u['id']]['A']['primary_retention'] else '未通过'} |"
    text=['# 九篇 EEG 接入 TRI-DEP 融合完成报告','',
      '25 个 EEG 条件快照、375 组种子折快照、四种融合视图共 1,500 个评价单元全部完成。375 组全部复用已保存概率，无补推理；本轮新增基础模型训练、融合网络训练、优化器更新、模型加载、神经前向和 ASR 均为 0。','',
      f"主要协议的系统保留条件通过 {len(kept)} 条候选路线。{'通过路线：'+', '.join(kept)+'。' if kept else '本轮不替换已保存系统；具体配对涨分、概率收益和次要协议结果仍完整保留。'}",'',
      'A 为固定 EEG/语音/文本=0.05/0.35/0.60；B 为原3人验证集上重新选权；C 为三模态等权；D 为历史EST权重直接替换。A 是主要协议，B/C/D 不因分数更高而改为主要结果。','',
      '## 基线重新计算结果','',
      '| 对照 | Macro-F1 | BA | MDD召回 | AUC | NLL | Brier |','|---|---:|---:|---:|---:|---:|---:|']
    names={'historical_ST':'历史固定ST','historical_EST':'历史固定EST','ST_A':'A去EEG并保持S/T比例','original_E_A':'原EEG按A融合',
           'ST_paper_reference':'论文公布ST权重0.45/0.55','ST_C':'语音文本等权','original_E_C':'原EEG三模态等权'}
    for k,label in names.items():
        m=r['baselines'][k]['primary'];text.append('| '+label+' | '+' | '.join(f'{m[x]:.6f}' for x in ('macro_f1','balanced_accuracy','mdd_recall','auc','nll','brier'))+' |')
    text+=['','A的ST消融为(0.35pS+0.60pT)/0.95。历史ST的0.724180不代替该消融。原EEG来源为阶段0原模型，匹配LaBraM来源为九篇计划中的相应输入窗长和训练路线。','',
      '## 九篇第一轮主结果','',
      'Δ均为Macro-F1的绝对差值，0.01相当于1个百分点。纠错/新增错误以111条种子测试记录计数，去重受试者数另见明细。','',
      '| 候选 | A融合F1 | Δ匹配LaBraM | Δ原EEG同协议 | Δ匹配ST | 纠错/新增错误 | 主要系统保留 |',
      '|---|---:|---:|---:|---:|---:|---|']+[row(u) for u in main]
    text+=['','## 七条有限微调扩展','',
      '| 候选 | A融合F1 | Δ匹配LaBraM | Δ原EEG同协议 | Δ匹配ST | 纠错/新增错误 | 主要系统保留 |',
      '|---|---:|---:|---:|---:|---:|---|']+[row(u) for u in ft]
    text+=['','微调路线所选检查点阶段如下；best处于热身阶段者不称为编码器更新后的最佳模型。','',
           '| 候选 | 所选阶段及组数 |','|---|---|']
    for u in ft:
        phases=Counter(x['selected_checkpoint_phase'] for x in units if x['unit']==u['id'])
        text.append('| '+u['method']+' | '+'；'.join(f'{k}: {v}' for k,v in phases.items())+' |')
    text+=['','## 全部协议与匹配控制','',
      '| 条件快照 | A固定权重 | B重新选权 | C等权 | D旧权重 |', '|---|---:|---:|---:|---:|']
    for u in reg['matrix']:text.append('| '+u['id']+' | '+' | '.join(f'{score(u,v):.6f}' for v in ('A','B','C','D'))+' |')
    text+=['','以下只列B/C通过完整系统门槛的路线，作为次要协议结果，不替代A。完整门槛要求同时对匹配LaBraM、原EEG同协议和ST通过配对门槛，并严格优于历史ST/EST。','']
    for v in ('B','C'):
        passing=[u['id'] for u in reg['matrix'] if u['candidate'] and r['comparisons'][u['id']][v]['protocol_retention']]
        text.append(f"- {v}："+('、'.join(passing) if passing else '无')+'。')
    text+=['','## 延长EEG训练能否传递为融合收益','',
           '| 骨干 | A融合60轮F1 | A融合300轮F1 | ΔF1 | ΔBA | ΔMDD召回 |','|---|---:|---:|---:|---:|---:|']
    for model,views in r['duration'].items():
        a=r['conditions'][model+'@60']['A']['primary'];b=r['conditions'][model+'@300']['A']['primary'];d=views['A']['deltas']
        text.append(f"| {model.split('-')[0]} | {a['macro_f1']:.6f} | {b['macro_f1']:.6f} | {d['macro_f1']:+.6f} | {d['balanced_accuracy']:+.6f} | {d['mdd_recall']:+.6f} |")
    text+=['','本表保持A融合公式和S/T不变，只比较已登记的60/300轮快照。B的轮次差值还包含重新选权影响，见results.json的duration字段。未选择120/200轮中的测试最优快照。','',
           '## 选权及概率收益','', '| 条件快照 | B选中EEG非零组数 | 存在非零EEG并列最优组数 | 最优并列组合数范围 | A对匹配ST的NLL/Brier同时改善 |',
           '|---|---:|---:|---|---|']
    for u in reg['matrix']:
        ss=[x for x in selections if x['unit']==u['id']]
        text.append(f"| {u['id']} | {sum(x['weights'][0]>0 for x in ss)}/15 | {sum(x['nonzero_eeg_tied_best'] for x in ss)}/15 | {min(x['ties'] for x in ss)}–{max(x['ties'] for x in ss)} | {'是' if r['comparisons'][u['id']]['A']['comparisons']['ST']['probability_benefit'] else '否'} |")
    text+=['','EEG权重为零受验证集和并列规则影响，不单独说明EEG无效；非零权重也不是有效性证据。NLL和Brier同时改善是独立的概率收益，不替代分类门槛。','',
      '## 验证和成本','',
      f"已核对 {preflight['sources']} 个绑定来源文件及检查点，输入共13,875条。独立实现重新连接原始批次和S/T来源，逐条复核55,500条融合概率、6,105条基线概率、11,100条测试纠错记录，并用NumPy重做375次候选选权及30次基线网格重放、用scikit-learn复算全部111组汇总。所有检查通过。",'',
      f"候选选权375次、验证候选评分86,625次；基线重放另含30次网格、3,780次候选评分。执行耗时{cost['elapsed_seconds']:.2f}秒，独立核验耗时{verification['verification_seconds']:.2f}秒；均含本地文件读取/哈希，不能解释为神经网络训练耗时。",'',
      '主指标先平均种子内五折，再平均三个种子；逐种子37人OOF另列。阈值固定p>0.5，相等为HC；NLL真类概率下限1e-15，裁剪不影响融合输入或分类。训练/验证指标保留为诊断，验证参与EEG选模及B选权，不能作为独立性能证据。','',
      '## 结论范围和文件','',
      '当前37人已反复用于开发，111条种子记录仍只有37名受试者。该报告是固定融合方案的开发比较，不是新的盲测或统计显著性结论。不同输入参考、幅值、窗长、训练范式仍按EEG原合同解释；EEGMamba仍为作者公开实现适配。未启动十篇融合或CCML/残差融合头训练。','',
      '[完整指标和配对判定](results.json) · [全部100行比较表](summary.csv) · [逐折权重与并列统计](selections.json) · [逐人预测](predictions.jsonl) · [纠错与新增错误](paired_changes.jsonl) · [独立核验](independent_verification.json) · [实际成本](cost.json)','']
    REPORT.mkdir(parents=True,exist_ok=True)
    copy=['results.json','summary.csv','selections.json','baseline_selections.json','predictions.jsonl','baseline_predictions.jsonl','paired_changes.jsonl','independent_verification.json','unit_sources.json','preflight.json']
    for name in copy:
        with (REPORT/name).open('xb') as f:f.write((attempt/name).read_bytes())
    attempts=[]
    for p in sorted(OUT.glob('attempt_*')):
        for n in ('failed.json','execution_complete.json'):
            if (p/n).exists():attempts.append(dict(attempt=p.name,status=n,record=read(p/n)))
    write_json(REPORT/'cost.json',dict(execution=cost,independent_verification_seconds=verification['verification_seconds'],all_attempts=attempts,
        new_training=0,neural_forwards=0,source_hash_bytes=preflight['bytes_hashed'],note='Hash I/O included in wall time; retries preserved.'))
    report_path=REPORT/'九篇EEG接入TRI-DEP融合完成报告.md'
    with report_path.open('x',encoding='utf-8') as f:f.write('\n'.join(text))
    sources=[descriptor(p) for p in sorted(REPORT.iterdir()) if p.is_file()]
    write_json(REPORT/'artifact_manifest.json',dict(files=sources,registration=active,execution_attempt=attempt.relative_to(ROOT).as_posix()))
    complete=dict(passed=True,units=375,condition_snapshots=25,view_units=1500,new_training=0,neural_forwards=0,
        retained_primary=kept,verification=descriptor(attempt/'independent_verification.json'),report=descriptor(report_path),manifest=descriptor(REPORT/'artifact_manifest.json'))
    write_json(OUT/'complete.json',complete)
    print(json.dumps(complete,ensure_ascii=False),flush=True)

if __name__=='__main__':main()

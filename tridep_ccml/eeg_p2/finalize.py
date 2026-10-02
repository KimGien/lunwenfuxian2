"""Human-readable closeout and artifact verification for the completed QC task."""
from __future__ import annotations
from .common import *
from .scales import METHODS


def main():
    v=read_json(OUT/'verification.json');q=read_json(OUT/'quality_summary.json')
    f=read_json(OUT/'flag_validation.json');s=read_json(OUT/'scale_summary.json')
    subjects=read_json(OUT/'subject_summary.json')
    require(v['passed'],'Numerical audit not verified')
    reg=read_json(OUT/'registration.json')
    for name,sha in reg['sources'].items():
        require(digest(WORK/name)==sha,'Registered core audit source changed')
    sources={'checked_date':'2026-09-28','official_upstream_commit_sha':None,
        'version_limit':'Public main-branch code was inspected, but an immutable upstream commit and linkage to the local Mumtaz checkpoint were not verified.',
        'items':[
            {'name':'MODMA data paper','url':'https://www.nature.com/articles/s41597-022-01211-x',
             'observation':'128 E1..E128 channels plus Cz reference; supports metadata interpretation, not individual physical calibration.'},
            {'name':'Official CBraMod Mumtaz preprocessing','url':'https://raw.githubusercontent.com/wjq-learning/CBraMod/main/preprocessing/preprocessing_mumtaz.py',
             'observed_operation':'raw.to_data_frame().values; discard time column',
             'observation':'EDF -> MNE resample/filter/notch -> default DataFrame export.'},
            {'name':'Official CBraMod Mumtaz dataset loader','url':'https://raw.githubusercontent.com/wjq-learning/CBraMod/main/datasets/mumtaz_dataset.py',
             'observed_operation':'return data/100, label','observation':'Explicit divide-by-100 model input scaling.'},
            {'name':'MNE Raw DataFrame units','url':'https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.to_data_frame',
             'observation':'Default EEG DataFrame scaling is 1e6 from volts to microvolts; locally confirmed with MNE 1.10.1.'},
            {'name':'EEGLAB units convention','url':'https://eeglab.org/tutorials/06_RejectArtifacts/Scrolling_data.html',
             'observation':'EEGLAB tutorial uses microvolt amplitude display. This is a format convention, not this cohort calibration proof.'}],
        'local_extractor':desc(WORK/'tridep_ccml/v2/extract_eeg.py'),
        'local_upstream_extractor':desc(WORK/'S00/upstream-dev/lib/feature_extraction/eeg/extract_cbramod.py'),
        'checkpoint_provenance_prior_audit':desc(WORK/'tridep_ccml/reports/ccml_v2_audit/CCML-v2-Audit输入核查报告.md')}
    save(OUT/'external_source_evidence.json',sources)
    title='# EEG-P2：无标签质量与尺度核查完成报告'
    text=[title,'','日期：2026-09-28。后续优先级 2 已完成。核查覆盖现有 37 人的全部原始活动通道，以及模型使用的 19 导联、60 个五秒窗。所有诊断不使用疾病标签；缩放比较只使用各折 train 人。无增强、无删人删窗、无模型重训，无新增 F1。','',
    '**本轮定位到三类输入问题：滤波边界瞬态、部分记录持续的大幅低频变化，以及公共标准化对高能量受试者的敏感性。另补上了官方 CBraMod 的归一化线索，但本地权重的尺度来源仍未完全闭合。**','',
    '## 1. 原始数据和预处理是否被写错','',
    '- 37/37 人的 FDT 与 staged MAT 全数组数值一致；37/37 人的现行预处理可逐位重放 R1 原始波形缓存。现有大幅差异不能归因于本次格式转换或缓存写错。',
    '- 250 Hz、129 路的顺序均符合 E1–E128 加 Cz。除参考行外，未检出非有限样本、完全重复通道、整路常数、连续至少 1 秒精确平线或连续至少 20 毫秒的端值平台。没有触发这些结构规则不等于没有伪迹。',
    '- Cz 的 REF 类型和全零数值在 37 人中一致；Cz 不在当前 19 路输入中，不计为坏导。局部 ref 字段是 common，本地 CBraMod 路径没有追加平均参考；数据论文也将末路 Cz 解释为参考。原坐标仍有已知模板冲突，因此未按这些坐标插值。 [MODMA 数据论文](https://www.nature.com/articles/s41597-022-01211-x)',
    '- 37 份文件均没有非空 ICA 权重，也没有设备单位/增益校准字段。reject 中的非空条目是 threshold、threshentropy 等设置；实际拒绝掩码为空。这说明缺少清洗轨迹，不能证明数据发布前绝对未清洗。',
    '- 旧清单中的 signal_quality_verified 没有被当作 EEG 清洁证明；本轮根据原始波形单独检查。','',
    '## 2. 新发现：边界瞬态可主导模型输入','',
    '固定规则在 2,220 个受试者窗口中标记了 95 个相对高幅窗，涉及 26 人。其中 41 个位于第一或最后一个窗口，54 个位于内部。它们是复核线索，未据此丢弃任何窗。','',
    '| 受试者 | 滤波后前 5 秒占整段平方能量 |','|---|---:|']
    for r in sorted(f['boundary'],key=lambda r:r['first5_energy_fraction'],reverse=True)[:5]:
        text.append(f'| {r["subject_id"]} | {100*r["first5_energy_fraction"]:.2f}% |')
    text+=['',
    '以 02030007 做只读阶段重放：在同一逐通道去时间中位数的统计口径下，前 5 秒能量占比从原始 1.84%、重采样后 1.84%、带通后 1.69%，变成陷波后 90.35%；陷波后的首窗 RMS 为内部窗中位数的 23.79 倍。陷波消除了大量持续窄带能量，但边界瞬态相对保留下来，不能把“陷波后占比上升”写成“总能量被放大 50 倍”。','',
    '额外使用已知平稳的 50 Hz + 10 Hz 正弦通过完全相同的旧流程，不涉及真实标签或训练增强。处理后也出现首尾瞬态：首窗 RMS 是内部窗的 9.55 倍，末窗占去中位数后能量的 94.13%。这是该处理链能制造边界失真的直接工程证据，但没有单独归因到某个填充选项，也没有证明所有真实记录的首尾异常都由处理造成。','',
    '![真实记录边界问题](figures/02030007_quality.png)','',
    '![平稳合成信号的边界验证](figures/synthetic_filter_boundary.png)','',
    '这里的合成正弦仅用于信号处理验证，没有加入任何训练数据。初始阈值和全部筛查结果保持不变；该追加检查作为事后无标签复核，单列于 followup_flag_validation_protocol.json 与 flag_validation.json。','',
    '## 3. 02010022：持续低频大幅变化，不能直接删除','',
    f'滤波后受试者 RMS 的最小/中位/最大值为 {q["rms_min"]:.2f}/{q["rms_median"]:.2f}/{q["rms_max"]:.2f}，最大最小相差 {q["rms_max_min"]:.2f} 倍。图中使用 native 数值，不将其直接写成已校准的 μV。02010022 占全体平方能量的 {100*q["largest_energy_share"]:.2f}%。','',
    '- 其 F7/F4 RMS 分别为 610.10/536.91 native；占各自同名通道全队列平方能量的 68.48%/78.40%。在它自身的 1–40 Hz 功率内，1–4 Hz 的占比分别约为 99.69%/99.81%。',
    '- 大幅变化分布于多个时窗。最高能量窗在 270–275 秒，图中可见多个额部通道的缓慢大幅起伏；不是只有首窗异常，也没有触发本轮平线、端值平台、通道内极端孤立窗或超 8×MAD 样本规则。',
    '- 这些证据更支持优先复核慢漂移、接触状态和缓慢伪迹等来源，但缺少专门 EOG/EMG 和可靠成分标记，不能确定为眨眼/肌电，更不能当作病理机制。未删除此人、未手工压小幅值。','',
    '![02010022 波形与频谱](figures/02010022_quality.png)','',
    '## 4. 其他坏导/坏段与频谱提示','',
    '- 02020018 与 02030009 的 T6 稳健尺度分别为本人通道中位数的 5.02 和 5.82 倍，触发相对高尺度复核。仅凭相对幅值不能判为坏导，也没有做插值。',
    '- 29 人、182 个通道触发 49–51 Hz 相对邻带峰值规则；这些通道的该频带占 1–75 Hz 功率的中位数约 9.12%，范围 0.14%–75.51%。既有陷波并非没有执行：50 Hz PSD 相对原始信号的中位衰减为 45.22 dB。仍有峰值不等于每个通道都有严重污染，也不能根据邻带比值直接扩大陷波。',
    '- 9 人有通道的 35–75 Hz 功率占 1–75 Hz 超过一半。该指标包含工频附近成分，不能直接标为肌电。对只输入 1–40 Hz 频谱的 SparNet，也不能把全部高频提示等同于其实际输入污染。',
    '- 完整逐人、逐导联、逐窗结果及时间位置见质量复核清单和 subjects/、arrays/。所有 flags 都是复核提示。','',
    '## 5. 单位和 CBraMod 输入尺度：推进到哪里','',
    '已验证所有人的 EEGLAB 读取器数值为 native×1e-6；本机 MNE 1.10.1 导出 DataFrame 又按默认约定把 EEG 乘 1e6。用原波形测试“转 SI→线性滤波→转回 native”与原 native 处理的最大差仅 2.82e-11，说明格式单位转换和模型归一化是两件事。 [MNE DataFrame 单位说明](https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.to_data_frame)','',
    'CBraMod 作者当前公开的 Mumtaz 预处理通过 to_data_frame 导出数值，随后数据加载器明确执行 data/100。因此按 MNE 默认单位约定，其公开路径给模型的是 μV/100 数值；本地提取代码没有对应的除 100。 [官方预处理](https://raw.githubusercontent.com/wjq-learning/CBraMod/main/preprocessing/preprocessing_mumtaz.py) · [官方加载器](https://raw.githubusercontent.com/wjq-learning/CBraMod/main/datasets/mumtaz_dataset.py)','',
    '这比此前只知道“缺少尺度说明”更具体，但仍有两处未闭合：未取得本地数据的设备校准记录，也未将本地 pretrained_weights2.pth 绑定到这个官方训练版本。此次网页核对也未固定到不可变的 Git 提交，来源 URL、日期和观察到的运算单列保存。因此没有擅自除 100 或乘 1e-6 重导特征。','',
    '**对已做训练折 z-score 的原始波形路线，统一单位倍数会相消。** 五折数值验证中，同时缩放输入及拟合均值/标准差，标准化结果的最大差不超过 2.85e-14。因此不能把单位问题当成 EEGNet/SparNet 低 F1 的统一解释；它主要关系到直接接收 native 数值的冻结 CBraMod 路径是否匹配权重预期。','',
    '## 6. 训练折内缩放诊断','',
    '四种候选均按原五折的 train 身份计算，没有使用 val/test 标签或它们的信号来拟合、排序或评价缩放。以下是五个训练折的范围，未选择获胜方案。','',
    '| 固定方案 | 人间 RMS 最大/最小 | 最大单人能量占比 | 留一训练人的共享尺度最大变化 |',
    '|---|---:|---:|---:|']
    labels=['S0 原 pooled mean/std','S1 受试者等权 median/MAD 汇总','S2 每人每导联 mean/std','S3 每人每导联 median/MAD']
    for m,label in zip(METHODS,labels):
        vals=[r['methods'][m] for r in s['folds']]
        a=[r['subject_rms_max_min_ratio'] for r in vals];b=[100*r['largest_subject_energy_share'] for r in vals]
        inf=[r['largest_leave_one_scale_factor'] for r in vals if 'largest_leave_one_scale_factor' in r]
        col=f'{min(inf):.2f}–{max(inf):.2f} 倍' if inf else '无共享尺度参数'
        text.append(f'| {label} | {min(a):.2f}–{max(a):.2f} | {min(b):.2f}%–{max(b):.2f}% | {col} |')
    text+=['',
    'S1 的共享尺度参数更稳定，但它没有清除大幅记录。在 02010022 属于 train 的四折，其输出能量占比由 S0 的 40.27%–47.89% 变成 S1 的 52.66%–62.68%。因此“公共稳健缩放能消除该人的主导”在本次诊断中不成立。','',
    'S2 把每人每导联方差固定为 1，能量等权是定义上的结果，不是质量改善证据。S3 降低了人间尺度差异，但仍保留较大瞬态尾部。两者均会丢弃绝对幅值信息，并可能放大低质量通道；未来使用还须固定完整记录的离线处理边界。本轮保持原始缓存和训练配置不变。','',
    '![训练折尺度比较](figures/train_scale_diagnostics.png)','',
    '## 7. 执行结论和可追溯交付','',
    '优先级 2 的核查已完成；其结论不是“应直接删掉高幅者”或“立即换成稳健缩放”。下一项有明确工程依据的工作是：在新版本中先验证滤波边界处理，检查强窄带成分与瞬态保留，再用训练折内部协议确定处理方式。只有固定输入处理后重训配对模型，才能评价 F1 是否改善。本轮没有这类性能结论。','',
    f'- {v["tests_passed"]} 项回归测试通过；37 人源数据、37 次预处理逐位重放、五折原标准化重放均通过。',
    '- 原始输入未改写；R1 的 986 项、P1 的 667 项冻结产物均核验不变。',
    '- quality_summary.json、subject_summary.json：质量汇总；subjects/ 与 arrays/：完整逐人结果与时频数组。',
    '- scales/fold_*.json：每个方法的训练名单、参数、每人分布与留一敏感性；scale_summary.json：比较摘要。',
    '- flag_validation.json：边界、实际滤波阶段、平稳正弦验证、窄带提示和单位相消测试；external_source_evidence.json：单位与官方代码证据边界。',
    '- [质量复核清单](质量复核清单.md)：37 人图像和异常时段入口；[通道幅值概览](figures/channel_rms_overview.png)。','']
    (OUT/'EEG-P2无标签质量与尺度核查报告.md').write_text('\n'.join(text),encoding='utf-8')
    checklist=['# 37 人 EEG 质量复核清单','',
        '所有提示不使用诊断标签，不是坏导/坏段裁决。时间区间以原记录起点计，右端不包含；窄带/高频列是触发固定规则的通道数。图的纵轴为 native 数值。','',
        '| 受试者与图 | RMS | 相对高尺度导联 | 相对高幅窗（秒） | 窄带峰值/高频通道数 |',
        '|---|---:|---|---|---:|']
    for r in subjects:
        times='、'.join(f'{i*5}–{(i+1)*5}' for i in r['high_rms_windows_any_channel']) or '—'
        checklist.append(f'| [{r["subject_id"]}](figures/{r["subject_id"]}_quality.png) | {r["rms"]:.2f} | {"、".join(r["channel_scale_review"]) or "—"} | {times} | {len(r["filtered_line_noise_channels"])}/{len(r["filtered_high_frequency_channels"])} |')
    (OUT/'质量复核清单.md').write_text('\n'.join(checklist)+'\n',encoding='utf-8')
    require(len(list((OUT/'figures').glob('*_quality.png')))==37,'Missing subject figures')
    history=history_check()
    save(OUT/'completion.json',{'complete':True,'finished_utc':now(),'priority':2,'scope':'label-free quality audit and training-fold numerical scale diagnostics',
        'all_core_checks_passed':True,'historical_artifacts_unchanged':history,'subject_plots':37,
        'reviewed_figures':['02010022_quality.png','02020010_quality.png','02030007_quality.png','channel_rms_overview.png','train_scale_diagnostics.png','synthetic_filter_boundary.png'],
        'classification_models_trained':0,'new_F1_claim':False,'preprocessing_selected':None,
        'implementation_sources':{p.relative_to(WORK).as_posix():digest(p) for p in CODE.glob('*.py')}})
    save(OUT/'status.json',{'state':'complete','subjects_complete':37,'priority':2,'classification_training':False,'updated_utc':now()})
    hashes={p.relative_to(OUT).as_posix():digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='deliverable_hashes.json'}
    save(OUT/'deliverable_hashes.json',hashes)
    for name,sha in hashes.items():
        require(digest(OUT/name)==sha,'Final artifact changed')
    print(f'P2 COMPLETE: {len(hashes)} deliverables verified; R1/P1 unchanged.',flush=True)


if __name__=='__main__':
    main()

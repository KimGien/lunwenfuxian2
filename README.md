# TRI-DEP / CCML 实验复现

MODMA 37 人实验的源码、固定配置、完成报告与验收记录。实验按完成顺序追加；原仓库 S00 历史保持完整。

[实验完成顺序与上传范围](docs/实验完成顺序与上传范围.md)

| 已归档版本 | 完成日期 | 入口 |
|---|---|---|
| TRI-DEP 修改版 / S00 全文基线 | 2026-09-16 | [报告](S00/docs/S00完成详细步骤与训练代码.md) |
| TRI-DEP + CCML：TC-00 至 TC-05 | 2026-09-21 | [报告](tridep_ccml/reports/tc05/TC-05实验完成报告.md) |
| TRI-DEP + CCML v2：固定与可选实验完整收尾 | 2026-09-24 | [报告](tridep_ccml/reports/ccml_v2_full_closeout/CCML-v2总体总结与收尾报告.md) |
| EEG 数据增强与三个终点重跑 | 2026-09-27 | [报告](tridep_ccml/experiments/EEG-AUG-JITTER-37-fulltext/EEG数据增强_三终点重跑完成报告.md) |
| EEG-R1：分支替换 | 2026-09-27 | [报告](tridep_ccml/experiments/EEG-R1-37-fulltext/EEG-R1执行报告.md) |
| EEG-P1：有效学习验收与 SparNet 修复 | 2026-09-27 | [报告](tridep_ccml/experiments/EEG-P1-37-fulltext/EEG-P1执行报告.md) |
| EEG-P2：无标签质量与尺度核查 | 2026-09-28 | [报告](output/analysis/eeg_quality_scale_20260928/EEG-P2无标签质量与尺度核查报告.md) |
| EEG-P3：泛化、BN 与校准 | 2026-09-29 | [报告](output/analysis/eeg_generalization_20260929/EEG-P3泛化BN与校准报告.md) |
| EEG-E2：首版工程验收停止记录 | 2026-09-29 | [报告](output/analysis/eeg_e2_20260929/EEG-E2输入边界修复阶段报告.md) |
| EEG-E2v2：边界修复与固定训练主线 | 2026-09-29 | [报告](output/analysis/eeg_e2v2_20260929/EEG-E2v2主线完成报告.md) |
| FUSION-G1：CE / EDL / CCML 六条件 | 2026-09-30 | [报告](output/analysis/fusion_g1_20260929/FUSION-G1整体改进主线完成报告.md) |

## 运行与来源

[基线准备与训练说明](S00/README.md) · [完整基线步骤与源码](S00/docs/S00完成详细步骤与训练代码.md)
[CCML / v2 说明](tridep_ccml/README.md) · [v2 完整收尾](tridep_ccml/reports/ccml_v2_full_closeout/CCML-v2总体总结与收尾报告.md)

从仓库根目录运行，路径结构与本地实验工作区一致。新克隆需自行准备符合锁定版本的环境、授权数据、骨干和父实验产物。原始数据、转写正文、逐人预测、受试者级输入清单、配对明细、个体诊断图、特征缓存和权重保留本地。新增报告中的受试者编号已省略；本地原件保持原样。

完成报告中的重载、配对和指标验收是原实验记录，本次上传仅核对文件、依赖结构与发布范围，没有重新训练。模型比较来自反复使用的 37 人开发样本。

本次不发布 Tridep-DFMU。尚未完成最终验收的 QPOOL 和未执行的 DAIC-WOZ 实验未纳入完成阶段。

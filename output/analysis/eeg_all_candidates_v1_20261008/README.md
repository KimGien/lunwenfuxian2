# 九篇 EEG 实验当前结果入口

2026-10-13 按用户最新指令更新保留规则：以本地原 TRI-DEP 的 CBraMod-Mumtaz＋原CNN（S00/B0-37，60轮）测试 Macro-F1 **0.4494011544011543，约44.94%**为基线，每条已登记路线的主要结果严格高于基线即保留。

- [当前完成报告](九候选统一完成报告_原TRIDEP基线修订版_20261013.md)
- [当前逐路线结果与保留判定](unified_comparison_original_tridep_baseline_20261013.csv)
- [当前规则、基线来源与九篇结论](retention_policy_revision_20261013.json)
- [保留模型清单：9条路线、135个种子/折模型](retained_models_original_tridep_baseline_20261013.json)
- [当前版本及文件校验索引](current_retention.json)

九候选保留 LUNA 冻结小头、CodeBrain 冻结小头和末层微调、BIOT 末层微调；LaBraM五条对照路线也保留。其他路线的结果和权重完整保存，但不标记为通过当前保留规则。

原 `九候选统一完成报告.md`、`unified_comparison.csv`、评价JSON及其完成清单保持封存字节，它们记录的是旧LaBraM匹配控制门槛。当前报告使用以上带日期修订版；旧结论不再作为当前保留决定。

本次仅修改候选保留规则与汇总，不改变主要快照、预测或分数，不新增训练，不自动接入融合。新规则是在已有测试结果公开后制定的开发决策；原S00与新路线输入和训练设置的差异在修订报告中说明。

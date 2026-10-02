# MacBERT有限微调阶段完成报告

有限文本微调未通过预设分类开发门槛，保留第二批 ST 均值损失候选，不替换。
继续按用户指示跳过数据缺陷排查，以数据无缺陷为工作假设。本轮仅适配文本 MacBERT 最后一层的 query/value；语音输入与 ST 均值损失沿用第二批。

## 执行与验收

冻结控制与 LoRA 候选各 15 组，共完成 30 个新模型、7,200 次更新。每组固定 60 轮、batch 8；rank=4、alpha=4、LoRA lr=1e-5、无 dropout，新增 12,288 个可训练参数。原 MacBERT 参数完全冻结。
4 项训练前工程测试通过；15 对初值、随机状态、初始输出、训练名单和预算匹配。所有候选的 LoRA 均实际更新，所有原末层权重保持原哈希。30 个模型与 1,110 条逐人输出精确重载，融合 90 组折内指标及全部端点 360 组折内指标独立复算通过。
全部拟合锁定后统一生成结果，并独立导出 30 套绑定检查点和 train 名单的文本题向量及 pooled 特征。共享缓存仅为无本队列标签监督的前 11 层输出；监督适配特征没有跨折共享。无验证选轮、参数或阈值搜索。

## 同版本配对结果

| 条件 | Macro-F1 | BA | MDD召回 | AUC | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|
| M3-frozen-ST | 0.781599 | 0.783889 | 0.661111 | 0.847222 | 0.519286 | 0.168269 |
| M3-lora-ST | 0.781599 | 0.783889 | 0.661111 | 0.831944 | 0.518695 | 0.167982 |

主口径为每种子五折均值，再对 3 种子取均值。每折、每种子及合并 OOF 指标保存在完整指标中。
LoRA 相对同版本冻结控制 F1 变化 +0.0000 个百分点，0/3 种子改善；BA 变化 +0.0000 个百分点，召回变化 +0.0000 个百分点。概率门槛通过。

| 种子 | 冻结 F1 | LoRA F1 | 变化（百分点） |
|---|---:|---:|---:|
| 11 | 0.768333 | 0.768333 | +0.0000 |
| 22 | 0.758131 | 0.758131 | +0.0000 |
| 33 | 0.818333 | 0.818333 | +0.0000 |

![分类结果](<C:/Users/jin  qiyan/Desktop/lunwen2/output/analysis/model_opt_lora_20261001/lora_classification.png>)

## 分支、训练与数值记录

| 条件 | train F1 | val F1 | test F1 | 语音分支 test F1 | 文本分支 test F1 |
|---|---:|---:|---:|---:|---:|
| M3-frozen-ST | 0.964183 | 0.821111 | 0.781599 | 0.766455 | 0.668187 |
| M3-lora-ST | 0.964183 | 0.821111 | 0.781599 | 0.766455 | 0.668187 |

15 个候选的 LoRA 相对初值位移 L2 中位数为 0.050772，范围 [0.041287, 0.055330]。候选全局梯度裁剪比例 100.0000%。这是更新确实发生的机制记录，不能直接证明泛化原因。
111 条重复种子测试记录中，错变对 0、对变错 0；这些记录来自同一 37 人。
缓存重放相对原完整模型题 CLS 最大绝对误差 5.7220459e-06，低于登记容差；同一计算路径重复和重载逐位一致。由于批次长度和 float32 计算路径不同，以本轮新冻结控制判定 LoRA 效应，第二批 0.781599 仅作历史开发参考。

![训练与概率结果](<C:/Users/jin  qiyan/Desktop/lunwen2/output/analysis/model_opt_lora_20261001/lora_training_probability.png>)

## 判定与停止

分类通过须 F1 提高至少 1 个百分点、至少 2/3 种子改善、BA 不降、召回下降不超过 0.05；NLL/Brier 同时降低才记概率改善。各项判定见 comparisons.json。
本轮预算已经结束，不自动增加 rank、学习率、微调层数或改换 HuBERT；没有新增 EEG 或组合训练。后续计划 v1 的固定候选筛查已完成，只有通过门槛的开发候选保留。
三批累计 150 个新神经模型、36,000 次更新，以及 10 次线性拟合；复用控制、零更新探针单列。37 人已反复用于开发，不能据此宣称独立人群泛化或临床验证。

## 产物入口

- [固定执行附表](<C:/Users/jin  qiyan/Desktop/lunwen2/output/plans/模型优化第三批_MacBERT有限微调执行附表_v1.md>)
- [完整融合指标](<C:/Users/jin  qiyan/Desktop/lunwen2/tridep_ccml/experiments/MODEL-OPT-v1-LORA-37-fulltext/metrics.json>)
- [分支指标](<C:/Users/jin  qiyan/Desktop/lunwen2/tridep_ccml/experiments/MODEL-OPT-v1-LORA-37-fulltext/branch_metrics.json>)
- [门槛判定](<C:/Users/jin  qiyan/Desktop/lunwen2/tridep_ccml/experiments/MODEL-OPT-v1-LORA-37-fulltext/comparisons.json>)
- [特征与训练名单](<C:/Users/jin  qiyan/Desktop/lunwen2/tridep_ccml/experiments/MODEL-OPT-v1-LORA-37-fulltext/feature_manifest.json>)
- [逐人配对变化](<C:/Users/jin  qiyan/Desktop/lunwen2/tridep_ccml/experiments/MODEL-OPT-v1-LORA-37-fulltext/paired_subject_changes.json>)
- [训练更新诊断](<C:/Users/jin  qiyan/Desktop/lunwen2/tridep_ccml/experiments/MODEL-OPT-v1-LORA-37-fulltext/training_diagnostics.json>)

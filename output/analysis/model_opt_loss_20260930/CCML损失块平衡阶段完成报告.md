# CCML损失块平衡阶段完成报告

本阶段完成。损失块均值方案通过分类开发门槛的范围为：loss_ST。
继续跳过数据缺陷排查，以数据无缺陷为工作假设。只调整单分支损失块的模态聚合，一致与互补损失、模型、输入、优化器、裁剪和推理均与控制一致。

## 执行与验收

完成 30 组新训练、7,200 次更新，复用首批 30 个完全同配置的冻结控制。另完成 90 个训练内损失块梯度探针，优化器更新为零。30 对初值、随机状态、参数量、训练名单及预算一致；60 个模型精确重载，2,220 条逐人输出保存，1,110 条控制输出与首批精确一致，180 组折内指标独立复算通过。
所有新增模型完成并锁定后才统一生成本轮新测试结果。没有使用 val 选轮、没有超参数或阈值搜索。

## 同版本结果

| 条件 | Macro-F1 | BA | MDD召回 | AUC | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|
| M2-sum-ST | 0.753386 | 0.757778 | 0.622222 | 0.841667 | 0.521717 | 0.169394 |
| M2-mean-ST | 0.781599 | 0.783889 | 0.661111 | 0.847222 | 0.519312 | 0.168279 |
| M2-sum-EST | 0.763172 | 0.765556 | 0.611111 | 0.843611 | 0.510886 | 0.166082 |
| M2-mean-EST | 0.748569 | 0.751667 | 0.583333 | 0.855000 | 0.508332 | 0.164673 |

主口径为每种子五折均值，再对 3 种子取均值；每折和合并 OOF 指标单独保存在 metrics.json。

| 比较 | F1变化（百分点） | 改善种子 | BA变化 | 召回变化 | 分类门槛 | 概率门槛 |
|---|---:|---:|---:|---:|---|---|
| loss_ST | +2.8213 | 2/3 | +0.026111 | +0.038889 | 通过 | 通过 |
| loss_EST | -1.4603 | 0/3 | -0.013889 | -0.027778 | 未通过 | 通过 |
| EEG_sum | +0.9786 | 2/3 | +0.007778 | -0.011111 | 未通过 | 通过 |
| EEG_mean | -3.3030 | 0/3 | -0.032222 | -0.077778 | 未通过 | 通过 |

EEG 增量的差中差为 -4.2816 个百分点；这是描述量，不等于显著性或独立因果证明。

![分类及概率结果](<C:/Users/jin  qiyan/Desktop/lunwen2/output/analysis/model_opt_loss_20260930/loss_balance_results.png>)

## 损失块机制诊断

下表比较各末轮模型在固定 train 小批次上的梯度。范数比为该模型实际使用的单分支损失块梯度与一致加互补梯度之比；余弦在同一个模块参数上计算，负值仅说明该次局部方向相反。eval 模式不含 dropout 随机性，不能代表完整训练轨迹或直接证明泛化原因。

| 条件 | 模块 | 梯度范数比中位数 | 余弦中位数 | 负余弦比例 |
|---|---|---:|---:|---:|
| M2-sum-ST | speech_projection | 1.449404 | 0.208465 | 0.2667 |
| M2-sum-ST | text_projection | 1.138903 | 0.488611 | 0.1333 |
| M2-mean-ST | speech_projection | 0.998316 | 0.222554 | 0.2667 |
| M2-mean-ST | text_projection | 0.766458 | 0.502040 | 0.1333 |
| M2-sum-EST | speech_projection | 1.537725 | 0.572230 | 0.0667 |
| M2-sum-EST | text_projection | 1.757122 | 0.798166 | 0.0667 |
| M2-mean-EST | speech_projection | 0.787857 | 0.560433 | 0.1333 |
| M2-mean-EST | text_projection | 0.721360 | 0.744826 | 0.0667 |

![同模块梯度关系](<C:/Users/jin  qiyan/Desktop/lunwen2/output/analysis/model_opt_loss_20260930/loss_block_gradients.png>)

## 停止规则与后续范围

分类通过须 F1 提高至少 1 个百分点、至少 2/3 种子改善、BA 不降、MDD召回下降不超过 0.05。概率改善单列，不以它替代分类门槛。
本轮不追加损失系数搜索；第一批没有通过门槛的改动，因此不组合。有限骨干微调未执行，仍需单独登记分支、可训练范围、预算与折内特征生成规则。
全部证据仍来自已反复用于开发的 37 人；多个种子不是新增受试者，也不是独立人群验证。

## 产物入口

- [固定执行附表](<C:/Users/jin  qiyan/Desktop/lunwen2/output/plans/模型优化第二批_CCML损失块平衡执行附表_v1.md>)
- [完整指标](<C:/Users/jin  qiyan/Desktop/lunwen2/tridep_ccml/experiments/MODEL-OPT-v1-LOSS-37-fulltext/metrics.json>)
- [逐项门槛判断](<C:/Users/jin  qiyan/Desktop/lunwen2/tridep_ccml/experiments/MODEL-OPT-v1-LOSS-37-fulltext/comparisons.json>)
- [逐人配对变化](<C:/Users/jin  qiyan/Desktop/lunwen2/tridep_ccml/experiments/MODEL-OPT-v1-LOSS-37-fulltext/paired_subject_changes.json>)
- [训练内梯度诊断](<C:/Users/jin  qiyan/Desktop/lunwen2/tridep_ccml/experiments/MODEL-OPT-v1-LOSS-37-fulltext/mechanism_summary.json>)

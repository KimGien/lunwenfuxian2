# 九篇 EEG 接入 TRI-DEP 融合完成报告

25 个 EEG 条件快照、375 组种子折快照、四种融合视图共 1,500 个评价单元全部完成。375 组全部复用已保存概率，无补推理；本轮新增基础模型训练、融合网络训练、优化器更新、模型加载、神经前向和 ASR 均为 0。

主要协议的系统保留条件通过 0 条候选路线。本轮不替换已保存系统；具体配对涨分、概率收益和次要协议结果仍完整保留。

A 为固定 EEG/语音/文本=0.05/0.35/0.60；B 为原3人验证集上重新选权；C 为三模态等权；D 为历史EST权重直接替换。A 是主要协议，B/C/D 不因分数更高而改为主要结果。

## 基线重新计算结果

| 对照 | Macro-F1 | BA | MDD召回 | AUC | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|
| 历史固定ST | 0.724180 | 0.732222 | 0.661111 | 0.716944 | 0.782353 | 0.230029 |
| 历史固定EST | 0.724180 | 0.732222 | 0.661111 | 0.725833 | 0.773737 | 0.228925 |
| A去EEG并保持S/T比例 | 0.719788 | 0.724444 | 0.622222 | 0.782222 | 0.586614 | 0.187045 |
| 原EEG按A融合 | 0.725503 | 0.730000 | 0.616667 | 0.782222 | 0.568573 | 0.186360 |
| 论文公布ST权重0.45/0.55 | 0.663895 | 0.675556 | 0.527778 | 0.782222 | 0.582916 | 0.186748 |
| 语音文本等权 | 0.648101 | 0.661667 | 0.566667 | 0.781111 | 0.584309 | 0.188190 |
| 原EEG三模态等权 | 0.682487 | 0.684444 | 0.572222 | 0.709444 | 0.606431 | 0.207178 |

A的ST消融为(0.35pS+0.60pT)/0.95。历史ST的0.724180不代替该消融。原EEG来源为阶段0原模型，匹配LaBraM来源为九篇计划中的相应输入窗长和训练路线。

## 九篇第一轮主结果

Δ均为Macro-F1的绝对差值，0.01相当于1个百分点。纠错/新增错误以111条种子测试记录计数，去重受试者数另见明细。

| 候选 | A融合F1 | Δ匹配LaBraM | Δ原EEG同协议 | Δ匹配ST | 纠错/新增错误 | 主要系统保留 |
|---|---:|---:|---:|---:|---:|---|
| EEGPT | 0.717884 | -0.002593 | -0.007619 | -0.001905 | 1/1 | 未通过 |
| CSBrain | 0.719788 | -0.008466 | -0.005714 | +0.000000 | 0/0 | 未通过 |
| REVE | 0.717884 | -0.010370 | -0.007619 | -0.001905 | 1/1 | 未通过 |
| LUNA | 0.728254 | +0.000000 | +0.002751 | +0.008466 | 1/0 | 未通过 |
| CodeBrain | 0.719788 | -0.008466 | -0.005714 | +0.000000 | 0/0 | 未通过 |
| BIOT | 0.712011 | -0.016243 | -0.013492 | -0.007778 | 0/1 | 未通过 |
| EEGMamba | 0.719788 | -0.008466 | -0.005714 | +0.000000 | 0/0 | 未通过 |
| EEGConformer | 0.735873 | +0.007619 | +0.010370 | +0.016085 | 2/0 | 未通过 |
| EEG2Rep | 0.719788 | -0.008466 | -0.005714 | +0.000000 | 0/0 | 未通过 |

## 七条有限微调扩展

| 候选 | A融合F1 | Δ匹配LaBraM | Δ原EEG同协议 | Δ匹配ST | 纠错/新增错误 | 主要系统保留 |
|---|---:|---:|---:|---:|---:|---|
| EEGPT | 0.735873 | +0.007619 | +0.010370 | +0.016085 | 2/0 | 未通过 |
| CSBrain | 0.727407 | -0.000847 | +0.001905 | +0.007619 | 1/0 | 未通过 |
| REVE | 0.728254 | +0.000000 | +0.002751 | +0.008466 | 1/0 | 未通过 |
| LUNA | 0.728254 | +0.000000 | +0.002751 | +0.008466 | 1/0 | 未通过 |
| CodeBrain | 0.719788 | -0.008466 | -0.005714 | +0.000000 | 0/0 | 未通过 |
| BIOT | 0.719788 | -0.008466 | -0.005714 | +0.000000 | 0/0 | 未通过 |
| EEGMamba | 0.727407 | -0.000847 | +0.001905 | +0.007619 | 1/0 | 未通过 |

微调路线所选检查点阶段如下；best处于热身阶段者不称为编码器更新后的最佳模型。

| 候选 | 所选阶段及组数 |
|---|---|
| EEGPT | lastblock_finetune: 7；head_only_warmup: 8 |
| CSBrain | lastblock_finetune: 9；head_only_warmup: 6 |
| REVE | lastblock_finetune: 12；head_only_warmup: 3 |
| LUNA | lastblock_finetune: 8；head_only_warmup: 7 |
| CodeBrain | lastblock_finetune: 12；head_only_warmup: 3 |
| BIOT | lastblock_finetune: 11；head_only_warmup: 4 |
| EEGMamba | head_only_warmup: 9；lastblock_finetune: 6 |

## 全部协议与匹配控制

| 条件快照 | A固定权重 | B重新选权 | C等权 | D旧权重 |
|---|---:|---:|---:|---:|
| EEGPT-frozen-smallhead@best | 0.717884 | 0.708466 | 0.628022 | 0.724180 |
| CSBrain-frozen-smallhead@best | 0.719788 | 0.705089 | 0.639874 | 0.724180 |
| REVE-frozen-smallhead@best | 0.717884 | 0.731799 | 0.612678 | 0.724180 |
| LUNA-frozen-smallhead@best | 0.728254 | 0.705089 | 0.701996 | 0.724180 |
| CodeBrain-frozen-smallhead@best | 0.719788 | 0.731799 | 0.601435 | 0.724180 |
| BIOT-frozen-smallhead@best | 0.712011 | 0.724180 | 0.741188 | 0.724180 |
| EEGMamba-frozen-smallhead@best | 0.719788 | 0.712708 | 0.618339 | 0.724180 |
| EEGConformer-scratch@best | 0.735873 | 0.739418 | 0.628974 | 0.724180 |
| EEG2Rep-supervised@best | 0.719788 | 0.731799 | 0.655667 | 0.724180 |
| LaBraM-native4s-frozen-smallhead@best | 0.720476 | 0.724180 | 0.722734 | 0.724180 |
| LaBraM-native5s-frozen-smallhead@best | 0.728254 | 0.724180 | 0.709000 | 0.724180 |
| CBraMod-frozen-smallhead@best | 0.728254 | 0.712708 | 0.677625 | 0.724180 |
| EEGPT-lastblock-finetune@best | 0.735873 | 0.705089 | 0.648842 | 0.724180 |
| CSBrain-lastblock-finetune@best | 0.727407 | 0.712708 | 0.630800 | 0.724180 |
| REVE-lastblock-finetune@best | 0.728254 | 0.731799 | 0.639689 | 0.724180 |
| LUNA-lastblock-finetune@best | 0.728254 | 0.712708 | 0.679715 | 0.724180 |
| CodeBrain-lastblock-finetune@best | 0.719788 | 0.708466 | 0.602784 | 0.724180 |
| BIOT-lastblock-finetune@best | 0.719788 | 0.724180 | 0.697035 | 0.724180 |
| EEGMamba-lastblock-finetune@best | 0.727407 | 0.731799 | 0.647890 | 0.724180 |
| LaBraM-native4s-lastblock-finetune@best | 0.728254 | 0.731799 | 0.622070 | 0.724180 |
| LaBraM-native5s-lastblock-finetune@best | 0.728254 | 0.731799 | 0.638339 | 0.724180 |
| EEGPT-mean-projection4s-originalCNN-300ep@60 | 0.728254 | 0.724180 | 0.618095 | 0.724180 |
| EEGPT-mean-projection4s-originalCNN-300ep@300 | 0.701640 | 0.724180 | 0.601438 | 0.724180 |
| LaBraM-native4s-originalCNN-300ep@60 | 0.717884 | 0.724180 | 0.689861 | 0.724180 |
| LaBraM-native4s-originalCNN-300ep@300 | 0.717884 | 0.724180 | 0.699146 | 0.724180 |

以下只列B/C通过完整系统门槛的路线，作为次要协议结果，不替代A。完整门槛要求同时对匹配LaBraM、原EEG同协议和ST通过配对门槛，并严格优于历史ST/EST。

- B：EEGConformer-scratch@best。
- C：无。

## 延长EEG训练能否传递为融合收益

| 骨干 | A融合60轮F1 | A融合300轮F1 | ΔF1 | ΔBA | ΔMDD召回 |
|---|---:|---:|---:|---:|---:|
| EEGPT | 0.728254 | 0.701640 | -0.026614 | -0.027778 | -0.038889 |
| LaBraM | 0.717884 | 0.717884 | +0.000000 | +0.000000 | +0.000000 |

本表保持A融合公式和S/T不变，只比较已登记的60/300轮快照。B的轮次差值还包含重新选权影响，见results.json的duration字段。未选择120/200轮中的测试最优快照。

## 选权及概率收益

| 条件快照 | B选中EEG非零组数 | 存在非零EEG并列最优组数 | 最优并列组合数范围 | A对匹配ST的NLL/Brier同时改善 |
|---|---:|---:|---|---|
| EEGPT-frozen-smallhead@best | 3/15 | 15/15 | 2–199 | 是 |
| CSBrain-frozen-smallhead@best | 3/15 | 15/15 | 8–125 | 否 |
| REVE-frozen-smallhead@best | 3/15 | 15/15 | 15–201 | 否 |
| LUNA-frozen-smallhead@best | 3/15 | 15/15 | 4–198 | 是 |
| CodeBrain-frozen-smallhead@best | 3/15 | 15/15 | 1–201 | 是 |
| BIOT-frozen-smallhead@best | 3/15 | 15/15 | 42–194 | 是 |
| EEGMamba-frozen-smallhead@best | 3/15 | 15/15 | 3–157 | 否 |
| EEGConformer-scratch@best | 3/15 | 15/15 | 2–187 | 是 |
| EEG2Rep-supervised@best | 2/15 | 15/15 | 6–140 | 是 |
| LaBraM-native4s-frozen-smallhead@best | 3/15 | 15/15 | 62–201 | 是 |
| LaBraM-native5s-frozen-smallhead@best | 3/15 | 15/15 | 50–201 | 是 |
| CBraMod-frozen-smallhead@best | 3/15 | 15/15 | 7–198 | 是 |
| EEGPT-lastblock-finetune@best | 2/15 | 15/15 | 10–224 | 否 |
| CSBrain-lastblock-finetune@best | 3/15 | 15/15 | 6–118 | 是 |
| REVE-lastblock-finetune@best | 3/15 | 15/15 | 6–194 | 否 |
| LUNA-lastblock-finetune@best | 3/15 | 15/15 | 6–186 | 是 |
| CodeBrain-lastblock-finetune@best | 3/15 | 15/15 | 1–195 | 是 |
| BIOT-lastblock-finetune@best | 3/15 | 15/15 | 34–192 | 否 |
| EEGMamba-lastblock-finetune@best | 3/15 | 15/15 | 1–141 | 是 |
| LaBraM-native4s-lastblock-finetune@best | 3/15 | 15/15 | 5–198 | 是 |
| LaBraM-native5s-lastblock-finetune@best | 3/15 | 15/15 | 5–199 | 是 |
| EEGPT-mean-projection4s-originalCNN-300ep@60 | 0/15 | 15/15 | 2–159 | 否 |
| EEGPT-mean-projection4s-originalCNN-300ep@300 | 0/15 | 13/15 | 1–158 | 否 |
| LaBraM-native4s-originalCNN-300ep@60 | 3/15 | 14/15 | 2–201 | 是 |
| LaBraM-native4s-originalCNN-300ep@300 | 3/15 | 15/15 | 3–201 | 是 |

EEG权重为零受验证集和并列规则影响，不单独说明EEG无效；非零权重也不是有效性证据。NLL和Brier同时改善是独立的概率收益，不替代分类门槛。

## 验证和成本

已核对 9899 个绑定来源文件及检查点，输入共13,875条。独立实现重新连接原始批次和S/T来源，逐条复核55,500条融合概率、6,105条基线概率、11,100条测试纠错记录，并用NumPy重做375次候选选权及30次基线网格重放、用scikit-learn复算全部111组汇总。所有检查通过。

候选选权375次、验证候选评分86,625次；基线重放另含30次网格、3,780次候选评分。执行耗时254.82秒，独立核验耗时31.81秒；均含本地文件读取/哈希，不能解释为神经网络训练耗时。

主指标先平均种子内五折，再平均三个种子；逐种子37人OOF另列。阈值固定p>0.5，相等为HC；NLL真类概率下限1e-15，裁剪不影响融合输入或分类。训练/验证指标保留为诊断，验证参与EEG选模及B选权，不能作为独立性能证据。

## 结论范围和文件

当前37人已反复用于开发，111条种子记录仍只有37名受试者。该报告是固定融合方案的开发比较，不是新的盲测或统计显著性结论。不同输入参考、幅值、窗长、训练范式仍按EEG原合同解释；EEGMamba仍为作者公开实现适配。未启动十篇融合或CCML/残差融合头训练。

[完整指标和配对判定](results.json) · [全部100行比较表](summary.csv) · [逐折权重与并列统计](selections.json) · [逐人预测](predictions.jsonl) · [纠错与新增错误](paired_changes.jsonl) · [独立核验](independent_verification.json) · [实际成本](cost.json)

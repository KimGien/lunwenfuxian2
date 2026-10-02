# TC 父实验依赖审计

审计时间：2026-09-17T12:11:39.433380+00:00

结论：未发现父实验数据、权重、划分锁或运行环境的硬阻塞。

本审计不训练、不重跑 ASR、不提取骨干特征、不读取原始访谈内容；仅核对缓存、元数据、哈希和保存概率。新 TC 实现和表征导出状态另行验收。

| 项目 | 结果 |
|---|---|
| cohort_lock | 通过 |
| backbone_features_and_provenance | 通过 |
| parent_checkpoints | 通过 |
| saved_probabilities_and_metrics | 通过 |

通过项包括：37 人（16 MDD、21 HC）及原五折锁；185 份指定骨干缓存（EEG 37、语音 111、全文文本 37）；45 个同种子同折父权重的哈希/训练名单/配置/有限性/严格加载；6 份旧概率文件共 1110 行的覆盖、阈值、融合和指标。实际覆盖数量以 JSON 明细为准。

可用环境：`S00/.venv/Scripts/python.exe`，PyTorch 2.8.0+cu126，CUDA 12.6，NVIDIA GeForce RTX 4080 Laptop GPU。系统默认 Python 缺少依赖，仅需使用现有虚拟环境。

## 阻塞

无父依赖硬阻塞。

## 持续披露的限制

- L01：37-person fixed-split development comparison; all outer tests were previously observed and are not new independent validation.
- L02：Random frozen speech CNN-BiGRU encoder and single-position attention retained; match seed-specific features.
- L03：EEG native scale, index mapping and backbone checkpoint physical compatibility remain disclosed uncertainties.
- L04：Fulltext and first-300-second EEG policies are documented local adaptations rather than a verified exact paper reproduction.
- L05：System-default Python lacks ML dependencies; use the verified S00/.venv/Scripts/python.exe interpreter.
- L06：This audit verifies hashed cached features and existing data audit provenance; it does not redo raw waveform/EEG content validation or execute training.

## TC-00 旧概率重算

| 融合 | Macro-F1 | MDD recall | Balanced accuracy |
|---|---:|---:|---:|
| 原验证集选权重 | 0.72417989 | 0.66111111 | 0.73222222 |
| 相同原始概率等权 | 0.68248677 | 0.57222222 | 0.68444444 |

以上为既有父预测的重新计算，不是新头训练成绩；口径为每种子五折均值再取三种子均值。

# 九篇 EEG 与 TRI-DEP 复现步骤

本次归档按训练和评价流程分成 12 次提交，保留原仓库历史。提交时间为上传时间，不伪造当时训练时间。源码为本地执行版本，上传过程未训练或重新选模。

[单模态与全部概率融合比较](EEG单模态与TRIDEP融合比较.md)

## 顺序与入口

| 顺序 | 内容 | 主要代码或报告 |
|---|---|---|
| 01 | 原 CBraMod 与 TRI-DEP 来源锁定 | [入口](../tridep_ccml/stage0.py) |
| 02 | LaBraM 与 EEGPT 配对控制及恢复 | [入口](../tridep_ccml/tridep_eegpt/train_eval.py) |
| 03 | 九篇 EEG 协议 作者资源与输入适配 | [入口](../tridep_ccml/eeg_all_candidates/protocol.py) |
| 04 | 冻结小头 自监督及原 CNN 长训练 | [入口](../tridep_ccml/eeg_all_candidates/frozen_heads.py) |
| 05 | 有限微调 确定性修复与训练恢复 | [入口](../tridep_ccml/eeg_all_candidates/continuation_queue_v14.py) |
| 06 | 统一评价 成本与完成验收 | [入口](../output/analysis/eeg_all_candidates_v1_20261008/九候选统一完成报告.md) |
| 07 | 原 CBraMod 基线下的保留规则修订 | [入口](../output/analysis/eeg_all_candidates_v1_20261008/九候选统一完成报告_原TRIDEP基线修订版_20261013.md) |
| 08 | 接入 TRI-DEP 的四种概率融合与核验 | [入口](../output/analysis/eeg_tridep_fusion_v1_20261013/九篇EEG接入TRI-DEP融合完成报告.md) |
| 09 | 十五 EEG 版本与五种融合方法训练 | [入口](../tridep_ccml/fusion_eeg_grid_v2/README.md) |
| 10 | 全组合汇总与诊断勘误 | [入口](../output/analysis/fusion_eeg15_methods5_v2_20261013/diagnosis_20261013/诊断结论与验收勘误.txt) |
| 11 | 后续诊断修复代码快照 未声明训练完成 | [入口](../tridep_ccml/fusion_eeg_grid_v3/README.md) |
| 12 | 单模态与融合比较及复现导航 | [入口](../docs/EEG单模态与TRIDEP融合比较.md) |

## 实际入口与完成状态

九候选协议先登记，作者源码与权重经哈希核验后才能训练。冻结路线使用 frozen_heads.py，EEGConformer 使用 supervised.py；EEG2Rep 的 ssl.py/ssl_v2.py 与 supervised.py 区分无标签预训练和监督阶段；paired_replay.py 负责 EEGPT/LaBraM 原 CNN 的 60/120/200/300 轮。有限微调原实现为 finetune.py/finetune_v2.py，后续 v3 至 v12、Q2 至 Q14 和工程修复按历史记录保留，不能把目录内每个版本依次当作新实验重跑。最终统一评价使用 evaluate_v13.py，旧失败入口同时保留。

九候选 360 次正式拟合完成，统一监督评价涉及 345 次拟合与 435 个固定快照。当前单模态保留政策为严格高于原 CBraMod＋CNN 的 0.4494011544011543，修订后保留 3 个候选家族的 4 条路线，另有 LaBraM 5 条对照。该政策是查看结果后的开发修订。原匹配 LaBraM 门槛及阴性结果不删除。

概率融合使用 eeg_tridep_fusion/run.py，25 个条件快照、375 个种子折单元，计算 A/B/C/D 四种规则；verify_v2.py 为最终通过的独立复核。全部复用旧预测，新增训练为 0。历史 verify.py 的浮点求和末位问题与修复记录保留。

后续 v2 的 15 个 EEG 版本×5 融合方法实际训练 1200 次。旧结果存在标准化尺度问题和 QMF 作者损失一致性漏验，须连同诊断勘误阅读，不能把旧验收证书解释为这些问题已通过。v3 仅发布本次读取时的后续代码和计划快照，不声明整个新计划完成，也不启动或恢复其队列。

## 运行环境与本地依赖

从仓库根目录运行；基础环境见 S00/requirements-core.txt 和 S00/requirements-main.lock.txt 及各源码 imports。九候选 Windows 路线与 EEGMamba 的 WSL/Linux 路线使用不同环境，不能直接混用二进制包。作者源码目录保留代码、说明与许可证；精确来源和文件哈希见 author_source_binding.json、download_manifest.json、LaBraM 的 labram_asset_manifest.json 及融合 upstream_sources.json。非代码上游资产和预训练权重按固定 revision、URL、哈希自行恢复，不能用最新版本替代。

需要本人合法持有的 MODMA EEG/语音、诊断及分折元数据、原 S00 同种子同折 E/S/T 检查点、原特征、逐人概率和本地执行登记。公开仓库不新增这些个体数据，不包含模型权重、特征、转写、虚拟环境或系统进程/电源日志。缺少这些文件时，运行历史训练器或完整哈希审计会按设计拒绝，不能绕过来源门。

旧来源审计中列出的哈希仍对应本地原件。汇总导出与去除受试者编号后的公开文档在本次清单中分别记录 source_sha256 和 published_sha256；上传清单仅证明公开文件完整性，不替代旧实验审计。历史报告指向未公开数据的链接保留为依赖说明。

可在仅有公开文件时检查协议与本次表格：

```text
python -B -m unittest tridep_ccml.eeg_tridep_fusion.test_protocol
python -B scripts/verify_eeg_publication.py
```

复现历史运行必须先恢复完整本地来源，阅读各模块 README 的当前入口和失败恢复边界；不要直接恢复历史 PID，也不要重复已成功且只允许一次的诊断。
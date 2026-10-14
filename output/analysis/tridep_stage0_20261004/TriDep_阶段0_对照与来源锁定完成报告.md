# TriDep 阶段0：对照与来源锁定完成报告

完成日期：2026-10-04。第0阶段通过，新增训练和优化更新均为0；阶段1及以后尚未启动。

来源固定为原 S00 B0-37-fulltext：S和E继承B0-37，T采用全文版。45个同种子同外折检查点均按原class严格加载，全部eval、停止梯度、inference_mode，原参数与buffer逐位不变。

1665个分支概率重放通过，最大绝对误差0；最后Linear接口重放误差为0。15份37×3×128表示与历史TC准备缓存逐位一致，45个分支重复同一输入输出逐位相同。7087个来源文件推理前后SHA-256相同。

## 固定来源

- S：`S00/runs/B0-37/seed_<seed>/fold_<fold>_speech.pt`；29×512输入，原70轮。
- T：`S00/runs/B0-37-fulltext/seed_<seed>/fold_<fold>_text.pt`；29×768全文CLS输入，原50轮。
- E：`S00/runs/B0-37/seed_<seed>/fold_<fold>_eeg.pt`；60×200原输入，原60轮。
- 新缓存：`features/seed_<seed>/fold_<fold>.npz`；包含37×3×128表示、37×3 MDD概率、人员、标签和split；维序E/S/T。每份JSON绑定原权重、输入、模型代码、历史概率和导出脚本哈希。

活动分折SHA-256：`a28ebe92c94428128313586d3a5524b2f03befcbc473215f4b0b2dcb49c3284d`。原38人分折仅移除[受试者编号已省略]，顺序保持。原始数据、staged副本、两种ASR转写、诊断工作簿、固定骨干资产、随机语音编码器、输入、权重和历史结果均已核对。完整逐文件来源见 [source_manifest.json](source_manifest.json)，逐折训练人员及权重见 [checkpoint_provenance.json](checkpoint_provenance.json)。

## 原ST和EST控制

EST主控制直接复用原全文版的val选权和精确历史概率，全部555条train/val/test记录及历史主F1一致。原流程没有独立保存ST控制；本阶段按事前附表构造确定性ST控制：令E权重为0，在原0.05步长的21个S/T权重候选中只按本折3人val Macro-F1选权，平局保留低S权重。共315个评分、15次选择、0次模型拟合。ST等权、EST等权仅作补充，不按test排序改主控制。

主口径：每种子五外折测试均值，再取种子11/22/33均值；p>0.5判MDD，相等判HC。AUC/NLL/Brier也按同口径平均，每种子37人OOF合并值另列于 [metrics.json](metrics.json)。

| 控制 | Macro-F1 | BA | MDD召回 | AUC | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|
| ST-val-weighted | 0.724180 | 0.732222 | 0.661111 | 0.716944 | 0.782353 | 0.230029 |
| EST-original-val-weighted | 0.724180 | 0.732222 | 0.661111 | 0.725833 | 0.773737 | 0.228925 |
| ST-equal | 0.648101 | 0.661667 | 0.566667 | 0.781111 | 0.584309 | 0.188190 |
| EST-equal | 0.682487 | 0.684444 | 0.572222 | 0.709444 | 0.606431 | 0.207178 |

| 种子 | ST五折平均F1 | 原EST五折平均F1 |
|---|---:|---:|
| 11 | 0.742698 | 0.742698 |
| 22 | 0.687143 | 0.687143 |
| 33 | 0.742698 | 0.742698 |

原EST相对固定ST的Macro-F1差为0。111条重复种子测试记录中新增正确0条、新增错误0条；仍仅有37位受试者，逐人变化见 [paired_test_changes.jsonl](paired_test_changes.jsonl)。

原EST有14/15组的EEG权重为0；只有seed22/fold3使用(0.15,0,0.85)，对应ST选(0,0,1)。因此原EST的名称不能证明EEG在所有折有贡献，本次也未验证新的EEG收益。全部21个ST候选与选择结果保存在 [validation_weight_selections.json](validation_weight_selections.json)。

## 隔离边界及保留限制

原每折train/val/test人员互斥，固定S/T监督训练没有使用本折原3人val或外折test；原val已经用于历史融合选权，人数极少。若从本折train再切内层val，固定S/T曾监督训练过那些人员，因此本缓存不能当作合法内层OOF，后续不得据此开展未登记选型。

37人均为已反复观察的开发人群。新增ST控制和重放不恢复盲测身份，也不产生口音、词汇能力或跨域泛化证据。原S00随机冻结语音编码器、退化注意力、短音频空特征和EEG原幅值/导联适配不确定性继续保留；本阶段不替换E2v2、G1、M2或其他改过S/T的版本。

全文转写与原第一段文本不同的录音共130份；原语音无切片记录三种子共288条（并非新增数据）。详情见 [data_and_split_audit.json](data_and_split_audit.json)。

## 交付与后续

阶段0已完成，可作为阶段1 EEG结构/骨干与阶段3 ST外部小头的原始控制入口。两阶段必须先各自登记条件、参数、训练预算和停止规则，本次没有启动后续训练。

- [执行附表](../../plans/TriDep_阶段0_对照与来源锁定执行附表_v1_20261004.md)
- [完成状态与推理核验](status.json)
- [分支逐人概率](branch_probabilities.jsonl)
- [四个控制逐人预测](control_predictions.jsonl)
- [完整各折/种子/合并OOF指标](metrics.json)
- [缓存入口](feature_bundles.json)
- [产物哈希](output_manifest.json)

复核入口：`S00/.venv/Scripts/python.exe tridep_ccml/stage0.py --verify`。它重新核对源文件及产物哈希、人员覆盖、全部指标、原EST历史结果、缓存和来源关系，不重训原S/T。

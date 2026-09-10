# TRI-DEP S00：B0-37 基线准备

对应 `baseline备选/2510.14922v2.pdf` 和 `output/plans/TRI-DEP_递进实验路线.md`。这里保存 B0 的来源、数据审计、环境、配置、验证工具和运行入口。**尚未完成真实 B0 训练，不能把合成输入检查解释为论文指标复现。**

## 当前结论

**已按用户指令启用 B0-37；所有后续 MODMA 实验固定为同一批 37 人。** 唯一人群配置为 `configs/cohort.json`，活动划分为 `manifests/fold_assignments.json`。人数、每折类别分布、缓存和结果规则见 [37 人统一实验协议](reports/37人统一实验协议.md)。

- 官方 `main` 源码保留在 `../TRI-DEP官方`，提交 `e54fd68004843d20b618af307c0e200bb6dced07`。
- README 指向的 `hele/dev` 分支另存于 `upstream-dev`，固定提交 `be6bcafd8063cd04a325279742ff27dca24ea280`。这是准备时取得的开发快照，并不能证明等同于论文实验时的历史代码。
- 在作者五折的每个 train/val/test 集合中移除 `02010004`，其他人的折和顺序不变；37 人各作为测试对象一次，各折无交叉且各分组均含两类。作者原版 38 人划分另存于 `manifests/fold_assignments.official38.json`。
- 活动名单的两份 MODMA 标签表一致：**16 MDD、21 HC**。原始 ZIP、raw、CRC 和 SHA-256 保留；切换前的 38 人审计在 `archive/cohort38`。
- **37 份 EEG、1073 份录音的数据审计通过**，无解码错误、无本次严重噪声标记、无相同 PCM 的重复记录。被排除者第 23–28 题的损坏证据仍见 [历史恢复记录](recovery/恢复记录.md)，无需等待其恢复才继续 B0-37。
- staged 现仅有 37 人、1073 份音频；37 份 EEG 转换全部逐值相等，没有重采样、缩放或选通道。被排除者的旧 staged 文件夹移入 `archive/cohort38/staged/02010004`。
- 主环境为 Python 3.13、PyTorch 2.8.0+cu126；官方 CUDA wheel 的 SHA-256 已与官方索引比对。GPU 为 RTX 4080 Laptop。三分支分类头通过 GPU 前向、反向与同种子重复检查。
- MacBERT、代码实际使用的 HuBERT、CBraMod-Mumtaz 已固定本地权重及来源。前两者没有骨干参数缺失；CBraMod 的 209 个保留骨干键匹配，移除的重建头和原任务分类头不用于特征提取。
- 转写使用独立 `.asr-venv`，因为 WhisperX 3.3.1 不支持 Python 3.13。medium ASR 权重已固定；最终环境及转写验证状态见 `reports/asr_smoke.json` 和 `reports/preflight.json`，缺少报告时不可推断已通过。

## 仍需解决的输入和协议问题

1. 数据缺失阻塞已通过用户授权的人群调整解决：B0-37 与所有后续 MODMA 实验共用 37 人。原论文 38 人成绩仅作背景参考；方法增益以同环境、同划分的 B0-37 为准。以下模型与预处理协议仍待解决。
2. 论文描述 Chinese HuBERT Large，但发布代码实际加载 `xmj2002/hubert-base-ch-speech-emotion-recognition`，输出 768 维。准备包下载的是代码指向的权重，并保留这一差异。不能自行更换为 Large 后声称没有修改 B0。
3. 语音 CNN-BiGRU-Attention 在发布提取脚本中随机初始化后直接以 `eval/no_grad` 运行，没有加载编码器任务权重，也没有优化器训练。二维输入被扩成每个片段长度为 1 的序列，注意力权重因此恒为 1。输出实际为 512 维；部分默认参数仍写成 256/1024，训练配置文件名还含 `lstm`。这些均需明确采取“发布代码复现”还是有来源的修正版，不能悄悄修改并归因于后续 S01。
4. WhisperX 入口只保存第一个 ASR segment；论文写的是每题转写。完整转写与首段转写必须分别留存和命名。错误占位符 `[ERROR]` 不能当作正常答案输入模型。
5. 发布语音预处理丢弃不足 5 秒的录音，并追加末尾窗口；论文描述保留短录音。需要锁定短音频、尾段和缺失题政策。
6. 当前 `.set` 通道是 E1–E128 加 Cz。官方索引直接照录在配置中，实际对应电极标签与国际 10–20 名称仍需核实；转换保留 FDT 数值，未猜测与作者 `.mat` 的物理单位关系。不要直接额外乘除 `1e6` 或 `100`。
7. EEG 实测时长 300.356–335.504 秒，按 5 秒可产生 60–67 段；发布读取代码只补零，不截断，而全连接层按 60 段构造。需要核实是否固定截取前 300 秒，不能静默裁剪或换成自适应池化。

以上问题见 `reports/差异与阻塞.md`。数据和代码准备不等于这些科学协议已经得到作者确认。

## 固定的实验规范

`configs/b0.json` 仅选论文的三条最佳分支，未引入 S01 及后续模块：

| 分支 | 官方配置 | 输入 | 优化器／学习率 | 轮数 | batch |
|---|---|---|---|---|---|
| EEG | cbramod_mumtaz_conv | 60×200 | Adamax / 0.005 | 60 | 16 |
| 语音 | hubert_bigru_conv | 29×512 | Adamax / 0.005 | 70 | 8 |
| 文本 | macbert_lstm | 29×768 | Adamax / 0.0001 | 50 | 64 |

种子为 11、22、33；确认时加 44、55。42 仅作为作者默认种子的单独对照。每个折和模态重新设种子，训练固定轮数并取最后状态；不以测试表现选择模型。

标签 0=HC、1=MDD。概率均指 MDD 概率。阈值 0.5；恰等于 0.5 时按两类 softmax 的 argmax 约定取 HC。报告受试者级 macro-F1、MDD recall、balanced accuracy、各折各种子成绩、参数量与训练时间，折间标准差用 ddof=0。

融合在每折内部验证集上搜索步长 0.05、和为 1 的 EEG／语音／文本权重，并列按权重字典序选择；论文权重 0.05/0.35/0.60 仅保留为固定参考，不用全体测试标签重新寻找最优权重。这个选择协议是 plan 要求的防泄漏实现，与原全体预测网格搜索不能混为同一协议。未来如果根据外层测试结果选择 S01 等方案，必须把这些结果标为开发性结果。

EATD 不进入 S00 训练；Mumtaz 仅用于说明作者已发布 EEG 权重的训练来源，本步骤不新增微调，不能声称该权重对 Mumtaz 是独立外部验证。

## 本地复查

以下命令均在 `lunwen2` 目录执行；路径按当前工作区组织。

```powershell
S00/.venv/Scripts/python.exe S00/scripts/prepare.py
S00/.venv/Scripts/python.exe S00/scripts/audit_audio_signal.py
S00/.venv/Scripts/python.exe S00/scripts/audit_modma.py
S00/.venv/Scripts/python.exe S00/scripts/test_protocol.py
S00/.venv/Scripts/python.exe S00/scripts/smoke_models.py
S00/.venv/Scripts/python.exe S00/scripts/smoke_backbones.py
S00/.venv/Scripts/python.exe S00/scripts/run_b0.py
```

`audit_modma.py` 当前返回 0，37 人数据通过；`run_b0.py` 当前返回 2，因尚缺协议说明与完整特征。`smoke_*` 使用合成输入，不创建真实受试者预测或报告模型 F1。`cohort.py` 可单独验证活动人群和审计来源。

补充录音后不要覆盖已保留的原始压缩包；将新来源和 SHA-256 记录后更新工作副本，再运行数据审计和 staging。`stage_dataset.py` 遇到现有文件与来源不一致会停止，不覆盖已有数据。新来源接入需更新 raw manifest。

## 特征与训练入口

当前尚未生成真实完整三模态特征缓存或受试者预测。解决剩余协议问题后，放置 `S00/features/B0-37/protocol_resolution.json`，记录 `speech_encoder`、`hubert_identity`、`asr_segments`、`audio_segmentation`、`eeg_units_channels`、`eeg_length_policy` 六项选择及依据。该记录用于追溯，文字存在本身不证明方法正确。

每个已审计特征在 `features_manifest.json` 中占一条：

```json
{
  "cohort_id": "MODMA-37-v1",
  "folds_sha256": "a28ebe92c94428128313586d3a5524b2f03befcbc473215f4b0b2dcb49c3284d",
  "subject_id": "02010002",
  "modality": "text",
  "outer_fold": null,
  "seed": null,
  "path": "02010002/text_embedding_macbert.npy",
  "sha256": "填写实际文件的64位SHA256",
  "producer_id": "填写骨干revision、预处理配置、提取程序与权重hash",
  "fit_subject_ids": []
}
```

这里只展示格式，不提供假特征。冻结、未在 MODMA 拟合的骨干可跨折共享缓存；监督拟合或学习标准化的组件必须列出真实拟合受试者并指定折，且这些人必须属于该折 train。按种子随机初始化的编码器必须写明 seed，并保存初始化权重，不能填写 null 来跨种子共用。

准备好后先执行预检，再显式启动：

```powershell
S00/.venv/Scripts/python.exe S00/scripts/run_b0.py --seed 11
S00/.venv/Scripts/python.exe S00/scripts/run_b0.py --seed 11 --train
```

对 22、33 重复相同流程。每个种子运行目录独立，已有目录拒绝覆盖。入口保存每折三分支最终权重、训练 IDs、配置、耗时、参数量，以及 `probabilities.jsonl` 中逐人的 train/val/test 概率和标签。`inner_fold=null` 表示当前使用作者固定的内部留出验证，不冒充嵌套内层交叉验证。结构定义见 `configs/prediction_record.schema.json`。

转写入口为独立环境，明确选择导出政策后先预检；添加 `--run` 才执行 37 人语料转写。当前数据预检已通过，输出按 `transcripts/B0-37/<export-policy>` 隔离：

```powershell
S00/.asr-venv/Scripts/python.exe S00/scripts/transcribe.py --export-policy first_segment_code
```

当前转写环境使用 CPU/float32，与作者默认值相同，已通过 10 秒真实音频的离线加载与解码检查；未配置 ASR 的 CUDA/CTranslate2 加速，也未完成当前约 5.30 小时语料的完整转写。主分类和骨干环境已经使用 GPU。

## 来源

- 本地用户指定论文：`../baseline备选/2510.14922v2.pdf`，校验值见 `manifests/source_sha256.json`。
- 论文：[TRI-DEP v2](https://arxiv.org/html/2510.14922v2)。
- 作者代码：[tri-dep1](https://github.com/annisaafitrinn/tri-dep1)，本地同时保存 main 和 hele/dev 固定提交。
- 作者补充网页：[TRI-DEP companion](https://annisaafitrinn.github.io/tri-dep/)，已存 HTML 快照和划分对照。
- 原始 MODMA：[官方申请入口](https://modma.lzu.edu.cn/data/application/)。当前两个压缩包由用户提供，未收到原下载 URL，不额外声称已验证其发布者或等同作者原实验数据。

最终状态与证据索引见 `reports/preflight.json`。环境安装清单见两个 requirements lock；不要把新版本 pip 自动解析的环境当作作者原始环境。

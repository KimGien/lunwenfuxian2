# TRI-DEP + CCML 实验与 v2 收尾

## 当前状态（2026-09-24）

**v2 固定阶段和全部已登记可选实验均已完成，总体总结已更新。** 当前总入口：[CCML-v2 总体总结与收尾报告（含全部可选实验）](reports/ccml_v2_full_closeout/CCML-v2总体总结与收尾报告.md)；[完整完成状态](reports/ccml_v2_full_closeout/status.json)；[全部指标与检查点索引](reports/ccml_v2_full_closeout/results.json)。累计 **20 个外层条件、570 组训练（外层 300、内层 270）、120,600 次更新**，TC-00-E 补充与旧训练重放单列。

S-control、S、G、F 的开发 Macro-F1 均为 **0.790859**，是当前完整范围内的最高观测值；S/G/F 相对同版本控制的分类均未改善。AS-R/P 的 F1 为 0.472030 / 0.465902，也均未通过门槛。五项可选候选全部完成，但均未达到预定分类提升门槛。新旧版本不能混作单因素比较；旧 NP/AS-NP 保留各自版本的历史开发参考角色。

本次总结更新重算 20 个条件、15 个配对比较，核对 11,100 条外层预测和 570 组训练预算，校验 2,310 个来源文件哈希；没有新增训练或重复检查点推理。图表、泛化诊断和未完成事项均已更新；旧收尾与阶段报告保持原样。独立人群验证及 EEG 幅值/坐标来源仍未解决，原 TC-06 不属于 v2，当前没有后续已登记训练。以下阶段记录按各自生成时点理解。

## 已完成的 AS-R/P 与 G/F 阶段

[剩余可选实验总结](reports/ccml_v2_optional_closeout/CCML-v2剩余可选实验总结.md)记录后续四项固定对照：AS-R/P 为 30 组、7,200 次更新，G/F 为 210 组、39,600 次更新。该阶段 240 个新检查点、7,008 条预测精确重载，30 份内层选择复算、60 个新增外层测试折指标交叉核验通过，5,667 项历史文件保持不变。这些是当时验收记录，不作为本次新增测试。

## 已完成的 S 阶段

**CCML-v2-S 模态证据尺度校准已完成。** 见 [S 完成报告](experiments/CCML-v2-S-37-fulltext/CCML-v2-S实验完成报告.md)、[完成状态](experiments/CCML-v2-S-37-fulltext/status.json) 和 [S 固定计划](../output/plans/CCML_v2_S_模态证据尺度校准固定计划.md)。本轮从无 MODMA 监督父权重的冻结骨干特征建立均值汇聚版本，在合法训练内三折中选择尺度收缩强度，配同版本完整 CCML 控制。90 组内层训练＋30 组外层训练、23,400 次更新全部完成；旧 AS-NP 仅作历史参照。

S-control 与 S 的 Macro-F1 均为 **0.790859**、MDD recall 均为 **0.666667**，三个种子 F1 及 111 条测试分类全部不变。NLL 0.494859→0.494695，Brier 0.158180→0.158218。S 未通过分类提升门槛，不自动升级或追加搜索。两组 train/val/test F1 均为 0.979528/0.770000/0.790859，泛化差距仍在；新版本较旧 AS-NP 的分数差异不能归因于尺度校准。

41 项测试、120 个检查点和 3,504 条预测精确重载、15 份内层选择复算、30 个测试折指标交叉核验通过，4,868 项历史文件哈希不变。下方为 S 开始前的收尾快照，其“未执行 S”等文字按当时状态理解。

## S 开始前的 v2 历史收尾

**以下为 S 开始前的固定阶段快照。** [原固定阶段总体总结](reports/ccml_v2_closeout/CCML-v2总体总结与收尾报告.md)；[历史收尾状态](reports/ccml_v2_closeout/status.json)；[原 14 个条件指标与检查点索引](reports/ccml_v2_closeout/results.json)。当前完整结论以上方新版总报告为准。

Audit、E、N/R、P、NP、A、AS、AS-NP、AS-NP-R 均已完成，共 14 个训练条件、210 组训练、50,400 次更新。本次收尾不增加训练。AS-NP（F1 0.658326）保留为完整语音覆盖版本的开发参考，旧 NP（0.680895）保留为 v2 开发 F1 最高的历史参考，N 保留为固定表征规范化对照。AS-NP-R 未改善分类，不自动升级。

用户单独指定的 [TC-00-E 原始 TRI-DEP EEG 修订对照](experiments/TC-00-E-37-fulltext/TC-00-E原始TRIDEP实验完成报告.md) 亦已完成；其 15 组新 CE 训练与 15 组旧 EEG 精确重放单独计数。过拟合、独立泛化验证及 EEG 输入尺度/坐标来源仍是未解决问题，可选 S 与原 TC-06 未执行。当前没有自动续跑任务。

以下为 TC 准备及早期 v2 的历史说明；旧阶段状态不覆盖上方总体收尾状态。完整后续阶段导航见 [计划索引](../output/plans/README.md)。

## TC 准备及早期阶段历史记录

依据 `output/plans/MODMA_TRI-DEP_CCML_实验计划.md` 开展 MODMA 37 人的 TRI-DEP + CCML 实验。以三个种子、五折父模型为来源，冻结上游表征，只训练三个新的 128→2 末层。TC-00 至 TC-05 已完成；最新执行状态见 `reports/experiment_status.json`，准备条件复核见 `reports/准备状态.json`。

TC-01、TC-02 均完成三个种子 × 五折、每折 60 轮训练并通过重载验证。主口径 Macro-F1 分别为 0.699762、0.690847；详见 [TC-01 实验完成报告](reports/tc01/TC-01实验完成报告.md) 与 [TC-02 实验完成报告](reports/tc02/TC-02实验完成报告.md)。两阶段的初始化、批次顺序和训练预算一致；TC-02 结果未达到整体提升。

TC-03 复用 TC-02 权重完成三种规则、共 45 组推理，无额外训练。证据均值/求和的 Macro-F1 均为 0.565301，CCML 为 0.605380，均低于 TC-02；详见 [TC-03 实验完成报告](reports/tc03/TC-03实验完成报告.md)。

TC-04 从共同初始权重独立完成 15 组完整 CCML 训练并通过配对验收。Macro-F1 为 0.611614，较 TC-01 下降 8.81 个百分点，MDD recall 为 0.488889，NLL/Brier 均恶化，未达到预设工程参考；详见 [TC-04 实验完成报告](reports/tc04/TC-04实验完成报告.md)，其中包含 TC-00 至 TC-04 完整矩阵、单模态结果及不确定性诊断。

TC-05a/b/c 各完成 15 组独立训练和重载核验，Macro-F1 分别为 0.601111、0.619550、0.598608。删除一致 KL 的 TC-05b 较 TC-04 提高 0.79 个百分点，仅种子 11 提高，其余两种子持平；三个消融均低于 TC-01，未满足预设工程参考。详见 [TC-05 实验完成报告](reports/tc05/TC-05实验完成报告.md)。第一阶段全部 135 组新头训练/复用推理已完成；TC-06 尚未启动，仍为第二阶段保留项。

用户于 2026-09-21 限定后续只使用 CCML。后续安排见 [CCML v2 性能优化计划](../output/plans/CCML_v2_性能优化与后续计划.md)。Audit 确认 EEG 电极选择偏移后，已于 2026-09-22 完成独立版本的 37 人 EEG 重导出、15 组 CCML EEG 适配和 15 组冻结新表征后的 CCML 控制。新控制 Macro-F1=0.517126，较旧 TC-04 下降 9.45 个百分点；所有种子均下降，本轮未实现性能提升。

EEG 修订阶段结果见 [CCML-v2-E 完成报告](experiments/CCML-v2-E-37-fulltext/CCML-v2-E实验完成报告.md)。该阶段 728 项受保护历史文件保持不变；原 TC-06 的 CE/CCML 配对设计继续保留，v2-E 的暂停记录为历史记录。

2026-09-22 已进一步完成 N/R 各 15 组固定对照。N 的 Macro-F1=0.616845、MDD recall=0.466667，较同版本新控制提高 9.97 和 6.11 个百分点，三个种子均提高，NLL/Brier 也改善，达到预定工程门槛。R 的分类不变，未达到门槛。N 目前仍接近旧 TC-04 的 0.611614；跨父版本比较不能当作单因素效果。详见 [最新 N/R 完成报告](experiments/CCML-v2-NR-37-fulltext/CCML-v2-NR实验完成报告.md) 与 [N/R 当前状态](experiments/CCML-v2-NR-37-fulltext/status.json)。共 58 项测试通过，1,027 项历史文件哈希不变。

2026-09-22 已完成独立 P 的 15 组有限投影适配，共 3,600 次更新，训练 836,742 个参数。Macro-F1=0.551739、MDD recall=0.450000；相对同版本控制 F1 提高 3.46 个百分点，但仅种子 11 提高、22 持平、33 下降，未达到预定跨种子门槛，也低于 N。当前保留 N 作为主要开发候选，不自动升级 P 或解冻完整 CNN/LSTM。详见 [最新 P 完成报告](experiments/CCML-v2-P-37-fulltext/CCML-v2-P实验完成报告.md) 与 [P 当前状态](experiments/CCML-v2-P-37-fulltext/status.json)。62 项测试通过，1,217 项历史文件哈希不变。

以上保留 Audit、E、N/R、P 完成时的记录；后续 NP、A、AS、AS-NP、AS-NP-R 也已完成，当前结论见总体收尾报告。P 的有效配置为 `experiments/CCML-v2-P-37-fulltext/configuration_v2.json`；首次缓存核查发现旧语音导出的 cuDNN 数值模式差异后，原配置及代码已归档，并在训练前恢复原模式通过复现验收。P 入口为 `tridep_ccml.v2.p export` / `tridep_ccml.v2.p train`，验收为 `tridep_ccml.v2.complete_p`；N/R 入口为 `tridep_ccml.v2.nr` / `tridep_ccml.v2.complete_nr`。均使用 `S00/.venv/Scripts/python.exe -B -m` 调用。

v2-E 恢复入口为 `S00/.venv/Scripts/python.exe -B -m tridep_ccml.v2.resume_eeg`，完成复核入口为 `S00/.venv/Scripts/python.exe -B -m tridep_ccml.v2.complete_eeg`。恢复入口保留受限检查点读取，仅兼容该版本的 PyTorch 版本元数据，不修改旧训练代码或检查点。

## 查看准备状态

从项目根目录运行：

```powershell
.\tridep_ccml\run.ps1 -Action check
```

检查会核验配置、父实验锁、15 组缓存、官方 CCML 代码版本、依赖与测试结果，并验证已完成的运行记录。报告写入 `reports/TC实验准备报告.md`；存在阻塞时同时生成 `reports/TC阻塞报告.md`。已经完成的运行会被识别为已完成，正式训练入口仍拒绝覆盖旧产物。

使用已经存在的 `S00/.venv/Scripts/python.exe`。系统默认 Python 没有本实验依赖，不能直接替代此环境。无需重新下载骨干模型或转写数据。

## 已准备的材料

- `configs/experiment.json`：固定超参数及计划哈希；60 轮、batch 8、Adamax 0.001、梯度上限 1。
- `configs/registry.json`：独立 TC 实验注册，与原人群文件哈希绑定；旧 S00 配置和结果保留。
- `features/seed_<seed>/fold_<fold>.npz`：每组 37×3×128 表征，以及受试者、标签和分组。
- 对应 `.json`：父权重、原缓存、训练名单、代码与缓存哈希。每个监督表征绑定自己的折和种子。
- `ccml.py`：新分类头、Softplus 证据、Dirichlet 均值和不确定性、CCML 解耦与损失。
- `train.py`：TC-01/02/04/05 训练入口；TC-03 复用 TC-02 权重。默认只检查，需显式执行才拟合新模型。
- `references/ccml/`：固定提交的官方参考文件及来源哈希。
- `reports/`：父依赖审计、重载复现、TC-00 原概率补算、测试与准备状态。

## 后续训练入口

单折检查示例：

```powershell
.\tridep_ccml\run.ps1 -Action train -Experiment TC-01-37-fulltext -Seed 11 -Fold 1
```

正式运行时在同一命令末尾加 `-Execute`。种子依次为 11、22、33，折号为 1 至 5。先分别运行 TC-01 与 TC-02，再从 TC-02 权重生成 TC-03-mean、TC-03-sum、TC-03-ccml，最后独立运行 TC-04；所有 ID 以 `-37-fulltext` 结尾。TC-05a/b/c 是独立初始化的机制消融。TC-06 为计划中的第二阶段，当前未实现。

每个直接训练对照独立采用相同初始化和受试者顺序，仅在该折 train 拟合。普通概率平均没有 Dirichlet 总证据，不填写 `u_final`。TC-03 不再训练。已有输出拒绝覆盖。

重新运行特征准备：

```powershell
.\tridep_ccml\run.ps1 -Action prepare
.\tridep_ccml\run.ps1 -Action test
```

源配置、代码或父产物变化会触发检查失败；不能通过删掉哈希检查继续使用旧缓存。

## 解释边界

本轮沿用已经查看过测试结果的固定划分，属于开发性比较。旧 TC 输入的通道选择问题已在 v2-E 独立修订；EEG 检查点尺度、原始坐标模板，以及语音随机冻结编码器、单位置注意力和短音频零向量仍需披露。v2-E 同时包含通道修订与 EEG 表征适配，不能将其相对历史 TC-04 的差异全归因于通道。TC-00 是原结果复核和等权补算，不代表新头效果。新增种子不能视为新增受试者。

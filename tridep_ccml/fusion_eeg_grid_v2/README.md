# EEG 十五版本 × 五种融合：当前有效入口

## 2026-10-14 后续计划入口

后续研究以[诊断修复与调参计划v3](../../output/plans/EEG十五版本与五种融合方法后续计划_v3_20261014.md)为准：错误归因、标准化及QMF独立修复、有限学习率搜索、CCML/ECML条件性扩轮，候选由各方法直接输出，禁止基线式额外固定权重融合。v3保留完整75EST及5ST目标，拟用独立实现和实验目录；当前只修订计划，未启动新训练，自动任务保持暂停。

下方为v2的诊断与历史执行记录。v2源码、检查点、预测和旧证书继续保留，不用旧队列执行v3，不把旧“全部完成”覆盖后续来源勘误或新阶段预算。

## 2026-10-13 完成后的诊断勘误（优先于下方历史完成表述）

用户要求查找低分原因后的源码复查确认：本地 QMF 在非零损失历史下的排序边距符号与锁定作者 RGBD 实现不等价，原工程门仅检查融合及首批全零历史，没有覆盖此差异。1,200 次训练和保存的分数是真实运行结果；但其中 240 个 QMF 拟合（225 EST＋15 ST）的严格作者来源一致性验收需重新处理。因此，下方及旧 `final_delivery.json` 的“全部方法最终验收通过”仅保留为历史记录，不能继续作为当前无保留结论。该问题不直接影响其余 960 个非 QMF 拟合，也不能据此解释它们的低分。

缓存复算另确认共享标准化尺度隐患：训练方差为零的维度被 `std=max(std,1e-6)` 处理；全缓存训练绝对值最大5.10，验证最大350605、测试最大209744，含所有方法复用的 S/T 分支。有限性检查不会发现这类有限但极端的值。实际分数未被改写；此隐患对 F1 的因果贡献尚未通过修正对照隔离，不能解释为已证实的唯一低分原因。

本轮诊断证据在 `output/analysis/fusion_eeg15_methods5_v2_20261013/diagnosis_20261013/`。已完成的训练代码、权重、预测、原报告与原证书保留；本次只分析已有产物并复算合成排序损失，未追加训练。自动任务保持暂停。需要任何后续修复或扩轮时使用独立版本和来源审查，不复用历史队列直接改写结果。训练轮数诊断显示 CCML/ECML 有明显晚期验证改善，不能统一归因为轮数充分或不足。

## 最终完成状态（2026-10-13）

本轮已真实完成80条件、1,200正式拟合（1,125 EST＋75 ST）、455,944次optimizer更新、统一评价与全部独立验收。`final_delivery.json`及`delivery_v3/complete.json`为实际passed/all_plan_complete证书，绑定27份交付文件；主报告、75EST/5ST表、8,880条OOF与3,375行配对已核验。`status.json`为complete；eeg-2自动接续已按完成条件暂停。下面的训练与修复入口是历史记录，不据此重启。

75个EST均未超过本地TRI-DEP复现基线72.550%。最高为E11 LaBraM 5秒冻结小头＋MMDynamics，Macro-F1 71.165%，相对基线−1.385个百分点；属于既有37人的开发结果。全部阴性条件保留。

评价一次失败与修复没有新增训练，17个原成功评价字节复用；同后端逐位验证和全部模型独立重载通过。成本未测部分与四次验证诊断未知耗时仍null。原始System时间包络内9条事件全部保存，相关低功耗/重启/清日志事件0条，无未解范围；不冒称完整active总耗时或独占编码器显存峰值。

最终报告：`output/analysis/fusion_eeg15_methods5_v2_20261013/最终全组合实验报告.md`；成本附页：同目录`最终成本与运行范围附页.md`；全部证据索引：实验目录`final_delivery.json`。原训练、失败、旧版本和全部来源保留。

当前版本 v2.3：`tridep_ccml.fusion_eeg_grid_v2.run_v3` 与 `features_v3`。以实验 `status.json`、当前 owner 和真实日志为实时进度依据。本轮1,200次正式融合拟合独立登记，原EEG和S/T冻结。此前源码与成功工程验收保留；v2.2单独核记旧九候选计划的后续人类文本修订，其余旧来源均严格哈希核验。v2.3恢复已封存来源的精确batch成员：冻结小头8、原CNN16、raw模型每人，避免改变原CNN卷积推理的批量数值模式。

2026-10-13：1,200/1,200正式训练已完成。原评价在ST-LMF/seed11/fold3因重载时关闭Parameter requires_grad标志改变CPU matmul数值路径触发逐位门，原17个成功评价、原失败与全部训练字节保留。当前评价接续入口是 `S00/.venv/Scripts/python.exe -X utf8 -B -u -m tridep_ccml.fusion_eeg_grid_v2.evaluation_only_v1 --run`，绑定 `evaluation_v2` 和 `evaluation_recovery_v1/review.json`。保留训练验证时Parameter标志，每个实际前向均inference_mode、无梯度且状态不变；同后端逐位门不放宽。此入口只评价，禁止重新调用原训练队列或重跑成功工程/表示。诊断四次val前向及未知成本单列，不读测试用于修复。

唯一接续入口：从工作区运行 `S00/.venv/Scripts/python.exe -X utf8 -B -u -m tridep_ccml.fusion_eeg_grid_v2.run_v3 --run`。仅启动一个后台进程，使用隐藏窗口和独立stdout/stderr日志。已存在健康owner时不能重复启动。查看状态用同模块 `--status`。

全批后独立验收入口：`S00/.venv/Scripts/python.exe -X utf8 -B -u -m tridep_ccml.fusion_eeg_grid_v2.final_audit_v1 --run`。仅在1,200拟合、统一评价与队列owner及native进程释放后运行；此前只能登记 `--prepare`，不读新outer概率，不重复模型探测。它独立重建优化器更新、batch/轮次/选模/早停、train标准化与NumPy指标，生成包含基线和全部75EST的第一张主表。`final_audit_v1/complete.json`仍不代表全部计划完成：随后须核系统事件/暂停/失去进程影响范围、真实成本的已知与null未知边界，并完成最终语义/成本/交付审查。新验收脚本及protocol单独绑定，原训练登记、代码和成功工程证据不改。

最终交付入口：上述核心验收实际通过后运行 `S00/.venv/Scripts/python.exe -X utf8 -B -u -m tridep_ccml.fusion_eeg_grid_v2.delivery_v3 --run`。它复算全部3,375行配对与本地基线、核对数据/披露/成本未知边界，并读取本轮固定完整时间包络内的Windows System事件和本机provider定义，保留原始XML、事件窗口及可能影响的fit；未配对或需要原生schema解释的事件必须进一步窄审，不直接颁发完成证书。v2单独追加真实出生身份绑定的资源采样；v3另绑定实际评价失败、零optimizer恢复、17个原评价复用与四次val诊断，正式训练无重跑。区分截至采样时进程峰值、整卡诊断与源码导出读取/传输计数；未知独占显存峰值和完整active时长仍null，此前来源不改。只在 `final_delivery.json`真实passed/all_plan_complete后交付并暂停本自动接续。该交付门不新增模型前向或optimizer更新。当前每20分钟自动接续。

实验：`tridep_ccml/experiments/Fusion-EEG15-Methods5-v2-37-20261013`。报告：`output/analysis/fusion_eeg15_methods5_v2_20261013`。固定计划：`output/plans/EEG十五版本与五种融合方法全组合计划_v2_20261013.md`。

入口依次完成：注册计划/代码/作者提交 → 合成数据公式对应、CPU/GPU测速与逐位恢复 → 全部冻结表示来源登记/导出/原分类层回接 → 75 ST → 1,125 EST → 全批评价与独立重载。最终语义、来源和成本审查以及交付证书由自动接续进一步完成；`evaluation_complete.json`不等于全部验收完毕。

作者原文件保存在 `vendor/`，固定提交及Git blob见 `upstream_sources.json`。ECML作者固定实现为顺序平均证据与get_dc_loss冲突一致性，保留完整单模态/最终EDL与KL，不能将其误称为Dempster融合。QMF选择作者RGBD实现，质量在融合中保留梯度、排序系数0.1、先排序后更新历史；首次等历史规定零目标/零边距，修复作者除零边界。这些在正式运行前登记。

每方法单一CPU/GPU后端与FP32由工程测速锁定。CPU的小融合头可能更快；EEG导出仍按原CUDA推理方式。没有开启新推理批量优化。原人员顺序、全60窗、原window chunk不改。训练batch8、末批保留；累计最多120轮含恢复、10轮热身、余弦降至10%；最低验证NLL保存best、同值取早；最早60轮且连续60轮无1e-4下降早停。

暂停：先原子设置实验 `control/user_pause.json`为严格 `{"paused":true}`。所有入口检查标志；队列在当前安全边界落盘后退出。冻结导出每人检查，训练每轮边界检查。暂停后禁止再启动，明确继续前不清除标志，并暂停本对话自动化。只管理本轮已绑定完整出生身份的进程。

续训：检查 `control/owner.json`与完整 native_tree 身份、注册文件和源码哈希。健康owner只观察。失去进程后留lost owner与未知尾记录，载入精确latest（模型、优化器、绝对轮次调度、随机状态、QMF训练历史、最佳与早停状态、采样边界）；不重置120轮预算。失败需先读对应failure记录并审查，保留旧代码与产物，修正版本独立登记。不要启动旧实验、旧自动化或重跑已封存基础模型。

验收产物：`engineering/complete.json`、`source_registration.json`、`feature_gate.json`、80条件各15个 `complete.json`、`training_gate.json`、`evaluation_complete.json`、完整CSV/JSON与报告，以及后续独立最终审查/成本/交付证书。未满足全部最终门不能暂停自动接续或声称实验完成。

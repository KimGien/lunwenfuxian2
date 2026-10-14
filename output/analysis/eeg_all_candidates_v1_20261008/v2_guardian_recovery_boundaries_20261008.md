# v2 监管与恢复边界（只读说明）

当前没有故障，不能执行恢复。只读观察为 `dispatcher_suspended_children_running`：旧 dispatcher 被暂停，原 Conformer 两级子进程继续完成；监管器持有精确创建时间的进程句柄。源登记37项通过，未改任何封源文件。

完整验收仍是 `evaluate.completed_report_verified()`：360次正式拟合、345个监督模型、23监督条件、435个规定快照；每条件主口径111条OOF仅来自37人；全部概率、检查点、源、gate与最终报告哈希重新核对。v2隔离函数的ROOT、register、sources及评价尾都正确绑定v2。新发现的独立阻塞是：v2生成的WSL owner路径在 `continuation_queue_v2/attempts`，而封源Linux入口仅接受旧 `continuation_queue/attempts`。它会在GPU之前拒绝Mamba任务，须另立版本化路径修复或窄范围Mamba补全接续，不能修改封源或假标完成。

| 中断发生位置 | 可审查的恢复边界 |
|---|---|
| 监管器仍活动、Conformer仍训练 | 继续等待；不重跑 `--arm`、不恢复旧dispatcher、不另起训练。监管器正常路径会先确认两子进程exit0和Conformer15全产物，再终止旧dispatcher。 |
| 监管器死亡、原Conformer仍活动 | 模型本身可继续。现有监管器没有resume入口且lease为exclusive创建；需另立仅接管已暂停父进程的监管恢复方案，重新核验精确身份、原lease/日志、未开始旧SSL。不能直接重跑arm、累计Suspend或恢复旧driver。 |
| 主机中断且尚无 `handoff_complete.json` | 原精确进程身份及正常exit0链已失效。保留旧lease、attempt、progress、checkpoint字节；确认无任务进程后，用原监督完整状态恢复Conformer。完成15组后另立真实恢复交接证明，不能伪称被中断的原两子进程正常exit0。 |
| 已有真实handoff证明，v2运行后中断 | 首先确认guardian、v2 dispatcher、Windows/WSL实际训练进程均退出；归档并核验stale v2 owner后才允许v2 `--resume`。既存完整模型直接复用，未完成模型从最后持久化epoch状态恢复。原登记及封源保持原字节。 |
| 拟合或评价失败但进程仍在 | 先核实际PID/创建时间与子树，不以状态文件代替进程退出证据。Windows与WSL都必须确认任务的实际GPU进程退出后再取得唯一GPU所有权。 |

最后持久化epoch之后已经执行的部分更新可能需要重算。恢复前保存progress/failure/log前像，单列已完成/已开始更新及重复成本；不能称为0额外计算。评价恢复只能复用源/gate/检查点/人序完全一致的已保存概率批次；不能用测试成绩选新模型、阈值或更换来源。

监管器会保留v2的Popen并持续等待其退出，避免正常监管收尾回收仍运行的接续子进程。硬性主机/工具宿主中断无法靠这条等待逻辑消除，恢复仍须遵守上表进程与产物边界。

[本次只读接口审计](v2_completion_interface_readonly_audit_20261008.json)。本说明没有执行进程暂停、终止、训练、数据前向或真实测试。

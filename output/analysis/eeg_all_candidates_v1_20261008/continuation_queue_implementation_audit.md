# 单 GPU 后续队列独立审查

2026-10-08，纯标准库模拟测试 9 项通过，见 [独立结果和当时源 SHA256](continuation_queue_stdlib_9tests_audit.json)。全部子进程 / 平台清理操作均模拟；没有实际训练子进程、GPU 模型前向或优化更新。

队列矩阵为先前 11 条件×15＝165 次，再执行剩余 13 条件×15＝195 次，覆盖原登记全部360。剩余包括 Conformer从头15、EEG2Rep SSL15及监督15、EEGMamba冻结15、七个候选与两项LaBraM有限微调135。没有根据冻结成绩取消微调。SSL明确要求真实逻辑64、bfloat16，没有拆分VICReg损失累积或替换骨干的成功捷径。

最初实际观察到两项回归失败：父进程中断没有终止child；重复dispatcher在第一个child仍活动时启动第二child。root在登记冻结前增加独占owner记录、Popen后立即持久化child PID、finally Windows进程树 / WSL实际PGID清理。复测通过。第二dispatcher的锁拒绝位于主except之外，不覆盖第一owner状态；无法确认Linux组退出时保留owner lease，避免另起GPU任务。

交接必须同时满足前序ready_queue终态和全部165份状态完成。当前 `all_complete` 再核对每条件工程源锁、每fold四份真实产物及SHA；模拟篡改实际checkpoint后会拒绝交接。prepare进程exit0不足以成功，真实工程锁 / Mamba审计 / 导出manifest的后置条件也须通过。失败依赖只阻塞依赖任务，其他独立路线仍执行；partial / blocked / failed不伪装成complete。

CodeBrain修复仅针对实际记录的该条件导出失败。在两个原始owner PID均退出后，串行运行独立versioned export，再恢复原ready_queue。修复临时将导出根目录切换到新版本，finally恢复根路径与模块映射，main manifest明确绑定codebrain_v2及修复登记；旧源码和失败目录没有写入。有限微调也已显式使用codebrain_v2。已建议root再保存旧失败产物前后SHA清单，并把initial_owner证据纳入后续queue源锁，作为“保留原字节”的可核验依据。

本次9项审查发生于最终统一评价尾任务接入之前。新增LinuxMamba评价和Windows统一评价之后，还需独立测试：全部360训练/产物/源锁gate未通过时，任何evaluate child均不能启动；全部通过时按Linux再Windows顺序执行，并由各评价入口再次检查gate。该部分不能由此9项结果提前声称已验收。

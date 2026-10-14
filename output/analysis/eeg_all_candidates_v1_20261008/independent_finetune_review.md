# 有限微调与统一评价独立审查

审查日期：2026-10-08。审查和合成回归只使用 CPU，没有真实外层测试前向，也没有新增正式拟合。

发现并处理的确定问题：

1. LaBraM 原适配 audit 仅返回裸 hash，微调登记无法递归提取 `{path,sha256}` 来源；EEGPT 实际 loader 未完整纳入来源锁。新适配 audit 已登记作者 checkpoint、loader、vendor 及 helper 文件。两者 CPU strict 加载与来源回归通过，证据 `independent_provenance_regression.json`。
2. CodeBrain 原 checkpoint 的 8 个 GConv kernel norm 初始化标志全部为 false，第一次冻结前向会改 16 个 buffer；原冻结导出与微调状态检查因此会拒绝完成。新的 `codebrain_v2.py` 从作者代码抽取完整 kernel 公式，只依赖正式权重初始化缓存，未改原已锁适配或作者源。CPU 核对原首次前向、第二次前向以及 v2 冻结输出：v2 与原首次冻结输出逐位相同，buffer 从首前向起不变。
3. CodeBrain 最后块的 `sn.scale` 和 `res_conv` 三个参数不参与最终 skip 汇聚输出，梯度为 None。v2 解冻名单排除这 4 个死输出参数，保留 25 个真实连接参数；合成受试者 CE 的实际一步更新和冻结状态检查通过。高尺度 kernel 7/8/9 在 5 秒输入下梯度为零，如实保留记录，不声称每个参数都有信号梯度。证据 `codebrain_v2_cpu_review.json`。
4. 统一评价的缓存恢复必须保留每批模型状态与无梯度证据，防止终端校验失败后仅凭已保存概率被标为完成。现每批写盘前校验 checkpoint 指纹，缓存复用再次核对；Linux Mamba 全概率缓存已齐而终端账本缺失时，可从已验证批次恢复零前向终端证据，并对实际落盘包再次执行统一 postcondition。

已核对的执行约定包括：新正式预训练骨干与新头初始化；前 10 轮仅小头、接着 5 轮骨干热身；每人 8 窗、8 轮覆盖 60 窗并明确重复 0–3 窗；按真实逻辑人数平均受试者 CE；冻结子模块和 buffer 保持不变；诊断恢复 RNG、模块状态及模型状态；完整恢复 checkpoint 后下一次随机批次与更新一致。Conformer 仅使用已保存的本折训练标准化统计，EEG2Rep 监督重建必须绑定同种子、同折已完成 SSL best，并加载完整监督 best 状态。

统一评价入口和公开只读 postcondition 均首先执行全部 360 拟合完成与来源 hash 的 gate。固定 300 轮诊断主要比较第 300 与第 60 轮，120/200 轮只作次要观察。概率阈值严格 `p_MDD > 0.5`，相等为 HC；每种子先平均五折，再平均三种子。每条件 111 OOF 预测仍只对应 37 人。

最终 CPU 合成评价回归为 10/10 通过，覆盖 blocked gate、错误 gate/旧来源拒绝、状态错误缓存拒绝、Linux Mamba runtime 拒绝、全缓存终端恢复、概率与 logits 一致性、tie、五折聚合和固定匹配控制门槛。当前真实 GPU 与外层测试未由本审查执行；不能将这些 CPU 工程证据写成正式训练或测试完成。

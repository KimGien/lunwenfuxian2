# EEGPT / LaBraM 固定 300 轮旧接口重放实现审查

审查日期：2026-10-08。独立审查范围为旧 60 轮训练的来源、初始化、批序、优化器和随机状态，以及新 `paired_replay.py` / `storage.py` 的诊断隔离和恢复路径。旧文件未修改；审查没有启动正式拟合，也没有计算任何新外层测试成绩。

## 已验证的结果

- [历史只读独立审计](replay_readonly_audit.json)通过：30 份旧初始化逐张量符合独立 PyTorch 配方，两个条件合计 1,800 个旧轮次批序完全符合独立随机数模拟，210 份 complete 绑定产物 SHA256 一致。此项模型构造、神经前向和优化更新均为 0。
- [首对正式完成审计](paired_completed_readonly_audit_02fits.json)通过：LaBraM 与 EEGPT 的 seed 11 / fold 1 均完成 300 轮、600 次更新。两者前 120 批训练 CE、原 60 轮加权 CE、60 轮模型以及完整 Adamax 状态与历史逐位一致。
- 上述两份新训练的全部 300 轮批序与独立 CPU 规则完全一致；第 60 / 120 / 200 / 300 轮 CPU RNG 符合独立预期。两个条件四个快照的 CUDA RNG 也逐位一致。四快照的 optimizer step 依次为 120 / 240 / 400 / 600。
- 六项工程测试通过：独立 300 轮批序、完整 RNG / 模式 / 参数 / 下一次 Dropout 隔离、异常时诊断恢复、缺失 CUDA RNG 拒绝、序列化恢复后下一轮与不中断训练精确一致、写盘仅重试替换且只序列化一次。
- 工程 CUDA 输入仅来自原 fold 1 训练人员；测试中的“val”角色刻意用原训练人员构造，无原验证 / 测试人员输入。恢复等价测试使用 CPU 小型合成模型，共 6 次合成优化更新 / 测试执行，不构成候选正式拟合。首轮测试因测试比较器区分 OrderedDict 与 dict 而失败，修正测试后完整重跑通过；正式 runner 未因此修改或重训。

## 必须保留的旧训练 API 和顺序

旧数学入口为 `tridep_ccml/tridep_eegpt/train_eval.py::fit_all`，而不是更早阶段 1 的泛化训练函数。后者训练日志字段和部分记录行为不同，不应用来替代历史 matched4s 实验。

模型通过 `tridep_ccml/tridep_stage1/run.py::original_model()` 构造，最终调用 `S00/upstream-dev/lib/models/models.py::EEGConvPoolClassifier(input_dim=200, seq_len=60)`。498,434 个参数全部可训练；两次 Conv1d / ReLU / MaxPool1d，随后 Flatten → Dropout 0.4 → Linear 1920→128 → ReLU → Dropout 0.4 → Linear 128→2。不能关闭训练 Dropout、改头结构或替换输入预处理。

每折先 `C.runtime(seed)`，随后在 CPU 构造原模型，再 `.to('cuda')`。此构造会消耗 CPU 初始化随机流。只加载旧 `initial.pt` 而跳过随机构造会导致后续 DataLoader 采样随机流偏移。新 runner 当前保留了旧构造，并核对初始化哈希。

`C.runtime(seed)` 对 Python、NumPy、torch CPU 与全部 CUDA RNG 设相同 seed；线程数为 4，cuDNN benchmark=False、deterministic=True，`torch.use_deterministic_algorithms(True)`，matmul TF32=False、cuDNN TF32=True。旧模块在导入 torch 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。重放为 float32，无 AMP。

训练 dataset 是原 `subject_ids` 全局索引和原 train 人员标签，原训练名单的顺序必须保持。原 loader 是 `DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)`，默认不丢弃末批。每轮 27 人为 16+11，26 人为 16+10，均恰好 2 个更新。

优化器为 `torch.optim.Adamax(model.parameters(), lr=.005, weight_decay=0.)`；原状态确认 betas=(0.9,0.999)、eps=1e-8、foreach=None、maximize=False、differentiable=False、capturable=False。每批 `zero_grad(set_to_none=True)` → 原未加权 CE → backward → step；没有梯度裁剪或学习率调度。应精确比较 optimizer 的 8 个参数 state 中 step、exp_avg、exp_inf 和 param_groups，不能只比较学习率与 step 计数。

纯时长实验只复用旧 `features_manifest.json` 的 labram / eegpt 冻结数组。旧 EEGPT 的随机正交 512→200 投影、均值和旧幅值规则都不能改动。新 512 维读出与作者参考适配属于另行登记条件。

## 独立 CPU RNG 核验的准确规则

独立初始化配方只使用局部 `torch.Generator(device='cpu').manual_seed(seed)`，依次构造 conv.0、conv.3、classifier.2、classifier.5 的 weight 和 bias，共 8 个 tensor。各层 Kaiming uniform 的 fan / gain / bias bound 与原 nn.Conv1d / Linear 默认初始化一致。每种子初始化后的 Generator state 已保存哈希。

本机固定的 PyTorch 2.8.0 DataLoader 在每轮创建迭代器时消耗一次全局 CPU int64 base seed；RandomSampler 随后再从全局 CPU RNG 取一个 int64 sampler seed，再建立其局部 Generator。样本排列来自局部 `randperm(n)`。其源码还计算一个最终空尾切片的第二次局部 randperm；这不会影响全局 CPU 流。CUDA 上的 Dropout 不推进 CPU 采样流。

独立模拟由初始化后的局部 Generator 每轮取上述两个种子并重建 perm，映射回固定 train 全局索引。它不调用 producer 的 DataLoader。历史 30 个批序全部通过；工程测试又用真实 DataLoader 跑全 300 个轮次，核对第 60 / 120 / 200 / 300 轮独立 RNG 哈希。

旧 checkpoint 没有保存完整 RNG，因此不能声称“历史末 CUDA RNG 已被读取并核对”。本次建立严格重放的主要证据是旧初始化、全部批序 / loss、60 轮模型和 Adamax 全状态一致；新四快照的随机状态另外保存，并核对两条件随机流配对。

## 诊断与恢复审查

新 `diagnostic()` 使用独立索引，不创建随机 DataLoader；在 `inference_mode` / eval 下诊断 train / val，并在 finally 恢复每个 module 的原 training flag 和 Python / NumPy / torch CPU / 全 CUDA RNG。前后参数 state digest 相同，随后训练的 Dropout 输出与不做诊断时逐位一致。异常路径恢复也已测试。

旧 `train_eval._predict()` 会永久 `requires_grad_(False)`，只适用于训练完成后的冻结评价，不能作为每轮训练诊断使用。新 runner 未调用此旧接口。

review 中发现的中途工程锁跳过、无 latest 的初始恢复、里程碑 latest / snapshot 写入不一致、重复 immutable 文件报错、durable 轮次误记以及 CUDA RNG list 长度遗漏，root 已在冻结执行源之前修正。恢复文件现在检查 epoch / history / order / sampler_position / 初值 / finite state / optimizer step，并从 latest 或 initial 恢复。每轮 snapshot 可与已存在同内容文件核对复用。

写盘源先序列化、flush、fsync，再重试同一个 temporary.replace。PermissionError 或 winerror 5 才能重试，不重复神经前向、梯度或优化 step。注入临时替换失败的测试确认只序列化一次。

## 后续运行的审查边界

1. 当前首对完成审计证明这两份正常训练与旧路径严格配对；其余 28 份固定时长拟合和九篇完整矩阵尚未因此完成。应对新增完成文件继续使用独立审计入口，不能把启动成功当作全计划验收。
2. `--limit 2` 的返回发生在 runner 的 pair require 之前，本次由独立审计补核了一整对的初始化、300 轮批序和全部四快照 CUDA RNG。最终统一核验应再次全量配对，不依赖单次限额队列是否走到 require。
3. 当前从磁盘恢复时，没有单独在 `restore_rng(saved)` 之后立即核对 `rng_equal(saved, current)`。正常 writer 的完整 RNG roundtrip 已通过工程测试；将来实际恢复前，应先独立确认 CUDA RNG 设备数与当前设备数一致、CPU/CUDA字节状态都可完整恢复，再继续训练。
4. 首对目前没有实际中断恢复。工程测试证明模型 / optimizer / RNG 序列化后的下一轮可精确继续；它不等同于已经对正式 300 轮任务的所有磁盘失败时点逐个注入故障。

## 本机运行入口和只读进程证据

应使用 `S00/.venv/Scripts/python.exe`。实际 Python 3.13.14，PyTorch 2.8.0+cu126，NumPy 2.2.6，与旧登记一致，CUDA 可用。PATH 的 `F:/develop/python.exe` 没有 torch，不能用作正式训练入口。解释器的“Failed to find real location”提示不影响该 venv 正常导入和执行。

训练开始前的 NVIDIA 只读查询为 RTX 4080 Laptop、驱动 596.08、0 MiB / 0% 且无 compute app；Get-Process 同时未列出 Python / WSL / Conda 训练进程。CIM CommandLine 查询因当前访问限制失败，因此没有将其结果写成完整进程排查。训练后只读观测到 root 的正式单卡训练；审查没有另行启动 GPU 正式拟合。

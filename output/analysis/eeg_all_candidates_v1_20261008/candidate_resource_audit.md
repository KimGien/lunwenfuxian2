# 九篇 EEG 候选资源与接口审计

审计日期：2026-10-08。该文档记录资源准备和 CPU 工程检查，不把下载或少量前向计作论文实验完成。

**九候选来源已经全部登记。** 本轮实际取得八个作者仓库固定版本、六份新的正式预训练权重，以及 REVE 官方名义电极位置库；EEGPT 继续使用此前已经严格验收的完整权重。EEG Conformer 从随机初始化监督训练，EEG2Rep 必须逐折训练人员自监督预训练，二者不要求不存在的通用权重。

机器可读总表为 `tridep_ccml/eeg_all_candidates/candidate_resources.json`；精确文件路径、每个源码文件 SHA256、源码 archive SHA256、权重来源 URL 与 SHA256 均在新实验 `assets/download_manifest.json` 和每个 `vendor/{方法}/author_source_binding.json`。八个作者源码快照保持原始字节，旧实验和旧环境未修改。

| 候选 | 固定作者版本 | 本轮正式权重 / 资源 | 当前检查 |
|---|---|---|---|
| CSBrain | `185aee55b24d0410a830df8dd08d03f675616998` | `assets/csbrain/CSBrain.pth`，35,722,626 字节 | 完整骨干 295/295 keys CPU strict 加载；D200、12 层、8 头 |
| REVE | `06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7` | `assets/reve/model.safetensors`，276,773,456 字节 | 140 tensor 元数据完整；官方 positions 库独立下载 |
| EEGPT | `ea7282f698c406cd423ccbedbe9960853761144c` | 历史已验收 `work/models/eegpt_author_bound_20261006/eegpt_mcae_58chs_4s_large4E.ckpt` | 历史完整 strict loader 可复用，正式新拟合另记 |
| LUNA | `f7e94379255c425819c9138ceca60c26bc13b88e` | `assets/luna/LUNA_base.safetensors`，26,712,092 字节 | 224/224 keys CPU strict 加载；D256 |
| CodeBrain | `22d350caf68246d2fda4f630ef837420db3fb130` | `assets/codebrain/CodeBrain.pth`，60,901,006 字节 | 269/269 keys CPU strict 加载；正式 EEGSSM，D200 |
| EEG Conformer | `9ae149ba62487ceae723277d13adac27837113d2` | 作者 `conformer.py` 源码 | 需 19 导空间卷积与受试者级任务头；从头训练 |
| BIOT | `d138e32634e52ae9fa6ec98ac9c4087b14ca869a` | `assets/biot/EEG-PREST-16-channels.ckpt`，13,791,969 字节 | 57/57 keys CPU strict 加载；D256 |
| EEG2Rep | `8a72c39c8b5b1c1bd05d527c4187f30965fa5198` | 作者模型、SSP、训练损失、EMA 源码 | 无通用权重；仍需逐折 SSL 与监督两阶段 |
| EEGMamba | `dbc83fa072744201e8897aeb9f65007b952ad323` | `assets/eegmamba/pretrained_EEGMamba.pth`，13,313,895 字节 | 独立 Linux 环境完整 117/117 keys CPU strict；GPU 工程检查等待顺序时隙 |

表中的 `assets/` 指 `tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets/`，`vendor/` 指 `tridep_ccml/eeg_all_candidates/vendor/`。Hugging Face 权重仓库版本与 GitHub 作者代码版本分别登记，不把二者混为同一 commit。

## 已交付的三个标准适配器

`adapters/luna.py`、`codebrain.py`、`biot.py` 已提供 `load_encoder(device, trainable)`、`adapt_raw(raw)`、`encode_windows(model, tensor)` 和 `exact_finetune_parameter_names(model)`。它们只读取 `[W,19,1000]` 未逐导标准化的 E2v2 raw，保留正式权重、原生表征维度和准确解冻名单。

三者均通过 CPU 合成输入的严格载权、重复前向完全一致、W=1 与 W=2 输出维度、有限微调真实梯度、一步实际参数更新及冻结参数不变检查。证据在 `remaining_adapters_cpu_check.json`；只使用独立高斯输入，没有读取队列人员 EEG 或标签，没有 GPU 调用。3 次合成工程 optimizer step 单独记账，正式拟合数为 0。

- LUNA：使用实际 19 导名称调用作者 MNE `standard_1005` 坐标 helper。返回的是名义头坐标、单位米，不是本次受试者实测电极位置；随后按作者代码内部逐轴 min-max。物理重采样至 256 Hz、5 秒 1,280 点，再按作者 `ChannelWiseNormalize` 的 unbiased standard deviation 逐窗逐导标准化。读出为最后 norm 后 32 时间 token 平均，4×64=256 维。预训练权重有通道统一查询 `cross_attn.query_embed`，没有下游 `classifier.learned_agg`；二者不能混写。
- CodeBrain：正式作者下游骨干是 `Models.SSSM.SSSM`，不是同仓库的 `modeling_finetune.NeuralTransformer`。1 秒 patch、5 个 patch、幅值 `/100`。完整权重中两个 tokenizer 预测头保留加载，但 `if_codebook=False` 让特征前向只经过 EEGSSM 和输出 norm。作者写死的 mask `.cuda()` 在运行时仅改为 `.to(h_s.device)`，不改方程，原 vendor 字节保留；作者 W=1 时 `.squeeze()` 去掉 batch 轴，适配器明确恢复输出轴。
- BIOT：固定 16 条双极导联顺序和标准命名映射，在共同参考 raw 上先差分，之后每条导联除以 `95th percentile(abs)+1e-8`。19 导已经覆盖全部端点，不需要 Fz/Cz。预训练 10 秒、本地 5 秒的差异写入输入合同。采用作者矩形窗 STFT 与 Transformer 原生均值，不静默替换窗函数。

## 其他方法的实施核验点

CSBrain 的默认 Mumtaz wrapper 使用 Fz/Cz 及约 19,000 维展开大头，且按 shape 过滤加载；本计划应对完整骨干 strict 加载，并重新登记真实 FCz/CPz 脑区拓扑和小头。作者 dataset 幅值为 `/100`，不能把这写成经过实测物理校准。本地 E2v2 原始数值的物理校准仍未确认。

REVE 下载了 `brain-bzh/reve-positions` 固定版本 `befa5b57a455b77cf302daf610c2e9ed8140bace`，包含 positions JSON、position names 和精确 embedding。应按本地通道顺序严格索引这些官方名义数字，未知名称直接报错；原 position wrapper 会丢掉未知名称，不能任其造成信号和坐标数量不一致。

EEG2Rep 作者 `trainer.py` 的有效损失为 mask target/prediction MSE + prediction 方差项 + prediction 协方差项；另外的大段双侧 VICReg 是注释，不能误抄。SSP 是两个可能重叠的连续片段。代码 `copy_weight()` 使用 `param_b.data=param_a.data`，会使目标与上下文共享存储，训练器又在每轮开始调用它。执行版必须声明修复：仅初始化时独立 clone 目标，之后每个逻辑 optimizer step 一次 EMA，验证不得更新。真实逻辑 batch 的方差协方差不能被普通梯度累积冒称等价。

EEGMamba 正式类在 `models/eegmamba.py`，采用 `modules/mixer_seq_simple.py` 的 Mamba2，12 层、D200、headdim50、state64。作者 `quick_example.py` 仍引用 CBraMod，不能将该例子当作 EEGMamba。MixerModel 每层后对 hidden states 和 residual 翻转序列，准确描述应为“逐层交替序列方向的作者 Mamba2 实现”；它不是每层并行两路 Mamba 后拼接。当前仍未取得公开全文，因此最后实验必须称作者公开实现适配。

EEGMamba 运行资源已准备：使用 `TRI-DEP-Ubuntu-24.04` 中既有 `/home/tridep/.venvs/tri-dep` 的 Torch 依赖作为只读基础，在本轮 `assets/eegmamba_linux_env` 创建新的独立环境，安装官方 GitHub release 匹配轮包 Mamba 2.2.4 和 causal-conv1d 1.5.0.post8。新环境 Python 3.12.3、Torch 2.5.1+cu124、Triton 3.1.0。完整 117 keys CPU strict 加载成功，12 层均为真实 `mamba_ssm.modules.mamba2.Mamba2`；随后按作者下游设置移除 reconstruction proj_out，特征骨干 3,277,243 参数。GPU 前向依赖官方 Triton/CUDA，等待单卡顺序时隙后验收；当前不能把 CPU 构造等同 GPU 兼容通过。证据为 `eegmamba_cpu_strict_load.json` 和 `assets/eegmamba_linux_runtime.json`，旧 Linux 环境未修改。

## 运行环境与可复用材料

默认系统 `python` 不能作为训练入口；已确认 `S00/.venv/Scripts/python.exe` 有 Python 3.13.14、PyTorch 2.8.0+cu126、numpy、scipy、einops、safetensors、MNE。GPU 查询为 RTX 4080 Laptop。三适配器所缺的 timm、rotary、opt_einsum、BIOT 线性注意力及其子依赖已安装到新 `assets/python_dependencies/`，不改变原 S00 环境；每包版本与 METADATA 哈希保存在 CPU 工程报告。

可直接复用既有正式 EEGPT strict loader、LaBraM 作者源与权重 manifest、E2v2 修复 raw、固定 five-fold 和 cohort manifest。新的 EEG 表征与逐折训练权重必须放在本轮目录，不能让旧特征、旧 60 轮最终头或一次前向替代新路线正式完成。

来源：[CSBrain 作者仓库](https://github.com/yuchen2199/CSBrain)、[REVE 作者仓库](https://github.com/elouayas/reve_eeg)、[LUNA 作者仓库](https://github.com/pulp-bio/BioFoundation)、[CodeBrain 作者仓库](https://github.com/jingyingma01/CodeBrain)、[BIOT 作者仓库](https://github.com/ycq091044/BIOT)、[EEG2Rep 作者仓库](https://github.com/Navidfoumani/EEG2Rep)、[EEG Conformer 作者仓库](https://github.com/eeyhsong/EEG-Conformer)、[EEGMamba 作者仓库](https://github.com/wjq-learning/EEGMamba)。每条工程结论以本轮固定 snapshot 代码与严格加载记录为准。

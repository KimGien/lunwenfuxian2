# EEG2Rep 逐折自监督实施与 CPU 验证

2026-10-08：正式训练器 `tridep_ccml/eeg_all_candidates/ssl.py` 已完成，CPU 验证通过；本审查没有执行 CUDA 工程或正式自监督拟合。目前不能将 EEG2Rep 的 15 次自监督状态标为 ready / complete。

准备入口：`S00/.venv/Scripts/python.exe -m tridep_ccml.eeg_all_candidates.ssl --prepare`。它只使用原 fold 1 训练人员的真实逻辑 64 窗进行前向、梯度和完整状态恢复工程，0 次 optimizer / EMA 更新，测量 bfloat16 峰值显存及耗时。仅成功后锁定实际逻辑 batch、精度和全部来源，再将 15 个拟合标为 ready。若 OOM，实际错误及开始 / 完成的前向、梯度成本均记录，状态保持 blocked；没有用分批 VICReg 损失梯度累积冒称 batch64。

正式入口：`S00/.venv/Scripts/python.exe -m tridep_ccml.eeg_all_candidates.ssl`；可用 `--seed 11 --fold 1` 指定单组；中断后用 `--resume`。初始 300 轮，最早 60 轮、patience 60、满足登记条件才延到 500；AdamW 1e-4、weight decay .01、10 轮热身与余弦至 0.1 初始学习率、延长后保持末学习率，梯度裁剪 1。只有本折验证固定掩码下的作者 SSL 损失用于 best / stop / extend。

每个训练人每轮完整使用 60 窗；先独立打乱每人的窗口，再在 60 个轮转中打乱人顺序，保持逐人等量。26 / 27 人在真实逻辑 64 下每轮分别 25 / 26 次更新，末批 24 / 20 窗。训练采样器和 SSP 使用各自独立随机流，所有状态存入 checkpoint。SSP 的两个 125 长度片段起点逐逻辑批次保存，可以精确恢复全部可见 / 遮蔽索引；保留作者可能重叠片段和重复可见索引的行为。

验证仅使用本折原 val 3 人的全部 180 窗，固定组批与独立固定 SSP 掩码；三组作者损失取算术平均。eval / inference 下没有梯度、optimizer 或 EMA 更新；模型状态、每个 module 模式和训练 RNG 前后恢复。当前训练器没有外层测试前向、诊断标签输入或测试评分入口。

作者准确非优化设置位于 `adapters/eeg2rep.py::DEFAULT_CONFIG / SSL_CONTRACT`：19×1000 输入、pool2 → 500 token、emb16、heads8、FF256、context4层、predictor2层、dropout .1、SSP 两段各125、visible ratio .5、EMA .99；MSE / std hinge / covariance 三项系数均1，std 使用 unbiased 方差和 epsilon1e-4，covariance 除以 B−1 再按 D16 归一。方差 / 协方差仅在一次完整逻辑 batch 上计算；损失统计保持 float32。适配保留 E2v2 native 共同参考信号，作者 main 的 Norm 默认关闭，没有额外全队列拟合标准化。

作者缺陷修复均显式登记：target 只初始化一次且与 online 存储独立，绕开原 `.data` 别名 / 每轮重置；监督读取不使用作者 `linear_prob` 的 no_grad / squeeze；不调用每5轮访问 test 标签的作者 trainer。预训练的 target 只有每次实际 optimizer step 后的一次 EMA 可以改变；没有 target 梯度。未使用的原 predict_head 保持初值不变，不能将 EMA 正常变化误写为冻结参数不变。

每轮保存完整 latest、验证 best 的完整模型 / optimizer / RNG / window generator / SSP RNG，及 history / order / EMA 计数；60 /120 /200 /300 /500 保留实际快照。恢复核对身份、原折人员、作者配置、完整轮次 / 窗口 / mask / optimizer / EMA 计数、best 状态哈希和全部 CUDA RNG。latest 是恢复的唯一已提交轮次边界；best 文件及其他曲线可以由 latest 修复。磁盘替换失败不会重复优化计算。

监督对接：构造 `E.build_model(device, stage='supervised')`，严格加载同 seed / fold 的 SSL `best.pt['state_dict']`。其 `identity` 绑定 train_ids、val_ids、encoder_fit_subject_ids、登记、原 raw manifest、工程设置、固定验证掩码清单和初值。监督时只开放 InputEmbedding / Norm / Norm2 / contex_encoder；通过 `E.encode_windows(model, tensor)` 读取 `[W,16]`，保留 W=1 维度及编码器梯度，随后接统一受试者小头。必须核对 checkpoint descriptor 和原训练名单，不能将其他折表征共用。

五项 CPU 测试通过，见 [结果](ssl_cpu_test_results.json)。其中真实 fit 函数在合成 CPU 模型上执行 3 轮、2 次逻辑更新 / 轮，在 latest 第1轮成功后注入保存失败；恢复后所有模型、target EMA、optimizer、RNG、采样顺序、掩码和 best 与不中断流程精确一致。此次最终测试共12次合成 CPU 优化更新；包含一次测试路径身份比较修正后的前次执行，实际共24次，不计入360次正式拟合。CUDA batch64 的资源可用性仍需由队列完成 `--prepare` 才能确认。

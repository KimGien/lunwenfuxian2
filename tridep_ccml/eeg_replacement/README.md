# EEG-R1：原始 EEG、无数据增强

本入口执行 `output/plans/EEG分支替换执行计划_R1.md` 的八个条件、三个种子、原五折，共 120 组正式训练。结果写入 `tridep_ccml/experiments/EEG-R1-37-fulltext/`。

从项目根目录使用现有 `S00/.venv/Scripts/python.exe`：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
& S00/.venv/Scripts/python.exe -m tridep_ccml.eeg_replacement.data
& S00/.venv/Scripts/python.exe -m tridep_ccml.eeg_replacement.prepare
& S00/.venv/Scripts/python.exe -m tridep_ccml.eeg_replacement.prepare --lock
& S00/.venv/Scripts/python.exe -u -m tridep_ccml.eeg_replacement.run
```

`data` 可复用通过哈希检查的原始缓存。`prepare` 首次测量和测试后写不可变验收记录；已经完成时无需重复测量。`--lock` 在任何正式训练之前锁定所有候选及源代码。已锁定的实验拒绝改动源码。`run` 跳过通过来源验证的已完成训练；不从未标记的半成品检查点继续。异常写入 `failure.json` 并停止。

仅需要复算和交付时可运行 `-m tridep_ccml.eeg_replacement.complete`。只有全部 120 个固定模型训练完毕，才能进行外层推理与结果汇总。

- `U0/U1/U2`：独立 EEG CE。
- `C0/C1/C2/C3/C4`：完整 CCML；所有共享语音/文本参数与证据头的初始值逐项配对。
- 标准化仅使用各折训练人；每人全部 60 窗进入同一训练样本，不扩增样本、不增强、不删除窗口。
- 在训练前冻结 SparNet 的频谱与脑区映射、token 平均和注意力公式；详细清单见 `structure_manifest.json`。
- 检查点和监督 128 维特征按条件/seed/fold 隔离。历史 v2、上游代码和历史结果保持原位。

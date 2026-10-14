# TriDep EEGPT匹配4秒备选执行附表

登记日期：2026-10-06。对应汇总计划v1阶段4的EEGPT条件备选；在阶段6收尾后按用户“进行下一步”推进。旧阶段4的延期记录与所有既有封存产物保持历史原字节，本附表不追溯改写旧预算。执行状态以新登记、逐行动账本及独立核验为准。

## 权重条件解除与来源

作者[EEGPT仓库](https://github.com/BINE022/EEGPT)固定版本`ea7282f698c406cd423ccbedbe9960853761144c`的README链接[Figshare共享页](https://figshare.com/s/e37df4f8a907a866df4b)。本次浏览器实际进入Files/EEGPT/checkpoint并打开File info，确认正式文件编号46452745、路径`eegpt_mcae_58chs_4s_large4E.ckpt`、显示973.32 MB以及作者页面MD5 `a165305f329f51b57691c8a3131f9b79`。命令下载仍返回202空响应，浏览器下载等待超时，均保留失败记录，不把页面可见等同下载成功。

固定[完整原文件镜像版本](https://huggingface.co/Zachary1150/ear-eeg-fm-benchmark-model/commit/501c7fa6a368c74f77de916a03b91a4f3178f712)为第三方发布，不称作者授权镜像。实际完整文件1,020,602,524字节，SHA256 `9d63ebc80bba1ad5e3277df4eecfa05d2745e6243a12ff743c4c0a93930dfac0`，且全文MD5与作者File info一致。作者页面只公开MD5，未公开SHA256；作者校验值用于原始文件身份绑定，本地完整SHA与固定镜像指针用于后续强哈希锁定。101MB转换模型不替代这个完整检查点。

原checkpoint是protocol2保存格式，包含旧优化器与调度器对象及POP opcode；本环境torch2.8的weights_only加载不支持该保存流。先严格验证全文大小、SHA和作者MD5，再使用仅允许事前枚举的9个GLOBAL的restricted pickle读取，未知类立即拒绝，保留原文件。白名单为`__builtin__.dict/getattr`、`collections.OrderedDict/defaultdict`、`torch.FloatStorage`、`torch._utils._rebuild_parameter/_rebuild_tensor_v2`、`torch.optim.adamw.AdamW`、`torch.optim.lr_scheduler.OneCycleLR`。提取`state_dict.target_encoder.*`并对目标encoder全部键与形状严格加载，无缺失、额外目标键、部分加载或随机残留；读取旧优化器元数据不恢复训练。加载资格审查仅构建CPU模型并读参数，实际EEG神经前向、拟合与更新均0。

## 两个匹配条件

| 条件 | 固定特征接口 | 后端 |
|---|---|---|
| LaBraM-native4s-originalCNN | 原200Hz每5秒窗取前800点，19×4预训练patch token均值，200维 | 原EEGConvPoolClassifier(200,60) |
| EEGPT-mean-projection4s-originalCNN | 每窗先200→256Hz得到1280点，再取前1024点；原encoder时间与summary均值512维，经固定512×200正交投影 | 同一原EEGConvPoolClassifier(200,60) |

原5秒LaBraM成绩只作历史诊断，不能作为4秒骨干直接控制。比较同时包含encoder、固定池化及接口适配，不能单独归因为纯骨干优劣。

使用阶段1已验收的E2v2 harmonic_guarded原始缓存，每人60×19×1000，不重新选择窗或依成绩改变预处理。沿用已经核实的19导映射及真实导联名称，T3/T4/T5/T6按作者字典映射为T7/T8/P7/P8，z大小写规范化，不补造导联。

EEGPT以float64 `scipy.signal.resample_poly(up=32,down=25,window=('kaiser',5.0),padtype='constant')`重采样，先确认1280点，再截取first1024、除1000并cast float32；不能把1000直接拉伸到1024。LaBraM取原first800、除100并cast float32。两条件沿用E2v2的Cz/common reference；与作者平均参考的差异及未建立设备物理校准继续披露，单位解释为reader约定。

EEGPT采用作者直接EEGTransformer：img_size[19,1024]、patch64、embed_num4、embed_dim512、depth8、heads8、MLP ratio4、qkv_biasTrue、norm eps1e-6、各drop0。输出[batch,16,4,512]按时间及summary以float32均值为512。投影仅生成一次：numpy.Generator(PCG64(20261005))的float64正态512×200矩阵，reduced QR；每Q列乘对应R对角符号（非负取+1），scale1、cast float32并保存文件哈希。投影输出固定以float64矩阵乘法累加后cast float32，不学习、不依赖人或标签，不切取前200坐标。

## 人员、训练与隔离

使用原37人、16名MDD/21名HC，原五折及种子11/22/33。每折train26/27人、val3人、test7/8人。37人已经反复用于开发，本次仍属开发比较，没有新增独立人员。阶段6已公布14人的结果不进入本次拟合、选择或评价；其既有确认产物受哈希保护。

每个条件3种子×5折=15次拟合，共30次。每模型498434个可训练参数，Adamax lr0.005、weight_decay0、batch16、固定60轮，每轮2次更新，每模型120次，共3600次更新及3600次训练梯度计算。仅新增CNN后端更新，两个预训练encoder与全部原S/T保持冻结。每配对模型相同初始化、相同人员顺序，保存全部初始参数、每轮损失、完整批次顺序及最终参数。不用val早停或选型，不额外做真实训练梯度探针；纯数学与合成合同测试另列，不藏入正式预算。

登记后的正式行动及每个拟合模型在首次模型构建或正式前向前保存started记录；登记前CPU strict-load资格只构建并读取冻结权重，无真实数据前向或更新，单独保存资格记录。途中失败保留记录，拒绝静默自动重试。全部30次拟合完整通过后才允许统一train/val/test推理并发布成绩，不按某个折或种子的结果提前扩展。

## 前向与评分预算

| 项目 | 实际登记总额 |
|---|---:|
| 每encoder首次窗前向 | 2220 |
| 每encoder重复窗前向（每人first8整批，同batch8） | 296 |
| 每encoder总窗前向/总批次 | 2516 / 333 |
| 原S/T新前向、训练及ASR | 0 |
| CNN主评价及完整重新加载的人次前向 | 2220 |
| 独立CNN重放的人次前向 | 1110 |
| CNN总评价人次前向 | 3330 |
| 两条件全部train/val/test逐人记录 | 1110 |
| 两条件×两视图×3种子×5折×3split的六指标组 | 180 |
| 原分支/EEG骨干预训练、微调、内层、探针、校准、早停、阈值搜索 | 0 |
| 新融合选权及候选搜索 | 0 |

每encoder batch8逐人提取；60窗为8批，再对first8同一整批重复一次。相同批形状输出必须逐位相等，权重及状态前后完全一致。保存已计算的scaled inputs、token、固定池化值与重复结果供独立零前向数学核验，不因为保存副本虚增或隐去前向。

固定ST控制与S/T概率直接复用阶段0同种子同折文件。固定EST采用原阶段0每折原EST权重，用本条件EEG概率替换原EEG概率，S/T概率保持原值；没有重新选融合权重或恢复任何grid。登记前原来源算术复核表明14/15组wE=0，至多一个种子能随新EEG变化，因此该固定配置本身无法通过“两种子改善”的融合分类门槛；这限制不能解释为EEGPT表示本身失败，单EEG配对仍有完整可检验性。若原wE=0，如实报告新EEG在该固定融合配置不影响输出，不通过临时调权争取收益。旧原EST、固定ST与5秒LaBraM只按各自历史角色列出。

## 主端点、保留与停止

所有六指标Macro-F1、BA、MDD召回、AUC、NLL、Brier全部报告。主指标每个种子先平均五个外层测试折，再平均三个种子，合并37人预测的指标另列。111条种子测试记录不当作111人。阈值固定p>0.5判MDD，等于0.5判HC；NLL沿用阶段0开发比较的概率clip1e-15（与阶段6新人员评价单独的1e-7记录区分）。完整train/val结果只作诊断。

单EEG保留：EEGPT相对匹配4秒LaBraM，F1增量至少0.01，至少2种子严格改善，BA不下降，MDD召回降幅不超过0.05。融合保留：EEGPT fixed_EST同时通过相对匹配4秒LaBraM fixed_EST、原EST及固定ST的同四条门槛；三项比较、概率改善及逐人新增正确/新增错误全部保存。NLL与Brier同时下降只作独立概率结论，不代替分类保留。

权重身份、strict加载、真实窗口/通道/单位、原人员/来源或数学合同失败则先停止，不能换来源不明checkpoint或降级为随机骨干。完成登记条件后未达目标结束该路线，不追试投影种子、单位、池化、参考、窗长、轮次、学习率或阈值。若确需处理实现故障，保留先前行动与预算证据，另立修正记录，不能隐去失败前向或拟合。

独立流程为：来源及零前向strict-load资格→冻结登记→特征提取→独立特征验收→30次配对拟合→统一评价/重载→1110人次独立重放→零前向数学/来源/预算全核验→完成报告及完整哈希清单→封存后只读核验。原0—6阶段产物与计划v1一并保护。

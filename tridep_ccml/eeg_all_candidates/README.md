# EEG 九篇候选与长训练执行模块

## 当前保留规则：原 TRI-DEP 基线（2026-10-13 用户修订）

当前保留基线为原 S00/B0-37 的 CBraMod-Mumtaz＋原CNN 测试 Macro-F1 **0.4494011544011543**，来源为 `S00/reports/b0_training_summary.json` 的 `aggregate.eeg.macro_f1.mean_of_seed_fold_means`。每条已完成路线的原主要快照严格高于该精确值即保留；不再附加超过LaBraM、至少提升1个百分点、种子数量、BA或召回门槛。

九候选当前保留 LUNA 冻结小头、CodeBrain 冻结小头和末层微调、BIOT 末层微调，共3个模型家族、4条路线；LaBraM全部5条对照也保留。合计9条路线及其135个种子/折模型。保留不扩展到同家族低于基线的其他路线。

当前入口是[修订版完成报告](../../output/analysis/eeg_all_candidates_v1_20261008/九候选统一完成报告_原TRIDEP基线修订版_20261013.md)和[当前保留索引](../../output/analysis/eeg_all_candidates_v1_20261008/current_retention.json)。[修订版比较表](../../output/analysis/eeg_all_candidates_v1_20261008/unified_comparison_original_tridep_baseline_20261013.csv)将历史匹配控制判定放入 `historical_*` 列，当前 `retention_passed` 使用本次新规则。

新入口仅从已封存分数重建保留判定，并复核原S00逐人预测、主要快照和模型来源；不导入训练器，不重训或推理，不改原权重、分数、预测、完成清单或旧评价源码：

```powershell
& 'S00/.venv/Scripts/python.exe' -B -m tridep_ccml.eeg_all_candidates.retention_revision
& 'S00/.venv/Scripts/python.exe' -B -m tridep_ccml.eeg_all_candidates.retention_revision --check
```

本次是已知测试结果后的用户指定开发保留修订，不改写为原预登记或统计显著性结论；不自动接入融合、部署或恢复已取消的训练。下面的完成门、旧匹配控制结论和执行入口均作为历史训练及验收记录保留，当前保留状态以上述新入口为准。

## 最终交付：九候选全计划已实际完成（2026-10-13）

- 360/360原正式拟合、135FT/405实际CPU门、统一评价（345监督拟合/435固定快照）、12条零optimizer资源补测均完成；原身份、分折、预算、选模、阈值不变，不重跑任何已完成任务。
- 同一Cost19 `final_full_plan_cost_1791878935037825200.json`（SHA1213c45eaceeb78b0644ade79aa71a9b9638d05a5b7e6bce3f87d4a228515354）与wide9 `full_plan_acceptance_v9_1791880718467386600.json`（SHAa25798add925ba129df97aff7f2c2e2043fc90e568b6977479676d84f27d65aa）已实际全30通过；五项独立语义、G3最终bundle、暂停/三loss附页和memory5门、系统全范围独立审查与standby6门全部实际exit0。
- 最终系统门 `system_standby_final_delivery_v6_verified_final_delivery_with_bound_standby_scope_1791882360842078300.json` SHA085536eec1a62d1b4c04a369c0222eefc2b9b5dd9ca6ebb64a7c33a9befbf8fc。最后窄manifest修复四源独立审查b507af33已绑定；原v4失败4b68f357保留，原Cost19/wide9/194源等旧字节未改。
- 最终交付清单：`output/analysis/eeg_all_candidates_v1_20261008/EEG九候选最终交付清单_1791882555531097700.json`，SHAff437ec257c5157005d5d1bd116713fb262d789012d70d75641a66442cd0afac，59份直接绑定，包含九候选逐项结论、比较CSV、OOF、真实成本/报告、全部最终门、恢复链、成本附页与三份I/O勘误。
- 九候选全部完成但均未达到预登记的工程改善门槛，非统计显著性结论；37个独立受试者不因三个seed成为111人。135末态encoder均更新，但最终48个选中checkpoint为head-only热身。Mamba反向顺序/布局改变、原CUDA梯度逐位等价未建立、外部预训练参与者重叠排除未知及非完整论文复现均如实披露。
- 完整系统范围登记→Cost19快照，1240计时记录/208原始事件/38窗口已独立核清；513未知端点保持，360 possible为保守包络，非全360真实暂停。精确active/历史总量/三loss未知退出与尾仍null，未混减wall-QPC或重复加嵌套/暂停成本。
- 交付已完成；按用户要求暂停eeg自动接续。无待训练、待评价、待资源或待验收任务。下列各入口仅历史记录，不得据此重启旧队列、资格、审计或永久intent。

## 当前有效入口：360训练、统一评价、12资源与最终Cost19已完成，最终验收中（2026-10-13）

- 最新实际：wide9已工具exit0，全30 verified_complete，full_plan_acceptance_v9_1791880718467386600.json SHAa25798add925ba129df97aff7f2c2e2043fc90e568b6977479676d84f27d65aa；G3v4实际bundle final_cost_sync_control_bundle_v4_1791881440239138400.json SHA63ed8127d959eee57599810aa4da2540d5fff07f553fc982f5c2649aacefe670。两者不重跑。fullscope helper1实际产物full_system_power_timer_scope_1791881536952146900.json SHA6b9f3f6cc223ba782eab79955e8fb2ed026324224a3fae56b460fa677f7c7e23，1240库存/513未知端点，可能受影响360为保守包络，非360都睡眠；正由原loss3实际独立审查。
- 当前仅附加门阻塞：pausev4实际读取固定Mamba predictions_manifest元数据时，被自身文件名禁读规则误拦（未打开该清单）。失败actual_pause_v4_manifest_guard_failure_1791881597248839900.json SHA4b68f357d86da043b19926f486fdfd8bab8948c5243d4b339fb602331010521a保留；原author正作最小pause5/memory5/standby5/systemgate6窄manifest path+SHA许可及必要pins，新源须独立审查后执行。旧v4不重试，已完成训练/Cost19/wide9/G3/源194均不改。

- 原360与CPU17实际135/405已完成，所有已完成训练/资格/CPU/评价均不重跑。E13真实all_evaluation_complete，all_windows int exit0，05:52终态及released_owner；原E12 Mamba15结果/source12严格复用，新Windows/source13完整23条件/345监督拟合/435快照已汇总，原evaluation/complete.json SHA7cc8a34d33f78497af0e5cc59242d095ddc27d1d8ebd9b64e8e54c48ff3c04a2，比较CSV/OOF/九候选报告已生成但全计划未验收。
- 最新10源Cost19/post10/resource9/finalizer9/wide9及五v4附页完成真实独立38项审查：independent_E13_final_compatibility_source_review_1791875707354688800.json SHA07a8c699862bbbedfdfe0fea4f91868d4a582d631f83eee2282655de6de5eabe，reviewer v2仅修复v1对序列化AST返回字符串的空格敏感检查；v1/失败证据保留，10源与原194均未改。
- resource9已实际--run工具exit0，12个原条件/0optimizer补测完成；supplemental_head_resource_v9_complete_1791876496108405800.json SHA1df4a4e95444ee44bbe650e7293f86530c11b21a30eabcfe334749b461247e16。不要重复补测。finalize_full_plan_cost_v9.py已实际--run工具exit0：final_full_plan_cost_1791878935037825200.json SHA1213c45eaceeb78b0644ade79aa71a9b9638d05a5b7e6bce3f87d4a228515354，verified_final_cost/360、input_read_errors=[]；成本报告SHA7abafcafb94062ffaa302e018575e3057f25368b763a36018c3e84743c2b6c9a。五项最终独立语义审查已实际通过：independent_final_semantic_review_1791879676618795700.json SHA4ea1fe8142f6bf5d50d4c829ccb04aca88b9e675ddfac7d79216b8ed1c231043，26项实质检查绑定同cost/E13/原360。wide9已实际--run --require-complete运行（工具session93345），仍等待真实30项结果，禁止重复启动。16:23只读快照launcher7884/parent20140/birth134363533311889474，worker37720/parent7884/birth134363533312006488；这些仅快照，控制前重核。Cost19已完成，不重跑collector。进程身份以当前CIM完整birth/parent/command核查，不沿用旧PID。
- 下一顺序：真实同一cost19/成本报告→五项最终独立语义审查R04/R06/R10/R27/R29→wide9 --run --require-complete全30→同cost/wide G3v4 bundle/暂停loss v4附页+memory gate4→实际全范围系统事件独立审查/最终standby门。resource/finalizer/wide9 --source-review使用上段07a8审查及其SHA，wide加同cost-artifact/cost-report/semantic-review。
- 已实际仅读完整Kernel-Power/PowerTroubleshooter前缀system_power_event_full_range_prefix_1791876271590.json SHA1269fec163aaf846676c88b54f2e252879467a641eeba5d7871111380c307a87，UTC Oct7 00:00到Oct13 07:24:30.7328387，208 raw事件、31个ModernStandby配对，另7个PT睡眠窗口（注册后5个）；日志最旧Jul8。已补真实无缝后缀system_power_event_full_range_suffix_1791879383255.json SHA2781a3d581eb0d1dcba0916e15e83b61e9bdd07baaa12f6a09a57b0d50caa66b，到最终cost captured_at UTC08:08:55.027167，原生查询零新事件；尚须同wide实际全范围分析与独立最终审查。旧standby gate4仅认LUNA33/f1不足以覆盖新增计时；最小full_system_power_timer_scope_v1.py（SHAc28bdd912c3a97510cd17d725ce7542c078640ec3e77f2c48340d6c0e923caec）/audit_system_standby_final_delivery_v5.py（SHA5bda427e754deaa5f0f1aa246a29dc0ff5274de36bc579598e1d93a2325bee29）已独立35项通过：independent_full_power_scope_source_review_1791878543512820500.json SHA4f8a16a25a90f2b2b677f2a62e3454e1ffac6f399c98215ea3799d873b3eb2dc；旧源/Cost19/wide9不改。等真实同cost/wide运行helper，原loss3独立审核actual scope v2，再运行gate5严格实际门。ModernStandby会话不能等同全程停算，boot220/scenario11 SleepEntered=false/零DRIPS；未知active=null，保守包络/可能重叠完整披露，不用QPC秒倒推wall或扣减。
- 部分完成不代表结束；全部九候选/比较/OOF/真实成本、三loss/暂停/全系统事件最终门及I/O勘误实际交付后才暂停eeg。当前继续ACTIVE/25分钟，用户无新暂停。


## 当前有效入口：Mamba15评价保留，E13接续Windows统一评价（2026-10-13 04:55）

- 原360与CPU17真实135FT/405CPU、零errors/deferred/pending已通过，原194/训练/CPU17门字节不变，全部不重跑。canonical post_training_cpu_recovery_v1/complete.json SHAfc3c163fcbd62a73cf4f01cdfb7ddfb919b86dc479dd6f4fc22a01f4f9aa1181仍是完整预测试门。
- E12 Mamba15实际评价全部完成，mamba_linux真实int exit0，worker SHAb2bcbb33634fd8a095de4b31e7664e673cce67bcb3815ce6143f804a522152c2，manifest SHA9d3a15c32609ae74f7db91f1ad9105f616a0135dfa51de5a0a484645a71d5661；保留原E12来源/预测/指标。Windows首epoch60已实际读取1次CPU checkpoint header，在首ledger写入前因原临时路径274字>MAX_PATH失败，0模型/forward/optimizer，ledger elapsed未知。旧E12父status仍running原字节，已真实child0/1与released_owner，主进程工具exit1，不能伪写全评价完成。
- 二源独立审查independent_evaluation_storage_recovery_source_review_1791838352376118100.json SHA4775863335c3d94bc17fb8f807d6e9cd373f97f3e0eeafcd066ee9e782ad2d1a已通过；只在私有storage改变同目录短UUID临时名，其余atomic/fsync/immutable/retry AST原样。全部435 canonical最长238/短temp218；实际Windows旧275失败、新217成功。Mamba15×3role Windows原metrics严格等于旧Linux，未用容差。
- evaluation_storage_recovery_v1实际--run工具exit0，canonical complete.json SHA27d9cbb08206c64bc60cbb4b74938a9e52a87971f334666b63d2a818307ea0ad。evaluate_v13已实际--run，04:55:52建立owned统一评价，只有all_windows；Mamba整个context/cache/完成验证用原E12且模型入口硬拒绝Mamba。新global汇总E13、Mamba15原E12来源分别绑定，不能改旧包标签。
- 真实control为evaluation_v13_control；初始ownerc90c2e9e4709425abfad906466af8ef7，Win40148/parent19464，birth134363120946116518/134363120946011185；子树依据当时实际owner/worker_owner重核，旧E12、Q14、CPU17 PID失效，不重复启动。健康单GPU评价持续，仍未全部结束。
- E13实际完整统一评价并释放owner后，才12资源补测、真正最终成本/语义/30项及同cost/wide的G3/所有暂停三loss/系统全事件范围附页与最终门。因实际E12/E13混合生命周期，Cost18及其10兼容源是封存准备，当前最小后处理迁移由原author/loss3独立审核；有效新入口以其落盘审查为准，不能把旧E12失败套完成门。Mamba原成功评价主账一次，Windows failed header1与unknown主秒保留、worker秒nested，stdlib复用/repro费用分别保留，不重加。
- eeg ACTIVE/25分钟；当前无人类暂停。只有实际九候选/比较/OOF/成本和全部最终验收交付后才停自动接续，现不宣称计划完成。


## 当前有效入口：360拟合与CPU17已通过，E12统一评价运行中（2026-10-13 03:41）

- 原360拟合实际全部training_complete，Mamba15原Q14子任务真实exit0。原Q14仅因CPU16跨平台LR校验失败，failed原字节保留，owner已释放；不得重启训练或伪写原Q14终态。
- CPU17真实完整报告completed_finetune_checkpoint_audit_v17_1791833719887124200.json，SHA7fa47f81891194cc1e71cf0e0005bc469a0699772091914c3ba5194a1d7656b1：observed135/verified135/405真实CPU读取、errors/deferred/pending全空、all135=true。全程0模型构造/GPU初始化。原失败CPU16实际120/360及15错误保留，两个审计主范围分别记账，不能重复或混填。
- 独立post_training_cpu_recovery_v1实际运行exit0，canonical complete.json SHAfc3c163fcbd62a73cf4f01cdfb7ddfb919b86dc479dd6f4fc22a01f4f9aa1181。三核心独立审查cc65a66d504d3e483c2ccfe2a7bab64462dca1df0749fe71646246f4c2679d84绑定原194/原Q14 failed与release/原childexit0及新完整CPU17。只改变Mamba三处期望为原Linux精确数学表，严格==、无容差；训练源和history未改。
- evaluate_v12已实际--run，经完整门后03:41:15登记owned统一评价，当前mamba_linux阶段；Windows全部候选汇总由同一controller自然接续。control为实验/evaluation_v12_control，运行状态/owner/子日志以实际文件为准，不重复启动。初始owner838c722c3f214658bc0a0d1eeb6e5f44，Win18168/parent33484（birth134363076181188052/134363076181079181）；WSL27836→1664，Linux313/parent312/ticks5932/5931，boot6ab9b46a-5c1d-4c7c-9f56-86955100e10f。这些仅当前快照，控制前必须重核完整真实身份，旧Q14和CPU17 PID不用。
- E12真正完成并释放owner后，才resource8实际12条零optimizer补测→finalizer8/Cost18真实最终成本→独立最终语义审查→wide8原30严格项→同cost/wide的G3、暂停三loss和系统低功耗v3附页/最终门。十源独立审查independent_post_training_final_compatibility_source_review_1791833968825479500.json SHA8b0f23b56fb645f5ed33b05cf86f67d772704a4a5ab31e93b9d6f36b89847f74已通过，未实际后处理。有效参数见下一节；旧v7/v2只是历史，不能套CPU16成功前提。
- 两平台stdlib参照主秒数各一次，failed16/新17无独立seconds字段则null；inclusive父记录/active未知/暂停等待/三loss规则保留，不混减wall-QPC、不嵌套相加。最终仍需完整系统事件范围独立审查与全部交付门、九候选/比较/OOF/真实成本及I/O勘误，尚未计划完成；eeg ACTIVE/25分钟。


## 当前有效入口：360训练已完成，CPU17独立恢复门与E12待实际执行（2026-10-13 03:27）

- 原360个正式status已全部 `training_complete`；Q14的Mamba15子任务于02:59:03真实exit0，owner已释放。原Q14 `failed` 原字节保留：自动CPU16实际核验120FT/360次CPU读取，15个Mamba仅因Windows重算学习率相差1ulp而报错，不能冒充已通过135/405。
- 原Linux3.12/固定protocol独立重算1..200完整学习率表及float.hex，实际15条完整历史1452个标量全部严格相等（Linux参考SHA `c7109d9ced13c853ecb6de99606b98ab1c2f2b9978535a0a2c614049fb682463`）；Windows参考保留15处差异。CPU17只改变Mamba三处期望表达式，仍严格 `==`，其他检查/194正式源/训练历史不改，不重训或重跑工程。
- 三源 `audit_completed_finetune_checkpoints_v17.py`、`post_training_cpu_recovery_v1.py`、`evaluate_v12.py` 已由独立34项审查 `independent_post_training_cpu_recovery_source_review_1791833255395495800.json`（SHA `cc65a66d504d3e483c2ccfe2a7bab64462dca1df0749fe71646246f4c2679d84`）通过，准备不代表实际审计或评价完成。先CPU17 `--run`；其真实135FT/405CPU且errors/deferred/pending全空后，运行恢复门 `--run --cpu-report <报告相对路径> --cpu-report-sha256 <SHA> --source-review <上述审查相对路径> --source-review-sha256 <上述SHA>`；再以返回的新门运行E12 `--run --cpu-gate <门相对路径> --cpu-gate-sha256 <SHA>`。恢复门只新增后训练证据，原Q14失败/CPU16失败360读取与实际退出记录全部保留。
- E12真实两阶段完成及owner释放后才进行资源、成本、语义审查与全部最终门。下列原v7/五v2后处理已审源码仍保留，但因新CPU17/独立门/E12的生命周期绑定须使用后续最小兼容入口；不能直接套旧CPU16成功前提。两次平台标准库采集耗时、失败CPU16和新CPU17实际读取成本分别保留，不重复累加。

- 最终必要兼容十源已独立44项通过：`independent_post_training_final_compatibility_source_review_1791833968825479500.json`，SHA `8b0f23b56fb645f5ed33b05cf86f67d772704a4a5ab31e93b9d6f36b89847f74`。有效后处理为 `repair_postprocessing_gate_v9.py`、`supplemental_head_resource_measurement_v8.py` → `finalize_full_plan_cost_v8.py`（Cost18）→真实语义审查→ `audit_full_plan_acceptance_v8.py`；资源/finalizer/wide沿用 `--run --source-review <此十源审查相对路径> --source-review-sha256 <此SHA>`，wide再加同一 `--cost-artifact --cost-report --semantic-review --require-complete`。仍必须E12实际两阶段完成且owner释放后执行12条零更新补测，不能用已审准备代替结果。
- 同一cost18/wide8后使用 `sync_boundary_v3_cost_companion_v3.py`（原G3的909042附页只复用）、`pause_memory_resume_cost_companion_v3.py`、`system_standby_cost_companion_v3.py`，均 `--run --final-cost <同一cost18> --final-acceptance <同一wide8>`，G3加 `--companion <原909042>`；再以原六输入运行 `audit_memory_pause_final_delivery_v3.py --run --require-complete`，以原四输入运行 `audit_system_standby_final_delivery_v3.py --run --require-complete`。三loss、原计时unknown/null、全部I/O勘误和完整运行事件范围要求不减。新增成本仅失败CPU16真实360读取、新CPU17真实405读取和两平台stdlib采集各一次；嵌套秒数不相加。最终交付清单须绑定上述十源独立审查；十源本轮仅准备，尚未实际执行。

## 历史记录：Q14开训与原CPU16/E11预定入口（现以顶部为准）

## 当前有效入口：最后Mamba15已实际单GPU开训，Q14/F12运行中（2026-10-13 01:25）

- 最终三入口 `supplemental_head_resource_measurement_v7.py`、`finalize_full_plan_cost_v7.py`、`audit_full_plan_acceptance_v7.py` 已由独立审查 `independent_final_postprocessing_v7_source_review_1791829456847964300.json`（SHA `9e7c8f634d6d4048d42f129cc3abcc1bfa8a3700b0a6d592c6614a7d18f09baa`）通过，尚未实际执行；194 份正式源未改。仅在真实 360/CPU16→Q14 owner 释放→E11 评价终态后，按此顺序运行，三者均需 `--run --source-review <上述审查 JSON> --source-review-sha256 <上述 SHA>`；wide 另需同一份 `--cost-artifact <cost17 JSON> --cost-report <成本报告> --semantic-review <真实独立语义审查> --require-complete`。G3、暂停三 loss 与系统待机最终附页及交付门另经兼容审查后执行；准备或部分通过不等于最终交付。

- 成本附页兼容五 v2 已独立通过：`independent_final_companion_v2_source_review_1791830805989404100.json`（SHA `19c03c3f1ef0e82d1960a12ef528d4e32b89db5e0b0c5fb12b6161a7716964a5`），尚未实际执行。真实 cost17/wide7 后用 `sync_boundary_v3_cost_companion_v2.py` 的 bundle-only 模式复用原 909042 附页；`pause_memory_resume_cost_companion_v2.py` 复用原 60217 中午终态并绑定三 loss；`system_standby_cost_companion_v2.py` 以 loss2 关闭旧 LUNA 父计时生命周期，原 `running`/未知秒数保留。附页均需 `--run --final-cost <同一 cost17> --final-acceptance <同一 wide7>`，G3 再加 `--companion <原909042 JSON>`；最后 `audit_memory_pause_final_delivery_v2.py` 原六输入与 `audit_system_standby_final_delivery_v2.py` 原四输入均加 `--run --require-complete`。系统门继续要求完整运行事件范围/全部配对和逐受影响范围证据，额外 fit 未核清仍拒绝；最终交付清单须绑定此五源独立审查。

- 用户继续与一直进行到结束授权有效。真实Q14已登记并hidden启动原Mamba15；前345保持完成，首Mamba11/f1已training、initial/latest真实生成。01:28快照epoch14/56updates/378back、durable13，已含lastblock阶段；不可把当前fit算完成。
- continuation_queue_v14/registration.json SHA93b19228fb8d3f032f87be4072f4ff2fb3a0bbe2d89adce93117896b5bc59742，194来源继承原175与全部历史。有效Q14/D9/F12/O9/W9/V9/CPU16/cost17/E11/review_actual_repair_engineering_v8.py/repair_postprocessing_gate_v8.py；194独立source审查046098ae6b1ba58d57add72390ff71baf5337d51d3b25efd2c9d16e66b68a0ed通过，真Qual10成功review1d108208...已绑定。原15blocked字节在handoff逐字节封存，登记后使用不可变证据，旧Qual10门/字节不热改。
- 初始owner361f8bb2b4c440bdbd17c642cd5d4a83：Win16292/birth134362995022014588→28180/134362995022556814→WSL22868/134362995114818021→shim39516/134362995115118312；Linux420/parent419/start_ticks600，boot42d96cca-b5e8-461f-b5be-8864f76d6636。已核Win四完整身份全match；每次重读实际owner/worker_owner和全birth/Linux身份，历史PID不可直接控制。
- 健康单GPU队列不重复启动、重测成功资格、追加探测或另启未经owner允许的CPU工作。Q13/CodeBrain15永久不重跑，Qual4-10与单次诊断所有intent禁止盲重跑。Q14在真Mamba15全部完成后自动跑原生CPU16全135FT/405CPU门；此前不读outer概率、不评价。旧CPU15仅120/360历史partial，不重复或冒充完整门。
- 全360/CPU16通过且Q14释放owner后，才执行真实E11统一评价（绑定原F11及新F12两实际来源）、12资源、真最终cost17/报告及独立语义/30项、同cost/wide G3及所有暂停/三loss/完整系统低功耗附页和最终交付门。资源/finalizer/wide 三 v7 与成本附页五 v2 的独立源码审查均已通过（见上段），实际执行仍须前置门齐备，不用准备报告替代实际产物。Q14只training/CPU，E11是另一个owned动作。
- 数值修复前向原样、反向布局/归约顺序改变、原CUDA梯度逐位等价未建立；额外native函数计数仅副账，不与wholeback相加，不是CUDA/autotune launch count，也没有提速结论。全部原身份/分折/预算/选模/阈值/micro1逻辑8/forwardchunk8/FP32不变。旧失败、诊断38.444831497及Qual10 66.171489932秒主账各一次，父秒只nested；暂停/低功耗/未知耗时按原规则保留。
- eeg ACTIVE/25分钟，用户无新暂停。再暂停先写三strict marker，再按当时真实Q14身份暂停、已有挂起不叠加，并暂停自动接续。原Q5固定控制脚本失效。只读小范围优先，健康运行静默；九候选/OOF/比较表/真实成本及全部最终门实际交付后才停接续。

## 当前入口：Mamba Qual10实际成功，最后15正式拟合链待注册（2026-10-13）

- CodeBrain15/Q13/CPU15全部真实完成：总345/360、120FT/360CPU、零错误/暂缓，Mamba15仍blocked；原Q13已释放owner。旧175全部原字节保留，不重启Q13或重测CodeBrain。
- 单次诊断mamba_repeat_diagnosis_v1/EEGMamba-lastblock-finetune/1791823118874568500实际failed：6F4B0step、38.444831497秒；损失/RNG/grad keys完全一致，只A_log、dt_bias及conv1d的weight/bias4参数差异（最大1.862645149230957e-9）。原intent/失败/来源008981不重跑。
- 新最小R1/G4/M5已实际修复私有反向batch分解/布局/参数归约，保留原前向/FP32/训练micro1逻辑8与forwardchunk8，未放宽逐位门，原CUDA梯度等价仍未建立。静态audit不含live计数；额外native函数call与传输成本单列，未宣称提速。
- Qual10实际exit0/engineering_verified：deterministic_repair_qualification_v10/EEGMamba-lastblock-finetune/1791824574948282100/report.json SHA85619b334e6eacf1983adee0c4b24c8d3be9f1f7e0a6fe8e49d8d9d224392ece。6F/4B/48窗/0step；native forward、phase1/11 repeat loss-grad-RNG逐位一致、零参数/缓冲更新及完整checkpoint恢复全部pass，owner释放。primary66.171489932秒，controller86.3614763秒仅嵌套，不相加。末层chunk/causal native函数各16started=returned、2wrapper，其余11层0；不等于CUDA kernel/autotuner数。
- 四源独立审查2c5e3545356936cf89581e328b1f0d798418e1d416ca8471be8fb30734a2f8c1已通过；原10字段强门真实WSL缓存预检7a9f8ffa...通过。Source Qual10 e361be46244a9b0d2a07a29dfe634ee2f191a488dfc355eba1d6a17027c0e39b、R1 28d5.../G4 c965.../M5 14e8...全部冻结，禁止重跑成功资格intent。
- 当前正准备最小Mamba-only Q14/D9/F12/O9/W9/V9/CPU16/cost17/E11/eng8/post7混合链：继承实际Q13的175、全部失败/诊断与三loss；待真实源与工程审查后才注册/启动原Mamba15，尚无正式Mamba拟合。不得伪填Q12或Q13为全360完成。下一次先读该链实际source/review/registration/owner，准备不代表启动。
- 新正式15、实际360/原生135FT405CPU门、统一评价/owner释放、12资源、成本/语义/30项及全部G3/暂停/低功耗门仍必做；全门前不读outer概率。后续cost必须恰计一次诊断38.444831497与资格66.171489932主账及原失败，native子计数不与whole-back相加；旧cost16不改。用户未暂停，eeg ACTIVE/25分钟。

## 当前入口：CodeBrain15及CPU15已完成，Mamba重复差异单次诊断待运行（2026-10-13）

- 实际345/360已完成；CodeBrain全部15训练complete、Q13子任务真实exit0且owner释放。Q13终态codebrain15_complete_mamba15_blocked_no_evaluation，保持175原封存来源；不重启Q13或任何旧永久intent。
- 真实CPU15报告completed_finetune_checkpoint_audit_v15_1791821696169074000.json，SHAb2e46f49da6cd79f09de15077641e15b4124fe9545d813b6ef45ea6c6834ab1e：120FT/360CPU、0errors/0deferred，仍15pending；不可当135/405或测试门。不要重做该审计。
- 已只读核实无实验native、GPUcontexts=[]。新diagnose_mamba_repeat_v1.py SHA008981ad5948bfc885772ed0aceda46b8ca00ffc332797a5b0defc2aa874775c继承Q9原数学/G3/M4，额外仅对已有CPU梯度捕获记录loss/RNG/梯度差异，严格门未改；固定seed11/fold1首train人8窗、最多6F/4back/0step。新目录mamba_repeat_diagnosis_v1，永久intent不可重复。当前正独立审查、尚未实际模型运行，不能冒称诊断完成。
- 该入口即使本次一致也只有diagnostic_complete_no_formal_qualification/eligible=false，永远不是Mamba正式资格。需根据真实差异修复并独立审查、实际资格，再跑原Mamba15；全360/真实135405前不读outer或评价。诊断attempt主账/CPU统计成本及controller嵌套须在后续新cost兼容中单列一次，旧cost16和所有旧来源不改。
- 用户未再次暂停，eeg ACTIVE/25分钟。下一次先查新诊断owner、controller、repeat_diagnostic、attempt/report及独立审查，先处理真实状态，禁止盲重复模型。全部最终评价、12资源、成本、30项及G3/暂停/低功耗门仍待完成。

## 当前有效入口：CodeBrain原15已实际单GPU开训（2026-10-12 13:55）

- 最新人类要求加快，已实际调整顺序：独立受审CB-only Q13正在连续执行原CodeBrain15；Mamba15仍blocked，未放宽资格门、未读outer概率或评价。原330完成数未增加前不把当前首折算完成。
- continuation_queue_v13/registration.json SHA d7ae04ca35c68513b2a19fe0c81a1ceb52e0a70f1e221af33837fa24ee830ba1，175封存来源继承原166/153/139/127/115/104/88及原47等所有历史字节。当前source Q13 398d78ae.../D8/F11/O8/W8/V8/CPU15/cost16；独立source审查61e5cfbd04edb29fc2d23040df4e786304a7d0022585ba3af12dcaf418c255c7真实通过，CodeBrain Qual5 report89f0251...和实际只读review b98a0533...真来源复用，未再运行成功工程。
- 13:55:06实际hidden启动：初始launcher36196/birth134362581066833012、dispatcher2280/birth134362581067174523；owner id c4ec21ba7add4925b241c8a6936c5f8e，初始训练launcher24592。PID仅此时快照，下次依据真实owner、worker_owner、当前全部birth/父子链核身份。13:56首CodeBrain11/f1状态training、initial/latest已真实生成，progress epoch5/20updates/135back、durable4；不是仅排队或模型工程。
- 健康Q13不重复启动、暂停或追加GPU资格。其只允许CodeBrain原15；全部完成后自身执行CPU15部分120FT/360CPU/15pending并明确终态codebrain15_complete_mamba15_blocked_no_evaluation，不冒充360或135/405。训练中不要另启未经owner允许的CPU审计/资格造成排他误拦；计数未变不重复核验。
- Mamba Qual9失败6F/4back/0step/39.218419368秒及全部Qual4-9失败/未知尾/模型构造成本由cost16保留，nested不相加；至少loss/grad/RNG合取一项未一致，不能假装资格通过。后续只在自然任务边界和真实GPU释放后运行新受审Mamba资格/原15；健康CodeBrain训练不为其停下。其余完整360/CPU、统一评价、12资源、成本/30项及G3/暂停/低功耗最终门仍待实际完成。
- eeg已ACTIVE/25分钟，三暂停marker仍absent。最新用户对额度/速度的要求意味着只做必要阻塞修复与真实运行核验，不追加可选性能工程、外围审查框架或重复大范围读写；优先小范围状态/日志，保持健康训练，重要阶段才通知。用户再暂停时按真实当前Q13身份暂停并暂停自动接续，不复用旧Q5固定PID。

## 当前优先接续：先接上已合格CodeBrain原15，Mamba仍待修复（2026-10-12 13:48）

- 最新人类要求加快并质疑额度；先将已真正合格的CodeBrain15接上，不再等Mamba资格，仍只用单GPU。原330/360及105FT315CPU不变；CodeBrain Qual5成功报告89f0251...保持真实复用，不重复模型工程。新最小CB-only Q13/D8/F11/O8/W8/V8/CPU15/cost16在准备、待独立审查后实际注册启动；尚未实际训练，禁止冒称已接上。
- Qual9实际完成6/6F、48窗、4/4back、0step/0formal后仍失败：至少loss/grad/RNG合取的一项不一致，不能判为通过或说每项都错。attempt路径deterministic_repair_qualification_v9/EEGMamba-lastblock-finetune/1791783935769696400/attempt.json，primary39.218419368秒，controller真实code1、nested57.74783720000414秒不相加、released_owner存在。原永久intent及全部166来源/旧失败字节保留。
- 新partial入口只允许原CodeBrain15真实身份；Mamba15仍blocked/pending，所有评价永久禁止于该入口。保持原预算、选模、分折、FP32/micro1/逻辑8/chunk8；CPU15只对新增真实完成FT做部分审计，345与120FT/360CPU仍不是360或135/405最终门。后续Mamba原15和完整评价/12资源/最终验收仍必做。
- 自动审批的旧暂停误判已通过同一诊断的合规重试解决；当前授权有效、三暂停marker不存在、eeg ACTIVE/25分钟。最新166独立审查40a39257...仅证明来源准备，绝不冒充Mamba资格通过或正式拟合完成。

## 当前接续：Mamba实际反向验证失败，准备覆盖遗漏的core前缀和（2026-10-12 13:20）

- 13:34更新：C2/G2/adapter3及Qual8/Q11/CPU13/cost14精确153源独立审查8a7337276c7eac3a74ea4580566668374f655c9c0cacbe1c666870e02dbc53b9已通过；实际Qual8却在模型构造时被G2“Loaded SSD core differs from pinned source function”拒绝，永久intent不可重跑。1791783245785244500/attempt.json SHA92a46ec27179bacaeba0759a919e7eb6ace5a69742180411e7638c66ff9bfa0b：0F/0B/0step/0formal，primary6.125486071000001秒；构造确实已开始，不能称零模型。controller真实code1/worker_returned、23.894923999992898秒仅嵌套，released_owner存在。正在仅标准库核对WSL3.12真实缓存、全文与孤立函数的编译字段差异，尚无新Qual9登记或实际资格；旧153全部保留。
- 自动接续已保持ACTIVE/25分钟并改为版本中性的入口说明，当前三strict pause marker均不存在。诊断子任务的自动审批曾误用Oct11历史暂停拒绝，正在以Oct12最新人类继续、QQ退出及真实controls_released授权对同一只读诊断合规重试；未绕过拒绝或自行执行新模型。用户未再次暂停，不清新标记、不恢复旧入口。
- 用户已退出QQ；本次Qual7实际启动且进程身份核验通过，模型也确实运行。原330/360及105FT/315CPU不变，尚未执行原剩余30正式拟合、outer概率或评价。139份来源独立审查8433357e...已通过，但不是GPU资格。
- 真实Qual7失败在deterministic_repair_qualification_v7/EEGMamba-lastblock-finetune/1791782206453969000：5/5 forward、40/40 windows、3 backward started/2 returned、0/0 optimizer、0formal；primary48.540251404秒。父controller69.25431119999848秒为嵌套，不相加；第3次未返回反向的内部工作未知。attempt SHA9e8b06aa9e32b3612332459a8305583f90fbc1660bd893552ed6304e2593aeb9；controller真实code1，released_owner存在，report不存在，永久intent不可重跑。
- 实际trace为ssd_combined.py:437的Tensor.cumsum：ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])。旧私有G1仅替换helper的torch.cumsum(out=alias)，遗漏core该方法式调用。保留旧G1/C1/adapter2及139全部原字节，新C2/G2/adapter3和完整Qual8/Q11/CPU13/cost14链正在准备；严格FP32/确定性与原native forward不变，不关闭determinism或改安装库全局。
- qualifier7_failed_cleanup_1791782397945.json已只读确认experimental_native=[]、GPUcontexts=[]、lease不存在，0进程控制。CodeBrain Qual5成功继续真来源0模型复用，不重复成功工程；Qual4/5/6/7所有失败、未知端点和真实成本保留。新源独立审查及实际Mamba资格通过后才注册原30；本节覆盖下方“启动器兼容修复中”历史状态。

## 当前运行接续：QQ已退出，CodeBrain成功，Mamba启动器兼容修复中（2026-10-12）

- 用户明确已退出QQ，11:29真实GPU contexts=[]，原330/105FT315CPU仍完成，未开始原30正式修复拟合、未读outer概率。不要恢复旧Q5或重跑任何永久intent。
- CodeBrain qualifier4真实6F/4B/48窗/0step后Windows fsync读句柄失败，旧记录16.700772199998028秒保留。实测rb errno9、r+b成功且字节不变；新Qual5 CodeBrain已实际exit0/engineering_verified，报告deterministic_repair_qualification_v5/CodeBrain-lastblock-finetune/1791777157876013500/report.json（SHA89f0251f35539416ec7821ee959e502a70da51d91b24e3ca1e8748881bdb9052），成功6/4/48/0、forward/重复梯度/RNG/零step恢复均真实通过；不重测这份成功工程，后续按condition旧源验证后0模型复用。
- Mamba5在Torch/模型前因uid1000不能读root父进程exe而未获身份准入，0模型、无attempt/worker_owner；旧controller仍worker_starting/elapsed未知，不伪填终态。已核Win9860/26268、WSL5132、旧Linux422全部不存在、GPU空，按原birth/owner SHA归档并释放失效lease；永久intent和原字节留存。闭合deterministic_repair_qualification_v5/failure_closures/pre_model_linux_permission_1791778566044163800/closure.json（SHA98bd04eb5256ef3d895eb750cb1a7ef9c4ac3f69ccce97b5dd37414739a21e37）。
- Linux身份观察已新版本linux_worker_identity_v2.py（SHAabe21d44b8bc663a6506e1d996c67db45212b4a6d5f4c42e4d6a8bc3cd0aec57）：root仅隔离(-I -S -B)只读/proc、绑定源码与标准库，模型仍硬性uid1000。真实nested记录actual_unprivileged_root_observer_bridge_1791780223916.json证明唯一caller及单列精确/init只读relay；未知额外进程不排除。此前127源链独立审查independent_linux_identity_reuse_chain_source_review_1791780532160912600.json（SHAdc64317ceb61f985815a91f90a06c0e6e2636337c2286e287b4ca62e7c060dd7）通过，但未因此完成GPU资格或注册。
- Mamba6又在模型前因旧检查假定一个WSL launcher安全拒绝；controller真实worker_returned/int code1、lease已正常释放，仍0模型/forward/back/optimizer、无formal。实测actual_wsl_launcher_identity_diagnosis_1791781021708556900.json显示Windows Popen7560→35020同完整命令的WSL shim父子，Win/Linux两端相同；清理qualification6_zero_model_cleanup_1791781276379.json已确认无实验native/GPU上下文。新native_launcher_tree_v1.py按完整birth/parent/module/nonce/command只接Popen primary+最多一direct exact-command child，7项正负例df21d6...证据通过。
- 目前原recovery_entry_oct12任务正准备新版Qual7/D5/F8/Q10/V5/O5/W5/E8/CPU12/cost13等与该native tree；原127、115、104、88和所有旧版本/失败字节保持，不热改。新代码必须先独立审查，再实际Mamba新资格，按condition复用CodeBrain Qual5真实成功；两项实际审查齐备才注册新队列真实原30。不要运行Qual4/5/6旧永久intent或旧Q7/Q8/Q9准备。未生成新正式登记；有效下一入口以实际新版源码和审查落盘后更新为准。
- 用户已释放QQ，无需继续询问退出QQ。当前无应重验的CPU审计；自动接续仍ACTIVE/25分钟，但其旧具体Q7入口只属历史，先读本节和实际状态再行动。所有后处理资源/最终cost/wide/暂停/低功耗兼容与严格最终门尚待完成；保持原科学协议与单GPU，准备、失败闭合、工程成功、330都不代表计划结束。

## 当前有效入口：原330已完成，修复资格零模型延期（2026-10-12 10:28）

- 原Q5/source5/F3/恢复V3已实际正常结束：status=required_work_remaining，唯一LaBraM4s原15任务complete，当前子attempt真实int returncode=0，owner已释放。原登记330/360实际training_complete、30 blocked；七组FT全15，LaBraM4s33/f4实际56轮/best26，f5实际97轮/best67。不要重启恢复3、旧Q5入口或把330当计划结束。
- 最新CPU8 session26299已真实exit0：completed_finetune_checkpoint_audit_v8_1791771646056528000.json（SHAd7ac6533370ce8f6a9453e48d80b7a9360d5de4b8eaf3c77710e408e4af48e42），105FT/315CPU、零错误/暂缓，partial_verified、仍30pending；不是135/405。此前90279也exit0，103/309报告completed_finetune_checkpoint_audit_v8_1791770378697159200.json（SHAabb844bfb376d58ca971e23740f25d5178d46a7040ac4f6877bb73b0facaeee0）。当前没有应重复的审计；下次完成数未变不重验。
- 新104源修复链独立实际审查通过，可供Q7消费：independent_recovery3_repair_chain_sources_review_1791771797426373100.json（SHA9f4c37179767a8e8dfd51060968094b6cf12e9a0b542cdb0d2e33c5e47ea17b6），status=new_repair_chain_sources_reviewed、passed=true、完整精确104 reviewed_sources，全部88封存继承。独立30项定点/真实JSON复核与当前161项标准库证据recovery3_control_hardening_contract_tests_1791771585598558700.json（SHA62586d9ce66277816cd553f3484866e1c7065722178721e2bbc74c5ece66dc4d）均绑定真实字节；只通过受控入口和准备链，不代表双资格、原30训练、完整后处理或最终验收。
- 已修复旧103稿独立发现的四项阻塞，失败审查independent_recovery3_repair_chain_blocking_review_1791770460762218300.json（SHA197f9487bccde3bea9b48510c144987bf8edd7c856768489ca5db830c3b6a39c）及所有旧稿保持。当前资格v4要求真实source5/base、恢复3三键、原15 child正常exit0/outcome、330 immutable完成事件；空GPU之后才对原330完成产物作R._validate_complete全字节核验并带入admission，D2复核不可变JSON。共享linux_worker_identity_v1.py由controller独立捕获唯一Linux module/nonce/pid/ppid/startticks/parent/boot/argv/exe，资格每call、F5每fit、E5每fit复核；worker_owner改为open('x')原子排他。模块main guard已移到全部定义之后，标签F5/CPU9/E5/Q7实际新版本已纠正。
- 实际10:18边界只读证据nonrepair330_boundary_inventory_1791771498507.json：Q5 owner不存在，原32352/30824/3372/12192均缺失、没有实验进程，唯一GPU仍有QQ14024 C+G图形上下文。10:22只读再查仍有该上下文。已通过异步问题请用户手动退出QQ或关闭其硬件加速；不控制无关程序、不把此事当用户暂停，不暂停自动接续。
- 真实零模型延期actual_repair_qualification_graphics_deferral_1791772055699746000.json（SHAa86e47da60fd25789f0efad187a72521cabf446375578bd0cae1b98925815127）绑定本终态/child exit0/105CPU315/104源审查/实际GPU inventory；两个资格尚未启动，0 forward/back/optimizer、0模型GPU探测、0永久qualification intent、新Q7尚未注册。此时不要使用--run来制造一条失败资格记录，也不应再次通知同一未变占用；有用户答复或下次定时核查时重新核GPU全部contexts与排他进程，真实释放后才进入下一步。
- 下一真实入口：qualify_deterministic_repairs_v4分别CodeBrain与EEGMamba，两项均固定原fold1首train人8窗、6forward/4back/48windows/0step；严格空闲/永久intent/身份/输入与数值门保持。实际GPU成功后用review_actual_repair_engineering_v2.py核真实产物，绑定上述104源独立结论才可continuation_queue_v7 --register，再单GPU --run执行原30身份。所有旧入口/intent禁止盲重跑，任何资格失败须先保留真实失败、owner/子树、完整checkpoint与成本并独立核实。当前F5/Q7无任何实际训练登记，Mamba数值顺序改变及原CUDA反向逐位等价未建立必须披露。
- Q7实际登记/训练后有效混合CPU审计才变为audit_completed_finetune_checkpoints_v9.py --run（委托CPU8/L3），当前不启动CPU9或改88。新Q7原生全135/405零错误/暂缓/pending与原360实际完成后，owner释放才可evaluate_v5实际统一评价。之后12资源、真正finalizer/cost10/独立语义/wide全30、G3、所有暂停/loss3和系统低功耗最终附页及交付门的新兼容源码仍待补齐；新repair_postprocessing_gate_v2仅门准备，旧cost/wide/资源不可直接替代。
- 自动接续继续ACTIVE/25分钟，三暂停marker仍无，用户未再次暂停。原所有来源/登记/训练micro1逻辑8人/chunk8/FP32/预算/选模/阈值、三失去事件和未知尾/elapsed/active成本、I/O勘误及EEGMamba外部预训练参与者重叠排除未知均保留。全360与实际全135/405前持续不读outer概率、不评价；交付全部最终证据才停止自动接续。

## 当前有效入口更新（2026-10-12 09:58）

- 当前原Q5/source5/F3/恢复V3单GPU健康继续，原88份封存来源不变。真实完成328/360、FT103/135、LaBraM4s13/15；seed33/f1与f2均200轮完成，f3也已完成，f4正在第11轮/durable10（44updates/297back）。这些只作本次快照，下次读真实产物；未将当前fit算完成，不读取outer概率或评价。
- CPU8上一实际session61459已exit0，100FT/300CPU零错误，报告completed_finetune_checkpoint_audit_v8_1791768505305021500.json（SHA1c1661d9b3d97f4b2d25ec8d04a16ba364483c7f70e52c826d07c6ffbfc2c46a）。随后session16975亦exit0，101FT/303CPU零错误：completed_finetune_checkpoint_audit_v8_1791769788344362500.json（SHA796d1d3b913a33bbaa1e4fb9006db73ad4e99a929c9a15fa87bd7705889fc16b）。两者是partial_verified，非135/405。新增完成后CPU8 session90279已实际启动，尚无终态报告；下次先查它，不重复启动、不中断健康训练。
- 已准备恢复V3兼容的新资格v4、修复合同v2/F5/Q7、owner/worker/观察器v2、CPU9（原CPU8三加载数学与全部3失去事件）、cost10（原V5库存后cost9.augment仅一次再加真实修复账）、E5及只读后处理门v2。当前候选执行来源实际去重为103，继承全部88；原Q6/F4/CPU7/cost8/E4及原稿字节均保持。尚无任何新正式注册、GPU资格或30次修复拟合。
- 新纯标准库契约报告recovery3_repair_chain_contract_tests_1791770065475595700.json（SHA53157b23d24b3e1eaa17bcc394163e97fcbdac05066e5b3e6f5b166228912fdb）实际139项通过：两个资格数值函数AST与原版一致，严格原330全部真实完成/owner释放/恢复3三键及closure边界，拒绝旧恢复、partial/held/缺失及alias；observer5与CPU9/cost10/E5来源严格区分，损坏/重复来源不通过。初稿的closure字段及cost版本分支已纠正，初稿/首次检查失败均封存new_repair_source_preparation_history/recovery3_first_draft_*与recovery3_contract_initial_failure_*；未影响88或GPU训练。
- 103源及真实恢复兼容的独立语义审查已交回原loss3审查任务，尚待其真实结果，不能将本139项契约验证冒称独立审查完成。审核代码与报告范围要明确：实际GPU资格、30正式拟合、135/405、E5统一评价、12资源、真实finalizer/wide以及全部成本交付门仍待实际完成。新Q7可消费的独立报告须status=new_repair_chain_sources_reviewed、passed=true、reviewer_source及精确103 reviewed_sources；当前尚无可消费结论。
- 自然原330终态且owner释放之后才按qualify_deterministic_repairs_v4实际排他与全部GPU contexts空闲门运行两个资格；如有QQ/游戏图形上下文，真实延期并等待用户处理，不控制无关程序或放宽门。资格成功和独立实际报告审查后才能注册Q7真实原30身份；下一有效CPU版本届时才改为9，当前仍CPU8。当前恢复V3永久禁评价；后续原科学协议、单GPU、25分钟安静接续与全部最终证书要求仍按下节。

## 当前有效状态：已实际恢复单GPU训练（2026-10-12 09:17启动）

- 用户今日明确“继续”，本次已从完整磁盘checkpoint恢复，不是仅准备。`recover_q5_after_process_loss_v3.py`（SHA69ad989cefdb000304c89f9baf13529f096faa58223bce986e99a33e88612b10）实际运行原Q5/source5/F3核心；`continuation_queue_v5/recovery_registration_v3.json`（SHA653df2183b1a9cdcd74755f69629d2cf89ca3de8a6b7e9bf26b7df0bfcabbaf4）继承原84并新增4源，共88份封存来源。`recovery_intent_v3.json`、`process_loss_recovery_v3/launch_intent.json`及明确继续控制intent均永久排他，禁止重跑任何恢复/启动入口。
- 真实closure为`process_loss_closures/lost_f3_v3_1791767849813215500.json`（SHA83d4558818cd8727a4ec4b04d18ebfc18653aed6f3a03a611edb5a41f659e791），ready为`continuation_queue_v5/recoveries/process_loss_v3_1791767848030118400/recovery_ready.json`（SHAdaa0a15d14672bfa4949e94af5cb8380571a115268de525390b54418415b1eb1）。原Oct11四对象已缺失，原F3和父attempt的running字节、未知退出原因/时间/代码/elapsed/未刷尾都保留；不得伪造自然退出或数值终态。
- 实际单链快照为launcher32352→Q5 dispatcher30824→FT launcher3372→GPU worker12192，owner `13efdfd0f7db4984976740f99e82925f`。完整出生/父子/CIM及真实推进证据在`continuation_queue_v5/recovery_launches/process_loss_v3_1791767847808/native_running_observation_1791767966656702500.json`（SHA9c23ba66817461334545d61d7c3989bf3e607736049af4016b2a1abda7f2da34）；当前attempt为`continuation_queue_v5/attempts/1791767857117600800-LaBraM-native4s-lastblock-finetune-fit/`。这些PID仅此时有效，后续控制必须重核实际owner和完整出生身份。
- 最近只读进度为LaBraM4s33/f1第105轮、420updates/2835back、durable104，从原checkpoint54/216/1458真实推进；原进度55/220/1485中额外4updates/27back只是已返回调用的子集，不再加一次，新续训重放由新attempt另记。当前完成快照仍325/360、FT100/135、LaBraM4s10/15，不把当前训练计为完成；后续以真实产物刷新。过夜暂停不计训练，原inclusive计时保留，未知active耗时null。
- 88源独立审查`independent_q5_recovery_v3_source_review_1791767704319970500.json`（SHA1262fcf6ee237ff5ab26ca8c79c79d51138186442f4674766ffba0540d4392bc）及实际恢复事务审查`independent_actual_q5_recovery_v3_transaction_review_1791767989938396200.json`（SHA5a5782ff5c2f9ef5ca7a9e8f7640ed4cf7a46e94f3b0192ce515fa68f77763c6）均通过。后者核双次实际排他、源/manifest/closure/ready/intent、原owner及checkpoint边界；不是360训练或最终验收证书。
- 三个严格pause marker及Oct11暂停pointer已在本次明确继续授权后归档并清除：`process_loss_recovery_v3/explicit_continue_1791767837018777900/controls_released.json`（SHA91e21f91eb9e898cb004bcea7a3f9254907d7b25c8a17d6aef5f06842cae5c34）。没有对已失去进程执行NtResume。原暂停/快照原字节保留；再次暂停按当时真实身份执行，不能复用旧固定PID脚本或叠加挂起。
- 有效已完成FT审计改为`audit_completed_finetune_checkpoints_v8.py --run`（SHA43ab48da0bd862399d13aed718ad87500a718e795e9742663d78ba97c958252f），原CPU5三checkpoint核验数学不变，并覆盖全部三次F3失去进程。当前实际审计session61459仍在运行，启动时100份完成FT；下次先查其真实报告/进程，不重复启动。旧CPU6的99/297是历史已通过范围，不冒称当前100/300通过。`cost_aggregation_v9.py`（SHA4d0863abf0a4dbfe30e8e2bd853ca7d7bb8242212fce65683be36d9cd70811e0）覆盖三失去事件，但尚非实际最终cost。
- 本恢复V3永久禁评价，仅续原LaBraM4s任务；原非修复330实际完成并释放owner之后，CodeBrain15与EEGMamba15仍必须真实修复并训练。旧Q6/F4/CPU7/cost8/E4/资格v3/后处理门是恢复V2的准备，不能直接注册或运行。下一套新兼容源须继承全部88、绑定L3/CPU8/cost9及实际修复分支，补齐运行owner/worker/观察器、完整135/405门、评价和后处理，再独立审查；实际源数按去重登记，不能机械改99为103。原科学身份、分折、训练micro1/逻辑8人/chunk8/FP32、预算/选模/阈值保持。
- 当前GPU admission仅允许已核QQ图形上下文与必做训练；未运行性能资格、不控制QQ。扩大batch及CPU–GPU同步优化仍未应用。全360及相应实际135FT/405CPU零错误/暂缓/待办之前不读outer概率，不做评价。其后还须实际统一评价、12条零optimizer资源补测、最终cost/独立语义/全30wide及同一产物的G3、所有暂停和系统低功耗附页/交付门；旧准备或pending不能替代完成。最终必须额外纳入Oct11夜暂停和本次loss3，保留全部I/O勘误与EEGMamba参与者预训练重叠排除未知局限。
- `eeg`自动化已实际恢复ACTIVE，25分钟间隔已读回确认；健康队列不重复启动。状态无实质变化保持安静，全部真实交付完成后才停自动接续。以下是保留的历史快照，尤其09:07“尚未启动”和昨夜“暂停”不覆盖本节。

## 当前状态：已获继续授权，核验新磁盘恢复（2026-10-12 09:07）

- 用户今日明确“继续”，撤销昨夜暂停要求。新鲜CIM与GetProcess核实Oct11暂停的9840/34592/23748/10720四对象已全部消失；原因、时点、退出码及未刷尾未知，不执行NtResume、不复用旧永久恢复intent。
- 当前325/360、FT100/135；LaBraM4s33/f1的initial/latest已仅CPU加载两次并完整验证模型/优化器/RNG/采样和原政策。真实CPU报告 recovery_cpu_preflight_labram4s33f1_1791767054086345800.json（2b575b700a086b5a14e668b0f8af563dae81173a5f1f7f0215d9ba7657d6f32e）；durable54/216updates/1458back，对应原进度55/220/1485，额外已返回4updates/27back仅作为原返回计数子集披露，不再加一次；旧inclusive时长和active未知仍保留，过夜暂停不算训练。
- 真实loss证据 process_loss_recovery_v3/loss_evidence_1791767102232649100.json（82b713c44721b0966ae0d6846925d7a5ac60c6b2009115b071ec0c6eba316693）及独立CPU/JSON审查 independent_actual_labram4s_loss_cpu_evidence_review_1791767150384408900.json（224685c60ed6b712da38e9a5e4095d4f423e55c3cee6c8dacaf7e865bec2633a）均已保存。原pause/owner/attempt/checkpoint字节未修改。
- 新recover_q5_after_process_loss_v3/closure3/CPU8/cost9正在准备与独立审查，继承原84并新增四源；尚未登记或启动，三暂停标记暂留到实际受审启动事务，自动化暂PAUSED。该次恢复仅续原LaBraM4s任务至原非修复330；CodeBrain15/Mamba15的实际修复仍必做，禁止以330结束或提前outer评价。
- 本节覆盖下方历史暂停状态；新CPU审计入口、实际恢复身份及最终兼容链以真实启动后的顶部更新为准。仍保留25分钟安静自动接续要求。




## 当前状态：用户要求暂停（2026-10-11 22:41）

- 最新人类要求“暂停，我要明天再继续”。训练及自动接续均已实际暂停，明确“继续”前不能清标记、恢复进程、启动资格/审计或新队列。三处sync_boundary、sync_boundary_v2、sync_boundary_v3/user_pause.json严格paused=true，eeg自动化实际PAUSED，原25分钟配置保留。
- 本次仅对实时核实的四个原Q5对象各挂起一次：9840/birth134361805664662981→34592/birth134361805665493947→23748/birth134361968365520371→10720/birth134361968365627510；全部线程已独立复核Suspended，没有kill/第二次挂起。当前owner09db383ec1c14168b435dbd4547077a0及原84字节保留；PID/出生只有此时有效，恢复前必须重新核真实身份。
- 新暂停源pause_current_q5_20261011_v1.ps1（SHAb96b621f8098bff13ee1745e9e75d2a241b7ce8885f93bbcb6459bf66b960002）；原Oct10/Oct9控制脚本身份失效，禁止复用。真实证据pause_events/q5_user_pause_1791729696123/native_pause_complete.json（SHA65977f9551a7621f9df0e1701173bf1845794a6c60b7a0d58012237092a98c40）、captured_identities.json、各native started/returned事件与两份JSON快照，及user_pause_q5_owner_20261011.json均已保存。resume_review待初始化代理已中断，不自动继续。
- 实际完成325/360、FT100/135，LaBraM4s10/15。当前暂停LaBraM4s33/f1：progress报告epoch55/220updates/1485back，last_durable_epoch54；内存保持，initial/latest文件存在。仅保存JSON及文件大小，没有新增Tensor审计/模型/检查点序列化；未刷尾/未返回调用精确量未知。不得把正在训练的33/f1算完成。
- session91501在暂停前已自行exit0；真实CPU6报告completed_finetune_checkpoint_audit_v6_1791728930489996600.json（SHA3ac434b911ea94726d6eefe58e9bff42a6c29719fab810d9fb285f47aa915ff2）核99FT/297CPU、零错误，不能称100/300已核；当前没有活跃CPU审计。暂停后不启动第100份审计或继续准备工程。
- 明确继续后先读本次native证据/owner/完整出生与线程状态，保持未重复的挂起计数；原进程仍在才能受审恢复一次。进程已失去则保留未知时点/退出/尾，核完整checkpoint后走新的真实恢复版本，禁止复用永久intent。恢复后新增完成才用CPU6；尚未实际注册Q6，CodeBrain15/Mamba15仍blocked，新修复/后处理99源及门准备均不代表完成。
- 此次原进程暂停也须纳入最终成本：LaBraM4s当前fit/F3父计时/Q5父计时的暂停重叠与未知active时长，及新控制源/证据必须由新兼容暂停附页与交付门绑定。旧三计时附页不能漏本次暂停；原inclusive计时保留，过夜暂停不算训练，不用wall/QPC混减、不伪造精确active秒数、不将同一过程的已返回调用再加一次。

以下为保留的历史执行入口，不能覆盖本次明确暂停。


## 当前有效入口（2026-10-11 22:27）

- 当前真实324/360、FT99/135，LaBraM4s9/15；同一Q5/source5/F3/恢复V2单GPU链继续，seed22/f4已40轮/best9完成、f5继续。四对象完整出生信息已动态核，未重启训练、运行资格或读取outer概率。
- CPU6新终态completed_finetune_checkpoint_audit_v6_1791728605427211300.json（SHA03e89349bf77763f61417684758e93dfbfe309206676cf7a137318e42f266cdd）实际98FT/294CPU零错误/暂缓，session83235已结束。随后新增第99份完成，新的CPU6审计session91501已启动，尚无返回证明；38200/birth134362023592559936→34824/birth134362023592686626仅此时审计身份。下次先查此session/真实报告与进程，不能重复启动或冒称99/297已通过。
- 新repair_postprocessing_gate_v1.py（SHA93df313551fa00a1a0c1444ce920b7ad83fdfe5ccf39d45aad6079735dbbd22b）仅只读后处理门准备：全360与释放owner之前只读状态；之后严格核Q6原生CPU7全135/405/零pending及135唯一深审计行，再核两个实际E4子阶段exit0、canonical运行目录/同一已释放父owner出生链和原435/23/345报告。CPU/训练硬门前不得调用现有outer报告核验。不是GPU资源入口、finalizer或最终验收。
- 20项标准库模拟护栏repair_postprocessing_gate_preparation_1791728566919973000.json（SHA5ab6bd608daffb32f58ce6f93fa71d3b32ad11c4896d072ce93d28bc527e6e9e）已过；--check-only真实pending且零模型/GPU/概率/文件写入。旧初稿字节在new_repair_source_preparation_history/保留。没有实际CPU7/E4/资源/最终cost/wide/交付门操作，完整后处理链与独立审查仍待补齐。
- 正式修复的当前Q6源836a8044与99源核查0a2840ae保持22:17节绑定；CodeBrain15、EEGMamba15仍blocked。原Q5自然终态/owner释放与实时排他之后才按资格v3及永久intent运行。后续原30真实修复、360/全135405、E4、12资源、cost8/语义/全30及G3/暂停/低功耗最终门必须实际完成，不能以pending或准备结束计划。

以下各节是保留的历史快照，当前入口以本节和实际产物为准。


## 当前有效入口（2026-10-11 22:17）

- 真实登记322/360、FT97/135；六组FT全15，LaBraM4s已7/15（seed22/f2已112轮/448updates，f3继续）。同Q5/source5/F3/恢复V2四进程链34592→23748→10720与launcher9840已动态核完整出生信息；数字与PID仅本次快照，下次重新核。健康单GPU训练未暂停、重启或追加模型GPU探测。
- 新CPU6已实际exit0：completed_finetune_checkpoint_audit_v6_1791728146844845000.json（SHAa2a789e068d4bc5efc8e0d761dcb9fff80a3ab69f8f8bbe41528fc737fe8fa7e），97FT/291次CPU加载、零错误/暂缓，session3680已结束。旧95/285及93631为历史；计数不变不重复。仍不是135/405或360证明，没有outer概率/测试评价。
- 尚未注册的Q6草稿已修正：836a80443b8e007d212833f2ec5235da50d267c100e7a1ef74f144196052fe8e。原405d字节封存new_repair_source_preparation_history/1791727969078540300/。21:50关于“CPU7前已复核完整来源”的准备描述有误：旧字符串替换没有落入实际cpu_gate；本次已实际加逐份S.checked并用损坏来源负例证明在CPU进程之前拒绝。旧源码/报告保留，未影响任何正式执行或当前84封存源。
- 当前99份候选执行源的自动化源码核查已通过：independent_complete_repair_chain_source_review_1791727969605779800.json（SHA0a2840ae26e2ab6fd34c1a82db3a3821c5b7ec6f592669201fe7ac2b01b3eded）。范围是来源、语法、控制和原代码委托检查，不是实际GPU资格、独立最终语义验收或正式注册。原Q5自然终态与owner释放前不得运行资格或Q6。
- 新实测工程证据审查入口review_actual_repair_engineering_v1.py（SHA43d1681073177af78527501b33c8fb666b27e412898b2e54a9c3c41dc4325054）已通过14项标准库/模拟护栏，证据actual_repair_engineering_review_preparation_1791728066356390200.json（SHA44604d3a1c978364e0d4fa61c98995ad9ff2632b87ff880142196e51fc912689）。能拒绝少返回反向、真实step、前向/重复梯度不一致、数值顺序错误、外部GPU占用、owner字节变更和测试角色；尚未审查任何真实GPU报告。
- CodeBrain15与EEGMamba15仍blocked。待原非修复330实际完成、Q5终态/owner释放后，按qualify_deterministic_repairs_v3.py实时身份/空闲/永久intent硬门执行两项资格；实际报告独立审查、当前99源审查与条件lock齐备后才注册Q6并真实跑原30身份。未建立原CUDA反向逐位等价；Mamba累加顺序改变、参与者预训练重叠排除未知均须披露。
- 后续实际顺序需新兼容源及受审绑定：360+Q6原生CPU7全135/405→E4实际统一评价并释放owner→12条零step资源补测→cost8真实finalizer/报告与独立语义审查→严格wide全30→同cost/wide的G3、原进程暂停、系统低功耗终态附页及交付门。资源/finalizer/wide/附页兼容与最终独立审查尚待补齐；不能用旧cost7/wide6、源码准备或pending当最终证书。

以下入口、PID、进度及报告为原字节保留的历史记录，以本节及当时实际产物为准。



## 当前入口与待办（2026-10-11 21:50 更新）



- 最新有效阶段：六组FT各15完成（新增LaBraM5s全15），20:53:46原Q5父任务真实complete/exit0后接续LaBraM4s。当前快照320/360、FT95/135（4s seed11全5已完成）；最新CPU6已exit0：95FT/285CPU零错误95aa4a，session93631已结束。旧48417/50980等均历史；计数未变不重复有效审计。CodeBrain15与EEGMamba15仍blocked；有效待资格入口qualify_deterministic_repairs_v3.py（880a51），无实际GPU资格。F4、合同核验、Q6/owned-worker/native-owner以及CPU7/cost8均只准备；E4也已source-only准备；当前99份候选执行源未形成实际资格/完整全链注册与最终门，无任何新正式登记。当前Q5/LaBraM4s单GPU健康继续，原84不改；原Q5自然终态/owner释放且实时排他后才实际两项v3资格。

- 原 Q5/LUNA 四个训练对象已实际消失；原因、退出时间、退出码及未刷新的调用尾部均未知，旧 running attempt 与全部原字节保留。本次不是自然 handoff，也不是重跑旧恢复入口。

- 新一次性 `recover_q5_after_process_loss_v2.py` 已于 16:22:46 实际启动；Q5/source5 与 F3/E3 原核心不变。有效恢复登记为 `continuation_queue_v5/recovery_registration_v2.json`（SHA525f6ed6e838fe267956c54d90d4e6d13d7da3bc710863aca90c35af466782cd），84 份来源继承原80/原47。永久 `recovery_intent_v2.json` 已存在，禁止再次运行该入口。

- 新失去进程证明 `process_loss_closures/lost_f3_v2_1791706969935734100.json`（SHA14286a281be2d896653fb717ea2faefb86b84b3d80e572e36795154b746bfc8e）绑定原 LUNA33/f3 的完整 CPU 恢复证明、精确快照、两次新进程排他核查及原 owner/父任务档案；此前 CSBrain closure、恢复登记、原 running 日志不改。

- 当前四对象快照：恢复 launcher9840→dispatcher34592→FT launcher3752→实际训练14296，出生身份在 `continuation_queue_v5/recovery_launches/process_loss_v2_1791706966406/native_running_observation_1791707007802.json`。下次必须重新核当前 owner/全部实际出生信息，不沿用这些 PID。Q5 owner/status 真实 source5/base registration/version5 与 recovery2/previous lineage 均已出现；子任务仍为 LUNA-lastblock-finetune-fit。

- LUNA33/f3 从完整第38轮（152updates/988back）恢复，当前开始第39轮。原已返回156updates/1014back中的额外至少4updates/26back单列，已包含在原返回计数中，不再加一次；新真实续训尝试自行记录重放工作。精确旧耗时/未返回尾仍null。当前完成快照282/360、FT57/135；未将当前training算完成。

- 随后真实进展：33/f3已40轮/best6/validation_patience完成；33/f4已40轮/best9/validation_patience完成。两者选中headwarm合法保留，终态末层训练分别记录。实际登记现284/360、FT59/135，LUNA14/15；33/f5已推进第17轮（68updates/459back、durable16），没有重启第二队列或改预算。

- 恢复 CPU 实证 `recovery_cpu_preflight_luna33f3_1791705459339343500.json`（SHA80b8a90beb18188eb1358ebf8306ed720d17fb6e3c7acc239c263cab3542722f）已实际仅CPU加载initial/latest两次，完整优化器、随机状态、历史及嵌入best核验通过，不重复运行。新入口独立25项复核与 runtime 独立69+18项复跑证据在分析目录 `independent_q5_recovery_v2_entry_review_1791706057973838000.json`、`independent_recovery2_runtime_source_review_1791706938811.json`（SHA52f93f333f6af91d6f022a118a255334fb39b9138bc08fb530a92df4f753bb66）。这些是恢复/准备范围，不是最终计划证书。

- 此后真实完成FT只使用 `audit_completed_finetune_checkpoints_v6.py`；计数未变不重复有效审计。恢复队列在测试前强制运行实际CPU6的135FT/405CPU、0errors/0deferred/0pending门，原360及全CPU门实际完成前不读外层测试概率、不执行测试评价。

- 首个实际CPU6部分审计已正常exit0：`completed_finetune_checkpoint_audit_v6_1791707482608571800.json`（SHA4cff4899ad9a7eeac17f6235cb8f48d3cbe3aa876ad66be948998788d0df1594）实际59份完成FT/177次CPU加载，零错误、零暂缓，status=partial_verified；没有模型构建/GPU/raw/测试概率或测试门操作。这不是135/405或全360证明。exec session27467已结束，旧11792/22232仅审计历史身份；完成计数未变不重复本有效审计，全135/405仍须最终队列原生门。

- 新 `cost_aggregation_v7.py` 保留两次独立失去进程事件和未知父任务elapsed；源码已绑定84份来源。后处理 `supplemental_head_resource_measurement_v6.py`、`finalize_full_plan_cost_v6.py` 已在磁盘准备，尚须完整来源绑定及独立复核，不能仅凭文件存在或旧版本通过直接执行。新的wide/交付兼容门也须受审；旧cost6/wide6/资源5/所有旧门与产物保持原字节。

- 最终仍依次实际完成：360及CPU6门→Q5/E3统一评价且owner释放→12条零optimizer资源补测→真实最终cost7与语义审查→全部30项wide验收→同cost/wide的G3最终bundle→原进程暂停成本终态附页及交付门→系统低功耗完整运行范围事件审查、终态附页与交付门。旧LUNA父任务因进程失去永久保持running/elapsed未知，不能伪填终态；系统低功耗附页需新兼容版本接受此真实失去进程范围，不能用旧helper待父任务数字终态或放宽事件全范围门。

- 三处user_pause标记当前均不存在，用户明确继续；若再暂停先写三处严格paused=true、停止任何自动新probe/恢复，以当前真实进程身份暂停并停自动接续。扩大batch与同步优化仍未应用，不追加模型GPU探测、不重启已结束守护或重复工程。保持单实验GPU与25分钟安静检查；实际全部交付后才停止自动接续。下文为保留的历史记录，旧PID/进度/入口不覆盖本节。



### 2026-10-11 21:50 实际CPU95与E4准备



- completed_finetune_checkpoint_audit_v6_1791726506885653100.json（SHA95aa4adc680efcd9632be689f5509cc00a4e1e57a95a7ae2b05eaf3d0d629a5f）已实际exit0，95FT/285CPU、零错误/暂缓；session93631已结束，不重复本有效范围。包含LaBraM4s seed11全5。当前真实登记320/360、FT95/135；数字非最终360/135405证书，队列仍Q5/source5/F3与原84，未做outer概率/评价。

- 新evaluate_v4.py（SHA69a954240a6d2c0b9eb5c6765affced06110aff6a8e20d8c172e123386c1f387）只source准备。实际--run第一步须Q5/Q6 owner都释放、原360真实complete、Q6原生CPU7 gate的135/405/0errors/0deferred/0pending/完整源哈希通过；然后新evaluation_v4_control排他native父子/birth/token，原WSL/Linux Mamba15 staged先返回exit0，再Windows all23/435与OOF汇总。默认0Tensor/概率/模型/GPU/query/owner写入，现在没有任何实际评价。

- E4私有E2/原E1字节码复用，原pMDD>.5阈值、指标、435快照、选模和selected/terminal披露不改。actual4训练用新repair verifier，旧3用原X；推理复用原native forward，私有backward修复不调用，不声称原CUDA反向逐位等价。每实际评价fit只读核owner/token完整born tree及严格pause；未知孩子保留owner，无重复GPU。

- 12标准库/模拟护栏owned_evaluation_v4_preparation_1791725830570824900.json（SHAba2e116e7faf1779a142ee34359c8fc490b2c25aea9a819639ede46b0011dce4）通过；独立源码/default/原84与指标CodeType相等复核independent_owned_evaluation_v4_review_1791726002370216800.json（SHA68156b73c17a843d13336c387d653e1dcd5a23526549fc0b5b7cde2f59442ab7）仅准备，未读outer预测/概率或构建模型，full_new_source_chain_review_passed=false。

- 尚未登记的Q6草稿当前源405d38a8f9071c31a4f177b4762bc1bde8e4d5afa4158a8c22e64d84a117569d，新增完整源列表显式含repair_observation_v1.py，并在实际CPU7 gate前复核整套源；旧4ebd字节与修订说明封存new_repair_source_preparation_history/1791726258258318600/。old d370/4ebd准备报告保留，不冒称当前版本正式通过。

- Q6.sources()只读构造当前99份候选来源（继承84）；这不是注册、真实GPU资格或完整新source review通过。后续必须在原Q5/4s全15实际自然终态及owner释放后，核实时单GPU排他，再实际qualify_deterministic_repairs_v3两项工程及独立实测报告审查；随后完整99源独立review/真实条件lock/Q6注册/正式30运行。当前训练不暂停、不改84、没有新增模型GPU探测。

- 最终后处理仍须为新Q6/E4/CPU7/cost8补齐真实资源、finalizer/wide全30与G3/暂停/低功耗门兼容和独立review，不能拿旧cost7/wide或E4源准备代替最终完成。EEGMamba外部预训练参与者重叠排除未知仍须披露。全360与实际全135405前继续不读outer概率。



### 2026-10-11 21:24 CPU7/observer4与成本8准备



- 最新真实CPU6终态：completed_finetune_checkpoint_audit_v6_1791725033559707800.json（SHA7b7b9f9447dc258696a2b8ab51b60dd412aefff9c01640d4cc2aaf0085503ca1）实际91FT/273CPU、零错误/暂缓，exit0；session48417已结束，覆盖4s11/f1。后面的启动段只是历史过程；此报告仍partial_verified，不是135/405或全360证书。



- 当前Q5/LaBraM4s仍同一健康链；11/f1已按原规则200轮/800updates/best200完成，11/f2已80轮/best73继续。快照316/360、FT91/135；未运行新修复/GPU工程或outer测试。新增CPU6审计session48417于21:20真实启动，报告未返回前不冒称91/273；下次先查报告与当时真实进程，不重复启动。

- 新repair_observation_v1.py（SHA5acb6cd2890f8ee8de858731642b1afa1c43f5932ede423089567d53d81a3aed）先核真实observer4的完整资格/来源/参数/记录，再用原X.verify_observer_record的counter_semantics之后完整AST护栏（不含旧actual3来源接受段）。原所有serial/alias/flush/unknown-tail/durable-save约束保留；输出真实4来源，临时F2副本只供旧物理计数算术，不改原状态/退出/秒数/计数，不把4称3。

- 新audit_completed_finetune_checkpoints_v7.py（SHA0ce9a1d9eb7e97c4b0786a553b706bc0e2650bfde1cfebc787da796efa0477e3）私有CPU6→CPU5原审计空间，只替换真正4支路；原2/3、两次失去进程和每完成fit三个CPU读取原数学不变。默认不运行，真实未来使用--run；现在仍只运行有效CPU6，尚无actual4完成可供核。14标准库/模拟护栏repair_observation_CPU7_preparation_1791724269548366900.json（SHA80a20121617cd01b071add40bdd1e1fa2160a43470af28d6a1a050aecd48585d）通过；独立source/default/旧84核查independent_repair_observation_CPU7_review_1791724479187982000.json（SHAa2a7b77fff57d2b3685e0456d6fce561ed6aa2650074160c28be84c78b6c58e7）仅准备，0Tensor/GPU/真实CPU7读取。

- Q6尚未实际注册的源码修订为4ebd650305181fc19a7865c84de1a32b0c373f05783e1c0e6154f49c259ae207，仅给CPU7实际gate命令补--run；旧d370788字节与修订说明原样归档new_repair_source_preparation_history/1791724196293545200/。旧Q6准备报告/审查留存但不是修订后全链正式通过证明；最终独立review仍必须绑当前整套源。

- 新cost_aggregation_v8.py（SHA64030d81f55e16179dec258aadfa3c0bf7873a1882ed559478d2318a3fa2d2bf）准备在私有旧V5库存中核实际4，然后旧Cost7/L2损失仅增广一次；新增qualifier-v3 attempt为一次零step工程主账，controller/worker/Q6父计时仅nested supplemental。旧通配已收Q6 status时核同SHA并复用，不增第二条；unknown elapsed/tail/exact总成本保持null，任何失败保留，默认不collect。

- 成本9项纯模拟护栏repair_cost8_preparation_1791724832922965900.json（SHA48bedd645e59c340f4aea7c0ca15dc6230a9e6d1aab8ec042ffe744adbc383c9）通过；独立源码/default/旧84复核independent_repair_cost8_review_1791724951312442500.json（SHA8a758310de8b911bae254173d1a997ab464ac087548922cbb6d78b6b41f86d86）仅准备，不是实际最终cost或全链ready。

- 下一步必须补实际排他E4及真实GPU资格/独立报告review、当前全部新源完整review再注册Q6；随后新finalizer/资源/wide及G3/暂停/低功耗严格门兼容，不能用旧cost7/wide准备冒充新最终产物。暂不运行任何新register/模型/收集，原84源与原队列保持原字节。所有全360/135405/12补测/30项及附页最终门仍是实际完成条件。



### 2026-10-11 20:59 LaBraM5s全15完成与Q6控制准备



- 最新实际CPU终态：completed_finetune_checkpoint_audit_v6_1791723557433273500.json（SHA38863e4f6321aa935e506f2e0b8725b7a8fb431543ce87b8ff4efb351a9e79ff）已exit0，90FT/270CPU、零错误/暂缓，包含LaBraM5s全15。session50980已结束，不重复本有效范围。下面启动/身份段仅保留历史过程；此最新报告仍partial_verified，不是135/405或全360证书。



- LaBraM-native5s-lastblock-finetune的原15身份已全training_complete；原Q5父attempt 1791713091744223600-LaBraM-native5s-lastblock-finetune-fit真实status=complete/returncode0、finished20:53:46.134601，父inclusive10134.382205500002秒不与F3/metadata嵌套加。33/f2停151轮、best121；f3停40/best1（选中headwarm合法），f4停55/best25；f5停106/best76。不是全九组或计划完成。

- 当时实际总315/360、FT90/135；Q5/source5/恢复V2与previous链仍原字节。同dispatcher34592/parent9840已自然进入LaBraM-native4s-lastblock-finetune-fit；父attempt 1791723236541078300、launcher23748/birth134361968365520371、实际GPU10720/birth134361968365627510为20:56快照，下次动态核。旧5s7744/21308已不在新native inventory，不复用身份，不重启队列。

- CPU6 completed_finetune_checkpoint_audit_v6_1791722482427138500.json（SHAa5c9b50b752986faeef4a0a93c4a82d226c1ff1416f3a67d2b8ed19a0bf55511）已exit0、88FT/264CPU、零错误，session22288已结束；它覆盖5s13而非全15，不能伪填90。新增完整15范围的CPU6审计session50980在20:56实际启动；native17280/32644仅当时CPU身份，先查真正报告/进程，未结束不另启。未读取outer概率/测试评价。

- 新continuation_queue_v6.py（SHAd370788492c33b5b5448a8f6c5f5ea9b7d4092e58a50e33829254f46af8f5df2）、repair_runtime_owner_v1.py（SHA9b71f685d0414c984059b8fc8f0097f6c227f5ac5a652ca2995a76974b2460aa）、repair_owned_worker_v1.py（SHA9073375ba3ca549eaa880062be62169e43099c77403de0ba308ce71ec01536a4）只准备。默认零Native/GPU/Tensor/模型/登记写入；实际注册须原非修复330完整核验、两项实际GPU资格及独立实际报告审查、完整新源码review、永久注册intent，无资格不能register。

- Q6正式worker方案原15两组顺序，父Popen后捕获实际native出生/父子/完整command，再发布子token与identity；Windows实际Python或原WSL/Linux出生均约束，首个GPU查询/模型前等待真实身份（20秒有界），每fit只读核live owner，未知其他实验进程/PID复用/旧owner拒绝。原F4的直接CLI不是生产受控入口；正式必须经绑定owned-worker。任何未知孩子/失败启动未核清保留owner，不盲重启或终止无关程序。

- Q6只负责真实修复30与实际CPU7全135/405，结束仍evaluation_pending/all_plan_complete=false；实际E4须独立排他且Q6owner释放后执行，所有资源/成本/最终门继续硬门。CPU7/E4/cost8尚未写完，Q6不能注册/启动，不把准备当已兼容；当前原Q5/4s训练没有暂停/终止或额外GPU工程。

- 16纯标准库/模拟护栏repair_queue_ownership_preparation_1791723147789387600.json（SHA5c6ac7fba28c0262f2b188bd1d794b5158d85bd307f1c8d24eb4534d08321bf7）通过。独立AST/default/原84字节复核independent_repair_queue_ownership_review_1791723312132168100.json（SHAbc1c60493630231b7dd07a13d23d6bcf8ae94593cb79229b50095c36b32197b0）范围仅控制准备，full_new_source_chain_review_passed=false、无真实native/WSL/GPU运行，不是全链ready或最终验收。下一步CPU7混合observer4严格验收、E4、cost8及全30/附页新版绑定优先。



### 2026-10-11 20:32 新增CPU证明与正式修复链准备



- completed_finetune_checkpoint_audit_v6_1791721755485936700.json（SHA111e87010066ecd60ae6e53b24c1032a88456e73c308b13a16bb041442f83bed）实际86FT/258CPU、零错误/暂缓，exit0；session51098已结束。当前登记快照311/360、FT86/135；LaBraM5s33/f1已原协议延长后200轮/800updates完成，f2已120轮/best120继续。完成计数未变不重复本有效审计，不提前测试。

- 有效工程待执行入口qualify_deterministic_repairs_v3.py（SHA880a51f30b229674e7b3ed6d810669d80b114a48cacb91678adf2a60c98ae01b）新增只在真实6forward/4back/48窗/0step、重复梯度/完整恢复全部通过后生成runtime_contract：精确更新/冻结参数、buffers不变、官方权重/作者源和原输入合同。该v3的15模拟护栏报告0c5d8ec3、独立源码/default/84字节复核31085b99已通过；对应文件deterministic_repair_qualification_v3_preparation_1791721053042920300.json、independent_deterministic_repair_entry_v3_review_1791721053309987300.json。v1/v2旧准备源与报告保留，不实际运行旧版本。

- 新finetune_v4.py（SHA8928371053e96511d7f4823221d3b6e670b1d61bdcc81c0108e3f1364455204d）准备委托原F1训练/选模/恢复、F2选中/终态披露与F3计数/flush，只在资格和完整新注册通过后路由私有适配器；首个磁盘计数即实际observer4，metadata另列actual_finetune_implementation/actual_deterministic_repair_execution。原构造函数在旧runtime包装前捕获，避免递归；异常恢复构造/save上下文。原源码不改，不伪填F3/Q5完成。

- 新deterministic_repair_execution_v1.py（SHAaa2dce2cdb42a81a6025245a2db8b062e02d5c10d1d1deeed21cce9ce8968618）只核真实qualifier/controller/worker返回、已释放owner、永久intent、原首train/15身份、精确6/4/48/0计数、参数分区、官方源、数值顺序披露与独立GPU报告审查。任何新登记须完整F4/Q6/E4/CPU7/cost8源及真实工程通过；当前缺这些完整链与报告，不能登记/启动。字节hash可能读取权重或工程resume_probe.pt，Tensor反序列化为0，不声称绝对JSON-only。

- 14项标准库/假模型护栏deterministic_repair_formal_preparation_1791721547417019500.json（SHAaaa0447ce133bfdc38c962a7ec9fdd022ebd6864364f3df0489c81e148640d6b）通过；独立源码/实际默认CLI/原84源复核independent_deterministic_repair_formal_sources_review_1791721796053707800.json（SHA9783efbbed07bd0a246b62b9e73ae1f5c24aefde374a82858aad05a0eb061425）。这是未完整链的准备审查，未运行作者模型/Tensor/GPU/正式fit或新注册。F4实际启动前还必须补齐Q6 dispatcher/训练孩子的真实出生/排他绑定，不能直接以草稿CLI竞争GPU。

- 下一步优先完成新Q6与CPU7/E4/cost8及剩余严格门兼容，保留全部历史文件；当前健康Q5/LaBraM继续，不重跑资格、原工程、恢复或守护。待原Q5真实自然终态且owner释放才实际v3工程资格；单GPU、源/角色/预算/阈值不变。全360+真实全135/405前仍无outer概率/测试评价。



### 2026-10-11 20:09 实际CPU核验与独立修复入口准备



- completed_finetune_checkpoint_audit_v6_1791720008259902600.json（SHAa902eacc3c82b15312d72a26fbcee89b15a79a18be0b61c284d7e9ff3f2e9824）exit0、85FT/255实际CPU读取、零错误/暂缓，仍partial_verified；session74119已结束。20:09登记快照310/360、FT85/135；LaBraM5s seed33/f1已第80轮/best80继续。实际owner仍同一Q5/source5/恢复V2链；当前四对象完整出生身份于19:xx动态核过，后续重新读取，不复用固定PID。

- 新qualify_deterministic_repairs_v2.py（SHAf6af1a87e056cf58fc949786c4cb888ce552e74619363bda2deeed4a4e969e1b）只用于两项必做确定性故障的独立工程资格。默认只给零操作说明，无模型/raw/Tensor检查点/GPU查询/输出写入；实际--run严格要求原Q5真实required_work_remaining等终态且owner释放、原15blocked身份、84旧源/原矩阵、无其他实验进程与全GPU上下文空闲、三处严格pause、排他owner及每condition永久intent。不控制QQ/游戏或恢复/挂起任何训练。CodeBrain Windows与Mamba原TRI-DEP-Ubuntu-24.04/官方环境分开执行。

- 私有克隆复用原F1 build_model同字节码，仅路由新adapter；计划原fold1首位train/8窗/chunk8/FP32，原native/private前向零梯度两次比对，原warm1/lastblock11各两次相同RNG反向，总6forward/4back/48窗/0真实step；检查重复逐位梯度、冻结/全状态不变及零步完整恢复。实际来源、调用始末、失败未知尾、worker初始化至checkpoint主耗时与controller嵌套时间分列；Mamba累加顺序改变、原CUDA反向逐位等价未建立、无提速声明必须保留。该计划计数不是实际GPU调用。

- 15项纯标准库/模拟护栏deterministic_repair_qualification_v2_preparation_1791720392255564200.json（SHAdc2a69f369ef5ad1e6a7bb3162796edad44890301d1ac9f97814c8ed45ca4b1d）通过。独立AST/真实默认CLI/旧84源逐字节复核由review_deterministic_repair_entry_v2.py生成independent_deterministic_repair_entry_v2_review_1791720498906502500.json（SHA8785483b2b7f3ceb5977d7a4df445fb8d1c01e0ab24fdb6f4b3b03feb3cc83f5）。范围仅entry准备：实际Windows/WSL孩子、Linux AMP/autograd和完整GPU模型均未运行，不是两故障修复成功或30FT完成证书。首版qualification_v1及其3f3715准备报告原字节保留；它因初始化成本范围与worker绑定改进被新v2替代，禁止实际执行v1。

- 当前没有资格intent/owner/GPU worker，没有改原84源/状态/预算或重复旧失败工程。真正工程通过后仍须新F4/Q6等正式来源、CPU/evaluator/cost/全部30项及成本附页绑定，不能将新完成伪填到旧Q5。当前健康LaBraM训练继续，测试门未开放。后续资格要先再次核实时排他、来源与暂停；若用户暂停，旧三strict标记及当前真实worker_owner/父子出生信息优先撤销/暂停，不盲重启永久intent。



### 2026-10-11 19:32 最新实际部分CPU证明



- completed_finetune_checkpoint_audit_v6_1791718325787954400.json（SHA90de29d21eff6a68e841fcd40709b9523e27e77f89d47eecc342f0959d977839）exit0、81FT/243实际CPU读取、零错误/暂缓、0GPU/outer概率；仍partial_verified，不能代替135/405。session15244已结束，计数未变不重复本有效审计。

- 当前LaBraM5s seed22继续；队列/恢复V2及当前四进程完整出生身份已只读复核，同一健康单GPU链。此次没有GPU探针、重启队列、修改84封存源或新增正式修复执行；此前CPU桥/私有适配/假图证据保持准备范围，缺真实GPU资格与完整新版本绑定不能宣称两故障已修复。



### 2026-10-11 19:08 当前CPU与Mamba私有实例准备



- 实际CPU6 completed_finetune_checkpoint_audit_v6_1791716599554237400.json（SHA3ab6145abf9dd501fd16e61405ed14f7b78926ce67ff11f58a84a2e4f30ea964）exit0，80FT/240CPU、零错误/暂缓、0GPU/outer概率，含LaBraM5s seed11五折；仍partial_verified。session51563已结束。当前seed22/f1已100轮/best91继续，未改预算/选模/阈值，不因完成数增加而提前测试。

- 新adapters/eegmamba_v2.py（SHAc516c843db45ab987da560d1c952de1d888d0f873e57e5b2c262aa1c0b655741）保留原Linux严格官方加载、12个Mamba2、输入/方向/投影/参数名单；只拟给12个mixer实例克隆相同forward代码与私有split-scan函数图，原native forward复用、私有backward指向FP32 CPU累加桥。固定旧adapter（1b7cef）、mamba2（605e44）、ssd_combined（3717ba）、ssd_chunk_state（c970c2）、桥0159ce与图417c24六来源；安装包/class/global torch均不改。此适配器未接入当前F3，报告始终GPU_qualification_complete=false/new_formal_execution_registration_required=true。

- 5项假加载/class工厂护栏 EEGMamba_private_adapter_v2_preparation_1791716575285075300.json（SHA6c3de8c20614d1f091c49f3ea3ff85170e1aad19ff03b7df7d0106127a67d925）通过，无实际Linux包导入/作者模型/权重/Tensor操作/raw/GPU。独立审查源review_mamba_private_sources_v1.py只做AST和独立标准库假图，报告 independent_mamba_private_sources_review_1791716822749643000.json（SHAa0af28d79eb69cdc73632f05e9b8e98efd4a0f1dd803ae36e4ef6bcf1226453c）核私有图原引用/AMP闭包/forward保留、源守卫先于真实加载、顺序变化/未资格标记明确，并逐字节复核当前84源。

- 所有这些仍是准备，不是实际Linux AMP/autograd图绑定或GPU数学/梯度/冻结/恢复工程。下一步应写源绑定资格入口及其默认零操作/排他/暂停护栏；在Q5健康LaBraM阶段不实际运行GPU探针。两条修复随后必须新正式执行、CPU/评价/成本/全30项及成本附页版本；不拿CPU6/cost7准备冒称支持尚未登记的新执行。CodeBrain15/Mamba15仍必须真实完成。



### 2026-10-11 18:39 当前核验与Mamba私有图准备



- 实际CPU6 completed_finetune_checkpoint_audit_v6_1791715021561371900.json（SHA970c9b7540ae41a17622349dae12378596cfc42ec3805f3dcab009dba2ba3fc2）exit0，77FT/231实际CPU、零错误/暂缓、0GPU/outer概率，覆盖LaBraM5s首两折。第二折167轮/best137/668updates完成，当前第三折继续；仍partial_verified，绝非135/405。session57138已结束，计数未变不重复此有效审计。

- 新mamba_private_backward_graph_v1.py（SHA417c2440d0ff5f1be7ea923d2f81bfff56faa3d8444019013b0df246a0efab5a）只提供私有Python图构建：原native forward函数复用、同一函数字节码/default/closure保留；对两SSD helper克隆私有torch namespace，再克隆core backward和AMP装饰器闭包、由调用者提供新autograd类工厂，不改旧函数/类/global/安装包。它不是有效GPU入口，也未导入或接入真实Linux图；调用者仍须独立绑定实际安装源/FP32桥/类工厂及完整GPU资格。

- 6项纯标准库假图护栏 mamba_private_backward_graph_preparation_1791715032437300600.json（SHA63a800c9ad4e201db00dbfc8d157b5dd226bf5ce8a81fb09b50a89ad1b7d767d）通过，不导入Torch/Mamba，无Tensor/模型/权重/raw/GPU/optimizer操作。尚须源码独立审查与真实Linux模型/梯度/冻结/恢复资格，不将假图通过称GPU修好或正式执行完成。

- 当时当前Q5/LaBraM链9840→34592→7744→21308完整出生身份已只读核过，没有同时运行第二实验或残留CPU审计。当前健康训练继续；CodeBrain和Mamba共30FT未完成，旧故障/成本/84源均保留，未来必须新执行/CPU/评价/成本/最终门版本，不能热改原图或盲重跑旧prepare。



### 2026-10-11 18:26 BIOT完成与EEGMamba故障



- BIOT15全部实际完成，原Q5该父任务status=complete/returncode0（18:04:04.980646）后继续后续任务；当前进程树已换为LaBraM，未另行取得两孩子各自独立exit记录；18:05登记300/360、FT75/135，随后LaBraM5s首折已130轮/best100/validation_patience完成并继续第二折。新的LaBraM孩子由当前owner核：launcher7744/birth134361866917530676、实际训练21308/birth134361866917616088；此身份仅当时快照，下次动态核对。原Q5/source5与恢复V2/previous链不变，只有一个实验训练链。

- 实际CPU6 completed_finetune_checkpoint_audit_v6_1791713823233894500.json（SHAf4466079739a07961e1f2ec6bc34e2bba9b4631b01cb23e7626706838463f6a9）exit0、75FT/225CPU、零错误/暂缓、0GPU/outer概率，包含BIOT全15；后来LaBraM首折完成不在本快照中，不伪填76/228或135/405。session12000已结束，计数未变不重做有效审计。

- EEGMamba工程18:04真实failed：Linux原Triton SSD反向ssd_chunk_state.py:845的torch.cumsum没有严格确定性CUDA实现；其源码:880另有同类同存储view累加，需要源绑定一起核。原工程e_1791713053065335929_6a08bb7b保留：2forward返回、2back开始/1back返回、0optimizer、16窗口，primary21.530495979秒与父31.91640709999956秒嵌套不相加，未返回back内部工作未知；15正式拟合仍blocked。故障附页 actual_EEGMamba_deterministic_cumsum_failure_1791714410080108300.json（SHAf4b32e8b37e0c0a786d0c91f19005f4ab2a431f21d68a980e5b92b06dcee6c0e）已绑定旧失败/日志/新CPU准备/当前84源。

- mamba_cumsum_cpu_fp32_v1.py（SHA0159cec62acd3c7d07d60bbdeef961110e28d5b5579d47267655399da4e3c0e4）只是未集成的CPU FP32顺序prefix-sum原型：保持同storage/view的out=语义、显式FP32逐项相加并传回原device，不关闭确定性、不改安装包/global torch。它改变加法结合顺序，不宣称与原生CPU/CUDA累加逐位一致；对抗样例显示FP32顺序与原生CPU结果可不同。任何正式采用须源/数值/真实GPU工程审查和新版本登记，不能因CPU通过直接应用。

- 6CPU合成护栏 mamba_fixed_fp32_cumsum_CPU_preparation_1791713923374887400.json（SHA29a5011e43bbcd15349be4b52950d98c6b51b1c561db8ae1162e5c06cf19fc64）与独立IEEE-FP32标量顺序参考5组测试 independent_mamba_fp32_cumsum_CPU_review_1791714270014298500.json（SHA115c401c3ae08d97b655b377629aee9b026eca0776e0dfae08485a1bbc44a9f5）通过，未构建作者模型/读官方权重或raw/执行GPU/optimizer。尚未准备实际Linux私有backward调用链和真实GPU工程；不得在当前健康LaBraM任务中重跑Mamba工程或修改封存包。与CodeBrain故障分别修复、分别计费，全部30个blocked FT仍必须完成。

- 最终总门仍未开放。当前原Q5其余必做任务继续；它即使因30个blocked任务而结束也不是计划完成，不评价或补测。修复后要有新的真实执行/CPU/评价/成本/全30项及暂停、G3、低功耗最终交付绑定；旧cost7/CPU6准备不足以支持未知新增版本。



### 2026-10-11 17:59 新增核验与私有适配准备



- 实际CPU6 completed_finetune_checkpoint_audit_v6_1791712059990604600.json（SHA1899c410098f9de0a194028e50c98a9c97a6aa0c5556c2a3df7f64907292d58c）已exit0，71完成FT/213实际CPU读取，零错误与暂缓、0GPU/测试概率，仍partial_verified。session89355已结束。16:xx旧审计及此报告保留各自真实范围；之后新增完成不能伪填进此快照。

- 新adapters/codebrain_v3.py（SHA0e2d9ee14c2b0d3855532e4bca152c725703838138a8e6ef13b8b0051cace767）保留原V2严格权重/8缓存初始化/参数名单；只拟对8个GConv实例克隆同一forward字节码、默认参数与closure，以私有F命名空间使用CPU反向桥，不改原作者类、全局torch.functional或当前模型。原V2（06c783）与SGConv（94f12f）原字节和当前84源保持不变。新adapter报告GPU_qualification_complete=false/new_formal_execution_registration_required=true；当前F3仍只调用原codebrain_v2，不自动启用新文件。

- 6项CPU/假加载护栏 codebrain_private_adapter_v3_preparation_1791712054378282000.json（SHA0f3df7719006c36ebe49b238369dd43a26ad3ce26a0d4c9d9078df42f1bbe2f7）通过，没有作者模型、权重检查点、raw、GPU或优化器。root工具内独立AST/隔离函数引用检查 independent_codebrain_private_adapter_review_1791712289390893400.json（SHA15116b34740cac28cff2962189f6ccaccd5d9f8e9f9ddb86ed328d809ed23521）核原函数代码/kwdefaults/closure保留且原全局不变；这是准备范围，不是可替代最终语义审查或真实GPU工程的证书。

- 下一步CodeBrain修复需在当前必做队列自然结束/安全排他后，核实时进程与owner，实际验证新私有绑定下原CUDA前向及warm/lastblock两阶段完整梯度、冻结状态、恢复与真实成本；不得在健康BIOT/Mamba/LaBraM训练中追加竞争GPU。正式15个CodeBrain拟合必须绑定新的执行、CPU/评价/成本与全30项验收版本；不复制Q5终态、不热改84来源或盲重跑旧失败入口。当前其它必做训练继续，测试门不开放。



### 2026-10-11 17:26 有效CPU与修复准备



- 新真实CPU6 completed_finetune_checkpoint_audit_v6_1791710468266391600.json（SHAea7e84f408755dc1abe410a673f2415b79c741598f5a2d8c44eec621fe3207cc）exit0、68完成FT/204实际CPU加载、零错误与暂缓，仍partial_verified。审计session36909已结束；不重复已有效快照，不作135/405或全360声明。

- 独立检查源review_codebrain_cpu_derivative_v1.py已对原型7e98df做AST边界检查，并按原SSSM与SGConv静态参数核对8×400×2、全部10个作者cat尺度（1,1,2,…,256），以独立坐标Jacobian转置计算得到CPU导数逐位一致。证据 independent_codebrain_cpu_bridge_review_1791710603266933700.json（SHAb6ef642c8556ad4cac49e2d98bd38a66bbd7d9f87776f373af0a11ac8557e2ba）；原SGConv/SSSM/adapter及安装的ATen头文件均来源绑定。该检查没有作者模型、官方权重/Tensor检查点、raw、GPU、optimizer或当前队列操作；GPU完整模型数值与高阶导数未验证，原型仍未正式集成。

- Q5原source5/basea9/恢复V2/previous链及当前BIOT孩子完整出生身份已只读复核，只有一个训练链，旧故障没有新增或重跑。旧CodeBrain故障仍需真实单GPU工程及新来源/CPU/评价/成本/最终门兼容，准备通过不代表完成15个拟合。



### 2026-10-11 17:00 心跳新增证据



- LUNA15已实际全部完成，当前健康队列已进入BIOT；16:50登记288/360、FT63/135，之后BIOT seed11五折全部完成并继续seed22。当前Q5/source5及恢复V2三键/previous链不变；BIOT孩子由实际owner动态核对，非旧LUNA PID。

- 新真实CPU6报告 completed_finetune_checkpoint_audit_v6_1791708904979102100.json（SHAa82e03c870246df72adc0867b849a3453a204776b6f86d00527836d1aabffe9d）exit0、实际64完成FT/192CPU，零错误/暂缓；包括LUNA全15及BIOT前4。后来BIOT第5完成不能冒称已由此报告核验，计数/核验范围分开。未执行测试或GPU模型，仍不是135/405。

- CodeBrain必做工程16:33真实failed：严格确定性模式下CUDA upsample_linear1d_backward_out_cuda没有实现。原工程attempt e_1791707602147260700_0b9d9b39完整保留，2forward返回、2back开始/1back返回、0optimizer、16窗口；原primary2.941266399999904秒与父5.441647800000055秒嵌套不相加，未返回反向未知。其15正式拟合仍blocked，不能省略或当完成；其余健康任务继续。

- CPU原型 codebrain_linear_backward_cpu_v1.py尚未集成：保留原native forward，只拟将插值反向交给同一ATen原生CPU导数并传回原device，不关闭确定性、不改FP32或原SGConv源码。8CPU合成护栏、38逐位相同前向/梯度case通过，证明 codebrain_linear_cpu_backward_synthetic_1791709048235933800.json（SHA1b1da9035af1b2d70718b1b77570c69f01cd86cdbfb5f293aaa3c1d761df4971）。这不是GPU/全模型资格或提速证明；CPU传输与同步真实成本尚待测。

- 故障与准备证据 codebrain_deterministic_backward_repair_preparation_1791709244666729500.json（SHAd692de5a4ce22f8a9845f804606b580c078f277c90196bf492d604d426118e2b）绑定原失败、SGConv、原型、CPU报告并复核当前84源原字节。未来须先独立审查、自然安全边界的真实单GPU工程以及新受审执行/CPU/评价/成本/最终门绑定；不热改当前84源，不在BIOT健康训练时追加GPU探测或启第二队列。旧CodeBrain失败入口不能盲重跑，CPU原型存在不代表已修好15个正式拟合。



本轮于 2026-10-08 实际启动。总登记为 360 次正式拟合：345 次监督训练与 15 次逐折自监督。九篇候选全部保留；工程探测不计作论文完成。旧实验和原始封存结果保持独立。



第一批 EEGPT 与匹配 4 秒 LaBraM 的 30 次固定 300 轮重训已全部完成，历史前 60 轮重放与全批独立核验通过。第二批冻结读出队列包含 EEGPT、CSBrain、REVE、LUNA、CodeBrain、BIOT，以及 CBraMod 和 LaBraM 的三个控制，共 135 次拟合，已于 2026-10-08 13:55 完成。此前两批合计 165 次，接续器核验完成证据后自动交接至下一阶段。



接续队列覆盖余下 195 次：EEGConformer 从头训练 15 次、EEG2Rep 逐折自监督 15 次及随后监督 15 次、EEGMamba 冻结小头 15 次，以及七候选加 LaBraM 两个窗口控制的有限微调 135 次。2026-10-08 13:29 已封存 31 份执行来源并启动接续器；13:56 完成 EEGConformer 实际 GPU 工程检查并开始其 15 次正式训练，后续按登记的 27 个子任务串行执行。每条路线先以训练人员完成真实 GPU 工程检查再开训，工程失败保持未完成并保留成本，不以替代架构或微批损失冒称真实逻辑批 64。微调均重新加载作者权重和新小头，不因冻结路线的验证表现取消。



CodeBrain 原作者 checkpoint 的八个卷积归一化缓存会在首次前向改变状态。已登记独立 v2：按作者权重公式先初始化缓存，排除末块四个不参与输出的参数。实际 GPU 修复导出已通过状态不变检查；37 人共 2,220 个窗口的原生特征与原始首次导出逐位相同。原适配器、原始导出记录及失败记录保留，冻结训练队列已使用修复结果恢复。



所有外层测试保持冻结，直到 360 次正式拟合的真实完成证据全部核验通过。接续器随后先执行 Linux EEGMamba 微调评价，再执行统一评价；每次评价还会独立复核完整门槛、固定检查点、来源和概率产物。判别阈值、快照和比较规则均已预先固定，完成报告位于 `output/analysis/eeg_all_candidates_v1_20261008/九候选统一完成报告.md`，只有实际统一评价成功后才会生成。



2026-10-08 暂停时完成 90/360 次拟合；恢复事件单独记录在实验目录 `resume_events/`，原暂停记录保持不变。恢复后 CodeBrain 的 15 次冻结读出训练已完成，BIOT 冻结读出正在继续。该段为恢复时的历史记录，实时数量以登记状态为准。



“继续进行下一步”后的只读检查确认接续器与 EEGConformer 训练子进程都在运行，Linux 隔离解释器及四个官方包版本与登记一致。该环境预检没有执行 GPU 模型；EEGMamba 的实际 GPU 资格仍由后续登记任务检查。外层测试门槛再次验证拒绝开放，未产生测试前向。



阶段交接的独立审计已核验此前 165 次拟合的 660 个完成产物绑定、31 份执行来源和原分折名单，测试前向计数全部为 0。Conformer 最早三折的逐轮停止规则、参数更新、BN 状态、完整恢复字段和仅训练人员重载证据均通过独立 CPU 审查。



后续 Linux 包元数据预检发现官方 Mamba 与 causal-conv1d 都声明的 `ninja` 依赖缺失。已从官方 PyPI 下载固定 1.13.0 二进制轮包并核对发布 SHA，仅补齐此依赖，所有原有包版本保持一致；安装后依赖一致性检查通过。下载与安装前后证据另存实验资产 `assets/eegmamba_dependency_repair/ninja_v1/`，原 runtime manifest 和已封存训练来源保持原字节。该修复没有模型前向或优化更新，EEGMamba 实际 GPU 工程检查仍待队列运行。



开训前的合成复核还发现原 SSL 在 BF16 上下文中的协方差乘法实际降为 BF16，与本轮声明的 FP32 损失统计不符。独立 `ssl_v2.py` 仅在完整作者损失计算时关闭 autocast，保留编码器 BF16、完整逻辑批 64、SSP、EMA、采样、优化器、早停和恢复流程；8 项 CPU 合成测试及独立审查通过，原 SSL 源码和适配器未改，旧 SSL 尚无工程准备或正式拟合。



新版接续与交接监督已封存 37 份来源，6 项纯模拟交接测试及独立审查通过，实际进程身份已按完整创建时间核对。监督器已启用：只暂时保持旧接续器等待，Conformer 的实际训练进程持续运行；待 15 次 Conformer 完整结束、两级子进程正常退出且全部产物核验通过，再进入修复后的接续队列。旧 owner、旧任务记录及旧源码字节保留，成功交接证明另存 `continuation_queue_v2/handoff_complete.json`。该队列复用原 195 次矩阵中的已完成 Conformer 15 次，随后执行其余 180 次，不增加或减少正式拟合。



上述第一轮交接已实际完成，Conformer 全 15 次独立审计通过，累计 7,424 次优化更新。EEG2Rep 实际 CUDA 工程已通过完整逻辑批 64、BF16 编码器与 FP32 完整损失检查，峰值分配 2,476,714,496 字节；工程没有优化器更新，随后逐折自监督正式训练已启动。



用户要求直接执行到全部计划完成后，已建立持续完成目标。进一步预检发现 Linux owner 路径与版本化队列目录不兼容，以及 LUNA 原作者固定 RoPE 频率被错误列入可学参数。独立 v3 执行入口修复 owner 路由、保留 LUNA 固定频率，补齐逐次前向/反向/优化步骤成本记录及最佳检查点相对初始和终态的独立更新证据；允许最佳模型来自前十轮仅头暖身，不改原选模规则。评价 v2 仅适配真实参数名单并披露这些证据，保持全 360 门槛、435 固定检查点、判别阈值和保留规则不变。



v3 已封存 47 份来源，联合 13 项合成检查与独立审查通过。实际交接监管已启用，只保持 v2 调度器等待，当前 15 次 SSL 训练完整连续执行；其整批正常退出且 195 次既有拟合证据通过后，v3 复用已完成的 Conformer 与 SSL 共 30 次，执行余下 165 次并最终统一评价。所有旧 owner、旧源和失败/核验记录保留；Mamba 和各微调路线仍必须经过各自真实 GPU 工程检查。



2026-10-08 19:26，SSL 全 15 次实际完成且两级子进程正常退出，v3 交接完成证明已保存。全 15 次 SSL 独立只读审计通过：终态共 49,287 次优化与 EMA 更新，所选最佳检查点合计 26,247 次，两种统计分别保留。v3 已完成 EEG2Rep 监督路线的真实训练人员工程检查并开始全部 15 次监督拟合；首三折的实际同折 SSL 最佳权重初始化、骨干更新、目标/预测器冻结、逐轮政策与完整恢复产物核验通过。当前外层测试仍锁定，Mamba 与九条有限微调路线尚待后续真实执行。



2026-10-08 19:58 按用户要求暂停时，已完成 203/360 次拟合，Q3 调度器和当前监督训练进程均已挂起。2026-10-09 用户要求继续；检查确认旧进程已退出，原检查点与暂停记录一致。仅归档已失效的 Q3 占用记录后，以原封存队列恢复：EEG2Rep 的 seed 22 / fold 4 从完整第 26 轮模型、优化器、随机状态与采样位置续训，并已实际推进到后续折。暂停前第 27 轮已有 4 次更新和 27 次反向未写入检查点，这些额外工作单列为恢复成本；未知诊断前向和旧尝试耗时保留未知，过夜暂停不计入训练耗时。暂停证明在 `pause_events/1791460680736346200/`，归档与恢复证据在 `resume_events/1791506856841705600.json` 及其同名目录。



2026-10-09 09:20，EEG2Rep 监督阶段全 15 次实际完成，累计正式拟合达到 210/360。整批独立 CPU 检查已核对各折实际 SSL 最佳权重初始化、88 个可训练参数的终态更新、目标编码器与预测器冻结、逐轮停止政策、优化器和随机状态，共 5,856 次正式优化更新；报告为 `eeg2rep_supervised_cumulative_audit_15fits_949f54a9169e.json`。Q3 随后进入 EEGMamba 真实 CUDA 资格检查，测试门槛仍锁定。



早期冻结读出及配对 CNN 的部分工程探针没有记录耗时或显存峰值，历史缺口保留为未知。已准备主队列实际训练及统一评价均完成、占用记录释放之后的独立零优化器更新补测；其新测量只使用原 fold 1 训练人员和相同缓存，不回填旧数值，也不改变正式模型或结果。最终成本与全计划验收需要这些真实补测及全部完成产物，当前预审不代表计划完成。



2026-10-09 09:32，EEGMamba 官方 CUDA 资格检查、37 人原生 200 维特征导出和全部 15 次冻结小头拟合均已完成，正式拟合达到 225/360。资格检查实际执行四次编码、一次反向及一次合成 SGD 更新；特征导出含 2,220 个新窗口和 148 个复算窗口，共 592 批，工程成本与正式训练分列。独立阶段审核为 `eeg2rep_mamba_stage_audit_1791509311136476900.json`，全矩阵阶段验收为 `full_plan_acceptance_audit_v3_1791509590514189500.json`：225 份拟合证据通过、零核验错误、未读取外测试概率。EEGPT 有限微调已通过真实训练人员工程检查并开始正式运行，剩余九组共 135 次有限微调仍全部保留。



最终后处理程序已分别审查并保留来源哈希：`output/analysis/eeg_all_candidates_v1_20261008/supplemental_head_resource_measurement_v2.py --run` 只在全 360、Q3 统一评价与占用释放后执行 12 条独立资源补测；随后 `finalize_full_plan_cost_v2.py` 生成独立最终成本 JSON 和可读报告。全计划还需要最终独立语义审查与 `audit_full_plan_acceptance_v3.py --cost-artifact ... --cost-report ... --semantic-review ... --require-complete` 全部通过。这些准备程序及当前阶段报告均不表示最终完成。



微调过程中当前使用同一分析目录的 `audit_completed_finetune_checkpoints_v3.py`。它只加载已真实完成的微调任务的三个 CPU 检查点，核验原始身份、head/backbone 参数组、矩张量、实际更新和选模状态；各次尝试的物理调用计数与一次持久训练轨迹分别统计，失败旁证不重复相加。历史 v2 审计器及其 16 项回归和独立审查保持原字节。EEGPT 首个微调任务 seed 11 / fold 1 已实际完成 67 轮、268 次优化更新，按验证 CE 选择第 37 轮；真实 CPU 审计 `completed_finetune_checkpoint_audit_v2_1791510698316559500.json` 已读取其三个检查点并核验通过。该报告为微调 1/135 的部分完成证明，不能替代最终 135 次验收。



用户在持续接续授权询问后明确要求“一直进行到计划结束”，本对话已于 2026-10-09 建立每 15 分钟一次的自动接续（id：`eeg`）。它检查现有单 GPU 队列、处理实际故障，并在全训练与统一评价结束后执行已审补测及最终验收；状态无实质变化时保持安静，实际全部完成后停止。用户之后再次暂停或停止时，须立即暂停自动接续并按确切身份处理正在运行的训练进程。



首个真实微调的独立复核 `independent_first_actual_finetune_review_1791510902423001600.json` 已通过：六个分类头参数与十二个编码器参数的分组、实际所选与终态编码器更新、冻结状态及逐次计数均核对一致。实际 v4 成本适配 `first_completed_finetune_cost_v4_review_1791510965583186400.json` 也通过：268 次更新、1,809 次反向、135,192 个编码器窗口；928.84 秒整次尝试与 911.25 秒嵌套训练时间分列，不重复相加。该证明仍仅覆盖一个微调任务，其余任务继续执行。



EEGPT 微调 seed 11 的前三折现已完成，累计正式拟合为 228/360。实际 CPU 审计 `completed_finetune_checkpoint_audit_v2_1791512527325217600.json` 已核验这三个任务的九个检查点，无错误。第三折是首个实际热身选模案例：训练按原规则完成 40 轮，验证 CE 选中第 4 轮；所选编码器与初始权重相同，终态十二个末层参数则均已更新，分类头累计 160 步、编码器累计 120 步。独立证据 `independent_head_warmup_selected_finetune_review_1791512696433956200.json` 已核对人员、全曲线、状态与实际调用计数。最终评价须如实披露所选模型的热身阶段，不将终态更新冒称为所选编码器更新，也不改动预定选模规则。其余微调及全部外测试仍待后续执行。



2026-10-09 首个种子 seed 11 的五折 EEGPT 微调均已完成，累计正式拟合达到 230/360，seed 22 已继续运行。真实 CPU 审计 `completed_finetune_checkpoint_audit_v2_1791513474513739600.json` 核验五个任务的十五个检查点，无错误；独立阶段审核 `independent_eegpt_first_seed5_review_1791513744038578800.json` 核对注册分折、来源、逐轮政策和实际调用，共 1,140 次更新、7,557 次反向。五折中两折选择末层解冻后的检查点，三折选择分类头热身检查点；五折终态均已实际更新末层及分类头。整次正式尝试耗时小计 3,806.26 秒，工程的两次反向和 32.83 秒独立记录，不与嵌套训练耗时重复相加。该阶段证明仅覆盖微调 5/135、EEGPT 5/15，其余任务和最终评价继续待完成。



首次跨种子初始化的独立证据 `independent_first_cross_seed_finetune_review_1791515952258828600.json` 已通过：seed 11/22 的 fold 1 使用同一划分和官方初始编码器，实际完整初始状态随新分类头种子改变；该结论绑定真实六次拟合的 CPU 检查，而非仅依据文件名。seed 22 / fold 1 按原政策选择第 52 轮并在第 82 轮停止，分类头 328 步、编码器 288 步。



只读 v2 核验随后在扫描无关、正在更新的 F2 attempt 文件时遇到短暂 PermissionError。原失败报告 `completed_finetune_checkpoint_audit_v2_1791516059601364100.json` 保留：八个已完成任务中七个核验、21 次 CPU 读取，未改变任何正式训练状态。新增 v3 先读取小 JSON 并筛身份，进行有界重试；目标记录仍变化时暂缓，未知身份读取失败仍报错，全 135 完成声明须无错误或暂缓项。31 项回归与独立审查 `independent_completed_finetune_auditor_v3_review_1791516620885347500.json` 通过，十个原有 CPU/数学/计数验证函数保持 AST 相同。实际 v3 重验 `completed_finetune_checkpoint_audit_v3_1791516632794218200.json` 已核验九次完成拟合、27 个检查点，无错误。自动接续已改为此版本，47 份封存训练来源未修改。



EEGPT 的 seed 22 五折随后全部完成，累计正式拟合达到 235/360，seed 33 已开始运行。真实 v3 审计 `completed_finetune_checkpoint_audit_v3_1791517081831612700.json` 核验十次完成拟合的三十个检查点，无错误；独立阶段证据 `independent_eegpt_second_seed5_ten_fit_review_1791517192298391400.json` 核对 seed 22 的全部曲线和实际尝试：1,224 次更新、8,118 次反向。合计两个种子为 2,364 次更新、15,675 次反向；所选十折中四折来自解冻阶段、六折来自分类头热身，终态则均完成实际末层训练。此时仅确认 EEGPT 10/15、全部微调 10/135，完整测试与最终验收仍待后续完成。



EEGPT 全十五次有限微调的 JSON 阶段审核入口已准备：同一分析目录 `review_eegpt_full_finetune_15.py --cpu-audit <真实v3审计报告绝对路径> --require-complete`。准备证明 `review_eegpt_full_finetune_15_preparation_1791518409137947800.json` 仅记录真实 11/15 待完成支路和独立静态复核；正式审核必须等待该条件全部十五次实际完成及对应真实 CPU 证明，准备本身不是完成证书。



CodeBrain 有限微调的缓存链只读静态核查 `codebrain_finetune_cache_static_review_1791516601655704200.json` 未发现确定问题：有效 weight 每次由最新 g/v 参数通过前向钩子重算，且不是保存状态；八组预初始化的 GConv 归一化基准按作者逻辑固定，因此不应造成诊断或检查点 SHA 漂移。该核查未执行模型、GPU 或正式微调；后续仍须队列中的真实工程和全部拟合。



2026-10-09 12:38，EEGPT 全部十五次有限微调实际结束，子进程正常退出 0，累计正式拟合达到 240/360。真实 v3 审计 `completed_finetune_checkpoint_audit_v3_1791521176166520500.json` 已核验四十五个检查点，无错误；正式阶段审核 `review_eegpt_full_finetune_15_passed_1791521228008585400.json` 与独立复核 `independent_actual_eegpt_full15_cost_review_1791521425626713200.json` 均通过。该条件实际 3,652 次更新、24,222 次反向；十五个终态均更新十二个末层参数和六个分类头参数，所选检查点七项来自解冻阶段、八项来自分类头热身。正式尝试耗时 11,139.57 秒，嵌套训练时间及工程的 32.83 秒另列；不将重叠组更新或时间相加。该证明仅为十五次微调完成，不是全 135 次或全 360 次完成。



Q3 随后完成 CSBrain 实际训练人员的 raw 微调工程检查并启动十五次正式拟合。独立资格证据 `independent_csbrain_raw_finetune_engineering_review_1791521450680645100.json` 核对官方严格加载、原生 D200、head-only 与 layer 11 联合阶段的实际梯度、两次反向/零优化器步骤及恢复验证；该工程记录不计正式拟合，也不回填早期冻结小头的资源缺口。当前全局外测试仍锁定，剩余八组有限微调继续按原登记运行。



EEGMamba 公开 checkpoint 的来源局限已单独核查：`independent_eegmamba_pretraining_provenance_review_1791521798559419700.json`。现有版本绑定的发布记录、作者代码和说明未能核实该权重实际预训练语料、参与者清单或是否含 MODMA 本 37 人；这些字段保持未知，不能从默认训练目录或本地冻结导出的空拟合名单推断排除了同一参与者。本轮本地折内拟合隔离与外部预训练的参与者隔离是两项不同证据。最终独立语义审查及来源/局限说明必须保留此事实，不把官方权重适配结果称为预训练参与者重叠已被排除的盲测结果；完整 360 条件、权重和既定预算均继续保持原登记。



CSBrain 首折已完成首次真实有限微调延长：第 120 轮最佳为第 117 轮，最近二十轮验证 CE 相对前二十轮改善 3.09%，按原规则将上限由 120 延至最多 200；最终在第 147 轮满足最佳后 30 轮耐心而停止，未增加拟合数量。稳定前 120 轮快照及独立延长证据分别为 `csbrain_fold1_epoch120_stable_prefix_1791524264695918300.json`、`independent_csbrain_registered_extension_review_1791524264695918300.json`。真实 v3 审计 `completed_finetune_checkpoint_audit_v3_1791525043354058900.json` 已核验现有十六次微调的四十八个检查点；独立完成复核 `independent_csbrain_first_extended_completion_review_1791525172742025200.json` 核对选中第 117 轮及终态第 147 轮的二十个编码器参数均有更新，273 个冻结参数不变。该折实际 588 次分类头更新、548 次编码器组更新、3,969 次反向；组更新属于重叠计数，不相加作为物理步骤。累计正式拟合为 241/360，CSBrain 第二折继续运行，外层测试保持锁定。



CSBrain seed 11 的前三折随后完成，累计正式拟合为 243/360。实际 v3 审计 `completed_finetune_checkpoint_audit_v3_1791528577327694600.json` 核验现有十八次微调的五十四个检查点，无错误。该方法的首个热身选模案例为 fold 3：完成 40 轮，最佳第 8 轮；所选编码器与初始状态相同，终态则二十个末层参数和六个分类头参数均已更新，273 个冻结参数保持不变。独立复核 `independent_csbrain_head_warmup_completion_review_1791528721401651600.json` 核对 26 人的尾批次、160/120 两组更新、1,040 次反向及停止规则，未强制改选解冻后模型。其余十二次 CSBrain 微调和完整全计划仍未完成。



CSBrain 首个种子 seed 11 的五折已全部完成，累计正式拟合为 245/360，seed 22 正在继续。实际 v3 审计 `completed_finetune_checkpoint_audit_v3_1791530442765370800.json` 核验现有二十次微调的六十个检查点，无错误；独立阶段审查 `independent_csbrain_first_seed5_review_1791530613827816600.json` 核对五折人员、初始化、曲线、状态和实际调用。该批 1,488 次更新、9,899 次反向；所选两折为末块微调、三折为分类头热身，最佳第 10 轮仍属于热身。五折终态均更新二十个编码器参数和六个分类头参数，273 个冻结参数不变。正式尝试耗时 8,888.81 秒，嵌套训练耗时和工程 9.97 秒分列，不重复相加。此证明只覆盖 CSBrain 5/15、全部微调 20/135，完整 360 及最终评价仍待完成。



CSBrain seed 22 / fold 1 随后完成，累计正式拟合为 246/360。该折第 120 轮的近期验证改善为 1.509876%，触发原登记的唯一延长，最佳第 145 轮位于延长区间；最终第 175 轮满足最佳后 30 轮耐心停止，上限仍为 200。真实 v3 审计 `completed_finetune_checkpoint_audit_v3_1791535793189396100.json` 核验现有二十一次微调的六十三个检查点；独立边界证据 `independent_csbrain_extended_region_selection_review_1791536181574323000.json` 核对完整曲线、选中和终态各二十项编码器变化、273 项冻结状态及 700/660 两组步骤、4,725 次反向。最佳检查点本身未保存的更新计数保持未知，历史第 145 轮计数单独记录，不由终态倒推。该证明仅为 CSBrain 6/15、全部微调 21/135，后续训练继续。



CSBrain 全十五次微调的正式阶段审核使用 `review_csbrain_full_finetune_15_v2.py --cpu-audit <真实v3审计报告绝对路径> --require-complete`。准备证据 `review_csbrain_full_finetune_15_v2_preparation_1791538861608747900.json` 记录真实 6/15 待完成支路及独立静态复核，12 项证据结构测试两次通过。该版本接受覆盖 CSBrain15 与 EEGPT15 的有效更大 CPU 快照，实际全局计数动态披露，避免后续模型已有完成项时被固定 30/90 计数误拒；十五个 CSBrain 身份、全部相关证据、每次三个 CPU 读取及零错误/暂缓等要求保持严格。原版本与全部来源绑定报告保留；当前准备不是十五次完成声明。



2026-10-09 用户明确授权执行“减少同步写盘”和“扩大验证批处理”两项提速后，新增执行观察器 F3 与单 GPU 接续 Q4，原 47 份封存来源保持原字节。F3 仍逐次观察实际前向、反向与优化器调用，只在逻辑更新前后、诊断前后、完整 checkpoint 成功、返回与异常时持久保存；训练 micro1、8 人逻辑批、8 窗口训练袋、FP32、预算、随机状态、选模和阈值不变。缓冲记录在正常返回后才声明调用观察闭合，硬终止的未刷新尾部和未返回调用仍保持未知。新数学契约继续绑定原 F2，实际观察器 version3 则单独绑定 `execution_acceleration_v1/<condition>/registration.json`，不把新观察器冒称原 F2。



20:07 的真实检查点交接核验 CSBrain seed22/fold5 第31轮的完整 CPU 模型、优化器两组、矩形状、步数、LR、随机状态、采样、冻结参数、最佳状态和原曲线；随后按持有的创建时间及父子身份结束旧 Q3 三个进程并归档 owner。第31轮已有124次更新和837次反向持久保存，旧进程另外实际返回两次反向而没有新增优化器更新；这些额外工作与原暂停的4次更新/27次反向分别计账，未返回前向/编码及旧尝试最终耗时继续未知。证据为 `continuation_queue_v4/handoff/safe_stop_1791547625592196000.json`，独立复核为 `independent_actual_acceleration_handoff_review_1791547861516469400.json`。



实际零优化器批处理资格使用这一稳定检查点和原 train/val 人员，外层 test 未读取。8/16/32/60 窗口诊断的实测中位耗时分别约21.61/11.59/4.83/2.88秒；扩大批处理的 logits 最大差异约2.81e-5至4.59e-5，CE 最大差异约1.48e-6至4.62e-6，未满足预设逐位一致与严格容差。因此 CSBrain 只启用减少写盘，诊断仍为chunk8，不能声称第二项已在该架构生效。真实资格结果 `diagnostic_batching_qualification_1791547784516320500.json` 与所有尝试和工程成本保留。后续未完成的架构分别在排他队列中做 train-only 零更新资格，合格且实测更快才启用扩大诊断；失败、OOM 或没有合格候选不放宽门槛，保留原chunk8。训练、重载和测试始终chunk8。



20:11 Q4 已真实恢复，CSBrain seed22/fold5 从第31轮继续到第40轮并完成，累计250/360。该折实际新增36次更新和243次反向，验证选择第9轮分类头热身状态，终态二十项末块和六项分类头参数均有真实更新。`completed_finetune_checkpoint_audit_v4_1791548447185245000.json` 已核验全部25份真实完成微调的75个CPU检查点，零错误；这不是全135微调或全360完成证明。每个27人折每轮的计数日志保存由原约710次降到11次，原 progress/checkpoint 的持久写入保持不变；这项写次数减少不等于已测完整训练加速比。



Q4 启动时原工程入口额外重测一次 Mamba，四次编码、一次反向、一次合成 SGD 均单独保留，不计正式拟合。新内层测量20.169247279秒与新尝试端点时间另列；旧140.135474809秒 GPU 报告已按其原完整SHA逐字节恢复，新真实报告归档在 `assets/eegmamba_recheck_report_history/`。恢复证据 `progress_events/mamba_recheck_report_restore_1791548331506439200.json` 同时绑定旧原字节、新报告和新primary ledger。后续如必须重启，需要新增版本复用已完成工程并保护旧报告，不能重复运行旧检查器覆盖历史证据。



最终门控同步新增版本：`audit_completed_finetune_checkpoints_v4.py` 支持原F2与实际F3的严格混合审计；Q4 在任何外层测试前需要真实全135份微调/405次CPU核验通过。全360、Q4统一评价及owner释放后，只运行 `supplemental_head_resource_measurement_v3.py --run`、`finalize_full_plan_cost_v3.py` 和 `audit_full_plan_acceptance_v4.py --cost-artifact ... --cost-report ... --semantic-review ... --require-complete`；新补测和成本仍需真实执行及独立最终语义审查。旧分析版本保持原字节，不把准备或局部提速当作计划结束。



用户随后明确要求尝试“减少 CPU–GPU 同步等待”。新增原型分别合并诊断损失/有限性回传、按原字节打包状态指纹、合并梯度统计与 CE 回传，并尝试每人60窗口数据一次传入后仍按原8/最后4窗口计算；这不等同于前一项扩大编码批量。诊断17项、训练15项、指纹14项以及零更新资格入口13项CPU/合成检查通过，原训练代码及Q4的59份来源均未改变。训练成功路径的显式同步点按源码及CPU观测从3M+P变为2M+1；CSBrain末块阶段8人批为50→17，这只是同步次数，不是GPU实测加速比。



2026-10-09 21:08 已实际启动独立自然边界守护 `sync_qualification_boundary_guard_v1.py`：只保持Q4调度器等待，当前CSBrain的GPU训练及其launcher均未暂停或终止，仍完整执行其十五次拟合。实时状态为 `execution_acceleration_v1/sync_boundary/state.json`，guardian身份在同目录 `guardian_owner.json`。全CSBrain15完成、原两个孩子正常exit0、来源/完成证据通过并确认GPU无人占用后，它顺序调用 `qualify_sync_reduction_v1.py`、`qualify_training_sync_v1.py`。诊断仅固定已完成seed22/fold5的train/val，训练梯度探针仅该fold前8名train、epoch1/11各warm1对与实测2对，总96次真实反向/零优化器更新；只改变工程探针，不增加正式拟合。守护核实资格进程退出、无残留孩子/lease且GPU释放后恢复原Q4，无重复启动或重跑Mamba工程。



同步尝试的GPU数值等价性、真实耗时与额外显存尚待上述实际运行。只有逐位一致、有限性/冻结/随机状态/优化器证据通过且实测更快，才考虑独立新版本集成，不在当前封存源内热改，也不能把排队或CPU通过称作已在GPU提速。守护与资格的独立审查在 `independent_sync_boundary_guard_review_1791551220284023700.json`、`independent_diagnostic_sync_qualification_review_1791551327405406100.json`、`independent_training_sync_qualification_review_1791551327424815800.json`。工程attempt放在已有 `supervised_engineering/CSBrain-lastblock-finetune/` 族供成本V5单次采集，全程不读取外层test。



若用户再次暂停，先在上述sync_boundary目录写入 `user_pause.json`（严格 `{"paused": true}`），撤销守护的任何新probe与自动恢复，再按真实身份处理正在运行的训练/资格进程并暂停eeg自动接续。该控制标记不会自行过期；用户明确恢复前不能清除。守护异常时若资格进程、CPU孤儿、lease或GPU清理未核实，则保持调度器等待，不能以GPU一时空闲推断所有工作进程已退出。



2026-10-09 自动接续发现GPU进程列表另有游戏客户端/游戏和NVIDIA叠加层（C+G）；当前EEG训练健康，但旧可选性能实测的“所有GPU进程必须消失”会在该组结束后等待不必要的条件，并可能阻塞后续必做微调。已只取消仍在等待的旧同步守护、恢复Q4调度器；GPU训练及launcher没有任何暂停/终止调用。真实取消证据 `sync_boundary/cancel_waiting_complete_1791553589427394800.json`（SHA986d4e...），原守护源码和状态原字节均已归档。



为保留正式训练并如实延期可选测量，新增Q5控制器与自然交接守护V2；训练仍原F3、micro1/chunk8/FP32，E3评价、FT CPU核验V4与costV5保持原实现。Q5实际登记已封存 `continuation_queue_v5/registration.json`（SHAa9db8e215f1d32cd215d550319c0fa1646985c54d2167783a7547b3a6ea5312f），完整继承原Q4的59份来源，共76份来源，36条任务及195个原正式拟合身份无变。Q5实际终态须带source/execution_registration/execution_version=5，不能复制Q4状态冒称完成；评价前的真实135/405 CPU证明只写Q5目录。



当前新守护 `sync_qualification_boundary_guard_v2.py`（SHAbecadd7aadb20a9ca415d661f9aa2291e7c62de4762d4159dd6dbeaec8e6199c）已实际启用，状态及owner在 `execution_acceleration_v1/sync_boundary_v2/`。仍只保持旧Q4dispatcher等待，当前CSBrain GPU训练继续完整十五次；自然exit0且全部完成产物核验后，空闲GPU才运行诊断/梯度同步对照V2。存在图形上下文时，记录零模型/零更新延期，不以其为提速成功。确认无另一个CUDA计算工作进程、无资格残留后才结束旧Q4parent并开始Q5；图形共存仅允许G进程及实际观察到的三个完整C+G路径，不允许未知C/C+G或另一Python训练。游戏及其他程序不被控制。



Q5的AP2对外部GPU上下文或其启动竞态同样记录真实0模型延期，正式训练仍按原F3继续；安静时会收集实际批处理资格V2，但该附件不冒充原B1资格，实际诊断仍8/qualification=null，只有后续版本化数值与速度核验完毕才能启用。已完成的旧工程先核旧任务来源/日志/完成产物并复用，不能重跑Mamba覆盖旧GPU报告。所有旧Q3/Q4来源、失败、成本及原Mamba报告保持原字节。Q5与交接的独立检查在 `independent_q5_natural_handoff_review_1791554815096725400.json`、`independent_sync_boundary_guard_v2_review_1791554750692615400.json`。



Q5实际全360训练/E3评价/owner释放后，后处理仅使用新增 `supplemental_head_resource_measurement_v4.py --run`、`finalize_full_plan_cost_v4.py` 与 `audit_full_plan_acceptance_v5.py --cost-artifact ... --cost-report ... --semantic-review ... --require-complete`。它们仍需真实135/405、12条资源补测和全部30项最终验收，不能提前运行最终分支。新型资格/延期成本及不可变交接旁证单独绑定，不把guardian等待或嵌套时长再加入训练。



用户暂停控制同时检查原sync_boundary/user_pause.json及新sync_boundary_v2/user_pause.json，严格paused=true立即撤销新probe、恢复及交接；明确恢复前不清除。当前旧守护V1已取消，不可重启；新守护等待时不能误启Q4/Q5第二实例。实际Q5 owner出现前当前训练仍Q4/F3，出现后以Q5真实状态为准。



2026-10-09 用户两次明确要求“暂停所有任务，明天再继续”，本轮已暂停，完成253/360。eeg自动接续与目标均为PAUSED/paused；两个严格user_pause.json为paused=true，守护已退出并撤销任何自动恢复。当前Q4dispatcher原已被守护挂起，没有再次叠加挂起；仅按原创建100ns及父子关系暂停launcher18224与实际GPU Python32880。OS核验dispatcher1/1、launcher1/1、GPU Python44/44线程均为Suspended，GPU利用率0%。没有终止训练、启动第二实例或读取测试概率。



实际暂停证据 `pause_events/q4_user_pause_1791558499460173100/pause_complete.json`（SHA247d0b0dc3329a248ef7ca384b4acb2b2956e403d6b512c6b493306ed697d910）记录CSBrain seed33/fold4正在第9轮、36次更新/243次反向、progress所示last_durable_epoch8；完整checkpoint与history/progress、缓冲attempt原字节已保存。正常内存恢复时保留运行现场；如夜间关机使进程消失，必须重新核对实际完整checkpoint及缓冲下界，未刷新尾保持未知，不能猜作零。另有本次暂停控制标记格式纠正旁证保留，守护的控制错误未影响正式模型；最终严格标记不含任何额外键。过夜暂停不计训练。



明天只有收到新的明确“继续”指令才恢复；不得自动唤醒、删除paused标记、重跑旧队列或把暂停当完成。先查三个原对象是否仍存在（40840/18224/32880，完整出生记录在暂停证据），若仍存活，GPU与launcher各只有本次一次挂起，dispatcher保留原守护一次挂起；恢复前用新受审控制处理已退出的自然守护及将来Q5交接，不能盲目重复NtResume或启动Q5。若任一对象已消失，保留原owner和失败/成本旁证，以新版本安全磁盘续训。Q5尚未实际启动，原76来源与原59/47保持封存。



2026-10-10 用户明确“继续”后，原Q4的40840/18224/32880及旧守护46716均已不存在；实际退出时间、原因及代码保持未知，不能据此伪造自然十五折exit0交接。真实CPU恢复检查 `recovery_cpu_preflight_csb33f4_1791596275747982600.json` 核验暂停快照、完整模型/优化器/RNG/采样状态后，通过独占一次性 `recover_q5_from_pause_v1.py` 恢复原Q5/F3/E3核心。新恢复登记 `continuation_queue_v5/recovery_registration.json`（SHAec1500ce2671600fca1588e41f14fc559e12300824aaa5c4a2625671f8d2ec9f）封存80份来源，完整继承原Q5的76、Q4的59及原47份；原正式身份、预算、分折、选模和训练数学未变。



本次起点253/360，CSBrain seed33/fold4从第8轮结束、32次更新/216次反向恢复，已实际推进到第12轮（第11轮检查点已提交）。实际Q5 owner最初为45980，其训练launcher33964、GPU Python21764；后续以真实owner和完整创建时间为准。旧running观察记录保持e190原字节，唯一不可变进程缺席闭包为 `process_loss_closures/lost_f3_1791597116209422300.json`（SHAe56bbb201c067bf0bab31c0d405520fb2860deb12796f088072bb30bdd73de65）。昨夜第9轮额外已返回的4次更新/27次反向只是下界，已包含于旧36/243物理过程计数，不再相加；未刷新尾、旧完整耗时和退出时间仍未知，过夜暂停不计训练。该恢复与Oct8暂停、Oct9 F2交接成本分别保留。旧owner及双严格暂停标记逐字节归档后撤销，eeg自动接续已重新ACTIVE；旧守护不再运行，不能按固定旧身份复用。



新恢复入口的进程扫描对非旧PID原要求命令行含工作区字符串，独立审查已披露此范围局限。追加的只读全模块CIM证据 `continuation_queue_v5/recovery_launches/all_module_observation_1791597115326.json` 不要求该字符串，实际只发现本次恢复的四个Python父子进程，没有另一训练树。以后监测及故障恢复必须使用全模块、实际owner和完整出生身份；不得热改已封存入口。永久 `recovery_intent.json` 防止并发或失败盲重试生成重复闭包，任何新的故障恢复都须独立核对后使用新版本。



恢复后的预测试核验使用 `audit_completed_finetune_checkpoints_v5.py`，真实135次FT/405次CPU加载及零错误、零暂缓仍为统一评价前硬门。本轮真实部分审计 `completed_finetune_checkpoint_audit_v5_1791597305776022800.json`（SHA1b8bf02377f07109c24eb22e80f942e3cb71fa8b4cf5b8e567f4867edd12bda8）已核验已完成的28次FT/84次CPU加载，零错误；不是全135完成证明，计数不变无需重复。新进程的原生出生100ns只读旁证在 `continuation_queue_v5/recovery_launches/native_birth_observation_1791597115326.json`，实际暂停或故障控制前仍须重新核当时完整对象。原始恢复观察仅以独立闭包解释，未知值不改成零。全360、实际Q5/E3评价及owner释放之后，才运行 `supplemental_head_resource_measurement_v5.py --run`、`finalize_full_plan_cost_v5.py`（成本V6）、独立语义审查和 `audit_full_plan_acceptance_v6.py ... --require-complete`。这些新后处理已通过纯标准库及独立复核，但准备和pending检查不代表真实补测或最终完成。第三项减少CPU–GPU同步仍待新的安全自然边界GPU资格；当前只有F3日志缓冲已启用，诊断仍chunk8，不能宣称同步优化或扩大批处理已提速。



2026-10-10 10:30，用户授权的第三项同步优化已实际启用新的自然边界守护V3（`sync_qualification_boundary_guard_v3.py`，SHA012cc88f26e6693c5e986a43e92ba72a990acd0d3c0c662e4de6c5ee431efc10），状态及guardian owner在 `execution_acceleration_v1/sync_boundary_v3/`。独立可选控制登记SHAcf14a3e5e33ff2b9e39d1859297bdb7a6d25281638b720ebad7db3b7d115a239保留原80来源并另外绑定17份控制/资格来源，不改变正式训练入口。独立资格20项与守护33项标准库/模拟检查通过；真实检查、登记和启用不是GPU资格。初始guardian实际20760，parent40484、creation100ns134360730002601748。当前 `waiting_for_CSBrain_full15` 只保持调度器45980等待一次，launcher33964及GPU21764没有暂停或终止，CSBrain33/fold4已实际继续到第104轮（durable103）。



该组十五次完整训练、90项产物及两个孩子真实exit0后，GPU所有类型上下文均空闲才顺序运行新的 `qualify_sync_reduction_v3.py` / `qualify_training_sync_v3.py`。固定已完成22/fold5及原测量/96次反向/零真实优化步骤预算不变，12次stub另列。若仍有已确认的QQ图形对象（15304、parent34776、birth134360680926134212、QQNT/QQ.exe及 `--type=gpu-process` 旁证）或纯G占用，记录零模型延期，清理核实后恢复同一调度器一次；其他未知C/C+G仍须核验，不能启动竞争者或控制无关程序。严格测量门不因QQ放宽。期间约2秒只读GPU/子树采样，出现外部上下文或观测错误时不作干净性能结论，采样间隙及观测开销披露。报告code0、排队及守护启用均不表示同步优化已集成，`actual_sync_optimization_applied=false` 仍保留。



V3保留独占lease、精确动态出生身份、原owner/attempt字节快照、hold/drain/probe区间及独立终态。Q5父attempt的elapsed包含挂起与可选测量间隙，不算纯训练或与F3/probe主账重复相加。两个真实engineering主账由cost6收集，新控制证据还须独立成本companion及最终bundle绑定；未完成区间保持未知。用户暂停时先写原双严格marker及V3的user_pause.json以撤销新probe/恢复，已经挂起的dispatcher不可再次叠加；旧V1/V2固定身份代码不能复用。清理、原生控制返回或资格CPU孤儿尚未核实，不盲恢复或开启第二GPU；V3完成后只恢复当前Q5，不终止它、不创建新的正式队列或伪造natural handoff。



恢复的CSBrain33/fold4现已实际完成120轮、480次优化更新/3240次反向，选中第90轮，因validation_patience停止且没有延长，总数254/360。真实CPU5报告 `completed_finetune_checkpoint_audit_v5_1791600085874324600.json`（SHA0e9435ada353b182ae07aa06d448c82e7f00758e2771d4cce6908bb2e7fb28bb）核验29次已完成FT/87份CPU检查点，零错误，包括首个实际完整的失去F3进程后磁盘恢复案例。旧未刷新观察与下界成本仍保留未知字段；不补零或重复加4/27。GPU孩子已继续33/fold5，十五次仍未全完成；V3正常等待这一真实边界。



新增只读 `sync_boundary_v3_cost_companion_v1.py` 已封存（SHAe2eb75c9b436c7d083a6446a92699d85c7261ce84db743141f1c15cfea0d3744），28项标准库及独立复跑通过，真实仅运行pending门，零模型/GPU/检查点/评价。旧finalizer5的通配会收V3 inspection/boundary/deferred及两个probe主账，但不收新control/terminal/snapshot；新companion在真实V3终态后生成独立JSON与中文成本附页补齐。全计划最终cost和wide6的360/135405/30项已实际完成后，用同脚本 `--companion <verified JSON> --final-cost <final cost JSON> --final-acceptance <wide6 JSON>` 生成不可变最终bundle，严格核每个probe主账只计一次，父attempt只能作supplemental，hold/drain/probe重叠范围不加入formal。这项准备及pending不是测量完成或最终交付。独立审查为 `independent_sync_boundary_v3_cost_companion_review_1791600012640312600.json`（SHA97edce262b3ffe5d0beb81f602ae672716c064c3d1ccba8d6812046bd0572fb6）。



CSBrain全十五次现已真实完成，累计255/360；原两个训练孩子正常exit0，V3核验完整90项产物后因已确认QQ图形占用记录零模型延期，没有启动两个同步探针。不可变终态为 `sync_boundary_v3/same_dispatcher_resume_complete_1791600567114286900.json`，延期证据SHA484ec27d1cfdd6a2a72bdb5fe1360cadaec7193a0a953ffabb5e47a6f219aa98。同一dispatcher45980只恢复一次，已经进入REVE正式微调；旧V3 guardian owner现在为历史证据，不能重新arm、重复resume或用旧held-owner的live/control验证判断当前队列。实际挂起1166.2466111秒中1163.7410983秒等待训练孩子退出，GPU训练始终继续，两项范围嵌套，不能当额外GPU训练暂停或追加成本。



真实CSBrain15阶段审查 `CSBrain_full15_JSON_stage_review_v3_1791601874868214000.json`（SHAfb18ae903abd0f520a1a4b30923c248eda7b6592d7f5d3297589ed4f50cdfe8a）绑定CPU5的30/90证明a461及同级复核2ca1ce1c/独立成本2f6e6a25。所选九项来自末层微调、六项来自head热身，十五末态均有20个encoder与6个head参数更新、273项冻结；持久5948次更新/39660次反向，去重物理下界5952/39689，只把旧停机0/2与丢失F3的4/27分别保留而不再重复相加。39项唯一formal尝试中22项零调用复用，五折按注册延长、最高200轮；已知整次formal尝试40027.39519秒与metadata39987.27774秒嵌套不加，两个历史尝试的完整耗时和丢失尾仍未知，工程11.509404秒/2次反向/0更新另列。该阶段不是全135微调或360完成。



REVE seed11/fold1也已实际结束：130轮、520次更新/3510次反向，按原120轮条件延长后在best100+30满足耐心停止，不能把延长说成跑满200。累计正式拟合256/360；第2折继续。最新真实CPU5 `completed_finetune_checkpoint_audit_v5_1791602586088653500.json`（SHAff98523b7cc17dacc273b3c69aa27746b2b105477182aae69dd92d25d6a40a1f）核验31次已完成FT/93个CPU检查点，零错误，包括首个REVE实际完成折。架构切换后的不同速度不是同步优化已提速的证据，当前诊断仍chunk8，sync未应用。



V3真实成本companion现已verified：`sync_boundary_v3_cost_companion_1791601316065099800.json`（SHA9090421f81639c495b81d3ae33f99b6500642f9311edc8ffc252bd58b79b50a3）及 `同步资格控制成本附页_1791601316065099800.md`（SHA8cf8c014aa4cc5f37b99e0688888418e34a87e669f2a228f09e173ae1e3dead8），同级review09c44c70核实0probe/0主账/0资格报告、同父一次恢复及重叠范围；没有最终bundle，勿重复collect。最终全计划验收后用原只读bundle入口绑定成本/验收/companion/附页，另在交付manifest绑定以下勘误。



I/O范围已更正且旧证据原字节保留：历史L/X验证经registry._condition_lock间接读取官方CSBrain.pth作SHA哈希，不能称为绝对仅JSON/source读取；不是Torch反序列化、模型构建或GPU工作，未读取raw/test概率。调用次数、字节数和耗时未仪表化仍为null，CPU90/93次实际检查点Tensor加载口径不变。勘误为 `historical_binary_hash_scope_erratum_1791601956592773300.json`（SHAc7542365ed904799da58242605a3d46dabfd164ec00cdb18a409887405279038）、`independent_closure_review_binary_hash_scope_erratum_1791602447024092800.json`（SHA462f2f3beea08b79640c4d8f1a0889cb02bc29fe9579dcb667232d3bb79b0962）、`sync_control_companion_read_scope_erratum_1791602404565633400.json`（SHAbc114598e11c57bb2cc5f09115c764054b6709932b9b3fa75b1f125edd8a2461）。最后一项更正909042顶层checkpoint_reads=0的过宽常量；其15个完成训练checkpoint窄范围未打开仍正确，成本数字不变。新的CSB阶段helper没有调用X/L或读取binary；未来原companion bundle路径本身仅读JSON/py/md，不需重跑collect或改原e2eb源码。



REVE seed11/fold2已完成首个本架构延长预算上限案例：原26/3/8人员角色、末批8/8/8/2保持，epoch120最佳120且最近20轮验证CE均值比此前20轮改善5.26030391246%，按原至少1%条件延长一次至200；最终在epoch200选择epoch200并因epoch_cap停止，head800/bone760/backward5200。选中与终态各6个encoder和6个head参数已更新、134项冻结，二者encoder SHA相等但更新证据分别保留。实际独立JSON边界审核 `REVE_first_cap200_selected_terminal_edge_review_1791604332035001700.json`（SHAf6ace5f422263b9826fc98987fa1bce16744306907d04a79dd1d2dc345c9cf81）绑定全部200轮原采样/LR/停止前缀及新CPU5证明，仅此1fit，不代表REVE15已完。



当前总数257/360，第3折继续；CPU5 `completed_finetune_checkpoint_audit_v5_1791604280247484800.json`（SHA5cfc697d45a1eeb7b63812faff20947e66de379768e125904f743460cb7b0d13）真实32fit/96次CPU读取，零错误。新的边界审核只AST提取原三个数学函数、读取17份JSON/.py，未调用X/L或额外读取binary、Tensor、模型、GPU、process、raw或test。原预算与选模没有调整，不能把更换架构后的不同速度说成同步优化已经应用。



REVE seed11/fold3已实际完成原最低40轮，160次head/120次backbone更新、1040次反向，best3且未延长，按validation_patience停止。第3轮所选仍为head热身状态，不能用第40轮的encoder更新冒称选中模型更新，也不改原CE选模或最低训练规则。当前258/360，第4折健康继续；真实CPU5 `completed_finetune_checkpoint_audit_v5_1791605194620123300.json`（SHAd3a15a4c7543e24d2d462ff943153eb79a9602c869eb0db5ee07d18e6a65544a）已核验33份完成FT/99次CPU加载，零错误。该部分记录不是全135/360证书。



REVE的seed11五折现已全部实际结束，累计260/360，seed22继续。真实CPU5 `completed_finetune_checkpoint_audit_v5_1791606459185020500.json`（SHA7dc8c0fb82fd60b218e41889bc116f145f24bad84ac815b3d59b33e1adbed72c）核验35fit/105次CPU读取，零错误。首种子阶段 `REVE_first_seed5_passed_1791606547152873800.json`（SHAafbf52aa09aa1161525b80e97088d912ff170decc5e7fc13b953b2876311338b）绑定该真实CPU证明及原五折角色；完成轮次130/200/40/119/92、best100/200/3/89/62，所选4项来自末层、1项来自head热身，所有终态更新6个encoder与6个head、134项冻结。两折按原条件延长，119轮停止不进入120轮扩展分支。



本seed5的5个唯一F3尝试均通过逐次强公式核验，持久与已记录物理计数相同：2324次逻辑更新/15447次反向；head2324与bone2124为组内子集，不相加。没有本批zero-call reuse、alias或未知尝试elapsed；本批formal5514.1150665秒与metadata5482.0752354秒嵌套不加，工程24.1807593秒/2次反向/0更新单列，峰值433029632字节。历史全计划精确总时长仍未知。新review_reve_first_seed5_v1.py（SHAb5b32889a76bef07b96d43d347f48e6b6a705a768f187c42a56d5fab5a9cfdcc）仅读JSON/source和原policy数学函数，不执行X/L、Tensor/binary、GPU、process或test读取；其8项标准库护栏及真实pending4/5均保留。本报告只确认REVE5/15及FT35/135，不代表全部360完成。



REVE seed11五折的独立复核已通过：`independent_REVE_first_seed5_review_1791606716620780400.json`（SHAd33c5e426c178983f522f95123f568cd41de6030c7e4a93107430c5c695c9c43）绑定真实passed afbf52与CPU35/105 7dc8、独立复跑8项护栏；全部581轮原policy/LR/sampler及5条单次物理公式一致。两次延长的相对CE改善分别10.2289138%与5.2603039%，均满足原至少1%条件；所选和末态分别核验。该复核未读取二进制、运行X/L、模型、GPU或process，也不把同seed11初始化一致推断为已核跨seed性质。当前继续seed22，其余两种子尚未完成，外层测试仍锁定。



2026-10-10 用户再次明确“暂停”（含重复请求）后，本轮已暂停Q5所有实际相关对象及eeg自动接续，未执行重复挂起、终止、恢复或新实验。真实暂停证明 `pause_events/q5_user_pause_1791607365360/pause_complete.json`（SHAfcbd717850d8cee8bc57ceb0594cf8dab482d10363f618f8d5cae7406bc37b3b）保存261/360、当前REVE22/fold2及全部initial/latest/best检查点、曲线、progress、owner/attempt原字节与缓冲观察。native完整出生对象45980/31128/16096/38860各一次NTSTATUS0，复核67/67线程全部Suspended且无其他实验/CPU审计/资格Python工作进程，进程内存当前保留。暂停时第12轮已开始第45次逻辑更新、已完成44次、last_durable_epoch11；进行中调用与未刷新尾未来丢失时保持unknown，不以暂停补成零。第36个已完成FT尚未新增CPU证明，恢复后再核，旧35/105证明仍仅其原范围。



三处严格user_pause.json均paused=true，eeg状态PAUSED，既有目标记录亦paused；明确新“继续”前不能删标记、自动唤醒、运行stage/评价或恢复。新暂停入口pause_current_q5_20261010.ps1只暂停本次动态核对对象，永久user_pause_q5_owner.json排他；旧固定Q4入口不能复用。再次“继续”必须先核四个完整creation100ns及真实owner、Native suspend started/returned记录：全部各已挂起一次，不盲目多次NtResume；正常内存恢复需要新受审精确一次恢复并归档intent/控制标记。如果任一对象已失去，先保存原字节与本暂停Proof，核完整CPU checkpoint，再以新版本磁盘恢复、保留未知物理尾/旧耗时及额外工作，不能重跑此前一次性recover入口或伪造natural handoff。暂停间隙不计训练。



本次暂停最后复核在 `pause_events/q5_user_pause_1791607365360/final_native_verification.json`：12:47仍为四个同birth对象67/67线程Suspended，无额外实验/CPU资格Python。先前一次枚举比较谓词拒绝后改为显式ThreadState/WaitReason字符串读取核实，没有额外NtSuspend/NtResume。缓冲F3已返回44次更新/286次反向，最近完整checkpoint11/44/286；progress第12轮started45仅表示开始，之后未刷新尾仍unknown，不能说额外精确0。如以后内存恢复同一进程，原perf_counter及metadata/queue wall时长包含暂停，必须另有真实暂停/恢复计时范围及正确成本口径；未仪表化的精确active秒数保留未知，过夜暂停不能计训练。当前所有控制保持paused，暂停管理结束后不运行新的实验/审计/评价。



2026-10-10 用户再次明确“继续”后，在15:37:54实际恢复原Q5内存现场。新独立审查过的控制入口 `resume_paused_q5_20261010_v1.ps1`（SHA1f227764bc5b82351067319d0815e5d5926bb80baf3a7446b2ff55f761684761）先以原owner SHA、全部原始birth、父子链、image/command及67个原线程核验，再按GPU31128→launcher16096→dispatcher45980→外launcher38860各一次NtResumeProcess；四次均返回NTSTATUS0，后核没有Suspended线程。首次只读观察中的新增EventPairLow线程未作任何单独控制，正式preflight时已不存在，来源未测；不据此重挂起或多次恢复。原队列及F3/source80没有重启、热改或重放拟合，也没有新process-loss closure。



实际恢复证明在 `pause_events/q5_memory_resume_1791617874108/native_resume_complete.json`（SHA87dc54eb1efc9e95aefa9fc9ce431e9294a628fb404e17151516c7ae4b257c78），逐对象started/returned及当时wall/QPC/frequency、完整preflight、原控制字节归档均保留。同目录 `resume_controls_released.json`（SHA4610f9117366c7e8a08e8ac8e472584c19f38869207a9dcd62414c7ced8d5ab6）记录三处严格暂停标记与user_pause_q5_owner.json核原字节归档后撤销。原暂停目录的 `resume_intent.json`永久排他，不再次执行该入口或盲重复NtResume。旧G3已结束，不重新启用旧守护；后续暂停只能按届时真实owner和完整birth处理。



恢复起点261/360，FT36/135；实际REVE22/fold2已从暂停epoch12推进到epoch18（72次已完成更新/468次反向），不是排队状态。eeg自动接续已经重新ACTIVE；既有目标UI记录仍由用户管理，不将其状态当作训练状态。旧CPU35/105只保留原核验范围，本轮对实际新增完成任务补CPU5审计；全360及真实135/405前测试门继续冻结。



同一F3内存恢复保留原计时器，REVE22/fold2原F3 attempt `f_1791607293628173200_cafa8752`、最终metadata及Q5父REVE15 attempt的原始seconds会包含本次暂停。暂停边界没有QPC，逐对象恢复记录的QPC只约束恢复调用范围，不能与旧wall时间混减；精确pause_monotonic及active_training_elapsed保持null，原inclusive秒数和44/286快照原字节不改、不重复加。暂停时间不计训练；需新增独立成本附页识别此范围并在最终已知有效训练汇总中排除该inclusive timer，不能宣称旧cost6自动处理了本次暂停。其余拟合独立新计时与原数学继续不变。



本次实际恢复独立复核 `independent_q5_memory_resume_review_1791618064344382300.json`（SHAab6666acd982cc1c08b4a1c52094b6b8574ab2ab9146657385ed410d1ddceedf）已通过，复核来源 `review_q5_memory_resume_20261010_v1.py`（SHAef7464e3d987ab4cf56ce9405ae93f3050546c6e847e3d070aecd83649e6a224）仅读不可变JSON/PS1/source；八份JSON暂停快照逐字节哈希核验，二份.pt仅携原descriptor而未打开或重哈希。各次恢复事件、67原线程与4控制归档一致；此前新增线程的来源未测如实保留。



恢复后新增真实CPU5审计 `completed_finetune_checkpoint_audit_v5_1791618071585700100.json`（SHA273c849ecafba4821afcbeca442931644a3e23bc4127737fcfdad5b4416e273d）核验36个实际完成FT/108次CPU加载，零错误、零暂缓。仍不是135/405或全360证明；模型构建/GPU操作/外层概率读取均为0。真实同REVE22/fold2已推进到epoch53（212updates/1378back、durable52），后续依真实最新产物记录，完成数未变不重复CPU审计。



原进程暂停成本附页已实际准备并独立核验：`pause_memory_resume_cost_companion_v1.py`（SHAc1709cfc9c2f96d810215bbd3d7960cd5feb7d2382a2a5a36c4570bb74d92ca4）19项标准库护栏通过，实际报告 `pause_memory_resume_cost_companion_1791618162504822400.json`（SHAde1c56a746687b72e6b15b924de2fa3b5f98a8b4ee420033a3ea98e2c0bf6880）及 `原进程暂停恢复成本附页_1791618162504822400.md`（SHAc05ec44a88ea09e3fcb64bbaee9d3b9e04e75b8304da1bea818d8b43a6dcf6a9），独立审查 `independent_pause_memory_resume_cost_review_1791618234453009800.json`（SHAb5277483b8d1c780c5da75a260b6485d2aadc1c55bed3e22f6e82bc31b8dab1b）。只读白名单JSON/source，没有旧L/X/R执行、二进制、Tensor、GPU/process或最终成本收集。



该报告当前为真实 `verified_pause_scope_with_terminal_timers_pending`，一条F3正式尝试及metadata/父调度器两个嵌套计时均尚未终态；active_known贡献和精确暂停/有效训练秒均null，仅一条正式fit未知，不当作三条fit或零耗时。必须在这三条真实计时均终态后重新生成附页，不能把这份pending报告充当最终结论。旧cost6保留原inclusive记录；本次暂停最终需独立来源绑定的交付门一起核真实finalcost6、wide6、语义审查、旧G3最终bundle及此附页，强化R28成本/R29控制证据再推导R01/R30；仅旧wide6通过不足以宣称本次暂停已被排除，准备/护栏通过亦不代表计划完成。



新增最终交付门 `audit_memory_pause_final_delivery_v1.py`（SHA25c1d3b4ae6ca9fbb29c7bf86326fd3b920f63f38dfba4b67352d24600a7211a）已封存；20项标准库护栏及独立复核通过，准备证明 `memory_pause_final_delivery_gate_preparation_1791619045949321200.json`（SHAf0487116aae432d83c279420ca485e4840f35cf1f3a6ae7fc9814ed1f1becb8b），独立审查 `independent_memory_pause_final_delivery_gate_review_1791618995345760400.json`（SHA676beacafdc10fb63232dda130cee2c03331cb4081e71519b4d777e9fa6b2c37）。真实无参分支输出 `memory_pause_final_delivery_pending_1791619027426215600.json`（SHA44533e68113f9a1f820ec47921ef5956b92bc061589cec78fdeacc74b3ef4ea8），passed/objective_satisfied/full_plan_completed全为false；本轮没有实际最终验收、模型、二进制、GPU、process或旧项目验证器执行。



最终仅在真实360/135FT/405CPU、Q5/E3评价、12补测、finalcost6、独立语义审查、wide6全部30项及原G3最终bundle完成后，刷新c1709暂停成本附页至三个实际producer计时均terminal，再运行新门。六份参数为 `--final-cost <实际finalcost6> --wide <实际wide6> --semantic-review <相同最终语义审查> --g3-bundle <相同cost/wide的G3最终bundle> --pause-companion <新的terminal c1709报告> --resume-review <independent_q5_memory_resume_review_1791618064344382300.json> --require-complete`。当前de1c pending附页不能通过；新门强化R28/R29后才派生R01/R30，缺输入或pending的严格入口exit2。



R28新输出只对category=formal_training_attempt且role=primary_observed_invocation的原记录过滤求和，排除本次F3原inclusive计时，保留其余unknown；没有已知记录时子和null。字段 `known_recorded_formal_seconds_excluding_affected_native_pause_timer`仅为这部分原记录的已知子和，不是全计划精确有效训练总时长，metadata及父计时不相加。本门不改原cost/wide或source80字节。最终交付manifest需一并绑定新门、terminal暂停附页/中文报告、两份独立恢复/成本审查、旧G3bundle及三份I/O勘误；所有实际交付及新门全部通过后才暂停自动接续。



本轮最后实际检查：同原Q5 owner45980/child16096及GPU31128完整birth均未换，recovery登记/source/closure三键仍真实绑定。REVE22/fold2已经按原上限200轮实际完成，总262/360、FT37/135；当前CPU5证明仍36/108，新增完成应在下次补核，不能把37完成误称37已CPU核验。该F3及metadata终态属于本次暂停影响范围，最终elapsed需保留inclusive/activeunknown口径；Q5父REVE15未终态，c1709附页仍待全部三计时终态后刷新，不重复生成pending或提前final。



## 实时记录



- 最新核查（2026-10-10 20:50）：登记281/360、FT56/135；EEGPT/CSBrain/REVE各15/15，LUNA11/15。LUNA seed22五折epochs200/120/40/40/40、best197/120/3/9/9，仅f1延长，f3–f5合法保留headwarm；新33/f1完成200轮/best199/epoch_cap/已延长，selected/terminal17encoder+6head，原单F3完整收尾800updates/5400back。现33/f2实际epoch68/durable67、272updates/1768back。dispatcher45980及launcher25396（birth134360988331542432）、GPU Python50296（birth134360988331626948）父子链、source5及恢复三键一致；系统低功耗后同原对象继续，未重启、未补操作、未改电源设置或增加GPU探测。动态身份按实际owner重核，作者freqs1项按原审计排除。

- 最新真实CPU5证明：`completed_finetune_checkpoint_audit_v5_1791636590572320100.json`（SHA147ab8d5968a4227f1aba436e5ec139b10b423335d9f45be2f5600d6445c7164），56个完成FT/168次CPU加载、零错误与零暂缓，模型构建/GPU操作/外层概率读取0；仍为partial_verified，绝非135个FT/405加载或全360证明。前一份55/165报告 `completed_finetune_checkpoint_audit_v5_1791635187184450200.json`（SHAafd12763a3de82993fbc206ccacea053671e78142298e156269916116554022c）受系统低功耗影响而延迟返回，最终正常exit0，未重复启动；其观察用时不加formal训练。计数未变不重复有效审计；全360测试门继续冻结，未评价或运行资源补测/最终成本/最终验收。

- REVE15阶段不可变JSON汇总 `independent_reve_full15_json_stage_bound_1791626326483.json`（SHA10d9459f019213675fbf0fc391b68b9757d38aacff06c4885ff4f82e76943793）固定绑定CPU5报告 `completed_finetune_checkpoint_audit_v5_1791626079722213500.json`（SHA221a1bc7291a009e6ea0e06d1f01b850596e36f698ba6befe27a514c9d2f56ee，45FT/135CPU），不随最新CPU报告滚动改变，并绑定下述暂停成本终态；1836轮、7344updates/48852back，head7344和bone6744为子集不相加。所选12末层微调/3headwarm（11/f3、22/f3、33/f3），15终态均6encoder+6head更新/134冻结，7延长、3上限/12耐心停止。15单F3尝试均complete/graceful/exact、started=returned，与持久末态一致；仅读JSON声明并复用实际CPU证明，没有额外Tensor/GPU/test读取。本阶段不作全计划完成或效果比较结论。

- 本次原进程暂停成本已实际终态：`pause_memory_resume_cost_companion_1791625875034611700.json`（SHA60217de0cb993be4b43bfe28cb4a0f79f9f5cfe314bf62249fd002b764a7b7db）及 `原进程暂停恢复成本附页_1791625875034611700.md`（SHAb4f294d45c1ad19727e5e47719909c4e7c064758ad11c53a0d434e6729b58bbc），独立实际审查 `independent_actual_terminal_memory_pause_cost_review_1791626127628607600.json`（SHA9cea6711095948febe08de39e83b138c657a5830ba6dc973559652ebc7d2442b）。状态verified_pause_scope_and_recorded_terminal_timers，三原producer及source/input SHA均真绑定，原F3含暂停11874.611828200053秒、nested metadata11869.32014909992、nested父24571.839099100092秒保留不相加；800updates/5200back已包含暂停前44/286，不再加。精确active/pause和affected active_known贡献仍null，只有1 formal+2nested；旧de1c pending与旧source原字节保留。当前final_plan_certificate仍false，最终需绑定实际cost6/wide6/G3bundle/新交付门。三producer及封存源未变时复用60217终态报告，不重复collect；禁止用旧de1c pending作最终输入。

- 系统低功耗事件真实证据：`system_power_event_window_1791635064778.json`（SHA8b55ffd71637b10b583daaadb1aef2f2263f10ec9e94b1c2c99194c3439d8670）与 `independent_system_sleep_overlap_review_1791635154808605000.json`（SHA29727e34ca640854af6e6594677da32a0495797dc490737193affbccda1c2544）。Kernel-Power506/507，Boot218/V2=253在19:30:15.1319533进入、20:21:31.4372363退出；非零active/DRIPS与SleepEntered同时存在，不能把整段认定0训练或用SleepDuration/墙钟精确扣除。20:22:02另506/V2=256截至查询20:23未配对，仅表示窗口内未解决，不推断当前仍睡眠。只读原生观察 `monitoring/system_standby/native_observation_1791635509483.json`（SHAc2b499e646c93698c4baee377be514104b0558b382d0153a2e5b19410a38c83c）证明20:31四原对象同birth/alive/0Suspended、进度142；过去67→81仅保留先前工具观察，未伪造过去native快照。owner原字节快照 `owner_exact_1791635509483.json`（SHA83dced47231f58bd28ef41c4f00d106cc36331c1b5d86851710865eb52eb0791）及link `owner_snapshot_link_1791635509483.json`（SHAadcb75e61501f9516fd54c1fb57fb20773ba4d5f5326355286a7716cd006fdcf）在同目录绑定观察；以后勿要求live owner/progress继续旧SHA。

- 新低功耗成本附页 `system_standby_cost_companion_v1.py`（SHA43495b71d717ce764bfa2ca25d6e6a46fb30e4a3379d42b432a43b8e62326a33）21标准库护栏及独立审查 `independent_system_standby_cost_companion_review_1791635830383264500.json`（SHA7931825f87372cc230b67426c64ef267ddac2a1b3038a247e4b279750e3d4847）通过。实际初报 `system_standby_cost_companion_1791635786956685400.json`（SHAebcb72a78328ad73c10a131475052b05d66bc841ac1b9e71b441c3d82ab4b6e9）与 `系统低功耗重叠成本附页_1791635786956685400.md`（SHA9777fd08978a0932c5568f3ef0355b908473ff50525325220e765e1d9e5ad479）为真实pending；LUNA33/f1原attempt `f_1791631397731004200_7e6d59cd/attempt.json`及metadata目前终态，原F3含低功耗4712.4451164999045秒、metadata4710.7238700001035秒仅raw不相加、有效秒未知；父LUNA15仍pending，等三timer终态再一次生成新terminal附页。未知active/standby/contribution为null，1formal+2nested不加CPUobserver、不造lossclosure或额外更新；失败原attempt不能借重试metadata。该固定scope不代表其他未来低功耗事件，final_scope_closure_required=true/ready=false；不重复pendingcollect。

- 新最终补充门 `audit_system_standby_final_delivery_v1.py`（SHA161852e96efd16cb0a91e5c62da268d3d7c682c447ab11da2c22cc439845e528）32标准库护栏/独立review `independent_system_standby_final_delivery_gate_review_1791636166460165100.json`（SHAd85de1567186d70f4c835398989f260c4daec611dbb0d149d088f41c2b3aa613）通过；准备 `system_standby_final_delivery_gate_preparation_1791636206103137400.json`（SHAd27ffb045f63aaf1fb8e4976f46a39b6ddb5b70a5affe0535cfc91d7061dd437），无参pending `system_standby_final_delivery_pending_1791636204689965100.json`（SHA13304c7f68d0f52037a39deb0e3d4e2e1bf5661517c1ee451b07cf43de602844）均非实际最终验收。最终在旧25c完整memory-pause交付实际通过后，四参数 `--memory-pause-delivery <实际旧完整交付> --standby-companion <43495新terminal报告> --event-review <29727事件review> --scope-review <后来真实独立完整范围review> --require-complete`。后者schema=independent_final_system_low_power_scope_review_v1须source/reviewer/samefinalcost-wide-360完成绑定、成功连续原始事件范围覆盖registered_at→finalcost.captured_at、原7XML一致、真实Boot218/V2=256后续507及全部506/507完整配对、无未定期间或额外affected ID/attempt。额外范围/fit须新版本审查，不放宽本固定集合。R28排除REVE与LUNA两受影响F3后仅报其余primary原记录已知子和，非精确active总时长；R29绑定真事件与观察，不把整段低功耗当0操作。缺scope先pending/严格exit2，helper.ready=false不能单凭终态升格。最终manifest须额外绑定本附页、事件/原生观察/snapshot链、新门及完整范围review；当前零实际final/collector/模型/GPU/process动作，原80/602/25c字节不改，所有真实最终门及交付完成才暂停eeg。

- 总登记：`tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/registration.json`。

- 第一批进度：同目录 `paired_replay/training_ledger.json`。

- 后续冻结队列：同目录 `ready_queue/status.json`。

- 已完成 Conformer 日志：同目录 `continuation_queue/attempts/`，旧接续记录保留。

- 精度修复交接：同目录 `continuation_queue_v2/handoff/state.json`；新版队列启动后的进度为 `continuation_queue_v2/status.json`，实际子任务日志在 `continuation_queue_v2/attempts/`。

- 已完成 SSL 日志：同目录 `continuation_queue_v2/attempts/`；v3 历史交接证明 `continuation_queue_v3/handoff_complete.json`，交接前状态与日志为 `continuation_queue_v3/status.json`、`continuation_queue_v3/attempts/`。

- 当前运行与恢复：只读 `continuation_queue_v5/` 的真实owner、status、当前attempt日志、恢复登记及135/405 CPU5预测试门。V3已经记录实际延期并恢复同一Q5，现为历史不可变控制/成本证据，勿重启。Q3/Q4与旧V1/V2守护仅为历史记录，本次没有自然交接证明。

- 每个训练任务：`runs/<condition>/seed_<seed>/fold_<fold>/status.json` 与训练曲线、checkpoint。

- 来源、权重及依赖：同目录 `assets/download_manifest.json`，模块 `candidate_resources.json`。

- 独立核查：`output/analysis/eeg_all_candidates_v1_20261008/`。



## 运行入口



在工作区根目录使用已有训练环境查看实时登记状态：



```powershell

& 'S00/.venv/Scripts/python.exe' -B -m tridep_ccml.eeg_all_candidates.registry status

```



当前真实Q5/F3健康训练已由独立磁盘恢复入口启动，不要重新启动。原 `continuation_queue_v5 --resume` 的自然交接入口仍保留原字节，本次不存在它要求的handoff_complete.json，不能伪造该证明或调用旧守护。一次性恢复入口也不能盲重试。Q5复用已完成工程并如实延期可选性能测量，不可再重启旧v1/v2/v3/v4检查入口覆盖历史报告。失效占用记录必须先保存原字节并以确切进程证据证明已无运行实例，再以新受审恢复处理。



GPU采用一个实验训练队列。历史v1/v2 owner保留，Q3及已失效Q4 owner已归档；当前Q5实际继续原矩阵，不同时运行两个实验CUDA工作进程。其他程序的图形上下文不被控制，严格性能对照会延期而不阻塞必做训练。EEGMamba保留隔离Linux官方Mamba2环境，其余为原Windows环境；异常保留状态、未知成本及原字节，核对完整恢复状态后使用新版本续训。待工程检查、失败、blocked或仅排队均不代表拟合完成。



当前声明仅为计划已开始实施；性能是否改善等待全部预定训练完成后的统一评价。


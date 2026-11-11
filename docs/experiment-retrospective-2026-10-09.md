# 训练与诊断复盘（2026-09-29 ～ 2026-10-09）

## 0 文档性质与范围

- **复盘对象**：`docs/` 根目录 73 份带日期的训练日志与诊断报告，以及 `docs/superpowers/plans/` 下 50 份执行计划与 `-progress` 执行记录。不新增任何实验，不产生新数值。
- **数值来源**：全部数值转录自各 `-results` / `-progress` 文档，并已按条目回到原文抽查命中。对外引用前请按第 10 节文件名复核原始表格，因为同名指标在不同轮次存在多套口径（见第 6 节）。
- **本文的立场**：把分散在 120 余份文档里的、跨轮才可见的经验抽出来固化，并显式标出归档内部相互矛盾之处。
- **口径声明**：本阶段所有结果均为 TRAIN 内部开发折、单 seed、`formal_go=false`；上游模块已见过相关 TRAIN 数据，任何数字都不构成未见受试者验证。

## 1 总体判断

1. 这两周的主要产出**不是指标提升，而是根因排除**。所有 Go/No-Go 均为 No-Go，但把「Verify 阈值 → OR 融合 → 候选未成熟 → Shadow 弱信号 → 初始化与 maturity gate → 读出容量 → 语义-物理互补 → 时间编码 → 残差融合」这条根因候选链逐层关闭了。
2. **瓶颈定位收敛到两处**：(a) 雷达 token 与冻结语言 reader 的**语义接口从未对齐**（单事件零步宏平均 18.98% → CE 500 步 76.14%，旧 QA 参考 80.47%）；(b) Proposal 分支的**低误报预算下选择性不足**，而不是候选缺失（旧 Radar 零阈值合法峰覆盖 66.19% 即饱和，新臂覆盖 96.59%–99.43% 却选不出真峰）。
3. **多事件完整顺序是硬边界**：雷达输入、文字 oracle、事件编号前缀、order/time 位置编码下均为 0/24，从未突破。「模型能 100% 读文字列表」不等于「能解释雷达前缀」。
4. **上游表示不存在信息断崖**：冻结层可解性审计显示 2.16 s 处 F 0.788 / Ê 0.743 / E 0.736 / z 0.705，明显高于同点距离型读出（0.38–0.44），逐层成对下降仅 F→H4 与 E→z 两处。因此「换编码器/重训 Encoder」目前**缺乏证据支持**，不应作为下一步。
5. **本阶段真正可交付的资产是那套实验纪律**（预注册门槛、0 步只读基线、matched 单变量、失败证据不覆盖、未经授权不自动训练）。它已被证明能捕获真实缺陷：provenance 覆写、字典序把 100 排在 11 前、核验脚本把来源类别当提问类别等，都是靠这套纪律暴露的。

## 2 实验线地图

| 线 | 时间 | 核心问题 | 结局 |
|---|---|---|---|
| 上游审计与定位 | 09-29 ～ 10-01 | 事件切分差（P 23.16% / R 44.90% / F1 0.3056）的根因在哪一层 | 逐层否掉 5 个候选根因，改判为「读出问题」 |
| 冻结层可解性 | 10-01 | 6 层容量匹配线性探针是否存在信息断崖 | 无断崖；gate 非根因；后端 98.63% 候选被确认 |
| 短时序读出 | 10-02 | 拼历史 / 加容量 / 语义-物理互补能否救活读出 | 三条全部否；距离型读出 ≈ 随机 |
| transition-aware | 10-02 ～ 10-04 | 过渡监督能否在不破坏检索的前提下改善决策 | pilot +2.48pp → 复现 +0.48pp → 第三折转负；50 步 +1.60pp 仅作研究基线 |
| stage4 记忆与 QA | 10-04 ～ 10-06 | 分层记忆 + LLM 能否形成可靠问答 | 链跑通、格式改善，可靠性未达标；定位出提示合同 bug |
| v2.5 架构迁移 | 10-06 ～ 10-08 | 新门控/证据/提案能否替换旧链 | 门控迁移 −3.04pp；须先建「复现旧输出坐标」的过渡架构 |
| 提案监督与融合 | 10-08 ～ 10-09 | 低召回是损失、触发器还是融合造成 | 加权、窗口辅助、迟滞、残差融合全部排除 |
| 双源候选池 | 10-09 | 高召回候选空间是否存在 | dev 91.48% 召回 @ 39.92 候选/min，需过滤 74.9% 才达 8 FP/min；负担未验证 |

因果链摘要：短时序读出三线皆败 ⇒ 矛头由「读出」转向「表示监督」⇒ 阻断式评审把方案改写为预注册 ⇒ transition-aware 系列 ⇒ 检索退化 −21.43pp 的定位 ⇒ 目标函数与评估口径错位审计 ⇒ 50 步研究基线；并行地，门控迁移失败催生兼容过渡架构，兼容架构保住检索后才获准训练 Proposal，Proposal 低召回又引出零更新归因与触发器研究。

## 3 必须固化的实验纪律

| # | 规则 | 证据 / 出处 | 违反的后果 |
|---|---|---|---|
| 3.1 | 门槛、指标定义、停止条件在接触分数数组之前锁定并写入 spec/SHA | stage3a-proposal-zero-diagnostic；v2.5-proposal-causal-trigger | 事后取反方向、事后选点，结论不可辩护 |
| 3.2 | 任何训练之前先跑 0 步 / 冻结只读基线与前置门槛 | stage3-verify-candidate-audit；stage3-fork-* | 会误把「未训头退化对照」当强基线（v2.5 零步 0% 召回即此例） |
| 3.3 | 消融一次只改一个变量；对照臂须与旧运行逐张量一致 | stage3a-proposal-loss-ablation（A 臂精确复现） | v2.5 A 阶段同时改初始化+去 maturity+tanh，无法归因 |
| 3.4 | 预算写死，「500 步」= 成功更新数；扩预算须显式传参并进 manifest | 2026-10-03 projection-progress | 曾用改全局 SETTINGS 的方式把 20 步记成 100 步 |
| 3.5 | 禁止按 dev 择优、禁止追加步数、禁止放宽收敛阈值或迭代数「过关」 | stage3-transition-aware-projection-recovery | probe 未收敛须记 undefined 并排除该 checkpoint |
| 3.6 | 失败证据不覆盖：v1 包/日志/半成品保留，修复另开 v2 | gate-migration、reader-fusion、radar-alignment 等 ≥6 例 | 无法区分「改了什么」与「换了什么实验」 |
| 3.7 | 阈值只在 fit 按 FP/min 上限锁定，dev 单次读，且必须同报实际 FP | stage3a-window-peak / stage3ag-residual-fusion | 名义预算 8、实际 7.17 或 6.23，跨臂比较失真 |
| 3.8 | 被保护面（EventFormer、检索向量/排序、旧 QA 权重）须逐张量核验前后完全一致 | stage4-reader-fusion | 「检索无损」被反复误 claim |
| 3.9 | 未达标不自动进入下一步；门槛降级为 `diagnostic-only` 须显式授权 | stage4-radar-semantic-alignment v1→v2 | 门槛形同虚设 |
| 3.10 | 交付判定以归档 SHA 为准；launcher 返回或 `status=completed` 不算完成 | 2026-10-08 window-peak progress | 半途状态被当终态写进结论 |
| 3.11 | 结论边界固定话术：TRAIN 内部开发折 / 单 seed / formal_go=false / 合成数据不作结果 | 各 plan 与 README | 对外夸大，后续无法回溯 |
| 3.12 | 一轮 fresh 只读 review + 一次 fix pass（RED→GREEN），不做二次评审循环 | 2026-10-03/10-09 progress | 评审循环消耗额度且无新信息 |

## 4 负结果台账（已否方案）

| 方案 | 否证数值 | 出处 | 重开条件 |
|---|---|---|---|
| 拼接更早历史（K3/K8） | F −0.0070 / E −0.0052 AUROC；forced age13 F 0.7881 高于任何八段历史 | stage3-frozen-short-horizon-sequential-results | 无 |
| 加大读出容量（MLP32） | 0.6283 < 131 参投影基线 0.6736；最佳关系 Logistic 0.6787 仅 +0.005 | stage3-radar-readout-pilot | 若改的是监督信号而非容量 |
| 距离型几何读出 | cos/L2 only 0.5040 / 0.5399 ≈ 随机 | stage3-frozen-short-horizon-sequential-results | 需保留方向信息才谈 |
| 语义+物理互补证据 | F+E、F+z 均不胜 F；F+E@age7 recall +6.00pp 但 held FPR +1.76pp、AUROC −0.0213 | stage3-semantic-physical-complementarity-results | 需先解决阈值跨 subject 不迁移 |
| OR 融合 | 两头相关 0.6443→0.5611；500 步 @8FP fusion 44.18% < Local 50.00% < Radar 56.39% | v2.5-physical-proposal-results | 与预设「Radar 只作物理证据」相反，暂停 |
| 迟滞触发（hysteresis） | release 更严、不解长平台，fusion dev 20.74–28.55% | v2.5-proposal-causal-trigger-results | 无（长平台需别的手段） |
| score 级残差融合（GRU→Radar head） | fit8 dev：C 56.11%、A 55.26%、B 37.36%（−18.75pp）；Rescue 57 / Harm 189 净 −132；8/8 cell 退 | stage3ag-residual-fusion-results | 只能改为「候选池 + 过滤」两阶段 |
| 以 Combined 读出为主检验 | 主/次 −3.55 / +3.41pp，仅 2 与 4/8 cell 达标 | proposal-readout-matched-results | 保留 State/Delta 作辅助臂（+8.66 / +6.68pp） |
| unweighted Gaussian BCE | Radar@8 −11.51pp（8/8 cell 退）、Local −3.27、OR −1.14 | stage3a-proposal-loss-ablation-results | 加权不是低召回根因，勿再走 |
| 窗口峰辅助 CE | Radar 三预算全降；peak_shape 失败 200→207；平台 P95 9.20→10.08 s | stage3a-window-peak-ablation-results | 无 |
| pairwise 排序损失 | **未执行**（window-peak 计划明确 No hard-ranking term） | stage3a-proposal-supervision-diagnostic-results | 唯一仍开放的假设 |
| 重标 hard-negative 的 Verify | F1@0.72 s 0.4167→0.3885、过分割 1.46→1.78 | stage3-verify-candidate-audit | 无 |
| 强制延长候选（成熟度假设） | 等长 1336 候选余弦 0.5135→0.5093，无成熟拐点 | stage3-forced-horizon-probe | 无 |
| 无学习比 Shadow 对照 | 最大 d(C,S)=0.572、同录音配对 0.562 | stage3-candidate-shadow-diagnostic | 只作起步 |
| 初始化 / maturity gate 2×2 | 四臂 AUROC 0.42–0.51；d(PP,PF) 90 分位 2e-7 | stage3-fork-init-gate-ablation、stage3-fork-layer-localization | gate 对 maturity 不敏感，但随状态大幅变化（0.746→0.228） |
| 简单差分 / 无训练距离当边界信号 | dev AP 0.115–0.118；21 条距离 AUROC 0.429–0.493 < 正例率 13.29% | radar-gru-two-diagnostics-results | 「变化越大越像边界」已被推翻 |
| 事件编号前缀 | 双/四事件存在性 81.6 / 75.0 → 62.2 / 56.2 | stage4-event-boundary-qa-results | 无 |
| order / time 位置编码 | 均不优于 none；完整顺序三组仍 0/24 | stage4-reader-fusion-results | 接口对齐之后才有意义 |
| 固定 0.5·T²、T=2 蒸馏当默认目标 | 47.53% < CE 76.14% | stage4-radar-semantic-alignment-v2-results | 需按任务重新标定系数 |
| 「压缩器是主因」 | 4 个冻结 L1 摘要 token 与 32 个原始 token：存在性 68.8% vs 70.8% | stage4-short-context-qa-results | 无 |
| subject 类别均衡权重 | −0.082 / −0.064pp；原权重 subject 质量本就 49.47 / 54.93 / 55.43% | stage3-transition-aware-subject-balance-results | 失衡假设不成立 |
| 继续追加步数 | 第三折 20 步转负、100 步回落；fit 目标下降不保证 dev 改善 | stage3-transition-aware-budget-results | 50 步为研究基线 |

## 5 当前对模型与数据的判断

- **接口 > 编码 > 容量**：雷达 token 与 reader 的接口在训练前 ≈ 随机（零步宏平均 18.98%、正反平衡 46.97%），500 步 CE 升到 76.14%，说明「可学但未学透」；失败不在时间编码。
- **覆盖 ≠ 选择性**：GRU 支路把零阈值合法峰覆盖从 66.19% 提到 96.16%，但 55.37 候选/min 的假峰密度使其在低 FP 预算下毫无用处。问题的形状是「有候选、选不出」。
- **点级与事件级可以反向**：残差融合 B 的 AP 升到 0.30603、AUROC 0.756，事件 recall 却 −18.75pp；固定 0.6 阈值下「配对覆盖」上升而「实际发出匹配」下降（623→671 vs 126→89）。只看 AP/AUROC 会做出错误决策。
- **TCN 分支的真相是监督路径遗漏**：`frame_hidden` 取自 RD-TCN 之前，`temporal_gate` 在四个导出检查点中仍为 0 ⇒ RD-TCN 对 Radar Token 注入系数为 0。可宣称的只有「已证实的遗漏」，**不可**升级为「TCN 架构无效」或「整个编码器未训练」；直接把 gate 改成 1 会让随机初始化的 RD 分支立刻进输出。
- **阈值不跨 subject 迁移**：目标 5% FPR 下 subject02 实测 14.65–16.47%，subject03/04 仅 2.5–3.8%。macro 平均掩盖风险，任何阈值结论必须逐 subject 报。
- **命令式 GT 的时间基存疑**：所谓「语音命令」实为 0.09 s 帧标签翻转，不是真实发声时刻；未来偏移不能当物理反应延迟。

## 6 引用历史数字的口径陷阱

1. **seed 不独立**：1701 / 3407 前向逐 bit 一致；三个 feature seed 只用于中位数聚合，9 个方向（3 subject × 3 seed）不得当独立重复或称显著。
2. **跨折绝对值不可比**：同一冻结 P、同一 dev 组基线可差 0.118（0.526596 vs 0.644525），因 probe 按各折 fit 重拟合。
3. **同名指标多套值**：terminal F K1 = 0.6794 / 0.6754 / 0.6736（拟合行集与标定不同）；旧 Radar@8 = 56.39 / 57.81 / 56.11（触发器 lookahead 与 NMS、阈值网格不同）。跨轮相减一律禁止。
4. **归一化与口径差异**：加权 vs 非加权的目标质量归一、dev 正例率 13.27% vs 13.29%、Local AP 0.3429 vs 0.3435，都源自有效支持集（前 6 update 是否排除）不同。
5. **「零阈值覆盖」一词两义**：任一合法峰覆盖（66.19 / 96.59 / oracle 98.58%）≠ 合并 episode 后的实际匹配（98.01%）。
6. **「通过容差 ≠ 无损」**：50 步已出现检索准确率 −1.7857pp、动作 macro-F1 −0.9091pp。
7. **「文字门槛」被两种含义使用**：多事件列表 100% 与单事件命名教师 85.71%（从未过 95%）不是同一件事，不能推出「解码器已修好」。
8. **等参数量 ≠ 等信息容量**：matched 读出九组名义参数同为 65,793，但源表示内在秩不同。
9. **paired delta 必须先算再聚合**：分组统计要用逐 feature-seed 的配对差值中位数，不能拿两个中位数相减。
10. **函数名不等于性质**：`balanced_candidate_weights` 归一后加权正类实际只有 17.35%，不是 50%。
11. **不能用末步 CE 判断过拟合**：CE 的 fit 宏平均 69.15% 低于 dev 76.14%，fit 也并未全对。
12. **R@k 是覆盖比例不是召回**，AP 为非插值 step 积分；同一指标存在 0.760989 与 76.10% 两种写法。

## 7 远程训练与核验的工程经验

- 环境：conda env 与 pkgs 全放数据盘，仓库内只留被 ignore 的软链；离线 wheelhouse 装完即删；权重/输出不进仓库。
- 无 git：隔离目录 + 源码快照 SHA + ledger 代替 commit 与 worktree；跨预算/跨权重禁止 `--resume`；新轨迹前 20 步须与旧对照逐张量核对；provenance 比较必须发生在写入之前（曾发现代码会覆写原 manifest）。
- SHA 不许手抄：至少三次因终端换行多抄一位导致预检拒解压（0 更新）。统一用服务器 `sha256sum` + 本地 `Get-FileHash` 直读文件比较。
- 启动与恢复：必须读回 PID + 实际 step 才算启动；SSH 掉线绝不重复起 worker；拿不到回传时结论标 `server-readback`，不得冒充本地独立重算。
- 算力：单卡 2080 Ti、热节流是主瓶颈（82 °C 暂停 / 78 °C 续、SM 58–78%、频率掉至 645 MHz）；batch 8、`num_workers=0`；A/B 臂串行以避显存竞争与 RNG 混淆。
- FP16 反向溢出协议：精确重放前 74 步 loss/梯度范数确认逐项一致 → 只对第 75 批做只反向探测（scale 256/64/16/1）→ 判定为溢出而非数据或损失坏 → 同批减半 scale 重试 ≤12 次、恢复 RNG、不 step、不推进采样；重试计数单独上报；诊断权重与 smoke 权重不得用于正式初始化。
- 数值容差：FP32 下 batch/chunk 等价差 1.43e-6 ⇒ 数学测试改 FP64，训练仍 FP32；GPU/CPU 读回差 2.5e-7～9.5e-7 属正常；float32 BLAS 批大小引入 4.77e-7 舍入 ⇒ feature 容差 2e-6。
- 依赖与既有失败：缺 h5py 时只排除相关测试且不得称「全仓库通过」；sklearn 不可用则删运行时依赖、用 NumPy 对齐到 2e-16，而不是装包；无关后端（Gemma 两项 `None_inv_freq`）须用未修改的旧代码复现后再裁定不动。
- 核验脚本自身曾误导结论：把来源类别当提问类别、`torch.median` 取下中位数、经验 CDF 造 tie、合成边界期望写 1.0 应为 1.3、文件名字典序把 100 排在 11 前。修核验不重训，且**改测试期望而不是改真实数据**。

## 8 归档内部矛盾与需修正项

| # | 问题 | 建议处置 |
|---|---|---|
| 8.1 | `radar+gru融合方案.md` 用 oracle union 98.58% 论证「互补成立」，而 `stage3ag-residual-fusion-results` 明文「未同时优于 A 和 C 不得宣称互补已验证」 | 该方案文档需按 AG 结论回改，删除互补结论 |
| 8.2 | 同文档中「18–20 候选/min」「过滤 ≥50% 即可」的示例已被双源池实测反驳（dev 39.92/min，需过滤 74.9%） | 更新为实测密度，标注为未验证假设 |
| 8.3 | `radar+gru融合方案.md` 残留「粘贴的文本 (1)」占位符 | 清理为正式表述 |
| 8.4 | `server-dataset-inventory` 建议保留 subject08/12 作未见受试者，实际 subject08 被并入 dev 分组多轮查看 | 在相关文档显式标注「08 的未见价值已消耗」 |
| 8.5 | 文件名日期滞后内容（如 `...-2026-10-06` 实为 10-07 终态；`...-10-07 progress` 含 10-08 数值）；文内「归档未下载」限制被同篇后补的更正推翻 | 引用一律带日期与终态标记 |
| 8.6 | 测试通过数 86 / 96+16 / 110 / 117 属时间演进 | 报告数字须绑定 commit/SHA 与日期 |
| 8.7 | 交叉指针错指：`stage4-text-decoder-progress` 标题指向 `decoder-repair`，后者又把「实际状态」指回前者 | 修链接方向 |
| 8.8 | 门槛字段名 `six_cells` 实为 8 个 cell；预算档 3/5/8 与 3/5/8/10/15/20 并存；「8 cell / 8 条记录 / 4 名受试者」三种说法混用 | 统一命名与档位定义 |
| 8.9 | SUP 建议的 pairwise 排序损失从未执行，但 WIN 结果常被当作「辅助监督无效」的证据 | 在 WIN 文档明确标注该假设仍未检验 |
| 8.10 | 提示改版后不可比（temporal-qa 删去 14 项候选表，14/48 与旧 0/24 不可相减）；「共同起点 80.5%」跨轮复用但 adapter/reader 已换 | 每轮起点须绑定 reader 版本与提示 SHA |

## 9 下一步判据（沿用日志中已写明的门槛）

1. 先做**标签核验**（150 例 onset 人工/proxy 核验），再谈候选可过滤性；门槛：FP 下降 ≥50% 且召回损失 ≤5pp，操作门 FP ≤8/min。
2. 候选池采用**两阶段**（高召回候选 + 过滤），主策略 Combined k6，k 不按 dev 选；池永不复用已失败的 residual B。
3. 单事件接口对齐优先于多事件：先逐条复现旧 297 条单事件、建同 reader 零步雷达基线，再谈位置/时间编码。
4. 只有在 matched 对照仍证明上游不足时，才允许带**检索保留约束**的编码监督；且「同 MD 下 GRU vs TCN」与「加 RD」必须拆成两个变量。
5. 冻结参照与影子候选不得混用：影子候选只重置自身状态、GRU 保持连续，其失败不能证明 Encoder 缺信息。

## 10 来源清单

- 上游审计与定位：`stage3-upstream-audit`、`stage3-verify-candidate-audit`、`stage3-candidate-shadow-diagnostic`、`stage3-forced-horizon-probe`、`stage3-fork-init-gate-ablation`、`stage3-fork-layer-localization`、`stage3-frozen-layer-decodability`
- 读出与互补：`stage3-frozen-short-horizon-sequential(-results)`、`stage3-radar-readout-pilot`、`stage3-semantic-physical-complementarity(-results)`
- transition-aware：`stage3-transition-aware-projection-plan-review`、`-remedies`、`-recovery(-results)`、`-rehearsal-audit`、`-relational(-results)`、`stage3-transition-objective-audit`、`-primary(-results)`、`-primary-replication(-progress/-results)`、`-budget-ablation(-results)`、`-subject-balance(-results)`
- stage4：`stage4-memory-pilot(-results)`、`stage4-chat-qa-pilot(-results)`、`stage4-single-event-qa(-results)`、`stage4-short-context-qa(-results)`、`stage4-event-boundary-qa(-results)`、`stage4-temporal-qa(-progress/-results)`、`stage4-decoder-repair`、`stage4-text-decoder(-progress/-results)`、`stage4-reader-fusion(-progress/-results)`、`stage4-radar-semantic-alignment(-progress/-results/-v2/-v2-results)`
- v2.5：`server-dataset-inventory`、`v2.5-implementation-status`、`v2.5-online-modules`、`v2.5-training-pilot-results`、`v2.5-compatible-evidence(-progress/-results)`、`v2.5-gate-migration-diagnostic-results`、`v2.5-physical-proposal(-progress/-server-readback)`、`v2.5-proposal-zero-diagnostic-results`、`v2.5-proposal-causal-trigger-results`
- 10-08 ～ 10-09：`stage3a-proposal-supervision-diagnostic-results`、`stage3a-proposal-loss-ablation-results`、`stage3a-window-peak-ablation-results`、`stage3ag-residual-fusion-results`、`proposal-readout-matched-results`、`md-gru-transition-signal-results`、`radar-gru-two-diagnostics-results`、`tcn-radar-encoder-inspection`、`radar+gru融合方案`
- 执行记录：`docs/superpowers/plans/2026-09-17-remote-training-bootstrap.md` 起共 50 份 plan/progress；冻结设计见 `docs/superpowers/specs/`；架构口径源头见 `计划/方案书v2.0 → v2.5`；逐轮状态与结论边界见根 `README.md`。

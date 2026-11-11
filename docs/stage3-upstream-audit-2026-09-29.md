# Stage 3 上游信号诊断（2026-09-29）

## 范围与方法

固定原始 `best_verify.pt`（SHA-256 `c359ad2a459e0ee5b9db93e7cc3a4f33ca1391c5099f0608e57202beaca4fbd6`）和原始 validation-only 阈值（SHA-256 `5066d916a0ef0648409a989c89bda564fefae478759a1d9a3f707cd0e32d08ef`），只回放 8 条 validation recording。没有训练、读取 test 或更改 controller 判决。观测钩子核对 Local/Radar 融合分数和 Semantic Difference logit 均能重现原 controller 输出。

新建只读分析代码 `src/continuous_mmwave/parrad_stage3_upstream_audit.py`。远端结果目录：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/upstream-audit-v1`；`upstream-summary.json` 声明 `source_split=validation`、`test_evaluation_allowed=false`。

语义对照每次只替换参考 key、候选 key 或 maturity：在线两 key；双 oracle key；在线参考＋oracle 候选；oracle 参考＋在线候选；双 oracle key＋oracle maturity。oracle 参考取 Proposal 前一 update 的 GT-run-reset key，oracle 候选取当前 update 的 GT-run-reset key。因此它是同录音反事实诊断，不是可直接上线的轨迹。

## Proposal：Radar 头与 OR 融合降低边界排序

validation 有 26,729 个 update，其中 19,884 个处于可发 Proposal 的 Active 状态；这些 update 中，距直接 GT 边界 ≤0.72 秒的比例为 18.13%。仅把这个时间邻近作为诊断标签：

| 分数 | 总体 AUROC | AP | xWR14xx AUROC | xWR68xx AUROC |
|---|---:|---:|---:|---:|
| Local Change | 0.7054 | 0.3343 | 0.7269 | 0.6828 |
| Radar Transition | 0.5601 | 0.2087 | 0.5666 | 0.5540 |
| 当前 OR 融合 Proposal | 0.6842 | 0.2990 | 0.7116 | 0.6563 |

实际 1341 个 Proposal 按发生时两头是否单独越过 0.55 阈值归类：

| 距 GT 边界 | 总数 | 仅 Local 越阈 | 仅 Radar 越阈 | 两者都越阈 | 两者均未越阈但 OR 越阈 |
|---|---:|---:|---:|---:|---:|
| ≤0.72 s | 470 | 211 | 31 | 10 | 218 |
| (0.72, 1.44] s | 366 | 154 | 27 | 6 | 179 |
| >1.44 s | 505 | 100 | 100 | 9 | 296 |

远处的 505 个 Proposal 中，xWR14xx 有 236、xWR68xx 有 269。远处“仅 Radar 越阈”分别为 33、67；近处则为 21、10，68xx 上 Radar 误触发尤其集中。远处另外 296 个是两头各自不足 0.55、但 OR 融合后跨阈。**这些只是已观察状态的单步归因，不是关掉一个头后的闭环结果**：persistence、Candidate 生命周期和后续状态都会变化，不能把 296 或 100 直接当作可消除的误确认数。

## Semantic Difference：换成 oracle key 也不恢复

对每个 Candidate 的首次与最后 scored update 各取一行，以“Proposal 前 GT run 到 Candidate 终点 GT run 是否不同”作候选级代理标签（373/1341，27.8%）。这比上一轮的“Proposal 当下到终点”代理多纳入已在 Proposal update 跨边界的情况，但仍不等于人工裁决的 Candidate 有效性。

| 首次更新的分数 | AUROC | AP |
|---|---:|---:|
| 在线 reference＋在线 candidate | 0.3993 | 0.2279 |
| oracle reference＋oracle candidate，在线 maturity | 0.3795 | 0.2222 |
| 在线 reference＋oracle candidate | 0.3929 | 0.2235 |
| oracle reference＋在线 candidate | 0.3872 | 0.2267 |
| 双 oracle key＋oracle maturity | 0.4054 | 0.2347 |

最后更新时相应 AUROC 为 0.4026、0.3734、0.3830、0.3970、0.3070；换 key 或 maturity 没有产生稳定的正向排序。为排除“首次更新还没真正跨边界”的时序混淆，又对 1341 个首次更新按**当下** oracle reference/current run 是否不同重算：已有 138 个不同 run，在线 AUROC 0.4778，双 oracle key＋在线 maturity 0.4071，双 oracle key＋oracle maturity 0.3503。即使只问当下 GT run 差异，分数方向也未恢复。

对所有 6837 个可比较的 Candidate update，同一时刻的 run 差异比例为 20.64%；在线 `p_diff` AUROC 0.4547，双 oracle key＋在线 maturity 0.3876。逐 update 样本有候选内相关性，这组数只用于方向检查，不作独立样本置信区间。

## 可定位的训练／在线配对分布差异

`SemanticPairDataset` 训练的是四类 oracle key 配对。实际索引审计：train 共 8928 对，其中 `same_class_mature`、`different_nascent`、`different_mature` 各 2232 对，**全部跨受试者且跨会话**；validation 共 2984 对，对应三类各 746 对，**全部跨会话**。只有 `same_event_nascent` 是同一游程内配对。在线 controller 则始终比较同一 recording 内的冻结 Active key 与逐步发展的 Candidate key。现有 SemanticPairDataset 的排序偏好以 `subject_id == anchor.subject_id`、`session_id == anchor.session_id` 为键升序，`False` 被优先选中，因而跨事件训练没有相邻同会话例子。

原训练日志中 Semantic 验证最佳 AUROC 0.7293、AP 0.7757（epoch 2），说明它能做现有**跨会话 oracle 配对任务**；并不证明它能做在线相邻事件的比较。本轮同录音 oracle 反事实仍反向，支持“训练配对／在线任务不一致”是主要可疑机制。它是强证据，但还不能断言是唯一原因：Candidate 由当前 Proposal 选择，GT-run-switch 仍是代理标签，受试者独立性只有 4 个会话。

### 排除 Proposal 选择偏差的相邻 oracle 对照

直接从完整 validation oracle cache 提取 738 个**无 `-100` 间隙的相邻 GT 游程对**，不依赖 controller 是否曾发 Proposal。对每一对固定上一游程成熟 key 为参考，比较新游程 key 与上一游程内 key，并对正负例使用相同 maturity：

| key 阶段 | Semantic Difference AUROC | 成对正例分数高于负例 | key 余弦距离 AUROC | 两传感器 Semantic AUROC |
|---|---:|---:|---:|---:|
| 新生 key 对上一游程初生 key | 0.5283 | 48.5% | 0.0987 | 0.5295 / 0.5266 |
| 成熟 key 对上一游程倒数第二个 key | 0.4916 | 40.9% | 0.9966 | 0.4875 / 0.4956 |

成熟阶段，oracle key 的相邻游程距离与同游程最近 key 距离确实可分，但当前语义头未把这一信息稳定变为高分；新生阶段，简单余弦距离本身也反向，不能据此声称早期 key 已可靠。这是匹配 maturity 的成对对照；成熟负例是同游程最近 key，较容易，故距离 0.9966 **不是**在线边界检测能力。现有语义头在此对照中的概率很低：成熟正／负例中位数约 0.00592／0.000565，新生约 1.52e-6／2.30e-6。跨会话验证 AUROC 0.7293 与同会话相邻对照接近随机的差距，进一步指向语义头的配对任务错位。

## 下一步决策

1. **下一项代码变更应修正 SemanticPairDataset 的配对分布**：在 train 加入同会话相邻 hard pairs，并让正负例 maturity 匹配；先写数据集测试，确认不跨 `-100` 间隙、无 split 泄漏。短训只更新 Semantic Difference，冻结其余模块，并预先固定同会话相邻对与在线候选两项 validation 门槛。不能只用原跨会话配对验证分数选模型。
2. Proposal 侧另做 validation-only 闭环消融（Local-only、Radar-only、当前 OR，固定 Verify 和其余设置），比较一对一边界 F1、false-confirm 与两传感器召回；单步归因不能替代闭环消融。消融仍未授权 test。
3. 在上述两项前，**不要继续训练 Verify、调它的阈值或上 H0/H1 序列决策器**。本轮定位了明确的训练／在线分布错位，但短训是否能修复闭环仍须实测，不能直接宣称成功。

## 验证与溯源

- 本地 `interview` 相关测试 37/37；远端隔离 SEER 相关测试 37/37。远端全套 pytest：227 passed、2 个此前已知的 Gemma adapter 失败；按用户决定未处理。`interview` 本地全套仍因缺少 `h5py` 在收集阶段停止，未安装新依赖。
- 最初 8 条录音回放使用的模块 SHA-256 为 `ee88c3880bc4098b1a61c21db507d82b26c9f2f0517dbeb412dc132647c1026f`；其后仅新增独立的相邻 oracle 对照函数，未修改原回放路径。最终新模块本地／远端 SHA-256 `a28cc184105fa256cac88deba111f0ea404c11b5833b655973e94ac1801cecb3`；新测试 SHA-256 `c8e1af2175cd351157dc02e2238fc98d930ddd496b3bb297240f22df4f3eb5dc`。
- 远端 `physical-updates.jsonl` 26,729 行，SHA-256 `f43eec35c41650306fd78ab5a45b72c09a45269ca68ff894d52fabad5d13f31a`；`semantic-candidates.jsonl` 6837 行，SHA-256 `fe91a502f273aa558459da4bf15584e87f1b5c6a5d724aee801d55b0cf4ab1c1`；`upstream-summary.json` SHA-256 `4570556a615a9ba321e178948e9447f9d975dc8777287bf7332f68fadb6b4de4`。

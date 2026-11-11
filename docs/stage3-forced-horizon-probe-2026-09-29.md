# Stage 3 Candidate 强制延长诊断（2026-09-29）

## 问题与方法

上一轮自然轨迹中，Candidate 超过 1.2 s 仅余 39 个，因此不能直接用后期 AUROC 判断语义 key 的成熟时间。本轮固定原始 `best_verify.pt`、validation-only 阈值和 8 条 validation 录音：在每个自然 Proposal 处复制一条独立诊断分支，冻结同一 Active reference，只在复制分支屏蔽确认、拒绝与 timeout 造成的终止，继续评分至 Proposal 后 12 个 update（2.16 s）。原控制器仍按原规则推进，以便产生原来的后续 Proposal。没有训练、读取 test 或更改正式控制器。

主标签固定为 Proposal 时刻距最近 GT 边界 ≤0.72 s；当下 GT-run switch 另作辅助诊断。两者都是代理标签，不是人工裁决的语义有效性。分数为在线 key 的固定 `p_diff=(1-cos(z_A,z_C))/2`，与原 Semantic Difference 头在同一行比较。

## 一致性与覆盖

- 1,341 个自然 Proposal 均建立强制分支；仅 5 个靠近录音结尾，未满 2.16 s。完整等长候选集为 1,336 个，其中边界附近 469 个。
- 原自然轨迹 26,729 个 update 的决策及 Proposal 分数逐行一致；6,837 条自然 Candidate 语义分数及余弦分数逐行一致。反事实仅在自然终止后增加评分。
- 到 1.26 s 时，自然轨迹仅有 39 个候选，而强制分支有 1,337 个；到 2.16 s 仍有 1,336 个。先前后期稀疏问题已被排除。

## 等长候选集结果

固定同一批 1,336 个候选、同一个 Proposal 邻近标签：

| age | 在线余弦 AUROC | 原语义头 AUROC | 自然仍存活数（全部强制候选） |
|---:|---:|---:|---:|
| 0.18 s | 0.5135 | 0.4572 | 1,341（本行等长子集 1,336） |
| 0.54 s | 0.5022 | 0.4648 | 1,339 |
| 1.08 s | 0.4916 | 0.4694 | 859 |
| 1.26 s | 0.4867 | 0.4711 | 39 |
| 1.44 s | 0.4865 | 0.4704 | 0 |
| 1.80 s | 0.4945 | 0.4719 | 0 |
| 2.16 s | 0.5093 | 0.4784 | 0 |

余弦 AUROC 先降后略回升，2.16 s 仍约为随机排序；xWR14xx 在 0.18/1.08/2.16 s 为 `0.5308/0.4964/0.4905`，xWR68xx 为 `0.4971/0.4889/0.5314`，没有一致的成熟拐点。当下 GT-run switch 的余弦 AUROC 从 `0.5280` 降至 `0.4029`；但该标签随时间变化，且长分支可能跨多次 GT 边界，只用于方向检查。

在线 Candidate key 到 oracle Candidate key 的归一化余弦距离中位数从 0.18 s 的 `0.00589` 降至 2.16 s 的 `0.00126`；在线 Active reference 到 oracle reference 始终约 `0.0007`。这说明 key 与 oracle 的对齐在改善，**不等于**在线 Active–Candidate 距离具备判别力。边界附近 Proposal 的 Candidate–oracle 距离在 2.16 s 中位数 `0.00023`，远处 Proposal 为 `0.00304`；用这个需要 GT-run oracle 的距离反向排序，AUROC `0.7429`，只是不可部署的诊断上界。

一种符合代码和观测的解释是：错误 Proposal 也会把 Candidate 从初始状态重建，在线 pair 距离可能因 fork 本身增长，而非真实边界。远处 Proposal 的在线 pair 距离中位数从 `0.01135` 增至 `0.05662`，边界附近从 `0.01219` 增至 `0.07772`；两组都有明显漂移且排序高度重叠。这是机制线索，不是因果证明。

## 判断

延长同一候选后，简单在线余弦距离仍未形成可靠的 transition evidence；不能靠调长 Candidate warm-up 修复当前语义判别。下一步宜先检验是否能从**同会话在线 hard pairs 与 Candidate/Shadow 对照**中学习到不同于“新建分支漂移”的特征，再决定是否短训语义头；不要直接进行长训或用 oracle 对齐分数上线。

## 溯源

- 远端结果：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/forced-horizon-probe-v1`；`forced-candidates.jsonl` 16,051 行，`forced-summary.json` 均声明 `source_split=validation`、`test_evaluation_allowed=false`、`diagnostic_only=true`。
- 固定 checkpoint SHA-256 `c359ad2a459e0ee5b9db93e7cc3a4f33ca1391c5099f0608e57202beaca4fbd6`，阈值文件 SHA-256 `5066d916a0ef0648409a989c89bda564fefae478759a1d9a3f707cd0e32d08ef`。
- 诊断模块 SHA-256 `3451b511f7d049495a936d2226be192453472db5d14b9460aa7c7b17b12f885f`；summary SHA-256 `ab2628a624689f363c51f25c747724d87b46233235edb6a5c45d223f9465ca14`；JSONL SHA-256 `dc8ffeb4f81afe30e7343b0502e19907b59d268516985a387753a02676cdb2c1`。
- 本地／远端相关测试 37/37；远端全套 238 passed、2 个已知 Gemma adapter `None_inv_freq` 失败，未处理。

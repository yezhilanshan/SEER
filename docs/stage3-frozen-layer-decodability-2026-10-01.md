# Frozen Layer Decodability Audit — execution status

The approved frozen audit traces the actual runtime chain **F → H4 → R → E_hat → E → z**. It is diagnostic only: network weights, production controller decisions and thresholds are unchanged. No validation/test data is used. The existing threshold JSON is a frozen configuration artifact, not new validation evaluation.

## Protocol

- Exact train cohort: subject_02/03/04 × set000..003 × TI_xWR14xx/TI_xWR68xx, 24 unique recordings. Index/train sidecars are validated before recording arrays are loaded.
- Natural Candidate survivors and independent forced horizons are reported separately. Ages: update 1/4/7/13, corresponding to 0/0.54/1.08/2.16 s since Proposal.
- Proposal-boundary and update-level same/different-GT-run labels are separate tasks; mixed/ignored/multiple-transition update trajectories are excluded.
- All primary probes: 128 fixed projected relational dimensions + cosine/L2, 130 inputs and 131 logistic parameters. Projection seeds: 3407/7211/12893. Three leave-one-train-subject-out folds, train-fold-only standardization/class weights, fixed CPU LBFGS, lambda 1e-3, threshold 0.5.
- Raw-dimensional sensitivity at age 13 and current-only/geometry baselines are secondary, not capacity-matched claims. The frozen base model has already seen these train subjects; this is a probe holdout, not independent unseen-subject validation of the entire model.
- Save exact Active reference identifiers, full current/reference gate and previous state, and verify both Candidate and Shadow at terminal confirm/reject as well as live updates. Any E/z mismatch >1e-5 aborts.

## Verification so far

Local targeted tests: 24 passed, 1 skipped. School-server targeted tests including cached Qwen: 44 passed. Full server suite: 279 passed, 2 known Gemma adapter failures; Gemma remains untouched as requested.

The original smoke retained 1222 rows from 181 proposals over 3344 updates and checked 5516 natural/forced steps; E/z errors were both zero. It predates review fixes and is retained under `frozen-layer-decodability-smoke-v1`; it will not substitute for the revised smoke.

Revised smoke v2 accepted: 5516 checked steps, max E/z error **0.0/0.0**; every layer and complete gate/previous-state array is finite, dimensions and hashes verified. It took 218.29 s. Natural rows and Candidate metadata are identical to smoke v1, consistent with the instrumentation not changing controller behavior.

Full extraction and probes completed on 2026-10-01 20:53:51 CST. Analysis below was performed read-only on the completed train-only artifacts; no model/controller changes or new training were started.

Formal chained job launched on 2026-10-01, parent PID **2895**. PID record: `frozen-layer-decodability-v1.pid`; log: `frozen-layer-decodability-v1.log`. The second command runs probes only if extraction exits successfully. Based on revised smoke throughput, feature extraction is roughly 1.5 hours; CPU probe/raw-dimensional sensitivity duration has not yet been measured.

## Remote locations

Project: `/home/xibeidaxue/SEER`.

Output root: `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2`.

Revised smoke: `frozen-layer-decodability-smoke-v2`; completed formal audit: `frozen-layer-decodability-v1`, then its `probes` subdirectory. Original weights and prior experimental outputs are retained.

Extraction uses a 0.02 s pause per natural update and pauses at GPU temperature 82°C until below 78°C. No training of Event Former or Verify is part of this job.

## 结果分析：主要结论

**这轮不支持“state gate 把 transition 信息压到随机水平”的判断。更有证据的解释是：在线表示有可线性解码的 GT-run 变化信号，但固定 cosine/少数标量不能充分读出；F→H4 和 E→z 的可解码性能有一致下降，晚期证据又晚于当前 natural Candidate 的生命周期。**

这是当前固定 probe 协议下的性能定位，不是严格的信息论损失证明，也没有通过改变模块获得因果证据。

### 1. 完整性、样本和收敛

- 24/24 train recordings，80,170 natural updates，4,244 Candidates，27,985 retained natural/forced rows；三 subject、四 session、两 sensor 范围已在读取数组前核对。
- 六层输出全部完成；2,232 条结果记录中 1,422 个 fit、234 个因无训练类别/样本而未定义的条目、576 个固定方向指标。未定义条目是 natural age 13 没有样本的预期结果，不是拟合故障。
- 主协议 `projected_relational`：756 个有样本的 fit 全部通过收敛标记，108 个 natural age-13 条目未定义；所有实际主 fit 均为 131 个参数。
- 28 个未收敛 fit 全部属于 `raw_relational_secondary_age13`：F 4 个，H4/R/E_hat/E 各 6 个。不能用这些高维辅助分数证明原空间信息缺失。z 的 6 个有样本 raw fit 通过收敛标记。
- 全部 completed-record integrity 的最大 E/z trace/controller 误差均为 0；日志没有 Traceback、CUDA OOM 或 mismatch。Probe summary 绑定的源 manifest SHA 一致；train-only、test_evaluation_allowed=false。
- 本地复制的指标文件与服务器 SHA 一致：summary `769a5b82ac80bca6b0d10ae168491c8ebaa617c0177cd815ae24bc2a27a1329e`；results `15d96b36706503b23f279609e13b9591b526793aef0ba4a707caeff36d7c1e38`。Manifest SHA `7cca37835b7ede171f67fda5855c46a511f82865b197747ceb377ff2ed25f21a`。

### 2. 主要任务：当前 GT run 是否已不同于 Frozen Active reference

下表是 **forced cohort、容量匹配向量 probe** 的 AUROC。计算方法：每 seed 先对三个 held-out subject 做等权宏平均，再取三个固定 seed 的中位数；不是 pooled-update 指标，也没有挑选最佳 seed。

| 层 | Proposal 当下，0 s | 0.54 s | 1.08 s | 2.16 s |
| --- | ---: | ---: | ---: | ---: |
| F | 0.497 | 0.582 | 0.683 | 0.788 |
| H4 | 0.493 | 0.552 | 0.636 | 0.737 |
| R | 0.482 | 0.568 | 0.645 | 0.733 |
| E_hat | 0.488 | 0.570 | 0.646 | 0.743 |
| E | 0.461 | 0.559 | 0.641 | 0.736 |
| z | 0.513 | 0.541 | 0.598 | 0.705 |

Forced update 标签有效样本随年龄变化：

| 时间 | 正例 | 负例 | 有效总数 |
| --- | ---: | ---: | ---: |
| 0 s | 153 | 3815 | 3968 |
| 0.54 s | 511 | 3096 | 3607 |
| 1.08 s | 790 | 2566 | 3356 |
| 2.16 s | 1126 | 1938 | 3064 |

不能把这个表的全部时间增益解释成纯粹的表示成熟：GT 是否发生变化本身随时间改变，mixed/gap 排除也改变有效集合。Proposal 当下的正例只有约 3.9%，每个 held subject 为 58/50/45 个，因此早期随机水平不证明“模型永远没有信息”。

2.16 s 的跨 subject 稳定性（每 subject 对三个 seed 取中位数）：

| 层 | subject_02 | subject_03 | subject_04 |
| --- | ---: | ---: | ---: |
| F | 0.756 | 0.795 | 0.814 |
| H4 | 0.707 | 0.740 | 0.763 |
| R | 0.705 | 0.743 | 0.759 |
| E_hat | 0.718 | 0.753 | 0.761 |
| E | 0.708 | 0.733 | 0.766 |
| z | 0.686 | 0.714 | 0.708 |

对应 seed-macro AUROC 全范围：F 0.787–0.789，H4 0.732–0.739，R 0.733–0.742，E_hat 0.741–0.747，E 0.734–0.738，z 0.698–0.705。信号不由单个 subject 或 seed 撑起，但只有三个 subject，不能称独立全系统泛化验证。

晚期 AP 的 subject-macro/seed 中位数分别为 F 0.658、H4 0.579、R 0.578、E_hat 0.583、E 0.588、z 0.531；对应 prevalence 常数 baseline 的 subject-macro AP 约 0.367。z 有真实优势，但不是接近完美的“刹车”。z 的 held-subject balanced accuracy 中位数约 0.650/0.658/0.664，阈值固定 0.5。

### 3. 哪一段最值得怀疑

2.16 s、forced update，同一 subject/seed/样本上作成对 AUROC 差，正值表示前层优于后层：

| 比较 | 9 折/seed 对比的下降中位数 | 范围 | 发生下降的对比数 |
| --- | ---: | ---: | ---: |
| F→H4 | 0.0494 | 0.0440–0.0658 | 9/9 |
| H4→R | -0.0025 | -0.0090–0.0171 | 3/9 |
| R→E_hat | -0.0085 | -0.0142–0.0033 | 1/9 |
| E_hat→E | 0.0090 | -0.0136–0.0217 | 7/9 |
| E→z | 0.0250 | 0.0144–0.0662 | 9/9 |

Natural update 在 1.08 s 也支持相同主趋势：F/H4/E_hat/E/z 分别约 0.688/0.651/0.646/0.655/0.589。F→H4 和 E→z 均 9/9 下降；E_hat→E 只有 3/9 下降，中位差 -0.0014，即 gate 后反而略高。

因此：

1. **不支持把 gate 当作首要根因。** 没有 E_hat≈0.7、E≈0.5 的断崖，门控差异小且有方向反转。
2. **F→H4、E→z 是更一致的可解码性能下降位置。** 前者涉及整个 LLM1–4/Cross-Attention/状态条件路径，不能仅归罪于某个 cross-attention block；后者提示现有 semantic projection 的 transition-sensitive 读出有损。
3. **在线 E/z 并非完全不可用。** 容量匹配线性 probe 在各 subject 上均高于随机水平，不能继续用“没有任何 transition evidence”概括。
4. **这不是严格的因果损失定位。** 不同原始维度的随机投影、固定正则、不同几何及状态参考都影响 probe；高维辅助还存在未收敛，不能据此宣布某层不可逆删除了信息。

### 4. 为什么前几轮 cosine 很差，却仍能线性解码

同一 forced age-13 update 任务：

| 层 | 固定方向 cosine | 两标量 geometry-only 线性 probe | 130 维向量主 probe |
| --- | ---: | ---: | ---: |
| F | 0.444 | 0.569 | 0.788 |
| H4 | 0.418 | 0.583 | 0.737 |
| E_hat | 0.425 | 0.576 | 0.743 |
| E | 0.407 | 0.596 | 0.736 |
| z | 0.382 | 0.614 | 0.705 |

简单“越远越不同”的几何假设在该任务下不成立。Geometry-only probe 的方向/组合只在 probe-training subjects 拟合，未在 held subject 翻转方向；即便如此，它仍明显落后于完整向量读出。不能看到 cosine<0.5 后临时反向、再宣称获得独立验证。

z 的 raw-dimensional auxiliary probe 在该晚期任务 AUROC=0.711，且该组通过收敛标记，与主 probe 的 0.705 一致：它保留可读出的信号，并不是完全塌缩。此前差几何分数不能等同于 Event Token 没有动作/变化信息。

### 5. 年龄和闭环生命周期是实质限制

Natural 的 4,244 个 Candidate 中，age 4 留存 4,239，age 7 留存 2,565，age 13 为 **0**。因此 2.16 s 的所有结论来自 forced horizon，**不是现有闭环在实际决策时的性能**。

实际 frozen thresholds：proposal=0.55、proposal_persistence=2、verify_confirm=0.45、verify_reject=0.2、candidate_timeout_updates=6。自然终止分布：confirm age 4/5/6/7/8/9 为 1112/239/319/2421/94/1；reject age 8 为 49；录音结束未决 9。

共 4,186/4,244=98.63% 的 Candidate 被确认。按本次严格 proposal-boundary 标签看，负例 1,371/1,401=97.86% 被确认，正例 1,720/1,732=99.31% 被确认。**这说明在这些 train 诊断轨迹里，当前后端几乎没有拒绝作用。** 这个比例不是之前 validation Verify 分类集的 false_confirm_rate，不能混成同一个指标。

为控制 update 标签随年龄改变的影响，proposal 标签固定且 forced 有效集合均为 3,133 个 Candidate（1732 正、1401 负）：

| 层 | 0 s | 0.54 s | 1.08 s | 2.16 s |
| --- | ---: | ---: | ---: | ---: |
| F | 0.575 | 0.682 | 0.746 | 0.744 |
| H4 | 0.603 | 0.618 | 0.675 | 0.683 |
| R | 0.596 | 0.611 | 0.675 | 0.675 |
| E_hat | 0.604 | 0.618 | 0.689 | 0.674 |
| E | 0.610 | 0.619 | 0.675 | 0.680 |
| z | 0.561 | 0.574 | 0.631 | 0.669 |

z 的晚期提升在这个固定标签/固定集合上也存在，支持证据形成需要时间的解释，但不能推导“一律等 2.16 s 就能改善闭环”：延迟、over-segmentation、GT-run 监督、两个 task 的语义仍需分别验证。Proposal task 的 current-only secondary 与 relational 分数很接近（例如晚期 F 0.740 vs 0.744，z 0.669 vs 0.669），也提示其中一部分可能是当前动作/过渡阶段线索，不能全部认作同/异事件的关系能力。

### 6. 两种 sensor 与数值尖峰

Late forced update 的 sensor 条件 AUROC，对 3 subjects×3 seeds 描述性取中位数：

| 层 | xWR14xx | xWR68xx |
| --- | ---: | ---: |
| F | 0.797 | 0.792 |
| H4 | 0.733 | 0.748 |
| E_hat | 0.742 | 0.751 |
| E | 0.738 | 0.738 |
| z | 0.689 | 0.728 |

这一轮不支持 xWR68xx 整体失效的判断。各层的 subject×session×seed 晚期描述性 AUROC 也全部高于 0.5（例如 F 最低 0.611、z 最低 0.573）；这些相关分层不是 36 个独立验证实验。

Forced age-13 的 E norm 中位数约 41.05，但有 2/4,223 条 >200，1 条 >400。最大 821.51 在 `subject_02__set001__TI_xWR68xx:2867`，H4 norm=1942.81；其 update 标签因 mixed update 被排除。另一个 E norm=319.78 在 `subject_03__set002__TI_xWR14xx:2954`，属于有效正例。这些值有限但仍是异常尾部，应保留排查，不能因 trace 误差为零而说没有数值问题；当前未做去尖峰重拟合，不能完全排除它对训练标准化/系数的影响，也没有证据把整体现象归因于这两例。

两例在 natural controller 中都已于 age 7 confirm，当时 E norm 仅 78.09/67.21；以上尖峰出现在 forced 延长到 age 13 的分支，不能直接归因于现有自然决策的失败。Gate mean 的 forced 候选中位数在 age 1/4/7/13 为 0.746/0.206/0.219/0.228；它取决于实际状态输入，不是随 maturity 简单单调开启的常数门控。

### 7. 下一步建议：先换证据读出，不立即重训大模型

**首选下一实验：冻结网络的短时序、向量关系读出诊断。**

1. 保留现有 Stage 1/2、Fork 初始化、gate 和正式 controller；不继续做 warm-up 阈值 sweep，不再以原三标量标签复制方式重训 Verify。
2. 使用真实 online Proposal，主要监督为 clean update-level same/different GT run，proposal-boundary 仅独立辅助任务。保留 ignored/mixed/multiple-run 排除和 subject-disjoint probe folds。
3. 比较 F-only、E/z relational、F+E/z relational；再比较单 update 与连续 3/8-update 证据。所有 primary 仍固定 130 输入/131 参数和相同训练折变换，避免“时序更好只是参数更多”。重点评估实际可决策年龄和 natural survivors，不只展示 forced 2.16 s。
4. 现有 shards 只保留 age 1/4/7/13，**不是连续 3/8 updates**；短时序实验需补采连续冻结 trace，不能把四个稀疏时间点冒充滑动时序。
5. 若短时序向量读出在较早决策年龄跨 subject 稳定改善，再考虑小型 Semantic Difference/Verify 替换 pilot，冻结 Event Former，使用 update-aligned GT-run 监督，并重新验证闭环 precision/recall/过分割/延迟。不能把 AUROC≈0.70 的 probe 直接上线。
6. 只有读出方案仍不够时，再优先考虑 transition-aware Semantic Projection 训练或保留 F 的 residual 路径；整个 Event Former 重训、gate 改造和扩大模型目前都不是已有证据支持的第一步。

本次分析没有执行这些后续实验，没有读取 validation/test，也没有变更模型或正式控制策略。

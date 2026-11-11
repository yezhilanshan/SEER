# Stage 3 Fork 分层定位：gate、Event State、Semantic Key

日期：2026-09-30。延续 [2×2 Fork 消融](stage3-fork-init-gate-ablation-2026-09-30.md)，只读 validation、冻结全部权重，不训练、不读取 test、不修改正式 controller。新增逐 update 的实际 gate 统计、四臂两两 Event State/Key 距离及原始 radar update 差异；所有分支沿用同一输入、同一候选与同一标签。

## 完整性

复用上一轮的 8 条 validation recording（subject_05，4 session × 2 sensor）、1,341 个自然 Proposal；其中 1,336 个具有完整 13-update horizon，970 个有严格真/假标签（468 真、502 假）。17,392 个逐 update 行。旧版四臂分数的所有 age×score AUROC 与上一轮逐位一致；Candidate/Shadow 的 Event State 和 Key 逐 update 最大复现误差仍均为 **0.0**，还核对了 26,729 个自然决策和 16,051 个既有 forced 行。`source_split=validation`、`diagnostic_only=true`、`test_evaluation_allowed=false`。

| 比较 | 改变的因素 | update 1：Event L2 / Key L2 / gate 绝对差均值（中位数） | update 13：同三项（中位数） |
| --- | --- | ---: | ---: |
| PP–PF | `P_base` 不变，只改 maturity | 0.105 / 0.000161 / 0.000922 | 0.056 / 0.000233 / 0.000314 |
| AP–AF | Active 初始化不变，只改 maturity | 0.037 / 0.000234 / 0.001955 | 0.058 / 0.000258 / 0.000363 |
| PP–AP | progressive 不变，只改初始化 | 53.136 / 0.211486 / 0.588077 | 16.072 / 0.089149 / 0.048558 |

PP/PF 的 gate 均值中位数在 update 1 为 0.74733/0.74827，在 update 13 为 0.23015/0.23038。AP/AF 对应为 0.20697/0.20951、0.23994/0.24061。因此“maturity 门控不起作用”的更精确表述是：**当前 checkpoint 的 gate 对 maturity 输入不敏感**；gate 本身会随初始状态及在线状态发生大幅变化。不同维度的 Event/Key L2 不能直接按绝对数值比较信息量。

## 真/假 Proposal 排序（相同的 970 个候选）

下表为 AUROC；值越大代表更倾向把较大的距离当成真边界。原始 radar 差异是当前 update 相对前一 update 的量；`physical proposal` 是 Fork 触发当时的原模型分数，后续年龄重复这个固定值。

| 特征 | update 1（Proposal 当下） | update 7（1.08 s） | update 13（2.16 s） |
| --- | ---: | ---: | ---: |
| 原始 radar update 余弦距离 | 0.501 | 0.490 | 0.620 |
| 原始 radar update L2 | 0.511 | 0.420 | 0.551 |
| physical proposal 概率 | 0.570 | 0.570 | 0.570 |
| PP–PF Event L2（只改 maturity） | 0.516 | 0.526 | 0.540 |
| PP–PF Key L2 | 0.468 | 0.501 | 0.585 |
| PP–AP Event L2（只改初始化） | 0.547 | 0.543 | 0.530 |
| PP–AP Event 余弦距离 | 0.502 | 0.544 | 0.572 |
| PP–AP Key 余弦距离 | 0.516 | 0.544 | 0.572 |

尤其要避免将极小的数值差异误读为可用分类器：update 13 的 PP–PF Event 余弦距离 AUROC 虽为 0.613，但正/负类中位数仅约 `4.47e-7 / 3.58e-7`，接近 float32 余弦计算的数值分辨率；Key 余弦距离大量取 0。PP–PF Event/Key L2 AUROC 则为 0.540/0.585，分离仍弱。对于改变初始化的 PP–AP，Event 和 Key 余弦 AUROC 在 update 13 **同为 0.572**，没有观察到“Event State 已强可分而投影抹掉信号”的模式。

分 session 看，update 13 的原始 radar 余弦 AUROC 为 set000/001/002/003 的 `0.647/0.648/0.608/0.485`，只在较晚时间出现且有一组失效；Proposal 当下四组为 `0.469/0.479/0.500/0.518`。它不能直接作为早期 Verify 信号。PP–AP Event L2 在四组为 `0.559/0.430/0.468/0.555`；Key 余弦为 `0.635/0.551/0.547/0.531`，仍未形成跨会话强信号。所有会话均属同一受试者，不能声称跨受试者泛化。

## 判断与建议

1. 成熟度输入在当前 gate 网络里确实几乎不改变输出；继续扫描 warm-up/maturity 无实际杠杆。但不能说整个 gate 无用，因为初始化会显著改变其输出。
2. `P_base` 和 Active snapshot 的状态在 Event 层就明显不同，投影后也有可测差异；真/假排序始终弱。**Semantic Projection 不是已证实的主要信息瓶颈，单独换初始化也没有证据能修好 Verify。**
3. 下一步不训练 Verify、不改正式 Fork。应转到上游“在线 Event State 是否编码了真实边界”诊断：在独立受试者或严格留出的 recording 上，固定比较边界前后 radar 表示、Event State 更新的方向/持续性与 label run，而非再用同一批 validation 会话挑选分数方向。若无稳定边界证据，再重新设计 Event State 的训练监督和形成机制。

## 产物与测试

- 远程结果：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/fork-layer-audit-v1/`，`fork-summary.json` SHA-256 `4e7778b6d46e095e50bb2deb19bf2077907620a5a14bc888692f278dfdaffba3f`，`fork-rows.jsonl` SHA-256 `1ff2955c83f81a17b1aabdf2c8a1c92e78620e82ec464a675cd89939adbeca1b`。
- 新版探针：[parrad_stage3_fork_ablation.py](../src/continuous_mmwave/parrad_stage3_fork_ablation.py)，代码 SHA-256 `4dfac02f637676ef161603a78eef99957c88fe5aa4e077331ff597b01ffa4876`。旧实验的原始探针源码原样保留在上一轮结果目录 `probe-source.py`，SHA-256 `59a27f604c26d5f20014602a18a61e764fc4b7d5a08d75d23192b2352ad2422a`。
- 远程定向测试 25 passed；全套 254 passed、2 个既有 Gemma adapter `None_inv_freq` 失败，未触碰 Gemma。

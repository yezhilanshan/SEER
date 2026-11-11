# Stage 3 Fork 初始化 × maturity 门控消融（validation-only）

日期：2026-09-30。实验只回放 validation，冻结所有权重；没有训练、读取 test split 或修改正式 controller。

## 设计与完整性

| 分支 | 初始 Event State | Event Former maturity 输入 |
| --- | --- | --- |
| PP（当前 Candidate） | `P_base` | progressive，随候选年龄增长 |
| PF | `P_base` | full，恒为 1 |
| AP | Active snapshot | progressive |
| AF（当前 Shadow） | Active snapshot | full，恒为 1 |

四条分支消费完全相同的 radar token、mask 和 prompt cache，逐 update 以 batch=1 顺序回放，以避免 fp16 的 batch-size 数值漂移。PP 和 AF 与既有 Candidate/Shadow 的 Event State、Semantic Key 的最大绝对误差均为 **0.0**（容差 `1e-5`）；另核对了 26,729 个自然 controller 决策及 16,051 个既有 forced-horizon 行。共 8 条 validation recording（subject_05，4 个 session × 2 种传感器），1,341 个 Proposal，5 个在录音末尾截断；1,336 个拥有完整 13-update horizon，其中按既有严格定义可标注的为 970 个（468 真、502 假）。所有 AUROC 比较均固定在这 970 个相同候选上。

## 主要结果

分数为归一化余弦距离 `(1−cos)/2`；较大的距离被视为“真 Proposal”证据。`1−d(A,AF)` 是实验前指定的反方向检查，不可称独立验证。

| age updates（约秒） | d(A,PP) | d(A,PF) | d(A,AP) | d(A,AF) | 1−d(A,AF) | d(PP,AF) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1（0） | 0.512 | 0.512 | 0.463 | 0.463 | 0.537 | 0.516 |
| 4（0.54） | 0.497 | 0.497 | 0.444 | 0.444 | 0.556 | 0.531 |
| 7（1.08） | 0.469 | 0.469 | 0.429 | 0.429 | 0.571 | 0.544 |
| 10（1.62） | 0.448 | 0.448 | 0.420 | 0.420 | 0.580 | 0.567 |
| 13（2.16） | 0.461 | 0.461 | 0.443 | 0.443 | 0.557 | 0.572 |

固定 Active snapshot 时，AP 与 AF 的 AUROC 几乎重合；固定 `P_base` 时，PP 与 PF 也几乎重合。直接比较两个门控分支的 Key 距离，在 update 13，`d(PP,PF)` 中位数为 0、90 分位约 `2.1e-7`；`d(AP,AF)` 中位数为 0、90 分位约 `5.1e-7`。因此目前 checkpoint 下，改变 Event Former maturity 输入对投影后的 Key 几乎没有影响。距离接近零时，对其自身计算的 AUROC（例如 update 13 的 `d(PP,PF)` 为 0.550）不应解释成可用判别能力。

初始化确实强烈改变表示几何：update 13 的 `d(PP,AP)` 中位数约 0.00199、90 分位约 0.0553，远大于门控效应。但 Active 初始化没有提高当前方向的真假区分：`d(A,AP)` 0.443，低于 `d(A,PP)` 0.461；即使预先反转方向，`1−d(A,AF)` 也只有 0.557。`d(PP,AF)` 达 0.572，仍属弱信号。

2.16 秒时各 session 的 `d(A,PP) / d(A,AP) / 1−d(A,AF) / d(PP,AF)` AUROC：

| Session | 可标注候选 | PP | AP | 反向 AF | PP–AF |
| --- | ---: | ---: | ---: | ---: | ---: |
| set000 | 266 | 0.347 | 0.332 | 0.668 | 0.634 |
| set001 | 244 | 0.430 | 0.400 | 0.599 | 0.550 |
| set002 | 198 | 0.517 | 0.489 | 0.511 | 0.547 |
| set003 | 262 | 0.452 | 0.454 | 0.546 | 0.531 |

反向 AF 的方向在四个 validation session 上均略高于 0.5，但强度从 0.511 到 0.668，且四组都来自同一受试者，不能当成跨受试者独立复现。按传感器汇总，其 AUROC 为 xWR14xx 0.604、xWR68xx 0.508。2.16 秒的同录音近邻 hard-pair 胜率，PP 0.417、AP 0.400、反向 AF 0.598、PP–AF 0.562；反向 AF 在 8 条 recording 中仅 6 条胜率超过 0.5。

## State / projection 漂移

逐行保存了 `ΔE_t=||E_t−E_0||` 和 `Δz_t=||z_t−z_0||`，以及 state/key norm。典型中位数如下；这里两个初态不同，因此不能直接比较 PP 与 AP 的漂移绝对值来推断哪一个“更好”。

| update | PP ΔE | PP Δz | AP ΔE | AP Δz |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 68.785 | 1.333 | 3.935 | 0.030 |
| 7 | 42.469 | 1.335 | 20.628 | 0.203 |
| 13 | 40.699 | 1.336 | 31.824 | 0.420 |

在同一 970 个候选上，以漂移量本身分类的 AUROC 也弱：update 13 的 PP `ΔE/Δz` 为 `0.432/0.470`，AP 为 `0.471/0.443`；PF/AF 分别近乎重合。说明状态确实在变化，但这些标量漂移量没有给出可靠的正向边界证据。仅凭“相对自身初态漂移”不能严格定位信息究竟在 Event State 还是 Semantic Projection 层丢失；需要进一步记录两个分支之间的 Event State 距离及 gate 激活分布。

## 判断与下一步

这轮没有证据支持“`P_base` reset 是主要问题，改为 Active snapshot 就能解决真假 Proposal 区分”；初始化影响表示，但四臂的正向 Active-to-branch AUROC 全在约 0.42–0.51。当前成熟度门控则对 Semantic Key 近乎无影响，不应继续做 maturity/warm-up sweep。两者都不是已证实的单点修复。

建议下一步继续**无训练、只读**定位：在同一 validation matched cohort 上记录门控网络的实际 gate 值与成熟度输入敏感性、同门控不同初始化和同初始化不同门控的 Event State 两两距离，并比较 Semantic Projection 前后差异；同时分 session/sensor 检查在线输入与 Event State 的真/假边界可分性。若上游 Event State 本身无稳定信号，再讨论训练目标/表示；在此之前不改 Candidate 初始化、不重训 Verify。

## 产物与校验

- 远程：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/fork-init-gate-ablation-v2/fork-summary.json`，SHA-256 `0021375ba6cf2c3aa85a3d1e29e36b82a76c915caa5ee01795e63cd6974689e27`
- 远程：同目录 `fork-rows.jsonl`（17,392 行），SHA-256 `3470955552f518b73b23cffb75bc77b778cb85333af5826503c202d37b4df2df`
- `source_split=validation`、`diagnostic_only=true`、`test_evaluation_allowed=false`；checkpoint SHA-256 `c359ad2a459e0ee5b9db93e7cc3a4f33ca1391c5099f0608e57202beaca4fbd6`，阈值 artifact SHA-256 `5066d916a0ef0648409a989c89bda564fefae478759a1d9a3f707cd0e32d08ef`。
- 探针代码：[parrad_stage3_fork_ablation.py](../src/continuous_mmwave/parrad_stage3_fork_ablation.py)；测试：[test_parrad_stage3_fork_ablation.py](../tests/test_parrad_stage3_fork_ablation.py)。远程针对性测试 20 passed；全套 252 passed、2 个既有 Gemma adapter `None_inv_freq` 失败，与本实验无关。

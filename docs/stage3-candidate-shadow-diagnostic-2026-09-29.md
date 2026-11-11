# Stage 3 Candidate/Shadow 无学习诊断（2026-09-29）

## 范围与口径

固定原始 `best_verify.pt`、validation-only 阈值以及同一批 validation 录音；未训练、未读取 test、未修改正式控制器。沿用上轮自然 Proposal 后强制延长 12 个 update（2.16 s）的反事实回放，同时逐 update 记录 Active 冻结参考 `A`、Candidate `C_t` 和旧事件延续 Shadow `S_t` 的投影 key。距离统一为 `d(X,Y)=(1-cos(X,Y))/2`；更大值预先解释为更强的切换证据，AUROC 没有事后取反。

固定比较 `d(A,C)`、`d(A,S)`、`d(C,S)`、`d(A,C)-d(A,S)`、`||(Ĉ_t-Ŝ_t)-(Ĉ_0-Ŝ_0)||₂/4`，以及每项相邻 update 的单侧、按真实时间差归一的斜率。带帽向量为单位归一化 key，`t=0` 是 Proposal；计算斜率只使用当前与前一个 update。

主报告同时保留原有“Proposal 距 GT 边界 ≤0.72 s”的代理标签，新增更严格的干净子集：正例是在 `[Proposal−0.72 s, Proposal+2.16 s]` 中恰有一个 GT 边界且它距 Proposal ≤0.72 s；负例是在该窗内没有边界；其余排除。这个标签仍只是边界时间代理，不等于经人工裁决的语义有效性。只在完整存活到 2.16 s 的相同 Candidate 集合上比较年龄。

## 轨迹一致性与样本

- 自然控制器的 26,729 条决策和 Proposal 分数、6,837 条已有 Semantic 分数逐项复现。
- 1,341 个自然 Proposal 中，5 个靠近录音末端；完整等长集合为 1,336 个。
- 干净子集为 970 个：468 正例、502 负例；另 366 个因未来边界或多边界而排除。数据来自 subject_05 的 4 个会话 × 2 个传感器，不能据此推断跨受试者泛化。
- 输出声明 `source_split=validation`、`test_evaluation_allowed=false`、`diagnostic_only=true`；checkpoint 和阈值 SHA-256 均与上一轮相同。

## 无学习特征 AUROC（干净等长子集）

| Proposal 后 | `d(A,C)` | `d(A,S)` | `d(C,S)` | `d(A,C)-d(A,S)` | C–S residual change |
|---:|---:|---:|---:|---:|---:|
| 0.18 s | 0.512 | 0.456 | 0.522 | 0.531 | 0.488 |
| 0.54 s | 0.497 | 0.444 | 0.531 | 0.557 | 0.479 |
| 1.08 s | 0.469 | 0.429 | 0.544 | 0.557 | 0.495 |
| 1.44 s | 0.450 | 0.417 | 0.563 | 0.554 | 0.509 |
| 1.80 s | 0.452 | 0.428 | 0.572 | 0.532 | 0.520 |
| 2.16 s | 0.461 | 0.443 | 0.572 | 0.540 | 0.523 |

2.16 s 时，`d(C,S)` 在 xWR14xx/xWR68xx 的 AUROC 分别为 `0.580/0.564`；但 8 条录音分别约 `0.483–0.634`。原始宽松代理标签下同龄 `d(C,S)=0.553`、`d(A,C)=0.509`。斜率没有给出更强且稳定的区分：2.16 s 的 `slope_d(C,S)=0.477`，`slope_residual=0.537`。`d(A,S)` 的 AUROC 长期低于 0.5 是一个反向线索，但把分数事后取反会使用同一 validation 数据选择方向，不能作为已证实的模型能力。

## 同录音在线 hard-pair

在每个年龄，只为每个干净正例选择**同一录音**中 Proposal 时间最近的干净负例；配对可复用负例，不按模型分数选对。所有分数仍来自在线状态，GT 仅用于离线诊断配对。共有 468 对，正负 Proposal 时间差中位数 `3.96 s`，所以“同录音”并不意味着极近时间或同一动作上下文，配对结果也不能当作 468 个独立试验。

| Proposal 后 | `d(A,C)` 配对胜率 | `d(C,S)` 配对胜率 | `d(A,C)-d(A,S)` 配对胜率 |
|---:|---:|---:|---:|
| 0.18 s | 0.498 | 0.513 | 0.519 |
| 0.54 s | 0.487 | 0.530 | 0.549 |
| 1.08 s | 0.455 | 0.558 | 0.543 |
| 1.44 s | 0.419 | 0.564 | 0.545 |
| 1.80 s | 0.410 | 0.543 | 0.524 |
| 2.16 s | 0.417 | 0.562 | 0.573 |

2.16 s 时 `d(C,S)` 在两传感器分别为 `0.568/0.555`，但逐录音为 `0.286–0.716`，仅 4/8 条超过 0.5。`d(A,C)-d(A,S)` 两传感器分别为 `0.527/0.621`，逐录音 6/8 条超过 0.5，仍存在明显的会话和传感器波动。`d(A,S)` 在 1.80 s 的原始配对胜率为 `0.363`，反向看似有信号，但方向是在结果后发现的探索性线索。

## 判断与下一步

Candidate/Shadow 对照比单纯 Active–Candidate 距离带来了一点可重复的弱信号，但没有达到“稳定可分”标准：最大预先指定原始特征 AUROC 约 0.57、同录音配对约 0.56–0.57，且逐录音显著波动；只含一个受试者。现在**暂停短训 Semantic Difference**，不凭这些结果调阈值或上线特征。下一项更有信息量的是对 Candidate 新建状态与 Shadow 延续状态做受控机制消融：在不改变训练权重的情况下，分别检查初始状态、相同输入更新后的 key 漂移及跨会话方向一致性；先确定弱信号是否来自真实事件变化，还是 fork 初始化差异。任何反向特征的确认应事先固定方向并用未参与方向选择的会话验证。

## 溯源与测试

- 远端结果：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/candidate-shadow-features-v1/forced-summary.json`（SHA-256 `f6f173991b06ed89f8ec33849da2e54f2f0028d80b905166d086842fc3e4b660`）；同行 `forced-candidates.jsonl` 16,051 行（SHA-256 `5c8ed4fae14c03be132c56e640f61dd1b2c68acdafbdb0fdfc9c7d09bddbfcd8`）。
- 同录音配对：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/same-recording-shadow-hard-pairs-v1.json`（SHA-256 `9ed16ac742960de298e30b1f346c484e4d4ae4b368bb42220433a32e392d6f59`）。
- 固定 checkpoint SHA-256 `c359ad2a459e0ee5b9db93e7cc3a4f33ca1391c5099f0608e57202beaca4fbd6`，阈值 SHA-256 `5066d916a0ef0648409a989c89bda564fefae478759a1d9a3f707cd0e32d08ef`。
- 相关测试 12/12 通过；远端全套 244 passed、2 个先前已知的 Gemma adapter `None_inv_freq` 失败，未改动 Gemma。本地 `interview` 环境缺少 `h5py`，所以本地全套在收集阶段停止。

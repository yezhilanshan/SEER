# Stage 3 Verify 候选级诊断（2026-09-29）

## 范围与不可变条件

- 仅回放 Stage 3 缓存的 `train`（24 条 recording）和 `validation`（8 条 recording）；未读取或评估 `test`。
- 使用原始 `best_verify.pt`（SHA-256 `c359ad2a459e0ee5b9db93e7cc3a4f33ca1391c5099f0608e57202beaca4fbd6`）及其原始 validation-only timeout/reject 阈值。轻量线性／MLP 探针只在 train 拟合，validation 只评估一次；没有启动正式训练，也没有改动 Stage 1/2、Physical、Semantic 或两个 Verify checkpoint。
- 候选审计模块：`src/continuous_mmwave/parrad_stage3_candidate_audit.py`，本地和远端 SHA-256 均为 `2b45840fa0b486a5ee17ae1cf462f232819abb72483260d167dc87f942667652`。
- 远端结果目录：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/verify-candidate-audit-v1`。

## 闭环复标定：hard-negative Verify 变差

三个 validation 分片各 6/6，指纹、网格和 checkpoint SHA 一致，合并状态为 `calibrated_validation`，`test_evaluation_allowed=false`。新 checkpoint 最优格为 proposal 0.55、persistence 2、confirm 0.47、reject 0.2、timeout 6。对同一 8 条 validation recording 做完整回放：

| 指标 | 原始 Verify | hard-negative Verify |
|---|---:|---:|
| F1@0.36 s | 0.2432 | 0.2419 |
| F1@0.72 s | 0.4167 | 0.3885 |
| Precision@0.72 s | 0.3243 | 0.2842 |
| Recall@0.72 s | 0.5827 | 0.6138 |
| 过分割率 | 1.4566 | 1.7778 |
| 欠分割率 | 0.6599 | 0.6179 |
| Confirm / Reject | 1326 / 12 | 1594 / 0 |
| 确认延迟中位数 | 0.81 s | 0.54 s |
| state norm | finite，max 1356 | finite，max 1357 |

新头用更多 Confirm 换取少量召回，精度和 F1 下降。合并阈值 SHA-256：`0af36e11ce4f2366db17febe70ffbdf88304c6a62c9e40e86ff1ae1f43d98dd2`；详细比较 SHA-256：`dfc03e3f426fd5f79976dc9f11cb99378454c0242cd0ac692000603133f6053b`。

## 一候选一记录的审计

在线 controller 每次 Proposal 建一个独立 episode，保留所有顺序 update、Confirm／Reject／recording_end 结束原因、最近直接 GT 边界距离和 GT label-run 身份。`-100` 间隙不生成伪边界。train 为 4244 个 Candidate（4242 个至少有一个 scored update）；validation 为 1341 个，TI_xWR14xx 661、TI_xWR68xx 680。validation 有 1326 个 Confirm、12 个 Reject、3 个 recording_end；候选寿命中位数约 1.08 s，scored update 中位数为 6。

validation 中，仅 316/1341（23.6%）个 Candidate 的生命期包含一次直接 GT label-run 切换；470/1341（35.0%）个 proposal 距最近边界不超过 0.72 s。距边界大于 1.44 s 的 505 个 Candidate 中仍有 499 个 Confirm。这些是候选级诊断计数，**不能直接等同官方一对一匹配的 false-confirm 指标**。

### 原定义：proposal 时间距 GT 边界

固定 20 epoch、seed 42、train-only 标准化的 `Linear(3,1)` 与 `Linear(3,16)-ReLU-Linear(16,1)` 探针，和冻结原 Verify 对照：

| 标签容差 / 特征时刻 | validation 正类先验 | 冻结 Verify AUROC / AP | MLP AUROC / AP |
|---|---:|---:|---:|
| ±0.36 s / 首次 scored update | 0.188 | 0.469 / 0.174 | 0.531 / 0.201 |
| ±0.36 s / 最后 update | 0.188 | 0.522 / 0.197 | 0.507 / 0.182 |
| ±0.72 s / 首次 scored update | 0.350 | 0.456 / 0.328 | 0.544 / 0.382 |
| ±0.72 s / 最后 update | 0.350 | 0.506 / 0.349 | 0.524 / 0.350 |

±0.72 s 首次 MLP 在两传感器的 AUROC 为 0.553 / 0.531；**以 4 个会话为重采样单位、将配对传感器保持在一起**的 bootstrap 总体 95% 区间约 [0.510, 0.582]。既未达到预先登记的总体 AUROC≥0.60，也未达到 AP 比先验高 0.05；终点更没有变好。

### 代理定义：GT 游程在候选生命期发生切换

这个标签比“proposal 靠近边界”更接近候选级有效性，但仍不能证明 Candidate 与 Shadow 的真实语义假设，不能用于正式训练。validation 正类 316/1341（先验 0.236），没有跨 `-100` 间隙误判的实例。

| 特征时刻 | 冻结 Verify AUROC / AP | train-only MLP AUROC / AP |
|---|---:|---:|
| 首次 scored update | 0.408 / 0.199 | 0.593 / 0.288 |
| 最后 update | 0.450 / 0.206 | 0.593 / 0.274 |
| episode 均值 | 0.454 / 0.207 | 0.569 / 0.260 |

首次 MLP 的传感器 AUROC 为 0.574 / 0.608；以 4 个会话为重采样单位的 bootstrap 总体 95% 区间约 [0.564, 0.629]。AP 比先验高约 0.053，但总体 AUROC 仍低于预登记的 0.60；且终点不比首次更好，不能据此宣称时序积累会提升 Verify。单独 `p_diff` 的 AUROC 只有 0.409；倒置方向为 0.591。首次 update 的真实切换／未切换 `p_diff` 中位数分别约 `4.8e-6`／`6.2e-5`，与原 Verify 的“差异越高越应确认”单调假设相反。首次 stability 恒为 0、maturity 恒为 0.3；到第 6 次 update，Verify 分数两类中位数都约 0.467，足以越过原 confirm=0.45，因此 Confirm 几乎只由候选成熟时间驱动。

按传感器和边界距离分桶，两种传感器都呈现相同方向：离边界 >1.44 s 的 proposal 没有 GT 游程切换，但 14xx 组 233/236、68xx 组 266/269 仍 Confirm。原始闭环 `p_diff` 的高值更常见于这些远离边界的候选。这说明问题同时位于高噪声 Proposal 和 online Semantic Difference 输出／监督分布不一致，不能只靠 Verify 阈值或 hard-negative 权重解决。

## 决策与下一步

**暂不重训 Verify，也暂不直接实现 H0/H1 序列判决器。** 先做有边界的上游诊断：

1. 固定现有 checkpoint 和录音，按 GT 切换／未切换与传感器比对 oracle Semantic Difference 与在线 Candidate 的输入、logit、概率和 maturity 分桶，定位 `p_diff` 方向颠倒发生在哪一层。
2. 将 Proposal 的 `Local Change` 与 `Radar Transition` 分数对齐真实边界，分析距边界 >1.44 s 的大量误触发，估计只修 Proposal 能降低多少不必要的 Candidate。
3. 若能证明 Candidate 与 Shadow 的在线证据在 GT 游程切换上具有稳定可分性，再正式定义一候选一标签，设计可等待／可拒绝的序列检验，并在 subject-disjoint validation 验证。现在的游程切换只是诊断代理，不能直接当最终 H1 标签。

不应把 ±0.72 s 的边界邻近标签继续复制到每个 update，也不应把本轮弱的 MLP AUROC 当作序列 Verify 已成立的证据。

## 复现与验证

审计命令（从 `/home/xibeidaxue/SEER` 执行，PID 27864 已正常结束）：

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 ./.conda/envs/seer/bin/python -m continuous_mmwave.parrad_stage3_candidate_audit \
  --config configs/parrad-md-gru-stage3-sequence-v2.yaml \
  --checkpoint /home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/best_verify.pt \
  --selected-thresholds /home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/timeout-reject-ablation-shards-v1/merged/validation_thresholds.json \
  --output-dir /home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/verify-candidate-audit-v1 \
  --device cuda:0
```

正式引用的修正后文件为 `candidate-diagnostic-session-bootstrap.json`（SHA-256 `5d8faba23daa3528d5942b9a204489cd4d87241b6fc38e5beeba67420943481c`）和 `candidate-transition-proxy-session-bootstrap.json`（SHA-256 `91d4c96ee5feac4d263d4914eae07c6e651bcd75397b85564ce5c6d878c04336`）。原 `candidate-diagnostic.json` 和 `candidate-transition-proxy-diagnostic.json` 保留作溯源，但其中的 bootstrap 错将双传感器视为 8 个独立单元，不应引用其区间；点估计不变。验证期间 GPU 温度约 50–69°C，未见持续超过 84°C。

本地 `interview` 环境的候选级及相邻 Stage 3 测试为 13/13；远端隔离 SEER 环境 Stage 3／controller 测试为 80/80。修正会话分组后远端全套 pytest 为 216 passed、2 failed，两个失败均为此前决定暂不处理的 `test_gemma3_adapter.py`（`Gemma3RotaryEmbedding` 缺少 `None_inv_freq`）。本地全套在收集阶段被 `interview` 环境缺少 `h5py` 阻断。没有为此安装或污染环境。旧的 Stage 3 hard-negative 监控自动化已在本轮任务结束后删除。

下一步我建议不要再继续修改 `Radar + GRU` 的分数融合，而是把工作拆成 **“标签核验 → 高召回 Candidate Pool → 候选可过滤性 → 正式 Verify”** 四步。这样每一步都只回答一个问题，不会再把 Proposal、Verification 和 Boundary Localization 混在一起。

之所以这样做，是因为最新实验已经很明确：直接 residual fusion 是 No-Go，fit=8 下旧 Radar 为 56.11% Recall，而加入 GRU residual 后降到 37.36%；但与此同时，GRU 确实把零阈值合法峰覆盖从 66.19% 提高到了 96.16%。这说明“GRU 没有信息”不是问题，真正的问题是它带来了大量无法区分真假的运动变化。粘贴的文本 (1) 更关键的是，Radar 与 GRU 在零阈值下的 oracle union 达到 98.58%，说明二者的候选覆盖确实高度互补。粘贴的文本 (1)

---

# 下一阶段总目标

不再要求 Physical Proposal 一步做到“又高召回又低误报”。

重新明确职责：

\[
\boxed{\text{Proposal：尽量不要漏掉真正变化}}
\]

\[
\boxed{\text{Candidate Verify：把动作内部波动过滤掉}}
\]

\[
\boxed{\text{Boundary Localization：最后再确定精确边界}}
\]

新的前半段流程应该变成：

```text
                        ┌→ Old Radar Head → Radar Candidates ─┐
MD → MD-GRU → h_t ──────┤                                    │
                        └→ GRU State Readout → GRU Candidates ├→ Candidate Pool
                                                             │
                                                             ↓
                                                   Candidate Episode
                                                             ↓
                                                 Sequential Verification
                                                             ↓
                                                  Reject / Confirm
                                                             ↓
                                                   Boundary Localization
```

注意：**Radar 与 GRU 不再融合 score。**

它们只是两个 Candidate 来源。

---

# 第一阶段：Physical Onset Audit

这一阶段先不训练任何模型。

现在最先应该确认：

\[
t_{\text{command GT}}
\]

和

\[
t_{\text{physical onset}}
\]

到底偏差有多大。

因为目前报告已经明确：GT 是命令动作转换时间，而不是逐个人工核验的真实身体运动起点。粘贴的文本 (1)

建议从 dev 中固定抽取大约 150 个案例：

| 类型 | 数量 |
|---|---:|
| Radar 正确命中的 GT | 50 |
| GRU-only、被当前评价视为 false 的峰 | 50 |
| Radar/GRU 都认为稳定的普通区域 | 50 |

对每个样本观察：

\[
[t-2s,t+2s]
\]

范围内的原始 MD / RD。如果 PARrad 有同步视频或其他人工可观察模态，就优先以同步视频判断真实身体运动变化；如果没有同步视频，**不要把 Radar 自己的变化重新定义成 GT**，只能做“可能存在标签错位”的 proxy audit。

每个 case 记录：

```text
command_GT
estimated_physical_onset
offset = physical_onset - command_GT
confidence
transition_type
```

其中 confidence 可以只分：

```text
clear
ambiguous
unobservable
```

这一阶段最重要的统计是：

\[
\Delta t=t_{\text{physical}}-t_{\text{command}}
\]

画出其分布，同时计算：

\[
P(|\Delta t|>0.36s)
\]

和

\[
P(|\Delta t|>0.72s)
\]

如果大量“GRU false peak”实际上对应清楚的真实身体变化，只是相对 command GT 提前或滞后，那么下一步不能继续用当前点标签强行训练 Proposal。

反过来，如果这些 far-GT 峰绝大多数就是走路摆腿、挥臂、周期性运动，那说明 GT 不是主要矛盾，可以继续下一步。

---

# 第二阶段：建立双源 Candidate Pool

如果第一阶段没有发现严重 GT 错位，就停止 score fusion，建立：

\[
\mathcal C
=
\mathcal C_R
\cup
\mathcal C_G
\]

其中：

\[
\mathcal C_R=\text{旧 Radar Head proposals}
\]

\[
\mathcal C_G=\text{独立 GRU State proposals}
\]

特别注意：

**不要使用刚刚失败的 residual B。**

GRU 分支应该回到上一轮独立的 `recurrent__state` readout，因为它才是在不污染 Radar score 的情况下读取 GRU 状态。

每个 candidate 保存：

```text
time
source ∈ {radar, gru, both}
radar_score
gru_score
```

如果 Radar 和 GRU 的两个峰时间非常接近，例如：

\[
|t_R-t_G|\le0.36s
\]

只建立一个 Candidate Episode：

```text
source = both
```

而不是重复建立两个。

---

# 第三阶段：先搜索 Candidate Pool 的合理 operating point

这里不要一上来规定：

\[
8FP/min
\]

因为 Candidate Pool 本来就是高召回阶段。

在 **fit 数据上**分别扫描 GRU threshold，Radar Head 保持现有输出，计算 union：

\[
Recall(\mathcal C_R\cup\mathcal C_G)
\]

对应：

\[
CandidateDensity/min
\]

真正想得到的是一条：

\[
\boxed{\text{Candidate Recall–Density Curve}}
\]

而不是 Proposal Recall–FP 曲线。

例如寻找：

\[
80\%,\ 85\%,\ 90\%,\ 95\%
\]

Recall 分别需要多少 candidate/min。

然后选择一个 operating point：

> **在达到至少约 85% Candidate Recall 的前提下，Candidate Density 最低的点。**

不要先规定必须 8 FP/min。

因为后面还有 Verify。

例如如果最后发现：

\[
Recall=88\%
\]

时需要：

\[
18\ candidates/min
\]

这未必是坏事。

只要 Verify 能把：

\[
18/min
\]

最终过滤成：

\[
4\sim8/min
\]

这套两阶段架构就是成立的。

反而如果即使放到：

\[
30/min
\]

Candidate Recall 还只有 70%，那说明 Proposal 本身仍然不够，应停下来改上游。

---

# 第四阶段：Candidate Filterability Diagnostic

这是我认为下一轮实验的核心。

先不要直接训练正式 Sequential Verify。

先回答：

\[
\boxed{
\text{进入 Candidate Episode 后，真假 candidate 到底能不能被区分？}
}
\]

对于每一个 candidate \(t_p\)，从它产生以后继续 causal rollout：

\[
t_p,\quad
t_p+1,\quad
t_p+2,\quad
\dots
\]

按照你当前 0.18s/update，大约观察：

\[
K=6
\]

个 update：

\[
6\times0.18
=
1.08s
\]

也就是 Candidate 最大观察约 1 秒。

对每一个 update 保存：

\[
h_t
\]

\[
RadarToken_t
\]

\[
z_t^R
\]

以及已有 Event representation。

不要使用未来窗口一次性编码，必须保持 causal。

---

## Candidate 的标签

一个 candidate 如果最终能够与 GT boundary 在容差内一对一匹配：

\[
|t_p-t_{GT}|\le0.72s
\]

先标为：

\[
y=1
\]

其余：

\[
y=0
\]

但是如果第一阶段发现 command/onset 有明显偏移，就应该同时保存：

```text
protocol_label
physical_onset_label
```

不能偷偷修改 GT。

这样后面可以分别回答：

> 模型是在预测数据集定义的语义边界？

还是：

> 模型是在预测真正的身体变化？

---

# 第五阶段：不要直接把所有特征拼起来

之前 Combined 已经给了教训：

> 信息多不等于好。

所以 Candidate Verify diagnostic 应做非常清楚的消融。

我建议只比较三种：

\[
\text{State Evidence}
\]

表示 Candidate 新状态与 Active 状态是否越来越不同；

\[
\text{Stability Evidence}
\]

表示这种新状态是否持续存在，而不是一次瞬时波动；

以及：

\[
\text{State + Stability}
\]

而不要刚开始就把：

- maturity；
- difference；
- evidence；
- consistency；
- source；
- Radar；
- GRU；
- history；

全部塞进去。

真正要验证的是：

> **GRU 假峰虽然变化大，但它们是不是无法形成一个持续稳定的新状态？**

这其实非常符合你的原始直觉：

\[
\boxed{
maturity是前提，
difference和stability才决定是否真正切换
}
\]

---

# 第六阶段：我建议这样定义 Candidate 状态

Proposal 出现时：

\[
t=t_p
\]

保存当前 Active reference：

\[
A
\]

然后初始化 Candidate：

\[
C_{t_p}
\]

之后每个 update：

\[
C_t
=
Update(C_{t-1},F_t)
\]

但是：

\[
A
\]

在 Candidate 验证期间保持冻结。

于是每一步都可以比较：

\[
d_t
=
1-\cos(C_t,A)
\]

或者通过一个小 semantic projection 得到：

\[
d_t^{sem}
\]

真正的事件转换应该表现成：

```text
difference:
低 → 高 → 持续高

stability:
低 → 高 → 持续高
```

而 walking 内部的一次摆腿更可能：

```text
difference:
低 → 高 → 又低

stability:
一直不够稳定
```

这就是 Candidate Verify 真正应该解决的问题。

---

# 第七阶段：先做 Probe，不立刻做 Sequential Accumulator

Candidate Episode 建好后，先训练一个非常小的 diagnostic classifier。

例如每一个 maturity \(k\)：

\[
k=1,2,3,4,5,6
\]

分别训练：

\[
P(y=1|E_{1:k})
\]

看看随着观察时间增加，真假 candidate 是否越来越可分。

最终报告：

| maturity | 观察时长 | AP | AUROC | Recall@固定candidate precision |
|---:|---:|---:|---:|
| 1 | 0.18s | ... | ... | ... |
| 2 | 0.36s | ... | ... | ... |
| 3 | 0.54s | ... | ... | ... |
| 4 | 0.72s | ... | ... | ... |
| 6 | 1.08s | ... | ... | ... |

这个结果特别重要。

如果出现：

\[
AP_{0.18s}=0.35
\]

但：

\[
AP_{0.72s}=0.75
\]

那就直接证明：

> 单个时间点没法判断，但随着 Candidate 持续观察，新事件逐渐稳定之后就能判断。

这会非常有力地支持你论文里的：

\[
\boxed{\text{Sequential Evidence}}
\]

设计。

---

# 第八阶段：为 Candidate Verify 设一个真正有意义的成功条件

我建议不要再用单纯 AP 做 Go/No-Go。

真正关注两个量：

\[
Recall_{before}
\]

也就是 Candidate Pool Recall，

和：

\[
Recall_{after}
\]

也就是 Verify 后真正保留下来的 Recall。

同时看 candidate density：

\[
D_{before}
\]

和：

\[
D_{after}
\]

一个理想结果例如：

\[
Recall_{before}=88\%
\]

\[
D_{before}=20/min
\]

经过 Verify：

\[
Recall_{after}=83\%
\]

\[
D_{after}=6/min
\]

那么只损失：

\[
5pp
\]

Recall，却过滤：

\[
70\%
\]

候选。

这就是成功。

我建议预注册一个第一阶段门槛：

\[
\boxed{
\text{False Candidate Reduction}\ge50\%
}
\]

同时：

\[
\boxed{
\text{Recall loss}\le5pp
}
\]

如果做不到，就不要进入正式 Sequential Evidence Accumulator。

---

# 第九阶段：只有 diagnostic 成功，才正式训练 Sequential Verify

如果前面的 maturity 曲线证明：

> 观察时间越长，真假 candidate 越容易区分，

这时候才进入你 v2.5 真正的 Candidate Verify：

\[
e_t=
f(
difference_t,
stability_t,
evidence_t
)
\]

然后累计：

\[
S_t
=
\lambda S_{t-1}+e_t
\]

或者采用你方案里已经定义好的 Sequential Evidence 形式。

最终形成：

\[
Reject
\]

或者：

\[
Confirm
\]

其中：

- maturity 只控制“什么时候允许判断”；
- difference / stability / evidence 决定“判断什么”；
- Sequential Accumulator 决定“是否积累到了足够证据”。

这样职责就非常干净。

---

# 最终的分流逻辑

整轮结束后，根据结果决定方向：

| 结果 | 结论 |
|---|---|
| command GT 与真实 onset 大量错位 | 先处理标签/边界定义 |
| Candidate Pool Recall 能到 85–90%，Verify 可过滤 ≥50% 假候选 | 正式进入 Stage3B/3C |
| Candidate Pool Recall 高，但 Verify 分不开真假 | 当前表示缺少“新状态形成”信息，需要 transition supervision / RD |
| Candidate Pool 本身 Recall 就上不去 | 回到 Encoder / MD 可观测性 / RD 输入 |
| GRU supplementary 几乎不增加有效 Candidate | 放弃 GRU 补充路径，保留旧 Radar |

---

## 我认为现在最值得让 Codex 先执行的任务

不是直接训练模型，而是先执行这两个诊断：

\[
\boxed{\text{A. Physical Onset Audit 数据导出}}
\]

以及：

\[
\boxed{\text{B. Radar + GRU 双源 Candidate Pool Recall–Density Sweep}}
\]

B 尤其重要。

因为只要我们得到：

\[
Recall_{Candidate}
\quad\text{vs}\quad
Candidates/min
\]

这条曲线，就能第一次真正回答：

> **你的两阶段 Proposal → Verify 架构，到底有没有足够的“高召回候选空间”。**

如果 Candidate Pool 能以可接受的密度覆盖 85%～90% GT，我会认为应该停止折腾 Proposal，正式把研究重心转向 **Candidate Episode + Stability + Sequential Evidence**。

如果连 Candidate Pool 都做不到，那就还没有资格进入后面的 Semantic Verify。
# v2.6.1：双状态联合预训练首轮设计

日期：2026-10-15。状态：**设计与输入清点完成，模型与执行器待实现，正式训练 0 步**。

本轮依据用户对双状态方案审阅建议的认可，完成联合预训练的设计与实施准备。本文的数值是下一轮候选协议；不把未实现的设计写成已通过检查或已获得实验收益。原 v2.6 方案、P0/P1/P1-C、旧权重和旧输出保留。

## 1. 要回答的问题

在同一冻结 MD-GRU 表示之上，让运动预测和语义分类共同训练一个共享 Adapter，是否比单任务训练得到更有用的共享表示？

首轮只检验**共享 Adapter 的多任务学习**。它不等价于已经验证 Encoder 主干联合微调，也不验证自主事件形成。Event Former 暂时固定权重；其状态会随 Adapter 输入变化。若结果支持共享学习，再另立协议开放 Event Gate/Cross-Attention/LoRA 或 Encoder 主干。

P1-C 的 Motion-only MLP 也读取已有 MD-GRU 历史，不能将其解释成完全没有历史。现有证据只是不支持额外冻结 E 的净点预测收益，不能据此否定历史或语义任务。

## 2. 首轮结构与参数边界

```text
已有完整连续 MD-GRU 冻结缓存 F_t：[4,256]
                 │
       共享残差 Adapter S_phi
                 │ Z_t：[4,256]
          ┌──────┴────────┐
   token masked mean      冻结 Semantic Event Former
   GRU Dynamics D_t       E_t：[8,896]
   残差预测头              已有结构的语义投影 + 14 类分类头
   h=1,3,5                 当前前缀动作代理分类
```

- Adapter：`Z = F + W_up(GELU(W_down(LayerNorm(F))))`，256→64→256；无 dropout；`W_up` 权重和偏置为零，初态逐元素恒等。名义参数 33,600。
- A：单层 GRU，input=256、hidden=256；只在 token 维 pooling，不在时间维提前 pooling。Dynamics 从固定上下文起点递推。
- A 预测头：`concat(D_t, m_ref_t)` 的 512 维输入，Linear(512,256)→GELU→Linear(256,768)，输出三个 256 维残差；末层零初始化。首轮预测均值，不训练方差头。
- B：复用 Qwen2.5-0.5B 前四层、Cross-Attention、Gate、初始 Event Tokens 与固定 prompt，全部冻结并 eval。语义投影和 14 类 classifier 从已有 Stage2 快照加载，在 B-only/A+B 中训练。
- 所有模型的冻结部分禁用 dropout。不能在整条 B 分支外套 `no_grad`/`inference_mode`；B 对输入 Z 和递归状态的导数必须保留。冻结 prompt K/V 可缓存，更新中使用现有 clone 机制。
- 首轮不加入 G/L 拆分、A→B 输入、B→A 修正、Keep/Birth、Reset Policy、Commit/Memory 或 Caption 训练。
- 原始 Stage1/Stage2 与 Qwen 不原地改写；每臂构造独立模型，按组件名固定初始化 RNG，保证共同组件逐元素相同。

## 3. 对照与可训练集合

| 臂 | 可训练参数 | 原生预测评价 | 原生语义评价 |
|---|---|---|---|
| M：Motion-only MLP | S_phi + 无新增递归状态的 MLP | 是 | N/A |
| A-only | S_phi + GRU + A 预测头 | 是 | N/A |
| B-only | S_phi + B 语义投影/classifier | N/A | 是 |
| A+B | S_phi + 与 A-only 相同的 A + 与 B-only 相同的读出 | 是 | 是 |

M 输入为 `concat(mean_normalized(Z_t), m_ref_t)`，Linear(512,564)→GELU→Linear(564,768)，末层零初始化。A 的 GRU+预测头名义 723,456 参数，M 的 MLP 名义 723,252 参数，均另加同一 Adapter；以实现后实际枚举为准。M 仍继承冻结 Encoder 的历史。

这四臂匹配样本、主训练步数和共同组件初始化，不声称总参数、FLOPs 或训练耗时完全相同。A+B 比单任务模型增加另一任务分支，必须分别报告总参数、可训练参数、时间和显存。Persistence 与不调参的线性趋势同时保留。

另做相同容量、相同预算的**冻结共享表示探针**，避免把 A+B 原生读出能力直接当成共享表示增益：每个完成模型固定 S_phi，以 `mean_normalized(Z_t)` 训练独立的线性残差预测探针和 14 类线性分类探针。原始 F 也训练同样探针。B-only 的预测和 A-only 的语义只由这些探针报告，不填造不存在的原生头结果。

## 4. 数据、标签与因果上下文

### 4.1 保留的数据总体

沿用 P1-C 的 18 条 fit、6 条 validation 录制：fit 受试者 01/02/03/04/07/08/09/13/19，validation 为 05/12/16，每人两传感器；09 使用 set001，其余为原清单选定会话。test 数组不读取。

本轮服务器只读核验了 24 个缓存 SHA、24 个原 NPZ SHA 和对应命令源 SHA，标签 ID 在两种角色均覆盖 0–13。共同预测起点 fit=61,299，validation=20,075；其中当前 Update 可用于单一代理类别监督的起点分别 59,540、19,609。逐录制证据见仓库 `.superpowers/sdd/2026-10-15-v261-preparation/input-inventory.json`。这些数字是输入清点，不是新训练结果。

标签来自命令时间轴转换，仍命名为 `command_class_proxy`，不是经过人工核验的当前真实动作或真实事件实例。语义类别需要同时满足：两帧 ID 一致且属于 0–13、观测有效、当前 Update 不混合。未通过的样本只屏蔽 CE/语义保持项，**不删除合法预测监督或改变状态更新**。本轮已逐项比较 phase1 与 expanded 的 `label_map.json`，14个 ID→命令文本对应关系一致，两份摘要已保存；实现准备时重新核验并并入源锁。

### 4.2 固定上下文，不使用命令边界 Reset

首轮不复用 Oracle E，也不使用命令边界重置 A/B。每个评价/训练起点 t 的可见上下文为所在连续观测段内的 `max(segment_start,t-31)..t`，最多 32 个 Update（5.76 秒）。这里 segment_start 仅由观测起始、缺测或时间不连续确定，不能由动作类别变化确定。

F_t 来自原冻结 Encoder 的连续历史；32 是新增 A/B 的固定上下文，**不是重新搜索 W，也不是 P1-C 的完整事件历史实验**。这一上下文限制及与部署长期状态的差距必须写入结果报告。

- A/B 在该上下文开始初始化；初始 maturity 不依赖事件结束时间。
- `maturity_j = min(observed_age_j * 0.18 / 1.2, 1)`，age 从本次可见上下文起点计数，与旧 P1 回放的因果公式一致。
- 前 `max(0,length-16)` 个 Update 使用当前权重无梯度 burn-in，再 detach **状态**；最后最多 16 个 Update 保留递归梯度，损失在最后起点 t 计算。B 参数冻结不等于 B 输入梯度被截断。
- 每个 optimizer step 使用当前 Adapter 重算 burn-in；不跨更新缓存学生 E/D，避免过时状态。burn-in 也不得使用未来 F、标签或完整窗口长度比例。
- 训练、验证及保存权重重放采用相同上下文规则。未来后缀/标签扰动不得改变当前 Z/D/E/预测/logits。
- 混合 Update 照常更新状态；不从下一个纯 Update 借标签。三种 Horizon 共用原 P1-C 物理合法起点，未来标签不作预测输入。

## 5. 固定预测目标与具体损失

### 5.1 运动目标

`m_ref_t = (masked_mean(F_t)-mu_fit)/std_fit`。复用原冻结参考 F、pooling 和 fit-only normalization，并锁定 SHA；不能用可训练 Z 生成目标。首轮不用 EMA。即使后续开放主干，也需要独立冻结参考分支与相同评价坐标系。

三个残差头：

\[
\hat m^{ref}_{t+h}=m^{ref}_t+\Delta_h(D_t,m^{ref}_t),\qquad h\in\{1,3,5\}.
\]

M 将 D_t 换为当前 pooled Z。统一目标空间使零残差精确对应同一 Persistence。各头末层全零时上游梯度暂为零是正常现象；不能用第一个更新前 Adapter 的零预测梯度误判断图。

令 \(R_h\) 为所有 fit 共同起点上的等录制平均 Persistence MSE，固定且大于零。逐样本：

\[
\ell_{A,t}=\sum_{h\in\{1,3,5\}}w_h\frac{\|\hat m_{t+h}^{ref}-m_{t+h}^{ref}\|_2^2}{256R_h},\quad
(w_1,w_3,w_5)=(0.1,0.3,0.6).
\]

先逐录制对本批合法样本求平均，再等录制平均，得到 \(L_A\)。h=5 为主要指标，h=3 为次要，h=1 为短期对照；不因验证结果更换主指标。

NLL 仅作为**共同固定方差的工作模型诊断**：每 Horizon 使用 fit-only Persistence 残差二阶矩估计逐坐标方差，裁剪到 `[exp(-6),exp(2)]`；所有臂和 Persistence 共用。这不检验模型学习的概率校准，也不直接比较 P1-C 学习方差 NLL 的绝对值。首轮不同时研究均值、方差和 Reset 似然。

### 5.2 语义与保持

\[
\ell_{B,t}= -\log p_{new}(y_t\mid E_t)/\log 14.
\]

只对语义合法起点逐录制平均得到 \(L_B\)，首轮不加类别重权、不使用属性/骨架/速度/生成 Caption；另报告逐类召回和 Macro-F1。

保持教师为独立冻结旧快照。教师与学生使用**相同可见上下文、初态、maturity 和初始化位置**，教师读取原 F，学生读取 Z；都不能读取未来 Commit。记旧冻结语义投影为 P_old、旧分类器为 C_old，归一化表示为 z：

\[
\ell_{pres,t}=1-\cos(P_{old}(E_t^{new}),P_{old}(E_t^{old}))
+KL(p_{old}(E_t^{old})\|p_{old}(E_t^{new})).
\]

KL 使用温度 1，类别求和；教师侧 stop-gradient，学生 E 到 Adapter 的路径保留。只在当前语义合法样本上等录制平均为 \(L_{pres}\)。这保护旧前缀语义读出，不代表已保护完整 Caption/检索/Memory 能力。

\[
L_M=L_A,\quad L_{A-only}=L_A,\quad
L_{B-only}=L_B+0.1L_{pres},\quad
L_{A+B}=L_A+L_B+0.1L_{pres}.
\]

保持约束是 B 任务训练包的一部分，因此 A+B 对 A-only 的差别包含新增语义监督与保持教师，不能将其全部归为纯 CE 的收益。首轮不调这些权重。记录各项数值、对共享 Adapter 的梯度范数及两任务梯度余弦，定位冲突但不自适应修改权重。

## 6. 候选训练预算与评估

预算在实现后 preflight 前写入不可变配置；本轮没有执行以下更新。

- 主训练：4 臂 × seeds `[42,43,44]` × 1,000 个**成功** optimizer step = 12,000；每步两个独立上下文。
- 每 seed 的 1,000×2 表在训练前固定，并由四臂共用：第一个样本等概率抽 fit 录制、再抽当前语义合法的共同起点；第二个样本等概率抽录制、再抽全部物理共同起点，允许混合。首样本保证 B 任务存在合法信号，第二样本保留混合监督；原始总体与实际采样分布分别报告。
- 所有臂：AdamW lr=1e-4、weight_decay=.01、clip_norm=1；不追加步数、不根据 validation 调参。AMP/GradScaler 只用于资源控制，成功 step 的判断以真实参数更新为准；最大尝试 1,100 次，未达 1,000 则记失败，不能宣称完成。
- checkpoint 为 step0/100/300/1000；只比较 final1000，不以验证集选择最佳 checkpoint。
- 探针：4 个最终模型/seed + 原始 F/seed，共 15 组，每组独立分类/预测线性探针各 200 步、batch64、AdamW lr=.001、wd=.01、clip1，共 6,000 个额外低成本更新；与主训练分开计数。每个batch由32组主采样规则的“语义合法+物理合法”配对构成，所有表示共用同seed的探针表。预测探针为Linear(256,768)零初始化残差头，使用同一L_A；分类探针为Linear(256,14)，仅使用L_B，不带保持损失。探针模型、优化器和数据均不得回写表示。
- smoke：四臂各最多 5 次独立更新，共最多 20 次；丢弃权重，不续接主训练。smoke 的 seed 固定 1042，不能代替某个正式 seed。

共享前向冻结部分 eval；Adapter/头不含 dropout。GPU snapshot 为 RTX2080Ti 11,264MiB（只读查询时占用约270MiB），**尚未实测新模型的显存/耗时**。先用真实 Qwen 的最大32步上下文、16步梯度、batch2 做 smoke 与峰值检查。要求 PyTorch allocated 峰值≤9GiB，并记录设备总使用、耗时和估算全轮成本。OOM、断图、数值非有限或预算不成立时停止，不偷偷缩 W/batch/模型后继续同一实验；修订协议另建版本。

评价逐起点保存预测与 logits，先等录制、再等受试者；报告逐 seed、逐人和 seed 均值/标准差。h1/3/5 的跨代理边界、干净内部、mixed_current/future、可见上下文长度分别列示。分类在每条录制的全部语义合法起点上计算固定14类 Macro-F1，再等录制→等受试者平均，缺失类规则固定为 zero_division=0 并列 support；不将不同长度录制的混淆矩阵直接相加后冒充等录制指标。由于按 Horizon 保留共同起点，语义尾部不在本轮主评价总体内。

预测报告各人的 `1-MSE_model/MSE_persistence` 后宏平均，不用两列宏 MSE 直接求 Skill。历史 P1-C 是背景；新方法与同协议重训 M/A 对照比较，不将不同预算、上下文或方差策略的差异归因于联合学习。

## 7. 成功标准与结论边界

先通过工程检查，才允许执行候选主训练：

1. 数据、目标、类别文本映射、旧权重与源码锁定；物理目标和语义 mask 分离。
2. Adapter 初态恒等、预测初态精确 Persistence、语义初态与同上下文教师一致。
3. 独立 \(L_A\) 和 \(L_B\) 均能到达共享 Adapter；冻结参数、参考目标与教师无梯度且 SHA 不变。预测通路在零末层经历有效更新后检查。
4. 真 Qwen checkpoint backward、prompt cache 隔离、CPU/原生相关回归、实际资源 smoke 通过；日志可验证 optimizer 成功更新数。

研究结果不使用单 seed 的微小差异放行：分别比较 AB vs A、AB vs B、A vs M，以及统一探针中的共享表示。预先报告主差值方向在三 seed 中是否一致，并完整展示所有人；三 seed 同向仍不等价于统计显著或达到收敛上限。

仅当共享探针或原生任务出现跨 seed 一致的互补收益，且另一任务的退化及资源成本可接受时，才有理由设计更大规模主干/Former 微调。不得事后发明可接受退化阈值并称作预注册通过；若存在权衡，呈现实际差值，由研究目标决定后续实验。

即使联合表示有收益，也只完成该阶段的问题。真实时钟/实例边界 G0 仍未通过，不能据此启动正式自主 Reset。未来 Reset 必须确保 resettable 状态实际影响预测/语义，否则总损失中写了 future 项也未必存在 policy 梯度。

## 8. 复用、实现范围与交付

- 复用 `v26/source.py` 的权重/来源校验、`targets.py` 的冻结 pooling 与物理预测契约、`losses.py` 的 NLL/显式归约，以及旧 Event Former 的 `stream_update`。
- 不直接复用 `stages/parrad_stage2.py._stream_view` 的完整窗口 maturity；不直接复用带 `@torch.no_grad()` 的 `replay_events` 训练；不使用保存的 Oracle E 充当可训练路径。
- 新增隔离命名空间 `src/continuous_mmwave/v261/`；不修改旧 v26 模型或旧实验胶囊。工程不是 Git checkout，用源码 ZIP/清单/SHA 封存，不运行 git commit。
- 实现交付需有 protocol/输入清单/采样表、具名可训练参数、组件初值 SHA、逐项梯度与资源检查、独立 smoke/主训练/探针计数、全部逐点预测/标签/mask、权重重放和本地独立指标复算。
- 实施清单见 `docs/superpowers/plans/2026-10-15-v261-joint-pretraining.md`。本次只完成文档和只读输入清点，不宣称以上软件已经存在。

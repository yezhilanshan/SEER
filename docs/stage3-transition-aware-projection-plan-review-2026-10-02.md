# Transition-aware Semantic Projection Pilot 方案审查

审查日期：2026-10-02。审查对象：本轮用户粘贴的方案。

结论：优先尝试 Projection-only 适配的方向合理，但原文存在影响训练目标、对照公平性和结果归因的问题，不能按原文直接启动训练。本轮只做方案审查；未修改生产代码、checkpoint、controller 或阈值，未读取 validation/test 数据，未启动训练。

## 1. E > z 支持干预假设，但没有证明信息被 Projection 丢掉

本地结果报告确认 natural age6 的 E/z AUROC 为 .6190/.5659，age7 为 .6489/.5753。

然而 E 的读出从 7168 维完整状态残差开始，z 的读出从 128 维归一化 key 残差开始；两者分别经过固定随机投影，再使用 cosine/L2 和 Logistic。131 参数只匹配最终分类器容量，并不让前端特征映射等价。

因此应表述为“当前协议下 E 的 transition 信号更容易读出，值得检验 Projection 适配”，而不是已证明 Projection 压掉了信息。E 的当前 probe 分数不是 z 的严格信息上界，也不是必须恢复到的理论目标。

来源：`parrad_stage3_sequence_features.py::pack_history_block/features_from_pack`；现有结果报告已明确 E 不是严格信息上界。

## 2. Projection 同时投影文本，事件 anchor 不足以覆盖它的职责

`models/semantic.py::SemanticProjection` 明确使用事件/文本共享 query 和 MLP。`stages/parrad_stage2.py:263` 同样用它生成 text_key。因此只更新该模块，也会更新文本侧坐标。

若采用“不重跑文本，至少 anchor E”的退路，只能约束采样事件上的输出变化，不能验证文本 query 或已有 Memory key 的兼容性。新 query 配旧 Memory key，以及旧 query 配新事件 key，都需要明确处理。

修正建议：

- 缓存冻结 adapter 的文本 hidden states 和 mask，训练时仍通过当前共享 Projection 生成 text_key；不能用缓存的旧 text_key 冒充原 Stage2 alignment。
- 用独立的 clean label-run semantic rehearsal 缓存覆盖 stream、commit 和文本侧；Candidate update 的 clean GT 标签不自动意味着递归 E 是纯单动作表示，不能直接机械套用 Stage2 标签。
- 保留 duplicate-label-safe 的多正例 alignment，并对事件、commit 和文本输出施加旧 Projection anchor。
- 固定新旧事件/文本交叉检索及同版本检索的评估。若以后采用新 key，另行定义已有 Memory 的重建或版本策略；本 pilot 不自动替换生产模型。

另外，现有 caption logits 直接使用 commit Event Tokens，而不经过 Semantic Projection。上游固定时，原 caption loss/生成结果对 Projection 修改不敏感，不能用它通过来证明语义空间保留。

现有 `parrad_stage2_diagnostics.py::alignment_diagnostics` 是 Event→Text 的实例对角 top1；同动作重复文本会造成不恰当的实例负例。新的 retention 验收应明确 Text→Event/双向、多正例或动作级指标，而不能直接把该旧 top1 当作所需指标。

## 3. 临时 relational head 与最终 probe 的信息接口不同

训练 head 可以看到 z_A、z_C、绝对差和乘积。当前最终 probe 看到的是固定投影后的有符号残差，以及 cosine/L2；并不直接看到两侧绝对 key、逐维绝对差和乘积。

即使临时 head 只有一层，它也有更多类型的输入特征；若带隐藏层，差异更大。它可以主要学习读出而非促使 Projection 改善最终接口。丢弃它后，收益不一定保留。

修正建议：以与最终 probe 相同的残差/cosine/L2 接口上的小型训练读出作为主要 transition 梯度来源；丰富关系 head 若保留，只作为辅助。另一种可行做法是保留原训练 head，但 checkpoint 选择必须使用 dev 上重新拟合的冻结 probe，而非临时 head 的 BCE/AUROC。

必须包含 epoch0 对照和冻结旧 Projection、仅训练临时 head 的对照，并保存 Projection 梯度范数及参数变化。这样才能识别收益主要来自 head 的情况。

原 No-Go 1 的结论应收窄：一次指定损失/训练接口失败，只说明该适配策略没有达到预注册门槛，不能直接证明 Projection-only 已不足，也不能直接推出需要修改 Event Former。

## 4. Candidate 基础等权不等于最终等权

原公式先设 b_it=1/N_i，再乘 age/class balancing 系数 c_age,y。最终 Candidate 的权重是：

    W_i = sum_t (c_age(t),y(t) / N_i)

它不一定为 1。例：Candidate A 有两条更新，系数为 1 和 5，则总权重为 3；Candidate B 有一条更新，系数为 1，则总权重为 1。两者基础权重都为 1，最终仍相差三倍。

如果类别平衡按未加权行数计算，而不是按 b_it 的质量计算，甚至不能保证基础加权意义下的 age/class 平衡。

修正建议：明确最终优化目标和约束优先级。若采用 age/class 平衡，按基础加权质量计算系数、固定各 age 的总质量，报告每 Candidate 的最终权重分布，并预先设定允许的权重集中程度；不得再声称最终 Candidate 严格等权。若必须严格等权，则需先固定 Candidate 内归一或受约束配权的算法，并承认一般情况下不同时保证精确 age/class 平衡。

没有某一类别的 age bucket 应明确标为不可平衡并使用预先固定的排除规则，不能静默补样或利用 dev/held 估计权重。

## 5. 必须区分 Projection 划分与最终 probe 划分

现有 `parrad_stage3_radar_readout.py::fold_masks` 把 outer-train subjects 的所有非 set003 行放入拟合集，包含 set002。它可以复用于原最终 probe，不能直接复用于新 Projection 训练。

建议把两个层次明确写成：

| 阶段 | set000/001 | set002 | set003 | held subject |
| --- | --- | --- | --- | --- |
| Projection 适配 | 更新 Projection/head | checkpoint selection | 不使用 | 不使用 |
| 选择阶段冻结 dev probe | probe fit/变换估计 | dev scoring | 不使用 | 不使用 |
| 模型锁定后的最终 probe | fit | fit | threshold calibration only | evaluation only |

这保留最终旧 Complementarity Protocol 的 set000–002 probe fit 口径。新旧 Projection 的最终 probe 都必须在同一 Candidate IDs、标签、端点、seeds、尺度估计和 solver 设置下拟合；最终阶段不再反过来改 checkpoint。

如果决定最终 probe 也只在 set000/001 fit，则必须重跑所有旧基线，原表的 .6190/.5659 等不能直接作新旧 paired 对照。

每个 outer fold 必须分别从原 checkpoint 初始化一个 Projection，禁止用全 subject 训练一个 g_new 后声称 held evaluation。两个 sensor 绑定同 subject/session；所有标准化、采样和 loss 配权只从对应 fit 数据估计。

## 6. Go/No-Go 尚未真正预注册，held 也不是全模型未见数据

“基本一致提高”“相当一部分差距”“没明显下降”都没有数值定义；原 .575→.59 的例子也没有对应的统计稳定性或训练不确定性。

正式训练前需要固定：训练 head 结构、学习率、epoch 上限、训练 seeds、loss 系数及允许的有限候选集合、dev checkpoint 选择规则、失败/缺失指标处理、语义 retention 非劣门槛，以及 transition 的最小 paired 提升与各 subject FPR 恶化容忍度。

建议以 natural age6/7 K1 为主要端点，age4/5 和 natural-terminal 为支持性端点。报告每 subject 的 paired AUROC/AP、两工作点 recall/FPR 和计数。以 Candidate 为单位做同录音内成组重采样可描述候选级不确定性，但仅三个 subjects，不能把更新行或投影 seeds 当独立受试者来证明总体泛化。

当前 Stage2 上游见过这三个 TRAIN subjects。新 g 的 held subject 对新适配/新 probe 是留出的，对整个模型不是从未见过。因此通过只允许进入下一阶段研究，不允许直接替换生产模型或宣布独立 subject-generalization 成立。若后续依据本轮 held 结果改超参数，必须标记为后续探索，不能继续称其完全 untouched。

## 可直接保留的部分与执行可行性

保留：只更新已有 query/MLP；保持128维；从当前 checkpoint 初始化；真实 Proposal→Fork→Candidate；Frozen Active 参考；沿用 clean update 标签与排除规则；natural age3–7 训练；保持旧 calibration 算法；validation/test 继续不读；不同时修改 controller/Former。

现有提取代码把完整 E/reference E 展平保存为7168维，并非只保存128维诊断投影；配置对应8×896。因此在核验源 NPZ、checkpoint/config SHA 和旧 z 重算一致性后，可 reshape 回完整 Event Tokens 离线训练 Projection，不必为了这个 pilot 重新回放上游。当前本地诊断目录主要是 JSON 结果，完整 NPZ 来源记录在远端 manifest；本轮没有读取或核验远端 NPZ，不宣称训练缓存已经验收。

age3–7 对应 Proposal 后 .36–1.08s，与现有 age=(current-proposal+1)、elapsed=(age-1)×.18s 一致。

已核对本地 Complementarity 报告、manifest、提取器、共享 Projection、Stage2 alignment、probe 特征/划分/配权实现；本轮未改训练代码，因此没有运行 pytest，也没有安装任何依赖。

下一步应先把上述修正及数值门槛纳入可执行方案，再启动训练。此审查不把存在问题的原方案视为已满足用户“如果没有问题就开始执行”的条件。

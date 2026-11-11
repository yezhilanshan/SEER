# Transition-aware Projection Pilot：问题解决方法

状态：从方案审查中提取并具体化的修订建议；本文件没有启动训练或修改生产模型。数值配置与验收门槛是建议的初版预注册值，不是实验事实或用户已经确认的标准。

对应审查：`stage3-transition-aware-projection-plan-review-2026-10-02.md`。

## 1. 用配对实验检验 Projection，不把 E 当理论上界

把实验问题固定为：“仅适配现有共享 Semantic Projection，能否改善 natural age6/7 的冻结 transition probe，同时保留原语义功能？”

- E/F 继续作冻结参考，不设“必须达到 E”这一理论目标。
- 每个 outer fold 从同一个当前 checkpoint 初始化，分别训练 g_new。
- 所有比较复用旧 closed-loop 轨迹、Frozen Active、Candidate IDs、标签和 cohort。
- 保存 epoch0 重算结果，要求旧 Projection 重算 z 与源缓存 z 在原协议容差内一致；不通过就先排查 checkpoint、reshape、dtype 和 Projection 实现。
- 最终重跑 z_old、z_new、F、E、F+z_old、F+z_new；同规则估计各自 fit-only 尺度与标准化，不强行共享新旧表示的数值尺度。

主结果解释为“当前协议下 transition 可读性改善”。失败解释为“该预注册适配策略未达到目标”，不能仅凭一次失败宣布 Projection-only 不可能有效。

## 2. 缓存三类语义输入，保护共享空间

保留事件/文本共享 query 与 MLP，以及128维输出；不拆分文本支路，不新增正式 Transition Projection。

在 outer-train subjects 的 set000/001 上建立独立 semantic rehearsal 缓存：

1. 原 Stage2 clean label-run 的 stream Event Tokens。
2. 同一 label-run 的 commit Event Tokens。
3. 对应原文本模板的冻结 adapter hidden states、有效 mask 和动作标签。

这些缓存只冻结上游；训练时仍对三类 hidden states 调用 g_new。旧 g_old 永久冻结，作为 teacher。不能只缓存旧 text_key 并声称完整复用了原共享 alignment。

训练损失建议为：

    L = L_transition + lambda_s * L_semantic + lambda_a * L_anchor

- L_semantic：复用 Stage2 同动作多正例双向 alignment，分别计算 stream↔text 和 commit↔text；保留 stream/commit cosine consistency。默认采用项目现有温度与 consistency 权重，在 manifest 写入实际数值。
- L_anchor：三类 rehearsal 输出分别求 mean[1-cos(g_new(X), stopgrad(g_old(X)))]，再对 stream、commit、text 三组等权平均；另外对实际 Candidate/reference 输出求 anchor，避免只保护完整事件分布。两类 anchor 等权平均。
- transition batch 与 rehearsal batch 每个 optimizer step 都参与，分别先求均值，避免缓存大小决定 loss 比例。rehearsal 采样在 fit 内按动作平衡，记录各 subject/session/sensor 覆盖。
- 建议第一版 lambda_s=1、lambda_a=1，不做无限超参数扫描。语义缓存不足时停止正式训练并报告缺失，不降级为“只有事件 anchor 也足够”。

GT label-run 仅用于语义 rehearsal；transition 正负例仍全部来自真实 closed-loop Candidate。不能把 rehearsal 的 GT-reset 状态当成在线 Candidate，也不能把当前 update 的 clean 标签直接当递归 E 的纯动作标签。

retention 必测：动作 probe macro-F1、Text→Event 和 Event→Text 的动作级/多正例 retrieval、正负 alignment margin、stream/commit consistency，以及新文本→旧事件、新事件→旧文本的交叉版本检索。新旧 semantic probe 必须使用相同 fit/eval 划分和容量，禁止只给新 z 换更强分类器。

原 caption logits 不经过 Projection，上游固定后不作为本 pilot 的有效 retention 证据。Memory 兼容性不合格时只保留研究 checkpoint；后续另行决定 key 重建/版本迁移，不能直接部署。

## 3. 用最终 probe 的接口提供 transition 梯度

第一版不使用能访问两侧完整 key、绝对差和乘积的丰富 relational MLP。

建议临时训练 head 为130输入、131参数的线性 Logistic，输入：

    phi = concat(P_seed @ ((z_C-z_A)/scale), 1-cos(z_A,z_C), ||z_C-z_A||/scale)

P_seed 复用项目实际矩阵生成规则；实现时按代码的矩阵方向与排列构造。scale 和输入标准化在 epoch0 的 fit 数据上估计并固定；公式中的 cosine/L2 必须实际复用现有特征定义。该接口的 Torch 实现需对照原 NumPy/Torch 冻结特征验证数值和梯度。

- 使用与最终 probe 一致的 K1 残差/几何特征；age/maturity/lifetime/decision 不输入。
- 可固定使用现有三个 projection seeds 对应的三个临时线性 head，平均三个 BCE，不挑选最好 seed；训练后全部丢弃。
- 每个 checkpoint 候选把 g_new 冻结，再重新拟合 dev probes；按 dev probe 指标选择，不按训练 head 的指标选择。
- 同时记录 Projection 梯度范数、参数变化、输出变化和 train/dev gap。

对照至少包括：epoch0 的 g_old；冻结 g_old、只训练同样临时 head；g_new+临时 head 的受限适配。head-only 的训练表现只作归因诊断，正式 old/new 比较都使用独立重拟合 probe。

## 4. 明确配权取舍：优先 Candidate 最终等权

建议优先保证 Candidate 最终等权，再使用有界 age/class 校正；承认校正后的 age/class 平衡是近似的。

只在 natural age3–7 的 clean fit 更新上计算，N_i 是经过范围和排除规则筛选后的有效更新数：

    b_it = 1/N_i
    M_ay = sum_{i,t: age=a, y=y} b_it
    c_ay = clip((M_a0+M_a1)/(2*M_ay), 0.25, 4)
    r_it = b_it*c_age(t),y(t)
    w_it = r_it / sum_{u in Candidate i} r_iu

于是每 Candidate 的 sum_t w_it=1；不再声称各 age/class 精确等质量。校正系数上限4与下限0.25限制 Candidate 内单个更新的相对放大；稀有类别不会无限放大。

loss 先在每 Candidate 内求 sum_t(w_it*BCE_it)，再对 Candidate 等权平均。若均匀抽 Candidate，不能再按更新数或权重重复抽样并双重加权。

若某 age 在 fit 中缺正类或负类，按预先规则排除该 age 的 transition 训练更新，重新计算 N_i；没有有效更新的 Candidate 不参与 transition loss。dev/calibration/held 不参与系数估计，不补样。

报告每 age/class 的原始数量、基础质量、最终质量，以及每 Candidate 总权重、最大单 update 权重。固定 age 的独立评估是必需的；删除显式 age 和近似平衡仍不证明隐含 maturity 捷径完全消失。

若研究目标坚持精确 age/class 平衡，则改用该审查中的质量平衡公式并接受 Candidate 最终不等权，或另做受约束配权可行性检查；不在本版同时承诺两个无法普遍兼得的性质。

## 5. 三种划分分别实现，避免直接复用旧 fold_masks

每次留下 subject02/03/04 中一名，另外两名为 outer-train；两个 sensor 绑定同 subject/session。

| 阶段 | outer-train set000/001 | outer-train set002 | outer-train set003 | held subject set000–003 |
| --- | --- | --- | --- | --- |
| Projection/rehearsal 训练 | 更新参数、估计训练权重 | 不更新 | 不使用 | 不使用 |
| checkpoint selection | dev probe fit/变换估计 | dev scoring 与 retention | 不使用 | 不使用 |
| 锁定后最终 probe | fit | fit | 只标定阈值 | 只评估 |

每折独立初始化并选 checkpoint。选择阶段尺度、标准化、semantic probe、class weights 只拟合 set000/001；最终 probe 在参数锁定后按旧协议重新用 set000–002 fit。set003 不用于挑 epoch、loss 权重或训练 seed。

dev selection 建议：先排除 retention 不合格的 checkpoint，再按 natural age6/7 K1 的 dev AUROC 平均值排序。先在每 age 对两个 dev subjects 等权平均，再平均两个 age，最后取三个固定 projection seeds 的 median；同分选更早 epoch。epoch0 必须参与，没有合格且改善的 checkpoint 就返回未通过，不强行选最后一轮。

## 6. 在训练前固定有限预算与 Go/No-Go

建议初版预算：AdamW；Projection lr=1e-4；临时线性 head lr=1e-3；weight_decay=1e-4；gradient_clip=1；最多20 epochs；训练 seeds1701/1702/1703。batch 每次最多16个均匀抽取的 Candidate，包含其有效 age3–7 更新；每步另取32个 rehearsal label-runs。最后不足一个 batch 正常计算实际均值。每 epoch 做 dev selection，三个 seeds 全部报告，不挑最好 seed。

主端点：natural age6/7 K1。支持端点：age4/5 K1、natural-terminal-min3 K1；K3 可作旧协议的支持比较。forced 晚期不参与 checkpoint 或 Go/No-Go 选择。

以下建议门槛在正式训练前锁定，不根据 held 结果修改：

| 项目 | 建议门槛 |
| --- | --- |
| 主端点 AUROC | age6/7 平均 paired macro提升至少0.02；三个 subject 的主端点平均 delta 均>0 |
| 主端点 AP | paired macro提升至少0.01 |
| 5% calibration 工作点 | paired macro recall提高至少3个百分点；每个主端点、每 subject 的实际 held FPR增加不超过1个百分点 |
| 10% calibration 工作点 | paired macro recall不下降；每个主端点、每 subject 的实际 held FPR增加不超过2个百分点 |
| 支持端点 | AUROC下降不超过0.01；不得只靠晚龄改善掩盖早期明显退化 |
| 动作语义 | macro-F1下降不超过2个百分点 |
| 双向及交叉版本 retrieval | 各指标下降不超过2个百分点，stream/commit分别报告 |
| alignment | 若旧正负 margin为正，新 margin仍须为正，且绝对下降不超过0.02 |

百分点是绝对差，不是相对百分比。AUROC/AP 为0–1尺度。以上门槛是项目工程标准，不表示统计显著或已由数据证明最佳。

聚合规则：每训练 seed、每 projection seed、每 subject、每端点保留完整 paired delta；主端点先平均age6/7，再按subject macro，取projection-seed median，最后取training-seed median。subject要求也按相同两层seed median判断；三个训练 seeds 至少两个达到主 AUROC与AP门槛。FPR/retention容忍度逐配置检查，不允许median隐藏某seed或subject的明显退化。

dev selection 使用同一语义容忍度和AUROC排序；不调用set003阈值，也不把需要calibration的held规则拿来选epoch。主端点或必需语义指标缺失、probe未收敛时标为不确定，不能视为通过。稀疏支持端点保留undefined和计数，不补held数据。

Go仅表示继续研究 Projection→正式 online Semantic Difference 的路线。F+z_new 是否支持融合，必须另外检查对F的paired AP/recall收益及逐subject实际FPR，不能由z_new改善自动推出。

No-Go按原因分为语义退化、transition无可靠提升、数据/优化不完整；先查梯度、head-only对照、dev gap与样本覆盖，再讨论Event State训练。不得自动重训Former。

这三个subjects对本次适配是留出的，但已被冻结Stage2上游见过。报告必须使用“TRAIN内适配留出评估”的范围；validation/test继续封存。通过不等于跨subject FPR已经受控，也不授权生产部署。

## 实施顺序

1. 固定配置、loss、权重算法、selection与上述门槛，保存版本与SHA。
2. 核验源manifest/checkpoint/config/NPZ，并重算epoch0 z；上游冻结参数与代码哈希检查通过。
3. 建立仅fit内的semantic rehearsal缓存及独立dev retention缓存；held retention缓存在checkpoint锁定后才建立并读取，禁止held输入参与训练或选择。
4. 验证分割隔离、feature接口一致性、Projection-only梯度、Candidate最终等权、多正例检索与checkpoint恢复。
5. 在interview环境运行相关pytest；先做一个fold的短smoke，验证梯度与冻结契约，再按固定预算运行三折三seeds。
6. 参数选择锁定后，重跑同cohort的old/new最终probe，使用原set003标定算法，产出paired结果与Go/No-Go判定。

本轮只生成修订方法文档，没有启动上述实施步骤。

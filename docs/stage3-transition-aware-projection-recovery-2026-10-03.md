# Projection pilot 中断后的处理建议

状态：根据真实训练日志、checkpoint 和 fit-only 梯度诊断制定的恢复建议；未修改训练配置，未启动新实验。

执行授权：用户随后回复“那请你开始吧”，批准下述有限恢复实验。原建议文字保留作为历史；实施状态在执行 ledger 中更新，不通过修改本协议重写已运行的实验。

固定本次预算：subject02为留出subject，seed1701；仅使用其余subjects的fit000/001和dev002。按顺序运行 rehearsal-only 1轮（lr1e-4、lambda_t=0、lambda_s=lambda_a=1），joint 3轮（lr1e-5，其余原配置），以及仅在该joint无合格改善时运行anchor10的joint 3轮。所有条件均从epoch0初始化；最终head-only对照采用对应条件同样轮数。各项原语义门槛保持不变。记录首批分项梯度范数和夹角，并隔离重放v1失败轮；重放不是候选科学结果。此诊断流程不自动启动新的三折全量pilot，也不读取calibration或held数值数据。

## 已确认的事实

- v1 在 subject02 / seed1701 的第6轮 dev 评估中停止；动作分类 probe 使用 FP64 LBFGS、400次迭代上限、gradient tolerance 1e-7，报未收敛。目前没有迭代数、终止梯度和目标变化的失败诊断，不能认定只是迭代上限不够。
- epoch5 的模型、优化器和随机状态完整；但 epochs1–5 均未通过语义保持。原配置原样恢复不解决这个研究问题。
- dev transition score 从 epoch0 的 .625848 提高到最好 epoch3 的 .641179，仍不足以接受新权重。
- subject03 的 commit Text→Event 在第1轮从 .857143 降至 .642857；第5轮新事件→旧文本 stream Text→Event 从 .910714 降至 .705357。退化不是第6轮数值错误才产生的。
- semantic loss 的 epoch平均值从 .641975 降至 .385540，而 anchor 从 .008706 升至 .038088。损失下降不代表保留旧检索排名；损失数值大小也不能代替梯度大小比较。
- 临时线性 heads 初始化为零，第一次反传时 transition 对 Projection 的梯度为零；在 teacher 相同的初始状态，anchor 梯度理论上也为零。初始 Projection 更新主要由新的 fit rehearsal alignment 推动。
- fit-only 诊断复用 epoch0 和真实 rehearsal，用代表性的平衡32样本，配合合成 pair 输入、零 heads，得到 semantic loss 1.18116、Projection 梯度范数5.98824。它不是原训练第一批的精确复现，仅说明旧模型对当前 rehearsal 目标存在明显非零梯度。

## 1. 先修评估失败的诊断与处理

保存每次动作 probe 的迭代数、目标变化、梯度最大值、有限性、fit标准差范围，并附上 branch/epoch/subject 和 fit数据指纹。使用失败轮的同一冻结输入重现；区分迭代耗尽、错误终止判据、特征数值病态，不通过降低正则/放宽门槛猜测修复。

dev probe 未收敛：记录指标为 undefined，并让该 checkpoint 不具备选择资格；继续有限训练预算。旧模型基线 probe 未收敛、必需数据缺失、非有限梯度则保留为明确的配置失败。最终必需 probe 未收敛仍标为 inconclusive，不能算通过。

在 dev 评估前保存优化器/RNG和模型的原子 checkpoint，明确标记评估是否完成。增加失败重现和恢复测试。若证据要求提高 solver 迭代上限，对 old/new 采用相同设置，记录协议版本；不静默修改v1。

## 2. 先做一个折、一个种子的有限诊断

仅使用原 fit / dev，不读取 calibration、held 数值、validation 或 test：

1. 从 epoch0 做一个 rehearsal-only 的短对照，核查不含 transition 梯度时是否同样伤害检索。保持 rehearsal采样、学习率和旧teacher不变；记录各项梯度及夹角。
2. 做 frozen-Projection head warm-up 对照，确认临时头优化正常；warm-up 不代表最终 probe 变强。正式 old/new probe 容量和拟合协议继续一致。
3. 做更保守的 joint 诊断，先只将 Projection lr 从1e-4降到1e-5，其余不变，最多3轮；若仍失败，再另建一个固定版本，仅将 anchor 权重从1改到10。数值是待预注册的有限候选，不保证有效。

不在这一步添加全容量 readout、改Former、反复试seed或放宽2个百分点语义门槛。若 rehearsal-only 同样退化，优先检查 rehearsal分布、文本模板覆盖及与原Stage2目标的差别；不能把全部损伤归因于 transition loss。

## 3. 用语义门槛决定是否扩大预算

新目标或学习率实验均从 epoch0 初始化，使用新输出目录；v1 历史和失败证据保留。不从已语义不合格的epoch5切换新目标并混称一个实验。

先要求所有必需的双向、跨版本检索、动作macro-F1、margin和consistency通过原retention门槛；然后检查独立重拟合的 dev transition probe 是否有收益。没有合格改善则选epoch0，不扩大三折三种子预算。

若较小lr和有界强化anchor仍无合格改善，记录为当前策略的No-Go。可另立方案评估 teacher关系分布蒸馏或约束优化，但不据此宣布 Projection-only不可能，更不自动重训Former。

## 4. 原模型继续保留

当前 adapted checkpoint 不用于生产，也不作为训练成功结果。最终结论仍受 TRAIN内适配留出范围限制；冻结Stage2见过这些subjects。

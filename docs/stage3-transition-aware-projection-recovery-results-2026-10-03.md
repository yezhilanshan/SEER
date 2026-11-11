# Projection 恢复诊断结果

真实数据执行时间：2026-10-03 11:54:38–11:58:36 +0800。状态 diagnostics_complete，进程正常退出。

## 结论

本次有限诊断完成；三个条件均选择epoch0，未扩大三折训练，未进行held/calibration数值评估，生产权重保持不变。结果支持“当前rehearsal适配目标也会导致语义漂移”的判断；不证明Projection-only不可能成功。

## 原故障重放

从v1 epoch5的AdamW及随机状态隔离重放第6轮。commit动作probe的fit输入及目标均有限，标准差范围 .07130–.10882；LBFGS耗尽400次迭代，最终gradient max为1.61456e-6，objective change为1.97603e-9，尚不满足原容差。两个dev subject使用同一fit分类器输入，因此出现两个相同的commit未收敛诊断，不能视作两个独立故障。

修复后此类dev指标记为undefined，该checkpoint不可选；流程继续完成有限预算。不放宽原1e-7/1e-9容差，不改变probe容量或400次迭代预算。

## 三个独立epoch0初始化条件

共同基线dev selection AUROC为 **0.62584790**。它是natural age6/7、两个dev subjects等权、三个固定projection seeds的规定聚合分数；不是held AUROC。

| 条件 | 轮数 | 最高dev AUROC | 相对epoch0 | 最差单项检索变化 | 选中epoch |
| --- | --- | --- | --- | --- | --- |
| rehearsal-only，lr1e-4 | 1 | 0.63677871 | +0.01093082 | -21.43个百分点 | 0 |
| joint，lr1e-5 | 3 | 0.62878757 | +0.00293967 | -15.18个百分点 | 0 |
| joint，lr1e-5，anchor10 | 3 | 0.62931636 | +0.00346847 | -5.36个百分点 | 0 |

表中的最高dev分数来自不合格的适配checkpoint，不能按收益接受。最差单项变化覆盖对应条件所有适配轮次、全部必需双向和跨版本检索；三个最差项均是subject03 commit Text→Event。joint lr1e-5还出现必需语义probe undefined，不是完整可接受评估。原门槛允许下降最多2个百分点。

anchor10明显减小该诊断内的退化，但仍没有达标。不能据此反复增加权重、扩大预算或宣布最终Go。

## 梯度证据与冻结检查

真实第一批的Projection梯度分项范数在三个条件相同：transition=0，semantic=1.87229943，anchor≈3.96e-9。零初始化heads与相同teacher造成该结果；首次更新主要由当前fit rehearsal alignment驱动。

关闭transition损失的rehearsal-only同样下降21.43个百分点，说明退化不能全部归因于transition目标。需要进一步核对当前rehearsal的数据分布、模板覆盖与原Stage2目标，而不是认为semantic loss下降就保留了旧检索几何。

所有条件的frozen teacher及head-only Projection不变。Source和依赖哈希验证通过；epoch0最大key误差2.235174e-7。validation/test继续封存。

## 验证与可追溯性

- 新增/相关测试：本地27通过；服务器27通过。完整服务器测试384通过，2项原有Gemma3 None_inv_freq失败；本次使用Qwen路径。
- 原始v1 loss/runner与新循环的两轮对照：model、AdamW全部状态和NumPy/Torch随机状态逐bit一致。
- 独立复核通过；评估中断恢复不重复optimizer更新的测试通过。
- 恢复代码archive SHA256：880dd17b3d80236114c10d5e37903db99f919a5f1c2a5609c8e8cc4c9d1edf64。
- 结果archive SHA256：015ef719f220da9a8e42baa8355d3fe1a810e219f3babf033f6a15980d3fd499。
- 服务器输出：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/transition-aware-semantic-projection-recovery-v1/`。
- 本地原始日志/指标存放在 `.audit-staging/results-projection-recovery-v1/`；不复制大模型checkpoint。

下一步优先核对rehearsal目标与数据分布，再制定以旧teacher关系为约束的独立小实验。保持epoch0为有效模型；这不是全三折最终评估或整条技术路线的No-Go。

# HARrad Reset Closed-loop Implementation Plan

> REQUIRED SUB-SKILL: superpowers:executing-plans inline；用户已批准上轮验证设计并要求开始。持续完成实现、预检及后台启动，不重复询问。正式启动前一次独立整体审查。

**Goal:** 启动真实Stage2初始化和预测梯度配对Stage3试验。
**Architecture:** 独立冻结A/B适配层，持续G、可重置L/E；Keep/Birth固定方差，Policy-only ST闭环。
**Tech Stack:** PyTorch/numpy；本地interview、服务器seer；无安装。
**Spec:** docs/superpowers/specs/2026-10-22-harrad-reset-closed-loop-design.md。

## Global Constraints
- 不覆盖旧模型/输出；subject7只评估，8–9不读；unknown不是Keep负例。
- 头1000×64，Policy初始化300×32，pilot两臂各50×32，正式2臂×3seed×1000×32成功更新。
- 参数冻结不切断状态梯度；块间detach不清空；GT仅Stage2/oracle参考控制状态。
- 源胶囊与完整配置SHA绑定，数据/权重锁；失败证据保留。

## Review Focus
- start/stop在奇数帧、相邻同类和同Update冲突时的监督及延迟。
- 未来目标泄漏到当前分布或Policy；两Adapter错误混用。
- 冻结Former被no_grad切断Policy梯度、FP16/FP32状态问题。
- calibration与fit角色、阈值闭环依赖、未知区间FP分母。
- pilot更新污染正式初值、control误切断语义路径、训练记录跳步造假。

### Task 1: Partial-label data and evaluation
- [x] RED→GREEN：reset_data.py，reset_evaluation.py；官方起止、未知mask、按人内calibration和一对一实例计数。
### Task 2: Actual resettable dual-state model
- [x] RED→GREEN：reset_model.py；G保留、L/E重置、因果分布、future/sem独立梯度、chunk衔接一致。
### Task 3: Training and persistent runner
- [x] RED→GREEN：reset_training.py，reset_runner.py；tiny Stage2/pilot/配对训练、预算及源冻结/重载；固定源配置与SHA。
### Task 4: Native verification and execution
- [x] 一次独立整体审查，重要修复RED→GREEN。
- [x] 源码封存、原生任务测试、真实Qwen梯度/状态/显存预检。
- [x] 后台启动分阶段driver，确认Stage2A实际1000成功更新并记录状态；真实pilot未通过不得自动继续正式Stage3。
- [ ] 训练结束后全组独立结果核查；未结束如实报告运行状态。

## Ledger
- 2026-10-22 用户批准开始。原v2.6旧Stage1硬门槛按已同意双状态调整不再作为永久阻塞；本轮为受控探索，未宣称联合表示优越。
- Ruling: 项目无Git，沿用新模块+不可覆盖源码胶囊隔离，禁止创建虚假commit。
- Ruling: HARrad官方stop同样初始化无类别的新episode，才能表达已知动作结束；unknown仍无背景标签或负例。代价是边界问题包含动作结束，需分清联合边界指标与旧仅start指标。
- Ruling: 采用选择输入状态后单次更新的ST近似，降低双分支Qwen开销；硬前向与选中分支严格一致，梯度近似形式需披露。
- Pre-flight: Task1注释/角色→Task2状态初值→Task3顺序块，统一n=floor(frames/2)、2/15秒Update；尾单帧单列丢弃，不引入padding未来。
- 本地5项数据/模型、2项训练、2项runner测试均先观察缺模块RED；包含旧Event/Joint相关回归共15项通过。阈值字符串'.5'与'0.5'的索引问题已复现并修正为数值键，未改测试期望。
- 一次独立整体审查已发起；正式训练尚未启动。先用v1封存版本进行原生测试和真实机制预检，失败及后续修复版本分开保留。
- v1原生15项及真实Qwen预检通过：Keep/Reset预测最大差0.3626；future/semantic→Policy梯度最大值0.01040/4.24494；G保持、先验分布与决策无关、源码未变，峰值约1.04GiB。v1仅预检，未训练。
- 独立审查结束，无P0/P1；两项P2（原始双锚点评分不合分辨率、缺实例一票语义）均先RED再修复。35条真实录制中28组原始不同帧锚点需合并、6个末端锚点censored；修复后精确oracle无FP/FN。实例指标与时长加权Update指标已分开。18项相关回归通过，不再重复整体审查。
- 补充final权重重载的完整轨迹逐值重放门槛，新增保存轨迹变化反例先RED再实现；正式job只有重放及calibration后才标记完整结束。
- v2原生19项及真实Qwen预检通过，独立下载核验配置/源码绑定与25 fit、6 calibration、4 validation录制；仅预检，未训练。
- 最后预算检查发现Teacher纯unknown块的零BCE仍计步并触发weight decay；新反例先RED（读取1块错误计成功），修复后读取3块、跳过2块、1次有效更新且状态推进到12。采用有界惰性连续读取，保留真实执行顺序；非Teacher配对训练顺序和预算不变。本地相关20项通过，封存v3后原生复验。
- v3原生20项通过、214源码文件SHA及真实Qwen机制复验通过，下载独立核验绑定后16:13:14启动PID29612。16:14:30确认Stage2A1000成功更新完成、源未变；后续效果未验证，见启动记录。
- 16:17:38后台正常，拟合后机制检查通过；Stage2B150/300有效更新，167读取块、17纯unknown跳过块，实际状态快照与Stage2A完成证据已下载。pilot及正式Stage3仍待执行，不将训练batch loss当成验证质量。

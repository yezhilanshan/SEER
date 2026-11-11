# HARrad Stage2–3 Reset闭环验证：启动记录

## 后续检查：10月22日16:40后

Stage2A完成1000步；Stage2B完成300次有效更新、318次读取、18块未知监督跳过，冻结源和预测头均未变。control/full pilot各50步完成，技术门槛通过。正式6组中尚无完成组，目前control-seed42已到80/1000步，进程29612正常使用GPU（约1344MiB），未见failed记录；full及另外两seed尚未执行。

已下载第一个正式job的0步完整验证结果（4条subject7录制、12620 Update，固定阈值0.5），这是Stage2初始化后的共同起点，**不是Stage3最终结果**：

|指标|初始值|
|---|---:|
|起止合并边界Precision / Recall / F1|34.67% / 76.67% / 47.75%|
|已知区间边界误报|79.26次/分钟|
|类别正确且IoU≥0.5事件F1|18.35%|
|按录制平均逐Update语义Macro-F1|45.21%|
|按录制平均逐实例语义Macro-F1|59.47%|
|h1 / h3 / h5按录制平均预测Skill|13.64% / 30.02% / 33.07%|

初值存在较多误切分：边界TP779、已知FP1468、FN237，未知区间未匹配输出1718另报。预测可学习不代表预测梯度已改善事件形成，需等待同seed两臂及多个seed结果。逐实例指标采用实例内平均logits、并按录制平均，与此前官方起点重置的末端读出77.01%口径不同，不能直接作净增减比较。本次仅核对下载JSON摘要，尚未对正式训练的最终权重/预测做完整独立复算。

原始下载：[0步验证JSON](../outputs/harrad-reset-closed-loop-20261022-v1/check-20261022-1640/initial-evaluation.json)。

---

本轮开始验证v2.6.1的核心机制：Reset保留全局G，重置局部运动状态L和语义状态E；未来预测和语义损失能否训练同一个自主Policy。旧语义对齐/调制实验已经结束，本轮独立使用原A-only和B语义预热权重，不假定此前联合表示已有效。

## 当前证据

- v3本地interview与服务器seer相关测试各20项通过，服务器214个源码/配置/测试/设计文件SHA核验通过。
- 真实冻结Qwen预检：强制Keep/Reset预测最大差0.3626001；future与语义探针到Policy的最大梯度分别0.0104018、4.2449379。全局G保持，当前到达前Keep/Birth分布不随当前Reset改变，源参数未变化。32 Update反向峰值显存约1.04GiB。
- v3原生预检下载后独立复核源码及配置绑定，25条fit、6条训练受试者内calibration、4条subject7 validation，test信号未读。
- 2026-10-22 16:13:14（北京时间）启动后台driver，PID29612。16:14:30检查时进程存活，Stage2A已完成1000/1000成功更新，冻结源未变；最后一个batch的loss0.79018、future0.66456只是训练batch损失，不是泛化结果。
- Stage2A之后还需经过拟合后机制复验、Stage2B、pilot，才能进入正式Stage3；本记录不把启动等同于训练完成或质量通过。
- 16:17:38最新快照：进程存活、无failed记录，拟合后机制复验已通过；Stage2B完成150/300有效更新，共读取167块，其中17块无可靠监督只推进状态。本地已保存[启动状态快照](../outputs/harrad-reset-closed-loop-20261022-v1/native-v3/launch-snapshot.json)和Stage2A完成证明。尚未进入pilot或正式Stage3。

## 训练安排

|阶段|训练内容|预算|
|---|---|---|
|Stage2A|冻结A/B主干，拟合Keep/Birth及未来预测头|1000更新，batch64|
|Stage2B|官方起止控制状态，只初始化Policy|300有效BCE更新，最多32连续Update/块|
|Pilot|control/full配对短训，检查有限损失、冻结资产及闭环路径|seed42，每臂50更新|
|Stage3|固定预测器和EventFormer，配对训练Policy|control/full × seeds42/43/44 ×1000更新|
|评估|固定阈值整流验证、final重载重放、训练内阈值标定、锁定阈值整流验证|全部subject7录制；test封存|

control与full的数值总损失相同，只有future项到Policy的梯度不同；两臂的边界、语义和事件数量成本均保留。两臂同seed从同一个Stage2B初值开始，pilot权重不进入正式实验。技术pilot通过只代表程序/梯度可运行，不代表边界或事件识别质量已达要求。新模块不训练语言、Memory或Stage4/5。

连续读取整条录制，块间只detach，录制末尾才初始化；自主阶段不使用GT重置。官方未知区间不当作Keep负例，也不新增背景类；纯unknown的Teacher块仍推进状态，但不占用300有效更新、不执行优化器，读取次数另报。运动目标h1/h3/h5在HARrad对应约0.133/0.400/0.667秒。

## 本轮修正和评价限制

独立审查发现两项评价问题并已修复：同一个双帧Update内多个原帧锚点应合并评分，录制开端初始化/无法完成Update的末端应单列censored；还需加入每官方实例一票的语义指标。35条真实元数据中28组不同原帧锚点合并、6个末端锚点排除，精确Oracle参考的独立元数据匹配无FP/FN。保留原始锚点用于审计。

正式报告同时列出边界F1/已知分钟FP、未知区间未评分输出、类别正确且IoU≥0.5的事件匹配、逐Update与逐实例语义、未来MSE/Skill及Reset数量。阈值只在训练受试者内的6条录制各前1024 Update选取；公共阈值0.5和标定阈值分开报告，标定预算相同不意味着验证FP率相同。校准录制曾被冻结源模型的原预训练见过，不能称为未见受试者。

最终权重重载后完整轨迹逐值重放需误差0。固定预测器训练过程中的输入状态仍会变化，所以不能用冻结参数推断预测输出恒定。ST是选择输入状态后的有偏直通近似，不能据此宣称离散Reset的精确梯度。

## 路径和溯源

服务器输出：`/home/xibeidaxue/newdisk2/WJ/HARrad/outputs/reset_closed_loop_20261022_v3`

服务器不可覆盖源码胶囊：`/home/xibeidaxue/newdisk2/WJ/HARrad/experiments/reset_closed_loop_20261022_v3`

后台日志：上述源码目录的`driver.log`；整体阶段看输出目录`status.json`，实际更新看各阶段或job的`status.json`，失败看`failed.json`。检查进度无需重新启动driver。

本地证据：[v3真实预检](../outputs/harrad-reset-closed-loop-20261022-v1/native-v3/preflight.json)、[独立绑定核查](../outputs/harrad-reset-closed-loop-20261022-v1/native-v3/local-audit.json)、[原生测试日志](../outputs/harrad-reset-closed-loop-20261022-v1/native-v3/native-tests.log)、[锚点审计](../outputs/harrad-reset-closed-loop-20261022-v1/oracle-anchor-audit.json)。

v3源码ZIP SHA256：`a70ef7d5203909189e6e3586a5a243a234679be5c1e32c280517f798e9351734`；源码manifest SHA256：`4aa51d65a5b1c4f65ab41e08e20aadab1f521f421d6586cd0b90801c01e3f71c`；配置规范化SHA256：`83e9552c9adc13ce4feb4c71f08e876f3d2ffdf6e26448f4eb5d0c71ae9126e0`。

v1/v2仅做预检，没有启动正式训练，版本与失败测试证据均保留。完整协议见[设计](superpowers/specs/2026-10-22-harrad-reset-closed-loop-design.md)与[执行清单](superpowers/plans/2026-10-22-harrad-reset-closed-loop.md)。

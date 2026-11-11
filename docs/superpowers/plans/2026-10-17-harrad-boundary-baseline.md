# HARrad连续监督边界基线 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this authorized plan inline, task by task.

**Goal:** 完成用户已批准的连续数据准备及监督边界头训练，不接入自主EventFormer切换。

**定位：** 本清单交付纯监督边界对照，不代表原v2.6的预测驱动Reset阶段已完成。后续把边界头串联EventFormer只能作为对照支路，正式主线仍需Dynamics/Keep-Birth以及Reset影响未来预测损失的闭环训练。

**Architecture:** 从已核验冻结前端的continuous_hidden读取完整录制，原CNN/GRU只在录制起点初始化。新增小型MLP读取当前Update末端GRU hidden128及其相对上一Update的差分128，输出独立start/end logits。边界标注只用于损失与评价，未标注区间不作为负例。

**Tech Stack:** 现有PyTorch、numpy；本地interview pytest，服务器seer环境。无Git仓库，以不可覆盖源码包、SHA清单和新运行目录留档。

**Spec:** 用户本轮已批准的聊天方案：第1项连续训练数据、第2项因果监督起止头；原subject划分，test不参与。下列参数是范围内的首轮实现选择。

## Global Constraints

- train subject1–6（31录制190,238帧、3,018实例），validation subject7（4录制25,244帧、511实例）；不读取test信号。
- 锁定原RD seed42 checkpoint、normalizers和完整录制continuous_hidden的SHA；绝不使用oracle_hidden作预测输入。EventFormer不参与本轮前向或训练，权重文件保持不变。
- 全录制按时间顺序处理，包含未标注帧，无实例切片训练。两帧Update的cutoff为2/4/…；录制尾部若单帧则在真实结束flush。输入hidden来自cutoff-1，差分只与前一Update比较；第一个差分为0。
- 起点s映射到第一个已观察到帧s的Update；终点stop（右开）映射到第一个观察完stop-1的Update。预测发出时间为当前cutoff，绝不回填真实边界时间。真实start/stop分别保留用于事件匹配，帧时间按标称15fps。
- 两头target独立，同类相邻实例也分别标记。负例只在Update内全部帧已标注时成立；每头正例前后1个Update不作该头负例，正例本身有效；未标注区间保留输入，loss mask关闭，边界正例可使对应混合Update有效。
- Normalizer仅用train的全部Update（含未标注特征）拟合每坐标均值/标准差，clip±8。首轮MLP256→128/GELU/Dropout0.1→2，共33,154参数，FP32；冻结前端缓存不反传。
- masked weighted BCE，pos_weight=全train每头有效负例数/正例数，两头有效元素分别求均值后再平均。seed42，20完整epochs，batch一条完整录制，每轮31次成功更新，共620步；AdamW lr1e-3、wd0.01、clip1。
- 实时解码仅采用阈值上升沿，最小发出间隔4帧；不使用未来峰值/NMS。阈值预先固定网格0.1–0.9步长0.1，validation按每头F1选阈值、同分取较高阈值；checkpoint取start/end F1均值最高且最早epoch。
- 主匹配容差4帧（0.267秒），同时报告2/8帧；按录制、头分别按时间顺序一对一匹配。所有预测先尝试匹配已标注真边界；未匹配且发出时最后一帧已标注的预测计FP，其余单列为unscored_unknown。该指标为部分标注条件评价，不能声称是整条录制全监督Precision。
- 报告TP/FP/FN、Precision/Recall/F1、匹配延迟（有符号及绝对值）、未标注区间未评分预测数及每标称分钟数。冻结特征变化量基线使用train能量分位数0.5/.75/.9/.95/.975/.99作为候选阈值，validation选择，使用相同实时解码/评价；另报告train估计周期的时钟基线，均不使用真实起点控制预测。
- 结束后best重新加载逐值重放，独立事件匹配算法复算，导出概率、阈值、逐事件匹配及输入输出SHA。当前不训练Dynamics、预测收益Reset、语言生成或边界驱动EventFormer。

## Review Focus

- 未标注区间、混合Update：独立mask，不偷换为背景负例；增加目标构造测试。
- 相邻同类、短实例：起止标签独立、匹配不重复计TP；增加小序列测试。
- 连续特征与batch：只用continuous_hidden，首尾cutoff正确、未来扰动不改早期输出；增加因果/分块测试。
- 验证阈值和局部峰值：无未来选择、阈值有限网格、对照同协议；增加解码及独立匹配测试。
- 以大量未知预测逃避FP：同时报告未评分预测数量/分钟、完整时间轴概率，不只报F1；增加未知区域统计测试。

## Tasks

- [x] 写`tests/test_harrad_boundary.py`并观察RED：目标/mask、因果输入、在线阈值解码、一对一事件匹配。
- [x] 实现`harrad/boundary.py`数据契约、MLP、loss、评价；目标测试GREEN。
- [x] 写小数据训练/重放测试并观察RED；实现`harrad/boundary_training.py`、`boundary_runner.py`，通过测试。
- [x] 固定配置、源包、原生测试；锁定35条continuous cache与源权重，完成数据清单。
- [x] 执行seed42×20epoch，核实620步、31完整训练录制覆盖、4完整validation录制预测。
- [x] best重放误差0，独立指标18组及38文件哈希通过，简单基线与结果文档完成。无Git提交要求，原实验与失败证据全部保留。

## Execution ledger

- 本地与原生首轮9项测试通过；真实20轮620步已完成，v1结束审计遇到DP最大基数相同时回溯选择不同预测的问题。训练权重和主评价未受影响，独立审查复现了合法反例。
- Ruling: 保留v1全部源码、权重、history、failed，修正独立DP回溯的确定性选择，新增回归反例先RED再GREEN；以独立v2评价目录重载既有best，不重复训练620步 — 原训练已完整且主评分不变，无需为评价实现错误重训 — 若v2重放不等于原概率则停止并定位，不改验收阈值。

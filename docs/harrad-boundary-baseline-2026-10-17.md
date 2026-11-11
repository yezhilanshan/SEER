# HARrad连续监督边界基线

用户已批准连续训练数据和监督边界头训练；本轮不接入EventFormer切换、不训练Dynamics或预测驱动Reset。

**研究定位修正：** 对照原《方案书v2.6》第1.2、5.4、Stage1–3及第13章，本实验属于独立监督边界对照，与其中boundary-head baseline同类，但架构并非原表中的TCN，不能直接称完整B1复现。它没有训练Keep/Birth预测分布，也没有让Reset经由状态影响未来预测损失，因此不是Stage2的预测型Reset初始化或Stage3自主事件形成。将它串联冻结EventFormer只会得到监督分段基线，不能替代v2.6主线。后续认可的v2.6.1双状态调整仍要求动力学/语义联合事件转移；HARrad上的A-only/A+B与Reset预测后果尚未验证，旧PARrad联合实验不能作为该项已完成的证据。

## 输入与训练范围

使用原RD seed42冻结CNN/GRU产生且已核验的`continuous_hidden`，每条录制从头持续运行，包含未标注区间。预测输入为当前双帧Update末端hidden128及其相对上个Update的变化128；不读官方重置后的`oracle_hidden`，不把类别、实例起点或终点输入预测头。

train为subject1–6：31条录制190,238帧、3,018实例；validation为subject7：4条录制25,244帧、511实例。test8–9信号不读取。原CNN、GRU、EventFormer权重文件以及原normalizers均锁定SHA并保留。

边界头为256→128/GELU/Dropout0.1→2，共33,154参数，输出独立start/end分数。首轮seed42、20完整epochs、每次一条完整录制，预计620次成功更新；AdamW lr0.001、weight_decay0.01、梯度clip1。新输入normalizer仅用train特征拟合。

## 标注、实时解码与评价

- 起点/终点分别映射到观察到对应帧的Update；发出时刻是实际cutoff，不回填真实边界时间。录制末尾若单帧，按真实结束flush。
- 同类相邻实例仍分别监督起止。35条录制的同一Update内同类边界冲突审计为0，不会静默合并两个边界。
- 正例附近一Update不当负例；其余负例仅来自完全已标注的Update。未知区间参与特征序列，但不作为背景负例。
- 损失为两个头分别归一化的masked weighted BCE，正权重只由train有效正负例计数得到。
- 推理仅使用当前概率的阈值上升沿，发出最小间隔4帧；不使用未来局部峰值或真实边界重置。候选阈值固定0.1–0.9，validation调阈值，按起止F1均值选择最高且最早checkpoint。
- 匹配容差主指标4帧（标称0.267秒），另报2/8帧。每录制、每头一对一匹配；未匹配且落在已标注帧内的预测记FP，未标注区域未匹配预测单列数量与每标称分钟数。因标注不完备，此Precision/F1不能解释为完整时间轴的全监督指标。
- 对照为train分位数固定候选的hidden变化量阈值，以及train估计周期的时钟预测，两者均不使用官方起点控制预测。报告逐事件匹配、完整时间轴分数、延迟、未知区域预测负担及best重载一致性。

## 已完成结果

连续数据准备和20完整epochs已完成，真实成功更新620步、全帧训练暴露3,804,760次；每轮31条完整训练录制，train/validation分别95,128/12,624个Update。最佳checkpoint为第16轮，validation选择start/end阈值分别0.9/0.8。本轮只训练33,154参数的小头；CNN/GRU使用此前核验的冻结连续缓存，EventFormer不参与前向。基于冻结缓存的20轮主训练及逐轮验证计时5.44秒，不包含实现、数据校验、测试、修复、最终审计和导出时间。

主指标容差±4帧（约±0.267秒），支持511个官方起点及511个终点：

| 模型 | 起点F1 | 终点F1 |
| --- | ---: | ---: |
| 训练集估计周期时钟 | 20.38% | 21.59% |
| 冻结hidden变化量阈值 | 41.83% | 34.99% |
| 学习的MLP边界头 | **55.94%** | **67.33%** |

| 头 | Precision | Recall | TP | 已标注区间FP | FN | 未标注区间未评分预测 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 起点 | 67.12% | 47.95% | 245 | 120 | 266 | 214 |
| 终点 | 68.99% | 65.75% | 336 | 151 | 175 | 237 |

匹配实例的绝对时差中位数均为0.133秒，P90均为0.267秒；这些统计仅针对已匹配边界，不能据此称全部事件都能及时检测。起点/终点未评分未知预测分别约22.46/24.87次每标称未知分钟。未知预测不是已确认误报，也不能因未进入FP分母就忽略。

容差±2帧时起止F1为40.14%/51.69%；±8帧时71.91%/82.26%。容差改变会改变匹配集合以及未知预测分类，不能将宽容差结果代替主指标。

结论：监督边界头优于两个简单对照，但起点仍漏266/511，未知区间预测负担较大；当前不足以直接接入自主Reset。仅单seed、单validation受试者，阈值和checkpoint均在该validation选择；不能声称跨受试者测试性能或最终连续事件识别已验证。边界F1与此前77.01%的动作类别Macro-F1是不同任务指标。

## 验收、修复和证据

本地及服务器最终10项测试通过，覆盖mask与未知区间、同类相邻实例、分辨率冲突、因果输入、实时上升沿、独立DP匹配、train-only normalizer、缓存SHA、完整录制训练与best重放。独立审查发现并复现了DP同分回溯可选择不同预测的评价问题：训练20轮620步均已结束，v1在结束审计停止。新增反例先RED再GREEN；修复回溯确定性，并用3,000组随机样例核对四项计数一致。

保留v1全部训练和failed证据，另建v2评价输出，只重载已有best，不调用优化器、不重训。v2于2026-10-17 **20:14:11（UTC+8）**完成：概率重放最大误差0，主评分与原best轮一致，源模型和元数据锁均未变，test信号未读取。源码v1包SHA256为`49952e7c779d88792a65e8fcb130e213b168b7ffc0ebd93fcff2366591e5af1b`；v2包为`6720d11cb60116f6b7b0a7ca42dd90ea2553899d6a45c824fde06e597e705591`。

- 服务器原训练：`/home/xibeidaxue/newdisk2/WJ/HARrad/outputs/boundary_baseline_20261017_v1`。
- 服务器独立复核：`/home/xibeidaxue/newdisk2/WJ/HARrad/outputs/boundary_evaluation_20261017_v2`。
- [本地最佳权重](../outputs/harrad-boundary-baseline-20261017-v1/finished-v2/training-v1/best.pt)，SHA256 `269994003e7828f605d077bfdb3551bd064ed7c723345d3bab46c7b80c7018c9`。
- [完整完成记录](../outputs/harrad-boundary-baseline-20261017-v1/finished-v2/evaluation-v2/complete.json)、[逐事件匹配](../outputs/harrad-boundary-baseline-20261017-v1/finished-v2/evaluation-v2/event-matches.json)、[本地独立复算](../outputs/harrad-boundary-baseline-20261017-v1/local-independent-verification.json)。
- 38个导出文件SHA与字节数通过，包含3份权重、20轮history、4份完整概率与4份变化量基线、data manifest/normalizers、原失败与新修复记录、两版原生测试和源码清单；完整特征数组仍留服务器。导出包1,241,376字节，SHA256 `4a974efcdecb0497a682ffce8e92deb16c512848136d0948ba627fc41e065424`。
- 神经头和两个基线、三个容差、两个头共18组指标，在本地使用DP独立复算TP/FP/FN/unknown计数与F1全部通过。

实施计划：[固定方案与检查清单](superpowers/plans/2026-10-17-harrad-boundary-baseline.md)。旧实验输出和失败证据均保留；本轮使用独立源码包与新输出目录。

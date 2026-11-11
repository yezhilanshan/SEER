# HARrad Stage 2–3预测驱动Reset验证

用户已批准先验证Reset改变预测及梯度、再进行Keep/Birth初始化和配对闭环实验。沿用讨论中的双状态v2.6.1；本轮不训练语言或Memory，不继续扩展对齐/调制。

## 固定资产与状态

冻结RD CNN缓存、A-only seed42的tokens/GRU和B语义warmup（含自己的Adapter）；两个Adapter独立，禁止相互覆盖。固定原train运动标准化及方差、h={1,3,5}（2/15、6/15、10/15秒）。Keep预测头从A-only初始化、Birth头同形同初值；Stage2A拟合后均冻结。

G为不重置的全局GRU状态；L为共享同一冻结GRU转移参数的可重置局部状态。E为可重置冻结EventFormer状态。一个Reset同时清空L、重置E及事件age，G保留。硬前向选择输入状态，反向采用sigmoid直通估计，再执行一次冻结GRU/EventFormer更新。此为选择输入状态的有偏ST近似，明确区别于两个更新后状态插值。

pKeep(m_t)、pBirth(m_t)只由截至t−1的L/G及m_(t−1)产生，固定高斯方差相同；m_t到达后计算证据。Policy读G/L、旧E投影、当前m_t、age及证据，不读GT或未来。新L预测真实未来目标，E给当前动作监督。

## 数据与不完整标注

subject1–6仍为train、subject7为validation、8–9不读。每个训练受试者按文件名最后一条录制划为policy calibration，剩余录制用于新头训练；既有冻结前端曾见这些训练受试者，明确不称独立未见人。所有新臂共享此划分。

HARrad有空白区间。官方start和stop均表示事件状态切换锚点：stop可以终止一个已标注动作并进入无类别标签的区间，不新增背景类别。映射到覆盖该帧的双帧Update，同一Update多个锚点合并；若同类重复实例相邻，仍保留切换。负例只取完整同一已知实例内、离边界>1 Update的位置。未知区间不作Keep负例；真实雷达预测继续。

评价记录Reset生效位置2t和实际可用时间2t+2，禁止回填GT。同Update锚点评分只保留最早原帧位置，全部组内锚点另存审计；自动初始化和未完成双帧之后的录制末端锚点单列censored。报告±4帧一对一边界匹配及已知时间FP/min，未知区间未匹配输出单报；还报告类别正确且IoU≥.5的事件匹配和实例语义、预测MSE。实例语义对官方完整实例内有效Update logits平均、每实例一票，与逐Update指标分开。未知episode不能默认为错误背景。官方边界参考、Always Keep/Reset用于排查退化。

## 训练和对照

Stage2A：种子42，1000成功更新，batch64，AdamW1e-4，拟合oracle局部状态的未来预测、事件内Keep当前似然及边界Birth当前似然；正负集合分别归约。固定源G/L轨迹可缓存，但训练Policy的闭环状态不得用oracle缓存替代。

Stage2B：300成功更新，每次读取最多32个顺序Update，只训练Policy，GT边界控制状态。BCE按正/负集合各平均，不让大量负例淹没边界；unknown不参与BCE。整块没有可靠监督时仍推进状态和游标，但不执行优化器、也不计入300次成功更新；分别记录读取次数、跳过块数和实际读取顺序。源状态无梯度。

Stage3：配对control/full仅训练相同Policy。总目标为boundary BCE + .25 semantic CE/log12 + .05 mean(reset probability)；full额外包含固定目标的未来MSE归一化项，control关闭这项对Policy的梯度，其他计算不detach。所有预测器和Former参数冻结，但full必须保留对状态输入的autograd。

先seed42两臂各50步pilot，有限梯度、状态影响与loss路径检查通过后，seeds42/43/44两臂各1000成功更新。每步32 Update，逐录制round-robin连续推进，块间只detach不清空，录制结束才重新初始化。0/300/1000保存；同一固定时刻评价完整validation录制。模型自主决策，不以GT或固定窗口重置。

固定公共阈值.5用于训练及阶段曲线。每组final权重重载后，以.5完整回放subject7，逐值比较q/logits/mean/episode/生效与可用边界，必须误差0；final另在训练calibration每条前1024 Update上，对.25/.5/.75/.9做真实闭环回放，按已知时间FP预算10/min内最高边界recall选阈值（同分优先更高阈值），无可行点选最少FP并声明未达预算；随后用锁定阈值评价完整subject7。报告实际FPR，不能因标定预算相同而声称validation FPR完全相等。

## 验收与隔离

先检查真实源checkpoint的强制Keep/Reset后续预测差异及future→Policy梯度；语义和预测梯度分别记录，control仅去掉future路径。未来扰动不改变此前决策，unknown负例遮罩，TBPTT不清状态，零步配对模型逐值一致。原生CUDA短流和显存预检通过后后台执行。

旧源码/权重/输出不改，新增harrad/reset_模块和独立版本输出；无Git，采用源码SHA胶囊和配置/数据锁。失败版本保留。全流程只检验小规模Oracle初始化至自主闭环，不等于Stage4/5完成。

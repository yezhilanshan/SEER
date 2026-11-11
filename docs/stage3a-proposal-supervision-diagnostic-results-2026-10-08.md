# Stage3A 固定500步监督诊断结果（2026-10-08）

诊断已完成：本轮新增 AdamW 更新0、模型前向0，全部操作使用冻结预测和原始Encoder缓存。现有权重、检索P、QA及生产路径保持原状。

主要发现：训练边界均获得过监督；简单相邻token/MD差分不能很好定位当前动作标签边界；已有Head明显优于这些简单差分。现有触发器下Local/OR主要失在阈值覆盖，Radar主要失在峰形约束。下一轮可检验“稳定区困难负例与边界正例排序”，暂不增加训练预算、扩标签窗或追加原始差分输入。这个实验方向是待验证假设，不是已证明的根因修复。

## 1. 实验范围与核验

- 18 fit recording、8 dev recording，fit1815个GT边界、dev704个；dev仍是上游见过的TRAIN开发数据，不能作为独立未见受试者测试或正式Go。
- 复用weighted500、unweighted500及18个此前fit锁定的Local/Radar/OR、3/5/8 FP/min配置；本轮无阈值搜索。matching仍按实际发出时刻±0.72s，峰中心及lookahead分开记录。
- 服务器原生测试15项通过；本地同样15项通过。新鲜只读审查发现3个统计边界问题，均在一次RED→GREEN修复中处理，使用v2隔离导出，v1证据保留。
- 独立服务器读回26个原缓存、416个token样本、2519个标签转换及3个检查点SHA；本地归档SHA/9成员安全路径/逐成员SHA通过。日志归档6成员同样核验。
- 全部45342个逐边界账目身份核验；36组fit/dev分区与原匹配结果一致；1770组特征/Head排名由sklearn独立重算，最大误差2.22e-16；8000个实际chunk与1815个边界曝光独立重算。
- fit-stable Q95分传感器先写入并锁SHA，再加载dev数值。所有缺失窗口与零范数保留NA，末尾缺少lookahead的高分归context不足，不生成虚构Proposal。

## 2. 现有触发器究竟在哪里漏检

下表是**第一个未满足的执行条件**，不是因果根因比例。高分检查对齐预定/实际决策时间，绝不以离线±2.16s邻近峰增加召回。每行总GT704、无excluded；未列matched可由Recall或704减各类miss数恢复。FP/min档是fit预算上限，dev实际FP/min不完全相等，不能理解为精确等FP插值。

| loss | 路径 | 档 | Recall% | 实际FP/min | 低于阈值 | 邻点不足 | 峰形不足 | context不足 | NMS抑制 | 匹配冲突 |
|---|---|---|---|---|---|---|---|---|---|---|
| weighted | local | 3 | 22.16 | 1.973 | 461 | 64 | 18 | 0 | 5 | 0 |
| weighted | local | 5 | 33.66 | 3.484 | 341 | 74 | 49 | 0 | 3 | 0 |
| weighted | local | 8 | 50.00 | 6.406 | 230 | 53 | 63 | 1 | 5 | 0 |
| weighted | radar | 3 | 27.98 | 2.285 | 424 | 4 | 77 | 0 | 2 | 0 |
| weighted | radar | 5 | 38.78 | 3.934 | 317 | 2 | 112 | 0 | 0 | 0 |
| weighted | radar | 8 | 57.81 | 7.817 | 92 | 0 | 200 | 0 | 5 | 0 |
| weighted | fusion | 3 | 20.74 | 1.536 | 502 | 39 | 11 | 0 | 6 | 0 |
| weighted | fusion | 5 | 21.59 | 2.760 | 502 | 39 | 11 | 0 | 0 | 0 |
| weighted | fusion | 8 | 48.01 | 6.331 | 286 | 37 | 29 | 0 | 14 | 0 |
| unweighted | local | 3 | 22.73 | 1.848 | 454 | 74 | 13 | 0 | 3 | 0 |
| unweighted | local | 5 | 35.51 | 3.671 | 330 | 68 | 53 | 0 | 3 | 0 |
| unweighted | local | 8 | 46.73 | 6.069 | 231 | 65 | 75 | 0 | 4 | 0 |
| unweighted | radar | 3 | 23.44 | 2.060 | 451 | 2 | 85 | 0 | 1 | 0 |
| unweighted | radar | 5 | 33.52 | 3.921 | 336 | 1 | 131 | 0 | 0 | 0 |
| unweighted | radar | 8 | 46.31 | 7.480 | 148 | 1 | 224 | 1 | 4 | 0 |
| unweighted | fusion | 3 | 23.86 | 1.711 | 481 | 29 | 14 | 0 | 12 | 0 |
| unweighted | fusion | 5 | 25.57 | 3.309 | 481 | 29 | 14 | 0 | 0 | 0 |
| unweighted | fusion | 8 | 46.88 | 6.681 | 279 | 45 | 34 | 0 | 16 | 0 |


weighted、8档：Local漏352个，其中230个低于阈值（65.34%），53个邻点不足，63个峰形不足，5个NMS抑制，1个尾部context不足；Radar漏297个，其中200个峰形不足（67.34%），92个低于阈值，5个NMS抑制；OR漏366个，其中286个低于阈值（78.14%）。NMS不是这三路主要缺口。峰形不足可能包括单调平台、峰不在匹配时窗等，不能直接全部解释为“平台”。低于阈值也不能解释成Head完全没有学会。

![全部固定8档漏检分区](D:/lab/SEER/docs/assets/v25-proposal-supervision-2026-10-08/failure-stages.png)

逐subject×sensor的8档Recall，A=weighted、B=unweighted，全部单元同报：

| 单元 | Local A | Local B | Radar A | Radar B | OR A | OR B |
|---|---|---|---|---|---|---|
| subject_02/TI_xWR14xx | 54.08 | 47.96 | 67.35 | 58.16 | 57.14 | 59.18 |
| subject_02/TI_xWR68xx | 67.35 | 60.20 | 64.29 | 57.14 | 68.37 | 62.24 |
| subject_03/TI_xWR14xx | 43.37 | 42.17 | 40.96 | 34.94 | 40.96 | 36.14 |
| subject_03/TI_xWR68xx | 38.55 | 42.17 | 56.63 | 44.58 | 36.14 | 39.76 |
| subject_04/TI_xWR14xx | 42.35 | 44.71 | 61.18 | 43.53 | 47.06 | 44.71 |
| subject_04/TI_xWR68xx | 41.18 | 35.29 | 52.94 | 38.82 | 36.47 | 35.29 |
| subject_08/TI_xWR14xx | 55.81 | 58.14 | 60.47 | 47.67 | 46.51 | 53.49 |
| subject_08/TI_xWR68xx | 53.49 | 40.70 | 55.81 | 41.86 | 46.51 | 39.53 |


Local A最弱单元为subject03/68xx（32/83），40/51个漏检属于阈值不足；subject04/68xx为35/85，40/50个漏检属于阈值不足。Radar A在subject02两路峰形失败占各自漏检81.25%和82.86%，subject04/68xx为50%。差异跨受试者和传感器存在，不能只报汇总均值。全部组及动作转换类别在原始JSON/CSV保留。

## 3. 实际训练曝光是否不足

真实500×16共8000个chunk、504845次valid update曝光；140个短chunk（1.75%），99个没有核心边界update的chunk（1.24%）。每个subject×sensor仍800次，但recording长度不同带来重复次数不同。

1815个fit边界核心（±0.36s）全部被采到，平均重复8.3543次、中位8次、P10=7、P90=15、最大16；0个完全未见边界。全部有效update重复次数范围6–16，逐recording计数在sampling-per-record.json。不能把桶均衡当成逐边界完全均衡，但没有“大量边界被漏采”的证据。

逐chunk平均Gaussian正例质量占比15.2968%，加权BCE中正项质量份额49.3707%；unweighted中为15.2968%。归一化分母独立核验为A: aM+b(N−M)，B: N。这里的“份额”是目标函数系数质量，不是实际梯度；chunk累计系数还需除16才能对应每步16-chunk平均的绝对系数。重叠边界核心可共享同一update，逐边界质量不可相加后声称独立总量。

结论：暂不优先重做采样。unweighted降低分数但同时减小正项质量，前轮结果也未显示固定FP档稳定召回收益；不能仅凭分数变低判定训练改好。

## 4. 输入中有没有简单可用的变化信号

固定5种因果差分，不训练读出、不改Head。token仅用前后共同有效slot；MD为缓存的归一化幅值/方差、帧绝对差及余弦差代理，**不是速度、姿态或高度真值**。hard标签为距GT≤0.36s正例、>0.72s稳定区，中间区排除；每项特征与Head用完全相同有效支持集。

| 分组 | 固定特征 | AUROC | AP | 同支持Local AP | 正例比例 |
|---|---|---|---|---|---|
| fit | token_cos | 0.4404 | 0.1281 | 0.3288 | 0.1522 |
| fit | token_relative_l2 | 0.4443 | 0.1289 | 0.3288 | 0.1522 |
| fit | md_level_delta | 0.4657 | 0.1377 | 0.3288 | 0.1522 |
| fit | md_frame_abs_delta | 0.4667 | 0.1384 | 0.3288 | 0.1522 |
| fit | md_frame_cos_delta | 0.4655 | 0.1372 | 0.3288 | 0.1522 |
| dev | token_cos | 0.4579 | 0.1166 | 0.3429 | 0.1327 |
| dev | token_relative_l2 | 0.4654 | 0.1184 | 0.3429 | 0.1327 |
| dev | md_level_delta | 0.4439 | 0.1172 | 0.3429 | 0.1327 |
| dev | md_frame_abs_delta | 0.4485 | 0.1151 | 0.3429 | 0.1327 |
| dev | md_frame_cos_delta | 0.4483 | 0.1151 | 0.3429 | 0.1327 |


![固定差分与已训练Local的AP](D:/lab/SEER/docs/assets/v25-proposal-supervision-2026-10-08/input-ranking.png)

在当前“数值越大越像变化”的预注册方向上，5项简单差分dev AP约0.115–0.118，正例比例0.1327；Local500 AP0.3429。fit同样弱，Local fit AP0.3288、dev0.3429，没有由此支持明显fit→dev排序崩溃。现有Head确实提取了简单差分没有捕获的模式。不能由低差分AP推断整个Encoder没有信息，更不能据此解冻Encoder；也不应直接把这些差分塞入Head并期待必然改善。

## 5. 标签与物理变化时刻是否错位

仅对固定特征超过所属sensor的fit-stable Q95时，取GT±2.16s窗口内最强峰。GT间距≤4.32s标记重叠，不参与孤立窗口的时刻结论。fit孤立1413/1815，dev孤立618/704。全部特征窗口均可评价；原始缺失统计仍保留。

| 分组 | 特征 | 孤立有效窗口/全部 | 孤立显著率 | 全部显著率 | 控制窗口显著率 | 孤立峰偏移中位(s) |
|---|---|---|---|---|---|---|
| fit | token_cos | 1413/1815 | 35.88 | 37.63 | 45.39 | 0.00 |
| fit | token_relative_l2 | 1413/1815 | 35.67 | 38.02 | 47.63 | 0.09 |
| fit | md_level_delta | 1413/1815 | 52.51 | 53.11 | 55.31 | 0.00 |
| fit | md_frame_abs_delta | 1413/1815 | 50.11 | 49.64 | 53.21 | -0.72 |
| fit | md_frame_cos_delta | 1413/1815 | 47.35 | 47.77 | 51.82 | -0.72 |
| dev | token_cos | 618/704 | 31.07 | 33.66 | 31.57 | 0.81 |
| dev | token_relative_l2 | 618/704 | 32.20 | 34.66 | 33.08 | 0.81 |
| dev | md_level_delta | 618/704 | 48.22 | 48.72 | 45.71 | 0.18 |
| dev | md_frame_abs_delta | 618/704 | 37.70 | 38.21 | 47.47 | -0.90 |
| dev | md_frame_cos_delta | 618/704 | 38.03 | 38.64 | 48.99 | -0.54 |


5种特征的孤立峰率大体接近控制窗，峰偏移分散，未观察到各特征共同支持的可靠固定平移。token dev显著峰中位+0.81s，帧abs proxy −0.90s，方向不一致；均只统计显著窗口的最大峰，并非真实物理边界。控制中心距GT>2.16s，但整段窗口不保证远离GT±0.72s，控制率只是描述性参照。窗口内max选择也会将噪声峰放大，控制窗口与同recording样本相关，不能作为独立统计试验。

因此不修改原标签、不平移GT、不放宽召回容差，也不以宽窗峰证明标注错误。若未来要判断动作标注与真实运动起点的差别，需要独立物理边界注释或同步视频，此轮不能回答。

以下案例在fit中按固定规则选取：weighted/local/8档，分别取below_threshold、neighbor_constraint、peak_shape类别中UID字典序最前且时间最早的GT。没有使用dev挑选成功例，案例不替代全部统计。

![fit固定规则案例](D:/lab/SEER/docs/assets/v25-proposal-supervision-2026-10-08/fit-representatives.png)

## 6. 推荐下一轮：只加入边界与困难稳定负例的排序约束

这是一个检验Head分数分离是否可改善的实验建议，尚未实现或启动。诊断不证明它必然有效。原因是：边界曝光充足，简单差分和统一标签平移缺乏支持，而Local在固定FP预算下大量边界得分不足；当前点级Gaussian目标未直接要求边界高于最容易误报的稳定段。

建议A复现原weighted Gaussian BCE；B保留同一BCE，唯一新增项为每个Head的pairwise排序损失：L_B=L_A+0.1·L_rank，L_rank=mean softplus(z_stable−z_boundary+0.5)。每个原chunk内，正例来自valid且距GT≤0.36s，负例来自valid且距GT>0.72s；取该Head稳定负例logit最高8个（不足8取全部，选索引stop-gradient），与chunk内全部正例配对。无正例或无负例时rank项为0并单独报告；每chunk/每Head先求均值，再沿用原两Head、16chunk汇总尺度。margin0.5、lambda0.1、k8预注册，不通过dev调参。

沿用原初始化、seed42、相同500×16抽样、LR0.001/WD0.01/clip1、FP32和状态carry；每臂**从原未训练Physical初始化重新训练500步**，不在现有500上追加。只训练Local/Radar，所有上游和P/Evidence冻结。候选对来自既有chunk，不换采样器、输入、Head结构、Gaussianσ或标签。

评估顺序仍先Head AUROC/AP、稳定分数和平台，再同一已定义peak家族/lookahead/NMS，阈值仅在fit重新校准3/5/8 FP/min；dev一次完整读取，三路及8个单元全报。不能把旧数值阈值套新模型，也不能选出单一路后隐藏其他退步。

建议提前锁定小实验门槛：Local为主，8档总体Recall较复现A至少+5个百分点且实际FP/min≤8；3/5档Recall各不得退>2个百分点；Local AP不退>0.01，8个单元至少6个改善且任何单元退步≤5个百分点。Radar/OR独立报告，不替代Local门槛；门槛只决定是否值得继续，不代表Stage3A已经合格。单seed结果不能宣称稳定复现收益，达到门槛后才另议重复seed验证。若排序项无收益，停止这一分支，再考虑独立监督定义或更长因果上下文，不自动开新训练。

## 7. 完整证据与可复查入口

- 结果：[逐边界漏检账目](D:/lab/SEER/.audit-staging/proposal-supervision-results-v1/all-boundary-failures.csv)、[全部组分区](D:/lab/SEER/.audit-staging/proposal-supervision-results-v1/failure-categories.json)、[8000个chunk质量](D:/lab/SEER/.audit-staging/proposal-supervision-results-v1/all-chunk-mass.csv)、[1815个边界曝光](D:/lab/SEER/.audit-staging/proposal-supervision-results-v1/all-fit-boundary-exposure.csv)。
- 输入：[1770组共同支持排名](D:/lab/SEER/.audit-staging/proposal-supervision-results-v1/ranking-summary.json)、[所有特征对齐](D:/lab/SEER/.audit-staging/proposal-supervision-results-v1/all-feature-alignment.csv)、[控制窗](D:/lab/SEER/.audit-staging/proposal-supervision-results-v1/stable-window-controls.csv)、[fit阈值](D:/lab/SEER/.audit-staging/proposal-supervision-results-v1/fit-signal-thresholds.json)。
- 验证：[本地独立数值读回](D:/lab/SEER/.audit-staging/proposal-supervision-results-v1/independent-local-readback.json)、[服务器原缓存独立读回](D:/lab/SEER/.audit-staging/proposal-supervision-logs-readback-v2/independent-native-features.json)、[原生测试日志](D:/lab/SEER/.audit-staging/proposal-supervision-logs-readback-v2/native-tests.log)、[导出日志](D:/lab/SEER/.audit-staging/proposal-supervision-logs-readback-v2/export.log)。
- 特征完整归档SHA：2180e42bc306fb24757270252fa58f44e31df66dc8e61dffeef0dd7bd8a09589；日志完整归档SHA：911881e71d6f319450dde8aba1261f58b02ec74c63599268cbb96f32d0c094b7。
- v2源包SHA：3541776ec9310611acde6719a871704af71b42a83e23693df01c08eae6e6634d；锁定18触发配置SHA：63636f5dff81f555566feeba8450830feae4160d248e96415eebb24bec9650ef。
- fit特征阈值SHA：4592d4bd374121256ab53cf5221a6dedb081d94cf94ebb0829bb7932047fbc3c；3个原检查点SHA详见features.json，前后不变。
- 本轮实现与分析源码按manifest固化，无Git checkout，不虚构commit或worktree；v1/v2隔离文件、原始输入和失败日志保留。

# Radar + GRU：onset导出与双源候选池诊断

本轮只执行方案最后明确的两个诊断，所有模型与标签均未修改，0个优化更新。A的150例数据导出完成，真实onset人工核验尚未完成；B的fit扫描与锁定点dev读回完成。

**主要结果：fit上85%目标对应阈值0.44、85.29% Recall、39.39候选/min；同一锁定点在dev为91.48% Recall、39.92候选/min、31.86假候选/min。高召回空间存在，过滤成本偏高，尚不能宣称两阶段系统已成立。**

## A：Physical Onset Audit 数据导出

| 样本组 | 可选数 | 导出数 |
| --- | --- | --- |
| radar_hit | 393 | 50 |
| gru_only_false | 857 | 50 |
| ordinary_stable | 215 | 50 |

每组50例，共150例；按8个subject×sensor轮转抽样，各组每cell6或7例，seed42/43/44。Radar组为旧fit8命中GT；GRU组为独立State零阈值未匹配峰、距任意Radar零阈值峰>0.36s；stable组为无GT±2s、无两路fit8 proposal±2s且窗口标签恒定。GRU未匹配包含可能近GT的重复峰，不能把它们全说成远GT物理假动作。分层抽样不能直接估计总体标签错位比例。

每例保存中心±2秒内的完整MD/RD、时间戳、标签、两路分数和GT上下文；提供150张MD/RD/score图、原始NPZ与CSV/JSON审核表。采样清单在查看信号前已固定，无替换样本。

**独立模态检查：只检查这些案例对应的4个原始recording目录及H5键，未发现视频或其他独立人体观测模态。H5包含radar cube、mDoppler、rdn。因此当前只能做proxy audit，不能把雷达峰重定义成physical GT。**

estimated_physical_onset、offset、physical_onset_label和confidence均留空待核验；proxy_change_time/proxy_offset另列，不覆盖协议GT。尚未计算offset直方图、P(|offset|>0.36s)和P(|offset|>0.72s)，也不能判断标签错位是否严重。缺失值未填0。

command_GT明确是canonical的帧标签转换时间(0.09s网格)，而非语音命令精确时间。审核表保留boundary_time_basis、frame_period_s、原始command CSV及SHA，并导出4份原始CSV。converter默认以first command作为录制起点，当前未独立核验录制时间原点/同步；以后测得offset也不能直接解释成人体反应延迟。proxy_confidence单独列出，避免和真实physical confidence混淆。

导出时发现subject_08的canonical RD是MD-only转换留下的空占位，原始H5仍有RD。其36例从H5的rdn按原frame索引补出；完整MD与H5逐帧精确一致、时间网格精确一致后才允许补充。其余114例RD来自canonical NPZ。补充只用于审核，不影响历史模型或本轮候选计算。H5全文件SHA未在导出阶段重算；clip自身SHA及frame对齐均已验证，来源路径和此限制逐例保留。

审核入口：[audit-index.md](D:/lab/SEER/.audit-staging/radar-gru-onset-readback-v3/audit-index.md)；模板：[review-template.csv](D:/lab/SEER/.audit-staging/radar-gru-onset-readback-v3/review-template.csv)；模态清单：[modality-inventory.json](D:/lab/SEER/.audit-staging/radar-gru-onset-readback-v3/modality-inventory.json)。

## B：双源 Candidate Pool

读取旧existing500__radar和独立recurrent__state，未使用失败的residual B，也未融合score。Radar固定此前fit=8选出的阈值0.38；GRU阈值0至1步长0.01，仅在18条fit流扫描。原peak触发器lookahead1update=0.18s、NMS0.36s、neighbor-high与±0.72s一对一matcher完全保持。

在线去重以首次emission建立episode，时间永久不变。0.36s内到达的另一来源峰可附着到一个最近、未配对的episode；同来源不合并，不允许链式延长、平均或倒写时间。最终both标签只在第二来源到达后可用，其时间单独保存。评估实际合并后的候选集合，不使用oracle命中并集代替可部署性能。

| fit目标 | GRU阈值 | fit Recall | fit Density/min | dev Recall | dev Density/min | dev FP/min | dev latency median/P95 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 80.00% | 0.51 | 80.17% | 33.41 | 83.95% | 31.90 | 24.51 | -0.27/0.45 |
| 85.00% | 0.44 | 85.29% | 39.39 | 91.48% | 39.92 | 31.86 | -0.31/0.45 |
| 90.00% | 0.32 | 90.63% | 47.30 | 96.45% | 50.97 | 42.47 | -0.36/0.45 |
| 95.00% | 不可达 | — | — | — | — | — | — |

fit95%不可达：扫描最高Recall为92.84%，Density=56.07/min（GRU零阈值）。预先指定的dev零阈值参考为98.01%、60.91候选/min；不是根据dev选择出的95% operating point。

达到目标时选择最低fit Density，平局依次取较高Recall、较高阈值。先保存fit锁定文件，再读取dev的三个唯一锁定点和预注册零阈值参考；没有dev阈值扫描或重选。

![fit候选Recall–Density曲线](D:/lab/SEER/docs/figures/radar-gru-candidate-recall-density-2026-10-09.png)

## 主点：8个subject×sensor全部结果

| cell | 命中/GT | Recall | Density/min | FP/min | Post-boundary Recall | Latency median/P95 s |
| --- | --- | --- | --- | --- | --- | --- |
| subject_02/TI_xWR14xx | 94/98 | 95.92% | 43.27 | 33.85 | 81.63% | -0.27/0.45 |
| subject_02/TI_xWR68xx | 93/98 | 94.90% | 39.16 | 29.85 | 72.45% | -0.27/0.36 |
| subject_03/TI_xWR14xx | 76/83 | 91.57% | 40.10 | 32.50 | 77.11% | -0.36/0.36 |
| subject_03/TI_xWR68xx | 74/83 | 89.16% | 35.30 | 27.90 | 69.88% | -0.36/0.48 |
| subject_04/TI_xWR14xx | 82/85 | 96.47% | 41.10 | 32.90 | 83.53% | -0.27/0.45 |
| subject_04/TI_xWR68xx | 72/85 | 84.71% | 34.99 | 27.79 | 65.88% | -0.27/0.40 |
| subject_08/TI_xWR14xx | 79/86 | 91.86% | 44.87 | 36.96 | 65.12% | -0.36/0.54 |
| subject_08/TI_xWR68xx | 74/86 | 86.05% | 40.56 | 33.15 | 55.81% | -0.36/0.36 |

| 最终来源 | 候选数 | 一对一命中数 |
| --- | --- | --- |
| radar | 257 | 45 |
| gru | 2183 | 376 |
| both | 751 | 223 |

所有锁定点逐cell/recording结果和每个episode的首次时间、来源到达时间、成员峰score与匹配GT见 [dev-selected-results.json](D:/lab/SEER/.audit-staging/radar-gru-two-diagnostics-v1/dev-selected-results.json)、[candidate-episodes.csv](D:/lab/SEER/.audit-staging/radar-gru-two-diagnostics-v1/candidate-episodes.csv)。

## 判断与边界

B回答了“有没有高召回候选空间”：有，85%fit目标在dev达到91.48%，但约40候选/min，不能按方案示例把成本说成18–20/min。A尚无独立physical onset标注，所以不能排除命令与运动错位。当前两个诊断不证明Verify可过滤、QA可用或完整闭环已跑通。

主点若要把假候选降至8/min，需要过滤约74.9%；方案中的“过滤≥50%”仅会降到约15.93假候选/min。当前已匹配真候选约8.06/min，因此不能把6个总candidate/min与保留91.48% Recall同时当成这份数据上的目标。此处只是负担计算，尚未训练或测量Verify。

继续条件应是先完成proxy/独立模态核验并单独锁定Candidate可过滤性实验；本轮没有训练Probe、Sequential Verify或改变Encoder/TCN/原Heads/损失/采样/trigger。所有数据仍为上游已见、反复使用的TRAIN内部fit/dev，不是独立未见受试者测试。

## 复核与证据

独立实现queue合并，使用原始emitter/matcher重算全部101个fit点、4个fit目标选择和4个dev阈值，共1850次record检查，数值误差为0。150份clip SHA、150×原始数组长度/有限值与全部归档成员SHA核验。新增7项测试通过；全项目pytest尝试在tests/test_parrad_convert.py收集时因interview环境缺h5py中断，未宣称全套通过，也未重装依赖。

读回证据：[independent-readback.json](D:/lab/SEER/.audit-staging/radar-gru-two-diagnostics-v1/independent-readback.json)；片段归档核验：[download-checks.json](D:/lab/SEER/.audit-staging/radar-gru-onset-readback-v3/download-checks.json)；导出冻结/来源核验：[export-checks.json](D:/lab/SEER/.audit-staging/radar-gru-onset-readback-v3/export-checks.json)。

A完整服务器归档SHA：`060b03fc33c7bb033a62b0a9d1434609207e0323c7d9e9c401546af16acac8ef`；B数据来源SHA见 [spec.json](D:/lab/SEER/.audit-staging/radar-gru-two-diagnostics-v1/spec.json)。完整代码、测试、固定清单、fit grid与审核原值均保留。

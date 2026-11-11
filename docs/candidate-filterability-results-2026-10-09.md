# 固定双源候选池：候选可过滤性诊断

全部18个线性诊断Probe完成各500步，共9,000次新Probe更新；原模型0次更新。它们没有安装进v2.5控制器。真实physical onset仍无独立同步标注，本轮评价沿用协议GT。

**预注册主结果（Combined、观察1.08s、calibration锁定阈值0.177）：dev Recall 85.51%，假候选24.407/min，总候选31.938/min；相对原池召回损失5.966个百分点，假候选减少23.40%。**

诊断门槛（过滤≥50%假候选且召回损失≤5pp）：未通过；8FP/min目标及同一召回约束：未通过。这两个判断均不是正式controller Go。

## 锁定配置与信息边界

候选池固定Radar0.38、独立GRU0.44；lag1=.18s、NMS.36s、neighbor-high、首次因果发出时间不倒写的跨源合并及±.72s原matcher不变。set000的10条fit传感器流训练Probe，set001的8条fit流校准阈值，set002的8条dev流一次性读出；同一recording的两路传感器同分区。上游此前已见这些TRAIN记录，不能称独立未见受试者验证。

冻结positive-coordinate兼容EventFormer/P。Active按recording连续前向，只在recording起点初始化，没有GT重置，也没有Confirm状态切换。每个候选于真正发出j之前冻结参考E[j-1]、h[j-1]；Candidate自己的EventFormer从初始状态消费j..j+6，GRU不重置。这是独立shadow可过滤性实验，Active约定不同于正式闭环，失败不能证明Encoder没有转换信息。

Difference含GRU和semantic-key的cosine/相对L2差异；Stability含两空间的短前缀方差和相邻移动。三组均Linear8→1、零初始化，单组证据未用位置补0；seed42、FP32、unweighted候选BCE、全训练候选batch、AdamW500步/lr.01/wd.01/clip1。没有Radar/GRU分数、source、subject、绝对时间或maturity输入。

训练与AP曲线使用共同完整6-update候选集；事件结果保留全池GT分母，对所需前缀缺失候选保守拒绝。完整性只用于离线队列统计，不作为在线输入。过滤后重新一对一匹配，因此原先未匹配的重复候选可以替代被拒绝的匹配候选。候选分类标签不等于事件命中数。

每组/k在calibration的0:.001:1网格上锁定三种策略：①召回损失≤5pp时尽量降低FP（主策略）；②FP≤原池50%时尽量保留Recall；③FP≤8/min时尽量保留Recall。18组全部锁保存之后才执行dev预测。主方法Combined/k6预先固定，未择优挑dev表现。

检测覆盖始终使用原proposal时间t_p匹配；decision时间为t_p+k*.18s，单独报告，未通过修改时间或扩大窗口提高召回。AP随观察时间上升只能支持这些证据变得更可分，不能单独证明时序因果或正式累计器有效。

## 主方法的三种校准策略

| 策略 | 阈值 | dev Recall | FP/min | 总候选/min | 召回损失pp | FP减少 | 决策延迟median/P95 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| retain_recall | 0.177 | 85.51% | 24.407 | 31.938 | 5.966 | 23.40% | 0.810/1.530 |
| halve_fp | 0.218 | 72.44% | 17.377 | 23.757 | 19.034 | 45.47% | 0.810/1.620 |
| fp8 | 0.255 | 44.46% | 8.107 | 12.022 | 47.017 | 74.56% | 0.990/1.710 |

原池参考：Recall 91.48%，FP 31.863/min，总候选 39.920/min。

![固定成熟度结果曲线](D:/lab/SEER/docs/figures/candidate-filterability-2026-10-09.png)

## 全部18组点级结果（共同完整6-update队列）

| 组 | 观察s | train AP | cal AP | dev AP | dev AUROC | dev正例/负例 |
| --- | --- | --- | --- | --- | --- | --- |
| difference-k1 | 0.180 | 0.247 | 0.258 | 0.226 | 0.557 | 642/2543 |
| difference-k2 | 0.360 | 0.249 | 0.268 | 0.236 | 0.570 | 642/2543 |
| difference-k3 | 0.540 | 0.245 | 0.277 | 0.240 | 0.564 | 642/2543 |
| difference-k4 | 0.720 | 0.248 | 0.282 | 0.246 | 0.560 | 642/2543 |
| difference-k5 | 0.900 | 0.253 | 0.291 | 0.251 | 0.560 | 642/2543 |
| difference-k6 | 1.080 | 0.259 | 0.292 | 0.250 | 0.556 | 642/2543 |
| stability-k1 | 0.180 | 0.228 | 0.265 | 0.223 | 0.541 | 642/2543 |
| stability-k2 | 0.360 | 0.234 | 0.238 | 0.226 | 0.546 | 642/2543 |
| stability-k3 | 0.540 | 0.242 | 0.235 | 0.230 | 0.544 | 642/2543 |
| stability-k4 | 0.720 | 0.233 | 0.231 | 0.218 | 0.541 | 642/2543 |
| stability-k5 | 0.900 | 0.252 | 0.242 | 0.227 | 0.564 | 642/2543 |
| stability-k6 | 1.080 | 0.264 | 0.258 | 0.231 | 0.573 | 642/2543 |
| combined-k1 | 0.180 | 0.261 | 0.269 | 0.239 | 0.577 | 642/2543 |
| combined-k2 | 0.360 | 0.269 | 0.274 | 0.250 | 0.593 | 642/2543 |
| combined-k3 | 0.540 | 0.275 | 0.270 | 0.252 | 0.585 | 642/2543 |
| combined-k4 | 0.720 | 0.271 | 0.270 | 0.260 | 0.586 | 642/2543 |
| combined-k5 | 0.900 | 0.278 | 0.278 | 0.270 | 0.596 | 642/2543 |
| combined-k6 | 1.080 | 0.288 | 0.294 | 0.275 | 0.598 | 642/2543 |

## 全部18组事件结果（主校准策略）

| 组 | 阈值 | dev Recall | FP/min | 总候选/min | 召回损失pp | FP减少 | 决策P95 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| difference-k1 | 0.178 | 83.52% | 26.647 | 34.003 | 7.955 | 16.37% | 0.630 |
| difference-k2 | 0.178 | 83.24% | 26.484 | 33.815 | 8.239 | 16.88% | 0.810 |
| difference-k3 | 0.179 | 85.23% | 27.185 | 34.691 | 6.250 | 14.68% | 0.990 |
| difference-k4 | 0.177 | 86.36% | 27.660 | 35.266 | 5.114 | 13.19% | 1.170 |
| difference-k5 | 0.173 | 86.36% | 27.873 | 35.479 | 5.114 | 12.52% | 1.350 |
| difference-k6 | 0.166 | 88.07% | 28.473 | 36.229 | 3.409 | 10.64% | 1.530 |
| stability-k1 | 0.203 | 87.07% | 27.635 | 35.304 | 4.403 | 13.27% | 0.630 |
| stability-k2 | 0.197 | 86.79% | 27.110 | 34.753 | 4.688 | 14.92% | 0.810 |
| stability-k3 | 0.184 | 86.65% | 26.672 | 34.303 | 4.830 | 16.29% | 0.990 |
| stability-k4 | 0.192 | 86.08% | 25.796 | 33.377 | 5.398 | 19.04% | 1.170 |
| stability-k5 | 0.186 | 86.36% | 25.496 | 33.102 | 5.114 | 19.98% | 1.350 |
| stability-k6 | 0.183 | 85.37% | 24.970 | 32.489 | 6.108 | 21.63% | 1.530 |
| combined-k1 | 0.171 | 85.80% | 26.134 | 33.690 | 5.682 | 17.98% | 0.630 |
| combined-k2 | 0.171 | 85.37% | 25.821 | 33.340 | 6.108 | 18.96% | 0.810 |
| combined-k3 | 0.169 | 87.22% | 25.596 | 33.277 | 4.261 | 19.67% | 1.080 |
| combined-k4 | 0.172 | 86.51% | 25.608 | 33.227 | 4.972 | 19.63% | 1.260 |
| combined-k5 | 0.180 | 86.65% | 24.658 | 32.289 | 4.830 | 22.61% | 1.350 |
| combined-k6 | 0.177 | 85.51% | 24.407 | 31.938 | 5.966 | 23.40% | 1.530 |

## 主方法8个subject×sensor

| cell | 命中/GT | Recall | FP/min | 候选/min | 缺前缀拒绝数 | 决策P95 s |
| --- | --- | --- | --- | --- | --- | --- |
| subject_02/TI_xWR14xx | 90/98 | 91.84% | 24.639 | 33.654 | 0 | 1.580 |
| subject_02/TI_xWR68xx | 87/98 | 88.78% | 23.838 | 32.552 | 0 | 1.503 |
| subject_03/TI_xWR14xx | 73/83 | 87.95% | 25.803 | 33.103 | 0 | 1.440 |
| subject_03/TI_xWR68xx | 73/83 | 87.95% | 21.702 | 29.003 | 0 | 1.530 |
| subject_04/TI_xWR14xx | 76/85 | 89.41% | 25.103 | 32.703 | 2 | 1.530 |
| subject_04/TI_xWR68xx | 65/85 | 76.47% | 19.996 | 26.495 | 1 | 1.530 |
| subject_08/TI_xWR14xx | 74/86 | 86.05% | 29.046 | 36.458 | 1 | 1.651 |
| subject_08/TI_xWR68xx | 64/86 | 74.42% | 25.140 | 31.550 | 2 | 1.426 |

## 候选组成与截尾

| recording | 分区 | 候选数 | 完整6次观察 | matched / nearGT-unmatched / remote-unmatched |
| --- | --- | --- | --- | --- |
| subject_02-set000-TI_xWR14xx | train | 414 | 414 | 87 / 61 / 266 |
| subject_02-set000-TI_xWR68xx | train | 369 | 369 | 75 / 55 / 239 |
| subject_02-set001-TI_xWR14xx | calibration | 415 | 415 | 68 / 47 / 300 |
| subject_02-set001-TI_xWR68xx | calibration | 382 | 382 | 71 / 44 / 267 |
| subject_03-set000-TI_xWR14xx | train | 385 | 384 | 82 / 55 / 248 |
| subject_03-set000-TI_xWR68xx | train | 335 | 335 | 70 / 45 / 220 |
| subject_03-set001-TI_xWR14xx | calibration | 407 | 407 | 86 / 70 / 251 |
| subject_03-set001-TI_xWR68xx | calibration | 423 | 423 | 88 / 67 / 268 |
| subject_04-set000-TI_xWR14xx | train | 394 | 392 | 90 / 60 / 244 |
| subject_04-set000-TI_xWR68xx | train | 387 | 385 | 90 / 67 / 230 |
| subject_04-set001-TI_xWR14xx | calibration | 396 | 394 | 96 / 64 / 236 |
| subject_04-set001-TI_xWR68xx | calibration | 381 | 379 | 93 / 60 / 228 |
| subject_08-set000-TI_xWR14xx | train | 464 | 464 | 81 / 40 / 343 |
| subject_08-set000-TI_xWR68xx | train | 456 | 456 | 85 / 48 / 323 |
| subject_08-set001-TI_xWR14xx | calibration | 440 | 439 | 98 / 66 / 276 |
| subject_08-set001-TI_xWR68xx | calibration | 380 | 379 | 102 / 56 / 222 |
| subject_13-set000-TI_xWR14xx | train | 397 | 396 | 99 / 70 / 228 |
| subject_13-set000-TI_xWR68xx | train | 342 | 342 | 87 / 60 / 195 |
| subject_02-set002-TI_xWR14xx | dev | 432 | 432 | 94 / 75 / 263 |
| subject_02-set002-TI_xWR68xx | dev | 391 | 391 | 93 / 62 / 236 |
| subject_03-set002-TI_xWR14xx | dev | 401 | 401 | 76 / 67 / 258 |
| subject_03-set002-TI_xWR68xx | dev | 353 | 353 | 74 / 58 / 221 |
| subject_04-set002-TI_xWR14xx | dev | 411 | 409 | 82 / 60 / 269 |
| subject_04-set002-TI_xWR68xx | dev | 350 | 349 | 72 / 46 / 232 |
| subject_08-set002-TI_xWR14xx | dev | 448 | 447 | 79 / 59 / 310 |
| subject_08-set002-TI_xWR68xx | dev | 405 | 403 | 74 / 46 / 285 |

nearGT-unmatched包括重复竞争峰，不能解释成没有身体变化的假动作；这组与remote分开保留在逐候选元数据。所有54种arm×k×policy结果及匹配记录完整保留于 [results.json](D:/lab/SEER/.audit-staging/candidate-filterability-readback-v2/results.json)。

## 主方法为什么未达到过滤门槛

| 原池标签组 | 候选数 | 完整6观察 | 平均score | 保留数 | 保留比例 |
| --- | --- | --- | --- | --- | --- |
| matched | 644 | 642 | 0.238 | 572 | 88.82% |
| near_gt_unmatched | 473 | 470 | 0.228 | 411 | 86.89% |
| remote_unmatched | 2074 | 2073 | 0.210 | 1570 | 75.70% |

| 正例matched vs 负例组 | AUROC | AP | 此子集正例占比 |
| --- | --- | --- | --- |
| near_gt_unmatched | 0.552 | 0.638 | 57.73% |
| remote_unmatched | 0.608 | 0.321 | 23.65% |

真匹配与近GT重复峰分数接近；即使只看远GT负例，区分力仍弱。主阈值保留了约75.7%的远GT假候选，因此过滤不足不能仅归因于重复峰标签。原matched保留572个，但过滤后重匹配有602个命中，差出的30个体现了重复候选替代，不可用572直接计算事件Recall。

这支持的结论仅是“本轮8个差异/稳定性标量＋线性Probe过滤不足”。它不等于完整GRU/事件表示缺少信息，也不证明重训Encoder必需；连续Active参考、Candidate冷启动、标量压缩及协议GT限制均需纳入解释。

逐候选决定：[primary-candidate-decisions.csv](D:/lab/SEER/.audit-staging/candidate-filterability-readback-v2/primary-candidate-decisions.csv)；分组统计：[primary-subgroup-analysis.json](D:/lab/SEER/.audit-staging/candidate-filterability-readback-v2/primary-subgroup-analysis.json)。

## 时间与物理起点核验

原150例MD/RD审核材料保留。原command CSV与canonical边界内部对照：704个dev边界全部在命令相对first-command时间之后0..0.09s内，符合帧量化；这不能证明first-command就是实际录制起点。所查原始记录未发现独立同步视频，physical onset/offset/confidence仍空，未修改协议GT，未计算物理错位比例。

时间检查：[candidate-filterability-time-basis.json](D:/lab/SEER/.audit-staging/candidate-filterability-time-basis.json)；150例入口：[audit-index.md](D:/lab/SEER/.audit-staging/radar-gru-onset-readback-v3/audit-index.md)。

## 核验与原始证据

独立读回确认9000次Probe更新、每参数AdamW500步、0步零初始化；服务器独立读回权重预测最大误差0.0，本地跨机器468组weight×record复算最大误差8.940696716308594e-08，均通过1e-7门槛。18×1001校准点用原matcher重算，144576次record检查；62088个前缀特征重算，最大误差1.6225931664948234e-07。每条流取首批Candidate进行未来输入扰动，26批的过去输出误差均0；前后frontend digest一致。

新增与候选池相关15项测试通过。全项目pytest仍在tests/test_parrad_convert.py因interview环境缺h5py收集失败，没有宣称全套通过。v1源预检查因人工复制SHA多一字符停止，0模型回放/0训练；v2改为从原审核归档自动取SHA并统一FP32标准化，未改变实验参数或读取dev后重训，v1失败证据保留。

完整归档SHA：`53ab74e2cd1d50cbfc94563037cf1bc1c7c6c1f809c2534f3cb177cf920cddb6`。前端bundle SHA：`57af183acfdcc1da33c76573b60b57ae60897fd9aca715484058b37340378602`。

- [spec.json](D:/lab/SEER/.audit-staging/candidate-filterability-readback-v2/spec.json)
- [source-manifest.json](D:/lab/SEER/.audit-staging/candidate-filterability-readback-v2/source-manifest.json)
- [extraction-checks.json](D:/lab/SEER/.audit-staging/candidate-filterability-readback-v2/extraction-checks.json)
- [probe-checks.json](D:/lab/SEER/.audit-staging/candidate-filterability-readback-v2/probe-checks.json)
- [calibration-locks.json](D:/lab/SEER/.audit-staging/candidate-filterability-readback-v2/calibration-locks.json)
- [independent-readback.json](D:/lab/SEER/.audit-staging/candidate-filterability-readback-v2/independent-readback.json)
- [download-checks.json](D:/lab/SEER/.audit-staging/candidate-filterability-readback-v2/download-checks.json)

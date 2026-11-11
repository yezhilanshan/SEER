# Stage3B：Oracle → Proposal 域差诊断（2026-10-10）

**结论：Oracle50 学会的关系判断没有迁移到时间局部任务；Proposal 选择并非本轮主要的可见解释。** 统一 k=6、12/7 update 窗口后，固定时间锚点已接近随机。匹配录制和动作配对后，Oracle 仍明显领先时间锚点与 Proposal。当前证据支持先修正语义配对训练任务，不能据此宣判 Encoder 或 P 无信息。

本轮新训练 **0 步**，原模型 **0 步**；Oracle50 checkpoint 原有的 50 步不是本轮更新。不训练 200 步或 adaptation，不改阈值、候选池、trigger，不安装新模型。

![诊断图](D:/lab/SEER/docs/figures/oracle-proposal-domain-gap-2026-10-10.png)

## 1. 三组比较的定义与边界

- O：原 Oracle 随机同录制配对，仅取 k=6；参考 key 为独立 12-update 窗口的最终值，候选 key 为独立 7-update 前缀。
- T：固定时间锚点 j=18,21,24,…，间隔 0.54s；参考 `[j-18:j-6)`，候选 `[j:j+7)`。窗口中心相隔 2.79s，候选观察延迟 1.08s。锚点及前向只依赖时间、有效上下文，不依赖 GT 或分数。
- P：原冻结 Proposal 的原始 index 和原保存 key，完全沿用上一轮 12/7 几何。T 不替换线上 trigger，不用于宣称事件级检出性能。
- Same/different 是**协议动作类别**；完整两侧窗口必须各自同类，且全部帧距离 GT 边界严格超过 0.36s。混合/模糊/无效窗口保持 mask。GT 仅用于离线标签。
- 26 条 TRAIN 开发流：train 10、calibration 8、dev 8；dev 为 4 个 recording 各 2 个 sensor。这些是已看过的上游 TRAIN 开发数据；本轮是回顾诊断，不能称独立未见受试者测试。
O 的候选 bank 原本要求完整 12-update 窗口同类且安全，虽然本轮仅读取其 7-update key；T/P 只要求实际 7-update 候选支持安全。因此 GT 窗口筛选强度也是域差的一部分，本轮不能把所有差距唯一归因于配对时间距离。


## 2. 原始分布结果（dev）

| 域 | 干净配对 / different | AUROC | AP | 50/50 AP | cosine AUROC |
| --- | --- | --- | --- | --- | --- |
| O | 3028 / 1514 | 0.7692 | 0.7810 | 0.7810 | 0.7363 |
| T | 2780 / 494 | 0.4807 | 0.2120 | 0.5375 | 0.5753 |
| P | 988 / 233 | 0.4548 | 0.2549 | 0.5138 | 0.5950 |

O/T/P 的 different 比例分别为 50.00%, 17.77%, 23.58%。因此原始 AP 不能直接横比；即便统一到 50/50 类别质量，时间局部任务仍未恢复到 Oracle 水平。

| 分区 | O AUROC | T AUROC | P AUROC |
| --- | --- | --- | --- |
| train | 0.7690 | 0.4743 | 0.4803 |
| calibration | 0.7280 | 0.4087 | 0.4909 |
| dev | 0.7692 | 0.4807 | 0.4548 |

train 上 T/P 也接近随机，说明现象并不局限于 dev。这里只评估已冻结 Oracle50，未证明同结构 Head 在局部配对监督下不可学。

## 3. 动作配对与录制匹配后的结果

匹配 stratum 为 `(stream recording含sensor, reference action, candidate action)`。仅保留共同支持，每个 stratum 的质量取各域数量最小值，再按每域样本数均分，最后正负各 1/2。权重不使用任何模型分数；缺失 stratum 明确排除。不同比较的支持/权重不同，其 AUROC 差值不能相加当作因果贡献。

| 共同支持 | strata | 保留数（O/T/P按参与顺序） | 保留覆盖率 | Head AUROC（按参与顺序） |
| --- | --- | --- | --- | --- |
| O-T | 222 | 1697 / 2629 | 56.04% / 94.57% | 0.6944 / 0.4778 |
| T-P | 215 | 2626 / 978 | 94.46% / 98.99% | 0.4653 / 0.4616 |
| O-P | 171 | 1511 / 911 | 49.90% / 92.21% | 0.6875 / 0.4533 |
| O-T-P | 166 | 1497 / 2513 / 906 | 49.44% / 90.40% / 91.70% | 0.6909 / 0.4613 / 0.4562 |

三域共同支持仍有 166 个 strata，但只覆盖 49.44% Oracle；不能把结果推广到被排除的配对。

| 域 | 共同支持 Head AUROC | 共同支持 AP（已50/50） | 共同支持 cosine AUROC |
| --- | --- | --- | --- |
| O | 0.6909 | 0.7225 | 0.7000 |
| T | 0.4613 | 0.4931 | 0.5139 |
| P | 0.4562 | 0.4934 | 0.5516 |

共同支持上 O→T 的差距仍约 0.230；T 与 P 相差约 0.005。单独 T/P 匹配且覆盖超过 94%/98% 时，T/P 为 0.4653/0.4616。**主要可见断点发生在“远距离 Oracle 配对 → 时间局部配对”，而非一定要由 Proposal 筛选才出现。** 动作组成确实影响原始结果，但没有消除域差；这里不能排除个别 action/cell 的候选选择效应。

## 4. 时间、方向与事件实例

| Oracle 中心时间差 | 样本 / different | Head AUROC | 50/50 AP |
| --- | --- | --- | --- |
| 5-30s | 186 / 140 | 0.6980 | 0.7073 |
| <=5s | 158 / 9 | 0.6659 | 0.6697 |
| >30s | 2684 / 1365 | 0.7717 | 0.7819 |

dev Oracle 中 2684/3028（88.64%）中心间距 >30s；≤5s 只有 158 对，different 仅 **9 对**。这直接显示原 Oracle 对“局部新旧动作”任务覆盖不足。近距离 AUROC 样本很少，不能把它当作稳定性能估计，也不能断言时间差是唯一原因；动作语义难度、窗口内容及历史仍纠缠在一起。

Oracle 候选在参考之后/之前的自然 AUROC 为 0.7690 / 0.7693。仅保留 forward Oracle 再匹配时，O/T/P 为 0.6469 / 0.4581 / 0.4691；方向差异不能独自解释断点。

Oracle 的 same 对中，167 对来自同一次连续 GT 动作，1347 对来自同类动作的不同出现；平均分分别 0.3650/0.3849。它们都是负类，单独 AUROC 为 null；不能把同类不同实例直接重标成 different。

## 5. 交换输入、cosine 与分数塌缩

| 域 | Head AUROC | 交换 R/C 后 AUROC | same均分 | different均分 |
| --- | --- | --- | --- | --- |
| O | 0.7692 | 0.7588 | 0.3827 | 0.5921 |
| T | 0.4807 | 0.4599 | 0.3229 | 0.3327 |
| P | 0.4548 | 0.4399 | 0.3088 | 0.3100 |

交换仅为冻结诊断，不重选方向或反转输出。交换没有救回 T/P；二者 same 与 different 的均分几乎重叠。P 的 cosine 自然 AUROC 0.5950、共同支持 0.5516，均好于 Head；说明仍有有限的可读差异，而已训练 Head 的配对规则不适用。但 cosine 远未达到强语义判别，不能声称只修 Head 就一定成功，也不能声称 P 已完全解决。

## 6. 八个 subject×sensor（dev）

| cell | 自然 O/T/P AUROC | 共同支持 O/T/P AUROC | 共同支持 O/T/P 数 |
| --- | --- | --- | --- |
| subject_02/TI_xWR14xx | 0.8230 / 0.5050 / 0.4489 | 0.7003 / 0.4565 / 0.5012 | 195 / 246 / 95 |
| subject_02/TI_xWR68xx | 0.7477 / 0.5038 / 0.4324 | 0.5833 / 0.4346 / 0.3817 | 191 / 233 / 91 |
| subject_03/TI_xWR14xx | 0.7506 / 0.4992 / 0.5754 | 0.6830 / 0.4367 / 0.4714 | 182 / 360 / 129 |
| subject_03/TI_xWR68xx | 0.7471 / 0.5020 / 0.5293 | 0.7639 / 0.4601 / 0.4743 | 181 / 355 / 113 |
| subject_04/TI_xWR14xx | 0.7616 / 0.3751 / 0.3624 | 0.5832 / 0.4771 / 0.4614 | 169 / 338 / 107 |
| subject_04/TI_xWR68xx | 0.7466 / 0.4256 / 0.3293 | 0.7160 / 0.3700 / 0.3582 | 178 / 352 / 98 |
| subject_08/TI_xWR14xx | 0.7889 / 0.5302 / 0.5065 | 0.7445 / 0.5701 / 0.4991 | 194 / 314 / 150 |
| subject_08/TI_xWR68xx | 0.7932 / 0.5265 / 0.5205 | 0.8154 / 0.5207 / 0.5697 | 207 / 315 / 123 |

八个 cell 的匹配 O 均高于 T 与 P。subject_08/14xx 的 T 仍有 0.5701，subject_08/68xx 的 P 为 0.5697；局部可用性有异质性，不能只报全局平均。

| 留出的 recording（同时去掉两sensor） | T−O | P−O | P−T |
| --- | --- | --- | --- |
| subject_02-set002 | -0.2442 | -0.2533 | -0.0091 |
| subject_03-set002 | -0.2190 | -0.2311 | -0.0121 |
| subject_04-set002 | -0.2291 | -0.2292 | -0.0001 |
| subject_08-set002 | -0.2285 | -0.2281 | 0.0004 |

这是分组敏感性分析，不是独立样本置信区间。排除 T 中恰好等于 P 的锚点后，重新三域匹配 O/T/P 为 0.6840 / 0.4598 / 0.4514，结论没有改变；剩余窗口仍可能共享历史，不能视为独立。

## 7. 标签覆盖与解释范围

| 分区 | T全部/可前向/干净/mask | P全部/可前向/干净/mask |
| --- | --- | --- |
| train | 11276 / 11246 / 2782 / 8494 | 3943 / 3929 / 985 / 2958 |
| calibration | 8848 / 8824 / 1966 / 6882 | 3224 / 3209 / 692 / 2532 |
| dev | 8840 / 8816 / 2780 / 6060 | 3191 / 3179 / 988 / 2203 |

dev T mask 68.55%，P mask 69.04%。干净配对不代表全部候选；本轮不对模糊 Candidate 强行补 same 标签，不重新计算事件 filtering/Confirm，也不改变原 704 个 GT 分母。没有新增物理 onset 标注，协议边界与物理变化是否一致仍未解决。

## 8. 冻结、数值与独立读回

- 相同原 Oracle batch 的 768 个 key：最大误差 **0**；每条录制新锚点首批均通过 reset、未来扰动不影响过去、独立原始前向循环一致检查（最大误差均 0）。
- 原 Oracle 56808 个 Head 分数和全部可用 P 分数与原保存值误差均 <1e-6；注意只有 k=6 的 9468 个 Oracle 对参与这轮主统计，其他 age 仅核验复现。
- 前后 frontend/Encoder/EventFormer/P/Head digest 完全一致；旧 Head/Evidence/gate 等均被包含在冻结 frontend 中。没有构建优化器；未读取 official holdout。
- 独立 CPU 重建 Head、标签重算、weighted rank 另一算法、完整导出行数核验通过。检查数见下表。

| 核验 | 数量/结果 |
| --- | --- |
| new_updates | 0 |
| original_updates | 0 |
| label_checks | 39203 |
| anchor_availability_checks | 28964 |
| CPU_forward_predictions | 192022 |
| CPU_GPU_max_error | 1.7881393432617188e-07 |
| independent_metric_checks | 513 |
| matched_cohort_checks | 57 |
| export_rows | 19661 |

**原跨 batch FP16 1e-5 证书仍未通过，不予改写。** 新 T/P 相同窗口不同 batch 的最大 key 误差为 reference 0.00027392245829105377、candidate 0.0004936624318361282，Head 分数最大误差 0.0003641843795776367。这与现有批次敏感性一致；不同 batch 不满足精确复现，但相同 batch 的算子检查通过。不能据此颁发 formal 数值证书，目前没有证据表明这些误差能解释约 0.23 AUROC 域差。

## 9. 下一步建议与执行边界

**不直接扩大 Oracle 到 200 步，不立即对 Proposal 做 closed-loop adaptation，不解冻 Encoder/P。** 下一轮先提交一个“时间局部语义配对”的固定小预算 pilot：只训练同一 Semantic Difference Head，使用训练录制中 T 几何构造的 stable same / stable different，保持上下游、12/7窗口和标签/mask不变；先有 0-step，再做 50-step。训练与开发只按既定分区，预锁类别/录制权重和门槛，报告自然及共同支持 O/T/P，不能根据 dev 挑 checkpoint。

其科学问题是：**已有 key 上，局部配对监督能否让同一个 Head 学会真正要用的局部语义关系？** 若 train T 仍学不起来，才拆窗口语义污染/局部动作可分性与 P；若 T 学会但 P 失败，才考虑 Proposal selection / cold-start adaptation。若 T/P 判别变好但事件过滤不够，再进入 Evidence + Stability + sequential controller，而不是训练 Verify classifier。本报告不构成新训练授权，本轮没有启动这些训练。

## 10. 原始证据与身份

- [批准计划](D:/lab/SEER/docs/superpowers/plans/2026-10-10-oracle-proposal-domain-gap.md)
- [执行账本](D:/lab/SEER/docs/superpowers/plans/2026-10-10-oracle-proposal-domain-gap-progress.md)
- [完整统计 JSON](D:/lab/SEER/.audit-staging/oracle-proposal-gap-analysis-v2/results.json)
- [全部干净配对 CSV](D:/lab/SEER/.audit-staging/oracle-proposal-gap-analysis-v2/all-clean-pairs.csv)
- [全部分组 CSV](D:/lab/SEER/.audit-staging/oracle-proposal-gap-analysis-v2/all-group-results.csv)
- [匹配支持明细](D:/lab/SEER/.audit-staging/oracle-proposal-gap-analysis-v2/matched-strata.csv)
- [独立读回](D:/lab/SEER/.audit-staging/oracle-proposal-gap-analysis-v2/independent-readback.json)
- [原始前向与冻结核验](D:/lab/SEER/.audit-staging/oracle-proposal-gap-readback-v1/extraction-checks.json)
- [下载与成员核验](D:/lab/SEER/.audit-staging/oracle-proposal-gap-readback-v1/download-checks.json)

服务器：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/oracle_proposal_domain_gap_v1`，launcher PID 9379，2026-10-10 08:58:35 CST 启动，已完成。原始归档 SHA256：`4c5bc5ab03d1efafb30b522b32f2c1bb58f7aea0e12ab9915083deba32126c00`，68,687,377 bytes，209 成员，208 逐成员 SHA 验证通过后才安全解压。

代码包 SHA256：`b3a94c4ccb6e5178ca2c5063ae3daa9b8bc63a9cabfc3acdbba7ba5e7efcac5f`；Head checkpoint SHA256：`2fc9e7b8b0a1688664a59626d51b26bc4841560ff8c242b0ee3d1e18f39194bd`。旧正式模型与旧失败归档均未覆盖。统计修正（零质量样本在加权排序前排除）仅在本地分析 core，服务器不可变包的 core 未调用统计函数；两者源文件 SHA 在最终交付 manifest 分别保留。

完整 clean NPZ、masked T NPZ、O/P原始 key、日志、代码、JSON 保留在上述 raw/analysis 目录；最终交付 manifest 与派生归档另附。**这是诊断完成，不是模型 Go 或正式闭环通过。**


最终审查修正：匹配 JSON 的辅助分数分布现也按共同支持及匹配质量计算；旧版 v1 未删除，最终请使用 analysis-v2。主 AUROC/AP 及结论未变。独立读回还从原始样本重建全部匹配支持集合和 minimum_count，而非信任统计输出。最终24项回归测试通过。
[审查与修正记录](D:/lab/SEER/.audit-staging/oracle-proposal-gap-analysis-v2/review.md)。派生归档：`.audit-staging/oracle-proposal-gap-derived-evidence-v1.tar.gz`；交付清单：`.audit-staging/oracle-proposal-gap-delivery-manifest.json`；最终核验：`.audit-staging/oracle-proposal-gap-delivery-checks.json`。


最终审查修正：匹配 JSON 的辅助分数分布现也按共同支持及匹配质量计算；旧版 v1 未删除，最终请使用 analysis-v2。主 AUROC/AP 及结论未变。独立读回还从原始样本重建全部匹配支持集合和 minimum_count，而非信任统计输出。最终24项回归测试通过。
[审查与修正记录](D:/lab/SEER/.audit-staging/oracle-proposal-gap-analysis-v2/review.md)。派生归档：`.audit-staging/oracle-proposal-gap-derived-evidence-v1.tar.gz`；交付清单：`.audit-staging/oracle-proposal-gap-delivery-manifest.json`；最终核验：`.audit-staging/oracle-proposal-gap-delivery-checks.json`。

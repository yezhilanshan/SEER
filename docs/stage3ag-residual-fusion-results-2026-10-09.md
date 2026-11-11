# Stage3A-G：冻结 GRU × 旧 Radar residual fusion 结果

本轮 A、B 各完成 500 个实际 AdamW 更新；C 与所有上游模块 0 更新。完整证据归档、权重重建及本地指标复算通过。

**预注册结论：No-Go（不安装，不自动追加训练）。**
fit=8 下开发集 Recall：C=56.11%，A=55.26%，B=37.36%。B−C=-18.75pp，B−A=-17.90pp。

B 的零阈值合法峰覆盖由C的66.19%增至96.16%，同时零阈值FP由14.162/min增至55.370/min。GRU增加了候选覆盖，但在固定误报预算下未能保住旧Radar的选择性。
Rescue=57，Harm=189，净收益=-132；8个cell全部退步。B的AP提高到0.30603，不能抵消事件Recall下降。

## 实验配置与来源

18 个 fit recording、8 个 TRAIN 内部开发 recording，共 26 条连续流；开发集 704 个可评估边界。上游已见过这些受试者，且开发集被反复使用；不是独立未见受试者测试，不能作为正式 Go。未访问 official holdout。

| 项目 | 锁定设置 |
| --- | --- |
| C/A/B | 旧 Radar / 零 GRU 容量对照 / 真实 GRU residual |
| Residual | 共享 Linear128→64；concat 两个投影和 zR；Linear129→64/GELU/Linear64→1；最后层零初始化 |
| 参数 | A/B 各 16,641 个参数、6 个参数张量，初始权重一致 |
| 输入 | 原始 h(t−1),h(t)，同一 causal update；无显式 delta、额外 TCN、Local、OR、gate |
| 优化 | seed42；各500 AdamW；16 chunks×64；lr0.001；weight_decay0.01；clip1.0；FP32 |
| 损失 | 原始 weighted Gaussian BCE，w+=3.341091295046995，w−=0.5879943564065809；逐 chunk 按加权目标质量归一化，再16 chunk等权均值 |
| 采样 | 复用上一轮固定 schedule，无新采样策略；共同前6 update有效性排除保持一致 |
| 冻结 | Encoder/Fused/Token/旧Local与Radar/EventFormer/兼容gate/P/Evidence/Semantic/Candidate全部0更新 |
| Trigger | 固定 causal peak；lookahead1 update=0.18s；NMS0.36s；原 neighbor-high；grid0:.01:1 |
| 阈值 | 各组只用 fit 锁定3/5/8/10/15/20 FP/min，之后正式dev一次；无dev选模型或调参 |

原 Stage3A BCE 按加权目标质量归一化；上一轮读取位置 probe 按有效点数归一化。因此跨实验不能把差值全部归因于 fusion；本轮 A/B 内部只有 GRU 输入是否真实这一变量。

| 证据 | SHA256 |
| --- | --- |
| 上传包 v2 | f81feeea156542849e3c23e2db8bf1dc28209a34452b16e42c9487012ac3f625 |
| 最终归档 | 2c3df9fb96a3aea9ad1a6b434c180536c30378a81a419ff719fd991314e57f6f |
| spec | 0eb8b7c63b4c102a1559bd68c3d5e2c7cba3edfcd1623448e7cc86753090b988 |
| source manifest | b1d634b912ae5c6953b57a6344378ba3c35ea6e2b392dbe69c2e4d8b08462a67 |
| 旧 weighted500 checkpoint | de06a21a0a12816ec82e0fbc71e972c415916ac172bf009867fb67b9923bd89e |
| Encoder state | a2c85175d7c95986682ca55dec886042fbe57ecfedeaa41d6155dbdfe2cb38e7 |
| 冻结 Radar state | 493e6d955368a95d189cc771d76bba4738e25af01739c9a2f967fd383f25c0c4 |
| Residual 初始 state | e3aaba2d393cac45ea5aa6b32d055a8d8b6657ddd00e625674cb8bf1011f2a13 |

完整来源/26份 layer SHA、代码与方案 SHA 见 [source-manifest.json](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/source-manifest.json)、[native-preflight.json](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/native-preflight.json) 和 [spec.json](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/spec.json)。

## Zero-update 与实际更新核验

| 检查 | 结果 |
| --- | --- |
| 原始 zR | 冻结 weighted500 Head 连续前向取得，不从概率反推 |
| C复现 | 六个预算的 dev 指标与上一轮 existing500__radar 精确一致 |
| A/B step0 max logit error | {'A': 0.0, 'B': 0.0} |
| 概率复现 max error | 0.0000000000 |
| 原模块字节核验 | True |
| Encoder digest | a2c85175d7c95986682ca55dec886042fbe57ecfedeaa41d6155dbdfe2cb38e7 |
| 实际 AdamW 步数 | {'A': 500, 'B': 500} |
| 每参数 optimizer step | {'A': [500, 500, 500, 500, 500, 500], 'B': [500, 500, 500, 500, 500, 500]} |
| Checkpoint | A/B 均保存0/100/500，含optimizer状态 |
| 接口测试 | 本地23项通过；服务器23项通过；缺历史绕过，recording reset、prefix因果和跨chunk前序一致 |
| CPU读回 | {'A': {'actual_steps': 500, 'parameter_steps': [500, 500, 500, 500, 500, 500], 'all_valid_predictions_checked': 87300, 'max_probability_error': 1.1920928955078125e-07, 'max_logit_error': 2.384185791015625e-07, 'checkpoint_sha': 'df6efa827a196ea351f65f26fddd82669a77ce0b3ef42f7b566aab83fd18c5fd', 'step0_max_logit_error': 0.0}, 'B': {'actual_steps': 500, 'parameter_steps': [500, 500, 500, 500, 500, 500], 'all_valid_predictions_checked': 87300, 'max_probability_error': 1.7881393432617188e-07, 'max_logit_error': 9.5367431640625e-07, 'checkpoint_sha': '7169daa129a6606b303deb07910fe30bbed54dc05c6d4910ddce6e5a9462d8fc', 'step0_max_logit_error': 0.0}} |
| 本地复算 | {'archive_members_verified': 159, 'score_metrics_recomputed': 120, 'fit_grid_recomputed': 303, 'dev_event_settings_recomputed': 18, 'original_trigger_record_crosschecks': 416, 'original_updates': 0, 'residual_updates': 1000, 'holdout_access': False, 'formal_go': False, 'all_prediction_points_reconstructed': 174964, 'max_cpu_reconstruction_error': 9.5367431640625e-07, 'all_boundary_rows_verified': 15114, 'all_legal_peak_rows_verified': 15972, 'complementarity_verified': True, 'success_recomputed': True, 'max_numeric_error': 2.220446049250313e-16, 'passed': True} |
| 归档核验 | 160 个成员，48139078 bytes，SHA和成员路径核验通过 |

缓存 recurrent state 使用上一轮已核验的26条流及其 layer SHA；本轮核对 Encoder 权重 digest 与缓存来源，不宣称重新全量运行 MD-GRU。旧 Radar 则本轮重新前向。测试中的数学 chunk 等价使用FP64排除不同batch矩阵乘舍入；实际训练与预测均为FP32，step0 logit误差门槛未放宽。

| 组 | step | project grad | input grad | output grad | project.weight grad |
| --- | --- | --- | --- | --- | --- |
| A | 1 | 0.00000000 | 0.00000000 | 0.00828336 | 0.00000000 |
| A | 2 | 0.00007820 | 0.00018780 | 0.03724983 | 0.00000000 |
| A | 5 | 0.00009247 | 0.00024374 | 0.02130220 | 0.00000000 |
| A | 100 | 0.00145549 | 0.00305045 | 0.01237605 | 0.00000000 |
| A | 500 | 0.00397307 | 0.00370748 | 0.01999307 | 0.00000000 |
| B | 1 | 0.00000000 | 0.00000000 | 0.03345119 | 0.00000000 |
| B | 2 | 0.00064792 | 0.00053081 | 0.04455320 | 0.00063762 |
| B | 5 | 0.00177004 | 0.00135440 | 0.03634915 | 0.00176196 |
| B | 100 | 0.02816971 | 0.01533307 | 0.05947011 | 0.02799213 |
| B | 500 | 0.09971769 | 0.04192180 | 0.09414144 | 0.09828378 |

梯度均为裁剪前范数。最后层零初始化使step1前级梯度为0；B从step2开始前级非零。A输入全零，投影weight梯度始终为0是容量对照预期行为。

## 点级结果（开发集）

| 组 | AP | AUROC | boundary mean | stable mean | stable P95 |
| --- | --- | --- | --- | --- | --- |
| C | 0.27951 | 0.71393 | 0.50876 | 0.36590 | 0.67737 |
| A | 0.27950 | 0.71393 | 0.54091 | 0.41777 | 0.67973 |
| B | 0.30603 | 0.75612 | 0.57043 | 0.40182 | 0.69680 |

点级 positive/stable 掩码沿用冻结来源；Gaussian过渡区域的中间点不用于二值AP/AUROC。

## Proposal 结果（每个阈值只在 fit 选择）

| 组 | fit cap | 阈值 | fit FP/min | dev Recall | dev FP/min | Density/min | Latency median s | P95 s | Post-boundary Recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C | 3 | 0.66 | 2.731 | 25.14% | 2.264 | 4.479 | 0.090 | 0.630 | 15.91% |
| A | 3 | 0.67 | 2.588 | 24.01% | 2.089 | 4.203 | 0.090 | 0.684 | 15.20% |
| B | 3 | 0.73 | 2.522 | 17.47% | 2.027 | 3.565 | -0.090 | 0.630 | 11.65% |
| C | 5 | 0.59 | 4.902 | 38.78% | 3.941 | 7.356 | 0.000 | 0.630 | 23.58% |
| A | 5 | 0.61 | 4.952 | 39.20% | 3.953 | 7.406 | 0.000 | 0.630 | 24.01% |
| B | 5 | 0.71 | 4.369 | 25.28% | 3.753 | 5.980 | -0.180 | 0.630 | 16.62% |
| C | 8 | 0.38 | 7.996 | 56.11% | 7.669 | 12.610 | 0.000 | 0.657 | 34.09% |
| A | 8 | 0.45 | 7.919 | 55.26% | 7.469 | 12.335 | 0.000 | 0.684 | 33.81% |
| B | 8 | 0.68 | 7.743 | 37.36% | 7.168 | 10.458 | -0.180 | 0.540 | 27.27% |
| C | 10 | 0.22 | 9.952 | 63.21% | 11.247 | 16.814 | 0.000 | 0.630 | 38.07% |
| A | 10 | 0.29 | 9.996 | 63.21% | 11.372 | 16.939 | 0.000 | 0.630 | 38.07% |
| B | 10 | 0.67 | 9.079 | 41.76% | 8.482 | 12.160 | -0.270 | 0.540 | 30.82% |
| C | 15 | 0.08 | 11.508 | 66.05% | 14.074 | 19.891 | 0.000 | 0.630 | 39.77% |
| A | 15 | 0.13 | 11.513 | 66.05% | 14.074 | 19.891 | 0.000 | 0.630 | 39.77% |
| B | 15 | 0.63 | 14.607 | 52.84% | 13.386 | 18.040 | -0.270 | 0.450 | 40.20% |
| C | 20 | 0.08 | 11.508 | 66.05% | 14.074 | 19.891 | 0.000 | 0.630 | 39.77% |
| A | 20 | 0.13 | 11.513 | 66.05% | 14.074 | 19.891 | 0.000 | 0.630 | 39.77% |
| B | 20 | 0.60 | 18.811 | 61.65% | 16.701 | 22.130 | -0.270 | 0.450 | 46.88% |

| 组 | 零阈值合法峰覆盖 | 零阈值FP/min |
| --- | --- | --- |
| C | 66.19% | 14.162 |
| A | 66.19% | 14.162 |
| B | 96.16% | 55.370 |

![Recall–FP曲线](D:/lab/SEER/docs/figures/stage3ag-residual-recall-fp-2026-10-09.png)

曲线只连接六个 fit 锁定阈值的 dev 点，横轴为实际 dev FP/min；数字为fit cap，不是dev扫描结果，也不保证开发集满足同名FP上限。负延迟允许早期提出怀疑；Proposal时间不是最终边界。

## 全部8个 subject×sensor

| cell | C Recall@8 | A Recall@8 | B Recall@8 | B−C pp | B−A pp | C/A/B实际FP/min |
| --- | --- | --- | --- | --- | --- | --- |
| subject_02/TI_xWR14xx | 67.35% | 67.35% | 35.71% | -31.63 | -31.63 | 8.013 / 7.913 / 7.712 |
| subject_02/TI_xWR68xx | 65.31% | 65.31% | 48.98% | -16.33 | -16.33 | 6.310 / 6.310 / 10.016 |
| subject_03/TI_xWR14xx | 42.17% | 42.17% | 28.92% | -13.25 | -13.25 | 7.901 / 7.901 / 5.901 |
| subject_03/TI_xWR68xx | 48.19% | 44.58% | 30.12% | -18.07 | -14.46 | 7.801 / 6.801 / 5.201 |
| subject_04/TI_xWR14xx | 57.65% | 57.65% | 37.65% | -20.00 | -20.00 | 7.401 / 7.301 / 5.901 |
| subject_04/TI_xWR68xx | 50.59% | 50.59% | 37.65% | -12.94 | -12.94 | 6.499 / 6.299 / 5.299 |
| subject_08/TI_xWR14xx | 59.30% | 59.30% | 41.86% | -17.44 | -17.44 | 9.515 / 9.515 / 9.115 |
| subject_08/TI_xWR68xx | 54.65% | 51.16% | 36.05% | -18.60 | -15.12 | 7.913 / 7.712 / 8.213 |

| cell | 组 | AP | AUROC | stable mean |
| --- | --- | --- | --- | --- |
| subject_02/TI_xWR14xx | C | 0.30827 | 0.74854 | 0.39356 |
| subject_02/TI_xWR14xx | A | 0.30827 | 0.74854 | 0.44091 |
| subject_02/TI_xWR14xx | B | 0.33550 | 0.79914 | 0.39655 |
| subject_02/TI_xWR68xx | C | 0.37233 | 0.76273 | 0.37687 |
| subject_02/TI_xWR68xx | A | 0.37233 | 0.76273 | 0.42624 |
| subject_02/TI_xWR68xx | B | 0.35770 | 0.79349 | 0.39304 |
| subject_03/TI_xWR14xx | C | 0.25196 | 0.69054 | 0.34094 |
| subject_03/TI_xWR14xx | A | 0.25196 | 0.69054 | 0.39503 |
| subject_03/TI_xWR14xx | B | 0.27782 | 0.74979 | 0.38668 |
| subject_03/TI_xWR68xx | C | 0.27752 | 0.68035 | 0.34395 |
| subject_03/TI_xWR68xx | A | 0.27752 | 0.68035 | 0.39922 |
| subject_03/TI_xWR68xx | B | 0.28765 | 0.72766 | 0.38645 |
| subject_04/TI_xWR14xx | C | 0.30081 | 0.71337 | 0.34879 |
| subject_04/TI_xWR14xx | A | 0.30081 | 0.71337 | 0.40318 |
| subject_04/TI_xWR14xx | B | 0.36957 | 0.77483 | 0.38847 |
| subject_04/TI_xWR68xx | C | 0.23757 | 0.67535 | 0.34786 |
| subject_04/TI_xWR68xx | A | 0.23757 | 0.67535 | 0.40417 |
| subject_04/TI_xWR68xx | B | 0.29791 | 0.73744 | 0.39653 |
| subject_08/TI_xWR14xx | C | 0.26500 | 0.71203 | 0.40314 |
| subject_08/TI_xWR14xx | A | 0.26500 | 0.71203 | 0.44972 |
| subject_08/TI_xWR14xx | B | 0.28640 | 0.74112 | 0.44113 |
| subject_08/TI_xWR68xx | C | 0.24456 | 0.70670 | 0.37436 |
| subject_08/TI_xWR68xx | A | 0.24456 | 0.70670 | 0.42557 |
| subject_08/TI_xWR68xx | B | 0.25921 | 0.70085 | 0.42556 |

全部 cell、六预算的 Recall/FP/Density/latency/Post-recall 原始指标见 [results.json](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/results.json)；下表完整披露各预算Recall。

| cell | 组 | R@3 | R@5 | R@8 | R@10 | R@15 | R@20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| subject_02/TI_xWR14xx | C | 30.61% | 53.06% | 67.35% | 70.41% | 70.41% | 70.41% |
| subject_02/TI_xWR14xx | A | 28.57% | 54.08% | 67.35% | 70.41% | 70.41% | 70.41% |
| subject_02/TI_xWR14xx | B | 11.22% | 20.41% | 35.71% | 44.90% | 61.22% | 74.49% |
| subject_02/TI_xWR68xx | C | 39.80% | 52.04% | 65.31% | 67.35% | 70.41% | 70.41% |
| subject_02/TI_xWR68xx | A | 38.78% | 52.04% | 65.31% | 67.35% | 70.41% | 70.41% |
| subject_02/TI_xWR68xx | B | 24.49% | 31.63% | 48.98% | 54.08% | 68.37% | 74.49% |
| subject_03/TI_xWR14xx | C | 15.66% | 28.92% | 42.17% | 45.78% | 50.60% | 50.60% |
| subject_03/TI_xWR14xx | A | 13.25% | 28.92% | 42.17% | 45.78% | 50.60% | 50.60% |
| subject_03/TI_xWR14xx | B | 14.46% | 16.87% | 28.92% | 33.73% | 44.58% | 55.42% |
| subject_03/TI_xWR68xx | C | 16.87% | 31.33% | 48.19% | 66.27% | 68.67% | 68.67% |
| subject_03/TI_xWR68xx | A | 15.66% | 31.33% | 44.58% | 66.27% | 68.67% | 68.67% |
| subject_03/TI_xWR68xx | B | 19.28% | 22.89% | 30.12% | 33.73% | 45.78% | 51.81% |
| subject_04/TI_xWR14xx | C | 23.53% | 37.65% | 57.65% | 65.88% | 69.41% | 69.41% |
| subject_04/TI_xWR14xx | A | 23.53% | 37.65% | 57.65% | 65.88% | 69.41% | 69.41% |
| subject_04/TI_xWR14xx | B | 15.29% | 24.71% | 37.65% | 41.18% | 47.06% | 56.47% |
| subject_04/TI_xWR68xx | C | 18.82% | 29.41% | 50.59% | 63.53% | 68.24% | 68.24% |
| subject_04/TI_xWR68xx | A | 18.82% | 30.59% | 50.59% | 63.53% | 68.24% | 68.24% |
| subject_04/TI_xWR68xx | B | 18.82% | 25.88% | 37.65% | 40.00% | 48.24% | 54.12% |
| subject_08/TI_xWR14xx | C | 27.91% | 40.70% | 59.30% | 65.12% | 66.28% | 66.28% |
| subject_08/TI_xWR14xx | A | 26.74% | 41.86% | 59.30% | 65.12% | 66.28% | 66.28% |
| subject_08/TI_xWR14xx | B | 18.60% | 33.72% | 41.86% | 44.19% | 53.49% | 59.30% |
| subject_08/TI_xWR68xx | C | 24.42% | 32.56% | 54.65% | 59.30% | 62.79% | 62.79% |
| subject_08/TI_xWR68xx | A | 23.26% | 32.56% | 51.16% | 59.30% | 62.79% | 62.79% |
| subject_08/TI_xWR68xx | B | 17.44% | 25.58% | 36.05% | 39.53% | 50.00% | 62.79% |

## Radar/GRU 零更新互补矩阵与 Oracle union

| 条件 | 数据 | 都命中 | Radar-only | GRU-only | 都漏检 | 总数 | Oracle union |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fit8 | fit | 586 | 335 | 268 | 626 | 1815 | 65.51% |
| fit8 | dev | 224 | 171 | 80 | 229 | 704 | 67.47% |
| zero | fit | 984 | 40 | 672 | 119 | 1815 | 93.44% |
| zero | dev | 452 | 14 | 228 | 10 | 704 | 98.58% |

Oracle union是两路独立一对一匹配后的边界命中集合之并，不是可部署检测器性能，也不控制联合误报。fit8使用两路各自此前fit阈值。

| 条件 | cell | 都命中 | Radar-only | GRU-only | 都漏检 | Oracle union |
| --- | --- | --- | --- | --- | --- | --- |
| fit8 | subject_02/TI_xWR14xx | 45 | 21 | 15 | 17 | 82.65% |
| fit8 | subject_02/TI_xWR68xx | 48 | 16 | 17 | 17 | 82.65% |
| fit8 | subject_03/TI_xWR14xx | 18 | 17 | 9 | 39 | 53.01% |
| fit8 | subject_03/TI_xWR68xx | 16 | 24 | 7 | 36 | 56.63% |
| fit8 | subject_04/TI_xWR14xx | 20 | 29 | 10 | 26 | 69.41% |
| fit8 | subject_04/TI_xWR68xx | 26 | 17 | 7 | 35 | 58.82% |
| fit8 | subject_08/TI_xWR14xx | 32 | 19 | 7 | 28 | 67.44% |
| fit8 | subject_08/TI_xWR68xx | 19 | 28 | 8 | 31 | 63.95% |
| zero | subject_02/TI_xWR14xx | 66 | 3 | 29 | 0 | 100.00% |
| zero | subject_02/TI_xWR68xx | 66 | 3 | 28 | 1 | 98.98% |
| zero | subject_03/TI_xWR14xx | 40 | 2 | 39 | 2 | 97.59% |
| zero | subject_03/TI_xWR68xx | 58 | 0 | 25 | 0 | 100.00% |
| zero | subject_04/TI_xWR14xx | 58 | 1 | 25 | 1 | 98.82% |
| zero | subject_04/TI_xWR68xx | 57 | 1 | 24 | 3 | 96.47% |
| zero | subject_08/TI_xWR14xx | 55 | 2 | 29 | 0 | 100.00% |
| zero | subject_08/TI_xWR68xx | 52 | 2 | 29 | 3 | 96.51% |

动作转换使用数据集 class-ID，不自行编造动作名。全部fit/dev逐边界与失败组合见 [zero-complementarity.csv](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/zero-complementarity.csv) 和 [zero-complementarity.json](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/zero-complementarity.json)。

| 条件 | 开发集转换ID | 都命中 | Radar-only | GRU-only | 都漏检 |
| --- | --- | --- | --- | --- | --- |
| fit8 | 0->10 | 43 | 3 | 5 | 7 |
| fit8 | 1->3 | 4 | 0 | 4 | 2 |
| fit8 | 1->4 | 7 | 3 | 5 | 1 |
| fit8 | 1->7 | 14 | 2 | 2 | 2 |
| fit8 | 10->11 | 9 | 12 | 5 | 12 |
| fit8 | 10->12 | 6 | 9 | 1 | 4 |
| fit8 | 11->1 | 14 | 14 | 3 | 15 |
| fit8 | 11->13 | 0 | 1 | 0 | 1 |
| fit8 | 11->5 | 5 | 5 | 2 | 14 |
| fit8 | 12->11 | 1 | 0 | 0 | 3 |
| fit8 | 12->6 | 6 | 18 | 6 | 32 |
| fit8 | 13->0 | 12 | 14 | 3 | 29 |
| fit8 | 13->11 | 0 | 2 | 0 | 2 |
| fit8 | 13->12 | 1 | 1 | 0 | 2 |
| fit8 | 2->12 | 7 | 8 | 0 | 9 |
| fit8 | 2->13 | 6 | 5 | 2 | 7 |
| fit8 | 3->2 | 1 | 1 | 6 | 10 |
| fit8 | 3->4 | 3 | 2 | 2 | 3 |
| fit8 | 3->7 | 1 | 6 | 2 | 3 |
| fit8 | 4->2 | 2 | 7 | 1 | 4 |
| fit8 | 4->3 | 1 | 11 | 1 | 3 |
| fit8 | 4->7 | 1 | 5 | 1 | 1 |
| fit8 | 5->8 | 6 | 7 | 1 | 12 |
| fit8 | 6->9 | 31 | 6 | 8 | 17 |
| fit8 | 7->2 | 3 | 3 | 3 | 3 |
| fit8 | 7->3 | 3 | 3 | 4 | 6 |
| fit8 | 7->4 | 5 | 2 | 3 | 2 |
| fit8 | 8->12 | 4 | 5 | 0 | 5 |
| fit8 | 8->13 | 4 | 4 | 2 | 2 |
| fit8 | 9->11 | 10 | 5 | 3 | 8 |
| fit8 | 9->13 | 14 | 7 | 5 | 8 |
| zero | 0->10 | 44 | 2 | 8 | 4 |
| zero | 1->3 | 5 | 0 | 5 | 0 |
| zero | 1->4 | 10 | 0 | 6 | 0 |
| zero | 1->7 | 16 | 0 | 4 | 0 |
| zero | 10->11 | 24 | 0 | 13 | 1 |
| zero | 10->12 | 14 | 2 | 4 | 0 |
| zero | 11->1 | 38 | 0 | 7 | 1 |
| zero | 11->13 | 1 | 0 | 1 | 0 |
| zero | 11->5 | 15 | 1 | 10 | 0 |
| zero | 12->11 | 2 | 0 | 2 | 0 |
| zero | 12->6 | 35 | 1 | 26 | 0 |
| zero | 13->0 | 43 | 0 | 14 | 1 |
| zero | 13->11 | 2 | 0 | 2 | 0 |
| zero | 13->12 | 3 | 0 | 1 | 0 |
| zero | 2->12 | 16 | 1 | 7 | 0 |
| zero | 2->13 | 11 | 0 | 8 | 1 |
| zero | 3->2 | 2 | 0 | 16 | 0 |
| zero | 3->4 | 5 | 0 | 4 | 1 |
| zero | 3->7 | 7 | 0 | 5 | 0 |
| zero | 4->2 | 10 | 0 | 4 | 0 |
| zero | 4->3 | 8 | 4 | 4 | 0 |
| zero | 4->7 | 5 | 1 | 2 | 0 |
| zero | 5->8 | 17 | 1 | 7 | 1 |
| zero | 6->9 | 42 | 0 | 20 | 0 |
| zero | 7->2 | 7 | 0 | 5 | 0 |
| zero | 7->3 | 6 | 0 | 10 | 0 |
| zero | 7->4 | 7 | 0 | 5 | 0 |
| zero | 8->12 | 10 | 1 | 3 | 0 |
| zero | 8->13 | 9 | 0 | 3 | 0 |
| zero | 9->11 | 17 | 0 | 9 | 0 |
| zero | 9->13 | 21 | 0 | 13 | 0 |

## Failure decomposition

| fit cap | 组 | matched | below_threshold | neighbor_constraint | peak_shape | nms_suppressed | invalid_context | match_conflict | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | C | 177 | 449 | 8 | 70 | 0 | 0 | 0 | 0 |
| 3 | A | 169 | 467 | 4 | 64 | 0 | 0 | 0 | 0 |
| 3 | B | 123 | 532 | 30 | 19 | 0 | 0 | 0 | 0 |
| 5 | C | 273 | 317 | 2 | 112 | 0 | 0 | 0 | 0 |
| 5 | A | 276 | 314 | 2 | 112 | 0 | 0 | 0 | 0 |
| 5 | B | 178 | 469 | 38 | 19 | 0 | 0 | 0 | 0 |
| 8 | C | 395 | 126 | 1 | 181 | 0 | 1 | 0 | 0 |
| 8 | A | 389 | 130 | 1 | 183 | 0 | 1 | 0 | 0 |
| 8 | B | 263 | 365 | 44 | 32 | 0 | 0 | 0 | 0 |
| 10 | C | 445 | 30 | 0 | 229 | 0 | 0 | 0 | 0 |
| 10 | A | 445 | 28 | 0 | 231 | 0 | 0 | 0 | 0 |
| 10 | B | 294 | 344 | 34 | 32 | 0 | 0 | 0 | 0 |
| 15 | C | 465 | 2 | 0 | 237 | 0 | 0 | 0 | 0 |
| 15 | A | 465 | 2 | 0 | 237 | 0 | 0 | 0 | 0 |
| 15 | B | 372 | 258 | 34 | 40 | 0 | 0 | 0 | 0 |
| 20 | C | 465 | 2 | 0 | 237 | 0 | 0 | 0 | 0 |
| 20 | A | 465 | 2 | 0 | 237 | 0 | 0 | 0 | 0 |
| 20 | B | 434 | 210 | 15 | 45 | 0 | 0 | 0 | 0 |

沿用原分类器优先级；peak_shape只表示匹配窗内未形成当前规则所需候选，不能一概解释成宽平台。fit的完整失败分解亦保存在 [failure-rescue-summary.json](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/failure-rescue-summary.json)。

## Rescue / Harm

| fit cap | 数据 | Rescue | Harm | 净收益 | 旧zero无合法峰的Rescue | 互斥Rescue分组 |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | fit | 73 | 295 | -222 | 54 | {'old_zero_peak_shape': 54, 'old_has_peak_threshold_filtered': 17, 'other': 2} |
| 3 | dev | 43 | 97 | -54 | 25 | {'old_zero_peak_shape': 25, 'old_has_peak_threshold_filtered': 18} |
| 5 | fit | 102 | 383 | -281 | 96 | {'old_zero_peak_shape': 96, 'old_has_peak_threshold_filtered': 4, 'other': 2} |
| 5 | dev | 43 | 138 | -95 | 36 | {'old_zero_peak_shape': 36, 'old_has_peak_threshold_filtered': 7} |
| 8 | fit | 177 | 436 | -259 | 176 | {'old_zero_peak_shape': 176, 'other': 1} |
| 8 | dev | 57 | 189 | -132 | 57 | {'old_zero_peak_shape': 57} |
| 10 | fit | 209 | 459 | -250 | 209 | {'old_zero_peak_shape': 209} |
| 10 | dev | 68 | 219 | -151 | 68 | {'old_zero_peak_shape': 68} |
| 15 | fit | 306 | 321 | -15 | 306 | {'old_zero_peak_shape': 306} |
| 15 | dev | 93 | 186 | -93 | 93 | {'old_zero_peak_shape': 93} |
| 20 | fit | 366 | 239 | 127 | 366 | {'old_zero_peak_shape': 366} |
| 20 | dev | 113 | 144 | -31 | 113 | {'old_zero_peak_shape': 113} |

old_zero_peak_shape是旧zero无合法峰的子集，与old_zero_no_legal_peak_other互斥；两者之和才是“旧zero无合法峰救回数”，不可再相加重复计数。旧峰存在但阈值/neighbor-high过滤归到old_has_peak_threshold_filtered。逐边界C/A/B命中、失败和B匹配见 [boundary-results.csv](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/boundary-results.csv)。

## B的 True/False peak 分数分析

| 数据 | peak中心是否在GT±.72s | 数量 | mean zR | mean Δz | mean zfinal | median Δz | Δz P10 | Δz P90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fit | near_GT_peak | 3223 | 0.19138 | 0.32213 | 0.51351 | 0.30089 | -0.10969 | 0.80185 |
| fit | far_GT_peak | 7646 | -0.40819 | 0.33368 | -0.07451 | 0.35649 | -0.17968 | 0.89463 |
| dev | near_GT_peak | 1356 | 0.03944 | 0.41388 | 0.45332 | 0.39434 | -0.01895 | 0.86958 |
| dev | far_GT_peak | 3747 | -0.54350 | 0.40113 | -0.14236 | 0.45214 | -0.11537 | 0.89670 |

这里的true/false是零阈值合法峰中心与GT的距离代理，多个近GT峰不能视为多个真实事件。完整CSV另存emission时间是否近GT及是否被8-budget唯一匹配、subject/sensor和前后class-ID；这些标签只用于离线解释。见 [B-legal-peaks.csv](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/B-legal-peaks.csv)。

## 预注册门槛与判断

| 门槛 | 通过 |
| --- | --- |
| recall_B_C | **否** |
| recall_B_A | **否** |
| ap | 是 |
| fp | 是 |
| latency | 是 |
| six_cells | **否** |
| worst_cell | **否** |
| budget3 | **否** |
| budget5 | **否** |
| rescue_gt_harm | **否** |

差值与cell明细：`{"assessed": true, "go": false, "tests": {"recall_B_C": false, "recall_B_A": false, "ap": true, "fp": true, "latency": true, "six_cells": false, "worst_cell": false, "budget3": false, "budget5": false, "rescue_gt_harm": false}, "delta": {"recall_B_C": -0.18749999999999994, "recall_B_A": -0.17897727272727276, "ap_B_C": 0.026520446296908584, "fp_B_C": -0.5004065803465316, "latency_p95_B_C": -0.11700000000000821}, "cell_deltas": {"subject_02/TI_xWR14xx": -0.31632653061224486, "subject_02/TI_xWR68xx": -0.16326530612244894, "subject_03/TI_xWR14xx": -0.13253012048192775, "subject_03/TI_xWR68xx": -0.18072289156626503, "subject_04/TI_xWR14xx": -0.19999999999999996, "subject_04/TI_xWR68xx": -0.12941176470588234, "subject_08/TI_xWR14xx": -0.17441860465116282, "subject_08/TI_xWR68xx": -0.18604651162790697}, "rescue": 57, "harm": 189, "net_rescue": -132, "scope": "Development next-experiment eligibility only; no production or independent-test Go"}`

这轮失败发生在候选选择性，而非完全没有新增变化信息：B救回57个旧zero无合法峰边界，但损失更多旧命中。peak_shape从181降至32，below_threshold从126升至365，neighbor_constraint从1升至44。fit为满足误报上限选出的阈值，过滤掉了大量边界候选。
B合法峰中的近GT与远GT residual均值分别为0.41388和0.40113，差距很小；这项条件统计支持“新增residual对真/假候选区分不够”的解释，但不能单独证明根因。
A与C近似、B在主指标同时落后A和C，属于当前融合实验失败。按预注册分流停止此融合方案，不追加seed、不加gate或其他结构，不安装。现有证据仍表明GRU能补部分盲区，不能据此断言Encoder没有转换信息或立即重训Encoder。后续若另行启动，应回到GT command与实际物理onset、MD可观测性、transition supervision或RD输入增益等独立问题，先锁定新方案。

若未同时优于A和C，不能宣称GRU/Radar互补已被验证。即便点级AP或合法峰覆盖提高，也必须通过固定误报预算、cell和延迟限制，才能进入下一轮开发验证。本轮不根据dev追加训练、重选checkpoint或调阈值。

## 完整证据

原始JSON/CSV、预测NPZ、aligned h/zR、A/B 0/100/500权重与optimizer状态、500步日志、完整代码和SHA manifest均保留。归档包含生成归档时的archiving状态快照；完成状态在服务器归档SHA写出后才更新，本地由完整归档SHA/成员核验确认交付完成。

关键入口：[archive-members.json](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/archive-members.json), [predictions.npz](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/predictions.npz), [aligned-inputs.npz](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/aligned-inputs.npz), [results.json](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/results.json), [success-checks.json](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/success-checks.json), [server-independent-readback.json](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/server-independent-readback.json), [local-independent-readback.json](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/local-independent-readback.json), [download-archive-verification.json](D:/lab/SEER/.audit-staging/stage3ag-residual-results-readback-v1/download-archive-verification.json)。

服务器隔离输出：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3ag_gru_radar_residual_v1`。没有修改旧模型、安装新分支、进入Semantic/Candidate或读取official holdout。

# Frozen Semantic–Physical Complementarity Pilot

用户已批准本轮限定诊断及独立 set003 标定划分。此文件记录运行口径，不是正式模型改动授权。

## 目标与边界

检查 F、E、z 的 transition evidence 是否互补。复用 `frozen-short-horizon-sequential-v1` 的连续因果轨迹，禁止加载 validation/test 数据，不重新回放或训练 Radar Encoder、Event Former、LoRA、Semantic Projection、controller、Verify。生产阈值只作原始轨迹 provenance 哈希校验，绝不改动。

仅24条 TRAIN recordings：subject_02/03/04 × set000–003 × TI_xWR14xx/TI_xWR68xx。三折 probe-held-subject，并非冻结上游从未见过该受试者的独立验证。训练标签只用 clean update 的 same/different GT-run；沿用源提取器的 mixed/ignored/gap/multiple-boundary 排除语义。

## 预先固定的实验矩阵

Natural fixed age4/5/6/7：K1/3；age8：K1/3/8。Natural actual terminal 分为 min-age3（K1/3）和 min-age8（K1/3/8），只含真实 confirm/reject，不含 recording-end。Forced fixed age4/5/6/7/8 和13分别使用相同历史规则，单独解释，不能代替自然早期结果。

每组先以最大 K 选公共 cohort。每个 Candidate 在一组中只有一个 endpoint。该组的所有 family/K 使用同一拟合、标定、评估 Candidate IDs/标签。共有13组、30个 group×K、5 families×3 seeds×3 folds，即1350条实验记录；不存在有效拟合条件时保留 undefined 记录。

输入仅因果 `current-reference` relational vector；F/E/z/F+E/F+z 原始残差历史先按拟合集的 source RMS 尺度归一，再共同投影128维，追加同一合并空间的 cosine distance/L2两项。固定 random-sign projection seeds3407/7211/12893，所有组130输入/131 Logistic 参数。不含 age、maturity、candidate lifetime、controller decision、Verify score、threshold。不做 MLP 或 +age secondary。

## 拟合与独立标定

每个 endpoint/cohort 独立拟合，不混 Natural/Forced。外层两名 probe-training subjects 的 set000–002 拟合、set003 标定，held subject 全4 sessions仅评估；两个 sensor 绑定同 session 划分。尺度、标准化与 class weight 只用拟合数据。

使用 float64 LBFGS strong-wolfe Logistic，ridge λ=0.001、max_iter400、tol_grad1e-7、tol_change1e-9；BA固定 decision threshold0.5。独立标定负例 higher-quantile，并拒绝 cutoff 全部 ties，为目标 calibration FPR5%/10%各生成阈值。不保证 held FPR5%/10%，必须并报实际 held FPR。

报告 AUROC、average precision（AP，不混称梯形积分 PR-AUC）、BA、两工作点 recall/precision/实际FPR、TP/FP/TN/FN、各划分正负数量与 prevalence，以及标定负例数、经验FPR分辨率。无标定负例时工作点 undefined，不能改用 held 标签标定。无拟合双类别时不拟合，held 单类别 AUROC undefined。

按 subject/seed/endpoint/cohort 计算 K3−K1、K8−K1，以及 fusion−两条 constituent 的 paired AUROC/AP/BA/recall/实际FPR/precision delta。先 subject macro 再 seed median/range，保留每折结果及缺失计数；稀疏 age8 不作为强结论依据。

## Provenance 与判读

新目录 `semantic-physical-complementarity-pilot-v1`，不覆盖旧实验。保存 source manifest/checkpoint/config/controller/Event Former/运行代码 SHA、源 cache fingerprint、完整 cohort reference/update metadata、每折 fit/calibration/held endpoint IDs、labels、predictions、模型状态、尺度/标准化/class weight、projection seed、thresholds、metrics 与每个 NPZ SHA。运行前后验证不可变来源；失败保留现场，不自行删除重跑。

Natural 早期 E/z 强→下一步考虑 online-distribution-matched Semantic Difference/sequence reader，不重训 Former。F强→保留 physical residual evidence path。Fusion在各 subject/seed 稳定改善低FPR recall/AP且实际FPR没有明显恶化→支持 Semantic+Physical Sequential Evidence。所有 early-natural 仍弱→停止读出扩容，下一阶段考虑 transition-aware representation training（Semantic Projection→Event State→Former），本轮不自动实施。

## 当前实现验证

新增25项测试（含完整合成矩阵和两个审查问题回归）及新旧相关共75项通过（116.63s）。完整合成矩阵实际产生1350记录，其中1080拟合、270为空的 Natural age8/terminal-min8记录；这些明确标为 undefined，没有补样。审查补齐了指标实现哈希，并将 manifest 中的 solver 配置直接传入实际拟合。

完整本地套件在 `tests/test_parrad_convert.py` 收集阶段因 interview 环境缺少 h5py 中止；不安装依赖。服务器独立 SEER 环境完整套件357通过、2失败（37.58s）：`test_gemma3_adapter_matches_native_prefix_and_targets_only_qv` 与 `test_gemma3_adapter_honors_local_sliding_window` 都是既有 Gemma `None_inv_freq` 兼容问题，用户已排除 Gemma 修复；另有一条既有 torch AMP FutureWarning。新增诊断测试无失败。

上传代码包 SHA256：`6a8a7a03be703491c3c78dc538495f3d74282bbe3bb8de131bd3185ce679f1ec`，服务器核验一致。使用 tar --keep-old-files 提取，只新增本轮文件；旧 sequence/controller/Event Former 本地SHA均与来源基线一致。

正式诊断由 `.audit-staging/run-complementarity-pilot-v1.sh` 于2026-10-02 20:16:45 CST在服务器 CPU nice10 启动，GPU屏蔽。Launcher PID20939、Python PID20942，启动后约2分钟检查 Python为RNl、CPU约100%单核、RSS约1.2GiB。输出根：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/semantic-physical-complementarity-pilot-v1`；日志同名 `.logs` 目录。来源校验已通过并开始拟合；首组 natural-fixed_age-4 已完成（3619个共同endpoint，耗时70.03s）。这只是启动健康状态，不是完整实验结果或互补性结论。

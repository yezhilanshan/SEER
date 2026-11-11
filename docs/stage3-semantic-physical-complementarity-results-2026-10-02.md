# Frozen Semantic–Physical Complementarity Pilot：结果与下一步

## 结论

Radar F 仍是最强的整体排序信息源。E 保留一定额外证据，F+E 在部分低 FPR 标定工作点显示局部 recall 增益；但没有稳定优于 F 的 AUROC/AP，也没有解决跨受试者误报漂移，因此不能宣布 v2.5 的 Semantic + Physical Sequential Evidence 已成立或上线。

F+z 没有稳定胜过 F；简单 K3/K8 残差历史拼接再压缩至128维，也没有带来可靠收益。停止继续扩大冻结读出 head、扫 warm-up/正式阈值；保留 F 作为物理证据参考，下一步建议申请受限的 transition-aware Semantic Projection pilot，而不是直接重训整个 Event Former。

这是对本轮冻结、固定投影、线性读出协议的判断，不是证明 E/z 完全没有 transition 信息，也不是否定所有语义—物理融合或所有时序模型。

## 完成、范围和完整性

2026-10-02 20:16:45–20:38:56 CST，流水线约22分11秒；probe manifest 计时1300.45秒，不含部分启动开销。13/13组、1350/1350拟合记录完成；全部131参数 Logistic 收敛。无 undefined fit、无 undefined calibration point。

仅24条 TRAIN recording，三个 probe-held-subject folds；外层训练受试者 set000–002拟合、set003独立标定，held subject全部session评估。上游冻结模型见过这些 TRAIN subjects，不能称整个模型的独立 subject-generalization test。validation/test 数据未读取，未修改生产模型/controller/阈值。

2026-10-02分析时重新下载只含诊断输出的四个 JSON 文件，核验 summary/results/counts SHA 与 manifest 一致；核对每条记录容量131、收敛状态、calibration FPR≤目标、同组同 subject/seed 全部 family/K 的 held/fit/calibration IDs/labels fingerprints 一致。前一轮完成检查还核验了全部1350个模型NPZ的SHA。

- Source manifest SHA：`5d9439e5b4c8cb533bc0de067faf7d2626307ec4a98c9a50e90e1b23638af440`
- Summary SHA：`bf89dbe26156bcec4b87ad169f1d4c971bf6610d61fc566be3190e80baedac1f`
- Results SHA：`c20f4537fda990be218014069fa13e2170072c0d008f3ed4ea5af081c7cc7aa9`
- Counts SHA：`17417780d638f5efeff42df5d2e04f3d610173b643556932b51a8d6ae499b597`

完整远程输出：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/semantic-physical-complementarity-pilot-v1`。本地输出位于 `.audit-staging/results-complementarity-v1`，复算脚本为 `.audit-staging/analyze-complementarity-v1.ps1`，完整表在 `analysis.txt`。

## 统计口径

下表数值先按三个 held subjects做macro，再取三个固定projection seeds的median。AP为average precision，不是梯形积分PR-AUC。BA固定classification threshold0.5。Recall5/10表示**在独立标定集负例上以目标FPR5%/10%生成阈值，再应用held subjects**，不是held FPR被控制在5%/10%。因此每项recall都与实际held FPR、precision一起看。

每个endpoint组使用同一max-K合格cohort；一组中每Candidate仅一个endpoint，所有family/K共享fit/calibration/heldIDs及标签。不同age组的cohort和prevalence会变化，不能把跨age AP上升直接视作时间带来的因果收益。

## 1. Natural实际终止点：F整体排序最好，F+E只有局部尾部增益

`natural-terminal-min3`：3442个endpoint，712正/2730负，pooled prevalence20.69%。

| Family | K | AUROC | AP | BA@0.5 | Recall5 | 实际FPR5 | Precision5 | Recall10 | 实际FPR10 | Precision10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F | 1 | .6754 | .3495 | .6310 | 15.28% | 7.28% | 42.31% | 29.94% | 13.40% | 40.26% |
| E | 1 | .6291 | .3161 | .6036 | 17.17% | 7.21% | 37.61% | 27.37% | 12.29% | 38.38% |
| z | 1 | .5931 | .2681 | .5719 | 9.91% | 6.34% | 30.37% | 17.53% | 10.77% | 29.92% |
| F+E | 1 | .6559 | .3414 | .6165 | 18.52% | 7.31% | 41.84% | 30.33% | 13.33% | 40.04% |
| F+z | 1 | .6647 | .3351 | .6215 | 13.05% | 6.39% | 33.39% | 26.97% | 14.07% | 36.29% |
| F | 3 | .6661 | .3392 | .6252 | 13.38% | 6.83% | 38.73% | 25.36% | 12.84% | 37.69% |
| E | 3 | .6286 | .3076 | .5963 | 12.82% | 6.49% | 38.28% | 23.28% | 12.60% | 35.83% |
| z | 3 | .6017 | .2784 | .5830 | 9.33% | 5.83% | 27.51% | 18.25% | 10.33% | 31.47% |
| F+E | 3 | .6623 | .3424 | .6135 | 17.08% | 7.59% | 39.60% | 30.63% | 14.94% | 38.11% |
| F+z | 3 | .6534 | .3292 | .6121 | 14.64% | 6.68% | 37.39% | 24.90% | 12.81% | 36.80% |

F+E K1相对F K1的真实paired delta：AUROC −.0194、AP −.0090、Recall5 +3.33个百分点、held FPR5 +.04个百分点。AUROC只2/9方向为正；Recall5在8/9提高，但只有2/9同时FPR不升。不能说全是更宽松阈值，也不能凭macro误报几乎不变就宣称稳定互补：各subject误报增减在平均时相互抵消。

Paired delta是每个subject/seed差值再聚合，**不是两个表格seed medians直接相减**。9个方向检查是3subject×3seed，不是9个独立临床/统计试验，不据此声称统计显著。

## 2. 实际决策年龄：age4弱，age6/7 F较强；融合未稳定超过最佳单路

下表为K1，展示同age同cohort比较。

| Natural age / elapsed since proposal | N / positives | F AUROC | E AUROC | z AUROC | F+E AUROC | F+z AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| 4 / .54s | 3619 / 515 | .5750 | .5560 | .5438 | .5792 | .5689 |
| 5 / .72s | 2567 / 454 | .6414 | .5954 | .5399 | .6212 | .6170 |
| 6 / .90s | 2312 / 493 | .6733 | .6190 | .5659 | .6618 | .6501 |
| 7 / 1.08s | 1996 / 513 | .6834 | .6489 | .5753 | .6629 | .6652 |
| 8 / 1.26s（稀疏） | 134 / 16 | .5455 | .6549 | .5669 | .6148 | .5261 |

age7 F：AP .3968、BA .6380、Recall5 8.82%/held FPR4.71%；F+E：AP .3865、BA .6193、Recall5 15.21%/held FPR6.47%。Paired Recall5 +6.00个百分点，held FPR +1.76个百分点，AUROC −.0213，只有2/9实现recall提升且FPR不升。因此有tail信号，但没有满足“稳定明显优于两条单路”的要求。

age6 F+E的Recall5也在9/9方向上提高，paired +6.94个百分点，但FPR paired +1.31个百分点且AUROC −.0129；只有3/9没有FPR上升。age4 F+E的AUROC/AP略有改善，幅度小且绝对能力仍弱。不是全部family严格随机，也不是可直接当可靠刹车。

删除显式age输入不保证表示没有隐含maturity信息。因此固定age比较比混合年龄terminal更能排除显式年龄捷径；本轮确实在固定age6/7看到tail增益，但仍未建立强互补性。

## 3. 跨受试者标定失配：macro隐藏了严重风险

Natural terminal K1，下表每个subject取seed median。各阈值独立由另外两个subject的set003产生。

| Held subject | N / pos | F AUROC | F Recall5 | F实际FPR5 | E实际FPR5 | F+E Recall5 | F+E实际FPR5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 02 | 1199 / 264 | .6300 | 27.27% | 16.47% | 12.41% | 27.65% | 13.69% |
| 03 | 1098 / 223 | .7077 | 8.97% | 2.86% | 5.71% | 10.76% | 4.23% |
| 04 | 1145 / 225 | .6906 | 10.22% | 2.50% | 3.59% | 14.22% | 4.02% |

真实calibration FPR约4.87–4.91%，标定负例448–472个，所有拟合满足目标上限。subject02 held FPR明显超标，是跨subject score分布/阈值transfer失败的实测表现，不是没有用独立标定或calibration数学出错；具体传感器/动作/场景原因本轮未进一步归因。

F+E terminal AUROC subject02 .6375、03 .6476、04 .6837：只在02略胜F，而03从F .7077降到 .6476。因此最弱subject的小改善不能代表全体泛化提高。

训练—held gap也明显：E各折训练AUROC约.751–.801，held约.595–.672；F+E训练约.778–.812，held约.638–.684。这提示offline probe存在泛化瓶颈，不能直接等同于完整Stage2模型过拟合，更不能凭此宣称优化器不学习。

## 4. 预测重叠：E并非完全冗余，但新正例伴随新误报

只读加载terminal K1的F/E/F+E已保存模型预测与已标定阈值，逐fold断言 heldIDs和标签完全一致。

在9个subject×seed组合中，E能单独找到F未接受的15–30个正例，同时带来23–72个额外误报。F+E相对F并非简单只增加接受：同时增删正例和负例。

例如seed3407：

| Subject | F+E新增TP | 丢失原TP | 新增FP | 去掉原FP | 净TP | 净FP |
| --- | --- | --- | --- | --- | --- | --- |
| 02 | 30 | 26 | 57 | 69 | +4 | −12 |
| 03 | 18 | 11 | 35 | 10 | +7 | +25 |
| 04 | 21 | 9 | 23 | 7 | +12 | +16 |

可见语义表示存在局部不同的错误模式，不能说E完全无额外信息；但目前读出无法稳定把有用额外正例和新增误报分开。“弱/局部互补迹象”比“互补性已成立”更符合证据。

## 5. K3/K8并未提供可靠时序收益

Natural terminal同cohort K3−K1 paired AUROC：F −.0092、E +.0045、z +.0131、F+E +.0003、F+z −.0113。z略有排序/AP收益，但Recall5没有同步提升（paired −.58个百分点）；F+E Recall5 paired −.41个百分点、FPR +.95个百分点。不存在稳定的sequence救活信号。

该实现测的是固定容量的**线性拼接历史**。更多原始维度压缩进相同128维，既不能新增非线性temporal interaction，也可能稀释有效方向。因此结论只能是这套冻结读出协议没有收益，不能证明GRU/attention/显式transition-aware memory必然无效。

Natural age8只有134样本/16正例，每held subject仅4/5/7正例；calibration每折仅13/19/18负例，FPR最小经验步长7.69%/5.26%/5.56%。5%标定点全部calibration FP=0，是取最大负例以上的非常粗糙阈值。部分拟合train AUROC=1.0而held不稳定，不能把K8 .60–.66的某个数字当作新架构证据，也不能用age8选择胜者。

## 6. Forced只是晚期/删失对照

Forced age13：3064样本/1126正例（prevalence36.75%）。K1 AUROC F .7837、E .7301、z .6951、F+E .7692、F+z .7698；F仍最高，融合没有超过它。F Recall5 28.98%/held FPR8.65%，不是在held FPR5%下召回28.98%。

这表明某些晚期forced表示可解码程度更高，不支持“所有表示彻底没transition信息”。但forced采样、age、标签和cohort不同；不能推断延长正式timeout就能获得该性能，也不授权新的warm-up/timeout sweep或上线。

## 下一步建议（本报告不授权自动实施）

1. 结束这一轮冻结读出扩容，不继续堆Radar MLP、投影seeds或正式controller阈值。保留F的原始冻结接口作为物理证据和比较基线。
2. 进入受限transition-aware representation pilot：优先仅训练Semantic Projection，冻结Radar/Former/LoRA/controller，使用真实连续Candidate和实际Frozen Active pair的clean same/different-run监督，并保留原动作/文本语义的约束，避免把动作表示改成纯boundary计时器。不要回到proposal距人工boundary的旧标签，也不要用摆拍maturity样本作为主训练分布。
3. 目标先是改善z在natural age4–7/actualterminal的跨subject可用transition证据，并恢复E中可读出的部分信号；不能承诺只改Projection就能到AUROC .75，因为E本身在早期也不强，且本轮E不是严格信息上界。
4. 用相同fit/calibration/held规则比较新旧z、F及融合，目标必须同时包括低FPR recall、实际held FPR、AP和各subject稳定性；不得靠调整held cutoff补救。若Projection改善有限，再对Event State施加transition objective；整个Former重训放最后。
5. 跨subject标定漂移列为单独验收问题：从训练受试者内预先规定鲁棒标定/风险约束，不准用held subject反向调阈值。架构与协议冻结后才按新授权使用真正未读取的validation；test继续封存。

以上是有限、按证据排序的下一阶段建议，不是这轮已经实现的正式Verify/evidence accumulator。

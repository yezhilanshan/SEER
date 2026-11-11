# 受试者均衡50步小实验结果，2026-10-04

训练已正常完成：2026-10-04T19:38:55+08:00至2026-10-04T20:04:57+08:00，共1562秒。原始JSON/log下载后SHA与服务器一致；主结果由保存预测重算，实际权重与AdamW在服务器独立读回。

## 固定50步的主要比较

所有差值为绝对百分点。两个固定关系系数分别报告，不择优合并。原聚合为受试者/age_updates等权、三个固定特征seed取median。

| 条件 | 均衡相对原权重ΔAUROC | ΔAP | 均衡相对原PΔAUROC | ΔAP | 12组不下降数量 | 最差组相对原P：原→均衡 | 全部50步资格 | 探索标准 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| joint-relation1 | -0.0823 | -0.0618 | 1.5137 | 1.4793 | 9/12 | -2.4812 → -3.2932 | 通过 | 未通过 |
| joint-relation10 | -0.0641 | -0.0348 | 1.5339 | 1.5210 | 8/12 | -2.3759 → -2.9323 | 未通过 | 未通过 |

**两个系数均未通过预注册探索标准，受试者均衡不升级为新基线。**

两个系数的平均AUROC均低于同预算原权重；前两折的小幅正变化没有抵消第三折退步。简单均衡没有解决本轮局部退步。

预注册标准：相对原权重macro AUROC至少+0.5pp、至少两折正且余折非负、macro AP不下降；最差组改善、至少8/12组不下降，三折50步原语义门槛完整通过。

## 每折20/50步的AUROC与AP

| 排除折/条件 | 原P AUROC/AP | 原权重20 | 均衡20 | 原权重50 | 均衡50 | 均衡50相对原权重ΔAUROC/ΔAP(pp) |
| --- | --- | --- | --- | --- | --- | --- |
| subject_02/joint-relation1 | 0.625848/0.334647 | 0.649809/0.351740 | 0.650461/0.352472 | 0.662696/0.368969 | 0.662811/0.368516 | 0.0115/-0.0453 |
| subject_02/joint-relation10 | 0.625848/0.334647 | 0.650679/0.352688 | 0.650834/0.352822 | 0.663249/0.368874 | 0.663294/0.368832（不合格） | 0.0045/-0.0042 |
| subject_03/joint-relation1 | 0.562906/0.282353 | 0.567182/0.286221 | 0.567436/0.284958 | 0.571007/0.290541 | 0.572887/0.293043 | 0.1880/0.2503 |
| subject_03/joint-relation10 | 0.562906/0.282353 | 0.567681/0.286492 | 0.568224/0.285179 | 0.570329/0.290274 | 0.572600/0.292501 | 0.2272/0.2227 |
| subject_04/joint-relation1 | 0.585991/0.299251 | 0.583993/0.297304 | 0.582486/0.297019 | 0.588921/0.302972 | 0.584458/0.299069 | -0.4464/-0.3903 |
| subject_04/joint-relation10 | 0.585991/0.299251 | 0.584257/0.297960 | 0.582672/0.297940 | 0.589109/0.303773 | 0.584869/0.300545 | -0.4241/-0.3228 |

折名subject_X表示本次适配排除该subject；成绩来自其余两个TRAIN subjects的set002 dev，不是该subject的held成绩。age_updates6/7表示候选状态更新次数，不是人的年龄。

## 全部局部分组，固定50步

| 排除折/条件 | dev subject/age_updates | 样本/正例 | 原权重相对原PΔAUROC | 均衡相对原PΔAUROC | 相对原权重ΔAUROC | 原权重/均衡相对原PΔAP |
| --- | --- | --- | --- | --- | --- | --- |
| subject_02/joint-relation1 | subject_03/6 | 166/36 | 4.2094 | 4.2949 | 0.1068 | 4.1177/4.1611 |
| subject_02/joint-relation1 | subject_03/7 | 138/37 | 4.7364 | 4.7899 | 0.0803 | 4.4338/4.1742 |
| subject_02/joint-relation1 | subject_04/6 | 180/41 | 3.1760 | 3.3164 | 0.0351 | 2.7643/2.7572 |
| subject_02/joint-relation1 | subject_04/7 | 156/41 | 2.3754 | 2.3330 | 0.0848 | 2.0755/2.0334 |
| subject_02/joint-relation10 | subject_03/6 | 166/36 | 4.3162 | 4.3803 | 0.0214 | 3.8094/3.8690 |
| subject_02/joint-relation10 | subject_03/7 | 138/37 | 4.7097 | 4.7632 | 0.0803 | 4.1462/4.1368 |
| subject_02/joint-relation10 | subject_04/6 | 180/41 | 3.0532 | 3.3866 | 0.0175 | 2.7330/2.7358 |
| subject_02/joint-relation10 | subject_04/7 | 156/41 | 2.4814 | 2.4602 | 0.0212 | 2.2351/2.1738 |
| subject_03/joint-relation1 | subject_02/6 | 211/47 | 1.2974 | 1.3492 | 0.1427 | 0.0049/0.1298 |
| subject_03/joint-relation1 | subject_02/7 | 183/50 | -0.5113 | -0.6316 | -0.1203 | -2.0237/-2.1047 |
| subject_03/joint-relation1 | subject_04/6 | 180/41 | 2.5268 | 3.1409 | 0.1930 | 0.8245/1.0293 |
| subject_03/joint-relation1 | subject_04/7 | 156/41 | 2.9268 | 3.2238 | 0.2969 | 2.3102/2.7474 |
| subject_03/joint-relation10 | subject_02/6 | 211/47 | 1.3233 | 1.3752 | 0.1427 | -0.0803/0.0853 |
| subject_03/joint-relation10 | subject_02/7 | 183/50 | -0.6466 | -0.6767 | -0.0301 | -2.0789/-2.1110 |
| subject_03/joint-relation10 | subject_04/6 | 180/41 | 2.4917 | 2.8426 | 0.2281 | 0.7186/0.9868 |
| subject_03/joint-relation10 | subject_04/7 | 156/41 | 2.6299 | 3.0965 | 0.4666 | 2.1231/2.8492 |
| subject_04/joint-relation1 | subject_02/6 | 211/47 | -1.6866 | -2.3223 | -0.5319 | -1.6249/-1.8509 |
| subject_04/joint-relation1 | subject_02/7 | 183/50 | -2.4812 | -3.2932 | -0.8120 | -1.1902/-1.8118 |
| subject_04/joint-relation1 | subject_03/6 | 166/36 | 2.5214 | 2.5641 | 0.0427 | 2.2751/2.4333 |
| subject_04/joint-relation1 | subject_03/7 | 138/37 | 1.7394 | 2.0605 | 0.0268 | 2.8253/2.6113 |
| subject_04/joint-relation10 | subject_02/6 | 211/47 | -1.4920 | -2.0109 | -0.4670 | -1.3917/-1.6919 |
| subject_04/joint-relation10 | subject_02/7 | 183/50 | -2.3759 | -2.9323 | -0.6917 | -1.2007/-1.5793 |
| subject_04/joint-relation10 | subject_03/6 | 166/36 | 2.5427 | 2.6496 | 0.1068 | 2.4014/2.6569 |
| subject_04/joint-relation10 | subject_03/7 | 138/37 | 1.7661 | 1.8464 | -0.0803 | 2.9189/2.7037 |

分组差值为三个固定feature seeds的逐seed paired差值median，不替代主AUROC/AP聚合。完整原始预测和absolute指标保存在原始history.json，summary.json保存重算差值。

## 全部50步检索/动作指标

| 排除折/条件 | 权重 | 检索准确率最差Δ(pp) | 动作F1最差Δ(pp) | margin最差Δ | 缺失指标/失败probe | 资格 |
| --- | --- | --- | --- | --- | --- | --- |
| subject_02/joint-relation1 | age-class | 0.0000 | 0.0000 | -0.000222 | 0/0 | 通过 |
| subject_02/joint-relation1 | subject-age-class | 0.0000 | 0.0000 | -0.000232 | 0/0 | 通过 |
| subject_02/joint-relation10 | age-class | 0.0000 | 0.0000 | -0.000024 | 0/0 | 通过 |
| subject_02/joint-relation10 | subject-age-class | 0.0000 | 0.0000 | -0.000029 | 2/2 | 未通过 |
| subject_03/joint-relation1 | age-class | -1.7857 | 0.0000 | -0.000047 | 0/0 | 通过 |
| subject_03/joint-relation1 | subject-age-class | -1.7857 | 0.0000 | -0.000108 | 0/0 | 通过 |
| subject_03/joint-relation10 | age-class | -1.7857 | 0.0000 | -0.000031 | 0/0 | 通过 |
| subject_03/joint-relation10 | subject-age-class | -1.7857 | 0.0000 | -0.000041 | 0/0 | 通过 |
| subject_04/joint-relation1 | age-class | -1.7857 | -0.7948 | -0.000097 | 0/0 | 通过 |
| subject_04/joint-relation1 | subject-age-class | -1.7857 | -0.7948 | -0.000140 | 0/0 | 通过 |
| subject_04/joint-relation10 | age-class | 0.0000 | -0.7948 | -0.000016 | 0/0 | 通过 |
| subject_04/joint-relation10 | subject-age-class | 0.0000 | -0.7948 | -0.000018 | 0/0 | 通过 |

通过容差不代表无损。全部branch/metric及相对原P的before/after/delta、缺失和失败probe见summary.json；原语义门槛没有改动，任何缺失指标阻止资格通过。

20/50步每个检索/动作/margin/consistency指标相对同预算原权重的逐项差值见summary.json的semantic_vs_original；以下列出50步所有变化或缺失的检索准确率/F1项，未列出的这些指标均精确相等。

| 排除折/条件 | branch | 指标 | 原权重 | 均衡 | 相对原权重Δ(pp) |
| --- | --- | --- | --- | --- | --- |
| subject_02/joint-relation1 | subject_03:action_stream | macro_f1 | 0.629101 | 0.620698 | -0.8403 |
| subject_02/joint-relation10 | subject_03:action_stream | macro_f1 | 0.629101 | 缺失 | 缺失 |
| subject_02/joint-relation10 | subject_04:action_stream | macro_f1 | 0.702151 | 缺失 | 缺失 |
| subject_03/joint-relation1 | subject_02:commit | text_to_event | 0.875000 | 0.892857 | 1.7857 |
| subject_03/joint-relation10 | subject_02:commit | text_to_event | 0.875000 | 0.892857 | 1.7857 |
| subject_04/joint-relation10 | subject_03:action_commit | macro_f1 | 0.526237 | 0.533436 | 0.7198 |

subject02排除折的均衡relation10在50步缺subject03/04 stream动作macro-F1：两个独立动作probe达到400迭代上限，finite但未收敛。训练每步转变头全部收敛，不能称为训练中断或已证实检索退化，也不提高solver预算补过关。

## 完整性和适用范围

- 六个run全部diagnostics_complete；12条适配轨迹各50步，共600实际P更新与每步3头refit。warmup及全部训练头均收敛，frozen-control实际0步，teacher和控制P未变。
- 三折原权重对照20/50步与旧100步轨迹的P逐张量、dev预测/分数及检索指标精确相等；均衡条件独立原P baseline不变。八个fit桶各1/8、每subject1/2，不读held/calibration/validation/test数值。
- 源manifest、Stage2、config、锁定方案及部署代码SHA通过；全部0/5/10/20/50 checkpoint身份、选中SHA、last与终态50步P、AdamW50步读回通过，原生产runner/pilot哈希未变。
- 原始JSON/log归档SHA256：`c54f4f57d36fb7b95a1233c18c58213ce9c094b4e631c1ba6478bc3596ecbb3c`。未下载pt/npz，权重与优化器证据来自服务器独立读回。
- 初始原权重subject总质量已接近均衡：约49.47/50.53%、54.93/45.07%、55.43/44.57%；本轮结果用于检验均衡假设，不据fit目标下降证明泛化机制。
- 本次仅相关TRAIN内部开发折；三折fit/dev重叠，上游已见TRAIN subjects，不是独立未见受试者测试，不是正式Go。没有替换生产模型、解冻Former或自动启动下一轮训练。

## 后续建议

本轮按固定比较结论保留原权重50步研究基线，均衡权重不替换原方案。检索难例保护或直接保护薄弱组的任务目标可作为后续单独实验，但须另定方案，不能把一次负结果解释成必须解冻主干。本次不自动启动下一轮。

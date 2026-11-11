# Stage3B：冻结完整 semantic key 的 50 步 Oracle pilot

**完成50次新 Semantic Difference Head 的 AdamW 更新，原模型0更新。Oracle calibration AUROC 0.722、AP 0.729，未达到预注册两项均≥0.80的门槛，因此回放适配0步，未执行200/500步扩展，也未进入controller calibration。**

结果说明：完整冻结key存在可读出的部分语义关系，50步读出尚不充分；不能据此断言P/Encoder缺信息，更不能自动解冻它们。固定Oracle50在真实Proposal支持区间上的迁移也很弱。

## 实际预算、配置与三层结论

| 项目 | 实际执行 |
| --- | --- |
| Oracle | step0基线 + step50；50次AdamW成功更新 |
| Proposal回放适配 | 0步，calibration gate失败；985个干净train回放样本未用于训练 |
| 扩预算 | 200/500均未执行 |
| 新增Head | 现有SemanticDifference(128,256)，完整key/绝对差/逐维乘积/cosine；50次更新 |
| 训练 | seed42，Head FP32，lr.001/wd.01/clip1，全固定train配对队列；record×age×class等质量权重 |
| 冻结 | Encoder/EventFormer/positive-coordinate gate/P/Evidence/原Local与Radar；无生产安装 |
| 损失 | Oracle平衡BCE；适配方案的replayBCE+.2OracleBCE未执行 |

所有0/50/余弦对照的阈值均在set001 calibration锁定，随后一次性读取set002 dev，不按dev择优或取反。set000训练10路、set001校准8路、set002开发8路，双传感器同recording同分区。上游见过这些TRAIN记录，非独立未见受试者测试。

## 第一层：Oracle semantic discrimination

| 读出 | 分区 | 配对数 | 正类占比 | AUROC | AP |
| --- | --- | --- | --- | --- | --- |
| step0 | train | 21696 | 50.00% | 0.335 | 0.397 |
| step0 | calibration | 16944 | 50.00% | 0.363 | 0.416 |
| step0 | dev | 18168 | 50.00% | 0.328 | 0.394 |
| oracle50 | train | 21696 | 50.00% | 0.764 | 0.768 |
| oracle50 | calibration | 16944 | 50.00% | 0.722 | 0.729 |
| oracle50 | dev | 18168 | 50.00% | 0.768 | 0.779 |
| cosine | train | 21696 | 50.00% | 0.718 | 0.709 |
| cosine | calibration | 16944 | 50.00% | 0.686 | 0.667 |
| cosine | dev | 18168 | 50.00% | 0.726 | 0.718 |

![Oracle0/50固定轨迹](D:/lab/SEER/docs/figures/stage3b-semantic-pilot-2026-10-09.png)

随机step0的固定方向不作强基线；相比冻结cosine，Oracle50的calibration AUROC仅提升0.036，AP提升0.062。Dev的Oracle50成绩不能推翻先前已失败的calibration门槛。

Oracle same定义为同一协议动作类别，不是同事件实例。参考12updates；候选2至7updates，共六个固定成熟度。同一recording里选择不同、不重叠窗口；每个参考在每个成熟度成对提供same/different。不以相同key复制作为same，不用recording/sensor变化作为different。GT只用于离线构造干净窗口，Encoder从不按GT重置；EventFormer窗口初始化属于Oracle构造。

六个成熟度共享参考及部分候选前缀，配对共享窗口/recording；表中配对数不能当成独立试验样本量，未据此计算独立样本显著性。非重叠token窗口也未消除连续MD-GRU更早历史共享。

| k | 候选消费updates | cal AUROC | cal AP | dev AUROC | dev AP |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 0.716 | 0.723 | 0.765 | 0.777 |
| 2 | 3 | 0.719 | 0.727 | 0.767 | 0.778 |
| 3 | 4 | 0.722 | 0.730 | 0.769 | 0.780 |
| 4 | 5 | 0.723 | 0.730 | 0.769 | 0.781 |
| 5 | 6 | 0.725 | 0.731 | 0.769 | 0.780 |
| 6 | 7 | 0.728 | 0.732 | 0.769 | 0.781 |

## 第二层：干净Proposal回放的语义判别

| 读出 | 分区 | same | different | 正类占比 | AUROC | AP |
| --- | --- | --- | --- | --- | --- | --- |
| step0 | train | 611 | 374 | 37.97% | 0.424 | 0.365 |
| step0 | calibration | 404 | 288 | 41.62% | 0.488 | 0.449 |
| step0 | dev | 755 | 233 | 23.58% | 0.348 | 0.189 |
| oracle50 | train | 611 | 374 | 37.97% | 0.480 | 0.394 |
| oracle50 | calibration | 404 | 288 | 41.62% | 0.491 | 0.407 |
| oracle50 | dev | 755 | 233 | 23.58% | 0.455 | 0.255 |
| cosine | train | 611 | 374 | 37.97% | 0.535 | 0.430 |
| cosine | calibration | 404 | 288 | 41.62% | 0.477 | 0.402 |
| cosine | dev | 755 | 233 | 23.58% | 0.595 | 0.326 |

这一层只评价干净语义标签，未训练回放适配。每个支持区间所有帧必须同一非负动作类别，并离协议GT转换>.36s；参考/候选混合或近边界均mask。未使用matched/unmatched、GT±.72命中作为语义训练标签。各组same/different分数均值、中位数及action-pair支持数保存在原始results.json；单类action-pair的AUROC无定义，不把该组AP当有效判别证据。

| 分区 | 全候选 | 因果上下文完整 | same | different | mask | 干净占比 |
| --- | --- | --- | --- | --- | --- | --- |
| train | 3943 | 3929 | 611 | 374 | 2958 | 24.98% |
| calibration | 3224 | 3209 | 404 | 288 | 2532 | 21.46% |
| dev | 3191 | 3179 | 755 | 233 | 2203 | 30.96% |

**参考约定变化必须保留：** 对真正Proposal发出index j，参考只消费tokens[j-18:j-6]，于j-7结束并冻结；Candidate消费j..j+6，1.08s后判别。窗口按固定时间规则构造，对所有因果有效候选都计算；GT不参与构造、reset或推理筛选。它解决了旧连续Active混合多动作后无法直接贴当前GT标签的问题，但并非正式Active/Confirm闭环，也不是与上一轮8标量Probe的单变量比较。

## 第三层：事件级诊断过滤

原候选池固定Radar.38/独立GRU.44，peak lookahead1=.18s/NMS.36s/neighbor-high/因果跨源合并不变。原池644/704=91.48%，FP 31.863/min，总候选39.920/min。

| 读出 | cal策略 | 阈值 | dev Recall | FP/min | Density/min | Recall损失pp | FP减少 | 决策P95 s | 过滤门槛 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| step0 | retain_recall | 0.484 | 90.62% | 31.438 | 39.420 | 0.852 | 1.33% | 1.530 | 未通过 |
| step0 | halve_fp | 0.487 | 45.88% | 15.062 | 19.103 | 45.597 | 52.73% | 1.530 | 未通过 |
| step0 | fp8 | 0.488 | 10.51% | 3.178 | 4.103 | 80.966 | 90.03% | 1.562 | 未通过 |
| oracle50 | retain_recall | 0.171 | 84.52% | 28.736 | 36.179 | 6.960 | 9.82% | 1.530 | 未通过 |
| oracle50 | halve_fp | 0.325 | 43.18% | 14.587 | 18.390 | 48.295 | 54.22% | 1.620 | 未通过 |
| oracle50 | fp8 | 0.394 | 21.59% | 5.992 | 7.894 | 69.886 | 81.19% | 1.710 | 未通过 |
| cosine | retain_recall | 0.002 | 86.65% | 29.499 | 37.130 | 4.830 | 7.42% | 1.530 | 未通过 |
| cosine | halve_fp | 0.105 | 43.32% | 14.024 | 17.839 | 48.153 | 55.99% | 1.620 | 未通过 |
| cosine | fp8 | 0.322 | 19.32% | 6.042 | 7.744 | 72.159 | 81.04% | 1.710 | 未通过 |

全部9组已锁定过滤策略均未达到“FP减少≥50%且Recall损失≤5pp”；8FP/min及同一召回约束也均失败。事件级对全池预测并重新一对一匹配，离线语义mask不用于在线过滤；上下文缺失保守拒绝，GT分母保留。Proposal时间不后移，决策延迟单独增加1.08s。p_diff阈值过滤仅是诊断，未安装Verify Head、未执行Confirm。

## 八个开发cell全部披露

| cell | Oracle数 | Oracle AUROC | Oracle AP | 回放干净数 | 回放AUROC | 回放AP | 事件Recall | 事件FP/min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| subject_02/TI_xWR14xx | 2172 | 0.817 | 0.823 | 104 | 0.449 | 0.389 | 91.84% | 30.749 |
| subject_02/TI_xWR68xx | 2172 | 0.743 | 0.766 | 104 | 0.432 | 0.301 | 88.78% | 26.142 |
| subject_03/TI_xWR14xx | 2304 | 0.755 | 0.749 | 136 | 0.575 | 0.373 | 81.93% | 28.503 |
| subject_03/TI_xWR68xx | 2304 | 0.753 | 0.762 | 123 | 0.529 | 0.376 | 80.72% | 25.603 |
| subject_04/TI_xWR14xx | 2268 | 0.756 | 0.765 | 125 | 0.362 | 0.183 | 82.35% | 29.403 |
| subject_04/TI_xWR68xx | 2268 | 0.747 | 0.764 | 102 | 0.329 | 0.204 | 74.12% | 24.095 |
| subject_08/TI_xWR14xx | 2340 | 0.794 | 0.819 | 164 | 0.507 | 0.296 | 88.37% | 34.956 |
| subject_08/TI_xWR68xx | 2340 | 0.787 | 0.803 | 130 | 0.521 | 0.283 | 86.05% | 30.449 |

## 核验结果与未通过项目

首次v1在记录已有partition字段的组装处停止，0 AdamW更新。修复函数显式校验旧分区而非静默覆盖；v1完整日志及SHA归档保留。v2实际训练50步，两个0/50检查点与50条逐步日志、全部参数AdamW counter已独立核对。16项相关pytest通过；Windows pyreadline析构时有非测试主体警告，退出码0。未宣称全仓库pytest通过。

独立补充核验：56808个Oracle配对、10358个回放标签/上下文、24024次原matcher校准record检查；权重读回预测最大误差1.7881393432617188e-07。冻结frontend前后digest一致。

**原自动核验并非全绿：** 52路混合batch的native key重放最大差异0.0001888088881969452，超过原1e-5检查。额外按原batch形状抽查重放21632个key，最大误差0；该数量是选定批次的抽查量，不是全部key库。已定位为批次形状相关数值差异，前端EventFormer混合FP16/FP32，P及新Head FP32。保留原失败日志与归档，补充核验显式保留cross_batch_1e_minus5_passed=false；没有放宽原阈值、没有重算替换训练输入或重新训练。该差异不能据此被宣称满足正式v2.5 FP32 replay资格。

## 判断与下一步边界

本轮No-Go：Oracle主门未过，实际回放语义迁移与事件过滤也未达标。冻结key并非完全不可读，但50步不足以证明Head已经读充分；train Oracle AUROC约.764，也未到.80，不能把calibration失败直接归因于P信息不足。Oracle比回放好也提示配对任务、相邻动作难度与候选初始化/参考约定的域差需要区分，不能直接解冻Encoder。

后续若继续，应单独批准同一Oracle任务的预注册200步趋势检查或有明确假设的诊断；本轮不自动执行、不按已看过dev选择预算。P、EventFormer、Encoder的任何适配另立受约束实验。所有物理语义判断仍使用协议GT，physical onset独立标注为0；本轮不是正式controller Go。

## 证据与SHA

- [results.json](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/results.json)
- [oracle-gate.json](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/oracle-gate.json)
- [zero-step-baseline.json](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/zero-step-baseline.json)
- [cohort-accounting.json](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/cohort-accounting.json)
- [predictions.npz](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/predictions.npz)
- [oracle-pairs.json](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/oracle-pairs.json)
- [oracle-bank.json](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/oracle-bank.json)
- [oracle-pair-predictions.csv](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/oracle-pair-predictions.csv)
- [proposal-semantic-predictions.csv](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/proposal-semantic-predictions.csv)
- [optimizer-checks.json](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/optimizer-checks.json)
- [oracle-step0.pt](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/heads/oracle-step0.pt)
- [oracle-step50.pt](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/heads/oracle-step50.pt)
- [oracle.jsonl](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/heads/oracle.jsonl)
- [download-checks.json](D:/lab/SEER/.audit-staging/stage3b-semantic-readback-v2/download-checks.json)
- [independent-readback.json](D:/lab/SEER/.audit-staging/stage3b-semantic-verification-readback-v1/verification/independent-readback.json)
- [batch-key-audit.json](D:/lab/SEER/.audit-staging/stage3b-semantic-verification-readback-v1/batch-key-audit.json)
- [2026-10-09-stage3b-semantic-pilot.md](D:/lab/SEER/docs/superpowers/plans/2026-10-09-stage3b-semantic-pilot.md)

v2原始归档SHA：`2fd55613f3b22c3f726d0b4e3021f3bf776a55b017f72feac575ab185bbea451`，41398849bytes，172成员逐路径/哈希验证后解压。

v1零更新失败归档SHA：`7eb5ef4365ba425a1ce6b341525c1788391d762b9bd97def35bf7325335712d4`。

原冻结frontend SHA：`57af183acfdcc1da33c76573b60b57ae60897fd9aca715484058b37340378602`；digest：`0699d993826ccb052a89d801f4c052e6a75ffe7052b152f7c47918da108d6cd0`。完整来源/方案/源码SHA见spec/source-manifest和补充核验归档，密码未写入任何文件。

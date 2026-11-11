# 新数据折复核结果

新折复核已经正常完成。subject_03折的主AUROC从 **0.562906 提升到0.567681**，增加 **0.48个百分点**；两个关系约束条件均有提升，且全部检查点通过原语义保留门槛。第一折约2.48个百分点的提升没有在新折同幅度重现，说明收益方向可以重复，但幅度依赖数据划分，目前不能认定为稳定的大幅改进。

原训练于10月3日直接读回核验；**10月4日原始JSON/log已完整下载并核验归档SHA**，本地脚本已重算成绩、全部选中语义差值和头收敛记录。原始归档SHA256为`0841c2020669273afbf720a661c610c95538fd1f11f41934e6d10ba0a478ddb5`，本地目录`.audit-staging/results-projection-replication-v1/`。前一日下载失败的限制已解除。

## 主任务与固定检查点

| 条件 | 本折原模型AUROC | 本折选中AUROC | 提升 | 选中步数 | 原语义门槛 |
| --- | --- | --- | --- | --- | --- |
| 冻结对照 | 0.5629059618 | 0.5629059618 | 0 | 0 | 通过 |
| relation1 | 0.5629059618 | 0.5671820566 | +0.4276个百分点 | 20 | 通过 |
| relation10 | 0.5629059618 | 0.5676813882 | +0.4775个百分点 | 20 | 通过 |

比较均使用本折同一独立分年龄probe及开发集聚合，没有把第一折的原模型0.6258与新折0.5677直接当作适配退化。两个折的开发受试者构成不同，必须分别比较各自原模型。

| 固定步数 | relation1 AUROC | relation10 AUROC | 两条件语义评估 |
| --- | --- | --- | --- |
| 0 | 0.5629059618 | 0.5629059618 | 完整、通过 |
| 5 | 0.5632879285 | 0.5627835934 | 完整、通过 |
| 10 | 0.5648008748 | 0.5647605739 | 完整、通过 |
| 20 | 0.5671820566 | 0.5676813882 | 完整、通过 |

relation10第5步略低于基线；第10/20步高于基线。两条件都按已固定规则选择20步，未追加步数。relation10仅比relation1高0.0004993，不据此认定更大的约束系数稳定更好。

## 与第一折的配对对照

| 方法 | 第一折held02增益 | 新折held03增益 |
| --- | --- | --- |
| relation1 | +2.3961个百分点 | +0.4276个百分点 |
| relation10 | +2.4831个百分点 | +0.4775个百分点 |

这支持继续评估组合修正“主年龄/类别目标对齐、每步拟合分类头、teacher关系保护”。两个折的fit/dev受试者有重叠，Stage2上游训练过TRAIN subjects，且流程是确定性的；不能把两个折或更换未使用的训练seed称为独立随机重复、显著性检验或未见受试者测试，也不能把全部收益单独归因于某一个修正。

## 完成和完整性核验

流程2026-10-03 **22:31:57–22:38:58(+0800)**，总计421秒。缓存阶段完成于22:35:49，Projection训练及评估用时189秒。fit/dev语义缓存分别446/216窗口，按held_subject_03的fit/dev角色检查通过，14类、每recording/class上限4；缓存artifact哈希核验通过。

服务器独立读取了common-warmup、最后检查点、选中检查点及原selection lock：共同warmup的Projection与teacher逐张量相等；所有最终teacher与共同teacher逐张量相等；冻结对照保留原Projection；两个联合条件实际AdamW步数均为20，所有最终状态为ready；40次每步分类头重新拟合全部收敛，每次3个头。选中权重SHA256与原锁定记录一致，原语义保留函数独立重算均通过。没有挂起的拟合或评估阶段。

源码、预注册方案、源manifest和Stage2 checkpoint的哈希核验通过。原生产runner/pilot哈希未改变，模型未替换。manifest确认没有读取本折held/calibration数值、validation/test，未启动全量pilot。读取和核验过程中未进行任何新训练或改变原记录。

服务器验证记录：

`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/transition-aware-semantic-primary-replication-v1/result-readback-checks-v1.json`

服务器结果目录：

`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/transition-aware-semantic-primary-replication-v1/subject_03-seed-3407/`

选中权重分别为 `experiment/joint-relation1/step-20.pt`、`experiment/joint-relation10/step-20.pt`。此前控制台转录`.audit-staging/projection-replication-live-readback-v1.json`仍保留作历史；最新结论以哈希核验后的下载原始JSON/log及本地重算`.audit-staging/results-projection-replication-v1/summary.json`为依据。归档不包含pt/npz；权重本身在服务器逐张量核验，并没有声称权重已下载。

## 10月4日补齐的语义差值和分组分析

| 条件 | 双向及交叉版本检索准确率最差delta | 动作macro-F1最差delta | margin最差delta | consistency loss最大增加 |
| --- | --- | --- | --- | --- |
| relation1 | 0 | 0 | -0.0000483394 | +0.0001558885 |
| relation10 | 0 | 0 | -0.0000294447 | +0.0001012981 |

这里的“准确率未下降”指选中20步检查点的已测量指标；margin和consistency仍有小幅变化，不能表述为所有语义指标完全不变。

relation10固定20步的分组paired差值（三个固定feature seeds取median，属于分组诊断，不替换原主AUROC聚合）：

| dev subject | age | 样本/正例 | AUROC delta百分点 | AP delta百分点 |
| --- | --- | --- | --- | --- |
| subject02 | 6 | 211/47 | +0.6227 | -0.0664 |
| subject02 | 7 | 183/50 | -1.4737 | -1.0432 |
| subject04 | 6 | 180/41 | +1.4038 | +0.3581 |
| subject04 | 7 | 156/41 | +1.3574 | +1.2104 |

整体+0.48个百分点掩盖subject02/age7的局部退化。尚不能单独归因于预算或泛化；新增预算实验应同时观察每折/每subject/每age的AUROC与AP，不能只汇报macro最高值。

## 下一步建议

原始下载和具体检索差值已补齐。用户已授权以相同20步预算补齐held04，随后在新目录做三折20/50/100步固定预算诊断，方案见`docs/stage3-transition-aware-budget-ablation-2026-10-04.md`。这不会改写本次20步实验或启动正式全量长训；新实验实际状态见对应执行记录。

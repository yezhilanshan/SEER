# RD-only ResNet18从头训练结果核验

20个epoch已完成，训练及保存链路正常。验证集选择第17个epoch。测试受试者06表现较好，但新增留出17/18明显较弱，当前主要问题是跨受试者泛化。不能将06的改善单独解释为ResNet结构收益，也尚未训练或验证新的自然语言生成路径。

## 完成状态与选优

- 北京时间2026-10-11 21:04:46启动，23:09:57完成，耗时2小时5分10.54秒。
- 20epoch，24,855次成功优化更新。5次AMP溢出均跳过并回退缩放，未中断训练；次数符合20×1,243−5。
- 11,898,062个可训练参数；启动证据已确认80个参数张量都更新，空间骨干和BN并非冻结。
- 第17个epoch按validation三个受试者的Macro-F1平均选中，平局依据合并帧Macro-F1与更低CE。第20轮不是最终选中的模型。
- 最佳权重SHA256：`d2c01cccb6646b827905e26a572df46893f73522639b644a7852aec89a05af21`。
- 2026-10-11 23:08:23先写入选优锁定记录，再开始最终validation/test评价；没有按test改选权重，本次检查没有重启训练。

## 逐帧识别结果

下列指标保留描述协议中的全部有效帧，包括标签过渡区。窗口12秒、步长6秒、最短8秒、最多3动作；使用全部1,850个validation窗口和1,286个test窗口，含不满12秒的合格尾窗。重叠帧重复计数，因此是窗口内逐帧观测，不是全recording去重帧指标。每个Macro-F1均固定对14类求平均。

| 集合/受试者 | 帧观测 | Frame Accuracy | Macro-Precision | Macro-Recall | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation合并帧 | 246,507 | 60.93% | 62.55% | 60.93% | 61.35% |
| test合并帧，06/17/18 | 171,329 | 51.55% | 55.67% | 53.47% | 52.39% |
| 新增留出合并帧，17/18 | 106,875 | 37.64% | 42.11% | 39.91% | 37.56% |

模型选择和跨受试者汇总使用**先计算每人的14类Macro-F1，再对人平均**：validation为57.33%，test三人为50.16%，新增17/18为36.75%。它们与合并全部帧计算的61.35%/52.39%/37.56%不同，应分别标注。所有validation/test受试者均有14类标签支持。

| 用途 | 受试者 | Frame Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| validation | 05 | 74.55% | 74.08% |
| validation | 12 | 61.63% | 62.09% |
| validation | 16 | 35.72% | 35.83% |
| 历史已观察test | 06 | 74.60% | 76.98% |
| 新增模型留出test | 17 | 29.82% | 29.14% |
| 新增模型留出test | 18 | 45.36% | 44.35% |

06此前已经多次用于历史实验；17/18是本轮扩充协议的新增模型留出，结果必须单列。validation中的16也明显偏低，说明差异在验证阶段已可见，不能只把问题归结于测试偶然波动。现有源数据未确认环境类别，不能凭编号认定为Homelab/Hospital域差异。

## 收敛与错误模式

![训练损失与仅验证集曲线](D:/lab/SEER/outputs/rd-resnet18-scratch-results-20261013-v1/derived/training-curves.png)

训练加权CE由第1轮1.7032降至第20轮0.0790。验证受试者平均Macro-F1在第8轮已到55.94%，第17轮57.33%，第20轮56.70%；后段训练损失下降没有转化为持续验证提升，提示过拟合或泛化瓶颈。训练CE使用训练频率权重，validation CE未加权，两者数值不作直接差值比较。

新增17/18的主要可见混淆：进入床→躺床，占真实进入床帧57.64%；坐到床上→躺床，占49.72%；走向房间→走向椅子，占30.54%；坐到椅子→从椅子站起，占25.94%。这些比例均为完整14类混淆矩阵的真实类行内比例。

![全14类测试与新增留出混淆矩阵](D:/lab/SEER/outputs/rd-resnet18-scratch-results-20261013-v1/derived/confusion-matrices.png)

| 新增17/18中的动作 | Recall | F1 |
| --- | ---: | ---: |
| gets into bed | 20.94% | 29.69% |
| gets out of bed | 42.97% | 48.70% |
| lies in bed | 73.89% | 23.59% |
| stands up from the bed | 46.66% | 44.80% |
| stands up from the chair | 44.13% | 41.85% |
| stands up from the floor | 27.62% | 36.37% |
| sits down on the bed | 11.62% | 16.35% |

躺床Recall高而F1低，Precision仅14.03%，与大量其他动作被误判躺床相符。坐床和进入床漏识别突出，下一步应优先检查这些动作及16/17/18的输入分布、标注对齐与BN统计敏感性。具体成因尚不能仅凭分数确定。

## 与旧实验如何比较

旧冻结随机ResNet18在同一06滑窗的逐帧Macro-F1为62.25%，本轮06为76.98%，数值增加14.72个百分点。但训练人数由3增至9、空间骨干由冻结变为可训练、读出损失和优化配置也改变，验证选优受试者与规则同样扩展，因此不是隔离ResNet端到端训练收益的受控比较。新增17/18更低的结果也说明，06提升不足以代表整体泛化提升。

研究上应保留这一检查点作为基线；优先完成同一9/3/3划分下的小CNN对照和低分受试者诊断，再决定语言对齐投入。本轮没有追加训练、调整测试阈值或计算新的文本序列准确率；逐帧结果不能换算成自然语言描述能力。

## 独立验证与交付

服务器检查全部16个训练产物文件SHA及148个执行源码SHA；最佳权重与锁定记录一致。对全部3,196个validation/test窗口，存储标签与真实canonical标签逐值一致；抽取涵盖05/12/16/06/17/18的6个批次、12个真实窗口重放最佳模型，保存logits最大绝对差为0。

本地独立从417,836帧观测的logits重算混淆矩阵、Accuracy、Macro-P/R/F1、逐类指标和受试者平均，全部与结果一致；另重算全部20轮验证混淆指标与选优排序，确认第17轮；核对窗口ID、帧区间、动作序列、AMP跳步和测试前锁定时序。最佳checkpoint以`weights_only=True`安全加载，元数据与实际划分一致。图表已生成并检查可读性。

服务器结果及权重目录：

```text
/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/rd_resnet18_scratch_expanded_20261011_v1
```

本地检查产物：`outputs/rd-resnet18-scratch-results-20261013-v1/`。

- [最佳权重best.pt](../outputs/rd-resnet18-scratch-results-20261013-v1/run/best.pt)
- [原始结果](../outputs/rd-resnet18-scratch-results-20261013-v1/run/results.json)、[20轮曲线](../outputs/rd-resnet18-scratch-results-20261013-v1/run/history.json)、[测试logits](../outputs/rd-resnet18-scratch-results-20261013-v1/run/logits-test.npz)
- [逐类指标CSV](../outputs/rd-resnet18-scratch-results-20261013-v1/derived/per-class.csv)、[曲线CSV](../outputs/rd-resnet18-scratch-results-20261013-v1/derived/training-curve.csv)
- [本地独立审计](../outputs/rd-resnet18-scratch-results-20261013-v1/derived/independent-audit.json)、[服务器真实数据重放审计](../outputs/rd-resnet18-scratch-results-20261013-v1/remote-inspection.json)

原始导出在`run/`，衍生指标与图在`derived/`，执行源码在`source/`；大型initial/last/recovery/first-update权重仍留服务器，未修改原训练结果。图片同时保留SVG版本供导出。

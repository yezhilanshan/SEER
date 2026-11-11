# Radar-to-Text 系统性错误归因报告（2026-10-11）

**结论：当前是稳态类别感知/分类读出不足、文本动作覆盖与窗口截断问题并存；不支持“BiGRU 破坏特征”或“Align 导致整体特征退化”。模型利用了雷达内容，但可靠的顺序理解仍未得到证明。**

全部六阶段已完成，复用已有 Stage1/Align 权重，Qwen 主体始终冻结。新增训练仅四个线性头和 8 个固定前缀向量，没有更新原始编码器或 LLM 权重。

## 评价口径与数据限制

- train=subject02/03/04、validation=05、test=06；探针/前缀训练与标准化仅用 train，平滑阈值与权重选优仅用 validation。subject06 已在首轮看过，本次 test 只能作为固定诊断的描述性结果，不是新的盲测。
- 逐帧评价覆盖完整记录的所有有效标签，每帧一次；按不重叠 12 秒块推理，保留末尾短块。BiGRU 不跨块，不能称为全记录上下文评价。
- 稳态诊断从完整记录的标签转换中点前后各排除 0.3 秒，包含到未知标签的转换；-100 不计分。它不替代全部有效帧结果。
- canonical 标签策略为 `command_active_until_next_command`。标注区间长度不等于实际动作执行耗时，边界也是指令标签边界。environment_id 统一为 `parrad`，不足以区分 Homelab/Hospital。
- 双雷达视图、重叠描述窗口、反序配对之间有关联；没有把这些条目当作独立实验，也未作显著性结论。

## 1. Stage1 逐帧分类

| split | 帧集合 | 帧数 | Accuracy | Macro-P | Macro-R | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | all_valid | 160328 | 38.10% | 39.77% | 35.83% | 34.82% |
| train | steady | 147083 | 39.27% | 41.06% | 36.98% | 35.92% |
| validation | all_valid | 53454 | 34.31% | 39.02% | 31.67% | 29.50% |
| validation | steady | 49030 | 35.19% | 40.24% | 32.46% | 30.26% |
| test | all_valid | 40088 | 38.81% | 39.58% | 35.79% | 34.62% |
| test | steady | 36393 | 40.08% | 40.93% | 36.98% | 35.69% |

排除过渡帧后提升有限：validation Macro-F1 只提高约 0.77 个百分点，test 约 1.08 个百分点。稳态同样存在明显错误，不能只修标签边界。宏平均始终包含全部 14 类，零分母置零。

| class | 动作 | val F1 | val steady F1 | test F1 | test steady F1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | falls onto the floor of the room | 46.61% | 48.04% | 48.01% | 49.82% |
| 1 | gets into bed | 8.93% | 9.19% | 8.85% | 8.61% |
| 2 | gets out of bed | 23.18% | 23.89% | 30.80% | 31.97% |
| 3 | lies in bed | 33.07% | 33.71% | 31.88% | 32.88% |
| 4 | rolls in bed | 40.21% | 41.16% | 31.72% | 32.25% |
| 5 | sits down on the bed | 13.48% | 12.87% | 16.99% | 16.85% |
| 6 | sits down on the chair | 34.77% | 35.52% | 38.57% | 40.61% |
| 7 | sits in bed | 33.02% | 33.67% | 36.69% | 37.42% |
| 8 | stands up from the bed | 9.22% | 9.98% | 24.51% | 25.96% |
| 9 | stands up from the chair | 23.13% | 24.75% | 39.08% | 40.24% |
| 10 | stands up from the floor of the room | 48.07% | 49.75% | 55.74% | 57.73% |
| 11 | walks toward the bed | 32.92% | 33.71% | 31.16% | 32.53% |
| 12 | walks toward the chair | 24.74% | 25.16% | 44.86% | 46.29% |
| 13 | walks toward the room | 41.60% | 42.28% | 45.78% | 46.57% |

六个重点类别的主要混淆（validation 全有效帧，比例以该 GT 类总帧数为分母）：

| GT 类别 | 最大非对角混淆 | 占该 GT 类 |
| --- | --- | ---: |
| gets into bed | sits down on the chair | 36.33% |
| gets out of bed | stands up from the floor of the room | 33.25% |
| lies in bed | falls onto the floor of the room | 30.35% |
| stands up from the bed | stands up from the floor of the room | 45.74% |
| stands up from the chair | stands up from the floor of the room | 58.69% |
| stands up from the floor of the room | falls onto the floor of the room | 18.51% |

![Validation 14-class confusion](../outputs/radar-text-diagnostics-20261011-v1/diagnostics/figures/validation-all_valid-row-normalized.png)

[全部逐帧指标与混淆矩阵 JSON](../outputs/radar-text-diagnostics-20261011-v1/diagnostics/frame-classification.json) · [逐类 CSV](../outputs/radar-text-diagnostics-20261011-v1/diagnostics/per-class-frame-metrics.csv) · [全部图表](../outputs/radar-text-diagnostics-20261011-v1/diagnostics/figures/linear-probes.png)

完整 14×14 counts 与行归一化矩阵分别输出 train/validation/test 的 all_valid/steady，共 12 张混淆图，各有 PNG/SVG。逐帧 NPZ 含 labels、timestamps、logits、predictions、steady、short 和 recording offsets；recording 元信息在 feature-index.json。

## 2. 冻结特征线性探针

四路特征均为 256D：stage1/align 各自 CNN 融合与 BiGRU 输出。统一 train-only 标准化、线性256→14、按训练类频率加权 CE、AdamW lr0.01、batch4096、固定20epoch、seed17，以 validation Macro-F1 选 best。使用全部有效训练帧与相同顺序/预算，不更新编码器。床相关类固定为1/2/3/4/5/7/8。

| 特征 | val Macro-F1 | val 床类 Macro-F1 | val 短动作 Recall | test Macro-F1 | test 床类 Macro-F1 | test 短动作 Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stage1_cnn | 18.80% | 15.67% | 22.58% | 17.96% | 15.32% | 20.56% |
| stage1_gru | 47.47% | 43.71% | 19.35% | 49.15% | 46.48% | 16.67% |
| align_cnn | 19.98% | 16.61% | 17.74% | 18.94% | 16.45% | 20.56% |
| align_gru | 50.55% | 48.79% | 12.90% | 51.44% | 47.02% | 19.44% |

短动作定义为完整标注 run 小于2秒，Recall 是该子集中有 GT 支持类别的帧 Recall 宏平均；仅40个 validation 帧、50个 test 帧，不能据此给出稳定短动作排名。与后文窗口可见时长的分桶不同。

CNN 融合特征的线性可读性弱，BiGRU 明显提高可读性；不能推断原始 RD/MD 完全没有区分信息。Align 后 BiGRU 没有出现整体退化。原 Stage1 分类头 validation F1 29.50%，同一冻结特征新探针47.47%，提示原分类读出/训练配置也有不足，而非只能归咎于 backbone。探针更长训练、类平衡和标准化共同改变了读出，不能单独归因某一个因素。

探针约50%的 Macro-F1 仍有较大感知错误空间；其指标与窗口生成 Recall 不同，不能直接相减以量化接口损失，也不能证明 LLM 具有同等读出能力。BiGRU 的收益可能同时包含运动上下文与动作流程先验。

[完整探针结果、训练曲线与逐类指标](../outputs/radar-text-diagnostics-20261011-v1/diagnostics/linear-probes.json)

## 3. 三组对照

### A. 平滑分类模板

按真实时间戳构造0/0.3/0.5秒邻域中值滤波 × 0/0.3/0.6秒最短预测段，删短段后合并相邻同类。整数 class-ID 中值依赖类别编号顺序，这是该候选方法的局限。未按 GT 动作个数截断。选优依次比较 validation exact、LCS、Recall、最简单阈值。

| 中值秒 | 最短段秒 | val 序列正确率 | val LCS | val Recall |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.0 | 13.68% | 36.94% | 50.07% |
| 0.0 | 0.3 | 13.39% | 37.38% | 47.17% |
| 0.0 | 0.6 | 13.53% | 37.62% | 44.80% |
| 0.3 | 0.0 | 13.68% | 37.24% | 49.76% |
| 0.3 | 0.3 | 13.25% | 37.33% | 47.17% |
| 0.3 | 0.6 | 13.53% | 37.61% | 44.80% |
| 0.5 | 0.0 | 13.39% | 37.41% | 49.36% |
| 0.5 | 0.3 | 13.25% | 37.35% | 47.27% |
| 0.5 | 0.6 | 13.53% | 37.60% | 44.80% |

最终选0.3秒中值、最短段0秒。test 完整序列仍54/484=11.16%，不能把首轮差距完全解释为分类抖动。

### B. 无雷达语言先验

训练8×896=7168个固定前缀参数，冻结同一个 Qwen，使用相同2187条训练描述、相同chat prompt/answer-only CE/greedy解码；3epoch、batch2、accum4、lr1e-3、seed17，按 validation loss 选 best。前缀对每个输入相同，因此greedy输出恒定。没有读取真实雷达进行前向。

固定前缀 best epoch=3，生成：`The person rolls in bed, then gets out of bed.`。另补算训练集众数静态对照（仅train选句，186/2187；此项为事后sanity check，不据test挑句）。

| 方法 | val 完整序列 | test 完整序列 | test Precision | test Recall | test LCS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 未平滑分类模板 | 13.68% | 11.16% | 38.47% | 55.51% | 36.67% |
| 选优平滑分类模板 | 13.68% | 11.16% | 39.02% | 54.89% | 37.14% |
| 固定前缀无雷达 | 1.99% | 1.24% | 15.08% | 11.09% | 11.09% |
| train众数静态句 | 7.41% | 9.09% | 27.00% | 28.17% | 27.00% |
| 真实雷达→Qwen | 20.09% | 25.00% | 50.72% | 46.28% | 45.59% |

真实雷达模型优于这两种无雷达对照，支持输入内容带来增益。但固定前缀不是所有语言先验方法的上界，训练集众数甚至更强；不能仅凭弱固定前缀基线宣称已排除动作流程先验。

### C. 样本错配、同步时间打乱与真实异序配对

validation/test 各均匀选择64个窗口。错配窗口要求同传感器、时长差不超过一个真实帧周期、动作序列不同；同记录内相互重叠的窗口不配。错配 loss 保持原目标文本，错配生成同时对原目标与其自身参考分别评价。RD/MD 时间打乱使用同一 permutation、固定seed，无优化器更新。

| split | 配对数 | 正确雷达 loss | 错配 loss | 匹配更好 | 原始序列正确 | 打乱序列正确 | 打乱 loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 64 | 0.2412 | 0.6191 | 95.31% | 15.62% | 1.56% | 0.5389 |
| test | 64 | 0.1739 | 0.5422 | 95.31% | 23.44% | 1.56% | 0.4766 |

两组均61/64配对在匹配时loss更好，支持利用雷达内容。同步打乱导致分布外输入，也破坏MD局部时间邻域，下降只能说明时间结构敏感性，不能单独证明动作顺序理解。

| split | 真实异序候选组合 | 评估组合 | 涉及不同窗口 | 两端动作多重集均对 | 两端完整顺序均对 |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 52 | 32 | 23 | 0 | 0 |
| test | 8 | 8 | 5 | 0 | 0 |

真实配对要求同动作多重集、不同顺序、同传感器、非重叠，并先去重同步视图。validation32组合中31组为三动作、1组双动作；test8组合全部三动作，仅涉及5个不同窗口。测试没有足够独立双动作反序样本，不能编造或据此宣称纯顺序能力的稳定结论。两端类别多重集同时正确也为0，顺序错误尚未从识别/漏动作错误中隔离。

去重核验发现一个同步视图末帧差异使原始validation候选窗口键多计1（352→351）。已新增红→绿回归并修正本地代码；重新计算确认52/8候选配对及全部实际评估配对不变，原始预测和服务器源码包保留。校正记录在 real-order-dedup-audit.json。

[错配与打乱逐样本输出、真实异序对](../outputs/radar-text-diagnostics-20261011-v1/diagnostics/controlled-perturbations.json)

## 4. 文本动作未覆盖的结构化分解

原先148/484只统计预测动作个数偏少，并不覆盖所有类别未匹配错误。现在test1358个GT类别实例中736个未被生成多重集覆盖（54.20%）；validation1875中1020个（54.40%）。这是文本层覆盖缺失，可能同时包含识别成别类，不等于736个已证实的物理漏检。

逐实例保留class_id、duration（可见）、full_duration、relative_start/end、at_window_edge、truncation（含both）、subject、sensor、environment、window动作数。重复类别无法确定具体哪次未覆盖，按missed_weight均摊；不使用任意先覆盖的布尔值定位漏检。

### 窗口内可见时长

| 桶 | val实例 | val未覆盖率 | test实例 | test未覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| 1-2 | 192 | 63.80% | 136 | 61.40% |
| 2-4 | 482 | 53.42% | 332 | 53.01% |
| <1 | 127 | 74.02% | 102 | 75.49% |
| >=4 | 1074 | 50.84% | 788 | 50.70% |

### 窗口位置（实例中点划三等分）

| 桶 | val实例 | val未覆盖率 | test实例 | test未覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| begin | 706 | 50.07% | 510 | 48.53% |
| end | 701 | 65.12% | 508 | 65.45% |
| middle | 468 | 44.87% | 340 | 45.88% |

### 截断类型

| 桶 | val实例 | val未覆盖率 | test实例 | test未覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| both | 2 | 100.00% | 0 | NA |
| complete | 503 | 46.12% | 414 | 47.83% |
| left | 682 | 49.93% | 470 | 49.26% |
| right | 688 | 64.75% | 474 | 64.66% |

### 窗口动作数量

| 桶 | val实例 | val未覆盖率 | test实例 | test未覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4 | 100.00% | 2 | 0.00% |
| 2 | 446 | 54.48% | 180 | 48.89% |
| 3 | 1425 | 54.25% | 1176 | 55.10% |

### 按完整标注区间长度复核

| 完整run长度 | val实例 | val未覆盖率 | test实例 | test未覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| <1 | 1 | 100.00% | 0 | NA |
| 1-2 | 2 | 50.00% | 2 | 0.00% |
| 2-4 | 52 | 70.19% | 48 | 60.42% |
| >=4 | 1820 | 53.93% | 1308 | 54.05% |

test可见不足1秒的102个实例全部被截断，不能当作102个真正短动作。完整run不足2秒只有2个窗口实例，短动作能力证据不足。test末端未覆盖率65.45%、右截断64.66%，均高于中间45.88%和完整实例47.83%；这是相关性，类别、窗口位置和可见长度相互混杂，不作因果断言。

### 按14类统计

| 类 | val实例 | val未覆盖率 | test实例 | test未覆盖率 |
| --- | ---: | ---: | ---: | ---: |
| falls onto the floor of the room | 152 | 45.39% | 128 | 55.47% |
| gets into bed | 132 | 96.97% | 82 | 82.93% |
| gets out of bed | 108 | 87.04% | 84 | 91.67% |
| lies in bed | 80 | 91.25% | 60 | 88.33% |
| rolls in bed | 120 | 31.67% | 66 | 57.58% |
| sits down on the bed | 86 | 84.88% | 58 | 70.69% |
| sits down on the chair | 144 | 35.42% | 108 | 37.96% |
| sits in bed | 78 | 41.03% | 56 | 50.00% |
| stands up from the bed | 78 | 93.59% | 54 | 77.78% |
| stands up from the chair | 152 | 25.66% | 104 | 26.92% |
| stands up from the floor of the room | 184 | 32.61% | 134 | 26.87% |
| walks toward the bed | 214 | 58.88% | 158 | 48.10% |
| walks toward the chair | 184 | 52.17% | 136 | 50.00% |
| walks toward the room | 163 | 41.72% | 130 | 53.08% |

环境字段只能报告parrad这一组。各sensor、subject、边缘标记统计见原始JSON；记录含重叠窗口和双视图，计数不是独立物理事件。

### 文本抽查与人工复核边界

已对队列前8条validation、前8条test进行AI辅助文本核对，逐条注释保存于text-sample-review.json。例如 `[9,11,5]→[9,11]` 未覆盖最后的床上坐下，`[5,8,13]→[6,9,13]` 是床/椅类别替代候选，`[4,7,4]→[4,7]` 无法确定哪次翻身缺失。

抽查还出现参考标签未支持的“站在桌前”和“滚下楼梯”等自由文本。14类短语匹配不会将这些词表外内容计入extra_class_occurrences；该字段为空不代表没有幻觉，当前代理指标不能衡量全部文本事实性。是否存在对应物理证据仍待人工雷达复核。

这不是人工雷达标注：当前输出没有可靠的逐时段对齐，不能把类别替代候选确定到某个时间段。每split40条人工复核队列保持pending，物理时段替代判断需后续人工雷达复核；未伪造人工完成状态。

[全部漏检分组](../outputs/radar-text-diagnostics-20261011-v1/diagnostics/miss-attribution.json) · [完整/可见时长交叉分析](../outputs/radar-text-diagnostics-20261011-v1/diagnostics/instance-extended-audit.json) · [16条文本抽查](../outputs/radar-text-diagnostics-20261011-v1/diagnostics/text-sample-review.json)

## 5. 瓶颈定位与下一步优先级

1. **首先查稳态类别感知与原分类读出。** 边界排除收益很小；上床、床上站起和椅子/地面动作存在跨类混淆。同一Stage1冻结特征换统一读出明显改善，优先检查训练预算、类平衡、读出校准与标签策略，不急于增加编码器复杂度。
2. **保留BiGRU，不把Align退化作为当前主因。** 四路统一探针支持时间上下文有益，Align后整体读出未变差；仍需注意单受试者验证和类别差异。
3. **下一步定位池化/Projector/语言目标和解码的覆盖损失。** Align特征仍可简单读出，且错配敏感，但生成存在系统性末端/截断动作缺失。当前试验没有单独隔离这些模块，不能断定唯一责任在Projector。建议先做冻结编码器下的接口与可控双动作验证，再决定是否LoRA。
4. **先补监督和评价数据。** 完整短动作、双动作反序和真实环境标识不足；构造类别平衡的完整单动作与可控双动作窗口，避免仅靠模板分数宣称细粒度能力。新增最终泛化验证需未使用受试者。

## 6. 产物与验证

服务器原生53项测试通过；最终本地相关回归48 passed,1 skipped（本地缺PEFT），同步末帧去重回归已红→绿。单worker于00:08:02启动、00:22:10完成全部6阶段。49项下载产物逐项验证SHA，119项服务器源码和原始Stage1/Align/Qwen/manifest/label-map SHA复核一致，原始权重未改变。逐帧混淆矩阵已从导出的逐帧预测重新计算。

服务器输出：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/radar_text_diagnostics_20261011_v1`。38个冻结特征缓存留在服务器，文件SHA在feature-cache-manifest.json；本地导出不含原始RD/MD或完整Qwen。

[机器核验记录](../outputs/radar-text-diagnostics-20261011-v1/diagnostics/verification.json) · [探针权重示例](../outputs/radar-text-diagnostics-20261011-v1/diagnostics/align_gru-probe.pt) · [固定前缀权重](../outputs/radar-text-diagnostics-20261011-v1/diagnostics/language-prior-best.pt) · [test逐帧输出](../outputs/radar-text-diagnostics-20261011-v1/diagnostics/frame-predictions-test.npz)

结果archive SHA256：`7ff72ff86c8b7ee1fd0992c0fc02c2e8252c5886aac6745114fdcc5a63736052`。源码capsule SHA256：`970f9a741b7386d5c608c3daedc0f85f71874ce1d35e81f3fab3cad0734053ad`。本地后续去重统计修复、图表和派生审查单独记录，不改写原始归档。

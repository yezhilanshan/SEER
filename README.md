# SEER：毫米波连续行为理解多方案代码仓库

> **修改前必读：本仓库同时保留多种研究方案、历史版本和对照实验。修改前必须明确目标方案及其代码范围；不能确认时，先向用户询问，只做只读定位，不开始修改。**

## 修改代码前的强制规则

这些规则适用于维护者、协作者和代码助手。

1. **先明确修改哪个方案。** 根据用户的明确指令或已经确认的上下文，确定方案名称、适用版本和目标功能。不能把“最近修改的方案”自动当作所有后续任务的目标。
2. **再确认对应入口与文件。** 对照下表及该方案文档，找出实际使用的模型、训练/推理入口、配置和测试。目录相近、模块同名或使用相同数据集，都不代表属于同一方案。
3. **不清楚就不动手。** 如果任务只说“改模型”“改训练”“优化HARrad”，且上下文无法唯一确定方案，必须先询问。确认前仅允许只读查看和定位，不修改代码、配置或权重，不启动训练，不生成或覆盖实验产物。
4. **改动限定在已确认的方案内。** 不顺带重构其他方案，不把一种方案的训练规则、状态重置、记忆结构或事件定义自动套到另一种方案。
5. **共享代码需要单独检查影响范围。** 修改前先查调用关系；如果会改变其他方案的行为，先说明并明确跨方案修改范围。无法确认兼容性时，优先在目标方案中增加独立实现或适配层。
6. **保留旧模型与实验资产。** 不覆盖其他方案的检查点、配置、原始数据、历史结果或失败记录。新实验使用独立输出目录；未经明确要求，不替换旧入口的默认模型或行为。
7. **验证与报告明确归属。** 使用对应方案的测试和验收协议，说明修改了哪个方案、哪些文件、验证了什么。单个组件测试通过，不等于该方案端到端能力已经验证。

开始修改前，应能清楚说明：

```text
目标方案：想法几 / 具体路线 / 版本
目标功能：本次要修改的行为
代码入口：模型、训练或推理入口及配置
改动范围：计划修改的文件；是否涉及共享依赖
验证方式：对应测试或实验协议
```

**如果“目标方案”或“代码入口”仍无法确定，停在定位阶段，先询问用户。已经在上下文中确认过的范围不必重复询问。**

## 当前方案导航

下表按研究方向归类，方便定位代码；历史版本、消融配置和训练阶段不重复算作独立大方向。各方向的实现与验证程度不同，不能因为代码存在就认定整套系统已经完成。

| 大方向 | 核心思路 | 主要代码范围 | 方案与使用说明 |
| --- | --- | --- | --- |
| **想法三：显式事件建模、预测驱动事件切换** | 维护事件状态，判断何时切换或Reset，再接分层事件记忆；包含v2.2–v2.6.1的历史演进 | `src/continuous_mmwave/v25/`、`v26/`、`v261/`；早期实现涉及`models/`、`stages/`及`parrad_*.py`；HARrad适配涉及`harrad/joint_*`、`am_*`、`reset_*`等，须按具体实验定位 | [想法三方案目录](计划/想法三/)、[v2.6方案](计划/想法三/方案书v2.6.md) |
| **想法五：无显式分段的潜在记忆** | 连续雷达流进入短/中/长期记忆，通过LMT检索与演化，再交给LLM描述；不以显式事件切分作为主路径 | `src/continuous_mmwave/latent_memory/` | [方案书](计划/想法五/方案书v1.0.md)、[实现记录](计划/想法五/实现记录v1.0.md) |
| **想法六：CIF/LIF事件门** | 积累语义变化证据，积分发火形成事件；当前首版使用HARrad，尚未接入LLM | `src/continuous_mmwave/event_gating/`、`configs/harrad-cif-lif-v1.json`、`tests/test_event_gating_*.py` | [想法六方案目录](计划/想法六/)、[当前实现设计](docs/superpowers/specs/2026-10-26-harrad-cif-lif-design.md)、[运行说明](docs/harrad-cif-lif.md) |
| **离线Radar-to-Text** | 固定雷达片段经CNN/BiGRU、雷达tokens和Projector送入Qwen，生成片段描述 | `src/continuous_mmwave/radar_text/`、`configs/radar-text-*` | [运行说明与实验记录](docs/offline-radar-to-text.md) |
| **识别与时序分割基线** | MD-GRU、RD-CNN-GRU、ResNet18、Split BiRNN及分类后处理等，用于独立识别或对照 | `src/continuous_mmwave/harrad/model.py`、`harrad/training.py`、`split_birnn/`；部分ResNet和后处理对照位于`radar_text/` | [HARrad语义基线](docs/harrad-semantic-baseline-results-2026-10-17.md)、[Split BiRNN](docs/split-birnn-training-results-2026-10-14.md) |

### 容易混淆的边界

- **HARrad、PARrad是数据集名称，不是方案名称。** 多种方案使用同一数据集，不能仅凭数据集决定修改哪套模型。
- **`harrad/`不是某一个方案的专属目录。** 它既包含基础分类器和流式前端，也包含语义预热、联合预测、调制及Reset实验；需要明确具体文件和入口。
- **共享前端不等于共享整套模型。** 想法五、想法六可以只读复用已有HARrad前端，但各自的记忆、事件门、损失与推理状态规则不同。
- **想法三的版本不能随意混用。** `v25/`、`v26/`、`v261/`对应不同阶段的实现和实验契约，修改时要确认具体版本。
- **CIF、LIF是想法六内部的两种事件形成方法。** `lif-distill`是训练阶段，`frame-baseline`是对照；不应误当成另外两条研究主线。
- **同目录的多个方案文档也不能自动合并。** 想法六目录同时保留草案、SAEF设计、最小可行架构和骨架预训练构想，实际修改以用户指定或已批准的设计为准。
- `configs/`、`scripts/`、`tests/`、`outputs/`中的文件也有方案归属，操作前要核对其调用入口、来源锁及实验协议。

## 环境与操作约定

- 本项目pytest使用名为`interview`的conda环境。
- 如果任务涉及AnotherMe项目，使用`AnotherMe-V2`环境测试，不能混用本项目环境。
- APK由用户自行安装，不代替用户重新安装APK。

## 历史实现与实验记录

以下保留原README的v2.2 Phase 1说明及后续累积记录。这里的“已启动”“已完成”“当前”等描述属于相应日期和实验范围；修改前仍需结合上方方案导航及对应文档核对，不能把某条历史记录当作整个仓库唯一的当前方案。

已启动[HARrad Stage2–3 Reset闭环验证](docs/harrad-reset-closed-loop-progress-2026-10-22.md)：冻结独立A/B前端，保留全局G、重置局部L与语义E，检验预测损失对Policy的真实作用。本地/服务器各20项测试与真实Qwen机制预检通过；10月22日16:13启动，Stage2A已完成1000步，后续依次为Policy初始化、短试验及两臂三seed配对训练。最终效果尚待整流评价和重放核查，不代表完整v2.6训练已完成。

已完成[HARrad语义对齐×事件引导调制2×2实验](docs/harrad-alignment-modulation-results-2026-10-22.md)：10月22日10:39结束，四组×三seed共12000次成功更新。验证Macro-F1均值为B0 62.02%、仅对齐62.96%、仅调制62.97%、组合61.54%，均低于同面板初值64.52%；关闭调制仅损失约0.05个百分点。48检查点读回、78面板独立复算、210源码/5源锁/70缓存SHA复核通过，权重保留服务器，预测证据已下载。尚未显示组合互补或历史调制实质收益；[启动与历史进度](docs/harrad-alignment-modulation-progress-2026-10-21.md)保留。

本地[工作区清理记录（2026-10-18）](docs/workspace-cleanup-2026-10-18.md)：移除逐成员核验的重复压缩包、中断下载及测试临时文件，释放约7.51 GiB；正式训练权重、预测和研究记录保留。

已完成[HARrad Dynamics与语义联合预训练及核查](docs/harrad-joint-prediction-results-2026-10-18.md)：M/A/B/AB四臂×三seed各1000步，共12000次成功更新，10月18日04:59结束。h5宏Skill为M29.64%、A30.91%、AB28.36%；语义Macro-F1为B60.43%、AB58.80%，当前共享Adapter等权联合训练未显示净收益。12组权重重放误差0，120任务文件与9元数据SHA、48检查点读回及全组逐点独立复算通过，180源码/70冻结缓存保持不变；权重和预测均下载。固定subject7面板、Oracle前缀、CNN/Qwen冻结；未启动自主Reset。

此前[HARrad联合预训练启动记录](docs/harrad-joint-prediction-progress-2026-10-17.md)保留预检、协议和首批进度；完整训练已结束，最终结果见上方报告。

已完成[HARrad连续监督边界基线](docs/harrad-boundary-baseline-2026-10-17.md)：31条完整训练录制，冻结CNN/GRU连续特征，只训练33,154参数起止头，seed42×20epoch共620步。subject7主容差±4帧起点/终点F1为55.94%/67.33%，变化量基线41.83%/34.99%、周期基线20.38%/21.59%。未知区间未评分预测214/237，起点Recall47.95%，尚不适合直接接自主Reset。独立匹配回溯问题修复后重载原权重完成v2复核，无重训；best概率重放误差0、18组指标与38文件下载哈希通过，原v1失败证据保留，test未参与。

已完成[HARrad完整录制与官方起点重置回放](docs/harrad-oracle-start-replay-2026-10-17.md)：4条validation录制25,244帧全部读取（含8,575未标注帧），在511个官方实例起点重置状态、年龄和双帧buffer。Macro-F1为76.73%，接近原片段77.01%，但3个类别变化、严格逐值检查未通过。4个分歧实例在相同特征与batch设置下流式/片段误差0，还原原batch4也误差0，差异支持CNN分块与FP16批尺寸数值影响。11文件下载哈希及4组预测独立复算通过；新增训练0步，未启动自主Reset。

已完成[HARrad流式前端与EventFormer语义预热](docs/harrad-event-warmup-progress-2026-10-17.md)：35条录制215,482帧导出及原分类复现通过；小样gate40步通过，正式seed42训练5轮共3,775步，于17:41结束。官方实例重置下最佳validation Macro-F1为77.01%、accuracy77.50%，best重放误差0、冻结基座保持不变；零CNN特征为1.27%、连续无重置仅12.89%，后者尚不具备直接长流使用能力。12份最终预测本地独立复算通过；test信号未参与，自主Reset尚未启动。

已完成[HARrad首轮语义基线](docs/harrad-semantic-baseline-results-2026-10-17.md)：两模型各3seed×30epoch，11:01结束，总耗时41分52秒。subject7验证集best Macro-F1均值：MD-GRU78.89%、RD-CNN-GRU84.52%，RD三seed均胜。6组权重重放误差0、255个导出文件哈希及186份预测的独立指标复算通过，权重/日志/预测已下载；测试集未参与，尚未训练EventFormer或联合分支。

已完成[HARrad格式与重训接入检查](docs/harrad-format-audit-2026-10-16.md)：9人37条连续录制、233,515帧、12类3,852个官方帧区间；原始数据为`float32[T,256,256]`，官方FFT小样得到RD`160×256`和MD256维。完整CSV独立核验及12类521帧派生小样核验通过。官方随机划分共享受试者和录制；建议按人划分，保留36.6%未标注区域mask，重新训练雷达与语义分支。原始数据未改，正式重训尚未启动。

已完成[14类42段命令代理片段雷达复核](docs/v261-clip-review-results-2026-10-16.md)：28训练/14验证段，12名受试者、84个双雷达视图，附逐段清单、MD/RD距离投影和原始RD快照。信号可能在命令前已变化、段内较晚增强或两视图不同；这些是复核线索而非错标结论。116份导出证据哈希、84视图线索值与168条判据独立复算通过；真实类别/动作边界仍未由独立参考核实，未改标签、未新增训练。

已完成[v2.6.1第1项语义诊断](docs/v261-semantic-diagnostic-results-2026-10-16.md)，新增训练0步：三seed四检查点的固定面板上，AB语义训练Macro-F1由19.40%升至30.54%，验证由13.85%升至15.69%；类别匹配后纯/跨动作前缀平衡准确率34.48%/29.25%，原始F也存在跨动作下降和旧类滞留。24条录制标签/生成时间契约一致；原始HDF5及目录未提供独立时钟/动作参考，真实对齐G0仍未通过。1152份新预测、96组曲线与匹配/滞留结果独立复算通过，下一步优先监督时序和语义单任务基线。

v2.6.1已完成[四臂三seed训练与结果核查](docs/v261-joint-results-2026-10-16.md)：主训练12,000次、统一探针6000次更新；48个主检查点、15个探针检查点和162份预测原生重放通过。h5 Skill为MLP22.78%、A-only21.53%、A+B21.49%；代理分类Macro-F1为B-only17.40%、A+B17.23%。共享Adapter联合训练未显示稳定净收益，主干仍冻结；已下载全部检查点、全量逐点误差/logits和原始预测抽样，本地独立复算通过，完整原始预测留服务器。正式Reset仍暂停。

已完成[v2.6.1双状态联合预训练设计](docs/superpowers/specs/2026-10-15-v261-joint-pretraining-design.md)与[实施清单](docs/superpowers/plans/2026-10-15-v261-joint-pretraining.md)：首轮冻结MD-GRU主干、训练共享Adapter，对照MLP/A-only/B-only/A+B，h5为主要预测指标，分开评价语义与共同表示。服务器只读核验24份原缓存/源数据及命令摘要，物理共同起点fit61,299/validation20,075，当前代理语义合法起点59,540/19,609。该文档记录设计阶段，最终结果见上方完成报告；[预检与启动记录](docs/v261-joint-preparation-results-2026-10-15.md)保留为历史。

已完成[v2.6 P1-C多Horizon与Event信息消融](docs/v26-p1c-results-2026-10-15.md)：45个配置各300步，GRU/TCN使用完整匹配历史；Oracle主实验h3/h5的motion-only MLP宏Skill为+11.73%/+19.24%，Event为+7.20%/+15.71%。同头常量E在三个Horizon、三验证人的9个点预测比较中均优于正确E，尚无Event额外净预测收益。NoReset平台存在坐标能量集中，状态仍变化，56个Gate片段exact；不将其作为Reset启动依据。51项本地/原生相关测试、225检查点读回、378预测/控制及本地独立复算通过。下载全量预测、诊断及90份0/300检查点；135份中间检查点留native，未启动正式Reset训练。

已完成[v2.6 P1-C完整因果回放与准备](docs/v26-p1c-preparation-results-2026-10-15.md)：24条录制的Oracle/NoReset每种覆盖81,518个Update，包含全部2,236个混合Update；Oracle在原63,603个选点逐元素误差0。48份状态文件读回、相关37项本地/原生测试、下载证据及本地独立复算通过。h1/3/5共用fit61,299/validation20,075起点；线性趋势均弱于Persistence。NoReset出现长时幅值平台，Oracle错配消融索引已固定，NoReset因无同年龄过去供体将B标N/A。仅回放、基线与协议准备，新增神经训练0步，P1-C不是Reset放行条件。

已完成[v2.6 P1-A/P1-B固定预算实验](docs/v26-p1-results-2026-10-15.md)：拆开观测与事件归属mask，h1恢复fit/validation的1,768/468个截止时刻跨命令边界配对；原共同位置上的四个零初始化残差头各训练300步。Residual-GRU受试者宏Skill为+1.21%（三人均正），Residual-Event为−1.62%，仍无事件状态额外收益。20份检查点、30份预测回放及105文件下载核验通过。新增全集缺合法E的位置未伪造，五臂跨界辅助评价未执行；P1-C和Reset未启动。

已完成[v2.6首轮预测P0只读诊断](docs/v26-p0-diagnostic-results-2026-10-15.md)：四个神经头在完整fit及实际抽到的位置上也输给Persistence，低/中/高变化及代理边界邻域分组均无反转。定位到mask混用了代理事件归属与观测有效性，导致全部合法h1预测对跨代理边界数量为0；首轮结果不能证明跨事件转换预测能力。31.8万逐对分解、96组时滞动态性及203文件证据核验通过；零新增优化步，未开始P1或Reset训练。

已完成[v2.6冻结前端预测预实验](docs/v26-prediction-pilot-results-2026-10-15.md)：18份训练/6份验证缓存，四个新增预测头各300步，预测未来0.18秒。受试者宏MSE：Persistence 0.03366、无事件MLP 0.07130、GRU 0.06755、TCN 0.06822、Event State 0.07351；本轮未显示事件状态额外收益。30份预测的独立指标复算和权重逐值回放、65个下载证据文件哈希核验通过。使用命令边界代理，正式实例/时钟G0仍缺失，test数组未读取，未开始Reset闭环训练。

已完成[Split BiRNN五折训练与结果核验](docs/split-birnn-training-results-2026-10-14.md)：23人153份MD记录，Split与普通BiGRU各五折、每次6400步；共同有修正输出帧上的五折平均Weighted F1为80.06%/79.30%，Split前向即时输出为58.23%。306份测试预测独立重算与权重逐值回放通过，权重、日志、逐帧预测和1298文件哈希核验产物已下载到本机。最终修正平均等待37.17秒；Hospital映射、实际时钟及CNN参数差异仍未确认，属于方法复现实验，尚未严格复现原文数值。此前[准备阶段报告](docs/split-birnn-preparation-results-2026-10-14.md)保留为历史记录。

已按新9/3/3划分完成[RD-only ResNet18从头训练与结果核验](docs/rd-resnet18-scratch-results-2026-10-13.md)：20epoch，validation选第17轮；test受试者平均Macro-F1为50.16%，06/17/18分别76.98%/29.14%/44.35%，新增17/18平均36.75%。真实标签、logits重放与指标独立重算通过；低分受试者泛化仍是瓶颈，本轮没有训练Qwen。

扩充PARrad的[新受试者划分已固定](docs/parrad-expanded-split-2026-10-11.md)：沿用旧3:1:1比例及旧角色，train为01/02/03/04/07/08/09/13/19，validation为05/12/16，test为06/17/18。06属于历史已观察测试，17/18另作新增模型留出视图，报告需分开列示。

方案六当前补全入口为 [HARrad SAEF-MVP](docs/harrad-saef-mvp.md)，以 `计划/想法六/最小可行架构.md` 为准：完整自然录制的有监督连续事件形成、B0–B3对照、训练集校准与闭环评估，暂不接LLM。

另有独立的 **离线 Radar-to-Text MVP**：双分支 RD/MD CNN → BiGRU → 8 个有序 Radar Tokens → MLP → Qwen2.5，直接生成动作描述。包含雷达预训练、冻结 LLM 对齐、可选 LoRA、无标签推理和分类模板对照；使用说明见 [离线 Radar-to-Text](docs/offline-radar-to-text.md)。旧在线实验代码保留供复盘。

新增下载的[PARrad数据已转换并校验](docs/parrad-expanded-preparation-2026-10-11.md)：15人、45个session、89份RD/MD记录，旧38份复用、新51份转换，共599,799帧及8,107个描述窗口。服务器数据目录为`/home/xibeidaxue/newdisk2/WJ/PARrad/canonical/offline_expanded_20261011_v1`；按人清单可重新划分，旧05/06视图仅用于兼容。本轮没有启动从头训练。

真实 PARrad + Qwen2.5-0.5B 首轮 pilot 已完成：独立 subject06 测试完整动作序列 25.00%（分类模板基线 11.16%），动作召回 46.28% 仍低于基线 55.51%，不能视为连续描述能力已通过。[结果、错误分析与权重](docs/radar-text-server-pilot-2026-10-10.md)。

2026-10-11 已复用原权重完成系统性错误归因：14类逐帧混淆矩阵、Stage1/Align四路冻结特征探针、平滑分类模板、固定前缀无雷达基线、错配/时间扰动与真实异序配对、动作覆盖分桶。结果支持BiGRU有益、Align未整体退化，稳态类别混淆与末端/截断动作缺失仍明显；人工雷达时段复核待完成。详见[瓶颈定位报告与全部诊断产物](docs/radar-text-error-attribution-2026-10-11.md)。

随后完成[原分类头H0–H4受控消融](docs/radar-text-head-ablation-2026-10-11.md)：先复现原读出，再在原优化器与旧探针配置下各训练5个冻结线性头。原优化器组测试Macro-F1由H0短训45.35%升至H1延长训练49.28%、H4组合49.42%；类权重单独的测试收益很弱，标准化不等同概率校准。CNN/BiGRU/Qwen与38份缓存未改变，分类改善尚未验证能否转化为生成收益。

已进一步完成[同滑动窗口强分类到文本对照](docs/radar-text-strong-baselines-2026-10-11.md)：重新提取原描述窗口特征并复现Qwen生成，validation选择共同平滑参数。原484窗口中B0/B1(H1)/B2(H4)/B3完整序列准确率分别为11.78%/18.18%/18.39%/25.00%；排除两个不足12秒尾窗后为11.83%/17.84%/18.05%/24.69%。B3序列优势缩小但保留，H1/H4动作Recall约60%与LCS约47%均高于B3约46%/45%；生成覆盖问题仍是重点。没有新增模型训练。

最新[P0同编码器公平对照](docs/radar-text-align-readout-p0-2026-10-11.md)已冻结Align重新提取特征并按原H1/H4策略只训练两个线性头。严格482窗：Stage1 H1/Align H1/Align H4/Align Qwen完整序列准确率17.84%/18.88%/18.67%/24.69%；Align H1 Recall65.91%高于Qwen46.06%，但Count MAE1.098高于Qwen0.336。Align提升线性可读性，生成路径仍保留完整序列优势；分类偏多输出，生成仍漏床相关动作，不能单独量化LLM因果贡献。

随后完成[P1动作证据保留与数量校准](docs/radar-text-p1-evidence-count-2026-10-11.md)：同一冻结Align Encoder下，严格482窗原分类/HMM/预测数量段Viterbi/Qwen完整序列准确率为18.88%/25.52%/27.18%/24.69%，分类多预测窗253→31。此前生成优势在加强数量控制后不再保留，不能归因于LLM本身。H/Z/P有限探针未显示H→Z整体退化，Z→P有序读取变弱；Oracle真数量34.02%仅作诊断。全部86个独对窗口有符号/文本复核，未做人工雷达时段复核；测试3动作占80.91%，数量头总体计数准确率低于训练多数基线，但完整序列更好。仅训练六个探针与独立数量头，Encoder/Projector/Qwen未更新。

已完成[RD-only X-Fi冻结骨干迁移](docs/rd-only-xfi-transfer-2026-10-11.md)：原Stage1 RD-CNN、随机ResNet18、预训练ResNet18各训练新时序/多任务读出20epoch，空间骨干与BN冻结。test逐帧Macro-F1为62.01%/62.25%/62.52%，严格482窗预测数量模板Exact为32.78%/31.33%/33.61%；验证集选择原小CNN，预训练收益随读出方式变化，尚不足支持直接替换。单通道适配、原权重保护、48组指标独立重算及27个真实RD/384个读出重建均通过。没有重训Qwen或声称语言生成改善，旧语义接口不能直接复用。

当前进度已扩展到PARrad Stage3和Stage4组件实验。沿用旧控制器的雷达→事件→分层记忆→LLM回放已执行。另已新增独立`continuous_mmwave.v25`代码路径，包含顺序证据、异步边界定位、Active重建及Memory/QA接口；该新路径仍需真实权重训练、校准及连续雷达验收，旧回放入口没有自动切换，新QA与多事件效果也未达标。范围见[v2.5实现状态](docs/v2.5-implementation-status-2026-10-06.md)及[新模块用法](docs/v2.5-online-modules-2026-10-06.md)。下述Phase1内容为最初实现范围。

本工程严格实现《方案书 v2.2》冻结的第一阶段顺序：

`Radar Encoder → Stage 1 Pose Pretraining → Event Former → Stage 2 Streaming/Commit Alignment`

最初Phase1未包含Proposal、Candidate与分层事件记忆。后续已有旧版Fork-and-Shadow控制器与分层记忆实验入口，具体状态见上方记录及后续Stage3/4章节。

## 已实现

- RD：每帧四层 `Conv-BN-GELU-AvgPool`；PointNet：`6→64→128→256→max`。
- 两分支均采用 stateful causal TCN，dilation 为 `1/2/4/8`；20 fps 每 4 帧更新一次。
- 两个 RD query 与两个点云 query 形成 `[R1,R2,P1,P2]`，加入模态与无长度上限的正弦时间编码，再经两层 Transformer。
- Stage 1 临时 Pose Head `256→512→J×3`，损失为 `MPJPE + λv Lvelocity`；导出时严格移除 Pose Head。
- 8 个 896 维 Event Tokens；固定 Event Prompt；Qwen2.5-0.5B-Instruct 原生前四层；仅 q/v LoRA（rank 16）。
- 每层使用显式 512 维多头 cross-attention，`tanh(alpha)` 且 `alpha=0` 初始化；`Wout` 恒等初始化；token-wise scalar state gate。
- 固定 Prompt K/V 缓存、Qwen BF16 路径、streaming activation checkpointing。
- Commit 从同一 `P_base` 出发，加入 `[0,1]` 相对时间编码，最多 256 个 radar tokens。
- 文本经过冻结 LLM 前四层；事件与文本共享 learnable semantic query 和 `896→256→128` 归一化投影。
- Stage 2 仅包含 `Ltext + λa Lalign + λc Lcons` 三项损失；Radar Encoder 以独立小学习率继续训练。
- 支持可变事件长度、可变有效点数、AMP、梯度裁剪、trainable-only Stage 2 checkpoint 与确定性断点恢复。

## 数据

JSONL 每行：

```json
{"path":"subject01/event003.npz","event_id":"s01-e003","caption":"a person sits down"}
```

NPZ 包含 `rd[T,2,64,64]`、`points[T,N,6]`，Stage 1 还需 `skeleton[T,J,3]`。可选 `point_mask[T,N]` 或 `point_counts[T]`。collate 自动填充不同的 `T/N`，并让全部损失与注意力忽略填充。Stage 2 的 InfoNCE 要求 `batch_size>=2`。

## 使用

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

PYTHONPATH=src python -m continuous_mmwave.train stage1 --config configs/stage1.yaml
PYTHONPATH=src python -m continuous_mmwave.train stage2 --config configs/stage2.yaml

# 同阶段续训
PYTHONPATH=src python -m continuous_mmwave.train stage2 --config configs/stage2.yaml --resume outputs/stage2/last.pt

# CPU 端到端冒烟验证
bash scripts/smoke_train.sh

# 单元/集成测试
PYTHONPATH=src pytest -q
```

生产配置默认使用 Qwen2.5-0.5B-Instruct，首次运行时需要准备相应权重和可用 GPU。合成数据只验证接口、梯度与 checkpoint，不可作为论文结果。

### 可选 Gemma 3 270M-IT 后端

Qwen2.5-0.5B-Instruct 是 `configs/stage2.yaml` 的默认生产路线。Gemma 3 270M-IT 仍作为可选的现代解码器，
可通过独立配置 `configs/gemma3-270m-stage2.yaml` 使用：

```bash
PYTHONPATH=src python -m continuous_mmwave.train stage2 --config configs/gemma3-270m-stage2.yaml
```

该后端使用 `google/gemma-3-270m-it`，取前 4 层作为 Event Former 的语义骨干，并仅在这些层的
`q_proj` / `v_proj` 上训练 LoRA；Gemma 主干其余参数保持冻结。其隐藏维度为 640，因此该配置与
Qwen 的 896 维 Stage 2 checkpoint 与 Gemma 的 640 维 checkpoint 不兼容，但两者均可继续复用相同的 Stage 1 Radar Encoder 导出。
当前 Gemma 路径使用正确性优先的 Prompt/Event 拼接前向，尚未实现 Hybrid KV Cache 快路径。

Gemma 权重受许可门控。首次运行前，需要在 Hugging Face 接受 Gemma 使用许可，并在本机配置有权
访问该仓库的凭据；训练程序不会代替使用者接受许可。

## Stage 2 go/no-go

继续实现 Proposal/Candidate/Fork-and-Replay 前，必须在真实数据验证：Event semantic probe 可分性、radar→caption 有效性、stream/commit cosine 与 caption 差异、5 Hz P50/P95 延迟。本工程不虚构这些实验结果。

## PARrad 验证分支

PARrad 路径是独立的验证适配层，不替换原有的 `RD + Point Cloud` 实现。它只为后续验证 Event Token、连续语义状态和事件边界机制提供输入，不支持 Pose 监督，也不把固定类别模板的生成结果解释为自由文本描述能力。

前端以 RD map 为主表示，以 Micro-Doppler 为运动先验：MD 经轻量卷积和因果 TCN 后，通过零初始化 FiLM 调制 RD 特征；增强后的 RD 特征保留较细的 Doppler 轴，并由四个查询固定重采样为每 2 帧一组的 4 个 Radar Tokens。按 PARrad 约 0.09 秒的帧周期，一次在线更新对应约 0.18 秒。`semantic_state` 来自 RD Radar Tokens，`motion_state` 单独保留给后续边界机制。

### 规范化数据格式

当前读入层使用转换后的 NPZ，而不是猜测原始发布文件的目录或序列化格式。每个 NPZ 必须包含：

```text
rd          float32 [T, 1, 93, 128]
md          float32 [T, 128]
labels      int64   [T]
timestamps  float64 [T]
```

JSONL manifest 每行必须包含：

```json
{"path":"recordings/s01_session01.npz","subject_id":"s01","session_id":"session01","sensor_id":"radar60","environment_id":"room01"}
```

训练、验证和测试集必须按 subject/session 隔离，并共享由全部已划分 manifest 构造的稳定 sensor vocabulary。

真实 PARrad 转换器只读取每个传感器的 `rdn` 和 `mDoppler`，不会加载原始 radar cube。它把两个雷达保留为独立 recording，以 0.09 秒帧周期生成相对时间戳，并按“命令时间戳起生效、持续到下一条命令”的规则生成逐帧标签；显式采集起点早于第一条命令时，前缀标签为 `-100`。命令文本会折叠多余空白，类别 ID 写入 `label_map.json` 并在后续转换中复用。

```bash
PYTHONPATH=src .conda/envs/seer/bin/python -m continuous_mmwave.parrad_convert \
  --h5 /path/to/processed_data.h5 \
  --commands /path/to/timestamp_speech.csv \
  --output-dir /path/to/canonical \
  --subject-id subject_04 \
  --session-id set000 \
  --environment-id parrad
```

### 当前可运行验证

```powershell
$env:PYTHONPATH='D:\lab\SEER\src'
& 'C:\User\anaconda\envs\interview\python.exe' scripts\make_synthetic_parrad_data.py
& 'C:\User\anaconda\envs\interview\python.exe' -m continuous_mmwave.parrad_audit data\synthetic_parrad\manifest.jsonl
& 'C:\User\anaconda\envs\interview\python.exe' -m pytest tests\test_parrad_data.py tests\test_parrad_encoder.py tests\test_parrad_stages.py -q
```

PARrad P1 已接入统一训练入口。真实 recording 会先缓存在主机内存。生产训练先按连续 label run 取样：每个动作游程两端按 `label_boundary_ignore_seconds` 裁剪，少于 `min_label_run_frames`（生产值为 8）的短游程会丢弃，较长游程被均衡切成不超过 64 帧的单标签样本，避免产生极短冷启动尾块；padding 不参与编码器归一化。随后执行连续流 TBPTT 微调：每个 256 帧序列按 64 帧分块，跨块及跨 256 帧序列持续携带 TCN cache 和全局 `update_index`，只在切换 recording 组时重置，并在每个 64 帧块边界 detach 状态。

验证和测试不按标签选择窗口，而是按固定帧数遍历整段 recording，并跨块持续携带状态；标签仅用于 loss 和指标。生产配置考虑到 RTX 2080 Ti 已接近温度上限，使用低负载训练 batch 8、连续评估 recording batch 8 和 `num_workers=0`：

```bash
PYTHONPATH=src .conda/envs/seer/bin/python -m continuous_mmwave.train stage1 \
  --config configs/parrad-stage1.yaml
```

可先追加 `--max-steps 1` 做 CUDA 冒烟。正式运行前使用 `configs/parrad-stage1-pilot.yaml` 做 3+1 epoch 试跑。`epochs` 控制 label-run 预训练轮数，`streaming_finetune_epochs` 控制后续连续流微调轮数；两阶段分别使用 `learning_rate` 和 `streaming_learning_rate`，均采用 warmup-cosine 调度。验证早停在各阶段独立计数，并分别写出 `best_label_run.pt` 与 `best_streaming.pt`。`streaming_sequence_frames` 必须是 `streaming_chunk_frames` 的整数倍。输出还包含逐更新学习率的 `metrics.jsonl`、可恢复的 `last.pt`，以及不含临时分类头的 `radar_encoder.pt`。

真实 P1 配置按受试者划分为 train=02/03/04、validation=05、test=06，并只用训练集统计 balanced class weights。每轮验证后按 validation accuracy（同分时取较低 loss）更新全局 `best.pt`；连续评估同时记录边界裁剪与原始帧准确率、macro recall/F1、动作段准确率、逐类别召回率和逐传感器准确率。训练结束后使用 `best.pt` 写出 `test_metrics.json` 和供后续阶段使用的 `radar_encoder.pt`。三个 manifest 可复现生成：

训练后可用同一连续流协议补充诊断 train/validation/test。诊断输出包含混淆矩阵、逐类 precision/recall/F1/support、按 subject/sensor/recording 分组结果、原始与边界裁剪指标，以及动作段内部预测切换率：

```bash
PYTHONPATH=src .conda/envs/seer/bin/python -m continuous_mmwave.parrad_diagnostics \
  --config configs/parrad-stage1.yaml \
  --checkpoint /path/to/best.pt \
  --output /path/to/diagnostics.json
```

```bash
PYTHONPATH=src .conda/envs/seer/bin/python -m continuous_mmwave.parrad_split \
  /home/xibeidaxue/newdisk2/WJ/PARrad/canonical/phase1/manifest.jsonl \
  --output-dir /home/xibeidaxue/newdisk2/WJ/PARrad/canonical/phase1 \
  --train-subjects subject_02 subject_03 subject_04 \
  --validation-subjects subject_05 \
  --test-subjects subject_06
```

`configs/parrad-tiny-stage1.yaml` 与 `configs/parrad-tiny-stage2.yaml` 保留 CPU 合成验证参数。P2 已接入统一训练入口：它加载 P1 导出的 MD-GRU Radar Encoder，只接受 `label_run` 单标签事件段，并用每类三个确定性轮换模板训练 streaming/commit Event Token。验证与测试记录三项损失、stream/commit key cosine 和 Event Token 状态范数，并按验证总损失选择 `best.pt`。

服务器单步 CUDA 冒烟使用独立输出目录，不会覆盖正式实验：

```bash
PYTHONPATH=src .conda/envs/seer/bin/python -m continuous_mmwave.train stage2 \
  --config configs/parrad-md-gru-stage2-smoke.yaml \
  --max-steps 1
```

冒烟通过后才能使用 `configs/parrad-md-gru-stage2.yaml` 启动正式 P2。两份配置都使用本地 `/home/xibeidaxue/SEER/Qwen2.5-0.5B-Instruct`、FP16 Qwen2.5-0.5B、前四层 q/v LoRA，以及 `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage1_causal_md_gru_pilot/radar_encoder.pt`。P2 仍是 14 类固定模板闭集语义对齐，不代表自由文本 caption 能力，也不应在 P2 探针通过前开始 P3 或分层记忆。

### Stage4 工程贯通试验（2026-10-04）

按用户补充的v2.5原方案实现了共享Query Compressor与L0/L1/L2事件记忆，入口为：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -m continuous_mmwave.parrad_stage4_runner \
  --config configs/parrad-stage4-memory-pilot-v2.yaml --mode all
```

这一轮经用户授权，先冻结原Stage1–3和LLM，训练记忆压缩器并贯通真实雷达到Full Qwen回答；上文旧的阶段门槛不阻止此工程试验。离线GT分段训练与自然在线控制器回放分开报告，不宣称Stage3 Go、硬实时或开放问答效果。配置、输出约定及限制见[Stage4首轮说明](docs/stage4-memory-pilot-2026-10-04.md)。

首轮已跑通，但自由问答未成功。后续混合记忆/chat对齐与事件问答监督实验继续训练压缩器500步，入口：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -m continuous_mmwave.parrad_stage4_qa \
  --config configs/parrad-stage4-chat-qa-pilot-v1.yaml
```

训练前后的固定证据问答、同问题正反证据、无记忆对照及自然在线回放分别报告，详见[Stage4问答实验说明](docs/stage4-chat-qa-pilot-2026-10-04.md)。

500步实验已完成：回答格式改善，但平衡存在性准确率54.8%、活动顺序0/7，配对问题未超过无记忆对照。完整原始预测和结论见[Stage4问答实验结果](docs/stage4-chat-qa-pilot-results-2026-10-04.md)。

随后新增独立的生成侧QA适配层，原事件/检索P/LLM保持冻结。单事件固定500步入口：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -m continuous_mmwave.parrad_single_event_qa \
  --config configs/parrad-stage4-single-event-qa-v1.yaml
```

单事件实验已完成：动作识别63.6%、平衡问答76.8%、配对均正确56.6%，超过无记忆但未达到预设开发目标。[训练说明](docs/stage4-single-event-qa-2026-10-05.md)与[完整结果](docs/stage4-single-event-qa-results-2026-10-05.md)保留基线、逐类失败和适用范围；尚未把该适配层接入长期记忆问答。

下一轮继续旧QA适配层，固定追加500次更新：单事件250、双事件125、四事件125。只训练生成侧适配层；训练后以同一四事件组和同一问题对比32个原token与4个冻结L1摘要token。

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -m continuous_mmwave.parrad_short_context_qa \
  --config configs/parrad-stage4-short-context-qa-v1.yaml
```

本轮已完成：单事件识别76.8%、macro80.5%、平衡问答84.3%、配对均正确74.7%；双事件部分改善，四事件顺序仍0/24，整体开发目标未通过。冻结L1对照也未解决顺序问题。见[短上下文QA说明](docs/stage4-short-context-qa-2026-10-05.md)与[完整结果](docs/stage4-short-context-qa-results-2026-10-05.md)。仍使用TRAIN内部295个事件，原检索路径不变，未自动追加预算、训练压缩器或启动在线回放。

事件编号对照继续同一QA起点，两组各500步、相同任务与采样，分别使用原始拼接与冻结文字编号。新增逐位置识别、加强完整顺序监督，入口：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -m continuous_mmwave.parrad_event_boundary_qa \
  --config configs/parrad-stage4-event-boundary-qa-v1.yaml
```

[事件编号QA方案](docs/stage4-event-boundary-qa-2026-10-05.md)规定每组200/100/200单/双/四事件步数、四事件完整顺序100步、各位置25步。只训练生成侧QA适配层，单事件和检索输入路径保持；无记忆对照为空上下文、冻结LLM，两组共享一次生成。

两组已完成各500步：编号组单事件macro82.1%、平衡问答90.4%、配对均正确80.8%，但双/四事件问答62.2%/56.2%，弱于原始拼接组81.6%/75.0%；完整四事件顺序两组均0/24。正确文字顺序对照也0/24，两组整体开发目标未通过。原检索路径保持，未追加训练或替换生产模型。[完整结果与逐类退步](docs/stage4-event-boundary-qa-results-2026-10-05.md)。

### Stage4 事件位置与相对时间编码对照

新增融合层只服务多事件生成，旧QA及原检索路径冻结，单事件绕过新层。none/order/time三组各500次多事件更新；先通过锁定纯文字顺序问答门槛才进入训练，门槛失败保留证据并停止。

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -u -m continuous_mmwave.parrad_temporal_qa \
  --config configs/parrad-stage4-temporal-qa-v1.yaml
```

预算、时间编码、数据与结论边界见[实验说明](docs/stage4-temporal-qa-2026-10-05.md)，真实状态见[执行记录](docs/stage4-temporal-qa-progress-2026-10-05.md)。

实现与服务器原生测试已完成，fit纯文字前置诊断未达门槛，实际优化0步。位置编码学习效果尚未验证，三组训练没有启动。[诊断结果与原始证据](docs/stage4-temporal-qa-results-2026-10-05.md)。

文字问答提示冲突修复后仍未达标，现新增独立QA LoRA固定500步入口，先训练正确文字证据下的存在性、位置与完整顺序回答。基础Qwen和原EventFormer/检索/旧QA权重保护，新的QA LoRA单独导出；此入口不启动雷达或时序融合训练。

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -u -m continuous_mmwave.parrad_qa_text_decoder \
  --config configs/parrad-stage4-text-decoder-v1.yaml
```

[文字问答修复方案](docs/stage4-decoder-repair-2026-10-05.md)与[执行进度](docs/stage4-text-decoder-progress-2026-10-05.md)记录锁定预算、95%门槛、真实生成和当前状态。仅文字任务通过不能代表雷达QA或原检索性能提升。

独立文字QA LoRA已完成500次真实更新：fit1014题、dev510题实际生成全部正确，规定门槛通过；额外重复/重排/相近名称110题109正确，保留一例位置混淆。原模型及检索路径未改，位置/时间编码训练尚未启动。[完整结果与原始核验](docs/stage4-text-decoder-results-2026-10-05.md)。


### Stage4 新文字QA接回雷达融合

多事件接入冻结的500步文字QA LoRA，单事件保留旧reader；none/order/time各500步，仅训练新增融合层。v3在第75步遇到FP16反向溢出，保留74步失败证据；v4加入同批减半loss scale重试，三组已完成共1500次有效更新。三组均未通过开发目标，四事件完整顺序均0/24；位置/时间编码未超过无编码组。单事件297条回答、原检索向量/分数/排序前后完全一致。

[执行方案](docs/stage4-reader-fusion-2026-10-06.md)与[实时执行记录](docs/stage4-reader-fusion-progress-2026-10-06.md)。本轮服务器30项相关原生测试全部通过；完整495项中493通过，原Gemma后端两项兼容性失败已在未修改的旧代码复现，未改原EventFormer或升级共享依赖。

恢复版相关34项原生测试通过，完整套件497通过、仅保留上述两项Gemma失败。交换对照核验将事件类别误当问题类别的错误已在独立脚本修复，9项服务器回归通过，全部保存权重/预测读回通过；训练源码及原始答案未修改。[完整结果](docs/stage4-reader-fusion-results-2026-10-06.md)。

### Stage4 单事件雷达语义适配

新增独立单事件入口直接使用冻结的文字QA reader；原8个雷达token通过全新残差适配层输入，不走旧单事件reader。ce与distill两组各500步，后者增加fit正确文字教师的答案分布蒸馏（0.5、温度2）；两组共同初始化/批次，最终500步固定选择，原检索和旧QA路径保护。

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src python -u -m continuous_mmwave.parrad_radar_alignment_train \
  --config configs/parrad-stage4-radar-alignment-v1.yaml
```

先验证单事件文字教师95%门槛，再执行fit/dev零步基线和两组训练；分别报告动作识别、正反问答及配对全对，训练后复测检索并独立读回全部权重/预测。失败及核验异常保留证据，开发未通过不自动进入多事件。[锁定方案](docs/stage4-radar-semantic-alignment-2026-10-06.md)与[执行记录](docs/stage4-radar-semantic-alignment-progress-2026-10-06.md)。

本轮结果：文字教师动作识别宏平均85.71%，正反判断100%，未达95%前置门槛；ce/distill实际0步，未执行雷达训练。冠词变体与“坐着/坐下”混淆已分别定位，完整原始预测归档及复算见[结果报告](docs/stage4-radar-semantic-alignment-results-2026-10-06.md)。

用户明确授权继续后，v2配置把文字门槛改为diagnostic-only，保留未通过记录并继续相同雷达CE/distill各500步；默认v1仍required。服务器相关24项测试通过，v2在2026-10-06 13:25启动独立后台worker，输出stage4_radar_alignment_v2。[v2方案与进度](docs/stage4-radar-semantic-alignment-v2-2026-10-06.md)。

v2结果已核验：两组各实际500步完成、原QA/检索精确保留。dev动作宏平均CE76.14%/蒸馏47.53%，正反平衡67.17%/66.16%；两组均未过开发门槛，CE仍未超过旧QA参考。完整归档复算与失败类别见[v2结果](docs/stage4-radar-semantic-alignment-v2-results-2026-10-06.md)。

# P0：冻结Align Encoder的分类与语言读出公平对比（2026-10-11）

**按validation完整窗口选择的分类流程为B1，完整12秒test序列准确率18.88%，低于同编码器B3的24.69%（分类减B3=-5.81个百分点）。**

本轮只训练两个Linear(256,14)头。Align Encoder完全冻结；现有8个有序Tokens、Projector和Qwen均未更新。B3复用上轮同滑动窗口输出，逐窗口重新读取RD/MD/时间戳并核验摘要后配对。

## 1. 固定协议与选择顺序

- 新训练特征：Align原best epoch3，eval/no_grad重新提取train受试者02/03/04与validation05；同上轮H1/H4非重叠12秒上下文，每个有效原始帧计一次。新特征与旧Align诊断缓存数值复核，训练不复用旧缓存。
- H1/H4沿用original_optimizer完整策略：20epochs、AdamW lr0.001/wd0.01、batch4096、clip1、seed17；均从相同原Stage1分类头初始化。H1不加权/不标准化；H4训练集逆频率权重与mean/std，标准化初始化保持原始logits。新头并非旧H1/H4权重直接挂载。
- 验证帧Macro-F1选权重，平分取最早epoch：新H1选epoch20，新H4选epoch8。选择用subject05，test06不参与。
- 评价重新提取滑动窗口Align特征，名义12秒/stride6秒/最短8秒/最多3个GT run，与原B3完全相同。原702val/484test各有2个不足12秒尾窗；严格700/482为主，原全集补充。没有从非重叠特征切片构造评价特征。
- 三个分类方法均用相同平滑函数、真实时间戳和9组候选：median0/.3/.5秒×minimum0/.3/.6秒，合并相邻同类，不限制GT动作数。主结果各自仅在完整validation上按序列exact→LCS→Recall选择；共同阈值敏感性也报告。整数class-ID中值存在类别顺序任意性限制。
- 权重与后处理于2026-10-11T13:02:28.118456+08:00锁定，此后才提取test滑动特征；本报告比较方案B1也仅按validation选择。所有训练统计来自train。

| 方法 | 独立validation阈值 median/min秒 |
| --- | --- |
| B0 Stage1 + H1 + 模板 | 0.0/0.6 |
| B1 Align + 新H1 + 模板 | 0.0/0.6 |
| B2 Align + 新H4 + 模板 | 0.0/0.6 |

共同阈值：median=0.0秒，minimum=0.6秒。

## 2. 相同完整12秒窗口：独立validation阈值主结果

| split | 方法 | 完整序列正确 | Precision | Recall | LCS | Count MAE | 预测数−GT均值 | 少/多/等数量窗口 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | B0 Stage1 + H1 + 模板 | 104/700=14.86% | 50.47% | 59.57% | 47.12% | 1.127 | +0.867 | 86/372/242 |
| validation | B1 Align + 新H1 + 模板 | 115/700=16.43% | 52.23% | 62.31% | 48.73% | 1.151 | +0.917 | 76/379/245 |
| validation | B2 Align + 新H4 + 模板 | 106/700=15.14% | 51.94% | 62.55% | 48.26% | 1.207 | +0.970 | 79/395/226 |
| validation | B3 Align + 原Projector/Qwen | 141/700=20.14% | 50.14% | 45.55% | 43.55% | 0.446 | -0.229 | 231/76/393 |
| test | B0 Stage1 + H1 + 模板 | 86/482=17.84% | 49.80% | 60.06% | 47.26% | 1.141 | +0.917 | 52/260/170 |
| test | B1 Align + 新H1 + 模板 | 91/482=18.88% | 55.02% | 65.91% | 52.25% | 1.098 | +0.857 | 57/253/172 |
| test | B2 Align + 新H4 + 模板 | 90/482=18.67% | 54.26% | 64.18% | 51.22% | 1.093 | +0.849 | 58/241/183 |
| test | B3 Align + 原Projector/Qwen | 119/482=24.69% | 50.52% | 46.06% | 45.37% | 0.336 | -0.282 | 148/13/321 |

![同编码器对比](../outputs/radar-text-align-readout-p0-20261011-v2/derived/align-readout-comparison.png)

Count MAE=mean|预测实例数−GT实例数|；符号偏差=mean(预测−GT)。负值表示更倾向少输出，正值表示多输出。数量相等不保证类别或顺序正确；少/多计数也不能自动定位具体时段的识别替代。

## 3. 原484窗口与共同阈值敏感性

| subset | 方法 | 阈值策略 | 序列准确率 | Recall | LCS | Count MAE | 少/多窗口 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| original_protocol | B0 | individual | 18.18% | 60.23% | 47.48% | 1.136 | 52/260 |
| original_protocol | B1 | individual | 19.21% | 66.05% | 52.44% | 1.093 | 57/253 |
| original_protocol | B2 | individual | 19.01% | 64.33% | 51.43% | 1.089 | 58/241 |
| original_protocol | B3 | 固定B3 | 25.00% | 46.28% | 45.59% | 0.335 | 148/13 |
| full12 | B0 | common | 17.84% | 60.06% | 47.26% | 1.141 | 52/260 |
| full12 | B1 | common | 18.88% | 65.91% | 52.25% | 1.098 | 57/253 |
| full12 | B2 | common | 18.67% | 64.18% | 51.22% | 1.093 | 58/241 |
| full12 | B3 | 固定B3 | 24.69% | 46.06% | 45.37% | 0.336 | 148/13 |
| original_protocol | B0 | common | 18.18% | 60.23% | 47.48% | 1.136 | 52/260 |
| original_protocol | B1 | common | 19.21% | 66.05% | 52.44% | 1.093 | 57/253 |
| original_protocol | B2 | common | 19.01% | 64.33% | 51.43% | 1.089 | 58/241 |
| original_protocol | B3 | 固定B3 | 25.00% | 46.28% | 45.59% | 0.335 | 148/13 |
| full12 | B0 | unsmoothed | 13.07% | 67.95% | 42.14% | 2.627 | 14/364 |
| full12 | B1 | unsmoothed | 13.90% | 73.76% | 44.12% | 2.929 | 12/368 |
| full12 | B2 | unsmoothed | 13.90% | 71.54% | 43.70% | 2.824 | 12/359 |
| full12 | B3 | 固定B3 | 24.69% | 46.06% | 45.37% | 0.336 | 148/13 |

## 4. Align是否提高线性可读性

第一张表使用完全相同非重叠12秒训练/评价上下文和训练控制，对照旧Stage1 H1/H4；第二张表是同滑动窗口的逐帧读出。两者上下文不同，不混用指标。Macro-F1仍不能直接等同于生成性能。

| split | 头 | Stage1非重叠Frame Accuracy / Macro-F1 | Align非重叠Frame Accuracy / Macro-F1 |
| --- | --- | ---: | ---: |
| validation | H1 | 47.67% / 45.65% | 51.01% / 49.75% |
| validation | H4 | 48.64% / 47.67% | 50.97% / 50.74% |
| test | H1 | 49.50% / 49.28% | 51.94% / 51.74% |
| test | H4 | 48.54% / 49.42% | 51.73% / 52.31% |

| split | Stage1 H1滑动Macro-F1 | Align H1滑动Macro-F1 | Align H4滑动Macro-F1 |
| --- | ---: | ---: | ---: |
| validation | 43.81% | 47.33% | 48.46% |
| test | 47.95% | 52.24% | 52.22% |

## 5. 逐类Recall与配对错误

完整12秒test动作实例多重集覆盖；重复同类按次数计。未作可靠时段对齐，故原类未覆盖可能包含分类替代，不能都解释为纯省略。

| 动作 | GT实例 | Stage1 H1 | Align H1 | Align H4 | Align Qwen |
| --- | ---: | ---: | ---: | ---: | ---: |
| falls onto the floor of the room | 126 | 50.79% | 56.35% | 56.35% | 43.65% |
| gets into bed | 82 | 47.56% | 74.39% | 76.83% | 17.07% |
| gets out of bed | 84 | 60.71% | 66.67% | 69.05% | 8.33% |
| lies in bed | 60 | 58.33% | 58.33% | 61.67% | 11.67% |
| rolls in bed | 66 | 50.00% | 48.48% | 48.48% | 42.42% |
| sits down on the bed | 58 | 53.45% | 55.17% | 68.97% | 29.31% |
| sits down on the chair | 108 | 66.67% | 72.22% | 65.74% | 62.04% |
| sits in bed | 56 | 50.00% | 62.50% | 62.50% | 50.00% |
| stands up from the bed | 54 | 59.26% | 66.67% | 87.04% | 22.22% |
| stands up from the chair | 104 | 60.58% | 75.96% | 67.31% | 73.08% |
| stands up from the floor of the room | 132 | 71.21% | 71.97% | 65.15% | 72.73% |
| walks toward the bed | 158 | 65.19% | 66.46% | 62.03% | 51.90% |
| walks toward the chair | 136 | 53.68% | 58.82% | 48.53% | 50.00% |
| walks toward the room | 128 | 59.38% | 64.06% | 61.72% | 46.09% |

| 分类流程 | 分类独对 | B3独对 | 两者都对 | 两者都错 |
| --- | ---: | ---: | ---: | ---: |
| B0 | 32 | 65 | 54 | 331 |
| B1 | 29 | 57 | 62 | 334 |
| B2 | 30 | 59 | 60 | 333 |

## 6. 结论与适用范围

- 完整12秒/individual：B3仍高于两个Align分类模板。
- 完整12秒/common：B3仍高于两个Align分类模板。
- 原全部窗口/individual：B3仍高于两个Align分类模板。
- 原全部窗口/common：B3仍高于两个Align分类模板。

**Align提高了该配方下的类别可读性，但分类模板仍未赶上B3的完整序列准确率。** H1非重叠test Macro-F1由49.28%升至51.74%（+2.46个百分点）；同滑动窗口序列准确率由17.84%升至18.88%，与B3的差距由6.85缩至5.81个百分点。这是本控制配方下的描述性差异，不能当作独立事件的显著性或LLM贡献分解。

**两条流程的错误结构不同。** Align H1完整test窗中多输出动作253个窗口、少输出57个；B3分别为13/148。B3的Count MAE更低，但仍存在明显动作覆盖损失。上床/下床/卧床这三类，Align H1覆盖Recall分别为74.39%/66.67%/58.33%，B3为17.07%/8.33%/11.67%。这支持优先检查池化、Projector、文本目标和解码中的证据保留，同时继续控制分类序列碎片与多余动作；本轮不能定位为某个单一模块的责任。

validation选出的B1在同编码器下，Recall=65.91%、LCS=52.25%、Count MAE=1.098；B3对应46.06%、45.37%、0.336。应按这些不同维度判断完整读出流程，不能将序列准确率等同于全面语言质量。

冻结同一Align编码器消除了本轮分类/语言比较的编码器权重差异；仍比较“分类头+平滑+模板”与“8Tokens+池化+Projector+Qwen”的完整读出流程，不能单独量化LLM本身因果贡献。特征可读性变化只对本训练配方成立，非所有可能分类上限。本轮未开展完整2×2。

限制：已有test subject06曾被观察；只有一名测试受试者，同步双传感器/重叠窗口相关，不能当独立物理事件作简单显著性结论。窗口通过GT≤3run筛选，动作duration是command区间；不代表任意长序列、真实短动作或跨环境泛化。短语匹配有同义表达/否定/词表外文本盲点，Recall是受限动作代理。

## 7. 核验与产物

Align全部参数和buffer逐位不变，梯度为空；Qwen未加载。原输入/权重/结果57项SHA前后不变，源码129项核验通过。重新提取40份特征，窗口输入摘要匹配validation702/test484。封存v2执行前相关测试本地23 passed、服务器67 passed（各1个既有依赖warning）；最终入口另增加跨pilot路径的Align/B3 SHA前置校验，修复回归后本地24 passed，原v2训练/推理封存包保持不变。

[机器结果](../outputs/radar-text-align-readout-p0-20261011-v2/experiment/comparison.json) · [训练曲线/逐帧指标](../outputs/radar-text-align-readout-p0-20261011-v2/experiment/head-training.json) · [validation选择](../outputs/radar-text-align-readout-p0-20261011-v2/experiment/threshold-selection.json) · [14类CSV](../outputs/radar-text-align-readout-p0-20261011-v2/derived/align-readout-per-class.csv) · [test逐窗口预测](../outputs/radar-text-align-readout-p0-20261011-v2/experiment/predictions-test.json) · [模型与输入核验](../outputs/radar-text-align-readout-p0-20261011-v2/experiment/verification.json) · [新H1权重](../outputs/radar-text-align-readout-p0-20261011-v2/experiment/heads/align-H1.pt) · [新H4权重](../outputs/radar-text-align-readout-p0-20261011-v2/experiment/heads/align-H4.pt)

服务器输出：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/radar_text_align_readout_p0_20261011_v2`。v1预检查失败记录保留，尚未进行参数训练；修复受试者检查范围后v2重新封存执行。原始雷达/Qwen不下载，特征留服务器；新分类头随结果包导出。

本地独立审计通过：27份原始文件SHA、10674条logits/时间戳重建分类序列、11860条逐窗序列主指标、40组汇总和27组validation候选。Count MAE、数量偏差及多/少窗口分解独立重算一致；B3文本与旧输出一致，旧输入摘要与Stage1 H1 logits一致。CSV含14类共560行；审计不替代人工文本事实性评价，也不声称独立重算全部派生字段。

[本地独立审计](../outputs/radar-text-align-readout-p0-20261011-v2/derived/local-verification.json) · [派生产物SHA清单](../outputs/radar-text-align-readout-p0-20261011-v2/derived/derived-manifest.json)

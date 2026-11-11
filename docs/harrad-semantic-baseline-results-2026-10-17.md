# HARrad 首轮语义基线结果（2026-10-17）

**六组训练全部完成。** 流程于北京时间10:19:20启动、11:01:14结束，含转换、核验、拟合预检及正式训练，总计41分52秒。GPU现已空闲。

两模型各3个seed，每次30完整epochs，共180epochs、21,330次成功更新、543,240次训练实例暴露。MD/RD小样gate分别在10/40步达到100%准确率，gate权重没有用于正式初始化。

## 验证集结果

以subject1–6的3,018段训练，在subject7的511段上每轮验证。每次独立选取validation Macro-F1最高且最早epoch；以下是各自best，**不是统一第30轮，也不是测试集结果**。

| 模型 | seed42 | seed43 | seed44 | 平均Macro-F1 ± seed总体标准差 |
| --- | ---: | ---: | ---: | ---: |
| MD-GRU | 79.86% | 77.73% | 79.09% | **78.89% ± 0.88个百分点** |
| RD-CNN-GRU | 85.87% | 83.59% | 84.10% | **84.52% ± 0.98个百分点** |

| 指标（三seed均值） | MD-GRU | RD-CNN-GRU |
| --- | ---: | ---: |
| 12类accuracy | 75.28% | 81.15% |
| 6类Events组内Macro-F1 | 92.87% | 96.72% |
| 6类Gestures组内Macro-F1 | 67.78% | 73.83% |
| 每seed正式训练时长（含逐轮train/val评估） | 96.25秒 | 397.96秒 |
| 参数量（全部可训练） | 134,284 | 132,732 |

组内F1仅在真实标签属于该组的样本上计算，预测到另一组仍计作错误；它不是全体样本上抽取六个类别F1的同一口径。12类整体F1为主指标。

RD的主指标在三个seed分别领先6.01、5.87、5.01个百分点，均值领先**5.63个百分点**。当前结果支持在这一固定按人划分上，距离—多普勒联合结构比距离汇总后的MD更有用；两路都能在官方实例监督下学到较强的语义分类。手势仍明显弱于Events。

## 核验

- 35条train/validation录制、215,482帧转换完成；官方FFT36个训练样本逐值对照误差0，全部canonical缓存哈希、形状、有限值、原始帧号与标签/实例数组核验通过。
- 六个best检查点与完整完成记录的SHA256吻合；每组30轮日志和30份validation逐样本预测齐全。
- 服务器在正式结束时逐组重载best，validation logits最大绝对误差全部为**0**，并用独立指标实现核对历史best。
- 已取回所有initial/best/last权重、全部轮次预测、日志、配置与gate结果；255个导出文件大小及SHA256核验通过。
- 本地再次对180份逐轮预测及6份best预测，独立复算混淆矩阵、Macro-F1、accuracy、CE、组内及逐人结果，**186份全部通过**。
- 此次只读取train/validation信号，subject8–9测试信号未参与本轮训练、归一化、选择或指标报告。此前格式抽样的限制见启动记录。

## 结论范围

这是**完整动作片段的离线12类分类基线**；每段初始化状态，片段长度由官方起止标注提供。尚未训练或评价EventFormer、预测分支、双状态联合模型、LLM、自主Reset或未知段检测。

验证仅subject7一人，三个seed不等于三个独立受试者实验。当前数值是模型选择所用验证表现，不能当作未见测试人的最终泛化成绩。两模型使用相同epochs和实例暴露，但batch、优化步数、输入信息及计算量不同，不能把差异全部归因于一个网络层。与旧PARrad的任务及数据不同，也不应直接作百分比提升比较。

建议以RD分支作为下一轮的主要输入候选，保留MD对照；下一阶段再设计真正的语义状态与预测分支实验。本次检查没有自动追加训练或打开测试集。

## 证据与路径

- 服务器结果：`/home/xibeidaxue/newdisk2/WJ/HARrad/outputs/semantic_baseline_20261017_v2`。
- 服务器源码：`/home/xibeidaxue/newdisk2/WJ/HARrad/experiments/semantic_baseline_20261017_v2`。
- [本地独立核验结果](../outputs/harrad-semantic-baseline-20261016-v1/finished-v2/local-verification.json)。
- [全部导出文件](../outputs/harrad-semantic-baseline-20261016-v1/finished-v2/semantic_baseline_20261017_v2_export_v1/extracted/)。
- 导出包25,639,287字节，SHA256 `de482e0fb9e1afa9ffd6fca505f1fba7927c24c1e1eff73aafccd175067490a5`。
- [启动与协议](harrad-semantic-baseline-progress-2026-10-17.md)，[固定配置](../configs/harrad-semantic-baseline-v1.json)。

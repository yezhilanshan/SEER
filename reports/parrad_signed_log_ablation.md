# PARrad Signed-Log P0/P1 严格受控消融

## 结果检查结论（2026-10-14）

**删除 Signed-Log 没有提高本轮主指标。** P0/P1 的 Validation 受试者平均 Macro-F1 分别为 **58.268% / 58.118%**，P1−P0 为 **−0.150 个百分点**。Pooled F1 为62.076% / 61.809%，下降0.267个百分点；床相关F1为64.279% / 64.502%，略升0.223个百分点。

受试者方向不一致：05提高0.143、12提高0.725、16下降1.317个百分点；设备方向也不一致：xWR14xx下降0.791、xWR68xx提高0.231个百分点。未出现跨对象的一致收益。

类别变化具有取舍：get into bed +1.641、get out of bed +2.353、lie in bed +2.022个百分点；sit down on bed −3.026、stand up from chair −6.796、stand up from floor −4.132个百分点。不能把少数床动作改善解释为所有静态/低速动作都改善。

预设追加阈值为主指标绝对差1个百分点，本轮未触发，所以没有训练seed29/43。当前只有一个配对种子，不能报告三种子Mean±Std，也不能宣称两种预处理统计等价。

两臂初始tensor及共享初始化文件、20轮完整采样序列、训练/验证窗口身份、类别权重和数值配置核验一致；都是20epochs、相同名义更新预算。**AMP溢出跳步为P0=4、P1=5，成功更新为24,856/24,855**，因此实际成功更新数差1步，不能声称有效更新次数完全相同。未改变超参数消除它；CUDA位级确定性仍不保证。

P0最佳第11轮，耗时2.095h；P1最佳第10轮，耗时1.950h。按原有规则分别选优，并非使用训练终点或历史P0分数替代。

**建议：保留现有方案作为基线，不因这次输入尾部现象全面重训Stage1、Align或LLM；本轮也没有足够收益证据优先推广至RD+MD。** 若后续研究仍需检验双分支，尤其MD分支，应另做同样的受控试验，不能由RD-only结果推断其必然相同。标签时钟、域差异等仍是独立未解决问题。

服务器训练与历史数据未改动；本地已取回两组best.pt、指标、日志及公平性记录，保存在 [完成结果目录](D:/lab/SEER/outputs/parrad_signed_log_ablation_20261013_v1/completed-20261014-v1)。源配置/函数行号见[source-lines.json](D:/lab/SEER/outputs/parrad_signed_log_ablation_20261013_v1/completed-20261014-v1/source-lines.json)，新的本地核验记录见[local-verification.json](D:/lab/SEER/outputs/parrad_signed_log_ablation_20261013_v1/completed-20261014-v1/local-verification.json)。


状态：**complete**。开始于 2026-10-14T00:27:30.347784+08:00。只使用 Train/Validation；测试集不参与训练、评价或方案选择。

## 实验目的与固定设置

检验额外 Signed-Log 是否影响 RD-only ResNet18 + BiGRU + 14 类逐帧分类；不改变网络或追求新最优方案。P0 为 Signed-Log + 有效窗口 Z-score；P1 仅移除 Signed-Log，使用官方发布 RD 原值做相同 Z-score。

数据与原 Stage1 一致：train01/02/03/04/07/08/09/13/19，validation05/12/16，test06/17/18；窗口12s/stride6s/min8s/max3动作，不改标签、不跨录制，不裁剪、不滤波、不限幅、不训练 MD/LLM。

每臂20 epochs，无 early stopping；batch2/累积2，AdamW，encoder lr=0.0003、classifier lr=0.001、weight_decay=.01、clip1，CosineAnnealingLR eta_min=1e-6；相同训练窗口计算的类别权重、FP16 AMP、frame_chunk64、dropout=.1。以 Validation 受试者平均 Macro-F1 选优，平分依次看 pooled Macro-F1、较低未加权 CE。

每个 seed 两臂使用同一保存的初始 tensor/RNG 文件，核验整个 state_dict（含 BN buffer）内容 SHA256；每 epoch 采样顺序有独立 RNG 并保存全序列与散列。数值配置保持一致，但不声称 CUDA 位级确定性。AMP 跳步属于运行结果，逐臂记录，不通过改超参数消除。

seed17 的主指标绝对差达到预先约定的1个百分点才追加29/43；未达到则不把多帧数量当作独立样本宣称显著。

## 代码修改

`rd_transfer.normalize_rd` 增加默认 signed_log_zscore / 可选 raw_zscore；`rd_scratch` 传递并保存模式，旧 checkpoint 缺失模式时默认 P0。训练入口增加共享初始化和 Validation-only 选项。辅助模块核验参数/采样顺序，按完整录制标签排除转换点±0.3s后再切窗。模型选优仍使用全部有效帧，steady 指标仅作诊断。

执行源码版本见每臂 provenance.json 的 source_sha256；具体函数行号见 source-lines.json。单元测试和 Smoke Test 记录见上级启动归档；不能用 Smoke 指标充当正式结果。

## Seed 17

| 指标 | P0 | P1 | P1−P0 (百分点) |
| --- | --- | --- | --- |
| 主指标：受试者平均 Macro-F1 | 58.268% | 58.118% | -0.150 |
| Pooled Macro-F1 | 62.076% | 61.809% | -0.267 |
| Pooled Accuracy | 61.635% | 61.501% | -0.133 |
| Pooled Macro-Precision | 63.547% | 62.546% | -1.001 |
| Pooled Macro-Recall | 61.522% | 61.776% | +0.254 |
| Pooled Steady Macro-F1 | 63.235% | 62.940% | -0.295 |
| 床相关 Macro-F1（1/2/3/4/5/7/8） | 64.279% | 64.502% | +0.223 |

Validation未加权CE：P0=1.630236，P1=1.586792，差=-0.043444。

### 逐受试者

| 对象 | P0 Macro-F1 | P1 Macro-F1 | 差 (百分点) |
| --- | --- | --- | --- |
| subject_05 | 74.223% | 74.366% | +0.143 |
| subject_12 | 61.852% | 62.578% | +0.725 |
| subject_16 | 38.728% | 37.411% | -1.317 |

### 逐设备

| 对象 | P0 Macro-F1 | P1 Macro-F1 | 差 (百分点) |
| --- | --- | --- | --- |
| TI_xWR14xx | 62.774% | 61.983% | -0.791 |
| TI_xWR68xx | 61.380% | 61.610% | +0.231 |

### 14 类评价

| 类 | 动作 | P0 P/R/F1 | P1 P/R/F1 | F1差 (百分点) |
| --- | --- | --- | --- | --- |
| 0 | falls onto the floor of the room | 71.160% / 60.360% / 65.317% | 63.777% / 68.777% / 66.183% | +0.866 |
| 1 | gets into bed | 82.150% / 61.580% / 70.393% | 82.673% / 63.821% / 72.034% | +1.641 |
| 2 | gets out of bed | 67.601% / 68.438% / 68.017% | 73.187% / 67.761% / 70.369% | +2.353 |
| 3 | lies in bed | 44.600% / 55.896% / 49.613% | 45.441% / 59.784% / 51.635% | +2.022 |
| 4 | rolls in bed | 76.458% / 75.325% / 75.887% | 75.747% / 76.057% / 75.902% | +0.014 |
| 5 | sits down on the bed | 74.978% / 59.957% / 66.631% | 59.875% / 67.832% / 63.606% | -3.026 |
| 6 | sits down on the chair | 60.437% / 66.954% / 63.529% | 65.284% / 66.900% / 66.082% | +2.553 |
| 7 | sits in bed | 62.253% / 64.552% / 63.381% | 60.947% / 64.317% / 62.587% | -0.795 |
| 8 | stands up from the bed | 66.931% / 48.187% / 56.033% | 66.050% / 47.681% / 55.382% | -0.651 |
| 9 | stands up from the chair | 58.400% / 59.906% / 59.143% | 55.886% / 49.229% / 52.347% | -6.796 |
| 10 | stands up from the floor of the room | 68.606% / 70.509% / 69.544% | 60.382% / 71.357% / 65.412% | -4.132 |
| 11 | walks toward the bed | 56.543% / 66.058% / 60.931% | 63.306% / 59.296% / 61.236% | +0.305 |
| 12 | walks toward the chair | 49.211% / 41.610% / 45.093% | 48.204% / 46.369% / 47.269% | +2.176 |
| 13 | walks toward the room | 50.334% / 61.969% / 55.549% | 54.885% / 55.677% / 55.278% | -0.270 |

P0 Best Epoch=11；总耗时=2.095h；AMP跳步=4；成功optimizer steps=24856。

P1 Best Epoch=10；总耗时=1.950h；AMP跳步=5；成功optimizer steps=24855。

### 重点类别混淆变化

| GT | 预测 | P0帧次 | P1帧次 | 差 |
| --- | --- | --- | --- | --- |
| stands up from the chair | stands up from the floor of the room | 927 | 2032 | 1105 |
| stands up from the chair | walks toward the bed | 1863 | 1055 | -808 |
| stands up from the floor of the room | walks toward the bed | 1344 | 591 | -753 |
| gets into bed | sits down on the bed | 534 | 1208 | 674 |
| stands up from the floor of the room | walks toward the chair | 1379 | 1988 | 609 |
| sits down on the bed | walks toward the bed | 1763 | 1246 | -517 |
| gets into bed | lies in bed | 3001 | 2530 | -471 |
| lies in bed | walks toward the bed | 437 | 10 | -427 |
| stands up from the chair | walks toward the chair | 735 | 1146 | 411 |
| lies in bed | stands up from the floor of the room | 248 | 645 | 397 |
| lies in bed | gets out of bed | 1115 | 745 | -370 |
| stands up from the chair | sits down on the chair | 1341 | 1710 | 369 |
| stands up from the bed | sits down on the bed | 920 | 1242 | 322 |
| gets out of bed | stands up from the floor of the room | 414 | 102 | -312 |

这里统计的是滑动窗口中的帧次，包含重复观测；混淆减少不等于对该动作的因果机制已确认。

公平性核验：{"initial_parameters_identical": true, "initial_file_sha256": "236eee6ca0ccd80c3233f4d5f96a3462a79344241cab88952a660717ef2603aa", "initial_parameter_sha256": "56b7c98f6e11fd96ffdeb737bbffe6b54e6b7bc7e1d57c00b043fad6ee47dddf", "all_epoch_sample_orders_identical": true, "paired_epochs": 20, "class_weights_identical": true, "numerics_identical": true, "test_values_accessed": false, "bitwise_cuda_determinism_claimed": false}

![训练与验证曲线](D:/lab/SEER/outputs/parrad_signed_log_ablation_20261013_v1/completed-20261014-v1/assets/curves.png)
![14×14混淆矩阵](D:/lab/SEER/outputs/parrad_signed_log_ablation_20261013_v1/completed-20261014-v1/assets/confusion.png)

## 机制解释与最终判断

此前输入审计发现 Signed-Log 会压缩强值对主体的对比，也会使极少数低值相对主体更突出（约−50.7σ）。本试验只能估计整个该变换的净影响，不能单独把性能差异归因为极端尾部；需要其他独立机制消融才能作因果分解。

| seed | P0主指标 | P1主指标 | 配对差 (百分点) |
| --- | --- | --- | --- |
| 17 | 58.268% | 58.118% | -0.150 |


受试者跨种子方向

| 对象 | 平均配对差 (百分点) | P1更高的种子数 |
| --- | --- | --- |
| subject_05 | +0.143 | 1/1 |
| subject_12 | +0.725 | 1/1 |
| subject_16 | -1.317 | 0/1 |


设备跨种子方向

| 对象 | 平均配对差 (百分点) | P1更高的种子数 |
| --- | --- | --- |
| TI_xWR14xx | -0.791 | 0/1 |
| TI_xWR68xx | +0.231 | 1/1 |

床相关 Macro-F1 的已完成种子平均差为 +0.223 个百分点；具体混淆和逐类变化见上表，不能把总体收益自动解释为床相关动作全部改善。


已完成种子中平均配对差为 -0.150 个百分点。逐受试者/设备方向以各表为准，不以相关帧数宣称统计显著。

当前未见达到预设幅度的主指标收益，不能确认 Signed-Log 是主要瓶颈；不据此重训双分支、Align 或 LLM。

## 结果限制

仅验证现有命令弱标签与当前窗口协议；标签时钟起点、发令反应时延、环境元数据和不同设备物理速度轴风险保持原样，未在两臂间改动。Test未读取。subject仅3名、设备2种，帧和滑动窗口高度相关；不把相关帧视为独立统计样本。

服务器自动绘图失败（/usr/bin/python3缺少NumPy）；本次已在本地interview环境补齐全部曲线及混淆矩阵。原始错误记录保留在下载归档中。

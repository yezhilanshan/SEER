# HARrad语义对齐 × 事件引导调制：实施与启动记录

2026-10-21 17:54:40（北京时间），v2正式任务已在服务器后台启动。进程PID为19544，首组为B0-seed42；18:01下载的状态已完成20次成功更新、20次尝试，无AMP跳步，train512/validation512零步预测及独立指标核验均已保存。本文记录实施、预检和启动，不代表训练已结束或方法有效。

## 已完成：2026-10-22 10:39

12组全部完成，12000成功更新/12000尝试。结果、48检查点核验、78预测面板独立复算及源文件SHA复核见[完整结果](harrad-alignment-modulation-results-2026-10-22.md)。下面保留原始进度记录，不能作为当前状态。

## 历史进度：2026-10-22 10:24

进程持续运行，无failed.json。11/12组已完整结束，均为1000成功更新/1000尝试，final重放差值0、冻结参数未变、独立指标核验通过。最后一组B_both-seed44为980/1000成功更新、980次尝试；总计11980/12000次成功更新（99.83%）。这只是优化更新进度，最后一组仍需1000步train/validation评价、final重放和关闭调制诊断，预计另需约15–20分钟，非完成承诺。

快照保存在outputs/harrad-alignment-modulation-20261021-v1/progress-20261022-1024.json。最终三seed四组比较和全组产物归档核查尚未完成。

## 历史进度：2026-10-21 20:00

进程运行，无failed.json。B0-seed42已完整结束：1000成功更新/1000尝试，用时77分17秒；final重放差值0、冻结参数未变、独立指标核验通过。当前B_align-seed42为630/1000成功更新、630次尝试；其余10组待执行。完整结束1/12组，成功更新总数1630/12000（13.58%）。

B_align的300步验证分录制Macro-F1为58.13%，对应B0的300步为58.82%；B_align尚无1000步结果，不能据此判定最终对齐收益。快照：outputs/harrad-alignment-modulation-20261021-v1/progress-20261021-2000.json。

## 历史进度：2026-10-21 19:10

后台进程仍运行，无failed.json。B0-seed42已完成1000/1000次成功更新、1000次尝试，没有AMP跳步；0/300/1000步的train/validation评价已保存，final.pt已写出，目前进行最终重载重放。完整任务计数仍为0/12，另11组等待顺序执行；优化更新总进度1000/12000（8.33%），不能将其等同于包含评价耗时的总时间进度。

首组基线在固定validation512面板上的分录制Macro-F1：零步64.52%、300步58.82%、1000步62.84%；1000步accuracy63.09%。这只是单seed基线，未超过自身初值，尚无alignment/modulation对照结果，不能判断两种方法的贡献。完整快照保存在outputs/harrad-alignment-modulation-20261021-v1/progress-20261021-1910.json。

## 本轮回答的问题

分别检验跨层对比监督和事件引导的前向调制能否改善HARrad事件语义表示。采用2×2对照：

| 组别 | 分类CE | 固定雷达参考对比 | 上一事件状态调制当前Radar Tokens |
| --- | --- | --- | --- |
| B0 | 有 | 无 | 无 |
| B_align | 有 | 有 | 无 |
| B_mod | 有 | 无 | 有 |
| B_both | 有 | 有 | 有 |

seeds42/43/44各1000次成功更新，共12000次。相同warmup、相同原实验1000×2采样表、相同train512/validation512诊断面板；0/300/1000步保存检查点与逐点预测，不选择best替代固定预算比较。

四组共同训练TokenAdapter、EventFormer新增Cross-Attention/Gate/输出投影/base tokens及12类reader。只有alignment组训练新增E投影，只有modulation组训练新增调制模块。CNN使用已锁定的冻结特征，Qwen含LoRA冻结；本轮不加载或训练Dynamics A，不启动自主Reset。

实际白名单中B0训练7,287,313个参数；对齐投影增加231,424个，调制增加207,936个。因此B_align为7,518,737，B_mod为7,495,249，B_both为7,726,673。本轮为功能2×2对照，未做参数量完全相同的容量对照。

## 如何改造DSA-Net思想

固定参考为warmup TokenAdapter的冻结副本：只汇聚当前已观察前缀的Radar Tokens，得到256维归一化向量。E经896→256投影，与自身因果前缀参考为正对；384个train参考（12类×32）中仅不同类作负例，同类全部排除。τ=.1，L=CE/log12+0.1L_clc；非alignment组只有CE/log12。未知/混合Update不参与语义监督。

调制分支以当前Z查询上一状态E，生成有界FiLM残差，末层零初始化；只在调制conditioning处detach旧E，保留正常递归的截断反传。官方起点Oracle前缀、最长32 Update、最后16 Update反传。没有直接复制论文的时间Gram或cycle losses，也没有强迫Dynamics与Semantic原始状态相等。

## 预检与修复证据

- 本地interview相关测试：22 passed、1项CUDA-only skipped；服务器seer/CUDA：23 passed，包含真实FP16状态回归。
- 210份封存文件逐项SHA核验通过。4组真实Qwen最大32前缀零步logits与原B差值均0；填充/未来扰动差值0；冻结参数不变。
- CE可传回TokenAdapter及调制输出层，CL可传回学生EventFormer/TokenAdapter；固定参考与Qwen无梯度。预检峰值CUDA allocated为1,194,207,744 bytes，约1.11 GiB。
- 本地独立重算通过：三seed完整采样表与原设计逐项一致，训练/验证面板各512点且标签来自相同官方实例；参考库384点无重复、全部train、每类32，向量及文件SHA一致；未读取test信号。
- 真实预检绑定完整config和source manifest SHA；正式运行复核后才使用已准备的参考库，启动器另核对全部封存文件。

v1原生旧20项测试通过，但真实预检因FP16 Qwen状态进入FP32 LayerNorm而失败，正式训练0步；独立CUDA回归复现后在新模块中增加float32转换。v1源码、native-tests、prepare日志及failed.json保留，不覆盖。

独立整体审查另发现旧JointModel会在burn=0时detach初始base tokens，造成短前缀无法更新它。短前缀梯度回归先失败，修复后通过；只修复新am模型、四组一致。零步前向仍等于旧B，但本轮B0优化行为已纠正，必须使用新B0作配对基线，不能用旧B结果替代。另补预检绑定、原始CE和不同类负样本余弦，均有先失败后通过的回归及独立指标复算。

本地全suite首次因旧test_parrad_convert依赖h5py缺失而收集失败；仅排除该文件的补充全suite长时间没有产生输出，主动中止，未声称全suite通过，未安装依赖。本次任务相关测试在本地与服务器均已完成。

## 位置与复核

- 源码：src/continuous_mmwave/harrad/am_{modules,model,data,training,audit,runner}.py。
- [完整设计](superpowers/specs/2026-10-21-harrad-alignment-modulation-design.md)、[实施清单与修复记录](superpowers/plans/2026-10-21-harrad-alignment-modulation.md)。
- 本地证据：outputs/harrad-alignment-modulation-20261021-v1/native-v2；失败证据在native-v1-failed；源码封存在capsule-v1/capsule-v2。
- 服务器封存源码：/home/xibeidaxue/newdisk2/WJ/HARrad/experiments/alignment_modulation_20261021_v2。
- 服务器正式输出：/home/xibeidaxue/newdisk2/WJ/HARrad/outputs/alignment_modulation_20261021_v2。
- v2源码zip SHA256：edaa9272f2c26b4e93d528fdbe5104f573ddb7f0380e02fe289671c8058a0b5e。
- 独立预检结果：本地native-v2/independent-preflight-audit.json；可运行outputs/harrad-alignment-modulation-20261021-v1/verify_preflight.py重核。

训练完成后应逐组验证1000次成功更新、冻结hash和final重放，再比较三seed分录制Macro-F1、训练/验证差距、前缀长度分组、正负余弦及调制关闭诊断。四组差分用于区分单独效果与交互作用。固定验证受试者、Oracle有限前缀的结论不能直接推广到自主事件边界或语言描述。

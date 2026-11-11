# HARrad因果语义对齐与事件引导调制：2×2实验

用户于2026-10-21授权开始前述四组对照。目的：分别验证跨流训练约束和前向调制对事件语义的贡献；不是DSA-Net复现，不训练自主Reset，不重新训练Dynamics A。本轮优先采用跨层对比思想，不引入Gram关系或循环logits损失。

## 输入、初始化与冻结范围

复用`configs/harrad-joint-prediction-v1.json`的5项源锁、语义warmup best、冻结Qwen（含LoRA）digest。完整train subject1–6、validation subject7缓存；test不读取。复制原三seed的1000×2起点表、512点语义验证面板（原1005联合面板的semantic标志），另固定512点训练语义诊断面板。每2帧Update、标称15fps；同一已知实例才有语义标签，保留原Oracle最近起点、最长32 Update、最后16步反传。未知/混合样本不进入CE或对比损失。

四臂B0/B_align/B_mod/B_both，seeds42/43/44各1000成功更新。四组都从相同warmup初始化并训练B已有TokenAdapter、Former新增Cross-Attention/Gate/投影/base tokens和分类reader。CNN、Qwen、参考Adapter和独立A全部冻结；不只训练投影头而宣称E被改善。B0为同协议继续语义训练基线，不与旧实例末端77.01%直接比较。

## 固定因果雷达参考与对比监督

参考分支是warmup TokenAdapter的冻结副本。对每个相同可见前缀[s,t]的所有4-token输出取均值，再无仿射LayerNorm及L2归一化，得到r_t∈R256。它只读取已观察双帧，不使用未来或整实例末端。与预测的m64/D状态不做强制相同化。

学生q_t=normalize(Linear896→256(LayerNorm(mean_slots(E_t))))。新增E语义投影的初值跨臂/seed一致，只有alignment臂训练该投影。对比参考固定，不随训练目标移动。

固定负样本库每类最多32个train合法起点，优先轮询各录制抽取，共最多384；规则种子314159，保存逐点rid/t/label与参考向量。标签只用于训练采样/排除负例，不进入模型前向。每query唯一正例为自身前缀r_t；库中同类全部排除，不把同类不同实例直接当负例。必须12类有支持、每query有至少一个不同类参考；不能混入validation/test或fallback伪负例。

L_clc=(logsumexp([q·r/τ,q·bank_neg/τ])-q·r/τ)/log(1+N_neg)，τ=0.1。先按batch内录制归约，再平均；未知标签不参与。归一化只统一数值口径，不声称已最优。B_align/B_both使用L=CE/log12+0.1 L_clc，其余只有CE/log12，记录两项原始量。

这是固定雷达锚点、单向因果跨层对比的改造，不冒称照搬论文原式或已验证语义充分性。分类监督和固定参考共同约束，仍需检查塌缩/正负相似度。

## 前向调制

当前Z_t(4×256)查询上一状态E_(t-1)(8×896)，Q/K/V经LayerNorm投影到64维；cross-attention上下文经64→128→512 MLP生成γ/β。末层零初始化，Z'=Z+0.1[tanhγ⊙LN(Z)+tanhβ]，保证零步恒等且缩放有界。conditioning E.detach仅切断调制支路对旧状态的梯度；原Former的递归梯度路径仍保留。调制只在B输入侧，A没有变化。

每个新Oracle前缀从P_base开始，首步调制条件同样是P_base；不缓存旧实例E，不允许Et指导同一步生成Et造成代数环。逐长度分组，右填充不影响输出，burn-in禁用AMP权重缓存以防后缀梯度被切断。

## 预算与评价

AdamW lr1e-4 wd0.01 clip1、batch2、AMP scale256、每批最多3次尝试；只有成功更新计步。4臂×3seed×1000=12000。固定保存0/300/1000并评价train512/validation512，final重载后逐值重放；不选best。每录制Macro-F1/accuracy后等权平均，另报pooled与类支持。

按可观察前缀长度分组1–4、5–16、17–32 Update，作为当前输入证据量分组，不称真实事件年龄（截断会重新初始化）。final另对mod臂在同一512验证面板关闭调制作配对诊断，不能视为独立训练消融。先不增加错配E训练或额外架构臂。

预检：本地/原生测试、四臂零步同前缀logits等于原B、CE到调制末层的梯度、CL到E/Adapter的梯度、参考/Qwen无梯度与hash不变、未来/填充扰动、Reset不携带旧条件、相同初值/起点、真实Qwen最大32上下文显存<9GiB。预检失败不缩小预算/上下文蒙混通过。

源/数据/原实验不覆盖；独立新源码胶囊和输出`alignment_modulation_20261021_vN`。v1仅作预检且失败，完整保留；修复后使用v2重新预检和启动。旧脚本不改，新增am_前缀模块复用只读数据/指标工具。结果仅为单验证受试者、Oracle有限前缀分类，不等于连续自主事件或语言能力。

四臂统一修复旧JointModel在burn=0时误detach初始P_base的问题：仅有burn-in时在反传段起点detach。零步前向仍与原B完全一致，但优化行为已纠正，因此本轮B0是重新训练的配对基线，不能直接用旧B训练曲线替代。新增调制的LayerNorm显式接收float32状态以兼容冻结FP16 Qwen。正式启动前比较预检的完整配置SHA和封存源码manifest SHA，逐文件SHA由启动器另行核验。

# HARrad Dynamics与语义联合预训练

用户授权继续补齐Dynamics A及A-only/B-only/A+B可比实验。本轮回到v2.6.1的预测/语义双状态预训练，不扩展监督边界头、不启动Keep/Birth或闭环Reset。

## 问题与可训练接口

复用HARrad RD seed42前端和已完成的语义warmup seed42 best。CNN固定；每帧4×64 cells经**同一个可训练RadarTokenAdapter**，每2帧产生4×256 token。A取token均值经新GRU256形成动力学状态D，预测未来固定CNN运动目标；B用已有EventFormer及12类读出形成E。A/B共享的是token表征，并非宣称本轮已经联合微调整个CNN。

A-only训练TokenAdapter、GRU与残差预测头；B-only训练同初值TokenAdapter、EventFormer新增Cross-Attention/Gate/投影/初始tokens及分类读出；A+B训练两组分支与共享TokenAdapter。Qwen（包括LoRA）全部冻结，但保留输入梯度；不冻结整个B分支。三个seed复用同一已有预训练来源，只表示新增训练随机性的三seed对照。

## 数据和上下文

train subject1–6共31录制190,238帧；validation subject7共4录制25,244帧；不读test信号。读取已校验的完整cells与frame_features缓存，不用oracle_hidden作预测输入。

每2帧一Update，标称周期2/15秒；舍弃各录制最后不足2帧的尾包以保持预测时距一致，逐录制列出舍弃数。m_t为两帧冻结CNN全局64维特征均值，经train-only逐坐标标准化；预测h=1/3/5 Update，即0.1333/0.4/0.6667秒，不沿用PARrad时间。

三Horizon使用同一合法起点集合（t+5仍在录制内），不因跨动作边界或未知标签删除预测目标。当前两帧必须属于同一个已标注实例，才能给语义CE标签；同类不同实例的混合Update也屏蔽语义。观测与语义mask分开。

本轮是Oracle初始化预训练：过去及当前已观察到的官方实例起点映射到所在Update，控制新增D/E的初始化路径；从当前起点回看最多32个Update，开始位置=max(录制起点、最近实例起点、t-31)。这是训练/评价的特权路径，不是自主边界结果。历史窗口截断会再次初始化状态；年龄仅按已可见帧计，不使用总实例长。每个优化步重算前缀，前max(L-16,0)个Update无梯度burn-in，其余最多16步保留梯度；不跨优化步缓存学生状态。GRU D是本阶段的可重置局部状态，尚无全局/局部双Dynamics或可训练Reset Policy。

## 损失和对照

残差预测hat_m_(t+h)=m_t+delta_h(D_t,m_t)，输出3×64；末层零初始化，零步精确Persistence。R_h由全train共同合法起点的等录制Persistence MSE固定，L_A=sum_h (0.1,0.3,0.6)_h MSE_h/R_h，batch按录制归约。L_B为有效当前类别CE/log(12)，不添加不存在的速度或Caption监督。

A-only用L_A；B-only用L_B；A+B用L_A+L_B。首轮不加教师蒸馏或动态权重，避免混入另一项变化；已有语义能力的保护通过共同初值、语义训练项及同协议前后评价检验，不宣称语言/检索接口已受保护。

额外Motion-MLP以固定当前m64为输入，64→256→192零末层，同样训练L_A；另计算Persistence与无调参线性趋势m_t+h(m_t-m_(t-1))。这些是强预测对照，容量/FLOPs不冒充匹配。报告各臂实际可训练参数。

## 固定预算和评价

4训练臂M/A/B/AB × seeds42/43/44 ×1000成功更新，batch2，共12000成功更新。每seed固定同一1000×2表，每步一个均匀抽录制后抽语义合法起点、一个均匀抽录制后抽全部物理合法起点；同一记录重复的两个样本不重复加权该记录。AdamW lr1e-4、wd0.01、clip1；AMP初始scale256，每batch最多3次尝试。保存0/300/1000，固定比较final1000，不用validation选checkpoint。

validation预先固定两个面板：每录制均匀抽128个物理合法起点（预测主面板512），另独立抽128个语义合法起点（语义主面板512），重叠只计算一次但分别归约。面板不是全量评价，需公布索引及逐类支持；模型同面板同输入同Oracle上下文。主预测h5、次h3、短期h1；逐录制MSE/NLL后等录制，Skill先逐录制计算1-MSE/Persistence再宏平均；只有一个validation受试者，不宣称跨人统计结论。单列跨实例边界、事件内部和未来未知区域。NLL只使用train Persistence残差逐坐标固定方差，所有方法共用；不声称已学会分布校准或Birth。

AB与A比较预测，AB与B比较语义，三seed分别报告。A/B-only不存在的任务头不填分。原77.01%完整实例读出与本轮固定当前Update面板不同，不直接相减宣称增益/退化。联合训练若无可重复互补收益，报告实际结果，不自动进入Reset。

## 放行检查

先验证源锁、CNN目标冻结、共享初值、零预测、同前缀初始语义、未来扰动、mask、forecast/semantic到共享Adapter的梯度、Qwen无梯度与SHA不变、TBPTT及AMP checkpoint反传。真实Qwen预检实测峰值显存和时间；OOM或非有限失败留证据，不自动缩短上下文或覆盖原输出。输入侧共享TokenAdapter可以训练，目标侧m永远来自固定CNN缓存。此轮即使预测通过也不等于已完成Reset预测后果训练。

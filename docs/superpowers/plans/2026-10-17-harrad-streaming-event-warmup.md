# HARrad流式接口与EventFormer语义预热

用户已授权第1、2步，沿用现有按人划分，不重新征求同一范围授权；本轮顺序实现并按测试先行核验。

## 固定设计

- 只复用RD seed42的best（验证选第29轮），不按三seed优劣再选模型。CNN、GRU、旧12类头全部冻结eval；源缓存、权重和normalizers锁定哈希。test仍只元数据，不读取信号。其它两seed权重保留。
- `FrozenRDStream.forward_chunk`：规范化RD[B,T,40,64]→CNN末层map64×5×8；输出原始全局均值64、2×2空间池化得到4×64、GRU hidden128与可继续状态。时序归一化用既有train统计，不按录制重估。frame_mask padding不更新状态。每个录制起点清零，块间携带GRU状态。
- 验证真实train/validation片段的流式hidden/均值分类与原best一致；允许FP32 CUDA批尺寸舍入≤1e-4且argmax不变。另核验同一连续录制整段/不规则分块等价（不跨录制状态），状态detach不改变前向；未来扰动不能改变过去输出。
- 只导出seed42的CNN cells、全局特征、连续GRU hidden及按官方实例清零的Oracle GRU hidden。Oracle/连续分类结果分别报告；它们不是同一状态协议。B本轮读取CNN cells，不读取GRU或标签来生成Tokens。
- 每2帧一个Update，标称2/15秒；4空间cell×2帧=8个候选64维向量。TokenAdapter：LN64→Linear64→256/GELU，加入4空间位置与2时间位置embedding，以4个可学习query做masked attention pool，再LN256，输出4×256。不复制同一向量伪造多Token。奇数分块保留pending帧；仅显式结束时可flush单帧，padding不计观察时长。
- EventFormer沿用现有StreamingSemanticEventFormer，8×896状态、Qwen2.5-0.5B前4层、cross-attn512/8heads、alpha初值0.001、gate初值0.1、output_projection恒等。只加载通用本地Qwen基座，全部冻结（包括新建LoRA）；不加载PARrad语义权重/14类模板。class reader：event均值→LN896→Linear896→128→GELU→Linear128→12，FP32参数。
- 实例起点初始化Event State；成熟度=min(已观察帧数/15/1.2,1)，绝不使用完整段长、未来标签或完成比例。完整实例用于最终CE；固定观察时刻3/6/12个Update（0.4/0.8/1.6秒）的在段内前缀也参与CE，权重0.25。prefix loss为当前batch可用固定时刻CE均值，final CE权重1。采用原train实例数反比权重，均值归一化。
- seed42、5完整epochs、batch4；AdamW lr1e-4、weight_decay0.01、clip1；TBPTT每8次Update截断状态梯度，跨块不重置状态。固定前缀和结束读出的图保留，较早块不因无loss而保留全部图。trainable旁路FP32，冻结Qwen FP16，CUDA autocast/GradScaler(init_scale256)；最多3次同batch重试，跳步、尝试与成功更新均记录；失败停止。
- 不训练Caption/Commit/Memory/预测分支/自主Reset，本轮是12类实例监督的语义读出预热，不宣称自然语言理解已通过。CNN/GRU冻结，因此保持用前后hash与原分类复现检查，不添加恒定无梯度的“保持loss”。
- 每轮报告完整实例与0.4/0.8/1.6秒前缀Macro-F1/支持数，原GRU在相同前缀的结果作参考；不把短动作被排除的不同前缀总体直接比较。选全段validation Macro-F1最高且最早检查点。
- 完成后最佳权重重放，独立复算；另做CNN cells置零的零特征对照以检查语义模块对特征的依赖（并非全零RD经过CNN）。评估无Oracle Reset的连续validation诊断，不把该诊断作为改变训练阈值或选择checkpoint的条件。
- 正式前需要真实Qwen FP16前/反传、输入梯度、冻结参数不变与padding/未来扰动检查，以及train每类一个实例小样拟合（最多300步lr1e-3，accuracy≥0.99且CE≤0.3）；gate失败保存原因并停止，不自动放宽条件。gate权重不进入正式初始化。

## 文件与实施检查

- [x] `harrad/streaming.py`：冻结RD流式输出/状态与双帧buffer；`test_harrad_streaming.py`验证不规则块、padding、未来扰动与保持原分类。
- [x] `harrad/event_semantic.py`：TokenAdapter、EventFormer封装、观察年龄、前缀与TBPTT；Tiny CPU测试和真实Qwen CUDA预检，端点长度不进入状态输入。
- [x] `harrad/event_data.py`：不可覆盖cache、实例dataset、Oracle/连续参考评价、源hash和只读限制；完整真实录制导出由后台执行。
- [x] `harrad/event_training.py` / `event_runner.py`：实现小样gate、5轮执行、重放/零特征/连续诊断、进度与失败标记；训练完成验收待运行结果。
- [x] 固定配置、目标测试、独立审查；不可覆盖v1源码包上传，原生预检通过后启动持久任务（PID24593）。
- [x] 完整35条录制导出及原best分类复现验收：215,482帧，logits误差5.7220e-6、类别变化0，8组参考指标本地独立复算通过。
- [x] 小样gate在40步通过、5轮共3,775步训练完成、best重放误差0，冻结基座保持不变。
- [x] 零特征及连续validation诊断完成；12份预测下载并通过本地独立指标复算，完成记录见进度文档。检查点仍在服务器保存。

本地pytest使用interview环境；HDF5/真实Qwen预检在服务器已有seer环境执行，不安装依赖。无Git仓库，用源码包与SHA清单保留版本，旧数据及失败证据不覆盖。

# Stage4 分层记忆首轮训练

## 目标和使用

按用户补充的 `计划/方案书v2.5 (2).md` 实现分层事件记忆，压缩器结构继承v2.4。使用现有896维Qwen事件接口：内部1024维、两层TransformerDecoder、4个可学习查询，同一模块处理L0→L1和L1→L2。前端及LLM/LoRA全部冻结，训练共享压缩器及其输入输出投影、位置/时间/层级参数。

服务器隔离目录执行：

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src /home/xibeidaxue/SEER/.conda/envs/seer/bin/python \
  -m continuous_mmwave.parrad_stage4_runner \
  --config configs/parrad-stage4-memory-pilot-v2.yaml --mode all
```

`--mode cache/train/replay` 支持分阶段执行；已有训练日志时拒绝重复训练。重训必须换输出目录。缓存、checkpoint保存并验证冻结来源的SHA与逐张量摘要。checkpoint包含AdamW、scaler和Torch随机状态用于后续恢复实现；本入口当前不提供中断续训CLI。

## 首轮数据和固定预算

- subject_02 / TI_xWR14xx，TRAIN set000/set001拟合，set002开发评估。没有读取validation/test数组或set003。
- 离线训练用干净GT label-run的commit Event Tokens，四个连续事件压成L1，再将连续L1组压成L2。标签只用于离线训练摘要模板及事后在线边界诊断。
- 摘要监督是14类动作模板拼接，不是人工写的自由文本或问答。当前只能证明小规模训练/流程是否运行，不能证明开放问答能力。
- `L = frozen Full LLM caption CE + (1-cos(summary key, source-group key))`，源/摘要语义键通过原冻结P计算。
- batch1，100个实际AdamW更新，seed42，lr1e-4，clip1，保存20/50/100步；用100步结果，不按开发集挑选checkpoint。FP16初始GradScaler取256：同批实际诊断显示默认65536溢出，256与1梯度均有限、loss一致。溢出不会作为实际更新计数。

## 实际在线流程

canonical雷达 → 原Stage1编码器 → 原Stage2 EventFormer → 固定Stage3控制器 → confirm时commit_refine → L0/L1/L2 → 同投影文本查询 → L2 top1、L1 top2、所有L0 → Full Qwen回答。

固定使用 `best_verify.pt` 与原阈值：proposal0.55、连续2次；confirm0.45、reject0.2、timeout6updates。不会通过调整阈值或用GT边界伪造在线事件。录制结束时单独标记未确认EOF尾段，置信度0。

L0最近8个事件；满时将最旧4个提升。L1容量8，L2容量32；L2满时最旧组再压缩。每次查询读全部L0与检索摘要，按时间排列每层。可选语义合并依据键余弦和时间间隔；首轮关闭，保持可见的晋升轨迹。摘要保存tokens/key/时间/置信度/来源ID/count，不保存原雷达数组。来源ID用于pilot可追溯性，其元数据随事件数增长；固定容量限制的是token张量。

在线finalization目前同步执行，用来验证完整接口；尚未实现生产异步队列/调度，因此不能宣称硬实时。另保存GT分段的层次演示，以便在线控制器分段不足时仍能单独验证L1/L2，但这份演示不能当在线结果。

## 必要验证和当前状态

新测试覆盖mask、两级递归梯度、冻结教师、caption因果目标、L0/L1/L2晋升/检索顺序/持久化、合并时间顺序、TRAIN限定、commit真实接口及来源一致性。有限代码审查提出的两个实际问题已修复并有回归测试。

部署和真实结果见执行记录及后续结果报告。此实验跳过原Stage3 Go前置门槛只用于用户授权的工程贯通；原模型文件保留，未见受试者/正式Go均不在本轮结论范围。


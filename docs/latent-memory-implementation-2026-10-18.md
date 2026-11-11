# 无显式动作分段：独立 latent_memory 实现

设计与约束见 `新计划/方案书v1.0.md` 和 `新计划/代码落地建议v1.0.md`。本实现只新增文件，旧入口、配置、模型、特征和实验记录保持原样。

## 已实现

- 完整 HARrad cells 流接入，复用 FrozenRDStream、PairBuffer、RadarTokenAdapter；新封装持有输入副本，查询快照不消耗 pending 帧。
- 三尺度、有界、按容量晋升的外部时间记忆。压缩使用按原帧权重加权均值，保存时间覆盖与有效位；这是有损基线，不是动作分割或 Jenks。
- 三组可学习 LMT；每次查询从同一 Initial LMT 参数重新 Bootstrap。各组分别调用完整冻结 Qwen，其候选范围为 S、S∪M、S∪M∪L，防止跨组间接信息泄漏。
- 每轮用当前 LMT 重新计算余弦分数、稳定排序和检索；注入证据按时间排列；记录候选、分数、选择、截止帧和状态变化。
- 回答 Token 的 CE 训练、优化器及随机状态恢复；只保存新参数。Qwen 与局部适配器冻结，但训练前向保留输入梯度。梯度检查点也作用于冻结 eval 模式的前向；答案词表头按64个位置分块并重计算，降低长描述反向的激活占用。
- 独立 `prepare/probe/train/evaluate/replay` 命令；显式步数、拒绝覆盖输出、来源指纹、失败状态持久化及独立结果核验器。

新包：`src/continuous_mmwave/latent_memory/`。配置：`configs/harrad-latent-memory-v1.json`。核验脚本：`scripts/verify_latent_memory_run.py`。

## 默认配置的含义

工程默认值：15fps、每30帧查询、每层16个时间单元、4个单元合并、每组4个LMT、2轮演化、每组检索4个单元。单元中的两帧是固定计算粒度，不是动作片段。使用 log-time 投影编码起止、年龄与跨度，另用普通 LLM 位置表示输入序列位置。

优先使用现有35条录制的完整 cells 缓存，以及 EventFormer checkpoint 中独立 `tokens.*` 的局部适配器。新 Projector/LMT 未经训练，不保证能够直接描述动作。所有这些数值和简单压缩策略都是可更改的工程配置，不改写方案书中的“待定”项。

首版实现采用单流、batch=1 的 LLM 接口。训练样本从固定查询点中抽取，每次从录制起点重建真实过去记忆，正确性优先，长录制训练速度尚未优化。暂未实现跨查询继承LMT、Jenks、学习型压缩、生成侧LoRA或PMO。

## 服务器使用

使用现有环境，不升级依赖。以下命令在包含本次新增源码的独立代码目录中执行；每个 `--output` 必须是尚不存在的新目录。

```bash
export PYTHONPATH=src
PY=/home/xibeidaxue/newdisk2/WJ/.seer-conda/envs/seer/bin/python

$PY -m continuous_mmwave.latent_memory prepare \
  --config configs/harrad-latent-memory-v1.json --output /path/to/new-prepare

$PY -m continuous_mmwave.latent_memory probe \
  --config configs/harrad-latent-memory-v1.json --output /path/to/new-probe --device cuda

# 仅在明确实验预算后执行；train 不接受省略 --steps。
$PY -m continuous_mmwave.latent_memory train \
  --config configs/harrad-latent-memory-v1.json --output /path/to/new-train --steps 100 --device cuda

# 恢复至总共200次成功更新，使用新的输出目录，保留原运行。
$PY -m continuous_mmwave.latent_memory train \
  --config configs/harrad-latent-memory-v1.json --output /path/to/new-resume \
  --steps 200 --resume /path/to/new-train/step-000100.pt --device cuda

$PY -m continuous_mmwave.latent_memory replay \
  --config configs/harrad-latent-memory-v1.json --output /path/to/new-replay \
  --checkpoint /path/to/new-train/step-000100.pt --device cuda

$PY scripts/verify_latent_memory_run.py /path/to/new-replay
```

`evaluate` 和 `replay` 目前使用同一完整录制回放实现，默认只评价 subject7。`--max-recordings`、`--max-queries` 可限制诊断规模；输入清单的全部35个 cells 文件仍先核验。未训练回放必须显式传 `--allow-untrained`。`probe` 进行反向与生成核验，但不调用优化器。

同一检查点的推理消融：`--rounds 0` 为 Bootstrap-only；`--rounds 2` 为两轮演化；`--fixed-evidence` 固定首轮证据以区分重新检索与多次LLM更新。各自独立训练的主对照应使用各自独立配置和输出，不能把同一权重切换轮数视为同预算训练实验。

## 监督与评价边界

训练/验证沿用 subject1–6 / subject7，test8–9信号不读取。运行时只拿到 file、frames、cells_path，标注在独立监督/评价函数中使用，不驱动写入、查询时刻、Reset或检索。

弱目标为已开始的标注实例按先后顺序组成英文描述，保留相邻同类实例，对尚未结束实例加 `(ongoing)`。参考目标不含未来终点数值；未知帧仍输入模型并单列覆盖数。区间标签不提供幅度、速度等细节，也不能证明没有其他动作。空前缀目标只表示“尚无标注动作”，不代表实际无人动作；因此未知区间需人工核验。

自动评分严格接受约定英文句式，其他自由表述标为不可解析，不能把“never stood up”等否定句提取成正确动作。报告解析率、顺序编辑距离/完整正确率、12类实例次数MAE、含进行中标记的完整状态序列正确率和生成截断情况。自由描述改写、动作内部重复、未知区间及细节能力需人工描述参考，不能用这些闭集弱指标代替。

回答监督超过 `max_answer_tokens` 或上下文预算会显式失败，保留状态，不静默截断；生成达到预算但未EOS会标记 truncated。恢复要求配置、代码、原特征、基座文件及冻结参数指纹一致，调整研究配置应另开运行。

## 验证

本地 pytest 使用用户指定的 `C:/User/anaconda/envs/interview/python.exe`。新增测试涵盖记忆覆盖、分块、buffer所有权、无未来、分组隔离、每轮重检索、冻结梯度、原生CE对齐、检查点激活节省、精确断点恢复、状态落盘、否定句、进行中状态及真实微型Qwen的CLI全链路。

初次完整本地套件因原测试 `tests/test_parrad_convert.py` 依赖的 h5py 缺失而无法收集；未安装依赖。排除此单文件的原有兼容性运行结果及最新新增测试结果保存在 `outputs/latent-memory-implementation-20261018/`。397份原文件SHA清单亦在该目录，可独立复核。

服务器验证与本地测试分开记录；真实Qwen工程贯通不代表已经验证连续描述准确率或实时性能。本次不启动正式长训练。

最终实测（源码v2）：

- 本地 interview：新增20项通过；此前排除缺失h5py所影响的 `test_parrad_convert.py` 后，兼容性运行972通过、16跳过。完整本地收集失败记录仍保留。
- 服务器现有 seer 环境：新增20项通过。完整最终套件 **995通过、1跳过、2项原有Gemma失败**。首次全量还出现3项CuBLAS确定性环境失败；以 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 重跑后消失，未改旧代码。
- 保留的失败为 `tests/test_gemma3_adapter.py::test_gemma3_adapter_matches_native_prefix_and_targets_only_qv` 和 `tests/test_gemma3_adapter.py::test_gemma3_adapter_honors_local_sliding_window`；报错为 `Gemma3RotaryEmbedding` 缺少 `None_inv_freq`。在旧 event_warmup 源码快照中单独重现2项失败；相关 `event_former.py` 的SHA与本地和新包均相同，不修改旧Gemma路径或共享依赖。
- 真实35条录制cells缓存全部逐文件哈希/形状/有限值检查通过；完整本地Qwen预检通过：新参数梯度非零、冻结权重不变、短组隔离、清空新参数后重新加载的LMT误差0。
- 固定一条训练录制90帧前缀，完成一次优化，再从检查点恢复至第2步；优化器步数/矩估计、新权重及来源核验通过。损失分别4.333819、5.807055，未显示小样损失下降，也没有据此宣称有效学习；本轮峰值CUDA分配1,232,811,520字节，只代表此短前缀，不代表长流预算。
- 服务器最终代码目录为 `/home/xibeidaxue/newdisk2/WJ/HARrad/experiments/latent_memory_20261018_code_v2`。预检、优化和恢复分别写入 `outputs/latent_memory_20261018_probe_v2`、`outputs/latent_memory_20261018_optimizer_smoke_v2`、`outputs/latent_memory_20261018_optimizer_resume_v2`，均位于服务器HARrad根目录。
- 本地证据目录为 `outputs/latent-memory-implementation-20261018/`；包含源码包、原397份文件哈希、测试日志以及下载的native证据/新检查点。旧方案文件逐一SHA校验不变。
- 下载后的probe/smoke/resume由本地独立核验器再次读回，优化器步数、张量有限值、来源和所有文件SHA与服务器记录一致；本地与服务器源码指纹相符。结果见该目录 `local-independent-verification.json`。

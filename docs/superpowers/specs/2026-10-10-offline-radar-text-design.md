# Offline Radar-to-Text MVP

用户目标：输入完整 PARrad 雷达窗口，直接自回归生成动作过程描述；先验证单动作和 1–3 个动作的顺序。离线处理，不使用事件检测、Fork-and-Shadow、Event Token 或长期记忆。用户已明确指定架构并要求实现。

## 数据和结构

复用 canonical NPZ/JSONL 数据约定、转换工具和受试者划分工具。新模块独立放在 `continuous_mmwave.radar_text`，不修改旧实验实现。先按受试者划分，再按真实时间戳切 8–16 秒窗口；默认 12 秒、步长 6 秒、最短 8 秒。窗口不跨 recording/session/sensor；未知标签窗口排除，不能把中间未知时段删掉后拼成连续动作。

RD `[B,T,1,93,128]` 每帧三层 CNN + global average pooling 得 128 维。MD `[B,T,128]` 以每帧为中心构造 9 帧局部时间片，每片三层 2D CNN 得 128 维；边缘复制只在真实序列内部进行。Concat → 两层 BiGRU（每方向 128）→ 按真实长度 adaptive average pooling → `[B,8,256]` → 两层 MLP → LLM embedding hidden size。变长序列使用 pack/pad，CNN 使用 GroupNorm，避免 padding 影响真实帧。

默认 Qwen2.5-0.5B-Instruct，可配置 1.5B 或本地模型目录。使用模型自带 chat template，将雷达向量插入 user 消息中的唯一占位符位置。语言损失只监督 assistant 的文本和结束标记；prompt、radar、padding 全部为 -100。冻结 LLM 不能切断输入 embedding 的梯度。

## 训练和验收

阶段 1：有效帧 CE 辅助分类，保留 encoder 权重。阶段 2：训练 encoder + projector、冻结 LLM；支持加载阶段 1 checkpoint。后续使用独立配置启用 LoRA 并加载对齐 checkpoint。提供 checkpoint 恢复推理、验证集选优、测试集生成、分类→模板基线和随机 tiny Qwen 冒烟模式（不下载模型）。

弱监督使用每个连续标签 run 的顺序，只合并相邻重复；A→B→A 保留。显式 phrase JSON 将真实 class id 映射为英文第三人称动词短语，不凭 14 类标签捏造速度、幅度或停顿。支持窗口级人工 caption 覆盖。报告动作覆盖、提取到的动作序列完全匹配、LCS 顺序分数、模板 exact match，记录未匹配文本；这些是受限词表代理指标，不代表细粒度语言质量。

主要测试：MD 局部性、padding 不变性、无标签推理、因果 LM loss mask、冻结主干梯度、先后顺序保留、受试者泄漏拒绝、checkpoint 往返和 tiny Qwen 训练/生成。真实学习效果必须另用真实 PARrad 和预训练权重验证。

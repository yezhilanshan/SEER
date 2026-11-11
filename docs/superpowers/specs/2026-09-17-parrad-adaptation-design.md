# PARrad 连续语义事件形成验证版设计

**状态：** 待用户复核  
**日期：** 2026-09-17  
**适用仓库：** `D:\lab\SEER`  
**设计性质：** PARrad 数据集适配与机制可行性验证，不替代完整 RD + Point Cloud 方案

## 1. 背景

当前项目按照方案书 v2.3 的第一阶段实现了：

```text
RD + Point Cloud
→ Streaming Radar Encoder
→ Stage 1 Pose-Guided Pretraining
→ Streaming/Commit Event Former
→ Stage 2 Radar-Language Alignment
```

PARrad 不提供点云与逐帧 3D Skeleton，而提供：

- Range-Doppler Map（RD）；
- Micro-Doppler Signature（MD）；
- 连续活动标签；
- 两台不同频段、不同视角的雷达记录。

MD 是 RD 沿 Range 维聚合得到的速度分布，不是独立于 RD 的第二传感模态。因此 PARrad 版本不能把 MD 当作 Point Cloud 的等价替代物，也不应把 RD 与 MD 简单视为两个相互独立的模态。

本设计将 PARrad 定位为早期验证数据集，只验证以下核心假设：

1. Radar Token 能否形成稳定、可持续更新的 Event Token；
2. Active/Candidate 状态能否表达连续语义状态；
3. Physical Proposal + Semantic Verify 能否形成更可靠的事件边界；
4. Fork-and-Shadow 能否减少错误 proposal 对在线状态的破坏。

## 2. 目标与非目标

### 2.1 目标

- 保留原始 RD + Point Cloud Encoder，不破坏完整方案的代码路径。
- 新增独立的 `PARradRadarEncoder`，将 RD 与 MD 映射到现有 Event Former 接口。
- 每次在线更新仍输出 4 个、每个 256 维的 Radar Token。
- 将 RD 作为语义主表示，将 MD 作为运动先验和变化检测特征。
- 使用约 180 ms 的在线更新周期验证连续事件形成。
- 使用活动标签及其固定语言模板训练 Event Token 的闭集语义表示。
- 使用受试者级连续序列评估边界质量、事件序列质量和确认延迟。

### 2.2 非目标

PARrad 验证版不用于证明：

- Point Cloud 分支的有效性；
- Pose-Guided Radar Pretraining 的有效性；
- 开放词汇或开放集行为理解；
- 自由文本 caption 生成能力；
- 分层长期事件记忆；
- 多人行为理解；
- 多雷达端到端联合融合；
- 默认语义骨干改为 Qwen2.5-0.5B-Instruct；5 Hz 下的最终延迟仍需在目标服务器实测。

论文与实验报告中必须将其称为 **PARrad adaptation** 或 **PARrad validation variant**，不得将其描述成完整模型。

## 3. 已冻结的设计决策

### 3.1 输入与时间尺度

PARrad 的典型输入约为：

```text
RD: [T, 1, 93, 128]
MD: [T, 128]
frame period: 0.09 s
```

实际文件读取前必须由数据审计程序确认 shape、dtype、数值域、时间轴和标签轴。模型层不依赖文件内部字段名，只依赖数据适配器输出的标准张量。

在线更新使用：

```text
frames_per_update = 2
delta_t = 0.18 s
effective update rate ≈ 5.56 Hz
```

第一版不对原始时间轴插值到 20 fps 或 10 fps。所有延迟、平滑系数和边界容差均使用秒作为单位。

### 3.2 输出契约

每个在线 update 输出：

```text
radar_tokens:  [B, 4, 256]
semantic_state:[B, 256]
motion_state:  [B, 128]
frame_hidden:  [B, 2, 256]
token_mask:    [B, 4]
frame_mask:    [B, 2]
stream_state:  RD/MD temporal caches + update index
```

其中：

- `radar_tokens` 输入 Event Former；
- `semantic_state` 用于辅助语义分类、诊断和探针实验；
- `motion_state` 只输入 Local Change、Radar Transition 和物理 prototype；
- `frame_hidden` 只用于逐帧辅助监督和诊断，不进入 Event Former；
- Event Former 不直接读取 `motion_state`。

Event Former 的输入 token 数保持为 4，不采用附件建议中的 32 个 token。这样可以保持当前 Cross-Attention、Streaming/Commit 路径和延迟预算的接口不变。

### 3.3 双 Encoder 并存

代码中保留两个 Encoder 变体：

```text
original_rd_point
parrad_md_guided
```

二者实现同一逻辑输出协议，但拥有不同的数据输入协议。PARrad 适配不得通过大量可空字段污染原始 `RadarSample` 和 `RadarBatch`。

## 4. 总体架构

```text
PARrad RD Map [B, 2, 1, 93, 128]
             │
        RD Spatial Encoder
             │
       Z_rd [B,2,C,R',D']
             │
             │               PARrad MD [B,2,128]
             │                         │
             │                  MD Frame Encoder
             │                         │
             │                  MD Causal TCN
             │                         │
             │              ┌──────────┴──────────┐
             │              │                     │
             │          gamma, beta          motion_state
             │              │                     │
             └────── FiLM modulation              └→ Physical Proposal
                            │
                  Enhanced RD Feature Map
                            │
                  Doppler-Aware Pooling
                     ┌──────┴──────┐
                     │             │
              Doppler Tokens  Frame Pooling
                     │             │
                     │       RD Causal TCN
                     │             │
                     └── Context Injection
                            │
                  4-Query Token Resampler
                            │
                 radar_tokens [B,4,256]
                            │
                       Event Former
                            │
                Active / Candidate / Shadow
```

设计原则是：

```text
MD 判断“运动模式是否发生变化”
RD + Event Token 判断“语义是否成为新事件”
```

## 5. 组件设计

### 5.1 PARrad 数据适配器

新增独立的数据类型：

```text
PARradSample
  rd
  md
  labels
  timestamps
  subject_id
  session_id
  sensor_id
  environment_id

PARradBatch
  rd
  md
  labels
  timestamps
  frame_mask
  metadata
```

数据适配器负责：

1. 读取 PARrad 原始文件；
2. 对齐 RD、MD、标签和时间戳；
3. 将连续记录切成训练片段，但保留原始全局时间索引；
4. 生成 padding mask；
5. 保留受试者、会话、雷达和环境标识；
6. 输出统一的浮点张量；
7. 在发现缺帧、长度不一致或时间倒退时立即报错。

第一步必须生成数据审计报告，至少包含：

- 每种文件的字段名、shape 和 dtype；
- RD 与 MD 的长度关系；
- 标签覆盖率和无标签区间；
- 时间间隔分布；
- 每类持续时间分布；
- 每个受试者、会话、环境和传感器的数据量；
- 提供的 MD 与 RD 沿 Range 聚合结果的相关性。

若提供的 MD 与 RD 聚合结果不一致，第一版使用数据集提供的 MD，并在审计报告中记录差异，不在模型内部重新计算 MD。

### 5.2 RD Spatial Encoder

RD 输入为单通道幅度图。第一版结构为：

```text
Conv Stem:      1 → 32,  stride=(2,1)
Residual Block: 32 → 64, stride=(2,1)
Residual Block: 64 → 128,stride=(2,2)
Residual Block: 128 → 256,stride=(1,1)
```

设计约束：

- Range 维可以较早下采样；
- Doppler 维不得在前两层下采样；
- 不在卷积骨干末端直接执行二维全局平均池化；
- 所有空间尺寸由输入动态推导，不硬编码为 `64×64`。

典型尺寸变化为：

```text
Range:   93  → 47 → 24
Doppler: 128 → 128 → 64
```

### 5.3 MD Motion Encoder

MD 分支只负责速度分布和变化模式，不独立生成语义 token。

每帧结构：

```text
[128]
→ Conv1d 1→32
→ Residual Conv1d 32→64
→ Adaptive Pool
→ Linear 64→128
```

帧级特征输入 Stateful Causal TCN：

```text
dim = 128
kernel = 3
dilation = 1, 2, 4
```

TCN 输出作为 `motion_state`。它用于：

- 静止到运动；
- 快慢变化；
- 正负 Doppler 能量变化；
- 速度分布扩散或收缩；
- 周期性运动变化；
- fast/slow motion prototype。

### 5.4 MD-Guided FiLM

由每帧 MD hidden 生成通道级参数：

$$
[\gamma_t,\beta_t]=\operatorname{MLP}(m_t)
$$

对 RD feature map 执行：

$$
\widetilde Z_t^{RD}=(1+\gamma_t)\odot Z_t^{RD}+\beta_t
$$

其中 `gamma` 和 `beta` 形状为 `[B,C,1,1]`，不生成逐空间位置参数。

初始化要求：

```text
gamma projection weights = 0
gamma projection bias = 0
beta projection weights = 0
beta projection bias = 0
```

因此初始状态严格满足：

$$
\widetilde Z_t^{RD}=Z_t^{RD}
$$

这使 MD 增强可以从恒等映射稳定开始，也允许明确比较 RD-only 与 MD-guided RD。

### 5.5 Doppler-Aware Pooling

FiLM 后的 RD feature 不立即做二维全局池化，而采用：

```text
沿 Range 维池化
→ [B,T,C,D']
→ 逐 Doppler bin 投影
→ doppler_tokens [B,T,D',256]
→ Doppler Attention Pooling
→ frame_hidden [B,T,256]
```

`doppler_tokens` 加入 Doppler 位置编码；`frame_hidden` 作为逐帧语义向量输入 RD TCN。该设计既保留速度分布局部结构，又提供稳定的帧级辅助监督接口。

### 5.6 RD Stateful Causal TCN

每帧语义向量输入：

```text
dim = 256
kernel = 3
dilation = 1, 2, 4, 8
```

TCN 输出为每帧 temporal context。该 context 通过加法或门控残差回注到同一帧的所有 `doppler_tokens`。TCN 必须保持 chunked forward 与 full-sequence forward 数值等价。跨 update 的缓存属于 `PARradStreamState`，不得由 DataLoader 隐式管理。

### 5.7 4-Query Token Resampler

每次 update 将回注时序上下文后的 `[T,D',256]` token 展平为 `[T×D',256]`，再由 4 个 learnable queries 读取：

$$
F_t=\operatorname{Attention}(Q,Z_t,Z_t),\quad Q\in\mathbb R^{4\times256}
$$

输出固定为：

$$
F_t\in\mathbb R^{4\times256}
$$

四个 query 不预先绑定固定物理含义，由训练学习不同的当前事件信息槽。它们读取时间与 Doppler 局部信息，而不是只读取两个帧级向量。`semantic_state` 由 `F_t` 的 attention pooling 得到。

## 6. 下游接口

### 6.1 Event Former

Event Former 保持以下契约：

```text
stream_update(previous_event, radar_tokens, maturity, radar_mask)
commit_refine(all_event_radar_tokens, radar_mask)
```

PARrad Encoder 不改变：

- Event Token 数量；
- LLM hidden size；
- 512 维 Cross-Attention bottleneck；
- token-wise scalar state gate；
- Streaming/Commit 一致性训练逻辑。

### 6.2 Physical Proposal

Local Change 和 Radar Transition 使用 `motion_state`，而不是 Event Token 或 LLM embedding。

fast prototype 使用实际更新时间：

$$
\alpha=\exp(-0.18/\tau_{fast})
$$

默认 `tau_fast=2 s` 时，`alpha≈0.914`。

Local Change 可同时读取：

```text
motion_state difference
absolute difference
element-wise product
```

Radar Transition 在 motion prototype 空间维护 fast/slow/Welford 统计。

### 6.3 Semantic Verify

Semantic Verify 只比较：

- Frozen Active Event Tokens；
- Candidate Event Tokens；
- Candidate stability；
- Candidate maturity。

MD、Local Change 和 Radar Transition 在 Candidate 建立后不参与 Confirm，保持 Propose/Verify 职责分离。

### 6.4 Fork-and-Shadow

PARrad 验证版沿用方案书 v2.3：

- proposal 时冻结 Active Reference；
- Candidate 从 `P_base` 启动；
- Shadow 从旧 Active 快照继续吸收同一批 Radar Token；
- Reject 时 Shadow 以 O(1) 方式接管；
- Confirm 时 Candidate 成为新 Active。

PARrad 适配不实现 Fork-and-Replay。

## 7. 训练设计

### 7.1 Stage P0：数据审计

在任何模型训练前完成数据审计。若 RD、MD、标签无法逐帧可靠对齐，则停止模型训练并修复适配器，不允许通过截断到最短长度静默处理。

### 7.2 Stage P1：PARrad Encoder 预训练

PARrad 不执行 Pose-Guided Pretraining。使用临时逐帧活动分类头：

```text
frame_hidden → Linear 256→num_classes
```

损失为类权重交叉熵：

$$
L_{P1}=L_{weighted\_CE}
$$

类权重只从训练集统计。P1 完成后分类头不属于部署模型，但可保留用于探针评估。

P1 的作用是让前端先学习活动相关的 RD/MD 表示，不用于支持 category-agnostic 或 open-set claim。

### 7.3 Stage P2：Event Token 闭集语义对齐

每个活动类准备 3 个语义等价的固定英文模板。训练时在同类模板中确定性轮换，避免单一标签字符串成为捷径。

示例：

```text
walk to bed
→ "a person walks toward the bed"
→ "the person is moving to the bed"
→ "someone approaches the bed"
```

沿用现有三项 Stage 2 损失：

$$
L_{P2}=L_{text}+\lambda_aL_{align}+\lambda_cL_{cons}
$$

这里的 `L_text` 表示生成固定活动描述模板，不表示自由文本 caption 能力。

### 7.4 Stage P3：Continuous Transition Learning

从连续帧标签生成真实事件段和边界。边界监督使用以秒为尺度的 Gaussian soft label。

训练：

- Local Change Head；
- Radar Transition Head；
- Semantic Difference Head；
- monotonic Verify Head。

损失保持：

$$
L_{P3}=\lambda_1L_{local}+\lambda_2L_{radar}+\lambda_3L_{semantic}+\lambda_4L_{verify}
$$

Stage P3 不训练独立 Stability Head。

## 8. 数据划分与泄漏防护

### 8.1 划分单位

数据必须先按受试者划分 train/validation/test，再展开会话、片段和传感器视角。

禁止：

- 同一受试者出现在训练集和测试集；
- 同一会话的 60 GHz 视角进入训练集、77 GHz 视角进入测试集；
- 先切滑窗再随机划分；
- 以帧为单位随机划分。

### 8.2 两台雷达的使用方式

第一版不融合两台雷达。每个传感器视角作为一个独立样本，共享同一 Encoder 权重，并加入可学习 `sensor_id embedding`。

测试报告分别给出：

- 60 GHz；
- 77 GHz；
- 两者合并；
- 跨传感器诊断结果。

“跨传感器诊断”固定采用受试者不重叠的协议：仅用一个频段的训练受试者训练，在另一个频段的测试受试者上测试。该结果同时包含受试者域偏移和传感器域偏移，因此不得表述成纯粹的 sensor-only domain transfer。此诊断运行时关闭 `sensor_id embedding`，避免测试传感器 embedding 未训练。

### 8.3 连续序列优先

训练可以采样有限长度片段，但验证和测试必须按完整连续会话顺序执行。模型状态仅在会话开始、明确长时间无人或数据中断超过规定阈值时重置。

## 9. 评估设计

### 9.1 对照组

| 编号 | Encoder | 事件机制 | 用途 |
|---|---|---|---|
| A | RD-only | 固定窗口分类 | 数据集常规基线 |
| B | RD-only | 普通 Change Detector | 物理边界基线 |
| C | RD-only | Active/Candidate | 验证事件机制本身 |
| D | RD+MD concat | Active/Candidate | 简单融合消融 |
| E | MD-guided RD | Active/Candidate | 推荐完整验证版 |

对照 C 与 E 的差异只允许来自 MD-guided Encoder；对照 B 与 C 的差异只允许来自 Active/Candidate + Semantic Verify。

### 9.2 核心指标

事件边界：

- Boundary F1，容差 `±0.36 s`、`±0.72 s`；
- Over-Segmentation Rate；
- Under-Segmentation Rate；
- Proposal-to-Confirm Ratio；
- False Proposal Rate；
- Transition Latency，单位秒。

事件语义：

- event-level activity accuracy；
- segment mIoU；
- semantic purity；
- event sequence edit score；
- event order accuracy。

流式性能：

- 单 update P50/P95 延迟；
- Candidate + Shadow P50/P95 延迟；
- Candidate duty cycle；
- 长序列状态范数；
- 显存峰值。

### 9.3 Go/No-Go 条件

PARrad 验证版只有同时满足以下条件才算支持继续开发：

1. Active/Candidate 相比普通 Change Detector 提高 Boundary F1 或明显降低过分割；
2. 确认延迟中位数不超过 1.2 s；
3. Event Token 的线性 probe 明显优于多数类基线；
4. Streaming/Commit key cosine 在测试集保持稳定；
5. 连续完整会话中不存在持续性的状态范数发散；
6. MD-guided RD 至少在主要边界指标或受试者独立测试上优于 RD-only，否则删除 MD-FiLM，不为复杂度辩护。

该阶段不设置虚构的绝对精度阈值。最终阈值应在完成数据审计后，以训练集类别分布和固定窗口基线为参照写入实验配置，而不是回填本设计文档。

## 10. 配置设计

新增独立配置文件，不修改原始生产配置的语义：

```yaml
model:
  encoder_variant: parrad_md_guided
  radar_dim: 256
  radar_tokens_per_update: 4
  frame_period_s: 0.09
  frames_per_update: 2
  rd_channels: 1
  rd_height: 93
  rd_width: 128
  md_bins: 128
  motion_dim: 128
  rd_tcn_dilations: [1, 2, 4, 8]
  md_tcn_dilations: [1, 2, 4]

data:
  split_unit: subject
  preserve_continuous_sessions: true

stage:
  variant: parrad
  semantic_supervision: class_templates
```

配置加载时必须验证：

- `frames_per_update > 0`；
- `frame_period_s > 0`；
- token 数恰好为 4；
- RD、MD 时间长度相同；
- MD bins 与配置一致；
- split 不包含重复受试者或重复会话。

## 11. 错误处理

以下情况必须抛出明确异常，不做静默修复：

- RD 与 MD 帧数不一致；
- 标签长度与雷达长度不一致；
- 时间戳非单调；
- 输入出现 NaN/Inf；
- 会话内传感器标识发生未声明变化；
- update 中没有任何有效帧；
- stream state 的 batch size、device 或 dtype 与当前输入不一致；
- 训练/验证/测试存在 subject 或 session 重叠。

对无标签区间使用显式 ignore mask，不把其自动映射为 Idle。

## 12. 测试设计

### 12.1 数据测试

- RD/MD/label/timestamp 对齐；
- 不同长度样本 padding 与 mask；
- 两帧 update 切分；
- 连续会话不被随机重排；
- subject/session/sensor 泄漏检测；
- NaN、Inf、时间倒退和长度不一致报错。

### 12.2 模型单元测试

- RD Encoder 支持 `[1,93,128]`；
- Doppler 维在前两层保持不变；
- FiLM 零初始化时输出严格等于未调制 RD feature；
- MD 分支输出 `[B,T,128]`；
- Encoder 每 update 输出 `[B,4,256]`；
- 全 padding 输入不产生 NaN；
- chunked TCN 与 full-sequence TCN 等价；
- stream state 能跨 update 正确延续。

### 12.3 集成测试

- PARrad Encoder 输出可以直接输入现有 Event Former；
- Streaming 与 Commit 路径均可前向和反向；
- P1/P2/P3 各完成一个 optimizer step；
- checkpoint 保存和恢复后输出一致；
- 两个 Encoder 变体互不影响；
- tiny 合成数据端到端 smoke test。

### 12.4 真实性检查

测试必须包含至少一个真实 PARrad 会话的短片段，只用于验证文件解析与时间对齐，不用于断言模型精度。

## 13. 预计代码边界

建议新增：

```text
src/continuous_mmwave/data_parrad.py
src/continuous_mmwave/models/parrad_encoder.py
src/continuous_mmwave/stages/parrad_stage1.py
src/continuous_mmwave/stages/parrad_stage2.py
src/continuous_mmwave/stages/parrad_stage3.py
src/continuous_mmwave/parrad_audit.py
configs/parrad-stage1.yaml
configs/parrad-stage2.yaml
configs/parrad-stage3.yaml
tests/test_parrad_data.py
tests/test_parrad_encoder.py
tests/test_parrad_training.py
```

建议小范围修改：

```text
src/continuous_mmwave/config.py
src/continuous_mmwave/train.py
```

现有 `event_former.py` 不需要修改：PARrad Encoder 继续满足其每次 update 恰好 4 个 Radar Token 的契约。

原始文件应保持兼容：

```text
src/continuous_mmwave/data.py
src/continuous_mmwave/models/radar_encoder.py
src/continuous_mmwave/stages/stage1.py
src/continuous_mmwave/stages/stage2.py
configs/stage1.yaml
configs/stage2.yaml
```

## 14. 实施顺序

1. PARrad 数据审计与 split manifest 生成；
2. PARrad Dataset/Collate；
3. RD-only PARrad Encoder 基线；
4. MD Motion Encoder 与 FiLM；
5. 统一 Encoder 输出协议；
6. Stage P1 闭集前端预训练；
7. Stage P2 Event Token 模板语义对齐；
8. Stage P3 Proposal/Verify/Fork-and-Shadow；
9. 完整连续会话评估；
10. RD-only、concat 与 FiLM 消融；
11. 根据 Go/No-Go 决定是否继续分层记忆。

不得在数据审计与 RD-only 基线完成前实现复杂融合；不得在 P2 探针通过前实现分层记忆。

## 15. 风险与缓解措施

### 15.1 RD 与 MD 信息冗余

风险：FiLM 只增加参数，没有带来有效新信息。

缓解：FiLM 采用恒等初始化，并把 RD-only、concat、FiLM 纳入强制消融。若 FiLM 无收益则删除。

### 15.2 类标签模板造成语义捷径

风险：Event Token 只学习 14 个标签词，而非动作结构。

缓解：每类使用多个模板；报告闭集限制；使用跨受试者、跨传感器 probe；不声称开放词汇能力。

### 15.3 类别不平衡与长 Idle 段

风险：多数类或静止片段主导训练。

缓解：受试者级划分、类权重、事件级采样与完整会话测试并用；无标签区间不自动视为 Idle。

### 15.4 传感器视角泄漏

风险：同一活动的双雷达记录跨集合出现，使结果虚高。

缓解：先按 subject/session 分组，再展开 sensor view；训练启动前执行泄漏断言。

### 15.5 低帧率限制短事件边界

风险：90 ms 帧周期与180 ms update 限制边界分辨率。

缓解：所有指标按秒报告；边界容差使用0.36 s和0.72 s；不宣称亚帧级定位。

## 16. 结论

PARrad 版本采用：

```text
RD semantic backbone
+ MD motion prior
+ identity-initialized FiLM
+ causal temporal encoding
+ four-token output contract
```

它是对原始系统 Sensor Front-End 的独立适配，而不是完整方案的替代。验证成功意味着 Continuous Event Formation、Event Token 和 Active/Candidate 机制在真实长时间雷达流上具有可行性；验证失败则应优先分析 Event Token 与状态机，而不能用 Point Cloud 或 Pose 监督缺失掩盖问题。

## 17. 参考资料

- PARrad 官方页面：https://www.imec-int.com/en/79GHz-140GHz-radar-systems/parrad
- PARrad 项目页：https://sumo.ilabt.imec.be/node/33
- Patient activity recognition using radar sensors and machine learning：https://doi.org/10.1007/s00521-022-07229-x
- Split BiRNN for real-time activity recognition using radar and deep learning：https://www.nature.com/articles/s41598-022-08240-x

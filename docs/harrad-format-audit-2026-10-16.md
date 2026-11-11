# HARrad 格式、动作标注与重训接入检查（2026-10-16）

已完成服务器文件检查、全量 CSV 元数据校验、本地独立复算及 12 类训练片段的官方 FFT 小样检查。**HARrad 具备直接关联雷达帧的动作实例区间，可以用于新的有监督训练；无需继续用 PARrad 的命令时间代理生成动作区间。** 原始数据未修改，尚未启动模型训练或全量特征转换。

服务器数据根目录：`/home/xibeidaxue/newdisk2/WJ/HARrad/harrad`。本地审计与脚本：[outputs/harrad-format-audit-20261016-v1](../outputs/harrad-format-audit-20261016-v1/)。

## 1. 实际数据格式

| 项目 | 本次检查结果 |
| --- | --- |
| 受试者 / 连续录制 | 9 人 / 37 个 HDF5 |
| 总帧数 | 233,515 |
| HDF5 总文件字节数 | 61,227,905,520，约 57.02 GiB |
| 文件名 | `subject1_01.hdf5` 等；文件名中有受试者及录制号 |
| 实际 HDF5 数据集 | 全部仅含 `radar`，无现成 RD、MD、逐帧标签或时间戳 |
| 原始雷达 | `float32 [T,256,256]`，每帧 256 chirps × 256 samples；非已处理的 RD 图 |
| 原始数组分块 | `[1,256,256]`；无压缩；各文件无时间属性 |
| 官方处理脚本输出 | RD `[T,160,256]`；MD `[T,256]` |
| 标注 | 无表头 CSV：`filename,start,stop,activity_class,subject` |

例如 `subject1_01.hdf5` 有 6,311 帧。作者的 [RadarDataset 加载器](https://github.com/baptist/harrad/blob/67f8da31dfddf8f60906f20920a485cd84a433b5/datasets.py#L134-L171) 直接使用 `dataset[start:end]`，样本长度为 `end-start`，因此接入时采用零基帧索引和 **`[start,stop)`**。保持原始整数帧号，不再从语音命令推算边界。

作者论文给出的录制速度约 15 FPS；[正式论文附录](https://link.springer.com/article/10.1007/s00521-019-04408-1) 也使用 30 帧对应 2 秒。可采用 `frame_period_s=1/15` 的标称时间网格，同时在元数据中标为 nominal；本批 HDF5 没有硬件逐帧时间戳，不能声称已验证每帧精确物理间隔。原论文关于 256 chirps 和 chirp 时长的描述也只是近似帧率依据。

## 2. 标注确实是实例区间，仍应保留未知区域

[发布方说明](https://www.imec-int.com/en/79GHz-and-140GHz-radar-solutions/harrad) 与实物文件一致：`activities.csv` 共有 **3,852** 个区间，事件 1,505 个、手势 2,347 个，每组 6 类。作者论文的录制与标注章节说明：连续录制由同步视频分段标注。官方公开雷达数据，不公开配套视频。

| 组别 | 类别 | 片段数 |
| --- | --- | ---: |
| Events | standing_up（站起） | 344 |
| Events | sitting_down（坐下） | 342 |
| Events | going_out_of_room（出房间） | 224 |
| Events | entering_room（进房间） | 221 |
| Events | putting_on_cloth（穿衣） | 195 |
| Events | putting_off_cloth（脱衣） | 179 |
| Gestures | drumming（敲击） | 390 |
| Gestures | shaking（摇动） | 360 |
| Gestures | swiping_left（向左挥动） | 436 |
| Gestures | swiping_right（向右挥动） | 384 |
| Gestures | thumb_down（拇指向下） | 368 |
| Gestures | thumb_up（拇指向上） | 409 |

本次检查全部标注：

- 3,852 行引用的文件及受试者均一致；`0 <= start < stop <= T` 全部通过。
- 无重复实例，无同一录制中相互重叠的标注区间；官方六份 split CSV 的并集与 `activities.csv` 完全一致。
- 实例长度 8–169 帧，平均 38.4203 帧；按标称 15 FPS 约 0.53–11.27 秒，均值约 2.56 秒。
- 区间合计覆盖 **147,995 帧，占 63.3771%**；剩余 **85,520 帧，占 36.6229%** 未被这些动作标注覆盖。
- 每名受试者都有全部 12 类，但数量差异很大：subject2 有 995 段，subject8 只有 141 段；subject9 穿衣、脱衣各仅 2 段。

因此，这是可用的官方动作实例监督，比命令标签有更直接的依据；**“精确”在这里指发布方给出了帧级起止索引，不代表本次独立验证了人工边界误差为零。** 缺少原视频不妨碍采用官方监督训练，也不应再要求用户重标整个数据集；只需明确其标注来源和未知的误差容限。

空白区间应设 `label=-100 / annotation_valid=False`，**不能自动沿用上一动作、当作静止、或当作 Keep/Reset 的可靠负例**。官方页面将 4.32 小时称为 effectively annotated，但按实际 CSV 并集和标称 15 FPS：总录制约 4.3244 小时，已标注区间约 2.7406 小时。训练统计应采用实际帧覆盖，不把全文录制视为密集标注。

## 3. 默认划分与连续状态泄漏

| 官方随机分层划分 | Events | Gestures | 合计 |
| --- | ---: | ---: | ---: |
| train | 1,085 | 1,927 | 3,012 |
| validation | 120 | 120 | 240 |
| test | 300 | 300 | 600 |

三种角色都包含 subject1–subject9。训练与验证每组共享 36 条录制；训练与测试每组共享全部 37 条录制。本次用区间比较和独立逐帧占用数组两种方式检查，跨 split **没有重复或相交的标注片段**；所以随机片段分类划分本身并不是错的，但它不能说明跨受试者泛化。

对我们的连续历史模型，若从录制开头无差别回放，再选出 train 端点，状态可能已经读取该录制的 validation/test 片段。仅仅不在 test 位置算 CE 无法避免这种输入暴露。预测目标、前后上下文窗口和归一化统计也都必须遵守角色隔离。

建议主协议按受试者划分，所有来自同一人的录制及片段归到同一角色；对当前 9 人数据最终使用留一受试者或多组按人划分，并在训练人中再留验证人。官方随机划分可作为独立的论文对照，但必须做片段独立状态初始化，或明确隔断跨角色上下文。**本次未固定新的训练划分、未用 test 信号调参。**

## 4. 官方预处理的小样实测

只导入下载包 `process.py` 的 `range_doppler()` 函数，没有执行其 `__main__`，因为原脚本以 `r+` 打开并原地增加数据集。每类选取排序第一条官方 train 标注，共 12 段、521 帧，区间与任何官方 test 标注均不重叠。派生结果保存至服务器独立目录 `HARrad/format_probe_20261016_v1`，已取回本地。

实际计算步骤与包内脚本一致：

1. 去掉每个 chirp 的第一个 sample（脚本注明 chirp sequence number，不能误当作雷达幅值）。
2. 距离加窗及 FFT，选 0.5–4.5 m 的距离区间。
3. Doppler 加窗、FFT、fftshift，取复幅值绝对值。
4. `RD_dB = 20 * log10(abs(RD))`，得到 `160×256`。
5. `MD = RD_dB.sum(axis=range)`，得到 256 维。

这是官方 **dB 图求和** 的 MD 定义；它与“先在线性域求和再取 log”不同，不能无记录地互换。521 帧的 MD 和 12 张中间 RD 均为有限值。12 个派生 NPZ 的 SHA256、形状、帧号和有限值已在本地核验。另在 37 个文件内各抽 3 个合法 train/validation 帧，共 111 帧原始值检查通过；这不是全量 233,515 帧的数值完整性扫描。

[12 类 MD 小样](../outputs/harrad-format-audit-20261016-v1/sample-md.png)；[3 张 RD 小样](../outputs/harrad-format-audit-20261016-v1/sample-rd.png)。已视觉查看；图按各面板独立色阶显示，保留静态 Doppler，不代表跨图幅值可直接比较，也不能用图独立确认类别真值。

两份 ZIP 的已有完整 CRC 日志均显示无压缩错误；本次读取 central directory 并逐项比对包内 CSV/Python 与解压文件，全部一致。没有重新解压或重新全量 CRC 扫描，也没有为 61 GB HDF5 做新的全量哈希。读取前后 HDF5 文件大小及 mtime 均保持不变。

## 5. 接入现有代码需要改什么

| 现有 PARrad 设置 | HARrad 接入要求 |
| --- | --- |
| `md_bins=128` | 原生 MD 为 256；首选保留 256，修改输入层并重训 |
| RD `93×128` | 原生 RD 为 `160×256`；先按原生尺寸建立转换契约，另行记录任何降采样 |
| 2 帧 / Update，0.09 秒 / 帧 | 标称 `1/15` 秒 / 帧；若仍 2 帧 / Update，约 0.1333 秒 / Update |
| h=1/3/5 即 0.18/0.54/0.90 秒 | 此时变成约 0.133/0.400/0.667 秒；必须重设物理 Horizon 并报告实际偏移 |
| 14 类命令代理 | 12 类官方实例；建立新类别表和 12 类头，保留 6 类事件/6 类手势分组指标 |
| 命令间隔作为 label run | 直接保留原实例 ID、`start/stop`；同类连续实例也不可合并 |
| 两雷达、受试者旧 manifest | 单 SISO 雷达数据；新 sensor vocabulary / manifest / subject split |
| 旧配置动作两端各忽略 1 秒 | 不可照搬；会删除大量短动作。新实验单独设定边界容差与 mask |
| 冻结旧 PARrad 编码器/Former | 重新训练 HARrad 雷达表征和语义模块；旧权重仅作为明确的迁移对照 |

主要接入位置：`src/continuous_mmwave/data_parrad.py`、`parrad_config.py`、`models/md_gru_encoder.py`、Stage1/Stage2 的采样与分类头，以及 `v261/contract.py` 的标签/物理时间契约。可以复用模型结构和训练设施，但需要独立 HARrad loader/converter/config，不能只把旧路径改到 HDF5 上。

归一化只用 train 统计；预处理不使用整段未来帧来归一化单个在线输入。未标注帧可在确定按人隔离后用于无标签运动预测，但不可凭空给出语义和边界负标签。预测状态只读当前及过去，不能使用完整未来片段池化特征或按整段时长归一化的事件年龄。

## 6. 建议的重训顺序

1. **完成独立 HARrad 转换和按人数据契约。** 原始 HDF5 只读，RD/MD、逐帧 label/mask、原始 frame index、实例表分别保存；统一验证长度、索引、角色隔离和因果归一化。保留连续流与完整实例两种视图。
2. **先训练 12 类片段语义基线。** 新雷达编码器 → 片段分类头，训练完整官方实例，报告 Macro-F1、各类召回和逐人结果，事件/手势分别列示；加入 RD 输入基线，避免把 MD 单输入限制误判成整个架构失败。小样过拟合与验证只用 train/validation。此处整段汇聚是离线分类，不能当作在线即时能力。
3. **再比较 A-only、B-only、A+B。** 相同按人划分、初始化原则和训练预算，明确各模块是否可训练；这次让雷达编码器及对应分支真正学习新数据，避免重复“冻结 PARrad 语义状态 + 短训 Adapter”的旧问题。未来运动目标必须固定定义，不能随预测模型一起漂移；最终分别检查预测和语义收益。
4. **之后才评估标注边界监督的事件形成。** HARrad 的实例表可支持训练区间上的 Oracle 状态实验和受监督边界试验；先定义动作结束到下一动作之间的未知段、背景与边界容差，再设计 Keep/Birth/Reset。不能仅将每次类别变化当作全部真实实例边界，也不能把 Oracle 回放等同自主 Reset 已验证。

这次检查使我们能够从“命令弱监督诊断”转向“官方动作实例监督训练”。它不会自动证明换数据后联合模型一定更好；新语义基线和后续受控联合实验将直接回答这个问题。

## 7. 核验与可复查文件

- `audit_native.py` / `audit_format_20261016_v1.json`：服务器结构、CSV、区间和原始抽样审计。
- `verify_local.py` / `local-verification.json`：独立占用数组复算、全量区间与角色隔离检查，`passed=true`。
- `official-datasets.py`：作者公开仓库固定 commit `67f8da31dfddf8f60906f20920a485cd84a433b5` 的加载器。
- `probe_native.py` / `format_probe_20261016_v1/probe.json`：每类一段官方 FFT 小样。
- `render_probe.py` / `probe-verification.json`：12 个派生数组的哈希、维度、帧号和有限值核验，`passed=true`。
- `labels/`：本次取回的官方 CSV 原件，只用于结构和划分审计；包含 test 元数据，未读取 test 雷达信号。

本次未修改生产训练代码、旧实验、原标签或原 HDF5；新增模型优化步数为 0。

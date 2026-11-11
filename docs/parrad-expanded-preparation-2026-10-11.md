# PARrad 新下载数据准备（2026-10-11）

目标：将服务器新增且完整下载的session转换为现有训练器可直接读取的canonical RD/MD NPZ；复用原phase1数据，保留14类ID、0.09秒时间格与原窗口身份。不在本轮启动ResNet或Qwen训练。

执行设计：沿用`parrad_convert.convert_recording`，新增`parrad_expand`批量入口，只读取完整最终文件。冻结一次下载清单，检查探测字节数、CSV SHA及下载完成侧车；排除`.part`。不读取巨大cube，仅转换RD和MD，并逐记录重读所有必需数组、独立重建标签、比较原始RD/MD的完整值摘要。生成总manifest、每受试者manifest、兼容旧05/06划分、12秒/6秒描述窗口索引、按类和按受试者统计与校验清单。

边界：H5全文件包含cube，约数百GB；其SHA记录自下载完成侧车，不重新声称独立校验全H5。实际使用的全部RD/MD值单独核对。时间原点沿用首条command，时间戳由0.09秒周期生成，command-active标签不是实际身体动作边界。源文件不能确认环境时保留`parrad`，不猜Homelab/Hospital。

旧38份记录原样复用，通过相对路径引用phase1；不删除或覆盖旧文件。`recording_id`显式继承旧manifest路径，避免搬迁清单导致原窗口ID改变。不同传感器保留各自帧数与时间格，不补造缺失传感器。部分受试者中已经完整的session可入总清单，完整性状态另外报告。

划分说明：`splits/legacy_05_06`只是便于继续旧实验的兼容视图（05validation/06test，其余已备好受试者train），06曾多次测试，不是新保留测试集。新正式研究划分尚未锁定；按受试者清单可重新分组，不随机拆同一人的session或传感器。

转换于北京时间2026-10-11 20:15:45启动，20:27:26完成，独立审计随后通过。仅处理启动时固定清单内的完整session，后续下载不自动并入这一版。

| 项目 | 已完成 |
| --- | ---: |
| 受试者 | 15人，其中13人全部session齐全 |
| session | 45 |
| 传感器recording | 89：旧38复用，新增51 |
| 传感器帧数 | 599,799 |
| session时长，双传感器不重复计时 | 7.5895小时，取各session最长传感器时长 |
| 传感器时长之和 | 14.9950小时，不能当作独立采集时长 |
| NPZ总大小 | 28,876,811,166 bytes，约26.89GiB |
| 实际新增NPZ | 16,654,456,266 bytes，约15.51GiB |
| 12秒/6秒描述窗口 | 8,107，其中8,077个满足旧严格完整窗判据 |
| 单/双/三动作窗口 | 386 / 2,745 / 4,976 |

## 可用入口

服务器数据根目录：

```text
/home/xibeidaxue/newdisk2/WJ/PARrad/canonical/offline_expanded_20261011_v1
```

| 相对路径 | 用途 |
| --- | --- |
| `manifest.jsonl` | 全部89份记录，分类训练可自行切窗，不受描述窗口过滤限制 |
| `subjects/subject_NN.jsonl` | 每人独立清单，用于重新制定按人划分 |
| `splits/legacy_05_06/{train,validation,test}.jsonl` | 旧实验兼容清单；不作为新研究已锁定划分 |
| `label_map.json` | 原有14类ID和命令文本 |
| `action_phrases.json` | 动作ID到英文动词短语 |
| `caption-windows-12s-stride6.jsonl` | 稳定样本ID、recording ID、帧区间、动作序列和弱监督文本 |
| `converted/subject_NN/setNNN/recordings/*.npz` | 51份新记录 |
| `download-snapshot.json` / `conversion-plan.json` | 下载快照、排除原因、来源、字节数、侧车摘要、文件状态 |
| `recording-audit.json` | 89份NPZ摘要、原始RD/MD值摘要、逐类帧数和来源核对 |
| `summary.json` / `artifact-manifest.json` / `status.json` | 统计、逐文件摘要、完成状态 |

旧38份记录通过相对路径指向相邻的`../phase1/recordings/`，因此此目录不是独立可搬迁的数据包。若迁移到其他机器，需要同时带上相邻`phase1`目录或显式重写清单并重新校验。源H5不是运行NPZ加载器所必需，但应保留用于来源追溯。

每份NPZ包含：

| 数组 | 形状 | 类型 |
| --- | --- | --- |
| `rd` | `[T,1,93,128]` | float32 |
| `md` | `[T,128]` | float32 |
| `labels` | `[T]`，原14类ID | int64 |
| `timestamps` | `[T]`，`arange(T)*0.09` | float64 |

保存原始RD/MD值，不预先按全体数据拟合标准化。RD-only训练直接使用`rd`；MD保留供后续对照使用。此处不将两个传感器堆为输入通道，两者是同一session的独立视角，划分时必须跟随同一受试者。

服务器现有seer环境已验证以下加载方式；先使用封存代码目录的`src`作为`PYTHONPATH`：

```bash
export PYTHONPATH=/home/xibeidaxue/SEER/.audit-staging/parrad-expanded-code-20261011-v1/src
/home/xibeidaxue/SEER/.conda/envs/seer/bin/python - <<'PY'
from pathlib import Path
from continuous_mmwave.radar_text.data import OfflineCaptionDataset, load_phrases, collate_windows
root = Path('/home/xibeidaxue/newdisk2/WJ/PARrad/canonical/offline_expanded_20261011_v1')
dataset = OfflineCaptionDataset(root/'subjects/subject_01.jsonl', load_phrases(root/'action_phrases.json'))
batch = collate_windows([dataset[0], dataset[1]])
print(batch['rd'].shape, batch['lengths'], batch['actions'])
# RD-only模型取batch['rd']，并按lengths屏蔽padding。
PY
```

实际总清单首尾样本批次已通过：RD`[2,134,1,93,128]`、MD`[2,134,128]`、labels`[2,134]`。12秒通常对应133或134帧；索引按时间戳切分，不强行拉伸。长度8至不足12秒的30个尾窗保留并标记`full_window=false`；严格比较可按此字段筛选。现有严格完整窗判据为`帧数*0.09+0.09 >= 12-1e-6`，沿用旧实验。

## 受试者库存

| 受试者 | 完成session/应有 | recording | 描述窗口 |
| --- | ---: | ---: | ---: |
| 01 | 3/3 | 6 | 538 |
| 02 | 4/4 | 8 | 716 |
| 03 | 4/4 | 8 | 742 |
| 04 | 4/4 | 8 | 729 |
| 05 | 4/4 | 8 | 702 |
| 06 | 3/3 | 6 | 484 |
| 07 | 3/3 | 6 | 540 |
| 08 | 4/4 | 8 | 686 |
| 09 | 3/3 | 5 | 490 |
| 12 | 4/4 | 8 | 748 |
| 13 | 1/4 | 2 | 176 |
| 16 | 2/2 | 4 | 400 |
| 17 | 2/2 | 4 | 398 |
| 18 | 2/2 | 4 | 404 |
| 19 | 2/4 | 4 | 354 |

09在检查时已经3/3；09/set000源文件只有TI_xWR68xx，另44个session包含两种传感器。保留源文件的真实库存，没有补造缺失视角。源H5内同一个传感器的RD/MD帧数一致，不同传感器之间可能略有差别。

快照排除32个session：3个仍有`.part`（10/set000、13/set001、19/set002），29个缺少完整文件。13和19已完成的session可用，其余尚未纳入。下载继续进行，不能把该清单当成服务器之后的实时下载状态。

兼容视图的描述窗train/validation/test为6,921/702/484；严格完整窗为6,895/700/482。train含其余13人，validation仅05，test仅06；session及传感器均不跨受试者拆分。

描述索引排除了148个不足8秒候选窗和801个超过3动作候选窗，没有未知标签候选窗。该过滤只作用于描述任务；完整逐帧记录仍保留全部动作证据。弱监督动作文本由标签游程生成，连续相同类合并，重复出现但被其他动作分隔的类保留。

## 验证与来源

初始服务器原生37项测试全部通过，包括真实HDF5转换、旧记录复用、窗口ID和数据加载。最终只读审查发现摘要块边界处可能漏检被截断的旧NPZ，因此新增257帧源/256帧保存的回归：修正前确实失败，修正后显式检查源数组形状。修正模块单独上传并运行38项原生回归，全部通过。本地interview为6passed/2skipped，两个真实HDF5测试因未安装h5py跳过，真实验证以服务器为准。

实际批量转换使用的142文件源码封存包保持不变，SHA256为`916b904be86f80c26e2a1532d924313056b7db2579ef3207b0c4a581faa69a75`。修正模块另行封存，不改写已执行源码。对于这批产物，后置独立审计额外检查了全部89份记录的原始数组形状，确认不存在截断，并检查45个源session的文件状态与CSV摘要。

独立审计通过：130个数据产物文件摘要、142个执行源码文件摘要、旧38个NPZ与历史P1 SHA完全一致；8,107个窗口的起止帧/动作/文本/ID/完整窗标记可独立重建；05/06窗口ID、动作与完整窗标记和历史P1一致。按人清单和兼容划分的路径解析、穷尽性、受试者互斥也通过。

本地取回目录：`outputs/parrad-expanded-preparation-20261011-v1/`。其中`dataset/`仅含元数据，雷达NPZ仍在服务器；`independent-audit.json`是服务器独立审计，`native-tests.log`及`supplementary/`保留测试、审查RED/GREEN与修正源码，`source/`保留实际执行包，`audit-source/`保留独立审计和导出脚本。元数据导出包SHA256为`eb256a930cf7b9f124e316b423c740e80bed81a96e0dd95822b6dc82657dae89`，本地验证92个导出文件摘要全部一致。

数据准备入口为`python -m continuous_mmwave.parrad_expand --raw <raw目录> --existing <旧canonical目录> --expected-sizes <probe_h5.json> --phrases <动作短语.json> --output <全新目录>`。输出目录必须不存在；后续新下载完成后应创建新版本，复用本批master manifest所引用的数据，而不是覆盖本批或旧phase1。下次执行应使用已修正的当前模块，并重新封存执行源码。

本轮没有训练ResNet18或Qwen。正式从头训练前仍需确定新的受试者划分；当前05/06仅用于旧实验兼容，不能作为新的未观察测试集。

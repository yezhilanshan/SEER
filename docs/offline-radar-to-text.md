# 离线 Radar-to-Text MVP

新入口 `continuous_mmwave.radar_text` 独立于 v2.5/v2.6。推理只读 RD 和 MD，不输入动作类别或事件状态。分类头仅用于预训练及对照实验。

```text
RD [B,T,1,93,128] → 3-layer 2D CNN → [B,T,128] ┐
MD [B,T,128] → local [128,9] → 2D CNN → [B,T,128] ┘
→ concat → 2-layer BiGRU (128 per direction)
→ ordered temporal pooling [B,8,256]
→ MLP → Qwen hidden size → autoregressive description
```

两路 CNN 使用 GroupNorm + GELU + global average pooling。MD 每个时间片覆盖当前帧左右各 4 帧，按真实长度复制边缘；不会把整张谱压成一个向量。BiGRU 使用 packed sequences，池化只处理真实帧。默认对每个输入窗口、每个分支独立做 signed-log 和 z-score；不会使用其他受试者统计。它可能削弱绝对能量属性，研究幅度时应对照 `normalization: none` 或另行设计训练集统计归一化。

## 环境与测试

依赖已列在项目 `pyproject.toml`，没有新增依赖。测试使用 `interview` 环境。当前本机可直接运行：

```powershell
$env:PYTHONPATH = 'D:\lab\SEER\src'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& 'C:\User\anaconda\envs\interview\python.exe' -m pytest tests/test_radar_text.py -q
& 'C:\User\anaconda\envs\interview\python.exe' -m continuous_mmwave.radar_text smoke --output-dir outputs/radar-text-smoke
```

`smoke` 生成三个独立的合成受试者，实际运行 stage1、冻结 tiny Qwen 的 align、checkpoint 重载、无标签条件生成和分类模板对照。tiny Qwen 为随机初始化，不下载权重，不代表任何语言或雷达识别效果。正式模型默认为 Qwen2.5-0.5B-Instruct；也可将 `llm.name` 改为 1.5B 或本地目录，projector 自动读取 hidden size。

## 准备 PARrad

2026-10-11新增下载转换已完成：[数据准备报告](parrad-expanded-preparation-2026-10-11.md)。新canonical位于服务器`/home/xibeidaxue/newdisk2/WJ/PARrad/canonical/offline_expanded_20261011_v1`，含15人/45session/89份传感器记录，原始RD+MD、固定14类ID、逐帧标签和描述窗口索引均已校验。旧38份通过相邻phase1路径复用，搬迁需同时保留它们。RD-only训练取`rd`即可；新的正式受试者划分尚未锁定，兼容05/06视图不能视为新的未观察测试集。

沿用现有 `continuous_mmwave.parrad_convert` 将原始数据转为 canonical NPZ，再使用 `continuous_mmwave.parrad_split` 按受试者分组。完整 RD 必须保留，旧 MD-only 占位 RD 无法用于双分支模型。

每个 NPZ：`rd: float32[T,1,93,128]`、`md: float32[T,128]`、`labels: int64[T]`、`timestamps: float64[T]`。时间戳单位为秒，严格递增、近似均匀；含时间缺口的 recording 应先拆开。manifest JSONL 每行示例：

```json
{"path":"../recordings/s01-session1-60ghz.npz","subject_id":"s01","session_id":"session1","sensor_id":"60ghz","environment_id":"homelab"}
```

train/validation/test 受试者必须互斥，同一受试者的不同传感器也必须同组。训练、评估在切窗前检查 supplied split 元数据。模型只使用一个传感器的对齐序列，不把不同传感器当作 RD/MD 两个模态。

创建 `data/parrad/action_phrases.json`，按真实 `label_map.json` 的 class ID 编写英文第三人称动词短语，覆盖全部实际类别（不假定任何固定 14 类 ID）。下面仅展示格式，不能直接套用到真实标签：

```json
{"0":"walks toward the bed","1":"sits down on the bed","2":"lies down on the bed"}
```

相邻重复标签合并，A→B→A 保留。默认 12 秒窗口、6 秒步长，末尾不足 8 秒的窗口跳过；超过 3 个标签 run 或含 -100 的窗口跳过并记录计数。当前窗口可能截断标签段，因此目标描述的是窗口内可见活动顺序，不宣称动作完整发生，也不自动声称“开始”“完成”。原始 command-active 标签仍是弱监督，不能作为精确身体动作边界。

`prepare` 可导出具体窗口和模板目标，供抽样审计或人工改写：

```powershell
python -m continuous_mmwave.radar_text prepare --config configs/radar-text-align.yaml --split train --output outputs/radar-text-train-captions.jsonl
```

人工目标格式为每行 `{"sample_id":"s01/session1/60ghz/abc123def456:0:134","caption":"The person ..."}`；使用实际导出的 sample_id。ID 中的 recording hash 默认取自 manifest 的 path；需要跨目录搬迁保持 ID 时，在 manifest 明确提供稳定的 `recording_id`。在 YAML `data.train_captions`、`validation_captions`、`test_captions` 中分别指定。重复 ID、空文本和不属于当前切窗参数的 ID 会报错。人工 caption 可以补充真实可验证的速度或停顿，但动作顺序代理指标仍依据原始标签。

## 训练

先修改三个 `configs/radar-text-*.yaml` 中的数据路径和硬件配置。YAML 数据和输出路径相对于 YAML 文件解析。默认使用 GPU；CPU 改 `training.device: cpu`，LLM 改 `dtype: float32`。默认 `local_files_only: true`；准备本地权重后可直接填路径，或主动改为 false 允许 Transformers 下载。

以下 `python` 指当前选定环境的 Python。训练命令：

```powershell
# 1. 双分支雷达侧预训练，有效帧交叉熵
python -m continuous_mmwave.radar_text train --stage stage1 --config configs/radar-text-stage1.yaml

# 2. 冻结 LLM，只训练 encoder + projector
python -m continuous_mmwave.radar_text train --stage align --config configs/radar-text-align.yaml --checkpoint outputs/radar-text-stage1/best.pt

# 3. 对齐收敛后继续 LoRA（环境需要 PEFT）
python -m continuous_mmwave.radar_text train --stage lora --config configs/radar-text-lora.yaml --checkpoint outputs/radar-text-align/best.pt
```

align 可以从随机雷达侧开始，但正式实验建议加载 stage1。LoRA 必须加载 align/lora checkpoint，作用于 Qwen attention q/k/v/o 投影。整个流程不需要 Event Former 或额外跨模态 attention。

radar embeddings 插入模型自带 chat template 的 user 消息中。只有 assistant 文本和回答结束标记参与自回归 CE；prompt、radar tokens 和 padding 标签均为 -100。冻结 LLM 不使用 `no_grad`，梯度仍通过 LLM 输入回到 encoder/projector。

输出 `best.pt`（validation loss 最小）、`last.pt` 和 `training.json`。checkpoint 保存 encoder/projector/可选 LoRA，不重复保存预训练 LLM；因此推理仍需同一原始模型及 tokenizer。`--checkpoint` 为权重 warm start，优化器和 epoch 从头开始，不是精确中断续训。tiny smoke 例外保存随机 LLM 以便复现。

训练时随机打乱 recording 和各 recording 内的窗口，集中处理同一个 recording，配合单记录缓存，避免每个小窗口都重新加载整份 NPZ。checkpoint 继承此前各阶段所有训练受试者及验证选优受试者的集合：训练受试者不能进入 validation/test，历史验证受试者可以继续用于 validation，但不能进入 test；分类基线也检查自己的来源和短语词表。

## 独立受试者评估与无标签推理

```powershell
python -m continuous_mmwave.radar_text evaluate --config configs/radar-text-align.yaml --checkpoint outputs/radar-text-align/best.pt --split test --baseline-checkpoint outputs/radar-text-stage1/best.pt --output outputs/radar-text-test.json
python -m continuous_mmwave.radar_text infer --checkpoint outputs/radar-text-align/best.pt --input data/my-window.npz --device cuda
```

infer 输入 NPZ 只需 RD 和 MD，必须已经按时间对齐，不能把多个 recording 拼成一个窗口。`--model-name` 可重定位同一模型权重目录。建议先剪出 8–16 秒窗口；本入口不自动切块，不支持实时输入。

评估输出生成文本、标签参考、可选分类模板 baseline、动作多重集 precision/recall、完整序列 exact match、归一化 LCS 顺序分数、模板 exact match、无匹配比例，以及每受试者和单/多动作分组。动作提取按英文短语匹配，不会可靠识别同义改写或否定；这些指标仅用于弱监督诊断，应审阅原始输出并追加人工评价。未计算 BLEU，也不把模板命中解释成细粒度描述能力。

## 当前验收边界

2026-10-11 已冻结原始Stage1/Align/Qwen权重完成[系统性错误归因](radar-text-error-attribution-2026-10-11.md)。诊断只训练四个线性读出和8个固定前缀向量，包含完整14类混淆矩阵、平滑模板和雷达对照。Align后特征整体未退化；稳态分类和生成覆盖均有问题。真实异序样本与完整短动作数量不足，人工雷达时段复核队列保持pending，不能把短语代理指标解释为完整文本事实性或可靠顺序理解。

同日完成[冻结分类头H0–H4消融](radar-text-head-ablation-2026-10-11.md)。零更新原头与旧探针预测复现通过；控制缓存与训练设置后，原优化器组延长训练有收益，类别权重收益主要见于验证受试者，标准化单独收益较小。全部权重、训练曲线、逐类指标和复现入口见报告；本轮没有改变雷达编码器或语言生成模型。

随后完成[同滑动窗口B0–B3强基线比较](radar-text-strong-baselines-2026-10-11.md)：在相同RD/MD输入上新提取特征、重新执行冻结Qwen，避免与非重叠缓存指标混比。validation选择不做中值滤波、最短预测段0.6秒；H1/H4各自validation选优也得到同参数。原484窗口B0/H1/H4/Qwen完整序列准确率为11.78%/18.18%/18.39%/25.00%，严格482个完整窗口为11.83%/17.84%/18.05%/24.69%。validation选出的H1与Qwen差距降至6.82个百分点，分类Recall/LCS更高，研究重点仍应包含生成漏动作。所有参数冻结，原权重和数据SHA核验通过。

复现入口为 `scripts/run_radar_text_strong_baselines.py --pilot <原pilot目录> --heads <分类头消融目录> --diagnostics <原诊断目录> --output <新输出目录>`，服务器沿用现有seer环境；源码封存包、原始结果manifest、逐窗口预测、logits和本地独立重算清单在 `outputs/radar-text-strong-baselines-20261011-v1/`。运行前按服务器现有环境与封存源码核验；新特征缓存留在服务器。`scripts/report_radar_text_strong_baselines.py` 生成报告/图/CSV，`scripts/verify_radar_text_strong_baselines.py` 从导出logits与时间戳重算平滑、逐窗口指标及validation网格。

[P0冻结Align Encoder公平对照](radar-text-align-readout-p0-2026-10-11.md)进一步只训练Align H1/H4，训练与选择上下文保持原非重叠12秒协议，评价重新编码相同滑动窗口。H1非重叠test Macro-F1从Stage1的49.28%升至Align的51.74%；严格482窗Align H1完整序列18.88%，低于Qwen24.69%，Recall65.91%高于Qwen46.06%。Count MAE分别1.098与0.336：分类更多余动作，生成仍存在床相关覆盖损失。原484窗Align H1/Qwen为19.21%/25.00%。同编码器比较仍是完整读出流程对比，未隔离LLM本身或扩展完整2×2。

P0入口：`scripts/run_radar_text_align_readout.py --pilot <原pilot目录> --heads <分类头消融目录> --diagnostics <原诊断目录> --previous <同窗口强基线服务器目录> --output <新输出目录>`。采用已有seer环境，无安装。输出与新分类头在 `outputs/radar-text-align-readout-p0-20261011-v2/`；v1预检查失败记录保留。报告/封装/独立审计分别使用 `scripts/report_radar_text_align_readout.py`、`scripts/archive_radar_text_align_readout.py`、`scripts/verify_radar_text_align_readout.py`；审计同时重建平滑、重算序列/计数主指标、核对旧B3输入摘要与Stage1 H1 logits。原Align和Qwen权重不更新，原始雷达及40份新特征留服务器。

[P1动作证据保留与动作数量校准](radar-text-p1-evidence-count-2026-10-11.md)完成了原B1/B3逐窗编辑分解、29/57独对文本复核、冻结H/Z/P的六个Presence/Count/Ordered探针及概率/HMM/预测数量段Viterbi对照。严格482窗HMM25.52%、预测数量27.18%均高于原Qwen24.69%，Count MAE降至0.216；Oracle真数量34.02%明确排除正式性能。H/Z/P Ordered有限探针未支持H→Z整体退化，Z→P读取较弱；不同输入维度、参数量和优化难度限制结论。数量分布严重偏向3，报告附事后训练多数数量诊断及逐计数Recall。全部操作未更新Align Encoder/Projector/Qwen。

P1入口：`scripts/run_radar_text_p1.py --pilot <原pilot目录> --p0 <P0服务器结果目录> --output <全新输出目录>`，依赖同封存源码的`source-manifest.json`，先核验所有原始输入与源文件。新特征取原始滑窗逐一重编码，只有train/validation参与训练和选优，锁定后提取test。七个小读出均20epoch、seed17；因子化线性rank32与GELU探针分别报告容量。`archive_radar_text_p1.py`校验并封装小型结果，H/Z/P缓存保留服务器；`verify_radar_text_p1.py`独立重算指标/选择/输入，`verify_radar_text_p1_cache.py`只读重建节点/探针/校准并做明确事后多数数量诊断，`report_radar_text_p1.py`生成报告/图/CSV/全部86窗记录。本地产物在`outputs/radar-text-p1-20261011-v1/`，原结果与衍生审计分开封存；真实雷达时段人工复核尚未完成。

本地合成 fixture 和 smoke 用于验证张量、loss、梯度、保存加载和调用链路。2026-10-10 已在服务器使用真实 PARrad RD/MD 与现有完整 Qwen2.5-0.5B 权重完成 stage1 3epoch + 冻结 LLM 对齐 3epoch。独立 subject06 的 484 个测试窗口，完整动作序列正确率 25.00%，本轮分类模板基线 11.16%；动作 recall 46.28% 低于基线 55.51%，仍有明显漏动作和类别混淆。详见[真实 pilot 结果与产物](radar-text-server-pilot-2026-10-10.md)。这些是模板短语代理指标，单动作样本和独立测试受试者不足，不能证明细粒度描述或跨环境泛化。

2026-10-10 本地验证：新增模型与现有 PARrad 数据/划分测试共 **39 passed, 1 skipped**；完整 CPU smoke 已通过。LoRA 测试因 `interview` 环境缺少 `peft` 跳过；原转换测试因缺少 `h5py` 无法收集。随后使用服务器现有环境的原生验证 **44 passed**，包含 LoRA 与 HDF5 转换，未安装或修改环境。本地 smoke 在 `outputs/radar-text-smoke/`，真实实验取回产物在 `outputs/radar-text-server-pilot-20261010-v1/`。

Qwen 加载和 chat template 接口参考官方 [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) 与 [Transformers Qwen2 文档](https://huggingface.co/docs/transformers/model_doc/qwen2)。

## RD-only X-Fi 迁移入口

`continuous_mmwave.radar_text.rd_transfer` 提供独立的单通道ResNet18骨干，不改变原RD+MD默认训练/推理入口。`make_backbone('xfi-resnet', <纯张量.pt>)`输出每帧512维特征；17通道首层权重求和为单通道，移除原55类头及Softmax，全部参数与BN缓冲冻结。`normalize_rd`按真实窗口有效帧归一化；`TemporalReadout(512)`包含新256维投影、BiGRU和逐帧/窗口多任务头，不包含Qwen或原Projector。

本地原权重是自定义类完整pickle；`scripts/package_rd_only_xfi.py`使用受限allowlist导入并校验全部张量，导出纯张量供服务器旧Torch安全加载。没有`weights_only=False`回退，不依赖torchvision，也不修改原文件。

首轮实验入口：`scripts/run_rd_only_xfi_transfer.py --pilot <pilot> --p1 <P1目录> --xfi <安全导出.pt> --output <新目录>`，使用封存代码的`source-manifest.json`。三组为已有Stage1 RD-CNN、随机ResNet18、X-Fi ResNet18；只训练新的时序与读出模块，每组20epoch。在validation锁定轮次、HMM与预测数量参数后才提取test。`--smoke --device cuda`验证训练链路。

运行后的原始产物由`archive_rd_only_xfi_transfer.py`封存，9份大特征缓存留服务器；`verify_rd_only_xfi_transfer.py`独立重算指标与选择，`verify_rd_only_xfi_cache.py`重建真实RD特征与读出并校验训练统计；`report_rd_only_xfi_transfer.py`生成报告/混淆矩阵/CSV。此入口是冻结特征迁移诊断，不能直接替代已对齐模型的编码器并沿用旧Projector；正式替换需重新训练时序和语义接口。

[首轮结果已完成](rd-only-xfi-transfer-2026-10-11.md)：R0/R1/R2测试Macro-F1为62.01%/62.25%/62.52%，预测数量模板Exact为32.78%/31.33%/33.61%；验证集主组仍选R0，预训练未显示稳定优势。全部三组读出、逐帧logits、逐窗文本/动作、14类混淆矩阵及审计见`outputs/radar-text-rd-only-xfi-20261011-v1/`。本轮停止于冻结迁移诊断，没有更新原语言模型或自动进入微调。

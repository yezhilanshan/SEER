# 想法六：HARrad CIF/LIF事件门

这是独立的 `continuous_mmwave.event_gating` 包。原 `harrad`、`radar_text`、EventFormer和旧训练入口均不改动；旧模型文件不被写回。

本轮实现事件门与对照，不包含LLM训练、语言日志、骨架预训练或周期动作次数估计。以[批准设计](superpowers/specs/2026-10-26-harrad-cif-lif-design.md)为依据。

## 数据与结果的含义

HARrad官方实例是零基 `[start,stop)`，约36.6%的自然时间轴未标注。首版从同一录制顺序抽取1–6个官方实例，删除中间未知间隔构造训练序列。保留相邻同类实例：`[0,0,1]`不会变成`[0,1]`。

构造序列使用冻结CNN的64维逐帧特征，拼接后重新回放原单向GRU产生128维特征。不能直接拼接 `continuous_hidden`，也不使用 `oracle_hidden`。官方边界被用于构造样本，所以不能声称本实验完全没有利用边界信息。

自然流评测读取整段 `continuous_hidden`，只在录制开头初始化状态，包含全部未知区间。CIF/LIF/对照使用同一前端；训练subject1–6、validation subject7，test8–9不读取。时间一律是15 FPS标称时间，不是硬件精确时间戳。

## 本地运行

不需要安装新依赖。在PowerShell中：

```powershell
$env:PYTHONPATH = 'D:/lab/SEER/src'
& 'C:/User/anaconda/envs/interview/python.exe' -m continuous_mmwave.event_gating smoke --output-dir outputs/harrad-cif-lif-smoke-v1
```

smoke使用小型人工特征，依次运行CIF、LIF蒸馏、LIF微调、逐帧分类对照，包含优化器更新、best读回和自然流接口。所有结果带 `synthetic_only=true`，不代表真实HARrad识别效果。

测试：

```powershell
& 'C:/User/anaconda/envs/interview/python.exe' -m pytest tests/test_event_gating_core.py tests/test_event_gating_model.py tests/test_event_gating_data.py tests/test_event_gating_metrics.py tests/test_event_gating_runner.py -q --basetemp .superpowers/sdd/2026-10-27-harrad-cif-lif/pytest
```

所有输出目录必须是新目录；重复运行换目录，防止覆盖旧实验。

## 真实HARrad准备

配置：[harrad-cif-lif-v1.json](../configs/harrad-cif-lif-v1.json)。默认指向现有服务器资产；服务器文件当前状态仍需 `prepare` 实测。路径不存在、SHA不符或缺少来源证明时会停止，保留failed记录，不自动下载或重建旧模型。

在已有服务器Python环境和完整源码目录内，设置 `PYTHONPATH=src` 后运行：

```bash
python -m continuous_mmwave.event_gating prepare --config configs/harrad-cif-lif-v1.json --device cuda --output-dir outputs/harrad-cif-lif-prepared-v1
```

准备过程验证前端、normalizer、feature-manifest及流式验收记录的锁，核对每个实际读取数组的SHA、shape和有限值，并验证冻结前端参数没有变化。派生NPZ含features、frame_labels、targets和原始source_frames；`prepared.json`记录完整来源。

如果要在本地使用真实资产，复制配置并修改路径，保持正确SHA；feature-manifest内的artifact_dir也必须指向实际可用的本地缓存，不能只修改最外层清单路径。

## 训练与阶段转换

```bash
python -m continuous_mmwave.event_gating train --config configs/harrad-cif-lif-v1.json --prepared outputs/harrad-cif-lif-prepared-v1/prepared.json --stage cif --device cuda --output-dir outputs/harrad-cif-v1

python -m continuous_mmwave.event_gating train --config configs/harrad-cif-lif-v1.json --prepared outputs/harrad-cif-lif-prepared-v1/prepared.json --stage lif-distill --checkpoint outputs/harrad-cif-v1/best.pt --device cuda --output-dir outputs/harrad-lif-distill-v1

python -m continuous_mmwave.event_gating train --config configs/harrad-cif-lif-v1.json --prepared outputs/harrad-cif-lif-prepared-v1/prepared.json --stage lif --checkpoint outputs/harrad-lif-distill-v1/best.pt --device cuda --output-dir outputs/harrad-lif-v1

python -m continuous_mmwave.event_gating train --config configs/harrad-cif-lif-v1.json --prepared outputs/harrad-cif-lif-prepared-v1/prepared.json --stage frame-baseline --device cuda --output-dir outputs/harrad-frame-baseline-v1
```

- CIF：原始alpha总量监督N，缩放alpha产生N个训练token；推理不输入真实事件数。
- LIF：按秒精确漏积分、真正的ReLU死区、受限可学习radius、低电位慢更新语义参考、自适应阈值。软复位为默认；硬复位可配置。
- LIF数量监督是N−1个内部切换，EOF另关闭尾事件。训练为逐帧软门控CTC代理与代理梯度，不把硬发火不足的样本静默置零。它不等于已经解决训练/推理差异。
- 蒸馏教师由CIF自由推理生成，采用最终事件的内部终点，排除EOF；教师数量不符样本保留并统计。
- frame-baseline是显式使用逐帧标签的强监督分类对照；因果候选持续1/4/8帧，validation选定，不能伪称与弱监督主模型监督量相同。

检查点选择：构造validation序列exact优先、count MAE次之、较短对照确认间隔及较早epoch优先。自然流结果作为诊断单独报告，不用其结果临时改选优规则。

每次训练保存initial、逐epoch、best、last、final、优化器/RNG、输入和受试者来源、history以及best重放。只有新目录允许写入。恢复训练使用同阶段checkpoint和 `--resume`，epochs配置是该阶段总预算，不是额外轮次；不要更换数据清单。

## 评测

```bash
python -m continuous_mmwave.event_gating evaluate --checkpoint outputs/harrad-lif-v1/best.pt --prepared outputs/harrad-cif-lif-prepared-v1/prepared.json --device cuda --output-dir outputs/harrad-lif-evaluation-v1
```

构造序列报告实例序列exact、count MAE、编辑距离、过分割比例及相邻同类实例召回。后者的支持集是参考序列中至少有一个相邻同类邻居的实例，用一对一有序类别匹配统计召回；例如`[1]`相对`[1,1]`为1/2，无此类参考实例时为null。自然录制输出完整events JSONL和每录制±2/4/8帧边界指标。

事件IoU使用原始预测区间，不能裁剪到真值后评分。匹配最大化类别一致且IoU≥0.5的匹配数量，再最大化总IoU。未匹配预测分为完全已知（FP）、完全未知（unscored_unknown）和混合覆盖（unscored_mixed）；后两类数量、时长、覆盖情况保留。所有未匹配官方实例为FN。该Precision/F1属于部分标注口径，不是完整时间轴真值下的准确率。

定位时间和实际确认时间分别保存。初始化/EOF强制边界不算预测；CIF为处理EOF低质量尾部，最多暂存一个已形成事件，到下一个事件形成或EOF再提交，所以其确认时间包含这段等待，不能伪造低延迟。

## 在线推理

从未重置的连续128维特征：

```bash
python -m continuous_mmwave.event_gating infer --checkpoint outputs/harrad-lif-v1/best.pt --prepared outputs/harrad-cif-lif-prepared-v1/prepared.json --input /path/to/continuous_hidden.npy --device cuda --output-dir outputs/harrad-lif-inference-v1
```

特征模式只接受匹配的prepared清单中已核验的自然录制及其SHA；没有来源的数组、oracle_hidden或另一前端特征都会拒绝。新录制可使用同一冻结前端的RD模式，不将任意128维数组冒充连续特征。

从预处理RD `[T,40,64]`（不是原始ADC）：

```bash
python -m continuous_mmwave.event_gating infer --checkpoint outputs/harrad-lif-v1/best.pt --input /path/to/rd.npy --rd --config configs/harrad-cif-lif-v1.json --device cuda --output-dir outputs/harrad-lif-rd-inference-v1
```

RD路径复用原CNN、normalizer和GRU，只在录制开头初始化前端；事件门切换不重置GRU。JSONL逐块写入磁盘，前端状态和短缓冲有界，不在内存中保存全程特征。

默认关闭回溯，先验证门本身。开启 `gate.retrospective` 时从低电位点回溯，缓存不足退回发火时刻并标记fallback。192帧缓冲在15 FPS约12.8秒；更长事件用累计和保存内容，不丢弃老帧。

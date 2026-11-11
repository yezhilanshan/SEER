# HARrad 片段语义基线实施计划

**Goal:** 在官方动作实例和按人隔离数据上，从头训练 MD-GRU / RD-CNN-GRU，验证新数据能否支撑语义学习。

**Architecture:** 独立 `continuous_mmwave.harrad` 包负责实例契约、只读原始数据转换、片段模型和训练执行。复用现有 `CausalMDGRUEncoder`；不改变旧 PARrad 实验及其生产入口。用户已批准报告中的接入与基线训练，本轮顺序执行。

**Spec:** [HARrad 格式与接入审计](../../harrad-format-audit-2026-10-16.md)。本计划给出本轮具体参数，不启动 A/B 联合或 Reset。

**Tech Stack:** Python 3.10、NumPy、h5py、PyTorch；服务器现有 seer 环境，本地 interview 环境。

## 固定协议

- train: subject1–6，共 3,018 实例；validation: subject7，511 实例；test: subject8–9，323 实例。所有 12 类均在各角色中存在；按帧号保留独立 instance ID，同类也不合并。
- 官方 CSV 的所有行只作元数据契约核对；转换、归一化、预检、训练和选择都只读取 train/validation 雷达数组。测试 manifest 只含源路径和实例元数据，不生成测试特征、不计算测试指标。先前格式审计曾抽读 subject8/9 各 3 帧，仅检查格式，未训练或作性能选择；不能称其从未被读取。
- 原始 HDF5 只读；逐帧复现包内官方 FFT，RD `160×256`；MD 为原生 RD_dB 的距离求和 `256` 维。为降低基线成本，RD_dB 再作明确的 `4×4` 块均值，另存 `40×64`；不谎称使用原生尺寸训练。保留全部 Doppler bins，不新增静态通道裁剪。保存公式、版本和数值等价核验。
- canonical: MD float32、RD40×64 float32、labels int16(-100 未标注)、instance_id int32(-1 未标注)、nominal timestamps float64；每条录制独立文件/哈希/源文件 stat 与原始数组流 SHA256。原始数据不改；缓存输出目录不得覆盖。
- 标称 15 FPS，无降时间采样。训练完整 `[start,stop)` 实例，不裁边、不截断、不重复最后帧；右侧 padding 有 mask，不参与编码器/池化/损失。每实例状态清零，只作离线片段分类，不声称在线即时输出或自主边界能力。
- normalizer 仅用 train 已标注帧；每个输入坐标 mean/std，scale>=1e-6；同一固定统计用于 validation；zscore 后裁至[-8,8]。不使用 test 或单个录制的未来值来估计在线输入统计。
- MD: 现有 causal MD-GRU，hidden128，256输入；RD: 40×64帧输入，3层stride2 CNN(16/32/64，GroupNorm/GELU)，空间均值后送 causal GRU128。两者 valid hidden mean → Linear128→12。dropout0.1，全部雷达与分类参数训练，从随机初始化开始，无旧模型加载。
- loss: train 实例数反比 class-weight CE，权重均值归一化；12类片段Macro-F1为主指标，另报accuracy、CE、12类recall、6类events/gestures组内F1、混淆矩阵和逐受试者结果。
- 两臂 seeds42/43/44，各30完整epochs；AdamW lr1e-3，weight_decay1e-2，globalclip1.0；无AMP，MD batch64 / RD batch16，validation batch16；训练输入顺序每seed/epoch一致，batch分组不同。预算按样本暴露匹配，优化步数和模型容量不同，分别披露。无earlystop，validation Macro-F1最大且最早的epoch为best，保存initial/best/last权重及每轮logits/metrics。
- CPU预处理4个worker，每进程数值库线程1；训练单GPU顺序执行，Torch threads4。不重装环境。
- 正式前必须完成：合成数据契约测试、现有相关编码器测试、官方FFT/12个已导出样本对照、完整转换读回哈希/标签/shape/finite核验，以及每臂12实例小样过拟合。过拟合 gate: 最多300步、lr0.005、dropout0；accuracy>=0.99且CE<=0.2。失败则保存原因并停止正式训练，不悄悄放宽gate。
- 每轮保存状态和日志；任务退出必须有完成或失败标记。正式完成后重载best并重放同批validation logits，独立重算指标；test不参与本轮。

## Review focus

实例结束帧排除、同类相邻实例不合并；未知区间不填动作；不同受试者角色禁止交叉；padding不得改变真实帧特征或预测；转换必须与官方数值一致，且不修改源文件。

## Tasks

- [x] 1. `harrad/data.py` + `tests/test_harrad_data.py`：CSV/按人划分、实例label/mask、train-only统计与完整片段读取。先失败测试，再实现，再运行目标测试。
- [x] 2. `harrad/preprocess.py` + `tests/test_harrad_preprocess.py`：官方等价 FFT、4×4 RD降采样、无测试数组读取的不可覆盖转换与读回核验。对照零输入有限值及官方样本。
- [x] 3. `harrad/model.py` / `training.py` / `runner.py` + `tests/test_harrad_training.py`：两臂变长完整片段、mask、CE/分组指标、300步gate与30轮3seed运行。固定配置 `configs/harrad-semantic-baseline-v1.json`。
- [x] 4. 运行相关测试和本地小型端到端数据；隔离source capsule上传服务器，源清单哈希核对、原生目标测试、官方FFT对照后启动持久pipeline。转换完成后自动核验、预检和训练；遇错停止并留下failed.json。
- [x] 5. 取回转换/预检/训练状态与报告；如正式仍在运行，明确报告进度，不冒称已完成。

无Git仓库，使用不可覆盖源码包和SHA256清单保留版本；无需创建Git worktree。

2026-10-17进度：实现与原生22项测试通过，v2后台pipeline已启动；完整转换、拟合gate及正式训练仍在执行链中，最终完成核验尚未发生。见 docs/harrad-semantic-baseline-progress-2026-10-17.md。

2026-10-17完成：11:01六组结束，255文件哈希及186预测独立复算通过，权重和全部日志已取回。见 docs/harrad-semantic-baseline-results-2026-10-17.md。

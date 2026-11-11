# HARrad 片段语义基线启动记录（2026-10-17）

**完成更新：** 六组训练已于11:01结束，结果和本地独立核验见[最终报告](harrad-semantic-baseline-results-2026-10-17.md)。下面保留启动时的历史状态。

**后台流水线已启动，当前处于数据转换，正式训练尚未开始。** 主进程 PID `8720`，4 个 CPU 转换 worker。流程依次执行完整缓存核验、两臂小样拟合检查、六组正式训练；失败会停止后续步骤并保存 `failed.json`。

## 已核实

- 独立 HARrad 数据、模型与运行模块已实现，旧 PARrad 实验和原始数据不变。
- 按人划分：train subject1–6，3,018实例 / 190,238帧；validation subject7，511实例 / 25,244帧；test subject8–9，323实例 / 18,033帧。
- 本地相关测试 **19 passed，3 skipped**；跳过 HDF5 和两项CUDA检查，未安装新依赖。
- 服务器 v2 **22 passed，0 skipped**，包含 HDF5 只读转换、CUDA 确定性正反传、padding、train-only统计、实例区间和保存权重重放。
- 独立审查指出一项P2：原重放复用了训练指标函数。已新增 `harrad/verification.py`，用逐类计数公式和 `logaddexp.reduce`，从保存的NPZ独立重算全局、组内及逐人指标，并与best epoch历史逐项对照；新增错误指标及跨组预测测试已通过。
- 官方FFT实际对照：12类×3帧，共36个训练帧；MD、降采样RD、既有MD小样的最大绝对误差均为 **0**。
- 启动后核实主进程存活、4个转换进程各已处理约5,000–6,000帧。该数字仅是启动期快照，最新状态以服务器文件为准。

此前额度限制导致的上传和审查已恢复。v1只做预检，没有启动pipeline；v2是实际运行版本。

## 固定实验设置

| 项目 | 设置 |
| --- | --- |
| 模型 | MD-GRU；RD-CNN-GRU |
| 输入 | MD256；原生RD160×256的4×4 dB块均值→RD40×64 |
| 编码器 / 头 | causal GRU128，整段有效hidden均值→Linear128→12；全部随机初始化训练 |
| 样本 | 完整 `[start,stop)` 实例，不裁边、不截断；每实例状态清零 |
| 预算 | 每臂 seeds42/43/44，各30完整epochs，无early stop |
| batch | MD64 / RD16，验证16；按样本暴露匹配，更新数及容量不同 |
| 优化 / 损失 | FP32，AdamW lr1e-3，weight_decay1e-2，clip1.0；训练实例数反比加权CE |
| best选择 | validation 12类Macro-F1最大、最早epoch |
| 预检gate | 每类一个train实例；最多300步，accuracy≥0.99且CE≤0.2 |
| 完成核验 | 重载best重放validation logits；保存预测独立指标复算 |

归一化只用train已标注帧；未知帧不填动作类别。此处是**离线完整片段分类**，尚未训练EventFormer、联合分支、LLM或自主Reset，不能称为在线即时识别。

本轮转换和训练不读取test信号、不计算测试指标。此前10月16日格式审计曾抽读subject8/9各3帧，仅检查格式；不能将这些受试者称为从未被读取。

## 后续恢复位置

- 源码：`/home/xibeidaxue/newdisk2/WJ/HARrad/experiments/semantic_baseline_20261017_v2`
- 输出：`/home/xibeidaxue/newdisk2/WJ/HARrad/outputs/semantic_baseline_20261017_v2`
- 源码目录：`launch.json`、`driver.log`、`native-tests.log`、`preflight.json`。
- 输出总状态：`status.json`；录制转换：`canonical/<recording>/progress.json` / `complete.json`。
- 训练：`runs/{md,rd}-seed{42,43,44}/status.json`、`history.json`、`best.pt`、`last.pt`、`validation-best.npz`、`independent-metrics.json`。
- 流程成功：根目录 `complete.json`；失败：`failed.json`。

v2源码包166文件、570,936字节，SHA256 `7b43ad8d3ca3c9ca8a1177add93d14e51ba07d31b25e311e5a8053c0ad570724`；服务器逐文件哈希核验通过。后续进度文档更新不修改运行中的源码。

[配置](../configs/harrad-semantic-baseline-v1.json) · [计划](superpowers/plans/2026-10-16-harrad-semantic-baseline.md) · [本地证据](../outputs/harrad-semantic-baseline-20261016-v1/)。本地目录沿用10月16日准备日期，实际启动为10月17日v2。

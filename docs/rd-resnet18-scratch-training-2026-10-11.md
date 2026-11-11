# RD-only ResNet18从头训练

2026-10-13已连接服务器确认20epoch全部完成，validation选第17轮。[最终结果及独立核验](rd-resnet18-scratch-results-2026-10-13.md)已交付，下面保留启动时的配置与状态记录。

目标是训练雷达感知与时序编码器，之后再接Projector/Qwen。与前轮冻结随机ResNet18特征的诊断不同，本轮随机初始化单通道ResNet18并更新全部卷积、BN、投影、BiGRU和分类头参数，不读取X-Fi或旧Stage1权重。

架构：RD`[B,T,1,93,128]` → ResNet18每帧512维 → Linear(512,256) → 双层双向GRU（每方向128维）→ Linear(256,14)。保留按时间有序的8×256平均池化token接口，但分类训练损失来自逐帧分类头。本轮不训练文本生成。

使用`offline_expanded_20261011_split_v1`，训练01/02/03/04/07/08/09/13/19，验证05/12/16，测试06/17/18。训练4,971窗、验证1,850窗；12秒/6秒步长、最短8秒、最多3动作。逐帧指标统计这些描述窗内的帧观测，重叠帧重复计数，不能直接当作全recording唯一帧指标。MD仍由兼容数据加载器读取，但从不输入模型。

固定seed17、20epoch，batch2、梯度累积2（通常有效batch4），AdamW，encoder学习率3e-4、分类头1e-3，weight decay0.01，cosine衰减、梯度裁剪1。类别权重由训练窗口内标签频率的倒数计算，不读取验证/测试标签来拟合。RD按有效窗口做signed-log及z-score，padding不进入CNN或BN。CNN以64个有效帧分块，启用CUDA FP16 autocast/GradScaler；投影、GRU、分类损失保持FP32。BN统计在训练时更新，验证时冻结。

最佳权重先比较三个验证受试者各自14类Macro-F1的平均，平局再比较合并帧Macro-F1和更小的未加权CE。仅train/validation参与训练/选优；训练前检查test清单元数据及文件状态，模型选择锁定后才实例化test数据并评价。最终结果分别保留06历史受试者、17/18新增模型留出、三人总体及逐受试者指标。

入口：`scripts/train_rd_resnet18_scratch.py --config configs/rd-resnet18-scratch-expanded-20261011-v1.json --output <全新输出目录>`。必须使用包含`source-manifest.json`的封存源码目录，输出目录不能已存在。`--smoke`在真实train上完成两次优化更新并读取两个真实validation窗口，保存/加载链路由单元测试验证，不读取test值。

关键产物：`initial.pt`初始随机模型、`gradient-proof.json`全部参数梯度和真实卷积更新证据、`first-update.pt`首更新权重、`history.json`完整epoch曲线、`best.pt`验证选优模型、`last.pt`epoch末模型和优化器/scaler/scheduler/RNG状态、`recovery.pt`每250优化步的覆盖式恢复快照、`selection.json`测试前锁定记录、`results.json`及validation/test logits。当前入口不提供自动续训；恢复快照的局部epoch状态不构成精确中途续训保证。

checkpoint格式为`rd_resnet18_scratch_v1`，包含独立RD编码器和分类头，不兼容旧双CNN训练器的encoder格式。后续语言对齐应显式接入这一编码器并重新训练语义接口，不能沿用旧Projector后直接声称生成改善。

启动前验证记录：本地新测试先RED再GREEN，覆盖全网络梯度及骨干/BN更新、padding无影响、末尾不完整梯度累积组、checkpoint与受试者元数据往返。初版服务器回归11passed/1skipped（原完整X-Fi pickle仅在本地），真实窗口smoke完成两次优化更新。审查发现并修正AMP溢出处理：溢出时交由GradScaler跳过更新并回退缩放，只统计成功更新；非AMP梯度异常仍立即失败。新增两个真实CPU GradScaler回归，验证溢出回退后可继续更新及非AMP异常不改权重。修正版复核与正式启动结果见下文。

## 正式训练已启动

修正版本地6passed，服务器13passed/1skipped；跳过项仅需本地完整X-Fi pickle，与本轮从头训练无关。只读复核确认AMP问题已修正，无剩余重要发现。修正版真实GPU试跑完成2次成功更新、0次溢出，validation读取和checkpoint写入正常，test值未访问。

正式训练于北京时间2026-10-11 21:04:46启动，PID13200，nohup后台运行；启动检查时已超过100次成功优化更新。全部11,898,062个参数可训练，80个参数张量均有有限梯度，首层卷积最大更新为0.000300236；首更新GPU峰值分配988,542,208 bytes，启动检查显存占用约1.5GiB。这里是启动证据，不是训练完成结果或最终性能。

源码目录：`/home/xibeidaxue/SEER/.audit-staging/rd-resnet18-scratch-code-20261011-v2`。实际执行148个文件逐一核对，封存包SHA256为`6f985a19307d4f13796fecd98583874f73492e559a932e312d12b79f6cec054b`。初版源码和两轮smoke产物均保留，正式模型从相同随机种子重新创建，不沿用smoke权重。

服务器训练输出：

```text
/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/rd_resnet18_scratch_expanded_20261011_v1
```

训练日志：`/home/xibeidaxue/newdisk2/WJ/PARrad/logs/rd_resnet18_scratch_expanded_20261011_v1.log`。运行状态看输出目录`status.json`，完整epoch曲线看`history.json`。最佳权重为`best.pt`，epoch末状态为`last.pt`，每250次成功更新生成`recovery.pt`；训练前期尚未完成完整epoch时，`history.json`/`best.pt`可能还未生成。

启动证据取回到`outputs/rd-resnet18-scratch-launch-20261011-v1/`。其中服务器导出的状态和日志是固定时刻快照，不能当作之后的实时进度。训练结束后`status.json`为complete，并写入最终`results.json`及validation/test逐帧logits；本次答复只确认训练已实际启动且持续更新，未提前报告最终指标。

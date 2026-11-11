# 方案六 SAEF-MVP：连续自然录制有监督事件形成

本入口依据用户选定并于2026-10-29再次确认的 `计划/想法六/最小可行架构.md`，位于 `continuous_mmwave.event_gating.mvp`。此前 `event_gating` 的拼接序列/CIF弱监督代理实验仍保留，本入口不调用那套训练器。

## 实现范围

- 读取经SHA验证的冻结CNN/GRU `continuous_hidden`，保留整段录制、未知间隔和原帧号，不读取oracle_hidden或拼接实例。
- 动作头H→12、证据头4H→128/LayerNorm/GELU→1，状态为2秒EMA，判断使用更新前状态。
- B0直接阈值上升沿、B1无漏电IF、B2局部证据LIF、B3局部加事件状态LIF。前三组共用同一份证据权重，四组共用同一冻结动作读出。局部/状态证据头容量和更新预算一致。
- IF采用dt×I，LIF采用exp(-dt/tau)×v+(1-exp(-dt/tau))×I；I最大3，硬复位，固定tau，不启用自适应阈值或回溯。
- 训练先做动作语义头，再用真实边界预热证据头；可显式配置 `closed_loop_steps`，用模型自主发火重建状态后继续监督训练。校准、validation始终无真实边界重置。
- 训练subject1–6，每个人按文件名最后一条录制留作calibration，其余用于fit。subject7只做预注册后的评估；subject8–9信号不读。每个训练受试者至少有两条录制，否则拒绝此划分。

## 命令

在仓库根目录运行，所有输出目录必须是新目录。

```powershell
$env:PYTHONPATH='D:/lab/SEER/src'
& 'C:/User/anaconda/envs/interview/python.exe' -m continuous_mmwave.event_gating.mvp smoke --output-dir outputs/saef-mvp-smoke-v1
```

真实HARrad使用已有服务器环境与锁定资产：

```bash
PYTHONPATH=src python -m continuous_mmwave.event_gating.mvp audit --config configs/harrad-saef-mvp-v1.json --output-dir outputs/saef-mvp-audit-v1
PYTHONPATH=src python -m continuous_mmwave.event_gating.mvp run --config configs/harrad-saef-mvp-v1.json --device cuda --output-dir outputs/saef-mvp-run-v1
```

配置引用既有资产锁文件，`audit`会检查SHA、来源、完整原生时间轴、标注和划分。时间为标称15 Hz，不伪造硬件时间戳；两帧一次的缓存不能伪装成逐原帧缓存。

## 产物与判读

`protocol.json`记录fit/calibration/validation文件、预算、seed及来源；`model.pt`保存共享动作头、两份证据头及优化器；`calibration.json`保存训练calibration上的完整网格与选择结果。每臂输出全帧trace.npz、事件、指标及时间轴SVG。

指标包含±0.2/0.5/1秒的一对一边界匹配、已知有效区域FP/min、未知发火单列、漏检率、有符号延迟及绝对误差、IoU .3/.5事件识别、过分割、受部分标注限制的序列/数量指标。未知区间不会自动变成idle或第13类。高斯监督宽度0.2秒，边界附近未知区域属于明确标记的弱过渡监督，其余未知帧不做负例；录制边缘的截断事件不作可靠边界负例。

`oracle-closed-loop-diagnostic.json`只在fit录制比较预热参照与自主状态，不能代替无边界输入的validation。所有真实数据收益需实际运行；smoke只有人工特征上的机制与接口验证，报告不得据此宣称G1/G2通过。LLM、骨架、分层记忆及复杂回溯属于后续阶段。

## 2026-10-29补全

- `audit.json`逐录制记录维度、原帧范围、更新间隔、时长、已知/未知帧数、监督范围和特征哈希，并保留上游权重与normalizer锁。标注使用已有HARrad格式审计核对的零基半开区间 `[start,stop)`。运行时拒绝录制时间间隔与模型配置不一致的输入。
- `summary.json`汇总全部边界容差和事件IoU，保留有效时长分母、按录制平均F1、逐类别事件支持数与P/R/F1、未知/混合区间负担和序列/事件数量指标；`report.md`提供表格。`calibration.json`保留训练校准集召回—误发火操作点，`tau-sensitivity.json`在校准集选定每个tau的阈值后评估，不按验证结果重选。
- `go-no-go.json`报告G1/G2数值候选检查：G1要求B1/B2召回差绝对值不超过2个百分点、有效区域误发火率相对下降至少20%；基线误发火为零或无GT支持时为不可判定。G2检查边界F1改善且额外过分割不增加。候选检查不等于研究结论：G3需要查看完整流诊断，G4需要语义非劣性复核，跨人/类别一致性和独立复现实验尚需完成。单次subject7结果不会自动宣称研究Go，合成结果始终标记 `synthetic_only`。
- 理想状态诊断严格按训练边界维护摘要，不受模型发火重置影响，与预热训练状态历史一致；自主路径只按自己的Fire维护摘要。
- 无语义标签的录制不参与动作头采样，无边界有效监督的录制不参与证据头采样；保留完整数据用于审计和评估，成功更新预算保持不变。
- 检查点复现绑定文件/受试者/角色/标注/时间轴身份及特征哈希；身份不一致或旧检查点缺少身份信息时拒绝复现，需重新运行本入口建立检查点。

复现已有检查点，不更新优化器：

```bash
PYTHONPATH=src python -m continuous_mmwave.event_gating.mvp evaluate --config configs/harrad-saef-mvp-v1.json --checkpoint outputs/saef-mvp-run-v1/model.pt --device cuda --output-dir outputs/saef-mvp-replay-v1
```

校准集必须包含至少一条有内部标注边界的录制。边界 F1 和召回仅在有真值边界的录制上取平均，FP/min 仅在有可计分时长的录制上取平均。未计分的未知区间发火单独记录。

截至 2026-11-04，本地 `interview` 环境的 SAEF-MVP 和旧 event_gating 回归共 66 项通过。CLI smoke 结果位于 `outputs/saef-mvp-smoke-20261104-v1`，包含检查点复现、四臂闭环推理和完整报告。

真实配置的本地audit在读取服务器路径 `.../HARrad/outputs/event_warmup_20261017_v1/feature-manifest.json` 时因文件不可用而停止，尚未训练或验证真实HARrad。需在已有服务器数据和锁定资产所在环境运行上述audit/run命令。保留既有配置、权重和历史输出。

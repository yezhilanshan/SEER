# TCN 雷达编码器检查结果（2026-10-09）

结论：历史 RD+MD TCN 编码器的动作识别性能确实偏弱，但四层 RD-TCN 没有被旧 Stage1 目标训练。这是已证实的监督路径遗漏，不能据此断言 TCN 架构本身无效，也不能直接把这些四层权重当作已训练的多尺度表示。

本次只读代码、历史指标和检查点，执行合成输入梯度检查；没有实际优化器更新，没有读取新测试样本或更改生产模型。服务器使用已授权直连。

## 1. 实际检查的是哪个编码器

- 历史 TCN：`src/continuous_mmwave/models/parrad_encoder.py` 中 `PARradRadarEncoder`，RD 四层 dilation=1/2/4/8，MD 三层 dilation=1/2/4。
- 近期 Stage3A 的上游配置 `configs/parrad-md-gru-stage3-sequence-v2.yaml` 指定 `causal_md_gru`，不是这个 TCN。近期 Proposal 的低 Recall 与窗口损失失败不能直接归因于旧 TCN。
- RD 四层本身的感受野分别为 3/7/15/31 帧。0.09 秒帧间隔下，最早到当前帧的历史跨度为 0.18/0.54/1.26/2.70 秒，不含 RD-TCN 输入已携带的其他历史。dilation 对应层的累积感受野，并非各层独立固定的“运动/语义”尺度。

## 2. 历史性能确实偏弱

完整 TCN run：`stage1_label_run_streaming_v2_full_lowgpu`，20 epoch label-run + 5 epoch streaming；best checkpoint 保存 step=6330、epoch=23（从零计数），last step=6645、epoch=24。

使用该 run 的 `diagnostics.json`，避免旧 `test_metrics.json` 把已裁掉边界的样本当作 raw 数据的已知问题：

| 指标 | Train | Validation | 历史 Test |
|---|---:|---:|---:|
| 去边界区帧准确率 | 43.21% | 35.82% | 35.42% |
| 完整流帧准确率 | 38.01% | 31.48% | 30.20% |
| 完整流 Macro-F1 | 38.79% | 30.46% | 30.00% |
| 真实动作段多数投票准确率 | 65.68% | 50.13% | 48.23% |
| 同一动作内相邻监督帧预测切换率 | 31.55% | 33.10% | 30.63% |

历史 Test 有 612 个真实动作切换、12,737 次预测切换，预测切换/真实切换=20.81。这里是动作分类 argmax 的切换统计，**不是 Stage3A Proposal 指标**。训练集准确率也只有 43.21%，所以问题不只是未见受试者泛化。

其他旧 TCN run 的历史 Test 帧准确率：3+1 pilot 18.55%；早期未裁边界 subject split 25.00%；trim1s subject split 29.66%。评价方式不完全一致，不把这些数字当成同条件实验趋势。

MD-GRU pilot 历史 Test：去边界准确率 40.00%，完整流准确率 36.57%，完整流 Macro-F1 36.04%，动作内切换率 10.34%。它与完整 TCN run 使用相同 subject 分割设计，但 batch=64 对 8、阶段预算15+10对20+5、GRU额外使用权重2的时间平滑项，且输入分别为 MD 与 RD+MD。因此只能作为历史参照，不能证明 GRU 在严格匹配条件下优于 TCN。

后续 MD-GRU token warmup 的帧准确率33.74%不是纯粹的编码器退化证据：该阶段冻结骨干、重建分类模块并训练 Token 路径，不能直接拿不同临时分类器的数字衡量骨干变化。

这些均为此前已经生成并查看过的 Test 结果，不是本次新开展的独立测试，不能据此调参后再次宣称独立验收。

## 3. 已证实的监督路径遗漏

代码路径：

```text
RD空间特征 + MD-TCN经FiLM调制
                 ↓
           frame_hidden ──→ Stage1动作分类损失
                 ↓
             四层RD-TCN
                 ↓
    temporal_projection × tanh(temporal_gate)
                 ↓
          Radar Token重采样器
                 ↓
           下游EventFormer
```

- `parrad_encoder.py:302` 构造 `frame_hidden`，`:306` 才执行 RD-TCN；返回的 `frame_hidden` 仍是 RD-TCN 之前的张量。
- `stages/parrad_stage1.py:120` 对 `output.frame_hidden` 分类。
- 四个旧 TCN run 的保存配置都没有启用 Token 分类权重；当前构建器对应默认值为0。
- `temporal_gate` 初始化0，四个导出检查点和 last 检查点中仍然是0，RD-TCN对输出Radar Token的注入系数因此为0。

不是只有“初始化门值可能太小”的推测。独立证据包括：

1. 四个 best 优化器里，RD-TCN、temporal_gate、temporal_projection、Token resampler 都没有 AdamW 状态；MD-TCN 12个参数都有状态。按保存的 trainable_names 与 optimizer参数顺序映射，数量匹配。
2. 同保存模型配置、seed=42重建编码器：四个导出检查点的 RD-TCN **16/16 张量**、temporal_projection **2/2**、Token resampler **5/5**、gate **1/1**，均与初始化逐元素精确一致，最大绝对差0。
3. MD-TCN **12/12 张量**均与初始化不同。不能把“RD-TCN未训练”扩大为“整个编码器未训练”。

合成输入梯度诊断没有执行optimizer.step：现有默认frame-only目标下，RD-TCN、门控和resampler没有非零梯度；打开Token目标后，初始门控与resampler有非零梯度，RD-TCN在精确零门值下首轮仍为零。这说明仅仅把RD分支加入计算图不够，必须接通监督并检查门控打开后RD层的实际梯度。MD-FiLM也是零权重恒等初始化，所以本次合成首轮MD-TCN梯度为0；历史MD参数已变化，不能用首轮现象推断MD长期未训练。

## 4. 对下一步的影响

不建议直接冻结这些历史RD-TCN权重开展 H1/H2/H4/H8 vs H8 的Probe。它只能测“随机时序层叠在已训练输入上的作用”，不能检验“已有训练的多尺度TCN信息是否被Proposal遗漏”。也不建议把零门控直接改成1后部署，随机RD时序分支会立即进入输出。

建议先做一个隔离的TCN预训练修复方案：

1. 明确监督作用于 RD-TCN 后的逐帧表示，同时监督最终Radar Token；保留原始pre-TCN表示另名，避免悄悄改变接口含义。
2. 确认RD-TCN、门控、投影、resampler均有实际非零任务梯度、优化器状态和权重变化。没有接通前不增加正式训练预算。
3. 使用现有fit/validation锁定初始化、预算和损失；把动作稳定区时间一致性作为单独明确变量，不能同时改多项后归因于TCN。
4. 先比较TCN前后表示的动作可解码性与时序稳定性，再在训练合格的骨干上做参数匹配的多尺度/最终层Proposal Probe。Proposal仍按fit校准、dev Recall@固定FP/min验收。
5. 现有MD-GRU/检索/QA路径保留；替换骨干需另做语义检索和QA保留核验。没有自动开始这项训练。

TCN监督遗漏解释了为什么不能评价四层RD-TCN的学习能力；它不是当前Stage3A全部失败原因的已证明解释，也不是修复后性能必然提升的保证。

## 5. 本地证据与验证

- `.audit-staging/tcn-encoder-inspection-2026-10-09/historical-evidence.json`：六个run的原有指标、诊断、配置、参数报告、优化器状态摘要及来源SHA。
- `.audit-staging/tcn-encoder-inspection-2026-10-09/initialization-comparison.json`：四个TCNrun与seed42初始化逐张量比较。
- `.audit-staging/tcn-encoder-inspection-2026-10-09/check_gradient.py`、`gradient-routing.json`：合成梯度路由诊断、零优化步。
- 服务器与本地历史证据JSON SHA均为 `e5fcac681d141552a3919a178196d2c6d1cb93bc0d1fb6b02e596b9e68cae53f`。
- 初始化比较JSON SHA均为 `96480beef1ba1e507b09ef7546de50e18a0bd66cde009e76aae1060916682122`。
- interview环境现有 `tests/test_parrad_encoder.py`：10 passed；`tests/test_phase1.py::test_tcn_chunked_equals_full_sequence`：1 passed。没有修改编码器训练代码。

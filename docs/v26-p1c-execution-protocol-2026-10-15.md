# P1-C 神经预测诊断执行协议

用户已确认2026-10-15建议：先做已有状态的小型幅值平台诊断，再实现历史匹配预测头，通过小规模检查后执行固定预算多Horizon比较与信息消融。此授权扩展上一轮“准备后停止”的范围；原准备协议和产物保持不变。本轮依然不是正式Reset训练，不读test、不改冻结前端、不追加预算。

## 执行顺序

1. 逐份读取已锁定的48份完整E，计算RMS、逐步变化RMS、相邻余弦、各Token RMS与每Token最大坐标的能量占比。RMS>3及连续32步高幅值只作描述定位，不是筛选模型/数据的门槛；保留所有原共同起点。选首次平台前后与尾部窗口，必要时只在这些片段从保存前态重放Gate，核对回放状态及冻结权重。
2. 完整实现GRU/TCN历史。Oracle从同一segment_start开始（最长334）；NoReset从录制起点开始（最长3665）。TCN kernel2、纯左padding、dilation=1/2/...，感受野分别512/4096；GRU无32步截断、无TBPTT。本轮仍用冻结完整F，因此两种对照都共享原编码器已有历史。
3. 通过零步残差、右padding、未来后缀扰动、逐前缀与高效整段推理等价、小规模单步有限性检查后，执行以下45份独立预测头训练。

## 三个固定比较总体

| 总体 | 状态策略与起点 | 神经臂 | Horizon |
|---|---|---|---|
| oracle_full | 原完整共同起点fit61299/val20075 | event_free / gru / tcn / event | 1/3/5 |
| oracle_ablation | 原固定供体共同子集fit58913/val19609 | event_free / gru / tcn / event / event_shuffle / event_zero | 1/3/5 |
| no_reset_full | 同完整共同起点，NoReset状态 | event_free / gru / tcn / event / event_zero | 1/3/5 |

共(4+6+5)×3=45个配置，每个300步，总13,500次正式有效更新；另有45个独立模型各1步smoke检查，单独计数，不续接正式训练。每个配置独立从种子42重新初始化，不从smoke或其他Horizon续训。原400k参数预算：MLP/GRU/Event沿用P1形状；长历史TCN只按参数数量选hidden/head_width，不用验证表现。A/B/C保留完全相同模型形状、初值、训练顺序与预算，差异只在输入E。event_zero为同头常量E；另列event_free以区分名义参数匹配和实际有效容量。

每总体使用预先冻结的同一300×8 recording/start表，三Horizon与所有臂共享；oracle_full与no_reset_full也共享实际起点，消融总体单独在其共同子集固定采样。每Horizon另报告Persistence和不调参线性趋势。Persistence NLL方差仅用该总体、该Horizon的fit残差等录制权重估计，clamp logvar[-6,2]；趋势不伪造NLL。

损失沿用P1-B：每坐标Gaussian NLL→每样本平均→批内同录制平均→录制平均；logvar[-6,2]，AdamW lr=.001/wd=.01/clip1，batch8。保存0/50/100/200/300完整模型/优化器，只评final300。训练样本覆盖率、梯度裁剪比例、时间/显存/参数差异同时报告。零均值残差初始化严格等于Persistence，方差初始logvar名义为0（沿用原sigmoid参数化，FP32舍入允许1e-6）。

验证用高效完整因果序列推理：每个状态段从起点重置GRU/TCN并输出各Update上下文，再取固定共同点；不引入后续输入。与独立逐前缀推理做数值等价检查（CPU1e-6；CUDA允许FP32不同批形状最大误差1e-4，不能以此放宽保存权重重放必须逐元素一致的标准）。同一整段算法、形状、权重的保存后回放需exact。

## 报告与因果限制

每录制→每人→宏平均MSE、Gaussian NLL/variance、相对Persistence与趋势Skill；分all/deadline_cross/mixed_current/mixed_future/boundary_support/clean_interior、age与幅值组。宏Skill先对每人求比例，不直接除宏MSE。空组N/A，实际样本数保留。

B继续原causal-age-v1供体表，同录制/同age/过去其他已完成段。不是严格保持经验边际的一一置换，不把过去同类活动当成无信息。NoReset B=N/A，不补未来供体。所有消融对照都在相同供体子集训练和评估，不能比较全量A与子集B。

NoReset平台定位仅解释状态幅值与变化，不将幅值大直接等价信息消失，不删除平台样本。必须比较A/B/C/D后才能讨论E是否携带额外预测信息；预训练容量、Oracle额外边界信息、单seed300步局限保留。任何结果均不自动启动正式Reset。

## 可复查产物

正式训练采用审查修复后的v2胶囊。先前v1仅完成诊断和45次独立单步预检，没有正式训练；v1证据保留。v2重新完成其45次单步预检，因此跨两个胶囊合计90次smoke，与正式45×300步分开计数。准备入口为一次性操作，诊断未成功不得进入准备，保存预测/基线/目标/分组/索引均须核验。

新独立源码胶囊、锁定输入SHA和采样表、45×5检查点、训练日志、全量验证逐点预测/分组索引、保存权重精确重放、完整文件清单。全量大产物保留native，本地下载完整逐点预测与必要checkpoint/日志/探针，明确下载范围，不冒充全部权重已下载。

# Semantic rehearsal 输入与目标核查

状态：两个真实数据只读诊断均完成。没有optimizer step，没有训练新权重；仅使用subject02折的fit/dev缓存和相应TRAIN原始窗口，未读取held/calibration数值数据或validation/test。

## 已查明的结果

### 缓存本身可以重现

- 原Stage2 checkpoint为epoch14；原模型和LLM配置与缓存构建配置完全一致。
- fit434个、dev214个缓存窗口全部对应原label-run枚举，窗口边界、动作标签、subject/session/sensor及text_index配对正确。
- 14动作×3模板均有覆盖；但模板在每个缓存窗口上固定，不复现原Stage2按epoch/batch位置轮换的行为。
- 42个文本hidden重新编码最大误差为0；21个fit和4个dev事件窗口重新推理通过cached hidden一致性检查。
- mask有效且连续右padding；重新打包混合长度窗口可正常推理。

这些检查排除了已检查窗口的缓存错位及全体缓存的标签/模板映射错误。事件重新推理是25个分层抽查，不能表述为所有648个窗口都重新推理过。

### 数值精度带来小幅batch差异

原adapter实际采用FP16。相同事件batch4与逐个推理的normalized key最大差异约7.55e-5–1.99e-4，text batch4/42约9.76e-5。将同一份checkpoint的计算转为FP32，对应差异降低为：

| 比较 | FP32 key最大误差 |
| --- | --- |
| stream batch4 / single | 1.04308e-7 |
| commit batch4 / single | 1.63913e-7 |
| text batch4 / batch42 | 2.27243e-7 |

该抽查支持主要由低精度计算的batch形状效应造成，而非mask内容错误；尚未全量测量这种微小差异对近邻排名的影响。新旧Projection留存评估使用同一份固定缓存，不能把适配后的大幅检索下降直接归因于这些误差。

### rehearsal并非原Stage2语义目标的等价复现

| 项目 | 原Stage2 | 当前适配rehearsal |
| --- | --- | --- |
| batch与负例池 | shuffled窗口，batch4 | 动作平衡batch32 |
| 窗口覆盖 | 完整label-run数据 | 每record/action最多4个窗口 |
| caption模板 | (epoch+batch位置) modulo3 | 每个缓存窗口固定一个模板 |
| Projection损失 | alignment + .1 consistency + .5辅助动作CE | alignment + .1 consistency；另加teacher anchor |

原caption LM不经过Projection，省略它不会直接删除Projection的梯度项。辅助动作CE经过Projection，省略它确实改变了优化目标，但下面的梯度证据不支持把它单独判为首要原因。

在允许读取的相同fit范围，缓存只保留434/898个窗口；缓存与完整fit窗口的类别分布total variation为0.174294。dev缓存214/456，TV为0.192163。这里的完整集合只包含允许的角色，不含held或calibration；这些是分布差异指标，不是模型性能下降幅度。

## 受控梯度诊断

全部分析使用冻结上游和旧Projection，未更新参数。

1. **同32个样本，不同负例池。** 八次配对比较同一32个样本作为一个batch32，与拆成八个batch4后平均损失。保持事件、文本、标签和样本权重不变；梯度方向余弦均值 **0.682587**，范围 **0.304163–0.948004**。平均loss分别 .931799 和 .424987；因目标不同，不能把loss大小比较解释为谁训练得更好。这直接证明contrastive负例池改变会显著改变本次观察到的梯度方向。
2. **仅改变模板轮换。** 相同事件/标签、同batch32，改为原epoch14的局部batch4模板轮换；与当前固定模板梯度方向余弦均值 **0.983961**。本诊断下影响小于负例池变化，但不等于模板因素无影响。
3. **补回冻结原辅助classifier的CE。** 八个动作平衡batch32中，current loss与补回原辅助项后的梯度余弦均值 **0.998493**。平均alignment梯度范数4.43829，auxiliary梯度范数.380318（权重后约.190159）。缺失项值得修正/明确，但不能声称补回它就能解决退化。

这些梯度测量定位了训练目标变化，没有证明其中某一项独自造成全部dev检索退化；也没有运行修正后的训练性能对照。

## 下一步建议

当前最应检验的是“保留旧语义”目标，而非继续对少量fit窗口重新优化一个不同的alignment目标。

- 优先把rehearsal语义保留改为对旧teacher的关系/检索分布约束；其old初始化应为零或近零梯度，避免零临时heads时第一步由新alignment目标独自推动。
- 若要继续复用Stage2 alignment，应单独做batch4、原模板轮换及冻结原辅助classifier项的有限对照，并明确这是重新适配，不是假定保留原空间。
- 上述为后续方案，尚未改变训练协议或启动新试验；仍保留epoch0和原语义门槛，不能根据本次梯度诊断直接扩大三折训练。

此前对“复用同一alignment公式”的判断不够充分：同一公式在不同负例池、样本分布与损失组合下，不是等价的训练目标。这是这次核查需要修正的项目理解。

## 原始证据

- `.audit-staging/audit-projection-rehearsal-v1.py` SHA437fe858b65e03e8b691ba1611edfb244e3b1773c3a493b1684db94152bc77a0。
- `.audit-staging/audit-projection-rehearsal-followup-v1.py` SHAc28b1972686deccba88bd4307400553838aefae0b976f6fc7558f06f08f619ed。
- 服务器两个输出目录位于Stage3源目录旁：`transition-aware-semantic-rehearsal-audit-v1`与`transition-aware-semantic-rehearsal-audit-followup-v1`。
- 原audit manifest SHA06ee45a8d09374e2b139f3d7685a5c69f4560d2aba3e5198509dbb7a43ac59d3。
- followup manifest SHA788f703a5830f66d2cdf9d89810929d1cd0910ecd68107d5811183ee5d2c647c。
- 本地JSON取回目录：`.audit-staging/results-rehearsal-audit-v1/`。

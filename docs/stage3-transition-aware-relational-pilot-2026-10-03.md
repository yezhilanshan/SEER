# Teacher关系保留小实验 v1

用户已授权执行先前提出的三个小实验。此协议在运行前固定；目标是检验Projection适配是否能同时改善TRAIN-inner-dev转换识别并通过原语义门槛，不承诺已解决退化。

- 仅subject_02折、训练seed1701。参数fit限另外两人的set000/001；dev限set002。held/calibration数值及validation/test封存。
- 从原Stage2 Projection初始化。冻结upstream和teacher，只允许query/MLP及三个临时130输入线性heads优化。128维接口不变。
- 使用已核查的fit434窗口及全部42文本模板，不重新构建/扩大数据。所有窗口等权。每次更新计算完整固定fit库关系；可以分块计算Projection但不改变损失权重。dev不进入bank。
- L_relation为stream/commit各自三个相似度矩阵(newevent/newtext、newevent/oldtext、oldevent/newtext)与old/old矩阵的MSE均值，全部相似度除以温度.07。
- L_consistency为每个fit窗口stream/commit余弦与teacher原余弦差的MSE，系数.1，不直接拉向1。
- L_anchor使用.5*单位key平方欧氏距离(等价1-cos但old初始化精确零梯度)，包含fit库stream/commit/text均值与当前/参考转换窗口均值，二者等权。系数1。
- 总损失L_transition + lambda_r*(L_relation + .1*L_consistency) + L_anchor。原label alignment、辅助CE均关闭。
- Candidate质量/年龄权重、临时heads、fit归一化、dev独立线性probe和语义评估完全复用原协议。
- 先仅训练heads1epoch，固定Projection。保存common-warmup的model/optimizer/RNG。三个条件共享完全相同的warmup模型和后续Candidate次序；分叉时统一重新初始化AdamW状态。
- 三条件：head-only(Projection冻结)、joint-relation1、joint-relation10；每条件再3epochs，总计4epochs。每16Candidates更新一次。head lr1e-3/decay1e-4；Projection lr1e-5/decay0；全局gradient clip1。CPU单线程，有限预算后自动结束，不自动展开三折/其它种子。
- 每epoch按原独立dev probe评估，head-only Projection分数应与epoch0相同(临时头指标另外记录)。必需未收敛指标记undefined，checkpoint不可选，数值非有限则停止。
- 语义每单项允许下降最多.02、正margin仍正、一致性loss最多恶化.02；不放宽solver和门槛。只有语义完整且合格、dev score严格高于epoch0才能保留；同分选较早epoch，三个条件同分优先较小lambda。初始有效模型仍epoch0。
- 保存每轮Projection、model/optimizer/RNG、history、probe诊断、源码/源数据/缓存/协议哈希；在评估前保存pending_evaluation，恢复时先评估，不重复更新。common-warmup中断直接保留证据，不覆盖输出。
- 训练前验证真实fit库新旧保留loss/梯度为零或数值近零，warmup后Projection/teacher逐bit不变，转换任务梯度非零。初始完整语义评估失败则不训练。

不能把434窗口的关系损失下降解释为对未缓存窗口或held人群的性能保证；soft关系损失不保证离散top1排序，最终仍以原检索门槛为准。

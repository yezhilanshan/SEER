# 新文字QA reader与雷达多事件融合：固定预算三组

用户批准接回雷达token、零步基线、none/order/time各500步及单事件/检索保留复测。沿用2026-10-05时序方案的数据、三组、初始化、每步三题平均CE、lr3e-4、seed42、100/250/500检查点与最终500步选择。

单事件使用原EventFormer reader与原累计1000步QA适配层，逐条复现297回答。多事件使用文字QA500步导出的独立rank8全24层Qwen reader，完整冻结；旧QA变换冻结，新473984参数融合层仅服务多事件。训练与生成共用同一路由，不共享可更新模型存储。验证reader导出SHA、原来源SHA、基础Qwen文件集合/逐文件SHA、base及完整reader摘要。

先重放fit1014文字题95%门槛、dev510文字题；共享零步雷达807题、原reader多事件510题及无记忆807题。新旧reader比较使用同一雷达提示，三组与新reader共同零步起点比较。文字列表只用于oracle诊断，雷达输入没有动作标签或参考答案，训练与推理提示一致。

每步Yes/No/内容题逐题反向累积三个CE平均后仅执行一次AdamW，双/四各250真实步。none不提供元数据，order仅提供相对顺序，time另提供相对开始及持续时间；当前位置/顺序成绩不能证明时间间隔推理。

每组后807真实生成与344交换证据诊断，单事件297逐条不变；三组和全部固定门槛均报告。原检索另做固定诊断：使用原EventFormer文字编码与原P，对缓存dev99个L0原token，以14条固定动作名称查询，重投影原token并报告逐类AP、macro AP、Recall@1/@5；不存在的类别为null。每组后重算向量、完整分数及排序，要求与训练前精确一致。这不是完整在线检索或独立受试者测试。

源归档、实际步数、冻结base、原始回答和完整日志均独立读回并归档；仅下载JSON/log。仍是上游见过的subject02 TRAIN内部开发录制，formal_go=false。

配置configs/parrad-stage4-temporal-qa-v3.yaml，服务器隔离代码stage4-temporal-qa-code-v3，新输出stage4_temporal_qa_v3，不覆盖v1/v2或文字reader。launcher拒绝已有输出/日志，SSH断线不重复启动。

恢复版本使用configs/parrad-stage4-temporal-qa-v4.yaml、stage4-temporal-qa-code-v4及stage4_temporal_qa_v4。v3第75批FP16梯度溢出已精确复现；v4只增加同批减半scale重试，失败尝试不推进采样/优化器，500步仍指实际成功更新。经推理AST、来源SHA、全题集和原检索数值一致检查，复用v3完整的文字/零步基线；正式训练均从原共同初始化开始，不复用失败/诊断学习权重。

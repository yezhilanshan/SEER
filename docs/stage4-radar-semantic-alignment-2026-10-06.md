# 单事件雷达接入新QA：固定预算答案/蒸馏对照

用户已批准本对话六点方案并要求开始。沿用subject02 TRAIN内部fit196/dev99事件，单事件8x896原token；原frontend/EventFormer/P/旧QA/压缩器与文字QA500步reader冻结，仅训练新的EventQAAdapter残差MLP（896→256→896、零末层初始化），不加载旧QA学习权重、不加时间编码、不压缩多事件。新reader在所有训练与生成分支直接调用，禁止按size1路由回旧reader。

训练ce/distill两组从seed42共同身份残差初始化，lr1e-4、AdamW weight_decay.01、clip1、initial scale256，各500次实际更新。各步按14类轮转（35或36次），相同问题Yes正例、相近类No反例、正例动作识别三题平均；负例由原build_single_tasks固定seed42构建，监督来源严格fit。每组最终500步统一报告，100/250/500检查点只核验，不按dev选择。

蒸馏组loss=平均答案CE+0.5*平均T²KL，T=2。冻结新reader在fit事件正确文字“Observed activity: <actual source label>”下做教师；雷达分支只接原雷达token及普通问题。用分别构造的答案target mask选择预测位置，检验两分支assistant标签完全相同，只对答案及EOS位置蒸馏，教师logits detach/no_grad。问题类别与来源事件类别分别保存query_label/label。ce组不调用教师计算损失。不同任务逐题反向后一次更新，控制11GB显存。

前置生成：fit588文字题macro动作与Yes/No各≥95%，dev297文字诊断全部报告、不用于模板或权重选择；fit教师门槛失败时保留完整证据、实际优化0并停止，不暗中补训文字reader。再评估新reader原token共同零步fit588/dev297、无记忆dev297、原旧单事件297复现及原P固定dev99/14动作检索快照。

每组训练后真实greedy生成fit588/dev297，另做198条dev配对交换证据，按实际交换事件与query_label重新确定Yes/No。报告动作总体/macro/14类召回、Yes/No及正反成对全对、混淆和无记忆/交换诊断。开发工程门槛macro≥85%、平衡≥90%、成对≥80%；失败照实报告，不自动进入多事件。原单事件继续保留旧路由；训练前独立生成297并与旧保存预测完全一致，主训练不持有其可更新权重，前后原上游digest/保护SHA与检索向量分数排序均必须完全一致。

非有限梯度重用已验证finite_optimizer_step：减半scale、恢复RNG、同一批重算、最多12次，不推进采样、不执行失败optimizer.step。前向非有限或失败超限停止。所有重试单独记录，不算成功预算。主正式权重从共同初始化开始，原生GPU预检的诊断更新另列、不复用。

实现分工：parrad_radar_alignment.py负责任务、对齐目标和完整预测核验；parrad_radar_alignment_train.py负责加载、原生训练/生成、冻结和检查点。独立verify-radar-alignment.py从cache重建全部任务/采样，核对源与代码SHA、500优化状态/全部检查点、预测全集/来源/评分、开发门槛、初始共同参数、原检索及297旧输出，再归档原始JSON/JSONL/log（失败/blocked也保留）。源码在新隔离目录stage4-radar-alignment-code-v1，输出stage4_radar_alignment_v1，拒绝覆盖或重复启动。SSH断线不重启worker。

本轮只证明既有TRAIN开发录制内接口可学与否，formal_go=false，不声称独立未见受试者或稳定推广。全部失败、两组结果均报告，不自动改原模型或继续多事件。

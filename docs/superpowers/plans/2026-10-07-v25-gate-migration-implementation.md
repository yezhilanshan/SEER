# v2.5 gate迁移实施计划

依据：`docs/superpowers/specs/2026-10-07-v25-gate-migration-review-draft.md`。用户2026-10-07明确要求按方案实施；直接执行主线，不重复询问既有授权。可选adapter/joint/P解冻/双路/seed43不自动执行。

## 固定准备与停止规则

先完成四族零步诊断：A原教师，B0/B1固定maturity，C-raw固定m=1折入bias并直接tanh，C-exact用正确的输出坐标变换，D复用原seed42零步证据。全部AdamW=0；新输出独占，原来源及留出封存不改。

在查看新dev结果前锁定C-raw热启动可用性：fit每个recording/sensor取一个最长完整GT run，记录前64步及完整run；所有输出有限；同一teacher输入的alpha负值比例不得超过50%；free-running完整状态/teacher状态范数比p99不得超过4。这里检查的是“兼容权重是否提供合理正更新起点”，不是最终检索资格。任一失败停止，不将C-exact或随机D自动改作训练初始化。阈值为本轮新工程停止规则，来自拟合正区间与防范严重外推的目的，不来自dev择优。

完整dev仍按原14查询与两路mAP/R@1、逐subject×sensor评分，不挑最优诊断分支；C-exact应与B1在数值容差内对应。FP32相同输入alpha/更新误差≤1e-5；原生递归计算还涉及FP16，完整轨迹误差原样报告，不混同controller FP32 replay门槛。

迁移候选固定C-raw/m=1、seed42、1000实际AdamW（250teacher-forced＋750free-running），gate lr1e-4、wd.01、clip1、有效batch16、TBPTT32；原P/主干/QA不动。损失预注册：alpha MSE 1、相对token MSE .1、两路key cosine 1、同查询正负事件teacher margin保持1、负alpha平方惩罚1。Teacher-forced只用前三项及负值惩罚；free-running同路径token/key/排序加负值惩罚；不得把旧状态作为学生接续初值。困难对从当前batch所有正负事件中选择teacher排序正确者，各查询独立均值、缺正负类不补零；不引入静态学生负例库。训练途中不据dev挑checkpoint，终态沿用原门槛。

若诊断停止，迁移配置保留为未执行建议，不创建optimizer，不为“开始实施”强行训练。通过则实现对应训练、独立读回，gate达标再Evidence500；其余完整主线3750预算沿用草案，所有后续frontend证书更新并重新验收。

## Task 1：零步诊断及权重变换

- 新建`v25/gate_diagnostic.py`及对应tests，先观察缺模块/缺功能失败。
- 测试固定m bias折入、极端logit稳定转换、raw tanh不等价、原teacher不改、无maturity输入、预注册fit停止规则及teacher-forced/free-running区分。
- 原生入口严格验证原checkpoint/Stage1/config、encoder/teacher索引、读取角色和record SHA；持续写状态及每record进度，无optimizer构造。
- 先fit后dev；即使热启动不合格仍完成各零步对照，最后明确stopped。

## Task 2：指标与覆盖契约

- 正式`fixed_query_retrieval`补top5命中率与最佳正/负margin，缺类保留None。
- `GateBatches`显式记录两路visible/target区间，保留前缀监督；测试长chunk接续、非恒定target、未读未来帧。
- 原有mAP/R@1算法和缺类门槛保持原值，相关v25测试整套运行。

## Task 3：服务器零步实验

- 单独源码包、清单SHA与输出路径；确认原worker已退、GPU可用、无同轮lock。
- 从严格锁定原模型装载；只读fit/dev，D来自原零步归档核对，不解封holdout。
- 独立读回模型digest与0更新，完整JSON/log/便携NPZ归档，服务器和本地SHA及成员路径一致后解包；本地重算指标。
- 根据预注册fit可用性推进或停止；不改架构守卫掩盖失败。

## Task 4：有条件的迁移训练

- 只有Task3 fit热启动规则通过才推进；否则生成失败归因报告，训练0步，Task4及后续标记未启动。
- 热启动标签白名单、cache/checkpoint provenance、当前gate输入以及真实AdamW计数先RED→GREEN；全源审查后启动1000步。
- 终态检索达标才Evidence500，再进入原C/D/E/F；任一门槛失败停止并保留证据。

Review focus：极端logit、maturity列折入、原生FP16舍入、长轨迹范数、缓存SHA与留出封存、重入不覆盖、GT只用于离线构造/评分、失败不重选初始化。

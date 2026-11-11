# Compatible gate Evidence500 implementation plan

用户已批准上一轮在对话中给出的修订，并要求开始。目标：从精确转换/m=1的检索兼容初始化训练新的MD统计代理Evidence头；本轮没有1000步gate迁移、负alpha训练或后续自动闭环训练。执行writing-plans/executing-plans/TDD，独立输出与源码SHA隔离（仓库无.git）。

## 锁定设计与训练

- 新架构`v25-age-free-positive-coordinate-v1`：旧gate第一层前2D列，maturity=1列折入bias；alpha=tanh(.5*softplus(z+log(2)))，不读取调用者age。保持FP32门控参数及既有原生dtype契约。本轮冻结全部frontend/P/gate。
- bundle保存显式frontend_gate_architecture，加载必须与实例精确匹配；旧bundle无此字段仅按旧v25-age-free-tanh处理。新兼容架构只允许development状态，不宣称signed gate或完整v2.5 ready；不放宽线上controller及Stage2联合训练守卫。
- 使用与前轮相同source Stage1/Stage2、配置与encoder/teacher索引SHA；只读dataset-v2 fit/dev，不读holdout/official。先确认新gate权重与C-exact零步诊断逐张量一致，并重建fit/dev事件特征。检查路径与run归属、record SHA、目标/可见帧范围、所有输出有限，固定GT run控制实验。
- Evidence四目标为归一化MD平均绝对幅度、标准差、相邻帧绝对差及cosine差的事件可见前缀均值，都是统计代理。两路均覆盖同一完整GT run，显式保存visible/target range和mask；不提供label/subject/sensor/age作为头输入。
- seed42，实际AdamW500步，lr1e-3，wd.01，clip1，FP32，无AMP。每步平衡抽8个fit事件并各送Stream/Commit各一次，实际batch16；5个fit受试者×2sensor轮转，每格400个事件draw/每路径400。类别按格可用类轮转；完整draw记录保存。tokens detach，只有Evidence参数进入optimizer。
- 保存0/100/500检查点，固定500终态，禁止dev择优。保存实际逐参数AdamW步数、全部冻结参数digest、缓存/代码/配置来源SHA，加载读回验证。复用train_fixed_head的500步路径；不调用Stage2Migration。
- 原检索门槛同前轮：dev/dev_regression mAP/R@1各路下降≤1pp，subject×sensor≤3pp。新特征检索key需与已归档C-exact逐位一致；训练前后frontend与features不变才可复用同一检索分数，明确是固定GT控制，不是新自动事件检索。
- 属性主门槛：dev两路合并masked MAE较fit每sensor各属性事件中位数基线下降≥10%；完整报告各路径、属性、sensor、subject与subject×sensor。无有效fit基线/无dev有效值/非有限输出不能填零或记通过。报告0步和500步，不挑中途结果。失败仅保存结果，不自动训练adapter或解冻。

## Task1：正式架构和bundle身份

覆盖修订（新dev训练结果读取前锁定）：本轮positive架构事件缓存屏蔽首帧两个delta统计，因为其全局前驱在事件可见范围之外。其余有效帧取均值，保存cross_boundary_delta_excluded；历史signed路线的目标定义不改。

- [ ] 先测试继承正输出/无age、权重不改、bundle精确身份匹配和只开发加载；观察RED后实现models/checkpoint。
- [ ] 保留诊断标签与线上严格守卫；本地相关测试GREEN。

## Task2：独立Evidence入口与监督/读回

- [ ] 新建compatible_evidence.py：配对平衡batch、独立属性评估、特征来源验证、冻结训练入口、原检索复算。
- [ ] 先测两路每格都覆盖/stop-grad、fit中位数缺值、dev缺值、前缀范围、封存分区、终态500步source契约，再实现。
- [ ] 锁定配置，运行全部v25测试及项目测试，记录任何环境失败；一次全源独立审查，Important/Critical一次修复。

## Task3：服务器训练与证据

- [ ] 独占源码包/输出，验证源/配置/diagnostic SHA、GPU、无同轮worker；原生相关测试。
- [ ] 重建特征并配对C-exact；原检索不合格或来源不一致则0步停止。训练500步，保存selected终态及bundle，只开发。
- [ ] 完整JSON/log/便携预测归档，服务器/本地SHA、成员路径验证后新目录解包；独立重算MAE与检索，生成结果报告。运行期间如连接中断保留worker，不重复启动。

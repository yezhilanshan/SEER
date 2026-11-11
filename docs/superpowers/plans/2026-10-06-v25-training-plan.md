# v2.5 第一轮真实雷达训练与整链验证计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, inline. 用户要求先批准方案再执行；本文件只是提案。没有授权新增子代理、开启训练、上传或切换生产模型。

**Goal:** 以固定预算训练新事件门控、Evidence Projection、物理 Proposal 和 Semantic Difference，验证 v2.5 的真实连续雷达、边界修正、状态重建、分层记忆及冻结 QA 链路。

**Architecture:** 从原 Stage2 权重复制出独立实例，先迁移 age-free tanh 门控并保护原检索语义，再训练分阶段头。Sequential Evidence、Boundary Localization、State Reconciliation 采用确定性控制与开发集校准，不训练新的 Verify。第一轮暂不解冻 EventFormer 主干、原 P、QA 和压缩器；这是受限迁移实验，不等同于方案书 Stage2 全量联合训练。

**Tech Stack:** PyTorch、现有 PARrad canonical NPZ/JSONL、Qwen2.5-0.5B-Instruct、896 维 Event Token、128 维检索 key、现有 conda 环境。

**Spec:** `计划/方案书v2.5 (2).md`；接口状态见 `docs/v2.5-online-modules-2026-10-06.md`；历史基线见 `docs/stage4-memory-pilot-results-2026-10-04.md` 和 `docs/stage4-radar-semantic-alignment-v2-results-2026-10-06.md`。

## Global Constraints

- 本轮仅写方案。用户明确批准后，才实施准备代码、上传、训练和回放；不提前执行。
- maturity/age 不进入可学习 Event State Gate 或 Semantic Difference；age 只用于离线配对、分层统计及控制器 warm-up/timeout。
- 不训练 Verify Logistic Head；使用 Sequential Evidence Accumulation，不宣称它等价于严格 SPRT。
- 在线输入不含动作标签、GT 边界、骨架或答案。标签只进入离线监督和评分。
- Confirm 后临时启用 Candidate；后台定位 refined boundary、重放并追赶，在同一 watermark 原子替换 Active。不得用修改时间戳冒充状态重建。
- 原 P 冻结不保证新事件 token 的检索能力；保留损失与真实检索验收都必须执行。
- 保存失败、缺失值、所有固定条件；不靠选中间检查点或临时延长训练过门槛。
- 同一 subject/session 的两种 sensor 视图必须同属一个数据分区；不以窗口随机打散录制。
- 本轮仅用原 TRAIN 内部录制。禁止读取官方 test 内容，也不宣称独立未见受试者泛化或正式 Go。
- 无 Git：独占新输出目录、完整代码/配置/数据 SHA；旧权重及旧结果保持对照。不安装环境或另换 LLM。
- 本地 pytest 使用 `C:/User/anaconda/envs/interview/python.exe`；原生训练使用现有服务器 seer 环境。认证不进入文件、命令参数或报告。

## Review Focus

1. 新 gate 与旧 gate 架构不兼容：新初始化、独立实例及版本化缓存，不能偷偷装旧状态。
2. 物理标签缺失或未同步：mask 排除未知属性；代理属性与骨架物理监督分开命名，不能伪造高度/速度。
3. Candidate 初始化捷径：必须测训练完成后的当前 head、真实年龄区间以及成熟同类/初生同类负例。
4. 冻结模块仍改变输入分布：检索和 QA 要实测，不以参数 SHA 未变代替功能保持。
5. 缓存 episode 回放不能替代闭环策略：最终选择必须做完整在线重放；失败后台任务、滞后和掉事件均需显式报告。

## 1. 本轮回答的问题与范围

假设是：新事件状态能保留旧语义，Semantic Difference 能区分“同一动作的新状态”和“动作真的改变”，顺序证据可以减少错误 Confirm，后台边界和重建可以减少记忆污染。

固定冻结：Stage1 雷达编码器、原 EventFormer 除 gate 以外的参数、LoRA、交叉注意力、原 Semantic Projection P、LLM、旧 QA adapter、共享 Query Compressor。QA 使用此前旧单事件参考模型，不把已退步的新 reader 自动升级为默认。

可训练：新 gate、Evidence Projection、Local Change、Radar Transition、Semantic Difference。没有时间编码 QA 重训、RL、检索投影适配或全主干解冻。第一轮通过后仍不能称“v2.5 全量训练完成”；全量 Stage2 联合训练和多事件 QA 属于后续单独提案。

## 2. 数据、标签与对照

### 固定分区

| 用途 | 录制 | 允许操作 |
|---|---|---|
| fit | TRAIN subject02/03/04，set000/set001，两个现有 sensor | 拟合网络、属性归一化、采样及训练 episode 收集 |
| dev | 同三个 TRAIN subjects 的 set002，两个 sensor | 训练后检索保留、初始化 probe、有限阈值校准及开发效果 |
| 本轮留出 | 同三个 TRAIN subjects 的 set003，两个 sensor | 锁定权重/阈值后一次性评分，不据此回调训练 |
| 官方 validation/test | 不进入本轮数值训练/调参 | 不读取其数组、预测或标签 |

以上是预期清单，批准后的 preflight 根据真实 manifest 核对；缺录制不自行换入其他分区。同一录制的两个 sensor 不跨区。set002 已多次开发使用；set003 即使本轮不参与拟合，上游也见过 TRAIN subjects。核对历史用途，若 set003 曾用于相应调参，结果标注为回归评估而非新盲测。所有录制都是 TRAIN 内部开发证据。

采样先等权 subject，再等权 sensor，再轮转动作/正反类别；同时报告自然分布结果。14 类标签沿用原映射，不能从训练难例中删掉“走进房间”等弱项。某类缺样、某年龄桶缺一类或某组指标无定义时标为缺失，阻止对应验收，不填零。

### Evidence 标签的两条明确路径

当前 `data_parrad.py` 仅暴露 rd/md/labels/timestamps，没有骨架接口。这是待核实的数据前提，不能据此断言服务器原始资料一定没有骨架。

1. **若有可用的同步骨架**：M=4，监督躯干高度、上肢关节运动强度、下肢关节运动强度、躯干运动强度。高度必须有可解释的坐标轴/地面参考；缺失属性 mask=false。批准后的准备阶段固定关节映射、单位、相机到雷达时序映射及公式；同步误差必须不超过一帧 0.09s。只用过去帧和当前帧计算运动属性。无法满足则不能启用该属性或声称完整骨架监督。
2. **若没有合格骨架**：本次批准范围内允许做 `MD-statistic-proxy` 受限实验，M=4，标签为每帧 `mean(abs(md))`、谱值标准差、与前帧的平均绝对差、与前帧的余弦距离。前帧缺失或谱范数退化时相应 mask=false；两个帧取均值形成当前 update 标签，事件状态监督使用其实际可见前缀平均。明确它们是雷达统计代理，不是人体高度、真实速度或上下肢活动真值，也不算方案书完整物理监督已完成。

四维属性只在 fit 中按 sensor 拟合 5%/95% 分位范围，再线性缩放并裁剪到 [0,1]；退化范围的属性不训练。dev/留出沿用同一范围。必须记录裁剪比例、可用覆盖率和各维 MAE。至少两维在每个 fit/dev sensor 有 80% 有效覆盖且非退化，否则停止 Evidence 训练并报告数据前提不满足。不得自动改成“纯语义训练也算完整通过”。两个数据路径由可用性确定，不能两条都跑后择优。

### 两类基线

- 原完整权重和旧在线控制器：相同录制、相同问句、相同原 P/QA/压缩器，重新保存功能基线。
- v2.5 新 gate/新头零步：记录初始化代价。最终既比较零步，也比较原模型；超过零步不等于超过原模型。

Stage2 起点为 `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage2_event_joint_formal_v1/best.pt`，历史已核验SHA为 `939aa977b2616d8ceb11f47ec1470701dc04bd7d1af64cc5ac20ebed3ab88f9a`。批准后重新核对，来源不符则停止；Stage1、原P、QA与压缩器身份也写入保护清单，不根据dev挑新的起点。

旧 subject02/set002 单 sensor 的边界 F1@0.36s=0.3056，仅是历史参考，不直接当作六录制总基线。旧单事件 GT 分段 QA 的 80.47%/84.34%/74.75% 也不能当作自动分段 QA 的分数。

## 3. 固定预算与训练顺序

主实验 seed=42。每个“步”均指实际 AdamW 参数更新，不指梯度累积次数。预算为每 seed 3,750 步；主 seed 全部通过后，允许 seed=43 从相同原权重独立重复同样流程，最多合计 7,500 步，失败run中的实际更新也计入上限。重复失败须并列报告，不挑较好的 seed 当成功。若工程修复导致必须从头重训，剩余总预算不足时停止并另提方案，不静默超限。

| 阶段 | 更新对象 | 固定步数 | 学习率 | 固定目标 |
|---|---|---:|---|---|
| A：Stage2 受限迁移 | 新 gate + Evidence Projection | 1,000 | gate 1e-4；Evidence 1e-3 | 保持语言/检索语义、使新状态带有可测属性 |
| B：Evidence 固定前端精修 | Evidence Projection | 500 | 1e-3 | 冻结最终 gate 后，以当前 token 提升属性预测 |
| C：Stage3A | Local Change + Radar Transition | 1,000 | 1e-3 | 边界高斯软标签，sigma=0.36s |
| D1：Semantic oracle warm-start | Semantic Difference | 250 | 1e-3 | age-matched Oracle 正反对，建立收集闭环难例的初始头 |
| D2：Stage3B | Semantic Difference | 1,000 | 1e-3 | 真实闭环 hard-pair BCE + 0.3 × 独立 Oracle BCE |
| E：Stage3C/D | 不更新网络 | 0 | — | 有限 episode 阈值及边界校准 |
| F：Stage3E/4 回放 | 不更新网络 | 0 | — | 原始雷达→状态重建→Memory→冻结 QA |

共同设置：AdamW weight_decay=0.01、global grad_clip=1.0、constant lr；heads FP32，原生 Qwen 使用已验证 FP16/AMP。gate 阶段 micro-batch=4、accumulation=4，有效 batch=16，每个 micro-batch 必须至少两种动作，避免对比损失无负例。显存若不足，只允许变为 micro-batch=2/accumulation=8，两样本动作不同，不改有效 batch、损失系数或成功步数；micro-batch 内对比项不同，不能宣称梯度完全等效，须在正式step前锁定并沿用至seed43。batch2仍无法运行则停止，不在本轮改为无负例的batch1或减少序列预算。

gate 序列最大 64 updates，TBPTT=32；始终按真实顺序推进状态，长 run 分成连续片段，不能每个片段伪装成新事件。Commit token 上限256沿用现有规则。缓存冻结 encoder 的四个 radar tokens，不缓存旧 gate 的 Event State 作为新训练输入。teacher keys 离线生成，避免同时常驻两套 Qwen。

物理头 batch=16 连续序列，每段64 updates，TBPTT=32；丢失帧和跨录制连接不产生伪边界。Semantic batch=64，各年龄桶同时有正负样本；先 subject/sensor 等权，再桶内正负等权。

### A 的损失及检索保护

`L = 0.1 L_text + 1.0 L_align + 0.1 L_stream_commit + 0.5 L_evidence + 1.0 L_retain`。

- text：原有动作模板 caption CE，不使用已暴露语义错误的新文字 QA 教师。
- align：原 P 对事件与固定模板的对齐损失，沿用 duplicate-label-safe 实现。
- stream/commit：当前 Stream 与 Commit 的语义一致性。
- evidence：有效属性的 masked SmoothL1，Stream 与 Commit 分别使用其实际覆盖时间的标签。
- retain：当前 Stream/Commit 分别向**同一输入范围、同一路径**的原 Stream/Commit key 做 cosine 保留。不能把同一旧 Commit key 同时冒充两个路径的 teacher。

这是原雷达检索表示的保护，不是此前失败的 QA 答案蒸馏。冻结原 P 不冻结 gate 输出，所以仍要实测检索保留。

100/500/1000步保存 gate 检查点，500步仅诊断；只以固定1000步判定，不据 dev 挑中间最优权重。阶段 B/C 保存100/500/终步；D1保存250、D2保存100/500/1000。任何未达本阶段固定终步的 checkpoint 明确标为不完整。

### D 的样本构造，避免循环依赖

- D1 只在 fit 的 GT 动作 run 上构造 oracle 对：成熟A↔初生A是重要负例，成熟A↔初生B是正例；同时包含成熟↔成熟及相近动作。GT 只确定离线标签和配对，不进入 head。
- D1结束后冻结这份初始化 head，按真实 Proposal/Fork/Candidate/Shadow/Accumulator 跑 fit，配置固定，收集完整 Candidate 生命周期，不凭 GT 触发/拒绝/确认。
- 收集配置固定：proposal=0.55、persistence=2，rho=.95、tau_diff=.6、tau_stability=.5、H+=3、H-=-3、lambda_e=1、tau_e=.2、warm-up=.54s、timeout=1.98s、pre=.54s。它是训练采集配置，不表示已校准合格。
- hard-pair 标签依据 Candidate 当时可见 GT 和生命周期结果；持续不同动作至少.54s才记正例；A→瞬态扰动→A为负例。mixed/ignored/覆盖不足记未知并排除，报告排除数。不能把整个生命周期标签复制给尚未发生变化的早期 update。
- Candidate 年龄分为1–3、4–6、7–9、10–11 updates四桶；统计原始 age_s，年龄不作为 head 输入。原先 age_updates=6/7是状态更新次数，不是人的年龄。
- D2的 closed-loop loss 和 oracle loss 独立求均值/加权，不把全部 pair 拼成一个“大数据集”。只生成一次 fit hard-cache；本轮不根据 dev 反复刷新难例。
- D2完成后，用**当前 head**在 dev 新收集 episode，重做 initialization probe；不沿用 warm-start probe 或旧前端缓存。

## 4. 校准不是新网络训练

先完成 initialization probe，再开始 E。probe不通过就停止：保留曲线、错例和训练终态，不用升高 Confirm 阈值掩盖初始化偏差。

开发 episode 上至多24个固定 accumulator组合：tau_diff∈{.55,.60}，H+∈{2,3,4}，H-∈{-2,-3}，lambda_e∈{0,1}；其余采用 D 的固定采集常量。所有组合完整报告。lambda_e=0是物理 Evidence 的功能对照；若最终只能选它，应报告物理证据未带来有效增益，不宣称完整 Evidence 路径通过。

对保存 observation 的回放只作有限预筛。每个候选决策必须按其模拟终止时刻重新定义离线 GT 标签，不将 timeout 时的标签沿用给早期 Confirm。预筛约束：episode FCR≤5%、positive episode recall≥70%；符合者按 recall最大、延迟最小、配置固定顺序选一个；没有符合者停止。

随后对选中的唯一控制器做**完整 dev 在线闭环重放**，重收集 episode、重新统计 FCR/召回及 initialization probe。早先 Confirm 会改变后续 Active/Proposal，不能声称缓存重放已评价这些变化。完整闭环不合格就停止，不继续试第二个配置。

若选中lambda_e=1，另做一次同一锁定配置仅lambda_e=0的完整闭环对照，比较FCR、召回、延迟和边界；不从这个对照重新选阈值。最终lambda_e=0，或lambda_e=1相对这个对照的GT边界召回下降超过2pp/FCR增加超过1pp，则完整Evidence路径资格不通过，保留负结果，不启动seed43。该对照没有额外网络优化。

Boundary 仅在这份完整闭环轨迹上比较四组固定权重：local/radar/semantic = (1,1,0)、(1,1,.5)、(1,1,1)、(1,.5,1)。以 F1@.36s 最大、MAE最小、固定顺序选一组；随后完整 dev 回放复验一次。边界改变也可能影响后续 Active；复验失败停止，不追加网格。最终锁定网络、控制器、边界、归一化和全部源 SHA 后才评分 set003。

## 5. 预注册门槛：建议值，批准后锁定

这些是本轮建议的工程门槛，不是方案书已有数值，也不是预告一定能达到。所有差值均为绝对百分点；bootstrap以 subject/session 录制为单位，同录制两个sensor绑定，不把重复帧当独立样本。仅三个subjects，区间不能支撑强泛化结论。

| 检查 | 进入下一步的门槛 | 失败处理 |
|---|---|---|
| A：语义/检索保留 | dev固定GT事件、固定14查询：macro mAP与R@1各不低于原模型1pp以上；各subject×sensor组下降≤3pp；Stream/Commit分别评分 | 停止，不继续训练转变头；新gate尚不合格 |
| B：Evidence有效 | dev masked MAE比fit常数中位数预测降低≥10%；同时报告每属性/每sensor | 停止或说明标签/特征无效，不伪装成完整物理证据 |
| D：初始化偏差 | early=.54–.72s AUROC≥.75；mature=1.08–1.98s AUROC≥.80；same early FPR@.5≤10%；same p_diff与age相关系数绝对值≤.30 | 停止校准；定位same初生状态错判，不自动加age或扩大训练 |
| probe覆盖 | early/mature各至少20个独立Candidate正例和20负例，报告每实际age均值与CI；缺类/指标未定义都不通过 | 数据不足，不能把空指标当通过 |
| E：完整闭环 | Candidate FCR≤5%、positive episode recall≥70%；另报全部GT边界召回与漏Proposal | 停止，不靠抑制所有Proposal实现低FCR |
| 边界质量 | 全录制macro F1@.36s比同数据旧控制器提升≥.10；precision≥.45；每subject F1下降≤.02 | 不升级在线基线；同时报.18/.54/1.0s容差和MAE |
| 状态重建/Commit完整性 | 同watermark replay状态与独立参考一致，FP32误差≤1e-5；无丢事件/未报告worker错误；flush后无待处理finalization | 工程失败，不能归结为模型准确率低 |
| 实时资格 | 按.18s节拍输入连续10分钟，主线程update p95≤.18s；无持续增长的后台积压；明确取消任务数与真正完成swap数 | 离线可跑，但不能宣称实时通过 |
| Memory/QA功能 | 同一问题集比较旧/新自动分段；平衡Yes/No及宏动作下降均≤2pp；单列GT分段参考、否定题、时间窗和多事件顺序 | 说明下游回归；QA错误不自动触发重训 |
| 留出/重复 | set003和seed43均完整报告上述适用项及逐组差值，失败不隐藏 | 不能称稳定收益；不追加seed44 |

检索门槛同时在最终自动事件上做任务级比较：GT事件与自动事件单位不同，须用GT时间窗匹配的检索命中/覆盖率和误事件率，不能直接比较两个不同事件库的mAP而声称保持。逐类弱项、R@5、分数margin、动作F1也报告，不用平均掩盖弱项。

Memory/QA的“通过”只表示本轮迁移没有明显损伤既有下游，不表示原先不合格的多事件推理已经解决。若动态QA提升但GT分段QA仍错，明确保留该问题。

## 6. 批准后必须补的训练入口与可测试任务

现有 `Stage2Migration` 和 `V25Stage3` 是接口，不是已经完成原生数据准备的训练程序。以下工作包含在本方案准备范围，不提前执行。

### Task 1：固定分区与Evidence监督

**Files:** Create `src/continuous_mmwave/v25/data.py`、`tests/test_v25_training_data.py`、`configs/parrad-v25-training-pilot-v1.yaml`。

**Interfaces:** `build_training_manifest(source_manifest, spec) -> dict`；`build_evidence_targets(recording, label_spec, normalization) -> targets, mask`；`fit_normalization(fit_records, label_spec) -> dict`。产物记录recording/sensor/时间范围、标签路径、SHA、缺样和排除原因。

- [ ] 写并运行分区泄漏、双sensor绑定、未来帧泄漏、缺失mask和fit-only归一化失败测试。
- [ ] 实现分区及选定Evidence数据路径；锁定字段定义、单位和固定配置。
- [ ] 在interview环境运行该测试文件，全部通过；原生检查标签同步与有效覆盖，导出清单。缺前提则不启动优化。

### Task 2：Stage2固定预算迁移

**Files:** Modify `src/continuous_mmwave/v25/stage2.py`；Create `src/continuous_mmwave/v25/train_stage2.py`、`tests/test_v25_gate_training.py`。

**Interfaces:** `train_gate(config, data_manifest, source_checkpoint, output) -> stage2_report`。batch显式提供 `teacher_stream_key`、`teacher_commit_key`、对应Stream/Commit属性target/mask，不复用当前单一teacher_key接口冒充分支监督。保留旧调用兼容或明确新schema迁移。

- [ ] 写原P/主干冻结、两分支teacher来源、梯度累积实际步数、AMP跳步计数、连续TBPTT状态和resume精确性测试。
- [ ] 实现1000实际更新及固定读回，真实梯度能到新gate和Evidence；训练日志区分计划迭代、成功更新和跳步。
- [ ] 原生最多4个只前向/反向的预检batch，不step；正式前20个成功step计入1000总预算，测显存/耗时后给ETA，不另开一次试训。
- [ ] 固定终步检索验收，失败保存证据并停止。通过后拷贝同一个Evidence状态至Stage3 heads，避免随机重建后冒充延续。

### Task 3：Evidence/物理头与真实Candidate数据

**Files:** Create `src/continuous_mmwave/v25/cache.py`、`src/continuous_mmwave/v25/collect.py`、`tests/test_v25_cache_collection.py`；Modify `src/continuous_mmwave/v25/stage3.py`、`src/continuous_mmwave/v25/__main__.py`。

**Interfaces:** `build_physical_cache(frontend, recordings, manifest) -> cache`；`build_oracle_cache(...) -> cache`；`collect_candidate_episodes(runtime, recordings, output) -> manifest`；`build_hard_pair_cache(episodes, offline_labels, spec) -> cache`。

- [ ] 测试旧缓存拒绝、frontend/head/config依赖变化、GT禁止进入在线调用、A→noise→A、mixed标签与小年龄配对。
- [ ] 执行B500/C1000，建立D1 oracle-only训练模式并执行250；与现有只支持closed+oracle的semantic_step明确区分。
- [ ] 从D1终态只收集一次fit真实闭环难例，执行D2固定1000；保存所有采样桶覆盖和独立AdamW读回。

### Task 4：当前head probe与有限校准

**Files:** Modify `src/continuous_mmwave/v25/training.py`、`src/continuous_mmwave/v25/checkpoint.py`；Create `src/continuous_mmwave/v25/calibrate.py`、`tests/test_v25_training_acceptance.py`。

**Interfaces:** `probe_semantic_head(..., age_seconds, early_range, mature_range, episode_ids, criteria)`；`calibrate_and_replay(bundle, dev_manifest, grid, output) -> report`。当前probe的 `early_max_age` 二分接口不足以表示本计划两个实际秒区间；实施时须显式更新证书schema、digest和验证，禁止仅改报告标题。

- [ ] 测试真实年龄区间、独立Candidate数、常量输出/缺类失败、训练后新head SHA及旧证书拒绝。
- [ ] 测试候选模拟终止时刻的标签重算、24组合全量保存、无可行组合停止、完整闭环复验失败停止。
- [ ] 执行开发probe、固定网格、闭环重放、四组boundary权重与唯一复验；锁定合格配置及完整provenance。

### Task 5：整链回放、留出、重复和交付

**Files:** Create `src/continuous_mmwave/v25/evaluate.py`、`src/continuous_mmwave/v25/runner.py`、`tests/test_v25_native_replay_contract.py`；Modify入口只加入显式新命令，不替换旧controller。

**Interfaces:** `run_training_plan(config, output) -> status_manifest`；`evaluate_continuous(bundle, recordings, memory, frozen_qa, questions, output) -> report`。只接受有准确SHA、完整终步及当前校准证书的bundle；单一worker锁。

- [ ] 测试同watermark读回、慢后台无丢事件、失败flush、EOF未确认尾段不计确认边界、时间窗评分和源问题集固定。
- [ ] 先跑开发整链与节拍回放，再按最终SHA跑set003；逐录制原始预测与缺失值完整保存。
- [ ] 主seed全部门槛通过后执行seed43独立全流程；用同一预注册网格规则及终步，不能改变标准。
- [ ] 服务器独立读回实际optimizer step、模块冻结SHA、checkpoint状态和预测；完整JSON/log归档，核验服务器/本地SHA和成员路径，再在本地重算指标。大权重保留服务器。
- [ ] 输出 `docs/v2.5-training-pilot-results-2026-10-06.md` 及进度ledger；将“完成、有效、保留检索、实时、泛化”分别结论。不替换生产模型，不自动继续新训练。

## 7. 输出、安全恢复与时间预估

建议服务器隔离代码 `/home/xibeidaxue/SEER/.audit-staging/v25-training-pilot-code-v1`；输出 `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/v25_training_pilot_v1/seed42`、`seed43`。目录已存在则停止或使用显式新版本，不能覆盖。

状态记录phase、实际step、PID、启动时间、输入SHA和最后成功checkpoint；SSH断线只恢复监控，不重复启动。网络恢复后先核对PID/lock/checkpoint，再继续读取。运行中不临时调整损失；非finite/来源SHA变化/后台错误停止并留证。

预计使用既有服务器单GPU；此前记录为2080Ti 11GB，但本轮没有重新查询其在线状态。不能把上次两组500步QA约16分钟直接当作本次多步展开gate训练用时。批准后在前20个正式更新给出基于实测的ETA：各阶段剩余成功步×step耗时＋缓存构建＋固定回放/归档；阶段分开列预计完成时刻。计划不是几十步探针，需按小时级实验安排资源，当前不承诺具体结束时间。

若新 gate 检索保留失败、Evidence无有效标签/收益或initialization probe失败，本轮在该点结束并给出一个明确失败归因及下一份有边界的建议；不反复跑相同审核，不自行延预算或解冻主干。真实工程bug可在授权范围内修复并重做相关检查，影响轨迹则保留失败run，用新目录说明重启原因，不能伪装成同一成功轨迹。

## 8. 审阅结论与批准范围

本计划已根据当前代码接口和用户v2.5章节做静态自审；所有准备任务未执行。上一轮本地相关测试86 passed/1 skipped仅说明模块接口，不替代本计划原生训练检查；上一轮独立审查曾因额度中断，不能称完整独立审查已通过。

用户批准表示授权：上述准备代码与原生预检、一个seed42固定预算主实验、达标后一个seed43重复、有限校准、整链及TRAIN内部留出评估，以及结果归档。若骨架不可用，按明确标注的MD统计代理路径做受限实验；不授权静默改为其他人体物理标签、不授权官方test、QA重训、主干/原P解冻、部署替换或追加第三轮。

**当前状态：等待用户批准。实际训练启动次数：0。**

# v2.5 第一轮真实雷达训练与整链验证计划（数据修订v2）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, inline. 用户要求先批准方案再执行；本文件只是提案。没有授权新增子代理、开启训练、上传或切换生产模型。

**Goal:** 以固定预算训练新事件门控、Evidence Projection、物理 Proposal 和 Semantic Difference，验证 v2.5 的真实连续雷达、边界修正、状态重建、分层记忆及冻结 QA 链路。

**Architecture:** 从原 Stage2 权重复制出独立实例，先迁移 age-free tanh 门控并保护原检索语义，再训练分阶段头。Sequential Evidence、Boundary Localization、State Reconciliation 采用确定性控制与开发集校准，不训练新的 Verify。第一轮暂不解冻 EventFormer 主干、原 P、QA 和压缩器；这是受限迁移实验，不等同于方案书 Stage2 全量联合训练。

**Tech Stack:** PyTorch、现有 PARrad canonical NPZ/JSONL、Qwen2.5-0.5B-Instruct、896 维 Event Token、128 维检索 key、现有 conda 环境。

**Spec:** `计划/方案书v2.5 (2).md`；接口状态见 `docs/v2.5-online-modules-2026-10-06.md`；历史基线见 `docs/stage4-memory-pilot-results-2026-10-04.md` 和 `docs/stage4-radar-semantic-alignment-v2-results-2026-10-06.md`。

**Revision:** 仅修订数据来源、分区、转换/去重及评估解封顺序。依据 `docs/server-dataset-inventory-2026-10-06.md`；原方案 `2026-10-06-v25-training-plan.md` 保留。模型、冻结范围、损失及其系数、学习率、每阶段步数、两个seed预算和数值验收门槛均沿用原方案。

## Global Constraints

- 本轮仅写方案。用户明确批准后，才实施准备代码、上传、训练和回放；不提前执行。
- maturity/age 不进入可学习 Event State Gate 或 Semantic Difference；age 只用于离线配对、分层统计及控制器 warm-up/timeout。
- 不训练 Verify Logistic Head；使用 Sequential Evidence Accumulation，不宣称它等价于严格 SPRT。
- 在线输入不含动作标签、GT 边界、骨架或答案。标签只进入离线监督和评分。
- Confirm 后临时启用 Candidate；后台定位 refined boundary、重放并追赶，在同一 watermark 原子替换 Active。不得用修改时间戳冒充状态重建。
- 原 P 冻结不保证新事件 token 的检索能力；保留损失与真实检索验收都必须执行。
- 保存失败、缺失值、所有固定条件；不靠选中间检查点或临时延长训练过门槛。
- 同一 subject/session 的两种 sensor 视图必须同属一个数据分区；不以窗口随机打散录制。
- 本轮使用原TRAIN录制与新增PARrad subject08/13作为fit/dev来源，subject12全受试者留出。原官方validation subject05及test subject06不进入本轮数组读取、训练或调参。只有训练来源审计确认subject12未被上游及下游训练/调参使用后，才可称其为本轮未见受试者；一人留出不支持总体泛化或正式Go。
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

## 2. 数据、标签与对照（修订内容）

### 固定分区：新增数据既用于训练，也保留整人评估

数据依据是2026-10-06服务器盘点，不将“文件头可读”当作全文件已校验。本表是批准后的固定候选清单；准备阶段完成下述来源和格式检查后才成为可用清单。两种sensor分别为TI_xWR14xx/TI_xWR68xx，一份H5对应两个视图。

| 用途 | subject与session | H5录制数／sensor视图数 | 允许操作 |
|---|---|---:|---|
| fit | 02/03/04各set000/001；新增08的set000/001；新增13的set000 | 9／18 | 拟合网络、属性归一化、训练pair/episode及teacher keys |
| dev回归子集 | 02/03/04各set002 | 3／6 | 保留原开发参照，检查原检索/QA功能，不被新增数据的平均收益掩盖 |
| dev新增子集 | 08的set002 | 1／2 | 新录制上的probe、Evidence和闭环控制表现；与回归子集合并为扩展dev |
| 录制留出 | 02/03/04/08各set003 | 4／8 | 两个seed安排与终态均锁定后一次性回放；是见过受试者的录制留出 |
| 受试者留出 | 12的set000/001/002/003，全subject保留 | 4／8 | 模型、阈值和评估规则完全锁定后一次性评分；禁止fit/dev使用 |
| 不使用 | 原官方validation的05全部录制；原官方test的06全部录制 | — | 继续保持封存，本轮不读取数组/标签/预测 |
| 不可用下载 | 13的set001 part、13的set002、15/20空目录 | — | 排除，不补零、不当作已有样本、不启动续传 |

因此fit从原来的3个subjects/12视图增加到5个subjects/18视图，视图数增加50%，优化步数不增加。全实验候选为21份H5/42个sensor视图，其中9份新增H5需转换成18个新视图；这些是录制/视图数量，不是独立受试者数量或训练题目数量。

subject08已有部分session参与fit，因此其dev/录制留出不能称“新模型没见过的受试者”。subject13只有set000正式文件，本轮全部用于fit，不从同一录制随机切一块冒充独立验证。subject12整人不参与学习和校准；其是否真正未见必须绑定Stage1、Stage2、旧Stage3、QA和压缩器各自的历史训练/适配/调参来源，不能只看当前canonical manifest。若审计发现历史使用，仍保持留出角色并如实改称“历史已见受试者留出”，不临时找别的subject代替、不把它转入fit。

旧02/03/04的set002、set003都已被上游见过；set002还多次开发使用，set003保留回归性质。最终受试者留出仅有subject12一人，须报告四个session/两个sensor的差异，不能推广为全人群泛化已通过。

### 来源准备、转换与严格隔离

- 原数据只读复用 `/home/xibeidaxue/newdisk2/WJ/PARrad/canonical/phase1`；保留原train/validation/test、label_map和NPZ，不把新subjects追加进原清单。
- 新H5来源根为 `/home/xibeidaxue/newdisk2/WJ/PARrad/raw`，使用上表明确列出的08四份、12四份、13/set000一份及各自timestamp_speech.csv，不扫描part猜测可用样本。
- 新的隔离canonical目录计划为 `/home/xibeidaxue/newdisk2/WJ/PARrad/canonical/v25_dataset_v2`；只读复用旧NPZ路径并保存新增NPZ。生成独立 `fit.jsonl`、`dev_regression.jsonl`、`dev_new.jsonl`、合并 `dev.jsonl`、`session_holdout.jsonl`、`subject_holdout.jsonl`、`dataset-provenance.json`。
- 对新增9份H5/CSV做一次来源完整性核对：记录实际SHA/字节数，与已有可信预期SHA比较；没有可信预期SHA时明确记录“来源未能独立验真”，不把自算SHA冒称官方校验。转换时验证可用MD字段、有限值、传感器长度与CSV时间单调性。准备失败停止，不按模型分数排除某个subject。
- 标签固定复用原14类label_map、原命令归一化和原 `command_active_until_next_command` 策略；不得对新增subject重新按字母排序建ID。未知command不得硬映射到相近动作；作为数据兼容失败保留完整原因，不只评分较容易的已知类后宣称完整通过。
- 帧周期仍0.09s、两帧一次update，继承原传感器对齐及录制起点规则；起点/环境信息必须有来源记录，不能据模型效果反推时间偏移或把未知环境伪称Hospital。
- 当前采用causal_md_gru，只按原接口转换/缓存MD，不为了“利用cube/RD”切换encoder、增加输入或读取全部cube作新训练任务。
- 训练/采样/标准化/teacher键/难例缓存仅接受fit；dev只评估和校准。当前前端及数据清单SHA必须进入缓存依赖，不能把旧数据缓存视为新增样本。
- subject12锁定时可做文件路径、字节数和H5字段等来源检查；其数组/动作标签转换、teacher/新模型预测和评分延后到全部终态锁定后。解封前不向训练流程提供其动作频率、属性分位数、初步效果或缺类分数。

同一subject/session两sensor绑定分区，不能用一个sensor训练、另一个sensor声称独立测试。重复事件的不同模板/前缀/视图不得拆分成跨区样本。新增文件失败不自动换入05/06或手势集。

### 数据采样与对照集合

沿用原“subject等权→sensor等权→动作/正反类别轮转”的采样方法，仅将fit subjects扩为02/03/04/08/13，每个subject目标质量1/5。subject13可用录制更少，会有更多重复采样；报告独立run/pair/episode数、复用次数和每类覆盖，不能把重复抽样计为新增样本。每个micro-batch至少两种不同动作的要求不变。

原14类保留，不删除“走进房间”等弱项；缺类和年龄桶缺一类照原规则标缺失并阻止相应验收。新增subject未覆盖某类时不制造样本；统计其真实覆盖，并在其可用类别中轮转。同步报告自然录制分布结果。

旧dev回归子集与完整扩展dev分别报告检索/QA，再按subject/sensor报告。原检索保留数值门槛在旧回归子集和扩展dev均检查，避免新增08的改善抵消旧02/03/04退步。控制器校准只用合并dev，网格和全部数值门槛保持原方案；另列两个子集结果，不据其结果换分区。

### 其他已下载数据的用途

1. **早期PARrad MD子集**：`/home/xibeidaxue/SEER/data/parrad/hospital_md_77ghz_normal_v1`。其04/05×set000/001是已有录制的提取副本，不增加样本数。只使用属于fit的04两段与对应原MD做轻量读取/转换一致性检查；05副本仍封存，不能绕过官方validation封存。若比较发现提取时间范围不同，按来源时间范围定位，不强行认为逐帧相等。
2. **pyRadar/pyRadar-main的4/5/10类手势资料及bin/MAT版本**：建立去重清单，记录标签、采样说明、字段、原始/处理版关联，为后续独立手势域实验保留。当前MAT allData为32×30/60等样本表示，尚无已验证的PARrad128-bin连续MD/14类/时间戳映射。本轮不进入主fit、dev、Evidence归一化或留出评分，也不任意插值/拼接成“连续行为”。这样使用资料管理价值，不引入用户未要求的新训练任务。
3. THUCNews/MOSI/MNIST/CIFAR/SIRST/UCF等目录不纳入当前雷达事件训练；part及空目录记录为不可用，不能计入规模。

### Evidence标签的数据依据

盘点已确认28份正式PARrad H5均只有radar的cube/mDoppler/rdn，没有同步骨架字段；当前搜索范围也未发现可识别的同步骨架数据。本轮固定使用原方案的 `MD-statistic-proxy` 路径，不从新增手势标签伪造高度、肢体强度或真实速度。若后续找到同步骨架，另行确认来源后再提出替换数据监督方案，不在本次运行中途切换。

M=4与原方案相同：每帧mean(abs(md))、谱值标准差、与前帧平均绝对差、与前帧余弦距离；前帧缺失/谱范数退化时mask=false，两帧取均值为当前update标签，事件状态使用实际可见前缀平均。它们是统计属性代理，不算方案书完整人体物理监督。

只用扩展fit的5个subjects按sensor拟合5%/95%分位范围，冻结后用于旧/新增dev及两类留出；不得用subject12或08/set002/003的统计反算归一化。退化属性不训练，mask处理与原门槛不变：每个fit/dev sensor至少两维80%有效覆盖，dev MAE较fit中位数常量预测降低至少10%。报告逐属性、逐sensor和逐subject指标及裁剪比例，不以雷达统计差异冒称人体姿态理解。

### 两类基线与统一评分

- 原完整权重/旧控制器与v2.5零步在同一扩展dev分别建立基线，固定同一P、QA、压缩器、14查询和问句生成规则。新08/13上的teacher键来自原冻结模型读取相同fit输入，不更新teacher，不将subject12键进入训练。
- 对两个锁定终态及同一原模型，在录制留出和subject12使用相同输入、问题、GT匹配与边界容差；保留所有失败/缺失。原模型在新subject的效果只能解封后记录，不拿来挑新训练目标。
- 原Stage2起点仍为 `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage2_event_joint_formal_v1/best.pt`，历史已核验SHA仍为 `939aa977b2616d8ceb11f47ec1470701dc04bd7d1af64cc5ac20ebed3ab88f9a`。批准后复核来源；旧P、QA和压缩器身份不变。
- 历史subject02/set002单sensor边界F1=.3056与GT分段QA80.47/84.34/74.75%仍仅作历史参考，不能替代扩展dev或subject12的同数据旧模型基线。

### 解封顺序与结果称谓

seed42先按原固定预算和开发门槛运行。只有全部开发资格通过才按原规则启动seed43；是否运行seed43必须在任何留出结果解封前决定，不以subject12成绩追加/取消重复。若主seed开发失败，本轮按原停止规则结束，两类留出继续封存。

计划内seed均到达固定终态并锁定权重、控制器、boundary、归一化和问题规则后，一次性解封02/03/04/08的set003及整个subject12，原模型与所有已锁定seed一并评分。不根据结果换训练集、追加seed、重挑阈值或隐藏不合格模型。录制留出、原受试者回归、subject12留出三类结论分开；未见身份得到完整来源证明时才使用该称谓，也不能据一人留出宣布普遍泛化或正式Go。


## 3. 固定预算与训练顺序

主实验 seed=42。每个“步”均指实际 AdamW 参数更新，不指梯度累积次数。预算为每 seed 3,750 步；主 seed 全部开发门槛通过后，允许 seed=43 从相同原权重独立重复同样流程，最多合计 7,500 步，失败run中的实际更新也计入上限。重复失败须并列报告，不挑较好的 seed 当成功。若工程修复导致必须从头重训，剩余总预算不足时停止并另提方案，不静默超限。

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

Boundary 仅在这份完整闭环轨迹上比较四组固定权重：local/radar/semantic = (1,1,0)、(1,1,.5)、(1,1,1)、(1,.5,1)。以 F1@.36s 最大、MAE最小、固定顺序选一组；随后完整 dev 回放复验一次。边界改变也可能影响后续 Active；复验失败停止，不追加网格。最终锁定计划内全部seed的网络、控制器、边界、归一化、问题规则和全部源 SHA 后，按数据章节的一次性解封规则评分录制留出set003及受试者留出subject12。

## 5. 预注册门槛：建议值，批准后锁定

这些是本轮建议的工程门槛，不是方案书已有数值，也不是预告一定能达到。所有差值均为绝对百分点；bootstrap以 subject/session 录制为单位，同录制两个sensor绑定，不把重复帧当独立样本。扩展fit有5个subjects、dev有4个；受试者留出仅subject12一人，须分开报告，区间不能支撑全人群强泛化结论。

| 检查 | 进入下一步的门槛 | 失败处理 |
|---|---|---|
| A：语义/检索保留 | 旧回归dev和扩展dev分别按固定GT事件、固定14查询：macro mAP与R@1各不低于原模型1pp以上；各subject×sensor组下降≤3pp；Stream/Commit分别评分 | 停止，不继续训练转变头；新gate尚不合格 |
| B：Evidence有效 | dev masked MAE比fit常数中位数预测降低≥10%；同时报告每属性/每sensor | 停止或说明标签/特征无效，不伪装成完整物理证据 |
| D：初始化偏差 | early=.54–.72s AUROC≥.75；mature=1.08–1.98s AUROC≥.80；same early FPR@.5≤10%；same p_diff与age相关系数绝对值≤.30 | 停止校准；定位same初生状态错判，不自动加age或扩大训练 |
| probe覆盖 | early/mature各至少20个独立Candidate正例和20负例，报告每实际age均值与CI；缺类/指标未定义都不通过 | 数据不足，不能把空指标当通过 |
| E：完整闭环 | Candidate FCR≤5%、positive episode recall≥70%；另报全部GT边界召回与漏Proposal | 停止，不靠抑制所有Proposal实现低FCR |
| 边界质量 | 全录制macro F1@.36s比同数据旧控制器提升≥.10；precision≥.45；每subject F1下降≤.02 | 不升级在线基线；同时报.18/.54/1.0s容差和MAE |
| 状态重建/Commit完整性 | 同watermark replay状态与独立参考一致，FP32误差≤1e-5；无丢事件/未报告worker错误；flush后无待处理finalization | 工程失败，不能归结为模型准确率低 |
| 实时资格 | 按.18s节拍输入连续10分钟，主线程update p95≤.18s；无持续增长的后台积压；明确取消任务数与真正完成swap数 | 离线可跑，但不能宣称实时通过 |
| Memory/QA功能 | 同一问题集比较旧/新自动分段；平衡Yes/No及宏动作下降均≤2pp；单列GT分段参考、否定题、时间窗和多事件顺序 | 说明下游回归；QA错误不自动触发重训 |
| 留出/重复 | 录制留出set003、subject12及seed43均完整报告上述适用项及逐组差值，失败不隐藏 | 不能称稳定收益；不追加seed44 |

检索门槛同时在最终自动事件上做任务级比较：GT事件与自动事件单位不同，须用GT时间窗匹配的检索命中/覆盖率和误事件率，不能直接比较两个不同事件库的mAP而声称保持。逐类弱项、R@5、分数margin、动作F1也报告，不用平均掩盖弱项。

Memory/QA的“通过”只表示本轮迁移没有明显损伤既有下游，不表示原先不合格的多事件推理已经解决。若动态QA提升但GT分段QA仍错，明确保留该问题。

## 6. 批准后必须补的训练入口与可测试任务

现有 `Stage2Migration` 和 `V25Stage3` 是接口，不是已经完成原生数据准备的训练程序。以下工作包含在本方案准备范围，不提前执行。

### Task 1：固定分区与Evidence监督

**Files:** Create `src/continuous_mmwave/v25/data.py`、`tests/test_v25_training_data.py`、`configs/parrad-v25-training-pilot-v1.yaml`。

**Interfaces:** `build_training_manifest(source_manifest, spec) -> dict`；`build_evidence_targets(recording, label_spec, normalization) -> targets, mask`；`fit_normalization(fit_records, label_spec) -> dict`。产物记录recording/sensor/时间范围、标签路径、SHA、缺样和排除原因。

- [ ] 写并运行分区泄漏、双sensor绑定、未来帧泄漏、缺失mask和fit-only归一化失败测试。
- [ ] 实现数据v2固定分区、原14类映射复用、新增H5/CSV完整性与时间对齐检查、fit-only属性归一化及独立数据清单；保留原phase1不变。测试part/空目录拒绝、双sensor绑定、未知命令失败、MD副本去重和subject12解封前不可进入缓存。
- [ ] 在独立canonical/v25_dataset_v2中准备08/13的fit/dev/录制留出数据，保存来源/录制角色/排除原因SHA；subject12数组与标签转换按解封协议延后。固定MD统计代理字段与mask，不切换人体物理监督或encoder。
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
- [ ] 先跑旧回归及扩展dev整链与节拍回放，按原数值门槛决定开发资格；不读取subject12或录制留出效果来决定是否重复。
- [ ] 主seed全部开发门槛通过后执行seed43独立全流程；用同一预注册网格规则及终步，不能改变标准。
- [ ] 计划内全部终态与最终SHA锁定后，一次性解封四个subjects的set003及subject12全部四组录制，原模型和全部seed按相同规则评分；逐录制原始预测与缺失值完整保存。
- [ ] 服务器独立读回实际optimizer step、模块冻结SHA、checkpoint状态和预测；完整JSON/log归档，核验服务器/本地SHA和成员路径，再在本地重算指标。大权重保留服务器。
- [ ] 输出 `docs/v2.5-training-pilot-results-2026-10-06.md` 及进度ledger；将“完成、有效、保留检索、实时、泛化”分别结论。不替换生产模型，不自动继续新训练。

## 7. 输出、安全恢复与时间预估

建议服务器隔离代码 `/home/xibeidaxue/SEER/.audit-staging/v25-training-pilot-code-v1`；输出 `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/v25_training_pilot_v1/seed42`、`seed43`。目录已存在则停止或使用显式新版本，不能覆盖。

状态记录phase、实际step、PID、启动时间、输入SHA和最后成功checkpoint；SSH断线只恢复监控，不重复启动。网络恢复后先核对PID/lock/checkpoint，再继续读取。运行中不临时调整损失；非finite/来源SHA变化/后台错误停止并留证。

预计使用既有服务器单GPU；此前记录为2080Ti 11GB，但本轮没有重新查询其在线状态。不能把上次两组500步QA约16分钟直接当作本次多步展开gate训练用时。批准后在前20个正式更新给出基于实测的ETA：各阶段剩余成功步×step耗时＋缓存构建＋固定回放/归档；阶段分开列预计完成时刻。计划不是几十步探针，需按小时级实验安排资源，当前不承诺具体结束时间。

若新 gate 检索保留失败、Evidence无有效标签/收益或initialization probe失败，本轮在该点结束并给出一个明确失败归因及下一份有边界的建议；不反复跑相同审核，不自行延预算或解冻主干。真实工程bug可在授权范围内修复并重做相关检查，影响轨迹则保留失败run，用新目录说明重启原因，不能伪装成同一成功轨迹。

## 8. 审阅结论与批准范围

本计划已根据当前代码接口和用户v2.5章节做静态自审；所有准备任务未执行。上一轮本地相关测试86 passed/1 skipped仅说明模块接口，不替代本计划原生训练检查；上一轮独立审查曾因额度中断，不能称完整独立审查已通过。

用户批准表示授权：上述准备代码与原生预检、一个seed42固定预算主实验、达标后一个seed43重复、有限校准、整链、扩展fit/dev、录制留出及subject12受试者留出评估，以及结果归档。数据v2按明确标注的MD统计代理路径做受限实验；新增原始数据仅按数据章节转换，手势资料只整理去重，不进主训练；不授权静默改为其他人体物理标签、不授权官方test、QA重训、主干/原P解冻、部署替换或追加第三轮。

**当前状态：等待用户批准。实际训练启动次数：0。**

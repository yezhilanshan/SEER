from pathlib import Path
import json, hashlib, re, csv, html, textwrap, zipfile
from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path('D:/lab/SEER'); OUT=Path(__file__).parent
OUT.mkdir(exist_ok=True,parents=True)
groups=[]
def group(title,period,conclusion):
    g=dict(title=title,period=period,conclusion=conclusion,items=[]);groups.append(g)
def add(title,problem,attempt,result,next_,sources,status='已实验'):
    groups[-1]['items'].append(dict(id=f'T{sum(len(g["items"]) for g in groups)+1:02}',title=title,problem=problem,attempt=attempt,result=result,next=next_,sources=sources.split('|'),status=status))

group('架构起点与数据适配','早期设计至 2026年9月','先确立连续状态和事件记忆，再以轻量模型与真实雷达数据逐步验证。设计版本的增加不等于各版本都完成了端到端实验。')
add('v2.0 七模块与双路径','固定片段识别无法表达连续行为与长期历史。','设计雷达编码、物理变化提议、语义状态、事件确认、分层记忆及按需语言生成。','形成持续理解中产生事件的总体目标；方案本身不是效果证据。','逐步冻结模块结构，降低前端和LLM计算负担。','计划/想法三/方案书v2.0.md','设计记录')
add('v2.1 Fork and Replay','候选误触发后需要恢复旧事件状态；前端Token维度过大。','触发时冻结Active快照、Candidate吸收后续雷达；拒绝后回放；前端收紧到4×256，采用Prefix LM掩码。','明确三个主要可学习模块和若干轻量头；为后续状态管理提供契约。','v2.2进一步解决跨模态参数量、查询依赖和时间尺度。','计划/想法三/方案书v2.1.md','设计记录')
add('v2.2 与 v2.3 计算和时间契约','直接跨模态映射参数过多，任意用户指令会改变客观事件状态。','512维瓶颈注意力、固定Event Prompt、5 Hz语义更新、按物理时间常数更新原型；逐帧骨架监督和Commit一致性。','文档固定了主要实现规格。v2.3沿用大量v2.2内容，不能仅凭文件名认定独立实验收益。','迁移到实际可用数据和小模型，分阶段验证训练链。','计划/想法三/方案书v2.2.md|计划/想法三/方案书v2.3.md','设计记录')
add('PARrad 前端与 label run 流式训练','真实数据以RD/MD与命令标签为主，不能假装具备完整点云骨架监督。','适配PARrad；按标签run预训练，再在完整录制上连续状态训练；修复padding污染BatchNorm及短尾窗口。','3+1轮pilot测试accuracy18.56%、Macro-F1 13.94%；工程跑通未等于识别可用，GPU温度与降频也限制吞吐。','扩训练并建立Stage2 EventFormer、Stage3转换学习及独立诊断。','docs/superpowers/plans/2026-09-20-parrad-label-run-streaming-progress.md')
add('v2.4 语义与物理双投影','事件Token既要保留语义又要包含雷达证据，Streaming与Commit双向约束可能冲突。','引入Semantic/Evidence双投影，Commit Teacher向Streaming Student蒸馏；Fork and Shadow解释为旧/新事件竞争。','形成后续可解释性和稳定性探针框架；不据设计宣称消除幻觉。','围绕真实在线分段质量展开根因排查。','计划/想法三/方案书v2.4.md','设计记录')

group('旧在线事件链的根因排查','2026年9月29日至10月4日','从Verify和候选成熟度一路检查到表示、读出与训练目标。多数修补没有达到放行门槛，但逐步排除了简单解释。')
add('起点 上游分段与 Verify 审计','原在线回放边界Precision23.16%、Recall44.90%、F1 0.3056，误分割严重。','重标hard negative并检查Verify、上游来源与候选。','重训Verify后F1由0.4167降至0.3885，过分割1.46升至1.78；更强负样本没有解决问题。','检查候选是否过早、Shadow信号是否不足。','docs/stage3-upstream-audit-2026-09-29.md|docs/stage3-verify-candidate-audit-2026-09-29.md')
add('Shadow 与强制延长候选','怀疑Candidate早期状态不成熟，无法与Active区分。','做同录制候选影子诊断、强制更长观察窗口和等长候选探针。','等长1336候选余弦0.5135→0.5093，没有清楚成熟拐点；不支持单靠等待救活。','隔离初始化方式和maturity gate。','docs/stage3-candidate-shadow-diagnostic-2026-09-29.md|docs/stage3-forced-horizon-probe-2026-09-29.md')
add('初始化与 maturity gate 的 2×2','候选初始状态与门控可能造成新旧状态难区分。','组合初始状态与成熟度门控，逐层定位其影响。','四臂AUROC约0.42–0.51；门控对maturity弱敏感，不能把失败简单归因于成熟度。','冻结层逐层做容量匹配可解性探针。','docs/stage3-fork-init-gate-ablation-2026-09-30.md|docs/stage3-fork-layer-localization-2026-09-30.md')
add('冻结层可解性','需要判断事件转换信息是否已在上游丢失。','比较F、隐藏层、事件状态和投影向量的线性可解性。','2.16秒处F/Ê/E/z约0.788/0.743/0.736/0.705，没有整体信息断崖；距离读出明显更弱。','保留表示，先比较历史长度、读出容量及方向信息。','docs/stage3-frozen-layer-decodability-2026-10-01.md')
add('短历史与读出容量','弱性能可能来自读出太小或历史不足。','K1/K3/K8拼历史，关系读出与MLP容量对照，比较cos/L2距离。','加历史没有稳定改善；MLP32 0.6283低于投影基线0.6736；cos/L2约0.504/0.540近随机。','转向监督任务和语义物理互补，而非盲目加宽模型。','docs/stage3-frozen-short-horizon-sequential-results-2026-10-02.md|docs/stage3-radar-readout-pilot-2026-10-02.md')
add('语义与物理互补','期望F与E/z结合能补足单分支。','冻结表示做F+E、F+z受控比较，报告固定误报率操作点。','组合未稳定胜过F；部分召回改善伴随更高FPR，阈值跨受试者迁移困难。','尝试transition-aware表示监督，保留检索约束。','docs/stage3-semantic-physical-complementarity-results-2026-10-02.md')
add('Transition aware 投影恢复','过渡监督可能使语义检索坐标漂移。','做投影恢复、rehearsal与关系监督，检查冻结参照及检索保护。','历史诊断出现检索下降21.43个百分点；任务目标与评估口径并不天然一致。','将主实验门槛预注册，固定预算并做重复折。','docs/stage3-transition-aware-projection-recovery-results-2026-10-03.md|docs/stage3-transition-aware-rehearsal-audit-2026-10-03.md|docs/stage3-transition-aware-relational-results-2026-10-03.md')
add('主试验 复现 预算与均衡','一次pilot收益可能来自折或预算选择。','主实验、独立复现、20/50/100步预算与subject均衡对照。','pilot约+2.48pp，复现约+0.48pp，第三折转负；50步+1.60pp仅保留研究基线，类别均衡无收益。','停止把单次开发折改善当稳定结论，转向更明确的接口和候选问题。','docs/stage3-transition-aware-primary-results-2026-10-03.md|docs/stage3-transition-aware-primary-replication-results-2026-10-03.md|docs/stage3-transition-aware-budget-results-2026-10-04.md|docs/stage3-transition-aware-subject-balance-results-2026-10-04.md')

group('分层记忆与雷达语言接口','2026年10月4日至6日','记忆晋升和问答链可运行，但自由生成、单事件命名与完整顺序暴露了不同层次的接口问题。')
add('Stage4 分层记忆首轮','需要判断L0/L1/L2压缩能否支撑长历史问答。','固定前端，训练压缩器100步，真实约600秒录制回放并触发两级晋升。','压缩CE下降，但实际回答不是有效事件描述；190个确认事件同时有大量边界误报。','先限制问答任务，再做单事件与短上下文诊断。','docs/stage4-memory-pilot-results-2026-10-04.md')
add('聊天 QA 到单事件 QA','复杂自由回答混合了输入、提示、解码器和多事件错误。','逐步测试聊天、单事件存在性、动作命名和短上下文。','存在性与格式可改善，可靠命名和完整顺序仍不足；不能据链路可运行宣称理解通过。','隔离压缩、事件编号及时间顺序编码。','docs/stage4-chat-qa-pilot-results-2026-10-04.md|docs/stage4-single-event-qa-results-2026-10-05.md|docs/stage4-short-context-qa-results-2026-10-05.md')
add('压缩与事件编号假设','怀疑压缩丢信息或没有明确事件分隔。','比较4个摘要Token与32个原始Token，加入事件编号前缀。','存在性约68.8%/70.8%，不支持压缩器为唯一主因；编号前缀反而降低部分存在性结果。','以纯文字Oracle检查语言读者本身是否可靠。','docs/stage4-short-context-qa-results-2026-10-05.md|docs/stage4-event-boundary-qa-results-2026-10-05.md')
add('时间 QA 前置门槛与解码修复','纯文字输入仍无法可靠完成多事件问答。','none/order/time残差融合方案先过文本门槛；检查提示和文本解码合同。','时间QA本轮门槛失败、实际更新0；文字四事件顺序14/48不能与旧dev 0/24相减。','先修复文本合同，再在独立实验中研究时序融合。','docs/stage4-temporal-qa-results-2026-10-05.md|docs/stage4-decoder-repair-2026-10-05.md|docs/stage4-text-decoder-results-2026-10-05.md')
add('Reader 融合与语义对齐','雷达Token没有自动进入冻结语言读者的语义坐标。','比较none/order/time融合及雷达语义CE、蒸馏等目标。','旧24个四事件dev完整顺序仍0/24；单事件零步18.98%，CE500步76.14%，固定蒸馏47.53%。','优先建立可靠单事件接口，再研究多事件；文字列表100%不代表雷达读出100%。','docs/stage4-reader-fusion-results-2026-10-06.md|docs/stage4-radar-semantic-alignment-results-2026-10-06.md|docs/stage4-radar-semantic-alignment-v2-results-2026-10-06.md')

group('v2.5 证据累计与候选过滤','2026年10月6日至10日','从门控迁移到Proposal训练、触发器和双源池过滤，结果集中指向低误报预算下选择性不足，以及Oracle训练与时间局部任务的域差。')
add('v2.5 拆分提议 验证 定位','旧Verify同时混合变化、成熟度和边界问题。','删除Verify Logistic与独立损失，使用Candidate Episode、顺序证据累计、确认后定位与Active重建。','实现了独立v25路径；设计中的负证据撤销与完整在线能力不能由代码存在推出。','训练前先验证旧权重迁移及检索保留。','计划/想法三/方案书v2.5.md|计划/想法三/方案书v2.5 (2).md|docs/v2.5-implementation-status-2026-10-06.md','设计与实现')
add('门控迁移失败与兼容过渡','sigmoid权重直接复制到tanh不等价。','做零步热启动、精确输出坐标转换和正alpha兼容架构。','raw负alpha72.46%、状态严重外推且有非有限值；兼容转换保住原检索，raw迁移训练未启动。','在兼容坐标下只训练Evidence头。','docs/v2.5-gate-migration-diagnostic-results-2026-10-07.md')
add('兼容 Evidence 训练','要验证事件Token能否读出雷达统计且不破坏检索。','冻结前端与事件状态，独立Evidence头500步。','可读出部分物理统计、检索基本保持；共享表示未更新，不能说物理信息被新写入事件状态。','训练独立物理Proposal，验证实际低误报操作点。','docs/v2.5-compatible-evidence-results-2026-10-07.md')
add('Physical Proposal 与 OR 融合','高召回提议希望补足语义分支。','训练Local/Radar及OR，在固定FP预算比较。','500步@8FP融合44.18%，低于Local50.00%及Radar56.39%；OR并未自然产生互补。','零更新归因：候选覆盖、触发器、损失与峰形。','docs/v2.5-physical-proposal-results-2026-10-08-server-readback.md|docs/v2.5-proposal-zero-diagnostic-results-2026-10-08.md')
add('迟滞触发与峰形','分数长平台可能导致峰被抑制。','测试因果迟滞触发与不同释放条件。','更严释放未解决长平台，部分融合召回仅20.74–28.55%。','检查监督和损失，而非继续叠触发规则。','docs/v2.5-proposal-causal-trigger-results-2026-10-08.md')
add('监督诊断与 unweighted BCE','怀疑加权Gaussian BCE损害选择性。','检查正负权重实际质量，再移除加权做A/B。','Radar@8下降11.51pp，8/8单元退化；加权不是当前低召回的简单根因。','尝试窗口峰辅助，但pairwise排序仍是另一未执行假设。','docs/stage3a-proposal-supervision-diagnostic-results-2026-10-08.md|docs/stage3a-proposal-loss-ablation-results-2026-10-08.md')
add('窗口峰辅助 CE','希望将宽分数平台收缩为可发出的峰。','固定主损失与预算加入window peak辅助。','三预算Radar召回均降；平台P95由9.20增至10.08秒。','不将该失败外推到从未执行的pairwise排序损失。','docs/stage3a-window-peak-ablation-results-2026-10-09.md')
add('GRU 残差融合与 matched 读出','GRU信息是否能直接改进Radar分数。','残差融合、等参数读出，以及State/Delta/Combined比较。','融合B召回37.36%，低于55.26%/56.11%对照；Rescue57、Harm189；点级AP上升不等于事件召回上升。','保留双源候选池，避免复用已失败残差分数。','docs/stage3ag-residual-fusion-results-2026-10-09.md|docs/proposal-readout-matched-results-2026-10-09.md')
add('变化量诊断与 TCN 监督遗漏','简单变化量是否可直接当边界信号，TCN是否真正参与输出。','MD GRU差分、方向/距离诊断并检查TCN梯度和gate。','简单距离接近随机；发现RD TCN注入gate仍为0、frame_hidden取点在其之前。只证明监督路径遗漏，不证明TCN无效。','区分增加RD和替换时序骨干，优先检查真实信息路径。','docs/md-gru-transition-signal-results-2026-10-09.md|docs/radar-gru-two-diagnostics-results-2026-10-09.md|docs/tcn-radar-encoder-inspection-2026-10-09.md')
add('双源池与候选可过滤性','合并后召回91.48%，但39.92候选每分钟的负担过大。','固定池训练18个差异/稳定性线性Probe，主方法观察1.08秒。','召回85.51%，FP24.407/min；召回损失5.966pp、假候选仅降23.40%，未过50%过滤且损失≤5pp门槛。','转而核查语义训练任务和协议GT，不能说完整表示没有信息。','docs/candidate-filterability-results-2026-10-09.md')
add('Oracle 到时间局部任务域差','Oracle配对上有效的语义关系未必能用于在线候选。','冻结Oracle50，比较随机Oracle配对O、固定时间锚点T及Proposal P。','dev AUROC O/T/P为0.7692/0.4807/0.4548；T已近随机，不能主要归咎Proposal选择。','修正语义配对任务；并行退回更简单的离线描述链。','docs/stage3b-semantic-pilot-results-2026-10-09.md|docs/stage3b-oracle-proposal-domain-gap-results-2026-10-10.md')

group('离线描述与数据及识别基线','2026年10月10日至16日','主动降低系统复杂度，并逐步加强公平分类基线。生成模型的表面优势被数量控制对照重新解释，标签时钟和跨人泛化成为独立问题。')
add('离线 Radar to Text MVP','在线事件链、分支状态与记忆机制使端到端归因困难。','固定片段RD/MD CNN→BiGRU→8个Radar Tokens→Projector→Qwen；先Stage1再冻结LLM对齐。','首轮subject06完整序列25.00%，模板11.16%；但动作Recall46.28%低于模板55.51%。','拆解编码、对齐、分类头与生成覆盖错误。','docs/offline-radar-to-text.md|docs/radar-text-server-pilot-2026-10-10.md')
add('错误归因与分类头 H0 到 H4','不确定生成漏动作来自上游、Align退化还是弱读出。','冻结多层探针、平滑模板、前缀无雷达、错配/扰动及H0–H4头消融。','BiGRU有益、Align未整体退化；延长线性头训练比单独类权重更有效，稳态混淆与尾动作缺失仍明显。','在同窗口、同编码器下建立更强公平基线。','docs/radar-text-error-attribution-2026-10-11.md|docs/radar-text-head-ablation-2026-10-11.md')
add('强分类模板与同编码器 P0','原分类基线太弱，难以判断LLM贡献。','严格482共同窗口、同冻结Align特征，训练H1/H4线性读出。','Align H1/H4/Qwen序列准确率18.88%/18.67%/24.69%；H1 Recall65.91%高于Qwen46.06%。','控制动作数量和分段，检验序列优势是否来自计数。','docs/radar-text-strong-baselines-2026-10-11.md|docs/radar-text-align-readout-p0-2026-10-11.md')
add('P1 数量校准 HMM 与 Viterbi','分类过度输出和生成漏动作可能解释序列差异。','固定Encoder，对比原分类、HMM、预测数量段Viterbi与Qwen。','482窗序列准确率18.88%/25.52%/27.18%/24.69%；强数量模板超过Qwen，原生成优势不再保留。','避免将优势直接归因LLM；继续检查有序接口和数据泛化。','docs/radar-text-p1-evidence-count-2026-10-11.md')
add('RD only X Fi 迁移','预训练空间骨干是否更好。','比较原小CNN、随机ResNet18和预训练ResNet18，空间骨干冻结、统一新读出。','test逐帧Macro-F1 62.01%/62.25%/62.52%，验证集仍选择原CNN；不足支持直接替换或生成改善。','扩充受试者后独立从头训练，检验泛化。','docs/rd-only-xfi-transfer-2026-10-11.md|docs/xfi-mmwave-rd-transfer-assessment-2026-10-11.md')
add('PARrad 扩充和 RD 从头训练','旧数据规模小，历史测试已反复观察。','15人45session、89份记录、599799帧；9/3/3划分，单列旧06与新17/18。','ResNet18从头test受试者平均Macro-F1 50.16%，新17/18平均36.75%，跨人泛化仍弱。','检查预处理和标签时基；不以扩大训练掩盖监督问题。','docs/parrad-expanded-preparation-2026-10-11.md|docs/parrad-expanded-split-2026-10-11.md|docs/rd-resnet18-scratch-results-2026-10-13.md')
add('预处理审计与 Signed Log 消融','怀疑已处理RD上额外Signed Log损害信息。','P0保留/P1删除，统一初始化、采样、20轮和验证选优。','主指标58.268%→58.118%，差−0.150pp；未达追加seed阈值。AMP成功更新差1次，不能称完全一致。','保留原基线，分开处理标签和域差。','docs/parrad-preprocessing-assessment-2026-10-13.md|reports/parrad_preprocessing_audit.md|reports/parrad_signed_log_ablation.md')
add('Split BiRNN 方法复现','需要有代表性的连续识别对照。','23人153份MD，Split与普通BiGRU各五折6400步。','修正输出Weighted F1 80.06%/79.30%；即时前向58.23%，最终修正平均等待37.17秒。','精度和等待同时报告，不当成已验证低延迟在线方案。','docs/split-birnn-preparation-results-2026-10-14.md|docs/split-birnn-training-results-2026-10-14.md')
add('命令标签与真实动作时间审计','PARrad CSV命令时刻不等于真实动作起止；首命令等于录制起点的假设缺独立依据。','做150例候选审阅及14类42段双雷达复核，并查询公开目录和H5元数据。','可见提前/滞后与双视角差异，但缺同步视频音频或硬件时钟，不能测出真实执行延迟或宣判错标。','迁移具有官方实例区间的HARrad，并保留未知区域。','docs/candidate-filterability-results-2026-10-09.md|docs/v261-clip-review-results-2026-10-16.md')

group('v2.6 预测驱动与 HARrad 重训','2026年10月14日至11月6日','先证明事件状态具有超越普通时序模型的额外预测价值，再谈预测梯度驱动自主Reset。两次重训与随机重置诊断提供了局部支持，但当前Stage1仍未放行。')
add('v2.6 用预测状态替代显式 Verify','v2.5大量手工候选与验证机制仍未解决选择性。','预测驱动Keep/Reset；冻结target防塌缩，并用语义/实例锚定避免纯动力学分段。','形成PSEM设计；运动学分段不必然等于语义事件，预测损失不能代替语义监督。','先以冻结前端做P0/P1预测对照。','计划/想法三/方案书v2.6.md','设计记录')
add('PARrad 预测 pilot 与 P0 mask 审计','首轮Event预测不如Persistence。','四神经头各300步、P0零更新分解动态性与采样。','Persistence MSE0.03366，Event0.07351；发现mask混用观测有效与事件归属，所有合法h1配对都未跨边界。','拆mask，用零初始化残差头做P1。','docs/v26-prediction-pilot-results-2026-10-15.md|docs/v26-p0-diagnostic-results-2026-10-15.md')
add('P1 残差预测和 P1 C 多跨度','需要排除坏初始化和过短horizon。','P1-A/B恢复跨界配对，P1-C统一完整历史、h1/3/5和E错配常量消融。','残差GRU Skill+1.21%，Event−1.62%；h3/h5 motion MLP+11.73%/+19.24%，Event+7.20%/+15.71%；常量E多处更好。','不以该结果启动Reset；重新检查表示和监督。','docs/v26-p1-results-2026-10-15.md|docs/v26-p1c-preparation-results-2026-10-15.md|docs/v26-p1c-results-2026-10-15.md')
add('v2.6.1 PARrad 四臂联合','希望Dynamics与语义通过共享Adapter互相帮助。','MLP/A-only/B-only/A+B，三seed固定预算，检查语义滞留和命令代理时间。','h5 Skill22.78%/21.53%/21.49%；语义B17.40%、AB17.23%，无稳定联合净收益；旧类滞留与跨动作下降仍在。','先做可靠监督的识别单任务基线，转到HARrad。','docs/v261-joint-results-2026-10-16.md|docs/v261-semantic-diagnostic-results-2026-10-16.md|docs/v261-joint-diagnosis-next-steps-2026-10-16.md')
add('HARrad 格式审计与语义基线','需要自然连续录制、明确实例区间和跨人划分。','审计9人37录制233515帧、12类3852区间；subject1–6训练、7验证，保留36.6%未知区域。','MD GRU/RD CNN GRU三seed验证Macro-F1 78.89%/84.52%，RD均胜；官方随机划分存在受试者录制共享。','冻结RD前端训练流式EventFormer，再检查重置依赖。','docs/harrad-format-audit-2026-10-16.md|docs/harrad-semantic-baseline-results-2026-10-17.md')
add('EventFormer 预热与官方起点回放','片段识别好不等于连续状态可用。','35录制导出，语义预热5轮3775步，比较官方起点Reset与全程不Reset。','旧预热best77.01%，连续无Reset12.89%；官方起点整流76.73%，小差异支持批/chunk与FP16数值影响。','建立监督边界基线，独立检验Reset需求。','docs/harrad-event-warmup-progress-2026-10-17.md|docs/harrad-oracle-start-replay-2026-10-17.md')
add('连续监督边界基线','自主Reset前需要可用边界信号。','冻结CNN/GRU训练33154参数起止头，31条训练录制，20轮620步。','subject7 ±4帧起/止F1 55.94%/67.33%，胜变化量与周期；起点Recall47.95%，未知输出负担仍大。','不能直接把头接成已通过的自主Reset；分离语义与Dynamics。','docs/harrad-boundary-baseline-2026-10-17.md')
add('HARrad Dynamics 语义联合预训练','PARrad上的联合失败是否由标签代理引起。','M/A/B/AB四臂三seed各1000更新，冻结CNN与Qwen。','h5 Skill M29.64%、A30.91%、AB28.36%；语义B60.43%、AB58.80%，共享等权联合仍无净收益。','改成独立表示，测试对齐与调制是否互补。','docs/harrad-joint-prediction-results-2026-10-18.md')
add('对齐乘调制 2×2','独立语义对齐和事件引导调制能否避免共享目标冲突。','B0、仅对齐、仅调制、组合四组三seed12000更新。','Macro-F1 62.02%/62.96%/62.97%/61.54%，均低于同面板初值64.52%；关调制只降约0.05pp。','保留独立A/B前端，研究Reset机制是否技术上连通。','docs/harrad-alignment-modulation-results-2026-10-22.md')
add('10月22日 Reset 闭环机制试验','要验证future和语义梯度能否真实影响Policy及状态。','保留G、重置L/E；Stage2A1000、2B300、pilot两臂各50；正式control/full三seed计划。','机制梯度预检通过；本地历史记录停在control seed42 80/1000，0步边界F1 47.75%且FP79.26/min，不是终态收益。','没有找到可信正式终态报告；与11月新Stage1路线分别记录。','docs/harrad-reset-closed-loop-progress-2026-10-22.md','有启动记录 终态未核实')
add('11月统一前端重训与 Stage1','为想法三/六比较消除前端差异，重新检验冻结事件状态预测。','旧RD seed42共用前端；新EventFormer片段79.67%、官方起点79.71%；未来3/9/15原帧对应0.2/0.6/1秒。','GRU MSE0.75589/0.92940/0.94881；Event0.76464/0.91524/0.95046，仅0.6秒均值更好，Stage1 No-Go。','做分场景、逐seed、随机边界和NoReset诊断。','docs/idea3-idea6-harrad-unified-results-2026-11-05.md|docs/harrad-stage1-event-prediction-results-2026-11-05.md')
add('Stage1 A B C 诊断','0.6秒收益是否由特定录制、慢转换或同类重复掩盖。','按场景复算，录制级配对Bootstrap；单份随机边界重训3seed。','慢转换与同类重复无稳定优势；稳定长动作内部较一致；0.6秒收益主要来自seed43。正确Reset优于随机，但不足超GRU全跨度。','重复随机边界分布并加入NoReset，避免一份随机排列偶然性。','docs/harrad-stage1-diagnostic-results-2026-11-05.md')
add('实验二 十份随机排列与 No Reset','需要分离真实事件对齐和限制状态历史长度的作用。','GT、NoReset、10随机组各3seed，共36模型36000更新；保持重置次数及更新间隔多重集，实际间隔最多差2帧。','GT MSE0.76553/0.92015/0.95065，随机0.78025/0.93875/0.96827，NoReset0.78939/0.94152/0.96707；GT均优，Random与NoReset无统一排序。','支持事件对齐价值；仍受subject7、GT预热、年龄输入和间隔近似匹配限制，Stage1仍No-Go。','docs/experiment2-reset-results-2026-11-06.md')

group('想法五 潜在记忆与生成诊断','2026年10月18日至19日','绕开显式切分不等于绕开雷达语言接口。已实现多尺度记忆与LMT更新，但核心动态重检索收益尚未进入正式G2验证。')
add('无显式分段的 Latent Memory','显式边界错误会传到下游记忆与语言。','连续cells进入短中长记忆，Initial LMT分组Bootstrap，Evolved LMT驱动多轮重新检索。','代码、真实模型前向与梯度链已有验证；均值压缩是有损基线，不是Jenks；E0–E4正式效果未完成。','设G0先确认直接读出与文本Oracle，再进G2动态重检索。','计划/想法五/方案书v1.0.md|docs/latent-memory-implementation-2026-10-18.md','设计与实现')
add('G0 32窗 200步','LLM未必能保留顺序、重复和ongoing，可能无法作为记忆测量器。','训练Projector/time projection，冻结Qwen与适配器，固定4/8/16秒32窗。','序列5/32、含ongoing2/32，严格32/32未过；8秒和16秒序列均0；G2未启动。','先做提示词与解码文本对照。','计划/想法五/G0诊断结果与G2准入结论v1.0.md')
add('提示与解码对照','原提示可能鼓励摘要和去重；嵌入路径重复惩罚也可能影响结果。','明确逐项转写，统一输入路径，比较重复惩罚1.1/1.0及自由输出。','109旧例完整状态5→53，惩罚1.0为69；仍有循环截断。事后宽容解析不能替换原严格指标。','暂停提示搜索，回放终止分支。','计划/想法五/提示词与解码对照结果v1.0.md')
add('终止机制与语法检查','同时存在漏输出与已完整后循环，不能靠增大512上限解决。','分析900输出、681错例回放；验证438正确编码路径及1680动作边界。','决定点在句号结束/逗号继续，句号后EOS被强制；语法允许合法重复，非空GT完整后循环12例。','固定输出前缀，仅改变剩余证据作配对实验。','计划/想法五/终止机制诊断v1.0.md')
add('结束继续配对及 A B 判别','分数响应是否能变成可靠结束判断。','144对相同前缀对照；另试独立A/B完成状态裁判与标签交换。','惩罚1.0配对正确101/144，1.1降至79/144；独立A/B接口840条件中835换标签仍选A，不可当终止裁判。','结束A/B支线，转向不依赖标签映射的完整序列似然。','计划/想法五/固定前缀结束继续配对诊断v1.0.md|计划/想法五/完成状态独立判断诊断v1.0.md')
add('序列似然文本与雷达对照','生成失败不能直接证明冻结Qwen没有序列区分能力。','新文本面板错序/错类/错重复/ongoing候选评分，再用旧200步Projector做正确/零/错配雷达。','文本胜138/144、144/144、96/96、144/144；雷达148对108/63/79，但等Token仅17/26、16/26、15/26。','扩充均衡等Token面板，检查已有训练是否真正增强序列信息。','计划/想法五/序列似然诊断v1.0.md')
add('96窗检查点与接口归因','正答案似然升高可能只是通用语言偏好。','固定96窗213对候选，比较0/100/200步及正确、零、错配、块置乱。','新fit优势未稳定提高；正确答案在错配/置乱下也升高，正确与错误margin未稳定扩大。4412前向、0新增训练。','CE与CE加序列对比监督是下一项建议，尚未实施；G0/G2状态不变。','计划/想法五/雷达语义接口检查点对照v1.0.md')

group('想法六 CIF LIF 与 SAEF MVP','2026年10月24日至11月5日','将事件形成收敛为语义证据积分发火，再从拼接弱监督转向自然整流有监督MVP。真实结果呈召回与过分割取舍，尚未证明LIF和状态证据普遍优越。')
add('CIF 到 LIF 草案与首版','普通积分的微弱噪声会累积，动作实例不等于语音Token。','CIF教师、LIF蒸馏微调、事件分类及有界流式缓存；保留相邻同类。','首版机制和CLI人工特征smoke完成；34新增测试与旧HARrad69通过4跳过，真实训练未在该首版验收中完成。','用户选择自然整流最小架构，独立于旧拼接训练器。','计划/想法六/毫米波连续行为理解：LIF事件门 + LLM描述生成 方案书（草案）.md|docs/harrad-cif-lif-implementation-2026-10-27.md','设计与机制验证')
add('SAEF MVP 自然录制四臂','拼接片段有域差，需直接验证连续事件形成。','B0阈值上升沿、B1 IF、B2局部LIF、B3局部加状态LIF；共享冻结动作读出；fit/calibration/validation自主发火。','完整流、未知区域、时间积分、阈值校准和闭环入口具备；未接LLM。11月4日本地audit缺服务器资产不能当真实失败终态。','以随后真实锁定评估和统一比较更新历史状态。','计划/想法六/最小可行架构.md|docs/harrad-saef-mvp.md','设计与实现后已实测')
add('闭环阈值 tau 扫描与锁定复核','高召回是否伴随过多发火，状态反馈是否改变工作点。','在训练calibration扫描tau/阈值，锁定B2/B3-A/B3-B并评subject7及8–9。','锁定CSV中subject7边界F1 B2/B3-A/B3-B为84.40%/83.06%/80.68%；8–9合并76.42%/74.96%/70.71%。','这些旧event_f1与后续保守事件F1定义不同，不能混用；以统一口径复核完整事件。','outputs/saef-closed-loop-pareto-v1/summary.json|outputs/saef-locked-evaluation-v1/subject7-metrics.csv|outputs/saef-locked-evaluation-v1/subject8-9-metrics.csv')
add('想法三与六统一前端比较','不同前端、读出和指标会制造方案优劣假象。','共用旧RD前端，原生系统及相同EventFormer读出；6完整录制43277帧834实例，严格类别正确且IoU≥0.5。','8–9等权保守事件F1想法三30.92%、B2 25.19%、B3-A29.02%、B3-B22.82%；想法三误报低但漏边界多。','B3-A在subject7较好、跨人略弱；B3-B增召回损害事件F1。单seed不同操作点，不能宣称机制定论。','docs/idea3-idea6-harrad-unified-results-2026-11-05.md')

group('ROMSL 骨架辅助运动结构预训练','2026年11月5日至6日已有报告','从精确骨架重建改为运动幅值、径向可观测性和回波支持度监督。A/B有可重复资产，C/D正式几何仍未核实；代理实验显示径向目标未胜出且D权重退化。')
add('从精确骨架到可观测运动结构','雷达未必能恢复所有骨架细节，需要更符合传感器可见性的监督。','提出A骨架、B五维运动幅值、C径向投影、D回波支持度加权，采用PointNet加GRU。','这是想法六目录中的独立预训练分支，不是已验证的SAEF或LLM模块升级。','MM-Fi E01先跑A/B并锁定训练验证测试协议。','计划/想法六/骨架监督的新想法.md|计划/想法六/雷达可观测性引导的运动结构预训练方案书.md','设计记录')
add('E01 A B 首轮','先确认监督标签和冻结探针链可用。','S01–06训练、07–08验证、09–10测试，16帧窗口，seed42。','A/B Clip Macro-F1均100%，B共同运动MAE0.140065低于A0.150482；分类饱和，B对属性有直接目标优势。','增加随机编码器、单位审计、三seed及少样本探针。','reports/romsl_formal/report.md')
add('随机编码器与数据版本诊断','高分类分数可能来自容易任务或输入结构。','随机编码器零更新、同探针；逐录制比较ZIP与filtered版本。','随机Macro-F1 71.67%，A/B100%；270录制中184条版本有差异，单帧相同不能说明版本相同。','固定受试者ZIP版本并审计运动单位、归一化与聚合。','reports/romsl_random/report.md')
add('单位计算审计与三 seed 少样本','速度单位是否缺dt，单seed分类饱和是否稳定。','独立重算270录制，检查dt0.1秒、相对根速度、训练归一化及反标准化；加seed43/44和1/2/3/6条每类探针。','标签最大差4.71e-7；A100±0%、B97.78±3.85%；B MAE更低，但分类无一致优势；一条每类B90±12.02%高于A84.44±16.78%，方差大。','名义m/s仍以骨架按米解释为条件；低类别标注不等于降低骨架监督成本。','reports/romsl_results_audit/report.md')
add('正式 C D 几何与 Doppler 闸门','缺可验证世界到雷达外参、真实雷达原点、Doppler列与单位。','实现R/t或4×4检查、LOS、速度投影与单位符号校验，预设权重退化检测。','合成和回归42项通过，但正式配置verified=false；不能把坐标轴统一当真实外参。','采用训练数据配准生成明确标记的代理候选，另立探索实验。','reports/romsl_cd_validation/report.md|romsl_mvp/README.md','实现完成 正式条件未满足')
add('数据驱动外参候选','作者资料缺失，希望先研究代理几何可行性。','仅训练S01–06拟合、07–08诊断；Kabsch加鲁棒点到骨架肢段ICP、多起点和敏感性。','验证中位距离约11.14cm，留一人局部旋转≤0.72°、平移≤3.38cm；PROXY_GEOMETRY_ONLY，verified=false。','局部稳定不排除散射中心偏差；只授权为C_proxy/D_proxy探索依据。','reports/romsl_proxy_geometry/report.md')
add('C proxy 与 D proxy 三 seed','径向目标和回波加权是否比B更有效。','同A/B缓存与预算，代理外参；第5列按m/s待确认解释，D沿固定公式加权。','C/D Macro-F1均84.88±7.79%，低于B97.78±3.85%；训练48055和验证16019有效帧权重全1，三个seed模型张量逐项相同。','D未形成有效干预，不能推出回波加权思想无效；正式C/D仍需外参字段核验与新预注册。','reports/romsl_proxy_cd/report.md')

# Source registry and complete local record inventory. Hashes make the reviewed snapshot traceable.
all_items=[i for g in groups for i in g['items']]
source_paths=list(dict.fromkeys(p for i in all_items for p in i['sources']))
sources={p:f'S{n+1:03}' for n,p in enumerate(source_paths)}
missing=[p for p in source_paths if not (ROOT/p).exists()]
if missing: raise RuntimeError(missing)
inventory=[]
for base in ['docs','计划','reports','.superpowers/sdd']:
    for p in sorted((ROOT/base).rglob('*.md')):
        if OUT in p.parents: continue
        raw=p.read_bytes();t=raw.decode('utf-8-sig',errors='replace')
        inventory.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),headings=re.findall(r'^#{1,3} (.+)$',t,re.M),used_as_primary=p.relative_to(ROOT).as_posix() in sources))
(OUT/'source-inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'research-data.json').write_text(json.dumps(dict(date='2026-11-07',groups=groups,sources=sources,inventory_count=len(inventory)),ensure_ascii=False,indent=2),encoding='utf-8')

# Printable visual mind map, with compact problem/result chains.
fontpath='C:/Windows/Fonts/msyh.ttc'; boldpath='C:/Windows/Fonts/msyhbd.ttc'
def font(n,b=False):return ImageFont.truetype(boldpath if b else fontpath,n)
colors=['#236579','#506c93','#7c638e','#a16c45','#3b7d69','#526ba3','#8b607b','#7c7644','#617c5b']
W,H=3300,2900
im=Image.new('RGB',(W,H),'#f7f9fc');d=ImageDraw.Draw(im)
d.text((100,65),'SEER 研究路线与问题驱动迭代',font=font(66,True),fill='#14283e')
d.text((104,160),'设计 → 实验 → 暴露问题 → 新尝试     |     截至 2026年11月7日',font=font(29),fill='#52647a')
boxW,boxH=980,760
for n,g in enumerate(groups):
    x=100+(n%3)*1060;y=270+(n//3)*840;c=colors[n]
    d.rounded_rectangle((x,y,x+boxW,y+boxH),radius=24,fill='white',outline='#d8e1eb',width=2)
    d.rounded_rectangle((x+25,y+27,x+75,y+77),radius=12,fill=c)
    d.text((x+39,y+31),str(n+1),font=font(30,True),fill='white')
    title=g['title'];ls=textwrap.wrap(title,width=21)
    for k,line in enumerate(ls):d.text((x+94,y+28+k*40),line,font=font(34,True),fill='#172b40')
    yy=y+125
    # Five synopsis nodes retain the major changes; every detailed attempt is in the HTML and report.
    summaries=[
      ['v2.0 七模块与双路径 → v2.1 状态恢复','v2.2/2.3 瓶颈注意力与固定事件提示','PARrad label run → 完整录制流式训练','pilot识别弱 + 计算受限 → 分阶段验证','v2.4 双投影与Teacher蒸馏'],
      ['Verify重训无益 → 候选成熟度审计','延长候选无拐点 → 初始化/gate消融','冻结层可解 → 无整体信息断崖','加历史/加容量未救活 → 改监督','Transition-aware复现不稳 → 不放行'],
      ['L0/L1/L2晋升跑通 → 实际QA仍错误','聊天 → 单事件 → 短上下文逐层收缩','编号/时间编码 → 顺序仍未突破','文本合同未过 → 部分训练0更新停止','CE对齐有效 → 多事件接口仍待解决'],
      ['tanh直接迁移发散 → 兼容坐标','Evidence可读 → 不代表状态学到新信息','OR/迟滞/损失/峰辅助 → 无稳定收益','残差融合退化 → 双源候选池','高覆盖低选择性 → 过滤门槛失败'],
      ['退回离线CNN BiGRU → Qwen描述','25%序列正确 → Recall仍漏动作','同编码器公平对照 → 数量控制','Viterbi 27.18% > Qwen 24.69%','扩数据/预处理/迁移 → 泛化仍弱'],
      ['v2.6预测驱动 → 先验证额外预测价值','PARrad mask修正 → Event仍不占优','HARrad重训 → 无Reset语义显著退化','联合/调制无净益 → 独立Stage1','GT胜随机/不重置 → 仍未全面胜GRU'],
      ['不显式分段 → 多尺度记忆与LMT','G0 200步 → 完整状态2/32未过','提示改善 → 漏输出与循环并存','终止裁判A偏置 → 转完整序列似然','等Token对照 → 接口增益未稳定'],
      ['CIF教师 → LIF蒸馏机制实现','拼接域差 → 自然录制SAEF四臂','B0阈值/B1 IF/B2 LIF/B3状态LIF','统一前端 → 召回与误报明确取舍','跨人事件F1 三30.92% / 六29.02%'],
      ['A骨架 / B幅值 → 分类饱和','随机编码器对照 → 预训练有用','三seed/少样本 → B分类优势不稳','外参未核实 → C/D仅作proxy','D权重全1 → C与D完全等价']][n]
    for k,s in enumerate(summaries):
        cy=yy+k*105
        if k<4:d.line((x+48,cy+30,x+48,cy+118),fill='#ccd7e2',width=3)
        d.ellipse((x+39,cy+15,x+57,cy+33),fill=c)
        for a,line in enumerate(textwrap.wrap(s, width=26)):
            d.text((x+78,cy+a*35),line,font=font(28),fill='#31465b')
    d.text((x+30,y+boxH-55),f'{g["period"]}  ·  {len(g["items"])}项',font=font(23),fill='#65768a')
d.text((105,2810),'状态分开记录：设计构想 / 机制跑通 / 真实数据实验 / 尚未核实    详细节点与来源见可交互导图及Word',font=font(26),fill='#52647a')
im.save(OUT/'SEER研究路线思维导图.png')

# Editable Word, manual pages contain up to three attempts to keep evidence readable.
doc=Document();sec=doc.sections[0]
sec.page_height=Cm(29.7);sec.page_width=Cm(21);sec.top_margin=Cm(2);sec.bottom_margin=Cm(1.8);sec.left_margin=sec.right_margin=Cm(2)
styles=doc.styles
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3']:
    st=styles[name];st.font.name='Microsoft YaHei';st.font.color.rgb=RGBColor(0,0,0)
    st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
styles['Normal'].font.size=Pt(11);styles['Normal'].paragraph_format.line_spacing=1.25;styles['Normal'].paragraph_format.space_after=Pt(6)
styles['Title'].font.size=Pt(25);styles['Heading 1'].font.size=Pt(18);styles['Heading 2'].font.size=Pt(12)
for name in ['Heading 1','Heading 2']:styles[name].paragraph_format.space_after=Pt(8)
footer=sec.footer.paragraphs[0];footer.alignment=2
f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');footer._p.append(f)
def p(t='',bold=False):
    a=doc.add_paragraph();r=a.add_run(t);r.bold=bold;return a
def h(t,l=1):doc.add_heading(t,l)
def page():doc.add_page_break()
doc.add_paragraph('SEER 研究方案与实验演进复盘',style='Title')
p('问题如何推动下一次尝试',True);p('截至 2026 年 11 月 7 日')
p('研究目标始终围绕毫米波连续人体行为理解：从连续雷达信号形成可解释的事件和状态，再支持按时间组织的描述与历史查询。实际探索分成多条并行路线，而非一条不断升级且已全部验证的模型。')
p('最稳定的阶段性认识是：边界覆盖不等于选择性，片段识别不等于无重置长流理解，文字可读不等于雷达语义接口可靠，真实边界重置有价值也不等于已胜过普通时序模型。ROMSL 的回波权重退化则说明，实验组名不同未必产生了不同的实际训练干预。')
p(f'本报告整理 {len(groups)} 个主题、{len(all_items)} 项方案与诊断节点。每项按“问题—尝试—结果—后续”展开，并附本地来源编号。主要证据来自方案书、阶段结果报告、执行记录、锁定评估表与相关聊天正文；当前聊天索引只用于补充核查。')
p('阅读方式',True)
for n,g in enumerate(groups):p(f'{n+1:02}  {g["title"]}  ·  {len(g["items"])}项')
p('使用边界',True)
p('数值均为历史记录及本地证据中的结果，本次不重新训练、不接入服务器运行实验。设计、人工特征机制测试、真实数据结果与缺少终态证据分别标注。未知区域不自动归类为Idle；不同数据集、时间粒度、前端、面板和评价定义的分数不作跨轮相减。')
page();h('研究路线总览')
p('图中的箭头表示问题推动的下一次尝试。各列是并行分支或阶段主题，不能把同一时间的不同路线拼成一套已验证系统。完整节点在随附可交互思维导图中展开。')
doc.add_picture(str(OUT/'SEER研究路线思维导图.png'),width=Cm(17))
p('核心状态',True)
p('想法三：预测额外价值尚未过Stage1；想法五：G0未过，G2未启动；想法六：SAEF已做真实比较，但增益存在召回与误报取舍；ROMSL：A/B有补充稳定性结果，正式C/D缺几何与字段核验，proxy不能替代正式验证。')
for gn,g in enumerate(groups):
    remaining=len(g['items']);count=(remaining+3)//4;chunks=[];pos=0
    for k in range(count):
        size=(remaining+count-k-1)//(count-k);chunks.append((pos,size));pos+=size;remaining-=size
    for start,size in chunks:
        page();h(f'{gn+1:02}  {g["title"]}'+('  续' if start else ''))
        p(g['period'])
        if start==0:p(g['conclusion'],True)
        for i in g['items'][start:start+size]:
            h(f'{i["id"]}  {i["title"]}',2)
            for label,key in [('遇到的问题','problem'),('新的尝试','attempt'),('观察到的结果','result'),('由此转向','next')]:
                a=doc.add_paragraph();a.add_run(label+'  ').bold=True;a.add_run(i[key])
            a=p('状态  '+i['status']+'    来源  '+'  '.join('['+sources[s]+']' for s in i['sources']))
            for r in a.runs:r.font.size=Pt(8);r.font.color.rgb=RGBColor.from_string('596675')
page();h('跨路线判断与仍开放的问题')
takeaways=[
('候选覆盖不是事件质量','高召回双源池需要足够选择性。降低误报、保住召回、避免重复峰竞争必须共同计分；AP/AUROC、零阈值覆盖和最终事件F1不能互相替代。'),
('语言接口是独立研究对象','从Stage4到离线描述和想法五反复出现“格式或正答案似然改善，但完整顺序或证据相关差距未改善”。应同时测文字Oracle、正确/错配/置乱雷达、等Token候选与完整生成。'),
('Reset 的证据层级必须分开','官方起点重置对语义预热很重要，GT重置也胜随机和NoReset，但GT预热、年龄及历史长度同时变化。当前证据不能推到预测梯度改善自主Policy，更不能推到跨人优越。'),
('SAEF 当前体现操作点取舍','状态证据LIF并非在所有受试者和指标上占优。应在相同读出、决策频率、误报预算和多seed下比较；subject8–9已经读出，后续只能称跨人复核或补充分析。'),
('ROMSL 必须验证实际干预','B在速度属性上有直接目标优势，分类已接近饱和。C需要真实雷达原点，D需要字段单位和非退化权重；本轮D与C相同不构成对有效加权方法的否证。'),
('优先级建议而非已执行事项','优先补跨受试者与年龄/周期长度控制；锁定正式外参和Doppler资料；为语言接口预注册CE对比序列对比目标；保留pairwise排序、正式动态重检索和LLM接入为未验证项。')]
for t,b in takeaways:h(t,2);p(b)
page();h('指标与证据口径')
for t,b in [
('百分点与百分比','两个F1百分值之差写为百分点。不同折、不同拟合probe及不同旧权重的绝对值不可直接比较。'),
('保守事件 F1','2026年11月5日统一比较要求类别正确、完整预测区间IoU≥0.5且一对一匹配。旧SAEF锁定表中的event_f1不能直接替换此定义。'),
('边界 F1 与误报','必须同时说明容差、起止合并规则、有效已知时长和未知区间输出。subject8–9合并计数与逐人等权平均不是相同聚合。'),
('在线 Macro F1','SAEF不同门共享同一动作头，因此相同逐帧分数不能证明事件门效果相同或发火改善了语义。未知标注不能伪造为第13个背景类。'),
('预测 MSE 与 NLL','未来标签与输入可用cutoff绑定，target固定。NLL改善需同时看MSE、方差和覆盖率；逐坐标90%覆盖不是256维联合覆盖。'),
('种子与统计单位','三个seed不等于三个新受试者；10份受约束随机边界不是10个独立人。subject7四录制Bootstrap仅描述该面板条件不确定性。'),
('MAE 的物理单位','ROMSL共同B属性为坐标单位每秒。骨架按米解释时是名义m/s；绝对单位缺采集方计量材料，不能声称已经物理标定。'),
('完成与验证','启动、某个status=completed、合成smoke或测试通过数不能代替真实任务结果。失败证据保留；只有启动记录的旧Reset不能填入新Stage1结论。')]:h(t,2);p(b)

# Source bibliography: compact numbered paths, exact files verified above.
for n in range(0,len(source_paths),25):
    page();h('来源索引'+('  续' if n else ''))
    if n==0:p('相对路径均以 D:/lab/SEER 为根。编号对应正文节点；按文内终态更新解释旧文件名。完整文件清单与SHA256见随附source-inventory.json，不将测试生成的报告当正式训练结果。')
    for s in source_paths[n:n+25]:
        a=p(f'[{sources[s]}]  {s}');a.paragraph_format.space_after=Pt(7)
        for r in a.runs:r.font.size=Pt(9)
doc.core_properties.title='SEER 研究方案与实验演进复盘';doc.core_properties.subject='问题驱动的方案与实验演进';doc.core_properties.author=''
doc.save(OUT/'SEER研究方案与实验演进复盘.docx')

# A standalone offline interactive tree. All detailed evidence stays local; no network dependencies.
branches=[]
for n,g in enumerate(groups):
    nodes=[]
    for i in g['items']:
        detail=''.join(f'<dt>{label}</dt><dd>{html.escape(i[key])}</dd>' for label,key in [('问题','problem'),('尝试','attempt'),('结果','result'),('后续','next')])
        refs=' '.join(f'<a href="{(ROOT/s).as_uri()}" title="{html.escape(s)}">[{sources[s]}]</a>' for s in i['sources'])
        nodes.append(f'<details class="node" id="{i["id"]}"><summary><b>{i["id"]}</b> {html.escape(i["title"])}</summary><dl>{detail}</dl><p class="refs">{html.escape(i["status"])} · {refs}</p></details>')
    branches.append(f'<section id="branch{n}" style="--branch:{colors[n]}"><h2><span>{n+1:02}</span> {g["title"]}</h2><p class="period">{g["period"]}</p><p>{g["conclusion"]}</p><div class="tree">'+''.join(nodes)+'</div></section>')
head='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SEER 研究演进思维导图</title><style>
*{box-sizing:border-box}body{margin:0;background:#f6f8fb;color:#213448;font:16px/1.7 "Microsoft YaHei",sans-serif}header,main,footer{max-width:1500px;margin:auto;padding:30px}header{padding-top:50px}h1{font-size:36px;margin:0 0 12px}header p{max-width:1000px;color:#53677b}.root{background:#213d57;color:white;padding:22px 28px;border-radius:16px;font-size:23px;font-weight:bold}.branches{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;margin-top:36px}section{background:white;padding:24px;border-radius:12px;border-top:5px solid var(--branch)}h2{font-size:21px;line-height:1.5;margin:0}h2 span{color:var(--branch)}.period,.refs{font-size:12px;color:#60768b}.tree{border-left:2px solid #d6e0ea;margin:24px 0 0 8px;padding-left:20px}.node{position:relative;margin:0 0 12px;background:#f6f8fb;border-radius:6px}.node:before{content:'';position:absolute;top:22px;left:-22px;width:21px;height:2px;background:#d6e0ea}summary{cursor:pointer;min-height:46px;padding:11px 13px;font-size:14px}summary b{color:var(--branch);margin-right:5px}dl{margin:0;padding:0 15px 10px}dt{font-weight:bold;font-size:13px;margin-top:10px;color:var(--branch)}dd{margin:3px 0 0;font-size:14px}.refs{padding:0 15px 13px;overflow-wrap:anywhere}a{color:#346b9c}button{font:inherit;padding:8px 16px;border:1px solid #b6c8da;border-radius:7px;background:white;color:#294d6c;cursor:pointer;margin-right:8px}.tools{margin:20px 0}footer{font-size:13px;color:#65778a}@media(max-width:1100px){.branches{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:700px){.branches{grid-template-columns:1fr}header,main,footer{padding:20px}h1{font-size:27px}}@media print{.branches{display:block}section{break-before:page}.tools{display:none}.node{break-inside:avoid}}</style>
<header><h1>SEER 研究演进思维导图</h1><p>截至2026年11月7日 · 每个节点展开后可查看问题、尝试、结果、后续与本地证据。各主题包含并行分支，不表示所有模块已经组合验证。</p><div class="tools"><button id="expand">展开全部节点</button><button id="collapse">收起全部节点</button><a href="SEER研究路线思维导图.png">查看总览大图</a> · <a href="SEER研究方案与实验演进复盘.docx">Word文档</a></div></header><main><div class="root">持续雷达流 → 可解释事件与状态 → 时间有序描述与历史查询</div><div class="branches">'''
tail='''</div></main><footer>结果状态独立标注。source-inventory.json记录审阅范围与文件SHA256；research-data.json保留全部节点及来源映射。旧“已启动”不代表终态完成；合成验证不代表真实数据收益。来源链接需本机文件访问权限。</footer><script>document.getElementById('expand').onclick=()=>document.querySelectorAll('.node').forEach(x=>x.open=true);document.getElementById('collapse').onclick=()=>document.querySelectorAll('.node').forEach(x=>x.open=false);</script></html>'''
(OUT/'SEER研究演进思维导图.html').write_text(head+''.join(branches)+tail,encoding='utf-8')

# OPML is importable by many mind-map editors and keeps all details editable.
import xml.etree.ElementTree as ET
opml=ET.Element('opml',version='2.0');ET.SubElement(opml,'head');body=ET.SubElement(opml,'body');rt=ET.SubElement(body,'outline',text='SEER研究方案与实验演进')
for g in groups:
    a=ET.SubElement(rt,'outline',text=g['title'])
    for i in g['items']:
        b=ET.SubElement(a,'outline',text=i['id']+' '+i['title'])
        for lab,key in [('问题','problem'),('尝试','attempt'),('结果','result'),('后续','next')]:ET.SubElement(b,'outline',text=lab+'：'+i[key])
        ET.SubElement(b,'outline',text='来源：'+'；'.join(i['sources']))
ET.ElementTree(opml).write(OUT/'SEER研究演进思维导图.opml',encoding='utf-8',xml_declaration=True)
print(json.dumps(dict(attempts=len(all_items),groups=len(groups),primary_sources=len(sources),inventory=len(inventory),docx=str(OUT/'SEER研究方案与实验演进复盘.docx')),ensure_ascii=False))

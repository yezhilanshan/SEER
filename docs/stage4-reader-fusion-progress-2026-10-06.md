# 新文字QA reader接回雷达融合实验进度

执行依据：用户批准本对话下一步计划及 docs/stage4-temporal-qa-2026-10-05.md。

Ruling: 仓库没有Git，保留本地修改并用新隔离服务器代码目录、源归档SHA及本ledger记录版本；不运行依赖Git的worktree/commit脚本，不删除唯一证据。

接口核查：旧temporal入口将原EventFormer adapter同时用于单/多事件；新text decoder导出为独立rank8全24层LoRA。必须加载新模型实例并分别路由训练与生成，单事件保留旧reader和旧QA变换。

锁定：none/order/time各500实际AdamW步，seed42，同一初始化/批次/lr3e-4，冻结新QA reader、旧QA/frontend/EventFormer/P/压缩器；fit监督196事件，dev99事件，不按dev选择权重；最终500步报告全部组，formal_go=false。

Ruling: 每步三条CE逐题反向累积平均后更新一次，与原平均CE目标相同，避免同时保留三个完整LLM反向图造成11GB GPU显存风险。

任务1：新路由及逐题累积四个测试RED确认缺功能，GREEN相关18 passed/3 skipped；native封存加载测试因本地transformers/peft缺失留服务器执行。
任务2：检索指标测试RED后实现，已接入original-reader零步比较、共享reader路由、来源及检索快照。相关完整GREEN待最新运行读回。
全项目测试：interview环境pytest在test_parrad_convert.py收集失败，缺h5py；未安装，服务器完整套件待运行。初始Python沙箱无法加载依赖，提权授权运行成功。
任务3：独立review、服务器原生测试、源归档和独占后台启动，待完成。

独立只读整体验收：无Critical，一项Important（读回未核对题目全集/重复/交换证据）、一项Minor（未重算开发门槛）。门槛问题按可能导致错误通过声明的影响提升Important，两项统一一轮RED→GREEN修复，不再复审。新增8个拒绝测试已RED，修复接入readback并等待GREEN读回。

审查未判事项裁定：原生兼容/完整套件、GPU真实预算、冻结及生成/检索数值、部署归档均需真实执行后报告。failed路径只归档证据并明确失败，不视为部分结果完整验证；如失败需分析实际阶段，不能宣称实验完成。

初始源归档9e72fde1ab0b299ed7226c24ef2d5cf00d7dbd9425b7b9b973a9fa8e83100dce，173成员，已校验安全解压到服务器新stage4-temporal-qa-code-v3。仅pytest运行，训练未启动。服务器完整测试收集发现源码包缺scripts.make_synthetic_parrad_data，补入scripts，不改实验目标/预算。已退出测试进程后补修隔离目录，再跑完整套件。

一次修复GREEN：tests/test_qa_evaluation_readback.py/test_qa_reader_routing.py/test_temporal_qa.py/test_qa_text_decoder.py 共27 passed、3 skipped（本地native依赖缺失）。缺题、重复、错字段/来源/reader、交换证据及开发门槛均有拒绝测试；新生成行增加cohort，readback验证准确全集及来源，再重算门槛。无遗留审查事项。
最终ready源归档08c7c6da90e24fb1472fdc1c126672bd8eb445cd76d47a8a016d848de8661ee8，368675字节、182成员，等待服务器验证及完整测试。

服务器ready完整套件：7 failed、488 passed、6 warnings，59.37s，实际exit_code=1。五项为源码包缺旧验证脚本verify-projection-budget-v1.py（4测试）/verify-text-decoder.py（1测试）；补入测试支持文件，不改已有脚本。另两项Gemma3Adapter测试触发Gemma3RotaryEmbedding缺None_inv_freq，将以原代码独立复现。
Ruling: 本轮使用Qwen2.5且要求保护原EventFormer代码/cache来源SHA，因此不改Gemma3分支或升级共享依赖来处理无关后端兼容性；记录全套真实失败及原代码复现证据，Qwen相关native套件必须通过后才启动。

原Gemma失败复现：使用stage4-text-decoder-code-v1原src，在同一服务器环境运行两项Gemma失败用例，2 failed in 3.00s；event_former.py/config.py/train.py及Gemma测试SHA与本轮隔离源码完全一致。确认并非新QA路由回归，完整错误日志留服务器original-gemma-reproduction.log。
最终完整源归档a3070ea40da32970e7d5a7b32b124be512aa402ab02be7071f7b51f49eadbd4f，372038字节、184成员，服务器SHA/成员路径/全部manifest文件验证通过。完整套件及本轮四文件native套件串行运行PID8495，结果写native-complete-result.json，训练仍未启动。

最终服务器本轮native四文件：30 passed、5 warnings，6.15s，无跳过，focused_exit=0。全部原生reader加载/梯度及拒绝检查通过。完整套件full_exit=1，启动前严格确认失败集合只有两项已复现的原Gemma测试；旧脚本漏带导致的五项失败已消除。启动前再核验全部源manifest SHA，授权独占launcher仅运行一次。

2026-10-06 00:35:08.564647北京时间：后台worker PID8599已独占启动，输出stage4_temporal_qa_v3，代码stage4-temporal-qa-code-v3。三组none/order/time各500；流程先加载/文字复现/零步基线，再自动训练和读回。SSH会话不持有worker生命周期。当前尚未读取实际训练步数，不能宣称三组开始更新或完成。

00:39:13服务器状态running/text-preflight：fit1014已完成，11项锁定95%门槛全true，dev文字复现已进展300/510；三组实际更新均0。运行worker8599、训练入口8600，GPU显存约2491MiB（此时是生成复现阶段，非训练峰值）。新reader封存SHA/base/storage检验、三组零残差等价检验均已先于门槛完成。

00:40:21状态running/baseline，fit1014/1014、dev510/510正确，与封存文字reader成绩精确复现。新reader零步雷达评估300/807进行中；新增文字训练次数为0，等待固定零步基线完成自动进入融合优化。

00:43:30：status running/baseline，新reader零步雷达700/807进行中，无failed/blocked记录，尚未建立优化器。长生成任务使基线比文字题耗时增加；保持后台单worker，未改预算、提示、权重或评估任务。

00:46:33零步基线：dev-before807已保存。独立比较旧short-context dev-after，297条single的query/reference/generated完全一致；single macro80.4678%、presence84.3434%、pair74.7475%。新reader双/四存在性65.3061%/66.6667%，位置平均均0%，四完整顺序0/24；这是融合训练前基线，不能归为本轮训练退步。旧reader多事件510已接近完成，无记忆对照200/807；三组仍按计划等待基线结束自动训练。

00:49:41：v3 none实际完成74步（训练日志74行），第75步在梯度检查失败，未执行第75次optimizer.step；order/time为0。状态failed/train-evaluate/none，reason=nonfinite temporal gradient。无100步检查点；失败证据已自动归档stage4-temporal-qa-v3-results.tar.gz，316056字节，SHA a488d971e6528a1784e628ab5dd19ac964d22593ed97730493566d3342722753。日志末条step74 meanCE0.95424982、norm4.67190075。
已纠正最初根据日志尾部误判断“第一步无有效更新”的用户说明。训练与读回进程退出后进行数值复现：精确重放74步并比较每步loss/norm，再仅反向探测第75批在256/64/16/1的prefix梯度与六参数；不发布诊断权重或重启v3，不修改预算/损失/原模型。

数值诊断v1完成：74步全部loss/norm逐项精确复现，40.17s，诊断参数不用于发布。第75批meanCE0.883477807三种scale相同；256时仅指定第二事件任务的reader prefix梯度非有限，六融合参数均坏；64/16/1时全部有限，unscaled norm约123.9/123.5/123.7。确认放大损失后的FP16反向溢出，非数据或损失变坏。诊断脚本SHA eece90905b2dd1d2fb33ec05eaf15ebd3b4673b9b7f67b79e2a5c303fdd5ba41f，服务器诊断PID8836已退出。
Ruling: v4保留seed/data/init/lr/loss/AdamW/clip1/initialscale256及500真实步，只将非有限反向改为最多12次减半scale重算同一批，失败尝试不执行optimizer.step、不推进采样、恢复RNG；记录全部重试。若仍非有限则停，不能跳批凑预算。三组相同控制策略，v3失败74步与诊断74步单独保留，不算v4三组预算或成果。
Ruling: 推理数学、模型/来源/任务/初始化均未改，v4复用v3已完整产生的文字与零步基线；对推理函数AST、其他依赖SHA、冻结reader摘要、输入/task/batch SHA、完整题集/来源/评分严格验证，并实际重算原P检索分数完全一致。仅避免重复生成已完成的基线，不能复用训练权重/优化器或隐藏失败。改变任何绑定条件时拒绝复用。
AMP两个实际半精度溢出测试RED→GREEN；基线复用推理代码及源变更拒绝两个测试RED→GREEN；恢复相关本地31 passed/3 skipped。v4需原生26步（75..100）硬例验证和完整套件后再独占启动。

v4源码归档23ff30e46b45a65dce328114cc82f2ee5ee97b48e5cc97eaf7722c0ce7a91abb，381167字节、191成员，服务器新隔离stage4-temporal-qa-code-v4验证全部SHA后启动完整/相关native测试PID9119。v3/诊断训练进程8599/8600/8836均已退出，原v3记录保持。已启动独立native-numerical-smoke：验证全部基线绑定并实际重算检索后，从诊断精确step74状态仅验证第75..100步数值控制，诊断优化26次与正式预算隔离；正式v4尚未启动。

恢复验证完成：服务器v4相关34 passed、5 warnings、6.23s，无跳过；完整套件仅保留已复现原Gemma两项失败，启动前严格核对失败集合。native数值验证PID9210：从精确74步状态成功执行75..100共26次更新，六个AdamW状态到100；所有冻结权重、原P检索向量/分数和基线绑定通过。第100步meanCE0.54584262、norm6.45181322、scale32。诊断权重未用于v4正式初始化。
启动前再核验191成员全部源SHA、native测试退出码与数值验证结果，然后独占启动stage4_temporal_qa_v4。预算从共同原始初始化各500真实步；复用基线仅限已验证推理部分，不复用v3/诊断学习权重。

2026-10-06 01:17:08.474157北京时间：恢复v4唯一worker PID9288后台启动。当前回读status=complete，三组training.jsonl各500条，实际训练入口duration867.1089秒（不含旧基线生成与独立核验），训练已结束，未再启动worker。

11:03服务器回读发现自动核验脚本失败，训练/评估均已完成；原因是validate_rows将task.label（来源事件类别）当成presence问题询问类别。例如walking toward the bed的问题，负例来源walking toward the chair，其task.label为12，而实际询问类别为11。交换后正确Yes被误判。修复仅使用重新构建pairs.activity核查原证据及实际交换证据，不放宽来源/全集/评分/冻结检查，不改训练源码或保存预测。

本地11项回归通过；独立服务器stage4-v4-readback-repair-v1文件SHA核对后，9项回归通过。使用隔离修复模块执行原v4源码和数据SHA检查、三组100/250/500 optimizer状态、500最终导出、1500实际步数/重试记录、全部题集独立评分、单事件297完全一致、原P检索完整向量/分数/排序一致、开发门槛重算，exit0。independent-readback.json状态complete、actual_updates1500、original_retrieval_exact=true。此修复不改模型行为，不需要重训。

v4完整原始JSON/JSONL/log归档35成员、923866字节，SHA70ccf8000d2cec2960e00a03c3dbd2851932b76e220ddef0205ccef4af0537e5。v3失败归档独立保留16成员、316056字节、SHAa488d971e6528a1784e628ab5dd19ac964d22593ed97730493566d3342722753；本地下载后按SHA/成员路径核验再解压。

本地归档下载完成，v3/v4 SHA与服务器一致，所有路径/成员类型安全核查后解压；从全部预测重新计算开发指标/门槛、原检索逐类指标和297单事件逐条一致，summarize-reader-fusion-v4.py exit0。完整结果见stage4-reader-fusion-results-2026-10-06.md，未追加训练。

# 执行记录：固定训练预算诊断

方案：`docs/stage3-transition-aware-budget-ablation-2026-10-04.md`。用户已授权“请你开始吧”。

- Task 1: 下载held03原始结果并补齐第三折20步。当前SSH首次重试在banner阶段超时，未认证、未启动任何新worker。本地Tailscale新读回为NoState/starting；已请求用户恢复在线状态，继续独立本地工作。
- Task 2: 新增显式最多100步预算、严格检查和恢复保护；保留默认20步。先写失败测试，再实现。
- Task 3: 本地相关测试、完整可用回归、一次只读审查；制作独立部署包。部署前核验缓存、生产源码/Stage2哈希、GPU与无重复worker。
- Task 4: 第三折完成后，三折固定0/5/10/20/50/100小实验；监控、下载、汇总。仅固定预算对比，不自动架构修改或正式扩训。
- 执行适配：工作区无Git元数据，沿用隔离部署、不可覆盖输出和SHA记录；不能使用Git提交/工作树脚本，保留本地记录与归档。
- 界面核对：run的实际配置必须与warmup/condition signature一致，否则会把20步预算记录成100步或允许错误恢复。部署shell必须显式传新参数，不以修改模块全局SETTINGS代替。

## 当前验证

- 新预算测试16项先失败（缺失steps/budget_limit API），实现后相关30项全部通过，71.60秒。真实合成优化比较20步与显式21步流程，前20步P逐张量完全相等，实际AdamW步数21、teacher与原P不变、恢复预算不匹配保持manifest字节。
- 可用完整本地回归：420 passed、2 skipped，187.37秒；延续已有h5py缺失排除test_parrad_convert.py，不安装依赖；已有librosa/pkg_resources弃用警告一项。
- 新core SHA256：`4d790148063022e707252878b9bfdc0ca6321a9dc21b3a5710e11915ec9d1fd8`。默认SETTINGS、目标、优化器、每步头拟合和评估器不变，只参数化有限预算。扩展预算进入manifest和warm/condition身份；20步默认设置字段保持兼容。
- 用户回复网络恢复后，SSH确实认证成功，服务器date返回2026-10-04T01:13:56+08:00；随后会话中断，未获得GPU、结果归档SHA或worker状态。后续SCP一次和SSH两次连接跳板机超时，Tailscale两次短ping无回复。尚无归档下载或新worker启动，不能声称训练在跑。
- 新部署shell固定顺序：验证02/03旧缓存、建04缓存、04默认20步，然后02/03/04各100步；新只读结果验证器核对旧20步前缀、warmup/baseline、teacher/控制、实际优化器步数、原语义门槛与所有refit。
- 最终一次只读审查已派发budget_review，部署等待审查和服务器连接恢复。

## 中断恢复与审查修复

- 对话中断后继续同一次初始审查，没有对修复重复派发审查。审查确认两个P2：缺失非选中20步checkpoint可能绕过prefix核验；文件名字典序将100排在11之前，合法第100步头失败被误判。4项真实文件/实际AdamW读回测试中3项先失败、1项成功路径通过；修复后4 passed，11.28秒。按实际已完成评估要求所有milestone及checkpoint存在；按整数步数排列refit。默认训练目标和主core SHA不变。
- 审查未裁定的服务器缓存/GPU/worker实时状态由实际SSH核验：10月4日09:22:26服务器返回GPU0 RTX2080Ti 270/11264MiB，无Projection worker；旧生产runner/pilot与Stage2 SHA再次匹配。真实数据20步复现/收益由尚未启动的后续实验决定，不据合成数据宣称真实收益；部署包完整性由tar列表和本地/服务器SHA核验。
- held03原始结果归档已下载并核验SHA`0841c2020669273afbf720a661c610c95538fd1f11f41934e6d10ba0a478ddb5`；成员路径检查后解压新目录，原锁定选择与语义差值/头收敛重算通过。relation1/10检索准确率及动作F1最差delta均0；分组发现subject02/age7 AUROC约-1.47pp，AP约-1.04pp，宏观增益不能代表每个组改善。更新原报告和理解，不修改预注册预算。
- 最终修复包v2 SHA`e3ce99cc42c507abb83a0d58aa9f5666dc69f5d0ef41a4a04d228be7b242c499`，上传已完成且服务器哈希匹配；v1上传保留但不训练。cp新隔离目录及解压v2命令已发出，随后SSH再次reset；不能假定完整部署。新SSH正在检查新目录/模块SHA、脚本语法和全部Projection相关测试。真实训练仍未启动。
- 最终本地可用完整回归在跑；服务器检查、最终回归均通过后才启动新pipeline。网络中断时先看目录/PID/日志，禁止重复worker。

## 最新停点（10月4日09:38）

- 最终可用完整本地回归已完成：**424 passed、2 skipped、1既有弃用warning，382.04秒**，命令`PYTHONPATH=src OMP_NUM_THREADS=1 interview/python -m pytest --ignore=tests/test_parrad_convert.py -q --tb=short`。没有安装依赖或修改原算法。
- 最终v2包成员已核对：主core、预算/读回测试两文件、固定方案、launcher、验证器、汇总器均在包内。服务器收到的归档SHA与本地一致。原生产runner/pilot和Stage2 SHA匹配。
- 服务器隔离cp+解压命令之后会话reset；其后的新SSH还未到认证就超时，服务器目录/模块/脚本和相关测试因此**尚未核实完成**。不能从上传成功推断已部署成功。
- 09:37–09:38当前Tailscale明确显示jump100.83.206.93 **offline,lastseen1m**；netcheck本地UDP不可用，但DERP可达。主要外部阻塞是跳板离线，不能用换密码或改训练参数解决。
- **未启动第三折或100步实验，尚无新worker/PID。** 实验代码与方案、本地测试、一次初始审查及修复、held03原始下载均已完成。待恢复连接先检查隔离目录是否存在/完整，继续服务器测试，确认没有重复worker及输出不存在，再nohup启动`.audit-staging/run-projection-budget-v1.sh cuda:0`；不重新复制覆盖历史目录，不重复启动。

## 用户恢复连接并要求上传（10月4日18:15起）

- 新SSH认证成功。服务器18:15:41读回最终v2归档SHA与本地完全匹配，原上传有效；新隔离目录确实不存在，前次cp命令没有成功，不能据之前发出命令声称部署完成。
- 在确认目录不存在后，从已验证replication隔离代码复制新`projection-budget-code-v1`，解压v2；命令退出0。七个包内文件逐项SHA与当前本地一致（主core、两个测试、固定方案、launcher、验证器、汇总器），bash -n退出0。
- 当前已运行服务器全部Projection相关测试，输出重定向到新隔离目录`.audit-staging/server-projection-budget-tests.log`；尚待读回完整结果。用户此前启动小实验授权仍有效，上传后继续必要验证及固定pipeline，不重复询问许可。

- 服务器全部Projection相关测试 **70 passed、23.04秒，exit0**。18:20:22新读回GPU0 RTX2080Ti仍270/11264MiB、无Projection worker；原生产runner/pilot和Stage2 SHA匹配，第三折/预算输出及日志目录全都不存在。
- 已nohup启动固定pipeline，launcher **PID19086**，进程读回确认`bash .audit-staging/run-projection-budget-v1.sh cuda:0`。代码在新隔离目录，先04缓存/20步，再02/03/04各100步；原模型/旧实验不覆盖。新日志目录服务器`stage3_transition_sequence_v2/transition-aware-semantic-primary-budget-v1.logs`。
- 刚启动，当前正读回status/cache日志及实际worker。尚无第三折/100步完成或收益结论，不能将启动报告写成训练已完成。

- 后续实测pipeline于**2026-10-04 18:20:59 +0800**启动；launcher19086、cache worker19089均活跃。旧02/03 fit/dev缓存验证通过后开始held04新缓存，fit总436窗口，最近读回已225/436，GPU0占1511MiB且在计算。当前是冻结Stage2数据准备，不是尚有新Projection成绩；数据准备结束后固定脚本自动执行第三折20步及三折100步，检查点20/50/100用于预算比较。
- 当前上传/隔离部署/服务器测试/后台启动已确认完成。后续检查首先读`transition-aware-semantic-primary-budget-v1.logs/status.log`，根据现有PID与manifest确认状态；不要因监控断线重复启动。训练是否完成及效果必须重新读回，不从启动推断。

## 用户授权定时检查（10月4日）

- 18:24:56实时服务器读回：pipeline仍运行，held04 fit缓存已完成，dev214窗口已65；尚未开始第三折P训练。结合以往20步训练约156–189秒及本次三折100步预算，给出粗略结束窗口18:45–19:00，作为耗时估计而非保证。
- 用户明确请求设置定时任务检查结果。已通过Codex原生automation_update创建当前对话heartbeat **SEER训练结果检查**，ID **seer**，ACTIVE，每10分钟。正常/未变化时安静，仅完成、失败或需人工连接/认证时通知；完成核验和报告后暂停，同一连接问题不反复通知。
- 定时任务只延续当前固定实验的监控与结果核验：不重复worker，不改变预算/目标，不存密码；读真实日志/PID/manifest，核验checkpoint20/50/100、teacher/检索门槛和前缀复现，下载SHA核验原始结果后按固定系数/折/年龄/subject报告。认证不能安全取得或持续断网时报告并暂停，后台训练保留。

## 定时检查结果：实验完成并报告

- 此次heartbeat在18:48:50触发；新SSH读回完整status，pipeline **18:20:59–18:47:05正常完成**（1566秒），launcher19086已退出。缓存18:25:53结束；第三折20步18:29:06结束；三折100步分别18:34:41/18:40:44/18:47:05结束。
- 服务器逐检查点核验通过`ALL_CHECKPOINT_INTEGRITY_OK`：四个新run的来源/Stage2/代码/方案SHA和禁读标志正确；全部teacher与控制P冻结、实际AdamW步数20/100、终态ready、选中权重SHA匹配；640次每步头refit（每次3头）全部按原规则数值收敛。三折100步的baseline/warmup/20步P及score、语义指标精确复现原20步对照。原生产runner/pilot最终SHA与启动前一致。
- 原始JSON/log归档下载成功，SHA **9325bce5343b505814daae913778f6499f6c7bf6908f5fa21557375112cd9d60**；成员路径核查后提取新目录`.audit-staging/results-projection-budget-v1/`。`analyze-projection-budget-results-v1.py`从保存的预测重算每组binary metrics、原主AUROC/AP聚合及资格门槛，全部断言通过；没有新拟合/优化器步。
- 第三折20步原AUROC.5859907519 -> r1.5839926455/r10.5842572839，均锁定step0。50步r1.5889213421/r10.5891091809转为小幅正收益；100步.5831385366/.5835024794又回落。fit head objective持续下降，说明继续降低fit目标不能保证dev改善。
- 50步相对20步三折全正，r1/r10平均额外AUROC **+.7214/+.6690pp**，均满足预定探索标准；相对原P macro AUROC **+1.5960/+1.5981pp**，AP **+1.5411/+1.5557pp**。100步虽然macro更高，但第三折相对20步负，故均不通过稳定预算标准；r10第一折100步还因stream动作probe400次未收敛、F1缺失而不合格，原规则选50。不是训练头失败或已证明检索退化。
- 50步全部语义门槛通过，但非所有指标不变：subject02 commit文本检索事件准确率及new_event/old_text交叉版本，部分配置最差delta **-1.7857pp**；动作F1最差 **-.7948pp**，均在2pp容差内。不能声称原检索完全无损。
- 全部固定预算和分组绝对AUROC/AP、delta、probe诊断见`.audit-staging/results-projection-budget-v1/summary.json`；可审阅报告`docs/stage3-transition-aware-budget-results-2026-10-04.md`。只提出50步有限预算作为后续研究基线、优先查局部泛化和检索难例；不加步、不解冻、不启动新实验、不宣布正式Go。
- heartbeat文件实际保存间隔为20分钟（与先前口头10分钟不同；不推断变更来源）。结果核验和报告完成后，原生automation_update将**seer置PAUSED**，保存原其余字段；工具及automation.toml均已确认PAUSED，后续不会继续重复检查。未修改密码或写入认证文件。

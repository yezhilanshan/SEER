# 单事件雷达语义适配v2：文字门槛改为诊断

用户明确指示：“不纠结这个了，继续雷达的训练吧，下一步应该干什么”。本次授权覆盖继续原定两组雷达训练，并解除文字教师门槛阻塞。执行采用executing-plans技能；仓库无Git，保留所有旧证据，隔离服务器源码和独占新输出，不做worktree/commit或删除。

Ruling: 新配置text_preflight_policy=diagnostic-only，默认配置仍required。fit文字题未过95%时保留passed=false及原始错例，proceed=true只表示用户授权继续雷达实验，不伪造教师通过。蒸馏教师已知会混淆坐在床上/坐到床上，可能拖累distill；因此CE与distill均保留、分别报告，不自动据蒸馏结果替换原模型。损失与教师算法不改，避免混入新的消融因素。

流程为重新记录文字诊断、共同零步雷达fit/dev、无记忆dev、旧单事件297复现及原检索快照；随后ce答案监督500次实际AdamW更新、distill答案监督+0.5*T²KL/T2再500次更新。每步fit正例Yes、相近反例No、正例动作识别三题，14类平衡轮转；共同seed42身份残差初始化、lr1e-4，全部参数与v1一致。reader、EventFormer、原P及旧QA/压缩器冻结，仅训练新896→256→896的EventQAAdapter。不加时间编码、不训练多事件；不能从本轮结论推断此前时间编码实验的因果。

每组固定第500步报告全部fit588/dev297及dev交换198题，100/250/500仅读回；保留严格动作评分，也分别解释冠词格式与真正语义错例。工程门槛宏动作≥85%、平衡判断≥90%、成对≥80%，开发门槛继续保留，不改分数、不按dev选权重。最终判断以相对共同新reader零步的进步、逐动作混淆、交换证据依赖和原检索精确保留为依据。当前teacher文字门槛未达，不做正式Go。

原始任务/采样来源及SHA、原保护权重SHA、全量原始预测、实际500步优化状态、所有检查点、原检索向量/分数/排序前后一致继续核对。独立核验允许诊断策略的完整实验，但必须保留teacher passed=false、policy和proceed；默认required配置不允许绕过失败门槛。失败或核验异常继续保留完整JSON/log归档。

服务器隔离源码：`/home/xibeidaxue/SEER/.audit-staging/stage4-radar-alignment-code-v2`。

服务器新输出：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage4_radar_alignment_v2`，同名.logs/worker.log，结果归档stage4-radar-alignment-v2-results.tar.gz。旧v1保留。

本次仅用既有subject02 TRAIN开发录制fit196/dev99，不能称独立未见受试者测试。没有自动启用新的定时任务。

## 执行记录

任务1：三个政策/真实训练控制流用例先RED（缺决策函数、required结果缺proceed、诊断策略仍停止），再GREEN。本地相关18 passed/1 skipped；跳过为本地缺transformers/peft的原生Qwen测试，留服务器执行。修改范围仅文字预检策略、状态来源记录、相应核验与配置化独占launcher/worker；原训练损失/优化/采样/生成代码未改。

任务2：服务器部署、原生回归和独占启动待完成，不宣称已进入实际更新。

本地含QA reader路由回归22 passed/2 skipped（本地缺原生依赖）。源归档df2c6676f0173e8a34d1cd08d5789307f99709a9035e1237f462dc6a51f0d7cd、391687字节198成员，已上传；服务器核验SHA、安全成员类型/路径及全部manifest一致。首次native测试从主仓库cwd启动，因tests跨文件导入路径缺隔离源码根导致收集失败，未启动训练；修正测试PYTHONPATH为隔离src+隔离代码根并在隔离cwd运行后24 passed/1 warning，无跳过，14.23s，原生Qwen/PEFT反向与冻结测试通过。仅修正执行环境，未改源包。

一次技能要求的独立只读审查完成，无Critical/Important：确认policy/worker/核验一致、旧证据保留、实际500步与冻结/检索检查未改变；与v1源归档比较，训练损失/优化/采样/生成逻辑未变。不再次审查旧框架或增加文字QA训练。

2026-10-06 13:25:26.929044北京时间，唯一后台worker PID18301启动成功，LAUNCH_RC=0。启动前再次核对全部源码manifest、固定v2配置、服务器相关24通过日志、无其他alignment worker。此时仅确认后台启动，尚未确认正式优化步数。文字诊断按diagnostic-only继续，所有旧v1权重和结果保留。

启动后读回running/baseline，teacher identify=false、Yes/No=true、passed=false、policy=diagnostic-only、proceed=true，确认真实流程已越过旧阻塞点。共同新reader原始雷达零步：fit宏动作14.2460%、正反平衡47.9592%、配对全对5.1020%；dev宏动作18.9796%、正反平衡46.9697%、配对全对3.0303%。这些是身份残差初始化的接口基线，尚不代表训练后效果；后续两组都相对该基线报告，不把新reader零步与旧QA混成同一模型的退步。

2026-10-06 13:30:43.771073北京时间：status running/training/ce，状态快照ce150/distill0；独立读取training.jsonl为155条连续实际更新，最后step155 loss0.4250、scale256、backoffs0，worker18301存活。原单事件复现及原检索baseline已保存。本轮任务2部署/回归/独占启动已完成，任务3实际训练进行中，最终500步效果及训练后检索保留尚待完成后报告。

之后只读尝试核对100步检查点实际AdamW状态时SSH响应超时，未收到该命令结果，因此不宣称此次已独立读回100步权重。最后确认仍为13:30:43正式ce155步；后台worker detached，SSH断线不据此判断训练停止，也不重启任务。完整检查点读回由终止后的独立worker核验完成，后续需要取得结果。

用户要求检查结果后只读重连：两组各实际500步全部完成，主流程13:41:47.090245北京时间结束，duration978.644秒。worker18301已退出，独立核验verification_status=passed/complete_checks_performed=true，全部100/250/500实际AdamW状态通过，0数值回退。旧单事件297输出与原检索全快照精确一致。任务3完成；任务4结果核验/报告完成。

完整JSON/log归档628520字节、28成员、SHA43a192d931bdd199d192202f7aef86289d7c12c302d9462634d8f1dd4061ac1c，服务器/本地一致；安全路径检查后解压至新radar-alignment-v2-results-readback-2026-10-06目录。本地重新绑定任务/来源并复算4233条新reader预测及297条旧QA、连续500步/采样/源码政策/结果/检索排名，exit0。实时重新核对4个保护SHA全部一致。完整报告stage4-radar-semantic-alignment-v2-results-2026-10-06.md。

dev宏动作/正反平衡/配对全对：零步18.98/46.97/3.03%，CE76.14/67.17/46.46%，蒸馏47.53/66.16/43.43%，旧QA参考80.47/84.34/74.75%。两组开发门槛三项均失败；蒸馏更差不是冠词别名造成，主要真实动作混淆。CE的No54.55%，走进房间0/10为突出弱项。本次不自动增加预算/训练新模型，formal_go=false。

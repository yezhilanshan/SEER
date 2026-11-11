# Execution ledger: docs/stage4-decoder-repair-2026-10-05.md

用户在解释文字测试含义后明确要求“先解决文字问答问题”，授权执行建议的独立QA LoRA方案。本轮先完成文字训练和验证，不自动开启位置／时间编码三组训练。

Ruling: 仓库没有Git元数据，因此沿用版本化服务器隔离代码目录、SHA清单和本地执行记录替代worktree/commit脚本，保留所有既有结果；代价是没有Git提交可用于回滚，依赖归档和来源SHA。

Pre-flight: 新reader复用Qwen2Adapter的caption_logits与现有inputs_embeds生成接口；原qa_loss拒绝可训练teacher，因此新增仅用于独立LoRA的答案CE，不改变原函数。完整fit门槛沿用oracle_gate，dev仅报告不选模板/检查点。固定批次复用temporal_batches，正反监督各一条且内容任务第三条；逐题反向累积三个CE平均，避免一次保留三份LLM计算图。

Task 1: 新reader、答案CE、批次及adapter checkpoint：RED为缺失模块4 failed；实现后本地2 passed、2 skipped（缺原生依赖）。服务器首轮需从隔离目录运行pytest，避免原checkout测试包路径冲突；实际Qwen隔离/两步LoRA更新/冻结后前缀梯度均通过。新增损坏optimizer参数名称测试原生RED为DID NOT RAISE；已补真实LoRA参数名和结构匹配检查，待GREEN验证。
Task 2: 独立500步训练入口、保护摘要、前后完整fit/dev真实生成、最终检查点读回。
Task 3: 有限完整代码审查、服务器原生测试、SHA核验、独占后台启动。
Task 4: 实际步数与原始回答核验、本地归档及结果报告。只有全部95%fit门槛通过才能称本轮文字任务修复成功；仍为TRAIN开发任务。

- Task 1/2代码及原生测试完成：服务器30 passed、4条torch/peft兼容性warning，无跳过。实际tiny Qwen测试包含LoRA学习、冻结后前缀梯度、原模型权重保护、导出及损坏参数名称拒绝。修复模块SHA b26634d1be524b9452f7df2971c215b6bcdc20415496bc468757ea1fbe6547e4。
- 初始源归档353737字节、168成员SHA 0295783e9ae7807178d4fb0620a10c963829037be7b716348e6ab8680862215c，服务器归档、路径类型及167来源文件SHA全部一致。原生RED后只在新隔离目录修复检查点校验，最终来源清单另存final版本。此初始archive不能当作修复后最终代码SHA。
- 独立训练入口仅文字任务。完整fit-before/after、500次实更新、100/250/500检查点、最终导出清零重载及生成一致，仍需服务器真实运行证据。
- Final review: Critical/Important无；Minor为source.json产生前的加载失败不能归档。Ruling: 将该项升为需要修复的失败处理问题，因为用户需要真实失败证据；代价是新增读回--root测试参数及缺失产物标记。RED在临时目录真实复现缺source.json，修复后进行GREEN全相关测试。不再进行第二轮审查。
- GPU原生tiny Qwen FP16、FP32 LoRA、GradScaler256、每次三题累积平均、clip1连续两次实更新通过。正式训练的500步不包含这些独立测试更新。
- Ruling: 归档下载全部JSON/log，LoRA权重和优化器在服务器独立核验，不下载三份优化器状态；代价是本地原始归档不包含权重，需要服务器导出路径加载新reader。没有改变权重检查或训练方案。
- 审查修复后，本地29 passed、2 skipped；服务器31 passed、4条兼容性warning，无跳过。无剩余审查事项。源码最终ready包351169字节SHA 701e122376f024645f6fd48664564ee2f62a1d636782421fe161110e9615dd67，167源文件清单与本地预期一致；初始及中间包均保留。
- 北京时间2026-10-05 19:51:16后台独占启动，launcher PID5756，代码/home/xibeidaxue/SEER/.audit-staging/stage4-text-decoder-code-v1，输出/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage4_text_decoder_v1。只训练独立文字QA LoRA500步；不会启动时序训练。实际阶段以status.json/worker.log/进程为准。
- 结合ps确认真实worker PID5757；先fit-before1014题，再训练新540672参数/96个LoRA张量。最后一次成功读取：实更新132、训练已用102.8秒，保存step100；最后loss0.0002737213、有限梯度，不能据低loss称问答通过。
- 新reader从原始本地Qwen加载，与原EventFormer LoRA起点不同。新reader文字fit零步基线双Yes28/97、No97/97；四Yes25/48、No47/48；完整四顺序35/48。正式效果应与此同reader零步基线比较，不把更换reader起点带来的变化归为训练收益。
- 约20:00监控SSH连接被重置，随后跳板100.83.206.93:22连接超时。已保存后台PID/输出路径；未重启worker或改变预算。断线不代表训练停止，500步及最终回答尚待恢复连接核实。
- 后置只读压力检查脚本已准备：六个固定文字列表，涵盖重复、重排、相近名称，每组全部14个存在性目标及每个位置/完整顺序，共110题。不进入训练或选checkpoint，不改变原95%fit门槛。尚未上传或运行，不能当作结果。
- 连接已恢复并确认训练正常完成：status=complete，actual_updates=500、text_fit_pass=true，主流程893.0738秒，5756/5757均退出。fit1014题及dev510题全部真实生成正确；所有规定95%门槛通过。100/250/500检查点实际状态各对应步数、96参数；基础模型及原上游摘要未变，导出清零重载逐张量及12题生成一致。
- 主原始归档160144字节、13成员，服务器/本地SHA 1309e964eb619d7e99c33442088474d7ea64cb2a57daa7138310f423bc4800b6一致。路径/类型检查后解压raw-v1；全部2538条前后/dev回答及500日志本地独立重算通过。
- 压力脚本SCP上传失败后通过已有已认证SSH传输，脚本SHA与本地一致；实际110题109正确。唯一失败为相近动作四事件列表第3项standing up from a chair生成standing up from bed，完整顺序正确。真实内容错误完整保留，不用此结果改提示、选权重或追加训练。
- 压力原始JSON服务器/本地SHA afec66bd0298e40b4c96c93e3b3bffc179ce21d97651aefd500f36a6e61e7442一致；110参考由文字证据独立重建，全部评分一致。最终reader导出SHA cc475127110350bb4707a8d69dbc20add273a05a64b5700fd7fb8947d83c5dbb。
- Task 1/2/3/4本轮执行完成，报告docs/stage4-text-decoder-results-2026-10-05.md。规定文字任务通过、压力检查残余1错；没有启动时序训练或替换生产模型。旧单事件297题尚未复现，本轮未评估雷达或重新测检索指标。

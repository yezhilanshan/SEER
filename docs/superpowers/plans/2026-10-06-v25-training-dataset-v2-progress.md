# SDD ledger — plan: docs/superpowers/plans/2026-10-06-v25-training-plan-dataset-v2.md

用户2026-10-06明确批准开始训练。固定预算seed42=3750实际AdamW步；开发全部达标才启动seed43，合计最多7500（失败run更新也计入）。subject12及session_holdout不得在终态锁定前进入数组/标签读取或评分；官方05/06封存。

执行技能：executing-plans、test-driven-development。沿用本对话授权认证，密码不进入任何文件。

Ruling: 仓库无.git，按已批准方案使用独立v25源码、新输出和SHA证据，不调用依赖Git的worktree/commit/清理脚本；保留本ledger与所有失败证据。误判成本是无法用Git恢复，但独占源包/清单提供恢复依据。

Pre-flight: Task1→Task2：独立fit/dev清单、MD统计代理target/mask和原teacher两路key必须显式匹配；当前Stage2单teacher接口待修订。
Pre-flight: Task2→Task3：新gate终态和Evidence权重必须同源，原旧EventState缓存不能沿用。
Pre-flight: Task3→Task4：D1 oracle-only入口、当前head dev收集及以age_seconds区间测probe待实现，不能直接用现有early_max_age整数接口冒充。
Pre-flight: Task4→Task5：当前probe、完整闭环复验、全部SHA与独占worker状态需要绑定；缓存episode预筛不能当完整策略评分。

当前阶段：Task1数据兼容与准备；实际优化更新0。SSH连接进行中。训练入口尚待实现，不宣称训练已启动。

Task1 RED：tests/test_v25_training_data.py首轮7项因缺失data模块失败。实现后首次6通过/1去重元数据失败，修复保存字典别名后7通过（interview，0.35s）。py_compile通过。
服务器实查：SSH已登录；RTX2080Ti11264MiB用270MiB，磁盘剩余3.0T；无训练进程。08/set000/001/002与13/set000均原14类，无未知命令；行数117/110/87/106，时间字符串单调。读取固定label_map.classes纠正初次预检字典层级错误，不将误报unknown作为数据失败。
Ruling: session_holdout的新增08/set003也延迟转换，保持标签/数组封存；12只计算来源SHA/头，不解析命令。准备包和最终训练包独立，准备没有优化步，不改变最后训练bundle源码SHA。
Task1当前：data.py已通过因果、归一化、去重、角色隔离测试；prepare.py已建，正在上传独占preparation-v1源码。实际优化0。

Task2双路teacher/连续状态RED 2失败，修订stage2显式新schema后与pipeline合计20通过。更新级target/clean_run/均衡采样RED3失败，实现后连同数据10通过。实际AdamW累积计数/nonfinite RED2失败→2通过；固定查询检索RED1→1通过。代码实际优化0。
准备源码传输SHA服务器读回一致，安全成员检查后解包，原生import成功。提交nohup时SSH超时退出，未收到PID或日志，不得认定启动成功或重复启动；新连接在跳板机超时。已异步请求用户确认连接，继续本地实现。
Ruling: 旧Stream teacher使用原在线控制器min(age_updates*.18/1.2,1)固定规则（来自parrad_controller.py:297），只进入冻结teacher；新gate不读取age。Commit教师是同一实际前缀的原commit路径。两路独立保存key和可见范围，历史event状态不得进入新gate输入。

Task3 visible-prefix标签/oracle-only RED2→GREEN2；连续物理Prototype及片段接缝RED1→GREEN1。Task4 seconds probe RED2→GREEN，连同pipeline/evidence34通过。新bundle资格要求probe_schema2和真实秒区间，旧桶证书不得冒充新训练验收。
Task2补充真实CPU AMP溢出0更新、保存AdamW恢复下一步逐位一致，optimizer测试4通过。所有仍为本地小模型/接口检查，不是原生训练。
服务器准备worker22228已确认后台启动，日志08/set000/001及13/set000、08/set002转换完成，08/set003仅来源SHA/headers，转换false；12仍SHA/header准备中。最后实查worker持续运行，优化更新0。
Ruling: 物理batch16按16条记录分别推进并均值归约，以携带各自独立Prototype.count，避免将不同真实年龄的标量计数错误强塞进一个批次；不改有效batch/步数/损失。Oracle配对按reference/class确定性轮转两条candidate run，限制RAM，重复抽样需统计。新状态缓存与最终gate绑定。

原生preparation-v1 PID22495在load_original来源检查处0步失败：原Stage2 trainable_only缺少的全部22个名称均为radar_encoder.*，原builder已用Stage1严格装载，Stage2保存时冻结encoder因此合理不导出。旧检查误用builder初始requires_grad集合。源SHA复核Stage1=42065f59311cb41e6c23c7734b66392028d634d7c11264bfa3325cb2be7b7289，Stage2历史SHA一致。修复只允许Stage1补充radar前缀，P/former等拟合参数缺失仍报详细名称；RED新来源测试→GREEN1。旧失败源码/log/output保留，实际优化0。
已建native-preparation-v2配置，使用preparation-v2输出。SCP首轮banner timeout；尝试现有SSH批量传输时自动审核因输入太大拒绝，无远端写入；改回常规SCP并提高连接超时，不要求用户额外批准。
Task1服务器数据已ready：fit18、dev6+2、两类holdout8+8；全部source SHA/header就绪，12/08set003转换false。数据provenance SHA ea65b45a347354da701740e7475f43956b899f444888a4501890d525e0e53d4d。
当前相关训练及v25模块汇总68通过（interview11.30s），后来新增源检查和calibration1分别通过。尚未启动实际AdamW。
Native-preparation-v2由原生v1源码独占copytree后做最小来源检查修复；远端cache.py SHA228073eb7a03814fef035a22733966b3be0c84d3e5a6efa45fb4e1a2652e1109，不宣称上传失败的本地v2 archive与其字节一致。PID22824已连续读回编码缓存与原模型参照生成，仍0 AdamW。
Task4/5新校准、原始雷达整链、固定问答、old matched baseline、holdout锁和runner已实现；新增holdout RED1→GREEN1、失败计数/录制配对RED2→GREEN2。失败AdamW状态独立读取并计入总预算，不静默重试。
Final review: executing-plans要求的一次fresh whole-source review，native_training_final_review。两项Important进入唯一修复pass：Candidate实际秒数/观察ordinal错位；旧dev QA退步被新增subject抵消。对应RED2→GREEN2、RED1→GREEN1，汇总suite见后续锁定记录。
Ruling: Oracle/闭环训练桶统一观察次数1–3/4–6/7–9/10–11；原Controller首观察age_seconds=0，Oracle age_seconds改为(ordinal-1)*.18，闭环ordinal=round(age/.18)+1。第12观察(1.98s)不入D训练，但原始轨迹、秒区间probe与终态评估保留。若理解错，将导致桶错位；现有clock测试显式锁定。
Ruling: 输出父目录已有preparation-v1/v2，正式worker使用新独占run-v1/seed42和seed43，不覆盖已失败准备证据；若路径理解错，监控需按本ledger的新路径调整。
锁定前本地v25完整suite80 passed(7.89s，interview)，两项review Important全部RED→GREEN；没有第二次review。正式源tar.gz SHA961dc072b637dc36d3c6f0b703e4e13274297eaebe05e7d84c5e88944b6cf2fb，360062bytes/171成员；服务器archiveSHA、成员路径和170项文件SHA全部一致，原生runner import成功，v25 code_digest=3dce92711841a7f85963007d8849d3d6dc1a9aa33c28bff966acd2457396289c。独占code-v1已准备。当前actual0，teacher尾段dev仍运行。
原生ready确认：两批micro4/U64均有限loss/梯度且冻结source不变，peak1429089280/1436201472bytes；选择micro4accum4，不触发fallback，优化0。encoder index SHA205d96c8993c1570c13b0cdb0a708faa7025d094d1b974b5a8ecc5420bbbdc60，teachers index SHA70e64fea61ccc64f9fca2a009d12960f8db79945c85a8426fcb4f31c57022792f。native prep config SHAc6ac6b5ed68f97f4018dd5f1bafebb84ca4bf5eae16802fe177b12e351e76de79。正式worker创建指令已发，须读回PID和实际step确认，不能凭发出命令认定更新成功。
Formal run-v1 PID23222在任何优化前因旧Stage3 SHA65字符抄录错误停止，权重服务器SHA与历史QA/raw归档一致；同时纠正threshold SHA多字符。配置全SHA格式测试RED→GREEN；native准备不重做，失败日志/源/输出保留。正式新code-v2/run-v2只修正来源录入与preflight错误状态保存，无训练预算/损失变更，累计成功更新0。
正式v2源码归档SHA0b8abd743ca4e136889ecd8308a589b9c5173a8043c15ad8ea4e09f33371a196，361156bytes/172成员，远端文件清单和全部7项锁定来源SHA逐项一致。worker PID23319，2026-10-06T19:24:38+08:00，独占run-v2；实际更新待读回。前run-v1零步不增加预算。
正式训练已确认A65步，AMP skipped0；第20步checkpoint SHA5dd20666829e74a0447f46c887254fcc7dafd6afc720442ea0509eb650eeb0fb，独立保存optimizer所有parameter step集合[20]。前20成功步mean2.382890197s，采样时A剩余2228s约37min，当前阶段预计北京时间20:05前后（不含最终检索验收）。startup证据归档SHA6984df6c3c05614d10dd4b4db9f3dbaf0c63eff9aa76b6c1104f371ba62ea01c，286798bytes，下载进行中。v25 runtime digest69731c6386e7bb211e0f5754103d548b8a3dbbe5f1008a23aee02dce385e1513，final config SHAf6114594a27ec75f8ebf982b54e4d29c3d29901f36dbe7bd741122d9dc3024dad，seed42 schedule SHAfdab404bed7fde451a2005f2df129e784de2cb830bd80c1c333cb079eefd3ee79。
2026-10-07继续：新SSH只读确认旧worker23319已退出，正常A retrieval-retention dev-failed终态，实际1000/3750；B/C/D/E/F、seed43未启动，holdout lock不存在。独立服务器检查20/100/500/1000保存AdamW全parameter step、checkpointSHA、source-manifest通过；冻结digest不变。完整JSON/log+712事件预测归档SHA8344148845ced9cef0c4cbdd08b628e49c745942812897583ee1cf6cc057637e，1575377bytes/13成员，下载本地SHA/member校验通过，独立NumPy重算所有已有组最大差1.11e-16。完整dev Stream mAP-3.04pp/R@1-7.14pp；Commit mAP-1.85pp/R@1不变。启动原计划zero-gate基线评估PID3592，无optimizer、dev-only、封存不解锁，用于归因而不改原终态或重开训练。
零步新gate对照PID3592已正常完成，actual_optimizer_steps0。归档SHAe2f63b164c54177db11491582577c21fb5901117a135dd9acaa13d3f20fdcec7，1386561bytes/7成员，本地服务器一致，成员路径校验后解包；独立NumPy重算最大差2.22e-16。原查询/teacher键/712事件UID/标签/角色与1000步终态完全配对。完整dev：原Stream60.85→零步57.56→训练57.81mAP，Commit61.66→零步58.32→训练59.80mAP。主要损失在更换新初始化gate时已出现，训练恢复部分而不合格；不将架构/初始化/去age等合并因素分开声称因果。所有训练仍1000步，未追加优化或选择中间checkpoint。报告已更新。
2026-10-07补全startup-evidence-v1.tar.gz下载：SHA6984df6c3c05614d10dd4b4db9f3dbaf0c63eff9aa76b6c1104f371ba62ea01c与服务器一致，安全成员及重复路径检查通过，解包startup-evidence-local-v1并保存逐成员SHA。原生数据准备/encoder/teacher索引/预检/旧零步失败日志/启动快照证据均本地保存。原训练和零步诊断均已退出，未开启新训练。

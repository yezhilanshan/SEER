# SDD ledger — plan: docs/superpowers/plans/2026-10-07-v25-gate-migration-implementation.md

2026-10-07 用户要求按草案实施，主线已授权；当前新优化0。

Ruling: 仓库没有.git，保留SHA隔离源码/新输出及ledger，不运行Git worktree/commit/清理脚本。用户明确“开始实施”覆盖技能默认再次询问执行方式/计划审阅流程，原可选支路仍不自动获准。

Pre-flight Task1→Task3：gate转换身份必须写入诊断来源；原随机D归档只作配对参照，不能混成热启动初态。
Pre-flight Task2→Task4：Stream接续状态覆盖完整prefix，监督区间需要显式声明；不得把detach误解为reset。
Pre-flight Task3→Task4：fit可用性规则先锁后看结果，不以dev最佳诊断决定训练架构。

SSH已认证，session21233；尚未上传或启动新worker。

Task1 RED：PYTHONPATH缺失的首轮收集失败不算功能RED；补src后缺gate_diagnostic模块失败。实现转换/固定条件/fit探针后测试忘传exact参数，修正测试。ToyFormer验证65步学生自己滚动及前64标记，未用teacher状态冒充学生轨迹。
Task2 RED：前缀监督覆盖2项因缺函数失败，top5/margin1项因缺字段失败。实现后新增相关测试12 passed（interview3.30s）。apply_patch结果报重复旧行但落盘已有主要改动，逐文件核对后用真实pytest验收，不按tool消息假定成功。
服务器GPU RTX2080Ti用270MiB/11264MiB；旧worker退出，无新训练，缓存目录存在。
Ruling: pytest tests -k v25仍收集无关H5转换，因interview缺h5py失败；改为显式test_v25_*.py列表，无依赖安装，不能把此运行叫全仓库通过。
`nTask1/2: v25 suite92 passed(interview16.84s). 新增配置SHA测试RED发现teacher摘要65字符，多一个c；按服务器直接读回64字符修正，新增相关13 passed(interview3.08s)。fresh whole-source gate_diagnostic_review进行中。新训练0步。
Task1/2 complete: interview全部v25测试95 passed(14.21s)。一次fresh source review发现FP16 trace调用、推理配置SHA锁、首步nonfinite必须记录不合格而非丢证据、B1/C-exact完整递归轨迹四项；唯一fix pass均加入RED→GREEN回归（FP16首轮dtype报错、配置/轨迹/None规则缺功能失败；最终95通过）。ToyFormer改用真实SemanticProjection遵循原生dtype接口。
Review declined native真实性/缓存SHA/归档/独立重算/尚未执行的条件训练；由Task3实际执行验证，不推断已通过。没有第二次source review。
源包source.tar.gz SHA6f7ef235c8c9a98b2bc7bb88c6dea6648e2b828250e41d86865ac4f54a6e0653，29183 bytes/10 members。服务器验证旧171项文件、新9项文件；独占code-v1共177项runtime文件。旧源码/权重/输出未覆盖。正在服务器运行15项相关测试，尚未启动原生worker。
Native v1: 15 tests passed(2.15s)，PID16007于2026-10-07 20:02:24+08:00启动；fit18个最长run探针完成，alpha负值比例0.724553，范数比p99=26767.703，出现非有限输出，明确热启动不合格。随后切换阶段时status调用prepare.write_json(exclusive x)产生FileExistsError；不是训练失败，优化0。v1代码、日志、fit结果保留。
Root cause: 不可变报告写入函数误用于可变status。新增原子status写入RED缺函数→GREEN；不改不可变报告exclusive协议。所有v25测试96 passed(13.33s)。新v2包SHAe141c0c9d832bde80c4dbfbb0802a7f844b4acadcbb67e4e4cef28b2257dc8ca，29421bytes/10成员；服务器旧171项及新177项来源完整校验，独占code-v2已就绪。仅修复状态记录，不改变诊断标准/架构/数据/预算。
Ruling: 根据先锁fit停止条件，C-raw不可用，不进入1000步训练，不把C-exact临时选作学生。继续完成全部零步dev对照，以定位maturity与参数化影响。
Native v2: 16 tests passed(1.58s)，PID16177，2026-10-07T20:10:31+08:00；输出/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/v25_gate_migration_v1/diagnostic-v2。runtime源清单SHAbfffda2814c9a5e64af3d6c768bda20de6154d36290cd2f8495e40e99a2d291c。fit已完成且不合格，正在完整dev零步A对照。准备本地独立NumPy读回和最终报告脚本；尚未归档，训练0步。
Native v2 A-original完整712事件两路检索完成：mAP/R@1差均0，Stream/Commit键与原teacher逐位相同，门槛passed。进入B-m0。该结果实测排除来源/推理设置改变，不能据此提前推断其他gate结果。

Task3 complete: v2 PID16177已完成，status=completed，next_action=stop-unusable-warmstart，实际AdamW更新0。五个新对照及复用D均已完整报告。C-raw fit负alpha比例72.4553%，范数比p99=26767.703，18条fit轨迹全部出现首次非有限状态；dev Stream704/712事件非有限，指标未定义。B-m1和C-exact全部原检索门槛通过；B-m0因subject_08/68xx Stream R@1单元失败；D失败。不得按dev最佳分支临时选择训练架构。
Evidence readback complete: 服务器归档12469807 bytes，36成员（35项内容摘要及成员清单），SHA3b5e3e61e1ea9ae772963a962cdba42c1cb30e5ed5adbdae5056558b00c1326b与本地一致。先验证归档SHA、成员路径、重复名及文件类型，再解压至全新v25-gate-diagnostic-readback-v2目录；全部35项SHA一致。177项服务器runtime源码再次核验。新对照事件元数据/查询/teacher键与A逐位一致；D原归档及数组来源验证。NumPy float64独立重算全部路径/分组AP、R@1、R@5，与正式指标最大差1.1102230246251565e-16。
Final report: docs/v2.5-gate-migration-diagnostic-results-2026-10-07.md；完整本地原始JSON/log/预测及local-readback.json位于.audit-staging/v25-gate-diagnostic-readback-v2。96项本地v25测试、16项服务器相关测试通过，不称全仓库通过。
Terminal ruling: Task1/2/3完成；Task4迁移训练与后续Evidence训练按预注册fit停止条件未启动。本轮没有新参数更新、部署、P解冻、双路守卫放宽或封存留出访问；旧权重/模型/历史证据保留。C-exact仅为正alpha兼容诊断，不具备已经训练的负alpha撤销能力；后续必须先显式修订架构/初始化再锁定训练路线。

# SDD ledger — plan: docs/superpowers/plans/2026-10-07-v25-compatible-evidence-implementation.md

用户“请你开始吧”授权精确转换正式架构及Evidence500；本轮不执行gate1000或负alpha创新支路。
Ruling: 无.git，使用新的SHA源码包/独占服务器输出，不清理历史证据。已批准对话设计覆盖技能重复文档审批握手；依据当前明确请求执行，不扩大到adapter/P解冻。
Pre-flight Task1→Task2: 兼容正alpha标签不能冒充原signed架构；bundle精确身份、开发状态独立，新训练入口不经Stage2Migration，线上controller守卫不放宽。
Pre-flight Task2→Task3: 现有EvidenceBatches交替路径与10个subject/sensor格的偶数周期绑定，会使同一格固定单一路径；改为每个抽样事件同时送两路，8事件×2=16，记录每格每路覆盖。
SSH默认沙盒连接被拒绝；通过已授权escalated SSH成功到认证提示，仅交互传入既有认证，未写密码文件。
Task1/2 RED: 新测试5项因缺正式gate/bundle及训练模块功能失败，随后实现。相关35 passed(21.94s)，新增缓存范围/检索配对实行为验证7 passed(6.68s)。同事件两路停止梯度与每格覆盖已实测。已锁seed42/500AdamW/lr1e-3/wd.01/clip1/8事件双路，不调用gate训练。
Ruling: 现有头训练采用masked SmoothL1，沿用它；验收仍用MAE，fit baseline改为NumPy实际中位数（偶数样本取中间两项均值），空属性保留None，禁止零补基线。中位数定义显式预注册，避免旧torch.median取下中位数歧义。
SSH首次登录后Connection reset；后两次跳板超时。后台worker尚未创建，无重复启动。本地代码/config部分apply_patch报错但落盘已包含主要改动，逐项读回确认后继续，未把工具报错视为成功。
Local whole v25 suite103 passed(17.26s)。全项目pytest收集在tests/test_parrad_convert.py因interview缺h5py失败（5.43s），未安装依赖，完整日志.audit-staging/compatible-evidence-full-suite.txt；不能称全仓库通过。Tailscale只读状态显示跳板offline/10m前最后在线，2次ping无响应；已异步请用户恢复跳板，继续独立审查/上传包准备。
Whole-source reviewer compatible_evidence_review已按executing-plans要求派出，fresh context只读；无服务器访问。训练实际0步。
Coverage RED→GREEN: 首帧两个delta代理引用事件前全局帧，构造首帧.9/其他.2回归观察错误均值.375；positive缓存屏蔽首帧delta，历史signed缓存不变，相关8 passed(4.16s)。这是保证可见区间契约的修复，先锁定再看新dev结果。已告知一次源码reviewer。
One whole-source review: 无Critical，两项Important为record顶层role与metadata分区缺核对、头训练异常缺实际AdamW/权重证据；两项Minor为array_equal非逐字节、0步checkpoint缺feature SHA。全部在唯一fix pass中修复。RED：开发误标不抛错，signed-zero/dtype误差被接受，失败回调缺参数；GREEN相关14 passed(13.43s)，小型真实模型500更新单元实验验证所有Evidence参数step500、其他头不变、20个subject/sensor/path格各400，单元fixture不计入原生实验预算。主入口每成功步原子更新status，失败保存当前head、optimizer、来源及抽样记录；phase计数表示完整更新，逐参数步数单列。
Task1/2代码完成，审查修复后整套v25回归进行中；本轮服务器实际优化0。线上controller/Stage2/cache旧signed守卫不放宽，positive bundle仅development并严格身份匹配。
Task1/2 complete: 一次fix pass后全部v25测试110 passed(17.76s)，含500步小型单元验证及两次已更新2步的失败证据回归。未重做第二次源码审查。项目全测试仍受已记录h5py收集错误限制。
上传包就绪：.audit-staging/v25-compatible-evidence-package-v1/source.tar.gz，352841 bytes/127成员（126项文件及manifest），SHA c504387ccb3b424591b404d4ac88f3b0f2b3ce8eab1773416966b6c091d483f3。代码/config/方案独立封装；未上传或启动原生worker。
Task3等待连接：最新Tailscale JSON状态显示100.83.206.93 Online=False，LastSeen=2026-10-07T13:30:00Z。已请求用户恢复跳板，当前唯一外部阻塞为跳板离线。原生实际AdamW0、无PID；不能把unit-fixture500步误报为服务器训练完成。恢复后先核实远端无同轮worker与GPU，再上传/来源验证/原生测试/重建特征/500步训练，不重复已有后台worker。
Task3 resumed: 用户要求ssh xibeidaxue@10.15.22.62直连，session97609认证成功，沿用交互凭据不落盘。RTX2080Ti用270/11264MiB，无同轮worker，新code/upload/output目录均原先不存在。SCP直传352841bytes；服务器先验证归档SHA和127安全成员路径，再独占解包到/home/xibeidaxue/SEER/.audit-staging/v25-compatible-evidence-code-v1，126项源码SHA一致。原生相关preflight正在运行；实验实际优化仍0。
Native preflight36 passed(3.89s)，exit0，日志隔离code-v1/native-preflight.log。一次独占后台launcher PID20195，2026-10-07T22:05:34.519550+08:00，直连transport，无重复worker。输出/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/v25_compatible_evidence_v1/evidence500，日志同根logs/training.log；源码清单SHA9726dd7d2452549857bc68caa16500bb362b1f2e5e7809c54dcd1fbc8add5217b。使用start_new_session及stdinDEVNULL，SSH断线不停止worker。训练预算500、seed42、gate0；先source验证/重建fit-dev冻结事件特征/配对检索，然后头优化。
启动读回：PID20195 elapsed39s存活，status.phase=frozen-event-features，实际优化0；已完成subject_02-set000-TI_xWR14xx。无错误栈，只有torch_dtype弃用提示。来源/分区校验已通过；尚不能宣称配对检索或Evidence门槛通过。用户要求“直连”已落实，后台流程正常启动，Task3训练/结果核验仍待完成。
Terminal check: 用户要求检查结果，直连session78577成功。PID20195已退出，status.completed/实际AdamW500，终态时间2026-10-07T22:23:05.218667+08:00。原生结果属性主门槛passed，dev MAE0.06904567325406961，fit中位数基线0.10038575729666856，相对改善31.21965%。gate更新0，原检索通过，尚在本地独立重算前不作最终核验完成声明。
Native independent audit-v1 exit0：所有126源码SHA及Stage1/2/配置/索引核验；0/100/500 checkpoint，100与500各6个Evidence参数实际AdamW步数一致，其余头逐张量不变；bundle tensor digests/开发身份/来源重建frontend digest一致，encoder与缓存源一致。1833fit/712dev全部事件覆盖、每record SHA与首帧跨边界delta屏蔽目标独立核验。4000fit事件draw/10格各400/双路20格各400，无留出/official访问。大型前端权重仅服务器读回。
终态证据归档results-evidence-v1.tar.gz，9404912 bytes/30成员，SHA ad726e016647e3c4b9503ebb0d425e681189698c5035aa8d649d9ca73cf9dc13b。包含完整JSON/log/预测、proxy-targets及0/100/500小型头检查点、源码与来源清单；下载及独立NumPy重算进行中，不含巨大frontend/events权重。核验脚本首次通过终端内联因输入审阅尺寸被工具拒绝，改成SCP上传脚本后安全执行，未重训或改参数。
SHA correction: 前条SHA从PTY换行抄录时多一个1（65字符），第一次本地前置校验因此失败且没有解压。直接重新读取服务器原始sha256sum和metadata，再与本地Get-FileHash比较，一致的正确64字符SHA为ad726e016647e3c4b9503ebb0d425e68189698c5035aa8d649d9ca73cf9dc13b。文件大小9404912bytes一致，文件未损坏；历史抄录错误明确保留。
Task3 complete: 本地先SHA/安全成员路径验证，再独占解包.audit-staging/v25-compatible-evidence-readback-v1；30成员、29内容SHA全部一致。服务器0/100/500小型检查点、原始JSON/log/便携数组/代理目标及来源完整保留，巨大的前端权重服务器验证不下载。NumPy独立重算全部fit中位数/MAE/分组属性指标最大差0；检索mAP/R@1/R@5各分组最大差1.1102230246251565e-16。与之前C-exact原始数组逐dtype/shape/bytes一致。最终属性passed、检索preserved。
Final outcome: dev MAE0.06904567325406961，基线0.10038575729666856，改善31.21965%；四属性合并改善50.6826/20.3892/27.2050/19.8159%；Stream30.2083%、Commit32.2310%。全部8个subject×sensor合并属性改善18.4030–44.8859%；6个细分属性不到10%，没有比常量更差的细分属性，完整表列出。检索Stream mAP60.87329%、Commit61.65692%，R@1均64.28571%；全部原门槛通过，相对兼容初始化训练前后完全不变。
报告docs/v2.5-compatible-evidence-results-2026-10-07.md及local-readback.json已生成。固定500终态选择，未择优；数据仍是上游已见受试者的GT开发录制，不能称未见测试/自动事件或QA提升/正式Go。正alpha兼容架构不冒充负alpha能力，未启动下一轮或修改生产模型。Task1/2/3全部完成。

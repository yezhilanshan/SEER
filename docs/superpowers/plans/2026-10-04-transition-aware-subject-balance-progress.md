# 执行记录：受试者均衡50步

用户已授权本对话的受试者均衡小实验，方案见docs/stage3-transition-aware-subject-balance-2026-10-04.md。

- 新API显式weighting=subject-age-class；默认age-class仍走原算法和身份。八个fit桶各1/8，缺必需桶停止，不用dev补样本；新权重只进入transition BCE及对应临时头，保留原teacher标准化和检索保护。
- 新5项测试先失败，均为缺weighting API；真实默认优化已运行，失败日志projection-subject-balance-red.log。随后实现权重、CLI、manifest/warmup/condition恢复保护，相关回归正在运行。
- 服务器19:28:44连接成功，无Projection worker，GPU0 RTX2080Ti 270/11264MiB。此刻尚未上传或启动新实验。
- 工作区无Git元数据，沿用独立部署包及SHA记录；不会覆盖旧实验/生产模型。当前MEMORY注册表没有相关项目历史，依据本地方案和原始结果工作。

## 已启动（10月4日19:38）

- 相关测试34 passed/61.37秒；本地可用完整回归429 passed、2 skipped、既有pkg_resources弃用warning一项，243.89秒。沿用h5py缺失排除tests/test_parrad_convert.py，没有安装依赖。
- 新包SHA256 504999e413675dc582fbe04a4ccd4010ae878dc60ea4ba4bed0d00e8b78d813e；上传后服务器一致。隔离复制projection-budget-code-v1，解包至projection-subject-balance-code-v1；六个部署文件逐项SHA完全匹配，bash -n通过。服务器全部Projection相关测试75 passed/26.00秒。
- 原生产runner/pilot SHA分别7a665f0d6868eb57e2de5f6c49045c8aaeb9cac395ad29575d2809446eaa97d5、2cb4d27263a3d8dbb1af5d27938fc58afae5b30a08196e9dc0e794779050b614。启动前无其他Projection worker，GPU0 270/11264MiB，新输出/日志目录不存在。
- Pipeline于2026-10-04 19:38:55北京时间后台启动，launcher PID20219。19:40:28三折来源/缓存/fit分桶检查全部通过SUBJECT_BALANCE_PREFLIGHT_OK，进入subject02 age-class原权重50步对照。不是从旧适配P继续训练。
- 原权重对照20/50步必须与旧100步轨迹精确一致，才执行本折新权重；新权重baseline的独立dev和检索数值须等于原P，而临时头fit不同是预期。全部原始证据保存在服务器新输出，不重复worker。
- 本地新只读结果重算脚本analyze-projection-subject-balance-v1.py已准备，语法检查通过；真实新结果尚未完成，没有宣称均衡带来收益。后续按锁定门槛完整报告两个系数及所有局部组。
- 按用户此前定时检查请求，原heartbeat seer已更新为本次实验并置ACTIVE，每20分钟，工具和TOML均确认；正常安静，仅完成/失败/需用户处理时通知，核验与报告后暂停。未存储密码。
- 更正此前表述：age_updates6/7是候选状态自提议后的更新计数，不是受试者年龄。锁定方案的age桶指这个状态更新字段，未改变样本或协议。

- 19:43:51第一折age-class原权重对照50步完成并独立读回通过，两个条件均实际50步/ready；20与50步P、dev与原语义值精确复现旧轨迹。随后launcher进入subject02 subject-age-class，尚未取得新权重完成结果。preflight原subject质量：held02约49.47/50.53%，held03约54.93/45.07%，held04约55.43/44.57%；不是极端总体失衡，均衡假设仍须实测，不能先宣称原因或收益。

## 20:00定时检查：训练完成，正在下载原始证据

- SSH认证成功，完整status读回：19:38:55启动；预检查19:40:28完成；六个运行依次结束19:43:51、19:47:38、19:51:46、19:55:58、20:00:27、20:04:57。最后一折均衡训练正常完成，launcher20219已退出，没有重启worker。
- 新只读审计脚本SHA0f3485c5263ce42465be07dc0e2879d35a509a8453dc228bf785e473e45d89cc，上传SHA一致。服务器独立重新核验输出ALL_INTEGRITY_OK：六个run、600实际P更新、每次3头拟合全部收敛；各checkpoint身份、终态50步与last参数、优化器实际步数、teacher/控制P、来源/config/code/spec与原生产SHA通过；缓存readback与新审计精确一致。
- 原始JSON/log归档projection-subject-balance-results-v1.tar.gz约6.4MiB，服务器SHA c54f4f57d36fb7b95a1233c18c58213ce9c094b4e631c1ba6478bc3596ecbb3c。下载中，尚不能声称本地归档核验或最终报告已完成。
- 服务器初步50步读回：新权重relation1三折AUROC .6628105118355316/.5728866219980602/.5844576479949803；relation10 .6632940630640667/.572600462481356/.5848685436191259。第三折比原权重退步；第一折relation10语义不完整。
- 第一折均衡relation10 50步独立动作评估subject03/04 stream probe达到400次、finite但不满足收敛，缺macro-F1；不是训练转变头失败。保留原失败/缺失，不修改solver预算。
- 本地读取器先用旧已核验50步真实预测重算，三折原权重AUROC精确匹配；分组新/旧权重比较采用逐feature-seed paired差值的median，不能用两个baseline差值median相减代替。等待新完整下载后再计算最终AUROC/AP与成功标准。

## 完成核验与报告

- 完整原始JSON/log下载成功；本地归档SHA与服务器c54f4f57d36fb7b95a1233c18c58213ce9c094b4e631c1ba6478bc3596ecbb3c精确一致，788成员路径/类型检查通过，解压至新目录.audit-staging/results-projection-subject-balance-v1。原权重/优化器pt未下载，证据由服务器独立读回提供。
- 本地从保存预测重算20/50步所有binary metrics、原AUROC/AP聚合、原语义门槛和逐subject/age_updates paired差值；验证来源/Stage2/code/spec SHA、六个run actual steps及600份每步头报告全部通过。报告docs/stage3-transition-aware-subject-balance-results-2026-10-04.md；摘要位于原始下载目录内transition-aware-semantic-primary-subject-balance-v1/summary.json。
- 相对同预算原权重50步：relation1 macro AUROC -0.082305pp、AP -0.061765pp；relation10 AUROC -0.064132pp、AP -0.034765pp。第三折分别-0.446369/-0.424064pp，未抵消前两折小幅改善。两个系数均未通过预注册成功标准，不将均衡权重升级为新基线。
- 最差局部分组为第三折dev subject02/age_updates7；相对原P paired AUROC差值：relation1从-2.481203pp到-3.293233pp，relation10从-2.375940pp到-2.932331pp；虽9/12和8/12组相对原权重不下降，最差组加重。这里group采用逐feature-seed paired median，与两个绝对median相减不等价。
- 50步全部已定义检索准确率及动作F1均在原容差内，最差检索相对原P -1.785713pp、可定义动作F1 -0.794844pp。均衡relation10第一折独立stream action probe两个未收敛、macro-F1缺失，故不能确认该条件完整保留。relation1各折均完整通过；所有不可用项明确保留，不改变solver。
- 新/旧权重逐项语义差值已加入summary.semantic_vs_original：第一折relation1 subject03 stream动作F1相对原权重-0.8403pp；第二折两系数subject02 commit text_to_event相对原权重+1.785713pp；第三折relation10 subject03 commit动作F1+0.7198pp，其余50步已定义检索准确率/F1精确相等。完整margin/consistency及20步差值也存档。
- 实验19:38:55至20:04:57，1562秒；原模型未替换。保留原权重50步研究基线；后续目标改动需另定小实验，不自动解冻Former或开启新训练。本次仍为相关TRAIN内部开发折，不是独立未见受试者测试或正式Go。
- 原生automation_update已将seer置PAUSED，保留其名称、完整prompt、20分钟间隔和目标对话；报告核验完成后停止重复检查。

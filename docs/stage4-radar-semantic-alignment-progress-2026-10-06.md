# 单事件语义适配执行记录

依据：用户批准docs/stage4-radar-semantic-alignment-2026-10-06.md对应对话方案。

Ruling: 仓库无Git，沿用新隔离服务器源码目录、源码归档SHA和独占输出，不执行worktree/commit/清理旧证据。批准的在聊天设计足以执行现有QA流程的限定实验，不再索要同一行动授权。

任务1：六项采样/问题-证据区分/答案位置与教师冻结/新reader反向/门槛测试先RED，再GREEN；增加完整预测来源及交换实际答案四项拒绝检查。真实生成入口RED暴露identify任务缺pair_id字段，固定为显式None，再GREEN。新增tiny原生Qwen完整反向及冻结/优化步测试，本地transformers/peft缺失留服务器执行。

本地相关17 passed/2 skipped；完整pytest在test_parrad_convert.py收集缺h5py，保留radar-alignment-local-full.log，不安装依赖。服务器SSH已连通，RTX2080Ti11GB显存270MiB，无已有stage4训练进程。

锁定lr1e-4（前一聊天方案未指定，固定在训练前，避免dev调参）；两组500、seed42、蒸馏0.5/T2、同批次，原始token与全新身份残差为零步起点。fit教师95%门槛仅作是否可训练判断，dev文字题全部报告但不选择提示或预算。当前尚未上传/启动。

初始源归档SHA9c7cdc26bbffe16a6a01f36c6ae2eecd7ff8c9d1de6802321ec553502dfac22f，387575字节196成员，服务器SHA/安全成员路径/manifest全部通过。原生预检PID17153：相关20 passed/1 warning，完整511 passed/2 failed/6 warnings（仍仅原Gemma两项兼容性失败）。真实GPU四次诊断更新，ce/distill各2，六AdamW状态到2；新reader/upstream/原检索前后精确一致，峰值2297922560字节。未复用诊断权重。
一次独立审查：无Critical，两项Important涉及失败/核验异常时的证据读回。新增两拒绝用例RED→GREEN：失败运行检查已有来源/预测/日志及已完成arm/检查点，保持整体failed；核验子进程异常时父入口记录错误并无条件归档原始JSON/log，exit非零、complete_checks_performed=false，不能假报通过。未改变训练或生成代码，最终包等待原生修复回归。

审查修复额外实际500步AdamW回归：失败运行已完成ce arm的所有100/250/500状态及输出均核对；将500步状态改499必拒绝。本地相关21 passed/2 skipped。最终ready源码归档bc210ab34928a79863abc02725e24051ed011560e3ddb6e19f80b860fdfd0404，391057字节197成员。训练/推理/原生probe代码与初始通过预检的版本完全一致，仅改核验/失败证据处理和相关测试。最终服务器完整/相关套件待执行，尚未启动正式训练。

最终服务器原生相关23 passed/1 warning，无跳过；完整514 passed/2 failed/6 warnings，75.41s，失败集合严格核对仍只有test_gemma3_adapter_matches_native_prefix_and_targets_only_qv与test_gemma3_adapter_honors_local_sliding_window。启动前再次核对全部最终源manifest SHA、固定配置及GPU四次诊断结果，正式预算与诊断隔离。

2026-10-06 12:19:39.613546北京时间：独占后台worker PID17456已启动，输出stage4_radar_alignment_v1，source-ready SHA为bc210ab34928a79863abc02725e24051ed011560e3ddb6e19f80b860fdfd0404。实际流程先单事件文字教师生成门槛，再零步/原模型基线，随后ce与distill各500；此时尚未读回实际优化步数，不宣称已完成训练。

启动后48秒读回：running/text-preflight，fit文字生成400/588进行中，ce/distill实际更新均0，worker17456 PPID1正常后台存活。原生测试/诊断/启动记录另归档native-evidence-v1.tar.gz，14367字节，SHA19b88d56bed11d61f63d67709e2380e587a394dff1c1987a5b000317d33816167。

后续只读状态检查期间SSH出现传输波动；native小归档的scp在banner exchange超时，未下载完成，本地不宣称已保存该归档。旧会话无新输出，另建只读SSH尝试。最后已确认后台worker17456存活、running/text-preflight、fit400/588，实际更新0/0。后台worker独立于SSH，不重复启动或改预算，原生日志与启动/源码证据均留服务器。

2026-10-06用户要求检查训练结果：SSH恢复，只读确认blocked/text-preflight、实际更新ce0/distill0，worker17456已退出。文字fit588/dev297全部完成，宏动作召回均85.71%，Yes/No各100%；fit未达95%宏平均门槛，因此按锁定方案停止，并非SSH断线停止。独立读回verification_status=passed、complete_checks_performed=false；无正式训练/检查点/雷达基线/训练后检索快照。完整原始归档37472字节10成员，SHA5454daf4cde8af4f096b6ca2ba68cef470c4382410bdaa0c9b5a6e39117fc7d8，已下载、安全路径检查并解压新本地目录radar-alignment-results-readback-2026-10-06；原评分与完整任务绑定复算exit0。

错误集中label6冠词a/the（fit17/dev8）和label7坐在床上误答坐到床上（fit9/dev7）。仅接受前者的解释性诊断宏平均92.86%，仍未过95%，不修改锁定分数或门槛。完整结果见stage4-radar-semantic-alignment-results-2026-10-06.md。此次未启动新训练。

实时SSH重新核对全部4个保护文件SHA均一致，训练日志/100/250/500检查点/推理导出列表为空。终止status更新时间为北京时间12:21:17.946827，主实验duration_s=96.8493。

# 位置与时间编码执行记录

- 已批准：三组none/order/time固定预算对照，新融合层零输出初始化、冻结旧QA、单事件严格绕过、原检索路径保护。
- 已实现：models/temporal_qa.py、parrad_temporal_qa.py、配置及专项测试。
- RED：缺失时序模块；随后纯文字门槛缺失的真实断言失败。GREEN：专项和Stage4相关回归40 passed。
- 网络：首次SSH跳板超时；重试已到服务器认证，Tailscale ping返回DERP可达，已使用本会话授权认证登录。无密码写入文件。
- 服务器GPU：RTX2080Ti，显存270/11264MiB、利用率0%；未发现旧QA训练worker。
- 有限独立审查：Critical/Important无；Minor为导出配置不兼容构造函数，已先复现失败再修复，增加模型重建输出逐张量一致测试。
- 本地可用全回归469 passed、2 skipped、1条既有warning，150.24秒；既有h5py转换测试排除，不安装依赖。导出修复后Stage4相关41 passed。
- 代码归档SHA：7b3cab8dfb27abb9a0ea9d034f445dd472a224d2a2b17bcfe22d82cc7ce9026c，348721bytes，163成员。服务器SHA、成员类型/路径、162文件SHA清单一致后解压到隔离目录/home/xibeidaxue/SEER/.audit-staging/stage4-temporal-qa-code-v1。
- 服务器原生Stage4相关测试41 passed，3.04秒；启动/worker/读回脚本本地py_compile通过。
- 后台于2026-10-05 18:25:03北京时间启动，launcher PID4531，实验进程PID4532；已结合ps、status.json确认真实阶段text-preflight。GPU任务为生成纯文字诊断回答，尚未进行优化更新。
- fit纯文字诊断1014题完成，未通过锁定≥95%门槛：双位置1/2均100%；四位置1/2/3/4为100/93.8/89.6/81.2%，完整顺序14/48=29.2%；双／四Yes均0正确，No均100%。status=blocked，实际新增优化0步，三个训练组／检查点未创建。
- 单变量只读引用探针：每种任务前8个fit四事件问题，把memory引用改为Evidence list，存在性与顺序均4/8→5/8，位置8/8不变。未改锁定模板、未更新参数、未启动第二轮训练。
- 已结合ps确认后台PID4531/4532退出。未启动自动定时检查。
- 最终9成员96248bytes证据归档服务器／本地SHA一致：2689a30f553a7f64c1cd122ed78e84c97e08462ad43f43040dec6773998a6120。SCP不稳定，通过原已认证SSH会话分块取回；路径／类型检查后解压新目录。
- 本地1014题及24条引用探针真实生成独立重算通过。实现及诊断结果见docs/stage4-temporal-qa-results-2026-10-05.md；位置编码学习效果尚未测试，不能称训练已开始或提升。
- 待完成的训练：需先解决当前冻结问答解码器的文字读取／顺序列表输出，再启动原批准的三组500步预算。本次未自动训练LLM／LoRA或修改生产模型。

## 用户授权修复后的 v2

- 只读2×2诊断：同一批24个fit问题，LoRA/关闭LoRA × inputs_embeds/input_ids。四种存在性均4/8；完整顺序分别4/8、7/8、6/8、6/8；位置8/8、8/8、7/8、7/8。不存在能够单独解决问题的接口或LoRA开关证据。原上游摘要未变，诊断保存在服务器stage4_decoder_rootcause_v1/probe.json。
- 明确源码错误：oracle存在性指令同时要求“只用活动名称回答”和“Yes/No”；此外文字证据问题仍引用supplied memory。v2仅修复oracle回答合同和证据引用；普通雷达问题、生成接口、LoRA、模型结构、固定预算及门槛不变。不能据此宣称整个解码器已修好。
- 新回归先RED（实际捕获冲突）再GREEN；当前相关26 passed。修复模块SHA为7e31ddba193bf40c0228ba41313b16e7b8f224ced5bf721063e5f9e08b433739。
- 服务器保留v1代码／结果，新隔离目录stage4-temporal-qa-code-v2。模块传输SHA一致，未修改的v1清单文件逐项验证；另存v2-source-manifest.json。
- v2于2026-10-05 19:04:58北京时间启动，launcher PID5160，输出stage4_temporal_qa_v2。先完整fit文字诊断1014题；仅通过原95%各项门槛才进入三组500步。当前结果需以服务器实际status和保存回答为准。
- 完整v2已结束并独立核验，worker5161、launcher5160均退出。status=blocked、actual_updates=0；双Yes2/97、No97/97，四Yes10/48、No47/48；四完整顺序29/48，四位置48/45/40/43。修复不充分，不能声称解码器已修复或时序训练已开始。
- 本地完整8成员归档95427字节SHA与服务器一致：3b4eb59022031b399c1231658a66decef31651b2b61917870bb6bb55e1fd9934。全部1014保存回答独立重评分与门槛一致；raw-v2、recomputed-v2.json已保存。
- 有限静态审查无问题；已向用户请求确认独立QA LoRA训练（改变Full Qwen/LoRA冻结约束）或专用分类/顺序头。在选择前不启动依赖该选择的训练。可执行边界及全部复测见docs/stage4-decoder-repair-2026-10-05.md。

# 主任务训练方法复核：已完成并核验，原始下载已补齐

最新结果：原流程于22:38:58正常完成，两条件各20步；新折原主AUROC0.562906，relation1为0.567182，relation10为0.567681，均选择20步且全部检查点通过语义门槛。源码/方案/源manifest/Stage2 checkpoint、角色及缓存artifact、选中权重SHA、冻结teacher和实际优化步数核验通过。10月4日完整JSON/log已下载，归档SHA`0841c2020669273afbf720a661c610c95538fd1f11f41934e6d10ba0a478ddb5`核验匹配；本地重算选中检索准确率和动作F1最差delta均0，margin/consistency有小幅变化。以下启动/连接记录保留作历史。详细报告 `docs/stage3-transition-aware-primary-replication-results-2026-10-03.md`。

用户已授权继续稳定性复核。本地实现、测试和独立审查已完成。按用户给出的SSH命令重试后，学校服务器成功连接。最终部署包和主模块的服务器SHA256与本地匹配，服务器50项相关测试全部通过18.14秒。

新折流程于2026-10-03 **22:31:57(+0800)** 启动，launcher PID7014，先生成fit/dev冻结模型语义缓存，再执行预注册的两条件训练。fit共有446个语义窗口，启动监控时缓存生成正常。原生产代码和Stage2模型哈希保持不变。

部署和启动时连接已恢复；随后SSH监控会话再次重置，新连接超时。最新Tailscale状态明确显示跳板机100.83.206.93离线、最后在线约1分钟前。已请用户保持跳板电脑和Tailscale在线。

流程用nohup独立后台启动，不会因SSH掉线而自动重启。当时断线前核实的是fit缓存289/446、worker7017正常工作，后续状态待重连核查。此次重连已确认正常结束并读回检查点；未重复启动训练。

## 已固定的实验

新折固定held_subject_03、seed3407，使用subject02/04的set000/001拟合、set002评估。subject03和set003数值以及validation/test均不读取。仍从原Stage2权重开始，转换目标age6/7与四个年龄/类别桶各25%质量保持一致，每步重新拟合FP64分类头。

冻结对照加关系系数1/10，各联合条件最多20次全批次Projection更新，检查点0/5/10/20；原任务指标与语义门槛、分类头预算均保持不变。原模型和第一轮实验结果不覆盖。详细预注册方案：`docs/stage3-transition-aware-primary-replication-2026-10-03.md`。

原缓存只有subject02，新折需用已验证的冻结Stage2缓存生成函数生成fit/dev语义缓存。每recording/class最多4窗口、14类、42模板，源manifest/config/checkpoint和上游源码/依赖哈希都要通过验证。

## 种子的解释已修正

目前使用固定预训练Projection、零初始化分类头、全批次优化，没有dropout或随机采样。合成输入的两个种子1701/3407已经真实运行，完整模型张量、AdamW状态和成绩逐项完全一致。因此更换这个种子不能形成独立随机重复；本轮实质检验是更换数据折。没有人为增加随机扰动或把相同权重当作多次独立证据。

不同折的fit/dev受试者仍有重叠，Stage2上游也训练过TRAIN subjects；即使新折成功，仍是方法在变换fit/dev划分下的开发集复核，不是独立留出测试或完整未见受试者泛化证据。

## 实现和验证

仅为 `parrad_projection_primary.run` 增加held/seed/spec/cache_parent参数，原默认值保留，训练算法不变；manifest和warmup身份记录实际参数。新增非法fold提前拒绝和不同恢复参数提前拒绝保护。

独立审查发现恢复参数失配会在拒绝前覆写原manifest。两个真实入口回归先复现失败，再修复为任何写入及数值checkpoint加载之前比较provenance和settings；测试验证原manifest字节不变。复审通过，无剩余重要问题。

相关回归48项通过77.94秒（修复恢复保护前）；最终新增复核5项通过4.27秒；最终全部可用本地回归 **404项通过、2项跳过**，80.23秒。既有h5py缺失的转换测试仍排除，未安装依赖。最终服务器相关50项测试通过18.14秒；真实效果待运行结束验证。

## 恢复入口

最终部署包 `.audit-staging/projection-primary-replication-v3.tar.gz`，新独立服务器代码目录预定 `/home/xibeidaxue/SEER/.audit-staging/projection-primary-replication-code-v1`。原primary-code-v1目录作为已验证基础复制，不修改原目录。

部署包SHA256 `fb49017f73469eca56365340b44a3354d5083b808d548391de6743d9ffda4506`；主模块SHA256 `bd1a0a9dbf76ae520c7dffd12952a79458aa694c2718185c1a156fa6d5fb802f`。上传后必须与服务器值匹配。

启动脚本 `.audit-staging/run-projection-primary-replication-v1.sh` 会先准备角色正确的语义缓存，然后执行新折两条件；必须显式传入经过实时检查可用的设备，例如cuda:0。上一次成功连接时GPU0 RTX2080Ti空闲；恢复连接后应重新检查，不能依赖旧快照。

预定新输出为服务器 `.../stage3_transition_sequence_v2/transition-aware-semantic-primary-replication-v1`。启动前先核对部署包及代码哈希、确认输出不存在、执行脚本语法检查和服务器相关测试；此前所有原实验保持原样。启动后须检查缓存/头收敛、实际P步数、固定检查点全部门槛、源模型完整性，再下载核验结果。

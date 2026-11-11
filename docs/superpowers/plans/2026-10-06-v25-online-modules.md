# v2.5 在线模块补全

Goal: 按用户书面方案补齐可运行、可训练、可校准的在线模块；保留旧控制器与旧权重作为对照。

Architecture: 独立 continuous_mmwave.v25 路径。age-free gate / semantic / physical evidence → Candidate episode → leaky sequential evidence → Confirm → boundary localization → provisional Active → 异步 replay/catch-up/atomic swap；Commit 与分层记忆单独异步队列。复用现有编码器、EventFormer、P、Query Compressor和QA接口，不自动训练或替换已部署权重。

Tech Stack: PyTorch、标准库 futures、pytest（interview conda）。

Spec: 计划/方案书v2.5 (2).md，特别是第16–28章及P0–P5。用户本轮“按照方案补全模块”批准按该现有书面方案实施。

Global Constraints:
- age只用于确定性warmup/timeout及诊断，不进入可学习gate/difference。
- 新模型/检查点显式v2.5标记；旧Verify检查点不能静默载入。
- Confirm/Reject不依赖GT；监督标签只由离线episode生命周期生成。
- 异步结果须检查generation和watermark；拒绝不重播，恢复持续更新的Shadow。
- 不把代码测试成功称为模型效果、实时性或独立泛化达标。

Review Focus: 不规则时间戳、mask无交集、超时高difference、慢/失败worker、重复reset、连续确认、边界前移后移、追赶过程中新增update、内存提交顺序、EOF尾段、新旧checkpoint混用、泄漏GT及训练oracle/closed-loop混合。

## Task 1: 可学习模块与顺序证据
- [x] RED: gate输入无age、tanh residual、513维difference、物理投影监督、公式/warmup/timeout测试。
- [x] GREEN: models / evidence / episode校准与bias probe、独立训练loss接口。
- [x] Verify: pytest tests/test_v25_evidence.py。
Expected: 全通过；梯度进入新模块，Verify不存在。

## Task 2: 在线分支、边界与异步重建
- [x] RED: reject=shadow、refined origin、后台慢任务不阻塞、同watermark替换、stale/reset/failure/pin清理测试。
- [x] GREEN: ring buffer、localizer、v25 controller、encoder-once runtime。
- [x] Verify: pytest tests/test_v25_controller.py。
Expected: 全通过；每条被接受观测均能在重建分支核对。

## Task 3: Commit/Memory/QA及显式checkpoint集成
- [x] RED: 有序异步finalization、当前Active检索context、EOF与旧检查点拒载、真实TinyLLM整链接口测试。
- [x] GREEN: pipeline、versioned runtime factory、配置/CLI与使用说明。
- [x] Verify: 新测试 + 相关旧controller/event/memory测试，86 passed, 1 skipped。
- [ ] 独立审查完整结束：reviewer额度中断；已复现修复反馈，详细记录见progress。本地验证完成，不能声称完整独立审查通过。
Expected: 新旧回归通过；报告仅宣称模块运行及接口验证，列出训练/校准尚未执行。

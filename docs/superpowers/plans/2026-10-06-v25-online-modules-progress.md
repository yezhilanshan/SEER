# SDD ledger — plan: docs/superpowers/plans/2026-10-06-v25-online-modules.md

Task 1: in_progress.
Task 1: complete — tests/test_v25_evidence.py RED(import missing)→GREEN 12 passed. 新gate/difference/radar/evidence、独立oracle loss、age配平/probe、episode label及校准函数已实现。
Task 2: in_progress.
Task 2: complete — 核心11项→补异步定位测试12项；阻塞localizer测试RED(Confirm等待)→GREEN。Boundary FIFO依赖前一refined boundary，后台rebuild分批追赶，连续Confirm只取消过期state swap、不取消旧Commit边界任务。
Task 3: in_progress — Stage2 gate/evidence迁移、原P保留loss、Stage3 physical/semantic/evidence固定步数训练、SHA/version bundle、factory、encoder-once runtime、异步Commit/Memory与独立QA接口及CLI已加入。

Verification before final review: 旧控制器/Memory/EventFormer与新模块组合73 passed, 1 skipped；之后异步定位重构的新模块23项(controller+pipeline)已通过。需要重新运行全部相关测试。

Task 3: complete — 全部相关新旧回归86 passed, 1 skipped（2026-10-06）；命令：interview/python -m pytest tests/test_v25_evidence.py tests/test_v25_controller.py tests/test_v25_pipeline.py tests/test_parrad_controller.py tests/test_stage4_memory.py tests/test_phase1.py -q。新模块46项。最终烟测outputs/v25-module-smoke-final-2026-10-06，40 updates/40 encoder calls，13 confirmed＋1 EOF unconfirmed，Memory三层各2，失败0，最终Active已atomic swap。

Final review: 一次独立reviewer只读审查，中途因模型额度限制退出，未产出完整severity/declined-to-judge报告，不能称完整审查通过。其静态反馈指出6类正确性问题，作者已独立复现并进行一轮TDD修复；没有新增reviewer。

Final: fixed FP16/BF16 evidence和QA输入dtype — test_native_event_dtype_is_accepted_by_fp32_evidence_head / test_native_half_memory_context_passes_generation_adapter_without_mutation RED→GREEN。
Final: fixed drained Commit丢失及失败raw payload丢失 — test_controller_jobs_are_retained_when_commit_queue_is_full / test_failed_commit_retains_radar_payload_for_readback RED→GREEN；Pipeline保留job直到成功提交。
Final: fixed archive pin释放滞后 — test_archive_pin_is_released_without_another_stream_update RED→GREEN。
Final: fixed boundary失败后的FIFO污染继续消费观测 — test_boundary_failure_stops_before_consuming_more_observations RED→GREEN；显式停止，reset/recovery再继续。
Final: fixed训练前/过期bias probe — test_post_training_probe_is_bound_to_current_frontend_and_head RED→GREEN；训练后dev独立测量，无人为默认pass门槛。
Final: fixed status绕过及校准来源验证缺失 — test_calibrated_status_cannot_bypass_readiness_validation / test_calibration_requires_passed_current_probe RED→GREEN；test_calibration_and_checkpoint_ready_path_and_stale_source验证正向及过期拒绝。Final suite 86 passed, 1 skipped。

Deferred minors: 无完整reviewer报告，未收到明确minor项。实际训练、episode数据重建、原生连续回放、实时性和独立泛化为后续验收，不在本轮代码验证中伪造完成。

Pre-flight interfaces: Task1 produces age-free heads/evidence configuration; Task2 consumes logits and projected evidence, not maturity or Verify. Task2 produces finalization intervals and generation/watermark; Task3 consumes them in chronological order. ModelConfig/old controllers remain unchanged.

Ruling: 当前目录没有Git，技能的git/worktree/commit脚本不能使用；保留此持久ledger和独立测试输出，不删除工作记录。代价：没有本轮Git提交回滚。

Ruling: 已有用户书面v2.5方案和本轮实施授权，不重复设计审批；PARrad沿用896维Qwen及实际update时间，128维语义空间仍按方案。代价：不是原方案3584维LLM实验。

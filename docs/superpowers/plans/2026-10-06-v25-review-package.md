# v2.5 最终只读审查包

No Git checkout. All implementation changes are new files in src/continuous_mmwave/v25 and tests/test_v25_{evidence,controller,pipeline}.py; legacy production Python source is unchanged.

Plan: docs/superpowers/plans/2026-10-06-v25-online-modules.md
Spec: 计划/方案书v2.5 (2).md
Ledger: docs/superpowers/plans/2026-10-06-v25-online-modules-progress.md
Usage: docs/v2.5-online-modules-2026-10-06.md

Review Focus: 不规则时间戳、mask无交集、超时高difference、慢/失败worker、重复reset、连续确认、边界前移后移、追赶过程中新增update、内存提交顺序、EOF尾段、新旧checkpoint混用、泄漏GT及训练oracle/closed-loop混合。额外检查半精度Qwen tokens与FP32新头的dtype边界、异步队列溢出/失败的证据保留，以及训练API与runtime的默认配置一致性。

All claims restricted to code and interface operation. No actual native-model training/calibration, no unseen-subject or realtime success claim. One final reviewer only; no new subagents by the reviewer.

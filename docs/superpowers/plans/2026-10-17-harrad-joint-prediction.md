# HARrad Joint Prediction Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. User has authorized the A/B/AB work; implement and verify without repeated approval requests.

**Goal:** 实现并运行HARrad Dynamics/语义的固定预算可比预训练。

**Architecture:** 冻结CNN cache→可训练共享TokenAdapter→GRU Dynamics与Semantic EventFormer；固定CNN64目标，统一因果Oracle前缀，M/A/B/AB四臂三seed。

**Tech Stack:** 现有PyTorch/numpy，interview本地pytest、seer原生CUDA；不安装依赖。无Git，使用新目录源码包与SHA。

**Spec:** `docs/superpowers/specs/2026-10-17-harrad-joint-prediction-design.md`。

## Constraints / review focus

- 以Spec为数值契约：h1/3/5，2/15秒，m64，prefix≤32、grad≤16，12000成功更新，两个固定512样本validation面板，test不读取。
- 不从未来边界/标签决定当前输入或删除困难预测；同类跨实例mixed也不得有语义CE。
- Qwen冻结但B输入可导；无梯度burn-in不得通过AMP缓存切断后缀梯度。
- 三臂共享组件初值及样本表一致；目标不随共享训练改变；zero残差首步上游梯度零不误判。
- 面板指标不是全量、Oracle不是自主Reset；旧77.01%不可直接与本轮语义面板比较。

## Tasks

- [x] RED/GREEN：`joint_data.py`连续Update/冻结目标/合法起点/Oracle前缀/混合mask/固定采样与面板。
- [x] RED/GREEN：`joint_model.py`共享Adapter、D/E、因果前向、TBPTT、残差头、分任务梯度与白名单。
- [x] RED/GREEN：`joint_training.py`固定预算/AMP成功计数/面板评价/重放与归约。
- [x] `joint_runner.py`源锁、缓存、配置/源码胶囊、原生与真实Qwen资源预检；独立整体审查。
- [x] 启动持久的4臂×3seed×1000更新，保留日志/checkpoint/逐样本预测。
- [x] 完成后重放、独立复算、下载证据及三seed比较；若后台尚在运行明确报告实际进度，不宣称训练结束。

## Verification ledger

- v1源码胶囊仅完成原生8项测试及真实Qwen预检，没有启动正式训练，原目录保留。
- 独立审查发现P2统计问题：已知混合Update被归入unknown_future。修复为独立unknown/mixed掩码，新增mixed_future，独立审计检查源掩码与分组指标。
- 新回归用例先失败后通过，本地9项测试通过；v2封存修复后的源码并重新进行原生预检。科学协议及固定样本表不变。
- v2原生9项测试、真实Qwen预检、本地固定索引/目标统计核验及独立修复复核通过；正式后台PID4761，输出`outputs/joint_prediction_20261017_v2`。后续状态和验收见`docs/harrad-joint-prediction-progress-2026-10-17.md`。
- 2026-10-18 04:59全部12组完成，共12000次成功更新/尝试；最终预测重放均0。120任务文件及9元数据SHA、48检查点读回、12组本地独立复算、共同初值、原实例mask、180源码/70缓存完整性核验通过。h5 A30.91%略高于M29.64%，AB28.36%；语义B60.43%、AB58.80%。联合净收益未显示，不自动进入Reset；完整记录`docs/harrad-joint-prediction-results-2026-10-18.md`。

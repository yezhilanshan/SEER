# PSEM v2.6 P0/P1 Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Recommended execution method: native implementation in this chat, after human review; no automatic subagent dispatch.

**Goal:** 建立冻结旧前端的连续未来目标导出和预测价值对照，首先交付真实标签审计与命令边界代理的预测诊断，不启动Keep/Birth或Reset训练。

**Architecture:** 新增独立`continuous_mmwave.v26`工具，严格加载配套Stage1+Stage2+Qwen基础模型，保持旧权重不变。冻结Encoder/EventFormer/Reader，从F导出未来目标与因果Event特征，只训练预测对照头。数据与权重SHA、缓存有效性、零步一致性先通过，才启动固定预算pilot。

**Tech Stack:** Python3.10、PyTorch、NumPy、PyYAML、pytest；本机interview用于测试，服务器现有seer用于CUDA推理/预测头训练；不安装依赖。

**Spec:** `计划/方案书v2.6.md`第十五/十六/十八章，`docs/v26-p0-readiness-2026-10-14.md`。

## Global Constraints

- 保留原causal MD-GRU，`frame_period_s=0.09, frames_per_update=2, radar_tokens=4, radar_dim=256, event_tokens=8, llm_dim=896`。
- Stage1 SHA=`42065f59311cb41e6c23c7734b66392028d634d7c11264bfa3325cb2be7b7289`；Stage2 SHA=`939aa977b2616d8ceb11f47ec1470701dc04bd7d1af64cc5ac20ebed3ab88f9a`。
- 固定目标分支；future target不参与当前输入；不跨recording/缺失断点；每recording/horizon归一，日志保存合法数。
- 保持冻结旧Encoder/EventFormer/semantic projection/LLM；旧代码与输出不覆盖；无Git时保存源码SHA与新输出目录。
- 本轮真实实例G0未通过。默认候选运行名为`command_boundary_proxy_prediction_diagnostic`，`formal_go=false`；必须确认代理实验范围后开训，不能自动进入Stage2/3。
- test雷达数组保持关闭；validation只用于预注册pilot检查与以后固定协议评价，不根据分数反复改架构、窗口、权重或追加预算。

## Review Focus

- manifest相对路径必须按manifest父目录解析，两个同步雷达不能跨角色；Task1固定测试。
- 旧trainable-only Stage2遗漏编码器时必须配套加载Stage1，实际缺失的非冻结权重不得静默接受；Task1固定测试。
- 边界位于update中间、未知标签或奇数尾帧时不能生成跨断点未来目标；Task2固定测试。
- EventFormer maturity只取已经观察的age，不能用完整GT事件终点；Task2固定未来扰动测试。
- Event臂与基线的信息/历史预算不同、NLL通过方差作弊或重复缓存seed被当独立实验；Task3/4固定验证。

## 冻结的预实验提案

- 数据根目录：`/home/xibeidaxue/newdisk2/WJ/PARrad/canonical/offline_expanded_20261011_split_v1`。
- pilot fit：每个train受试者按session_id排序选择第一个完整session，保留两种传感器记录；预期18份。审计发现subject09/set000仅有68xx，按元数据完整性选择set001并记录，所有臂使用相同子集；不存在完整session的受试者停止预检，不按分数选session。
- pilot validation：05/12/16各按相同元数据规则取第一个完整session、两传感器，预期6份。仅在协议/config SHA保存后访问其数组。
- 从原CSV行号形成`command_episode_id`，不合并或重命名为真实实例。unknown/mixed update保留mask；不声称检测到真实动作起点。
- 严格保持编码器跨完整recording更新；事件状态在命令代理边界重置，maturity=`min(observed_episode_updates*0.18/1.2,1)`；训练和评价均不提供总事件长度。
- 固定目标`pool(F)`为四个有效token的masked mean，再用fit-only逐维均值/标准差标准化，std下界`1e-4`；目标映射和统计保存SHA。
- h=1，即完整update网格0.18秒。第一版不增加多horizon、属性、LoRA或Reset损失。
- 公平短前缀视图仅使用命令episode前32个合法update；Event/GRU从相同代理边界开始，TCN感受野至少32。未来目标只由真实观测合法性决定，不依模型边界筛选；所有臂共享完全相同的时间索引。
- 预测臂：Persistence（均值复制当前m，方差只从fit残差估计）、event-free MLP、causal GRU、causal TCN、Event-conditioned head。GRU/TCN读取完整flattened F（1024维），不能只读取m均值。无Event State臂没有E；Event臂读冻结E与当前m。
- 可训练参数预算拟定400000、容差10%；在任何validation分数产生前，仅按参数数量从确定的hidden-width网格选择最近合法配置，不增加未使用的假参数。不同时匹配时报告差值、标记capacity-unmatched，不宣称公平收益；冻结前端总参数与FLOPs另报。
- 预测头使用对角Gaussian，logvar范围`[-6,2]`，优化每维平均NLL，MSE独立报告。
- 初始pilot：seed42、每神经预测臂300次成功AdamW更新、batch8、LR`1e-3`、weight_decay`0.01`、clip_norm`1.0`；同recording/time采样清单，各臂使用同采样顺序。超参数未由本项目实验证明最优。
- 分层采样按recording均匀、再合法位置均匀；GRU/TCN按因果前缀计算，不把跨边界片段随机拼接。缓存只保存无梯度特征，训练头每次保持所需prefix图。
- pilot以固定末步权重评价，不择优选择checkpoint。所有臂成功后读取validation一次；P0/数值失败立即停止，不追加步数。pilot只检查可训练、可回放、方差合法与有无预测信号，不据单seed授予正式G1。
- 若继续做正式Stage1，需要真实G0通过与新预算预注册；拟定G1关键效应为Event臂对GRU/TCN两者均有至少5%受试者宏MSE下降、NLL不劣，且3个独立训练seed中至少2个同方向。此门槛需审阅后锁定，不能把proxy结果当正式通过。

## Task 1: 来源、划分与冻结前端加载

**Files:** Create `src/continuous_mmwave/v26/__init__.py`, `source.py`; test `tests/test_v26_source.py`。

**Interfaces:** `resolve_recording(manifest: Path, row: dict) -> Path`; `audit_sources(config: dict) -> dict`; `load_frozen_frontend(config_path: Path, stage1_path: Path, stage2_path: Path, expected_shas: dict, device: str) -> tuple`，返回frontend/tokenizer/source_report。复用既有构建/严格加载方法，不调用会自动迁移gate的v25 factory。

- [ ] RED：相对路径正确解析；split有同subject或同session双雷达跨角色时拒绝；SHA不符拒绝；Stage2省略冻结encoder时配套Stage1成功，缺语义投影时失败。
- [ ] 用interview运行`tests/test_v26_source.py`，确认新增接口未实现导致预期失败。
- [ ] 实现接口；锁定`trainable_only`结构和全部fitted非encoder键，冻结模型/eval；保存四核心源码与Qwen本地权重索引SHA，验证legacy gate类型。
- [ ] GREEN：上述测试及现有`test_v25_training_source.py`通过；真实服务器严格加载、固定短训练前缀零步回放两次输出一致，保护面SHA不变。
- [ ] 保存源码快照/清单；无Git时不执行commit或创建worktree。

## Task 2: 连续缓存、命令代理与未来目标

**Files:** Create `v26/targets.py`, `v26/features.py`; test `tests/test_v26_targets.py`, `tests/test_v26_features.py`。

**Interfaces:** `build_command_episodes(row: dict, frame_timestamps: np.ndarray) -> list[dict]`; `future_indices(update_intervals: np.ndarray, valid: np.ndarray, horizons: tuple[int,...]) -> dict[int,np.ndarray]`; `fixed_motion_target(tokens: Tensor, mask: Tensor) -> Tensor`; `export_prediction_cache(frontend, manifest_path: Path, rows: list[dict], output_dir: Path, config: dict) -> dict`。缓存含F/m*/E、start/end/available_at、episode_id、age、valid、来源SHA。

- [ ] RED：2帧/update、h1的6个完整update产生5个合法未来对；跨recording/gap不配对；7帧只导出3个完整update并记录1个尾帧；连续相同命令保留两个episode ID；unknown/mixed update不成为伪确定边界。
- [ ] RED：将未来雷达或未来episode结束时刻改变，不改变当前F/E/预测输入；fit统计不读取validation/test；目标与所有旧参数无梯度。
- [ ] 运行两份新测试，检查预期失败，再实现：Encoder全录制stateful处理、代理边界的因果event缓存、fit-only归一化；不修改canonical数据；输出目录必须新建。
- [ ] GREEN并完成pilot fit输入MD有限性检查；核对cached F与直接prefix回放一致、E固定前缀回放一致、保护面SHA保持。
- [ ] 保存cache/index/config/目标映射SHA，合法与尾帧数量清单；完成前不训练。

## Task 3: 预测对照与损失

**Files:** Create `v26/predictors.py`, `v26/losses.py`; test `tests/test_v26_predictors.py`, `tests/test_v26_prediction_losses.py`。

**Interfaces:** `make_predictor(kind: str, config: dict) -> nn.Module`，接口`forward(motion: Tensor, radar_prefix: Tensor, event_tokens: Tensor|None, prefix_mask: Tensor) -> tuple[Tensor,Tensor]`；`gaussian_nll(mean: Tensor, logvar: Tensor, target: Tensor) -> Tensor`返回每位置每维均值；`reduce_prediction_losses(losses: Tensor, recording_ids: Tensor, valid: Tensor) -> Tensor`按recording归一；`fit_persistence_variance(fit_current: Tensor, fit_future: Tensor) -> Tensor`。

- [ ] RED：mean=target=0/logvar=0时每维NLL=`0.5*log(2*pi)`；mean误差1时增加0.5；不等recording长度不改变其平均权重；无有效项跳过并阻止空优化步。
- [ ] RED：未来扰动不影响之前预测；GRU/TCN只读完整相同F前缀；E变化可影响Event臂，而不影响event-free臂；logvar始终在[-6,2]，输出和梯度有限。
- [ ] RED：参数计数选择仅使用结构与固定网格；同label/episode不传入预测head；Persistence方差仅使用fit；不能用dummy参数充预算。
- [ ] 运行RED，再实现最小预测头与损失；参数统计同时报告新增可训练与冻结基础模型，matched范围不足时报未匹配。
- [ ] GREEN并运行全部v26新测试，完成合成训练信号probe；合成结果只验证实现。

## Task 4: 固定预算pilot与独立读回

**Files:** Create `v26/prediction_runner.py`, `v26/__main__.py`, `configs/parrad-v26-prediction-pilot-v1.yaml`; test `tests/test_v26_prediction_runner.py`。结果报告实际交付为`docs/v26-prediction-pilot-results-2026-10-15.md`（跨日完成），使用新服务器输出，不覆盖现有文件。

**Interfaces:** `run_prediction_pilot(config_path: Path, mode: str) -> dict`，mode=`audit/export/smoke/train/evaluate/verify`；`evaluate_predictions(prediction_rows: list[dict], source: dict) -> dict`返回NLL/MSE/variance与按recording/session/subject分组；CLI `python -m continuous_mmwave.v26 --config ... --mode ...`。

- [ ] RED：固定300成功更新、不由validation择优、无有效loss不记成功step；config/source/cache SHA不符或目录已存在时拒绝新启动；test访问保持关闭。
- [ ] RED：同seed同采样清单可重放；若以后多seed，实际重新训练head，不把同一冻结特征换seed当独立实验；验证按subject聚合，两个雷达不当独立受试者。
- [ ] 运行RED并实现runner/配置，记录源与目标哈希、随机状态、采样列表、optimizer步数、各臂参数与梯度；保存固定末步检查点、prediction arrays和loss日志。
- [ ] GREEN；确认技能要求的计划审阅与proxy运行范围后，在独立源码capsule用现有seer环境做1步CUDA烟测，profile100次冻结update的P50/P95/显存。
- [ ] 只有smoke/冻结保护/目标一致性全部通过才启动约定pilot；异常保留日志并停止，不改变全局环境或其他任务。
- [ ] 保存所有臂终态后执行validation一次，重新加载权重回放predictions、独立重算NLL/MSE与subject均值；与保存结果一致。
- [ ] 交付diagnostic-only报告，分别列05历史验证与12/16视图；保留test关闭与正式G0未通过状态，不自动训练Reset。

## 当前执行状态与交接

用户已授权在当前对话执行命令边界代理diagnostic。2026-10-15已完成新增v26产品入口、18fit/6validation缓存、四臂各1步烟测及300步训练、五臂固定末步验证和30份预测的独立复算/权重回放；本机下载65份证据文件哈希通过。test数组未读取，预算未追加。详细结果见[交付报告](../../v26-prediction-pilot-results-2026-10-15.md)。

| 任务 | 实际交付与边界 |
|---|---|
| Task1 | 来源/资产锁、严格配套加载、冻结/划分检查完成；正式实例与物理时钟G0仍未通过 |
| Task2 | 连续缓存、代理边界、fit-only目标统计和24份初始32步事件参考完成；独立episode_id字段及全录制逐步Encoder参考延期 |
| Task3 | 五臂、损失、约40万新增参数匹配完成；全前端实际参数另报，完整FLOPs未测 |
| Task4 | 固定预算训练、一次评价、独立读回及下载核验完成；完整在线P50/P95/峰值显存等扩展验收延期 |

上方复合清单保留为最初计划，不将含延期项目的条目整体勾为已验收。新鲜审阅三项Important已修复，其余Minor及无Git快照/数值加速裁定详见结果报告与`.superpowers/sdd/2026-10-14-v26-p0-p1-prediction/progress.md`。本机和服务器各35项相关测试通过，本机完整套件因h5py缺失未能通过收集；补充全套运行中断，没有完整全套通过结论。

本轮Event宏MSE比GRU/TCN分别高8.82%/7.75%，Persistence仍最好，没有获得事件额外预测收益的正向证据。后续先调查预测头/短时目标，真实实例与时钟验收另列数据工作；本轮不启动Reset或自动扩展seed/预算。

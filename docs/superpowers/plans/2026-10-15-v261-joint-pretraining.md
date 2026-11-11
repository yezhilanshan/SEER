# v2.6.1 Joint Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现可核验的共享 Adapter 多任务对照，回答运动与语义监督是否改善共同表示。

**Architecture:** 冻结 MD-GRU 参考 F，经一个可训练共享 Adapter 输入轻量 GRU 与冻结 Event Former。分开构造 M、A-only、B-only、A+B，同样本同预算训练，以原生任务和统一线性探针评价。数据/语义教师/目标被冻结，更新状态保留输入梯度。

**Tech Stack:** Python3.10、PyTorch、NumPy、PyYAML、现有 Qwen 后端、pytest；不新增依赖。

**Spec:** `docs/superpowers/specs/2026-10-15-v261-joint-pretraining-design.md`

## Global Constraints

- 本轮状态：设计与输入清点完成，产品模块和训练待实现；不得将文档完成当成训练通过。
- 复用现有 18 fit/6 validation，不读取 test 数组；标签仍为 `command_class_proxy`。
- F[4,256]、E[8,896]、共享 Adapter 256→64→256、GRU256；新增上下文最多32 Update，后16步有梯度；maturity=`min(age*.18/1.2,1)`。
- 冻结主干/Former/教师/目标，B 对 Z 的梯度必须存在；首轮不训练 LoRA、Reset、Caption、Memory。
- 3 个 Horizon 权重 .1/.3/.6，主指标 h5；主训练4×3×1000，独立 smoke≤20，探针6000更新。完整损失、采样与归约以 Spec 第5–6节为准。
- 阶段实现失败应留存证据；不得缩短上下文或更改预算后覆盖原目录。
- 本地测试使用 `C:/User/anaconda/envs/interview/python.exe`；原生使用 `/home/xibeidaxue/newdisk2/WJ/.seer-conda/envs/seer/bin/python`。
- 无 Git checkout；每任务保存本地修改清单，原生执行使用新增源码胶囊与 SHA，不修改旧实验。

## Review Focus

1. 混合标签或类别 ID 错位：屏蔽语义监督，保留合法预测，映射不一致则拒绝准备；Task1测试。
2. 学生跨 optimizer step 复用旧 E/D 或未来上下文：每样本重算32步可见前缀，后缀扰动不影响当前；Task2测试。
3. 冻结 Former 时误用 no_grad、零残差首步误判、checkpoint cache 污染：检查实际输入梯度、更新后梯度与独立缓存；Task2测试。
4. 多臂可训练参数或初始化不同、AMP 溢出虚记 step：分组白名单、组件 SHA 与真实成功步；Task3测试。
5. 预测/分类总体、宏平均或探针泄漏不同：保存逐点数组并独立重算，目标/索引被篡改须失败；Task4测试。

## File structure

- 新建 `src/continuous_mmwave/v261/{__init__,contract,model,training,evaluation,runner}.py`，依次负责版本入口、数据协议、模型/梯度边界、训练/损失、预测与指标、一次性执行状态机。
- 新建 `tests/test_v261_{contract,model,training,evaluation,runner}.py`。
- 新建 `configs/parrad-v261-joint-pilot-v1.yaml`、`docs/v261-joint-preparation-results-2026-10-15.md`；修改 README 的当前状态入口。
- 已有只读准备证据在 `.superpowers/sdd/2026-10-15-v261-preparation/`；正式软件不要反向依赖该目录中的临时工具。

## Task 1：训练数据契约与不可变目标

**Files:** 新建 `v261/contract.py`、`v261/__init__.py`、`tests/test_v261_contract.py`。

**Interfaces:**

- `JointRecord`：uid/role/subject、`radar[T,4,256]`、`motion_ref[T,256]`、`intervals[T,2]`、`observation_valid[T]`、`semantic_target[T]`(-100忽略)、`common_start[N]`、`source_lock:dict`。
- `build_joint_record(cache:dict, labels:Tensor, physical:dict, role:str)->JointRecord`。
- `prefix_bounds(record:JointRecord,t:int,max_updates:int=32)->tuple[int,int]`，闭起点、开终点。
- `joint_schedule(records:list[JointRecord],seed:int,steps:int=1000)->list[list[tuple[int,int]]]`；坐标为 recording index 与实际 Update index。
- `fit_motion_contract(records:list[JointRecord],normalization:dict)->dict`：复用原 mean/std，输出 R_h 与公共 logvar_h；拒绝非 train 统计来源。

- [ ] 写失败测试：两帧不同类、未知类、mixed、缺测、短前缀、跨角色或 source SHA 不符、类别映射错位；断言仅语义 mask 改变时合法 future 和 common_start 不改变。
- [ ] 用 interview 运行该测试文件，确认因缺失模块/函数失败。
- [ ] 实现上述数据结构/接口；从 NPZ 读取 labels，按原 manifest 解析相对路径，复核 source/cache/command/label_map/normalization SHA；不读取 test 数组。
- [ ] 测试 train-only R_h/logvar、三个 h 的共同合法索引、32步窗口不跨观测断点；sampler每批首样本语义合法、两臂同 seed 表相同、输入列表顺序显式固定。
- [ ] 保存24录制来源、物理/语义样本和各类计数；与本次库存中的61,299/20,075、59,540/19,609对照；差异必须解释后另版锁定。

## Task 2：共享 Adapter 与双状态因果前向

**Files:** 新建 `v261/model.py`、`tests/test_v261_model.py`。

**Interfaces:**

- `SharedRadarAdapter(dim:int=256,bottleneck:int=64)`，`forward(F:Tensor)->Tensor`。
- `JointPilotModel(frontend:nn.Module,arm:str)`；`forward_prefix(F:Tensor,mask:Tensor,m_ref:Tensor)->dict[str,Tensor]`，F为[B,L,4,256]，返回存在的 mean[B,3,256]、logits[B,14]、shared[B,256]、E/D末态；不返回不存在的任务结果。
- `trainable_manifest(model:nn.Module)->dict`：组件名、shape、requires_grad、参数数量与初值SHA。

- [ ] 写失败测试：Adapter 初值 exact 恒等、M/A/AB 三头初值 exact Persistence、B/AB 零步语义与同前缀教师一致；按 arm 验证参数白名单。
- [ ] 先运行 model 测试确认失败，再实现 Adapter、GRU/MLP、语义读出及独立冻结教师；保持旧类和源码不变。
- [ ] 实现相同的上下文初始化、因果 maturity、最多16步无梯度 burn-in 与最多16步有梯度尾部；仅 detach burn-in 状态，不切断尾部 B 对 Z 的导数。
- [ ] 测试未来 F/标签后缀和 padding 扰动不改变当前输出；长/短 batch 与逐前缀推理等价；模型 A/AB 共同组件初值 exact；每次读取 prompt cache 不污染其他样本。
- [ ] 在 tiny 后端测试语义损失对 Adapter 梯度非零；预测头经历有效更新后预测损失对 Adapter 梯度非零；教师/旧权重无梯度且不变；检查 `checkpoint(use_reentrant=False)` 与普通 backward 的梯度等价。

## Task 3：损失、固定样本表与训练核验

**Files:** 新建 `v261/training.py`、`tests/test_v261_training.py`、`configs/parrad-v261-joint-pilot-v1.yaml`。

**Interfaces:**

- `joint_losses(outputs:dict,teacher:dict,batch:dict,fit_stats:dict,arm:str)->dict[str,Tensor]`，显式返回 total/motion/semantic/preserve 及合法数量。
- `train_joint_job(model:nn.Module,records:list[JointRecord],stats:dict,schedule:list,config:dict,path:Path)->dict`。
- `gradient_diagnostic(model:nn.Module,losses:dict)->dict`，分别记录共享 Adapter 的预测/语义梯度范数与余弦；零范数标N/A，不生成NaN、不改变主训练梯度。

- [ ] 写失败测试：逐录制归约、不同 h 的 R_h、温度1 KL、忽略类、teacher stop-grad、空合法任务不伪造loss、完全空批不更新、A/B/AB损失与Spec一致。
- [ ] 运行失败测试，再按Spec实现权重与预算；B-only没有预测头，不因同一训练预算而偷偷优化额外预测器。
- [ ] 按组件 RNG 初始化并保存每 seed 共同采样表；所有臂独立optimizer，记录实际唯一anchor覆盖、任务样本数、gradient clip及学习率。
- [ ] 测试故意制造 AMP skipped step：成功计数不增加、attempt增加；1000成功步或1100次尝试停止。step0/100/300/1000保存参数、optimizer、RNG、源锁与表SHA。
- [ ] 测试 checkpoint 载荷不能仅靠目录名宣称步数；测试结束后所有冻结组件SHA不变；无效配置（未知 arm/seed/负R_h/改预算）必须拒绝。

## Task 4：公平评价、共享探针与独立复算

**Files:** 新建 `v261/evaluation.py`、`tests/test_v261_evaluation.py`。

**Interfaces:**

- `predict_joint_record(model:nn.Module,record:JointRecord,stats:dict)->dict[str,np.ndarray]`。
- `fit_shared_probes(features:dict,records:list[JointRecord],stats:dict,seed:int,steps:int=200)->dict`：固定表示，独立线性预测/分类探针，不更新 S_phi。
- `summarize_joint(predictions:list[dict])->dict`；`verify_joint_outputs(root:Path)->dict`。

- [ ] 写失败测试：逐人Skill再平均与宏MSE比值不同；14类 Macro-F1固定分母、空组N/A、mixed仅屏蔽语义；B-only原生预测为N/A。
- [ ] 实现同一 prefix_bounds 前向、三个H共同起点、跨代理边界/内部/混合/长度分组；公共固定方差NLL明确单列，不称作校准评估。
- [ ] 实现4模型×3seed和原始F×3seed的两类统一探针，每个200步batch64；预先固定fit采样表，evaluation不训练；保存探针前后表示SHA证明未更新。
- [ ] 在测试中篡改future/target/mask/mean/logits，核验应失败；读回同权重在原环境同形状重放，保存误差/精度设置，不能泛称CPU/CUDA bitwise相同。
- [ ] 用独立NumPy/混淆矩阵实现从逐点数组重算全部指标；输出逐seed逐人和seed均值/标准差，保留负结果与任务权衡。

## Task 5：一次性准备入口、原生资源检查与交付

**Files:** 新建 `v261/runner.py`、`tests/test_v261_runner.py`、准备结果文档；修改 README。

**Interfaces:** CLI `python -m continuous_mmwave.v261.runner --config <path> --mode audit|prepare|smoke|train|evaluate|verify`；每阶段有输入/代码/配置身份与完成标记，依赖前阶段成功。

- [ ] 写失败测试：重复prepare/已有输出拒绝覆盖；失败audit或smoke不能train；更换code/config/source/sample表必须失效；smoke主训练探针计数独立。
- [ ] 实现新增输出目录与不可变source ZIP，fit/validation证据随协议封存；old v26/旧模型不得修改。先完成本地相关新旧回归。
- [ ] 在seer运行真实Qwen smoke：四臂各最多5个成功步、batch2/最长32/尾部16梯度，独立seed1042。检查B输入梯度、冻结SHA、缓存隔离、非有限、AMP计数和allocated峰值≤9GiB。
- [ ] 输出实际forward/backward/eval吞吐与全轮成本估算，确认是否适合当前设备。预算或显存失败则保留失败证据并停止，不自动替换结构/上下文继续训练。
- [ ] 只有工程与资源检查全部通过、候选配置已锁定并处于用户授权的执行范围内，才执行12个主任务；不得把本次“准备设计”自动当作已启动正式训练。
- [ ] 主训练后执行探针、原生验证与本地独立复算；保存完整逐点结果、最终/中间checkpoint位置与SHA、预算完成情况和真实限制。README按“准备/检查/训练/核验”实际状态更新。

## Verification commands

本地工作目录 `D:/lab/SEER`，PowerShell：

```powershell
& 'C:/User/anaconda/envs/interview/python.exe' -c "import sys; sys.path.insert(0,'src'); import pytest; raise SystemExit(pytest.main(['-q','tests/test_v261_contract.py','tests/test_v261_model.py','tests/test_v261_training.py','tests/test_v261_evaluation.py','tests/test_v261_runner.py']))"
```

上述文件目前尚未创建；此命令是未来实现验收命令，不是本轮已通过的测试。

原生在新源码胶囊根目录，使用同一组测试及既有 `tests/test_v26_*.py` 做相关回归；不得只展示新测试通过而省略保护原路径。

## Self-review and current state

- [x] 对照Spec检查4臂、目标、监督、状态、损失、初始化、预算、探针、评价、资源与输出契约。
- [x] 五项Review Focus均有明确负责的任务和失败测试。
- [x] 区分旧Encoder已有历史与新增32步上下文；区分冻结Former参数与冻结输入梯度。
- [x] 区分主训练12000、探针6000、独立smoke≤20；未以预算代替实际完成计数。
- [ ] Tasks1–5的软件实现与原生检查；本轮未执行。

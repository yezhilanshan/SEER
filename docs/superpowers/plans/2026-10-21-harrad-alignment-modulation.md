# HARrad Alignment / Modulation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. The user explicitly authorized starting the proposed four-arm experiment; carry the implementation through preflight and persistent launch without repeated permission requests. One independent final review before formal launch.

**Goal:** 实现并启动可复核的HARrad语义对齐×调制2×2因果预训练实验。

**Architecture:** B-only语义路径，独立固定雷达前缀参考库，学生E投影对比；可选Eprev→当前Z的恒等初始化FiLM。A/CNN/Qwen冻结，所有臂相同B训练范围。

**Tech Stack:** 现有PyTorch/numpy，本地interview测试，远程seer CUDA。无依赖安装。

**Spec:** `docs/superpowers/specs/2026-10-21-harrad-alignment-modulation-design.md`。

## Global Constraints

- 三seed42/43/44，每臂1000成功更新；原训练表batch2、语义validation512，固定train语义512。test不读。
- 参考仅train，每类≤32；不同类负例，τ=.1、λ=.1；前缀≤32，反传≤16。
- 保留旧输出、源码及所有检查点；新am模块/胶囊，禁止覆盖。
- 当前目录没有Git，不创建虚假commit；用封存源码manifest/SHA和ledger代替Git记录。

## Review Focus

- 正例和参考库是否相同因果前缀；同类/未知是否变成负例或监督。
- 部分padding/短前缀/burn-in及重置首步是否错误读取旧/未来E。
- 参考冻结、Qwen冻结是否仍允许学生E梯度；四组B共同初值与采样一致。
- AMP跳步预算及保存/重载评价是否真实；独立核算分录制与短前缀指标。
- baseline B0是否被额外模块初始化消耗RNG或额外损失改变；实验不是完整DSA-Net复现。

### Task 1: Modules and causal semantic model

Files: new `am_modules.py`, `am_model.py`, `tests/test_harrad_am_model.py`。

- [x] RED：恒等FiLM、原B零步logits等价、正/负例手算、同类排除、未知mask、学生梯度及冻结参考。
- [x] GREEN：`EventModulation.forward(z,previous)`；`AnchoredContrastive.forward(query,positive,labels,bank,bank_labels)`；`AlignmentModel`复用JointModel接口及白名单。
- [x] 核验padding、可见前缀、梯度截断及fresh reset与原B一致。

### Task 2: Data, training and independent evaluation

Files: new `am_data.py`, `am_training.py`, `am_audit.py`, `tests/test_harrad_am_training.py`。

- [x] RED：train-only固定类覆盖bank、固定诊断面板、mask/前缀长度、微型训练成功计数/初值/检查点/重放。
- [x] GREEN：固定设计和负例库，CE+optionalCL；0/300/1000两角色面板输出与独立指标，final重放、frozen digest及关闭调制诊断。
- [x] 相关回归测试通过；本地全suite若有环境缺项明确记录，原生任务测试必须通过。

### Task 3: Production capsule, real preflight and launch

Files: new `am_runner.py`, `configs/harrad-alignment-modulation-v1.json`, deployment helpers under new outputs folder。

- [x] RED/GREEN：runner契约、预检配置/源码绑定；tiny训练入口、固定预算与重放通过；正式prepare-only后重新打开数据比较并启动。
- [x] 封存源码、原生测试、真实Qwen32上下文梯度/显存/初始语义预检。
- [x] 独立整体审查后修复有意义缺陷（RED→GREEN），保留所有失败证据。
- [x] 后台启动12×1000成功更新，确认实际成功步数，下载预检/索引/启动证据，更新README与进度文档。
- [x] 完成后全组权重重放、逐点独立复算和三seed2×2比较；后台未结束则如实报告运行状态。

## Ledger

- 2026-10-21：用户批准开始。沿用已讨论四组；具体数值及固定参考改造已记录Spec。未增加自主Reset或新Dynamics训练。
- Isolation：在无Git项目中新增独立am模块与封存胶囊，旧模块保持不变；不创建worktree或安装依赖。
- Task 1/2：分别观察5项模型、2项训练/数据测试因新模块缺失而RED；实现后相关20项GREEN。源/目标、mask与bank接口一致。
- Full baseline：旧`tests/test_parrad_convert.py`因interview缺h5py收集失败，无依赖安装；完整可用suite（仅排除该旧文件）另跑，日志保留。
- v1原生20项通过，但真实Qwen预检失败，未启动正式更新：冻结Qwen输出FP16，新增调制LayerNorm的FP32权重与其类型不匹配；单独CUDA回归已复现RED，改为LayerNorm前float32转换。旧胶囊/日志/failed.json全部保留。
- 独立整体审查完成：P1为短前缀初始base tokens被无条件detach（旧JointModel同样存在）；新增短前缀梯度测试确认RED，只修复新am模型且四臂一致。两项P2为预检配置/源码绑定、负相似度/原始CE记录不足；新测试均先RED，再补完整配置+manifest SHA绑定、负样本余弦独立复算和raw CE日志。不重复发起整体审查。
- 修复后本地相关22项通过、1项CUDA跳过；v2原生23项全部通过。真实预检4组零步等价差值0、扰动0、冻结未变、CE/CL学生梯度通过，CUDA allocated峰值约1.11GiB。独立原始官方区间核验、采样表逐项比较、train-only384点参考库与两角色512面板核验通过。
- v2于17:54:40后台启动PID19544；首组B0-seed42，首个阶段为step0面板评价。全suite补充运行无输出后主动中止，完整日志保留；不将本次相关测试通过扩写为全suite通过。
- 18:00首组train512/validation512零步预测、curves与independent-audit已保存并下载；实际10成功更新/10尝试，无AMP跳步。启动阶段完成，剩余各组训练及最终重放/比较在后台进行，最终核查保持未完成状态。
- 随后下载的启动状态为20成功更新/20尝试；本地native-v2/B0-seed42-start-status.json保留该快照。
- 2026-10-22 10:39:10全组结束，12000/12000次成功更新，无AMP跳步。14时后核查48检查点、78面板（39936预测）、210源码/5输入锁/70缓存，全部通过；结果证据下载在results-20261022-v1，权重保留服务器。F1三seed均值B0/align/mod/both为62.02/62.96/62.97/61.54%，均低于同面板初值64.52%；关闭调制差约0.05个百分点。结果报告已更新，未增加新训练。

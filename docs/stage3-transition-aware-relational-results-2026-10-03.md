# Teacher关系保留小实验结果

真实执行：2026-10-03 **13:19:10–13:22:07 +0800**，正常完成，status=`diagnostics_complete`。单折subject_02、训练seed1701；共享heads预热1轮，三个条件分别再训练3轮。生产权重未修改，未展开三折或其它训练种子。

## 结论

本次缓存dev评估中，两个联合训练条件的全部适配轮次，双向检索及跨版本检索指标均未低于epoch0。上一轮严重检索退化在此次有限实验中没有出现；不能据此宣布对未缓存窗口/held人群也有效，或把改善归因于某一个单独改动。

但转换识别没有改善。三个条件最终都选择epoch0；没有合格的新Projection。部分轮次的动作probe未收敛，按原协议记undefined，而不是把它解释为动作F1下降或训练进程崩溃。

## 原协议dev结果

基线独立转换probe AUROC为 **0.6258478965738886**，使用natural age6/7、两名dev subjects等权和三个固定投影seed的规定聚合。不是held成绩，也不是临时训练头的AUROC。

| 条件 | 全部适配轮次最高dev AUROC | 完整语义门槛通过轮次 | 通过门槛的最佳dev AUROC | 最差检索变化 | 选中epoch |
| --- | --- | --- | --- | --- | --- |
| head-only，Projection冻结 | 0.62584790 | 3/3 | 0.62584790 | 0个百分点 | 0 |
| joint-relation1 | 0.62461836 | 1/3：联合第1轮 | 0.62461836 | 0个百分点 | 0 |
| joint-relation10 | 0.62481698 | 1/3：联合第3轮 | 0.61633344 | 0个百分点 | 0 |

最差检索变化统计全部适配轮次、每subject的stream/commit、双向和两种跨版本配对；最小delta为0，部分项目可能改善，不能表述为全部指标都恰好不变。relation10最高分发生在不完整的语义评估轮次，不可选。即使不考虑undefined，其最高转换分数仍低于基线。

head-only模型实际完成了转换头训练，但Projection未改变，独立probe评估旧表示的分数自然不变。临时头指标另存为diagnostic pooled dev，不与原分年龄/分subject选择分数直接比较，也不用于挑选checkpoint。

## 未通过完整门槛的原因

- relation1：总epoch3/4(联合第2/3轮)的stream动作分类probe未收敛。
- relation10：总epoch2/3(联合第1/2轮)的stream动作分类probe未收敛。
- 各次fit分类器供两名dev subjects共同使用，所以每个轮次的两个undefined诊断源自同一次fit，不是两个独立故障。
- 没有发现其它数值语义指标超过原门槛。必需动作F1缺失仍使这些轮次不可选；没有提高400迭代预算或放宽1e-7/1e-9容差。
- 对完整通过门槛的两个适配checkpoint，转换AUROC均未改善；所以即使解决评估收敛，也不能据当前分数获得Go。

## 新目标与冻结检查

采用固定fit434窗口及全部42文本模板，每次更新使用完整参照库；只读取其它两人set000/001作为参数fit、set002作为dev。held/calibration数值及validation/test继续封存。

- 原label alignment关闭，损失为转换任务 + lambda_r*(三种新旧事件文本关系MSE + .1旧stream/commit关系MSE) + 新旧向量锚定。
- 初始真实fit-bank relation损失0、consistency损失0、Projection最大梯度0。
- 共用heads预热之后，Projection和teacher逐bit不变；128个正权fit行的转换任务梯度范数 **0.10872630029916763**，检查未执行optimizer step。
- 两个联合条件首批Projection分项梯度完全相同：transition **0.14097221195697784**，relation/consistency/anchor均 **0**。首次Projection更新由转换任务驱动。
- 共享预热82次更新，三个条件各3×82次更新，实际共820次优化步骤；没有为每个条件重复预热。三个条件保存的fork模型完全相同，后续Candidate次序一致。
- Teacher在三个条件中均逐bit不变；head-only的Projection亦不变，两个joint Projection均确实更新。
- relation1最后一轮平均训练transition loss .56629194，relation10 .57573880；head-only .61610188。损失下降没有转化为独立dev收益，这些训练损失不是模型通过标准。

下一步更值得核查转换训练目标与评估收益的衔接：训练使用age3–7、Candidate质量权重及共享临时头；选择使用age6/7的独立fit线性probe。两者存在已知差别，但本轮没有证明哪一项导致无收益。先诊断这一差别，不依据当前结果继续增加语义约束权重或扩大训练。

## 验证和证据

- 新增9个真实行为测试；全部Projection相关测试本地 **36通过**、服务器 **36通过**。包括初始零梯度、跨版本旋转、分块不变、fit边界、预热冻结、共享fork、评估中断恢复不重复更新、拒绝覆盖未完成预热证据、拒绝零任务梯度。
- 本地裸pytest因`tests/test_parrad_convert.py`缺少已有h5py依赖在收集阶段中断；排除此文件后 **390通过、2跳过**。未安装依赖。
- 服务器完整回归 **393通过、2失败**；失败仍为既有`test_gemma3_adapter_matches_native_prefix_and_targets_only_qv`、`test_gemma3_adapter_honors_local_sliding_window`，均是Gemma3 `None_inv_freq`错误。当前实验走Qwen路径。
- 只读复核发现的预热恢复/梯度保护已测试修复，最终复核通过，无剩余重要问题。
- 最终实验档案SHA256 `e3092febe3315b74c39f010a823d00e36f0899b88e3dba200ded1a58b01b78a0`，上传前后相同；原pilot/runner及upstream未修改。
- 原始结果档案SHA256 `8923402e9e9d5c20cd2ffdb24d9925c2fde94148e5681d15533f02f4e89dfbc7`，下载前后相同。原始JSON/日志已保存于`.audit-staging/results-relational-pilot-v1/`，汇总`summary.json`可复查。
- 服务器输出：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/stage3_transition_sequence_v2/transition-aware-semantic-relational-pilot-v1/`；完整实验权重保留在服务器，未下载大模型权重。
- 实现：`src/continuous_mmwave/parrad_projection_relational.py`；固定协议：`docs/stage3-transition-aware-relational-pilot-2026-10-03.md`。最终source manifest/checkpoint完整性校验通过；launcher/worker均已退出。

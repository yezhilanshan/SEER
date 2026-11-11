# HARrad CIF/LIF首版实现与验收

日期：2026-10-27。实现范围为用户批准的首轮事件门；未接入LLM，未在服务器启动真实HARrad训练。

## 交付

- 新增 `src/continuous_mmwave/event_gating/`，提供CIF、LIF动力学、因果证据状态、事件分类头、有界在线缓存、显式EOF、可选回溯和强监督逐帧分类对照。
- 新增HARrad适配：按实例构造多动作序列后重放原冻结GRU，拒绝拼接原历史hidden，不访问oracle_hidden。
- 新增prepare、train、evaluate、infer、smoke命令，以及CIF→LIF蒸馏→LIF微调的检查点转换、恢复、best读回与输入来源锁。
- 构造序列和自然连续录制分开评测；保留相邻同类实例，并单列未知/混合区间预测。
- [使用说明](harrad-cif-lif.md)、[配置](../configs/harrad-cif-lif-v1.json)、[设计](superpowers/specs/2026-10-26-harrad-cif-lif-design.md)。

## 旧模型保护

原HARrad模型、训练入口、EventFormer、radar_text及原权重均未改写。实现开始前记录33个 `harrad/*.py` 的SHA，结束时逐个复核。新代码只读导入已有前端，派生产物写入新目录；重复输出目录拒绝覆盖。

当前目录不是Git仓库，因此在原目录新增独立模块，没有创建Git分支或提交。旧HARrad检查点含NumPy normalizer，只在通过可信SHA核验后以兼容方式读取；新的事件门检查点使用 `weights_only=True`。

## 测试证据

解释器：`C:/User/anaconda/envs/interview/python.exe`，PyTorch 2.11.0+cpu、pytest 9.0.3。

- 新增5个测试文件，最终 **34 passed**。覆盖实际优化更新、CIF长序列质量闭合、无硬发火的CTC梯度、相邻重复、时间积分、分块一致、未来扰动、padding、尾事件、缓存有界、来源冲突拒绝和恢复选优。
- 旧HARrad回归：**69 passed、4 skipped**。跳过项包含CUDA及当前环境缺少的HDF5依赖，未修改旧测试。
- 独立命令行四阶段smoke：[完成记录](../outputs/harrad-cif-lif-smoke-20261027-v2/complete.json)。CIF、LIF蒸馏、LIF微调、frame-baseline均发生真实优化器更新，best重载与评测入口通过。数据为人工小特征，记录明确标记 `synthetic_only=true`、`real_harrad_training=false`，不可作为识别效果。
- 仓库全量检查：标准pytest在收集 `tests/test_parrad_convert.py` 时因 `h5py` 缺失中止；未安装依赖。`--continue-on-collection-errors`补充运行在输出6%进度后长时间无新增记录，已中断，日志保留在 `.superpowers/sdd/2026-10-27-harrad-cif-lif/final-full-pytest.log`。没有完整套件全绿结论。

## 独立审阅与修复

独立审阅发现4项Important，均增加回归并修复：

1. 900帧FP32 alpha的CIF逐帧累计可能少一个token：改为FP64累计/归一化，输出回原特征精度，验证质量守恒和梯度。
2. train可用另一前端的配置冒报来源：对比配置与prepared来源锁，检查点继承prepared真实来源，冲突拒绝。
3. epoch检查点在best刷新前保存：调整保存顺序；恢复还比较当前检查点与snapshot，兼容已生成的旧顺序产物，保留真正最优及最早平分轮次。
4. 特征infer可接受未知来源或oracle：正式特征模式强制匹配prepared、输入路径、SHA和维度；新录制可从同一来源的预处理RD输入。

审阅指出相邻同类召回缺失，已按设计补齐：支持集为参考序列中具有同类相邻项的实例，一对一有序匹配后计算召回。未留下此项延期。

审阅未裁决的范围在此明确：真实HARrad效果/GPU数值/服务器资产可用性待实际运行；CIF教师使用最终事件内部终点符合设计；可选回溯与软训练代理的差异仍是研究近似，默认关闭回溯，不宣称已消除该差异。

最终验收清单及新源码/旧源码SHA见 `outputs/harrad-cif-lif-implementation-20261027-v1/verification.json`。全量测试的依赖限制不被隐藏；新增模块及相关旧HARrad回归和实际CLI smoke结果分别记录。

## 后续真实数据运行的边界

配置保留现有服务器路径及SHA，`prepare`会实测检查。当前本地无完整特征数组，未声称服务器资产仍然可用，未启动正式训练。

训练样本使用官方边界拼接，具有构造序列的域差异；自然录制只做无真实边界重置的自由回放。未知区间不当作背景，部分标注F1不能当作全时间轴准确率。周期动作重复次数、未知动作开放集能力、骨架预训练和LLM描述均尚未包含在本轮交付中。

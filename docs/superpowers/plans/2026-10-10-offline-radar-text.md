# Offline Radar-to-Text implementation plan

执行：当前会话直接实现用户已指定的架构。工作目录没有 Git 元数据，不创建提交。

1. `tests/test_radar_text.py`：先添加顺序模板、窗口过滤、MD 时间片、padding 不变性、LLM 梯度与 mask、checkpoint、指标测试；在 interview 环境运行并确认缺模块失败。
2. `src/continuous_mmwave/radar_text/data.py`：实现 `OfflineCaptionDataset`、`collate_windows`、显式短语表与 ordered run 文本。按时间戳索引窗口；单 recording 缓存；检查所有 split 的受试者隔离。
3. `config.py` / `model.py`：独立配置，双 CNN、pack BiGRU、有序 pooling、projector、chat template 雷达注入、generation、可选 LoRA。用随机初始化微型 Qwen 测真实 forward/backward/generate。
4. `runner.py` / `__main__.py`：stage1、align、lora、evaluate、infer、smoke 命令；保存轻量 checkpoint；验证 loss 选优；真实数据动作顺序指标和分类模板对照。
5. `configs/radar-text-*.yaml` / 使用文档 / README：提供真实训练参数、数据契约、英文短语表示例、人工 caption 格式、无下载烟测和完整运行命令。
6. 在 interview Python 中运行新增测试及 PARrad 数据/转换/划分回归测试；运行 CPU smoke 和 checkpoint 推理，写明真实训练未完成。

Review focus：CNN 不读取 padded 值；MD 时间片不跨窗口；忽略标签不桥接；assistant-only loss 无 label 泄漏；checkpoint 不暗中保存随机替代的真实 LLM；训练和评估受试者隔离。

## 完成记录（2026-10-10）

- 已实现数据窗口、人工 caption 覆盖、局部 MD CNN、packed BiGRU、有序 pooling、projector、冻结 Qwen 接口、可选 LoRA、三阶段训练入口、checkpoint 推理、prepare 和基线评估。
- 独立审查指出的生成 RoPE 位置、跨阶段训练/历史验证受试者来源、分类基线隔离与词表一致性、多个 recording 的窗口 ID 冲突均已修复；相应回归测试先复现失败后通过。
- 实际验证：`interview/python.exe -m pytest tests/test_radar_text.py tests/test_parrad_data.py tests/test_parrad_split.py -q`：39 passed, 1 skipped。LoRA 实测因环境缺少 PEFT 跳过；额外尝试收集 `test_parrad_convert.py` 因缺少 h5py 失败，未将其计入通过测试。
- 最终代码 `python -m continuous_mmwave.radar_text smoke --output-dir outputs/radar-text-smoke`：完成 stage1 + align 训练、best checkpoint 重载、独立合成受试者生成、分类模板对照、无标签推理。使用随机 tiny Qwen，不能证明任何描述效果。
- 正式训练配置及用法见 `docs/offline-radar-to-text.md`。本机无真实 PARrad recording 或预训练 Qwen；未开展正式训练，未安装依赖。
- Ruling：采用用户明确指定的架构，直接实施独立模块；默认 0.5B，1.5B 可配置；当前目录不是 Git 仓库，未创建提交。受限运行在 Torch 系统路径解析处卡住，常规权限运行同一 interview Python 完成动态验证。

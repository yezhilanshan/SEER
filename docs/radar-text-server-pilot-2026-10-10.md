# Radar-to-Text 真实数据首轮 pilot 结果

已完成并核验。北京时间 2026-10-10 22:30:18 启动，22:53:06 完成两个训练阶段及 validation/test 自由生成评估；之后完成预先固定的 validation-only 错配诊断。没有追加训练。

**结论：真实 RD/MD → 冻结 Qwen 描述链路可训练，输入条件敏感性已有初步证据；动作覆盖和类别区分仍不足，尚不能视为可靠的连续动作描述模型。** 测试完整序列 121/484 正确（25.0%），超过本轮分类模板基线 54/484（11.2%），但动作召回低于基线。这里全部是弱监督模板短语代理指标，没有验证细粒度描述能力。

## 实际资源及设置

- 服务器 `10.15.22.62`，账号 `xibeidaxue`；认证交互输入，未写入脚本或配置。
- Python `/home/xibeidaxue/SEER/.conda/envs/seer/bin/python`；Torch 2.4.1+cu121、Transformers 5.4.0、PEFT 0.18.1、h5py 3.10.0。未修改环境。
- RTX 2080 Ti，compute capability 7.5，使用 FP32。
- 数据 `/home/xibeidaxue/newdisk2/WJ/PARrad/canonical/phase1`，使用完整 RD/MD。
- 原始 Qwen `/home/xibeidaxue/SEER/Qwen2.5-0.5B-Instruct`，hidden 896、24 层，主体冻结；本轮没有训练 LoRA。
- 独立代码 `/home/xibeidaxue/SEER/.audit-staging/radar-text-code-20261010-v2`。
- 输出 `/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/radar_text_pilot_20261010_v1`。

固定 seed 17；窗口 12 秒、步长 6 秒、最短 8 秒、最多 3 个动作。8 个有序 radar tokens，双 CNN + 两层 BiGRU + MLP。训练 subject02/03/04，验证选优 subject05，测试 subject06；checkpoint 中保存并核验了训练和选优受试者，没有与测试重叠。

| 集合 | 窗口 | 单动作 | 双动作 | 三动作 |
| --- | ---: | ---: | ---: | ---: |
| train | 2187 | 20 | 582 | 1585 |
| validation | 702 | 4 | 223 | 475 |
| test | 484 | 2 | 90 | 392 |

测试只有一个独立受试者；484 个窗口含两个雷达视图，各 242 个，且窗口有重叠，不能解释成 484 个独立实验或充分的跨环境验证。单动作样本尤其不足。

## 训练与权重

stage1 batch4、accumulation1、encoder/classifier lr1e-3；align batch2、accumulation4、encoder lr1e-4、projector lr1e-3。两阶段各 3 epoch，分别按 validation loss 选择 best。

| epoch | stage1 train CE | stage1 validation CE | align train CE | align validation CE |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.239363 | 2.046084 | 0.393076 | 0.240216 |
| 2 | 1.840283 | **1.807447** | 0.176973 | 0.207611 |
| 3 | 1.453759 | 1.972535 | 0.147302 | **0.197782** |

stage1 best 来自 epoch2，align best 来自 epoch3。stage1 第三轮训练下降而验证回升，已有泛化差距；align 验证 loss 继续下降，三轮不能据此判断已收敛。两种 CE 的监督目标不同，数值不能跨阶段比较。

stage1 可训练参数 815,758；align 可训练参数 1,846,144。最终 `align/best.pt` 7,399,667 字节，仅包含雷达侧和 projector 等元数据；不包含完整 Qwen，推理仍需原始 Qwen 权重。

## 自由生成结果与分类模板对照

以下 precision/recall 为逐窗口动作多重集分数的平均；顺序 LCS 是最长公共子序列长度除以参考与预测的较大长度。英文短语匹配不可靠处理同义改写或否定，不能代替人工评价。

| 指标 | validation 生成 | validation 分类模板 | test 生成 | test 分类模板 |
| --- | ---: | ---: | ---: | ---: |
| 完整动作序列正确率 | 20.09%（141/702） | 13.68%（96/702） | **25.00%（121/484）** | 11.16%（54/484） |
| 动作 precision | 50.00% | 39.41% | **50.72%** | 38.47% |
| 动作 recall | 45.42% | **50.07%** | 46.28% | **55.51%** |
| 顺序 LCS | 43.42% | 36.94% | **45.59%** | 36.67% |

测试完整序列提高 13.84 个百分点，同时 recall 下降 9.23 个百分点。分类模板基线直接合并逐帧 argmax 的相邻相同类别，没有平滑，部分输出会因类别抖动很长；本轮优势只针对这个实际基线，不能外推为超过所有分类方法。

测试双动作 31/90 完整正确（34.44%），三动作 90/392（22.96%），单动作 0/2。两个雷达视图完整序列分别为 56/242（23.14%）和 65/242（26.86%）。

## 主要错误

- 148/484（30.58%）测试窗口预测动作数量少于参考，13/484（2.69%）多于参考。数量相同的窗口仍可能认错动作。
- 测试未发现“动作多重集完全正确、仅顺序错误”的窗口；验证有 1 个。这说明当前不能把全部失败归因于顺序编码不足。部分动作已漏掉的窗口也会同时有顺序错误。
- 测试生成 51 种动作序列，参考 65 种；最常见生成序列占 52/484（10.74%），没有坍缩成唯一描述，但存在输出偏好。
- 床相关类别覆盖弱。以下是按动作出现次数统计的 recall，与上表逐窗口均值不是同一聚合方式：

| 动作 | 参考出现次数 | 匹配次数 | 出现次数 recall |
| --- | ---: | ---: | ---: |
| gets out of bed | 84 | 7 | 8.33% |
| lies in bed | 60 | 7 | 11.67% |
| gets into bed | 82 | 14 | 17.07% |
| stands up from the bed | 54 | 12 | 22.22% |
| stands up from the chair | 104 | 76 | 73.08% |
| stands up from the floor of the room | 134 | 98 | 73.13% |

按数据顺序取出的真实样例（完整 sample ID 与文本见原始 JSON）：

1. 正确：`subject_06/set000/TI_xWR14xx/e816907bbacd:267:400`，参考和生成均为：
   > The person walks toward the room, then falls onto the floor of the room, and then stands up from the floor of the room.
2. 类别混淆：同 recording 的 `:0:134`，参考“走向椅子 → 坐到椅子”，生成“从椅子站起 → 走向床 → 坐到床”。
3. 漏动作并换序：同 recording 的 `:200:334`，参考“走向床 → 走向房间 → 跌倒”，生成“走向房间 → 走向床”，遗漏跌倒。

## 雷达错配诊断

该规则在 test 预测产生前固定：均匀取 32 个 validation 窗口，以后一个动作序列不同的窗口替换雷达，保留同一目标文本；冻结最终 best 模型，optimizer 更新为 0。前 8 对额外进行自由生成。

- 匹配 radar 的平均 teacher-forcing loss：0.254438。
- 错配 radar 的平均 loss：0.470200，平均差 0.215763。
- 26/32（81.25%）配对在匹配时 loss 更低。
- 前 8 对更换 radar 后生成都变化；匹配生成 4/8 完整正确，错配生成相对于其自身 radar 的参考也为 4/8。

这支持对输入条件的敏感性，不能证明细粒度 grounding 或稳定泛化，也没有直接隔离“时序顺序”因素。现有标签模板不能监督速度、幅度、停顿等细节。validation 参与选优，上述诊断不能当作独立测试证据；unmatched=0 仅代表匹配到了短语，不代表零幻觉。

## 核验与产物

原生测试日志记录 **44 passed, 1 warning**（包括 LoRA 保存加载与 HDF5 转换）。真实 Qwen 预检通过 forward/backward/generate，雷达侧各模块梯度非零、LLM 可训练参数为 0。

训练后重新读取 best checkpoint，核对 epoch、非 tiny 模型、受试者来源；重新校验 117 个源码/配置、原始 Qwen 权重、原始 manifest 和 label map 的 SHA，与预检记录一致。本地取回 26 个产物并逐项核验大小与 SHA；独立从原始生成文本复算 validation/test 的动作提取、precision/recall、序列与 LCS 分数，生成和基线共 2372 条均一致。只读独立审查未发现明显受试者泄漏或计数错误。

- [原始 validation 预测](../outputs/radar-text-server-pilot-20261010-v1/pilot/validation-evaluation.json)
- [原始 test 预测](../outputs/radar-text-server-pilot-20261010-v1/pilot/test-evaluation.json)
- [错配诊断逐样本结果](../outputs/radar-text-server-pilot-20261010-v1/pilot/grounding-probe.json)
- [本地复算和逐动作统计](../outputs/radar-text-server-pilot-20261010-v1/local-audit.json)
- [权重、来源和 SHA 核验](../outputs/radar-text-server-pilot-20261010-v1/pilot/verification.json)
- [最终雷达侧/projector 权重](../outputs/radar-text-server-pilot-20261010-v1/pilot/align/best.pt)
- [归档逐文件清单](../outputs/radar-text-server-pilot-20261010-v1/result-manifest.json)

结果归档 SHA256：`a381c19abbf3f2c389d835b9af0bfd2c7b92f058ef2e1573a07d2ee7a0bc94a3`。

源码 v2 archive SHA256：`d2a622e8f931094973a735584675fc3fe85bb70c15f297cc5a1d173f7e26fbfd`。

Qwen weights SHA256：`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`。此前进度文字手工转写多了一个 `f`；这里以机器生成的 preflight 和复核 JSON 为准，原始权重没有变化。

v1 因源码包漏带旧测试辅助脚本而 43pass/1fail，未启动训练；v2 补齐后重新测试及预检才启动。本地 helper 的后续通用路径加固没有修改正在运行的 v2 capsule。

## 建议的下一轮

优先构造类别平衡的单动作短窗口和可控双动作验证，先查床相关动作在雷达编码器与语言接口各自的混淆，再比较冻结接口和 LoRA 的增益。补充平滑分类模板基线及同动作反序诊断，继续保持简单架构。新增方案只在 train/validation 上开发；subject06 已看过，本轮不据其结果挑选训练参数，后续最终泛化结论需要新增未使用受试者。当前没有启动下一轮训练。

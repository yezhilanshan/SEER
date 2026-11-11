# HARrad第1、2步：流式前端与EventFormer语义预热

本轮已实现流式前端与语义训练流程，已在服务器启动后台任务（PID24593，2026-10-17 13:34:34，UTC+8）。第一步完整录制导出及原分类复现已通过；第二步12类小样gate在40步通过（accuracy100%、CE0.1603、冻结基座保持不变），随后从独立seed42初始化进入正式5轮训练。

2026-10-17 19:05:42（UTC+8）实时复核：全流程已于 **17:41:07完成**，总耗时4小时6分29秒，PID24593已退出。正式5轮完成3,775次成功更新、3,775次尝试及15,090次实例暴露，未发生AMP跳步。最佳为第5轮，重载best预测误差0、冻结基座保持不变；本轮test信号未读取。训练目标为12类实例语义，不含文本生成、预测分支或自主Reset。

## 完成结果

第1至5轮validation完整实例Macro-F1依次为57.5854%、68.9525%、73.6432%、76.5547%、77.0069%，最佳轮accuracy77.4951%。固定前缀0.4/0.8/1.6秒Macro-F1分别46.6884%/68.8843%/76.6574%，支持数511/511/381，不把不同支持集的总体分数当成同一样本的时间曲线。

完整实例Oracle重置下EventFormer Macro-F1为77.0069%，原冻结RD-GRU实例读出参考为85.8734%。本轮新语义分支只有5轮预热，不能视为同预算优劣结论。CNN cells置零时Macro-F1降至1.2658%，支持语义模块依赖输入特征；不代表已证明状态的历史信息价值。

无Oracle Reset的连续validation诊断，实例末端Macro-F1仅12.8889%、accuracy14.0900%。这说明当前实例重置预热模型尚不适用于直接长流无重置使用；训练/推理状态协议差异、长时状态与未标注区域的影响仍需诊断，不能仅凭此确定失败原因，也未启动自主Reset训练。连续读出截止与请求相差0或1帧，本地逐预测核验无未来读入。

已下载完成记录、5轮history、服务器独立指标、最佳/零特征/连续三组共12份预测；本地独立实现对12份预测的指标复算全部通过。权重仍保存在服务器原输出目录，尚未下载全部检查点。

后续已完成[完整录制、官方起点重置诊断](harrad-oracle-start-replay-2026-10-17.md)：全25,244帧恢复到76.73%，接近77.01%，但严格逐值复现仍未通过。4个分歧实例同特征/同batch的流式与片段误差0；原batch4也逐值复现，支持有限精度及CNN分块/批尺寸数值影响。新增训练0步，原结果保留。

## 使用的新权重与接口

- 复用RD-CNN-GRU seed42的验证best（第29轮），不从三个seed中再次挑优。原CNN、GRU、分类头与normalizers冻结保留。
- CNN末层空间图为64×5×8：原全局池化64维继续供GRU；另作2×2空间池化形成每帧4×64维cells。GRU继续输出128维历史状态，跨块传递，录制起点清零。
- 语义分支从CNN cells读取，不把GRU状态再次作为输入；GRU流式状态供下一阶段动力学分支及本轮参考评价使用。
- 每2帧Update，8个空间/时间候选经可学习QueryPooling形成4×256 Radar Tokens，再递推8×896 Event State。奇数分块保留尾帧，显式结束时才允许单帧flush。
- EventFormer使用通用本地Qwen2.5-0.5B前4层；基座及LoRA全部冻结。新训练参数共 **7,287,313**，包括Token Adapter、事件跨注意力、门控、状态投影/初始状态及12类读出；冻结基座参数494,212,992。CNN/GRU在独立冻结前端内，未计入此基座数。

## 已通过的检查

- 本地新增及相关测试14项通过：不规则分块与完整前向等价、padding不推进状态、原分类复现、未来扰动、奇数帧buffer、Token区分、事件输入梯度、冻结保持、实例数据与小模型训练/重放。
- 服务器原生30项全部通过；2条warning来自现有PyTorch checkpoint内部使用弃用的CPU autocast接口，不是测试失败。
- 真实Qwen FP16/AMP预检通过：Token投影梯度最大值0.001749，输入梯度非零且参数梯度有限，冻结基座不变。
- 真实Qwen未来扰动与额外padding检查，对应早期/最终logits最大绝对差均为0。
- 完整真实35条录制、215,482帧已导出；源RD逐文件SHA核验通过，原权重和normalizers保持不变。真实不规则分块hidden误差3.0100e-6；原best分类logits最大误差5.7220e-6，511个验证实例argmax变化0。
- Oracle/连续GRU的完整实例与3个前缀共8组预测已下载，并由本地独立指标实现复算通过。Oracle完整实例Macro-F1仍为85.8734%；连续GRU不按实例重置时为76.4936%，两者协议不同，不能称在线性能等同原实例分类。

## 首轮语义训练设置

train=subject1–6（3,018实例），validation=subject7（511实例），test8–9信号不读取。仅seed42，5完整epochs，batch4，AdamW lr1e-4、weight_decay0.01、clip1；新旁路FP32、冻结Qwen FP16、AMP初始scale256，每batch最多3次尝试。每8次Update detach事件状态，跨块不重置。

每段使用官方起点初始化Event State；完整实例末端加权CE为主损失，另加0.25倍固定观察时刻0.4/0.8/1.6秒可用前缀CE的均值。成熟度只用已观察帧数/15/1.2秒（上限1），不用完整段长或未来标签。不同前缀的样本支持数分别报告；短动作不通过补帧凑前缀。

正式训练前，以每类一个train实例做小样gate，最多300步lr1e-3，准确率≥0.99且CE≤0.3才放行；gate权重不用于正式初始化。失败会保存原因并停止，不自动放宽。

每轮保存完整实例与前缀指标、逐样本预测；按validation完整实例Macro-F1最高且最早epoch选best。结束后重载best、独立重算预测指标，另做CNN cells置零的零特征对照和不使用Oracle Reset的连续validation诊断。零特征对照不等于全零RD经过冻结CNN。无Reset读出取请求截止时间之前的最后完整双帧Update，最多提前1帧，不读取截止之后的数据；需单列这一对齐差异。

本轮仅证明已知实例边界下的语义读出能否学习，不代表自主事件边界、自由文本或跨所有受试者泛化已验证。

## 源码与证据

- [固定配置](../configs/harrad-event-warmup-v1.json)、[实施计划](superpowers/plans/2026-10-17-harrad-streaming-event-warmup.md)。
- [本地准备文件](../outputs/harrad-event-warmup-20261017-v1/)。
- [流式验收及启动证据](../outputs/harrad-event-warmup-20261017-v1/launch-v1/)、[独立复算](../outputs/harrad-event-warmup-20261017-v1/launch-v1/local-independent-verification.json)。
- [完成及预测证据](../outputs/harrad-event-warmup-20261017-v1/finished-v1/)、[最终本地独立复算](../outputs/harrad-event-warmup-20261017-v1/finished-v1/local-independent-verification.json)。
- 服务器实际源码目录：`/home/xibeidaxue/newdisk2/WJ/HARrad/experiments/event_warmup_20261017_v1`。
- 服务器实际输出目录：`/home/xibeidaxue/newdisk2/WJ/HARrad/outputs/event_warmup_20261017_v1`。
- 源码v1包175项清单、589,030字节，SHA256 `f527aef29666ee046caf824462609c66d0913c8a758c604afad41eb959a4ed8f`；已上传并通过原生预检。后续如有修正会保留旧版本，另建版本运行。

# v2.6 第1–2步：冻结前端预测预实验结果

本轮已完成来源与数据审计、冻结缓存、五臂预测比较，以及保存权重与逐录制预测的独立核验。**在本轮固定300步、seed42、预测未来0.18秒的诊断中，Event State没有带来额外预测收益；Persistence最好。**结果只支持命令边界代理诊断，`formal_go=false`，真实实例/物理时钟的G0仍未通过，不能据此宣布正式Stage1或闭环Reset训练条件已具备。

后续[P0只读诊断](v26-p0-diagnostic-results-2026-10-15.md)已完成：全fit也未达到Persistence；低/中/高变化和边界邻域均无神经预测优势。另定位到首轮valid屏蔽全部跨命令混合update，致使包括截断前全部合法h1对在内的cross组均为0。原指标没有改动，但必须限定为不跨该代理边界的合法预测视图，不能解释为已检验跨事件转换。

## 固定实验与实际执行

方案和范围在读取本轮validation分数前固定，见[执行计划](superpowers/plans/2026-10-14-v26-p0-p1-prediction.md)、[P0审计](v26-p0-readiness-2026-10-14.md)与[运行说明](v26-prediction-pilot-usage.md)。新增实现位于`src/continuous_mmwave/v26/`，旧Stage1/Stage2权重保持冻结。

- 使用原causal MD-GRU编码器，2帧/update，转换时间网格0.09秒/帧；目标为冻结4×256 Radar Tokens的masked mean，256维。没有改成尚未实现的RD/Point前端。
- 事件状态为原Qwen2.5-0.5B EventFormer的8×896状态，按原命令CSV代理边界重置；maturity只使用已到达update计数，不读取未来事件总长度。命令时间不等同于真实动作起点。
- fit：9位训练受试者各取首个双传感器完整session，共18份录制；subject09的set000仅有68xx，按元数据选择set001。其他人取set000。validation：05/12/16各两传感器，共6份。test雷达数组未读取。
- 全录制连续编码，主比较限每个代理episode前32个合法update，h=1。各臂使用同一当前/未来索引；fit共50,044对，validation共13,559对。29个奇数帧训练录制是全55份数据审计的结果，本轮选中18份仅舍弃10个不完整尾帧，不能混用两个分母。
- 归一化仅由fit计算，按recording等权，std下限1e-4。Persistence复制当前latent，方差按fit未来残差的recording等权二阶矩估计；所有臂logvar范围[-6,2]。
- 四个神经臂均seed42、batch8、AdamW、LR=0.001、weight decay=0.01、gradient clip=1，各执行300次成功更新。采样清单完全相同。先完成各1步独立烟测；使用固定末步权重，只执行一次正式validation评价，没有按分数改架构、追加步数或选择checkpoint。

新增可训练参数分别为：无事件MLP 394,240；GRU 395,008；TCN 394,816；Event 396,832，均落在预设40万±10%内。GRU/TCN读取同一代理边界开始的完整4-token雷达前缀；TCN感受野恰为32。Event读取当前motion与冻结事件状态。实际严格加载的全前端含501,893,267个参数，可训练数为0，其中Radar Encoder为265,728、EventFormer为501,362,309。**匹配的是新增训练参数，不是总预训练容量或推理计算量；本轮未测完整FLOPs。**

## 验证集结果

每份录制先对有效时间与256个坐标求均值，再将同一受试者两份录制等权平均，最后对05/12/16等权平均。两传感器不当作独立受试者。MSE使用fit标准化后的latent，不是动作分类错误率；NLL为每维Gaussian负对数密度，数值为负是允许的，两者均越低越好。

| 预测臂 | 受试者宏MSE | 每维宏NLL | 平均预测方差 |
|---|---:|---:|---:|
| Persistence | **0.033655** | **-0.335863** | 0.033183 |
| 无事件MLP | 0.071295 | 0.224875 | 0.072775 |
| GRU | 0.067550 | 0.138235 | 0.056379 |
| TCN | 0.068222 | 0.069268 | 0.058273 |
| Event State | 0.073509 | 0.160063 | 0.065788 |

Event的MSE比GRU高8.82%、比TCN高7.75%、比无事件MLP高3.10%；比Persistence高118.42%。Event的NLL优于无事件MLP，但劣于GRU、TCN与Persistence，不能只用某一个loss数值宣称事件状态有益。

| 预测臂 | 05历史验证MSE | 12 MSE | 16 MSE | 12/16平均MSE | 12/16平均NLL |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.031880 | 0.031958 | 0.037127 | 0.034543 | -0.323215 |
| 无事件MLP | 0.070173 | 0.065023 | 0.078690 | 0.071857 | 0.230461 |
| GRU | 0.067164 | 0.062468 | 0.073018 | 0.067743 | 0.131018 |
| TCN | 0.065076 | 0.064316 | 0.075274 | 0.069795 | 0.081049 |
| Event State | 0.070488 | 0.068890 | 0.081148 | 0.075019 | 0.171156 |

三个受试者上Event的MSE都高于GRU、TCN和无事件MLP。05单独报告是因为旧前端训练使用过它做验证；12/16是当前扩充分割下的额外验证视图，不声称它们从未被任何历史实验观察。

这里能确定的是“本轮读出没有兑现额外预测收益”。尚不能将原因归为事件状态必然没有信息：单种子短训、只看h=1、短前缀、Gaussian读出和命令代理边界都限制结论。当前预测头直接输出未来均值，没有把`当前motion + 残差`作为均值参数化；全部神经臂都输给Persistence，提示下一轮应先核查预测头是否充分学会短时恒等/残差预测。这个解释是后续待验证假设，本轮没有修改实现或补训来追分。

## 核验与实现边界

- 本机interview与服务器原生seer环境，各有35项新增及相关测试通过。完整本机pytest在`tests/test_parrad_convert.py`收集时因缺h5py失败；排除该文件的补充全套运行被中断，没有完整全套通过结论，也没有重装依赖。
- 每份缓存首个encoder update重复输出逐值一致；每份录制最初32个update中的事件前缀与原顺序实现逐值一致，24份最大绝对误差均为0。两个role导出前后全前端状态SHA相同；训练和验证role之间也一致。
- 真实模型严格加载时，先恢复Stage1 Encoder，再恢复Stage2的68个fitted非encoder参数，并锁定Qwen权重/tokenizer/配置及嵌套前端YAML。不存在用随机Encoder补齐Stage2省略权重的情况。
- 每个神经臂最终检查点SHA、结构、训练config、源码SHA及所有Adam状态步数均核验。所有臂的300×8采样清单已在下载后再次比较，完全一致。
- 五臂×六录制共30份预测：独立NumPy float64重算NLL/MSE/variance及受试者宏平均，与保存值在rtol=1e-5、atol=1e-6内一致；重载真实头权重后的mean/logvar/target逐值一致；当前/未来索引与缓存完全一致。
- 无Git仓库，按已记录的实现裁定使用独立源码capsule和SHA快照代替commit/worktree。服务器v1保留首次元数据预检失败；v2保留因耗时停止的部分顺序缓存；v3保留数值不合格的batch16尝试；v4为本报告唯一训练和评价来源，没有覆盖前面的目录。
- batch16虽通过CPU因果测试，但真实FP16前端参考最大绝对误差0.008789，超过事先设定的容差，未用于训练。最终使用batch1 CUDA Graph保持原递推，恢复rtol=0/atol=0的参考检查，没有放宽数值门槛。

新鲜代码审阅没有Critical，三项Important均已补回归并修复：嵌套配置/Qwen资产锁、最终权重及评价完整性保护、Persistence的recording等权方差。以下Minor明确延后，未当成已完成验收：全录制逐步Encoder回放、缓存独立command_episode_id字段（现保留prefix_start/age范围）、边界内/跨边界分组、联合NLL、完整在线P50/P95与峰值显存。当前缓存中的Event Graph回放P95只计图执行，不含输入复制、Encoder或整个在线系统。真实实例边界、动作物理起点和独立时钟仍缺乏验证，因此正式G0/G1不授予通过。

实际导出的fit/validation分别为61,407/20,111个完整update，分别舍弃10/3个尾帧。24份录制各自的batch1 Event Graph回放P95为2.366–2.414毫秒，仅作上述局部执行时间记录。

## 产物与后续判断

服务器保留完整F/E缓存和全部原始来源引用：

`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/v26_prediction_pilot_20261014_v4`

独立训练源码：

`/home/xibeidaxue/newdisk2/WJ/PARrad/experiments/v26_prediction_20261014_v4`

本机证据目录：[outputs/v26-prediction-pilot-20261015-v1](../outputs/v26-prediction-pilot-20261015-v1/)。包含四臂末步权重/optimizer、各1步烟测、300步loss/采样清单、30份逐录制预测、协议/缓存索引/统计、日志及完整SHA清单；大体积F/E缓存保留在服务器。源码快照见[source-capsule-v4.json](../outputs/v26-p0-inventory-20261014/source-capsule-v4.json)。

[本机下载核验](../outputs/v26-prediction-pilot-20261015-v1/local-verification.json)已通过：压缩包210,659,161字节，包SHA一致，65个文件逐个大小与SHA一致，四臂300步采样清单一致，30份预测SHA均与服务器权重回放证明对应。精确指标与逐受试者结果在[evaluation/results.json](../outputs/v26-prediction-pilot-20261015-v1/evidence/evaluation/results.json)。

| 保护对象 | SHA256 |
|---|---|
| Stage1 | `42065f59311cb41e6c23c7734b66392028d634d7c11264bfa3325cb2be7b7289` |
| Stage2 | `939aa977b2616d8ceb11f47ec1470701dc04bd7d1af64cc5ac20ebed3ab88f9a` |
| 完整冻结前端状态 | `4ddd405609d027cb87e2ca496a061392a7b954215c135c6755a966f3f1276501` |
| 训练模块源码 | `fefe79c36f3d488a9b1a283582e1f0152d38c168e4592f95137c134c623fd086` |
| pilot YAML | `5d6d375c13aaeed03d738043f0c9e88a1af1e34f8126b03066e25f84486f3bdd` |
| v4源码capsule | `b27cbac0bf375458cf92fd76c185c55de9369ac67a0cc166c446b7839df0a982` |
| 证据压缩包 | `707966b92ca3de24a8434b42d5d4e26f84b6060c482616d99d95358b438b851b` |

**后续应先定位全部神经预测头输给Persistence的原因，再按新协议判断是否重做残差读出/收敛诊断。**正式事件收益仍需真实G0、预先固定的新预算和独立seed验证。本轮停在已经核验的代理预测诊断，没有追加训练，也没有开始Keep/Birth、Reset、ST/TBPTT、Commit/Memory或LLM微调。

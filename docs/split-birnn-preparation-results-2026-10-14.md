# Split BiRNN 实现准备与验收结果

按用户审查完成了独立的数据转换/时间覆盖审计、Split BiGRU 与普通 BiGRU、损失、流式修正回放、等价性测试和 2080 Ti 探针。**正式五折训练没有启动，当前不能声称严格论文复现成功。** Hospital 映射、真实雷达时钟和 CNN 精确结构仍缺依据。

## 本轮修正

主实现采用每批统一长度、批量 GRU、随机片段零初态；前向流式状态跨窗口持续传递。两分支共享一个分类头，最后隐藏状态取平均；后向 loss 不 detach 前向缓存。主损失为原文标签分布真类0.8/其他各0.025 的 CCE + 全局 Dice，per-class Dice/CCE-only 作为显式备选，未做训练消融。

保存 Forward/First-corrected/Final-corrected 三套 logits、每帧首次/最终修正端点、等待时间及覆盖次数。Split 未覆盖尾帧保留前向预测；普通 BiGRU 没有因果头，尾帧显式 unavailable，完整分母指标计为 abstention，不能静默丢掉。代码实现因果回放和流式状态接口，本轮没有部署实体边云网络。

独立代码审查发现3个 Important 和2个 Minor问题，全部加入失败回归后修正：诊断图缺失非零时间原点；源 MD 在值校验前可能损失 float64 精度；全未标注批次可能错误通过 Adam 探针；超出录制的后续命令被误记为外推尾部；无 MD 的最终 H5 session 可能被计为完成。V1源码保留，V2修正；本批153份MD原dtype均float32，V1已有真实数据没有精度损失。

## 数据与时间审计

新数据目录：`/home/xibeidaxue/newdisk2/WJ/PARrad/canonical/split_birnn_preparation_20261014_v1`。只读当前原始 MD/CSV，不读 cube、不覆盖旧 canonical。原始 MD 保存值没有 signed-log 或预先标准化；归一化在模型里逐帧进行。数据1,033,155帧的 source/saved MD 完整值、源文件状态、CSV/NPZ摘要和九类标签都通过独立重算。

| 项目 | 结果 |
| --- | ---: |
| 受试者目录 | 23 |
| H5 / CSV session | 77 |
| 传感器记录 | 153 |
| MD 帧数 | 1,033,155 |
| CSV 命令 | 6,939 |
| 合并后连续标签段 | 13,400 |
| 有效相邻标签转换 | 13,247 |
| 每记录转换次数范围 | 33-120 |
| 命令落在录制结束之后 | 0 |
| 时间格碰撞的命令边界 | 0 |
| 最后命令到录制结束，最小/中位/最大 | 0.04064 / 0.797463 / 20.226724 秒 |

连续段是近似标签序列中的游程，不直接等于论文动作实例。不同传感器统计重复观察同一 session，不能用乘二证明原文13,359动作一致。

| 九类 | 有效帧数 | 标签连续段数 |
| --- | ---: | ---: |
| Walk | 347,057 | 3,842 |
| Fall | 78,135 | 1,196 |
| Stand up | 205,650 | 3,002 |
| Sit down | 123,552 | 1,836 |
| Get in bed | 62,523 | 762 |
| Lie in bed | 45,397 | 687 |
| Roll in bed | 61,266 | 662 |
| Sit in bed | 46,600 | 679 |
| Get out bed | 62,975 | 734 |

**未标注帧为0是协议造成的结果：默认首条命令=首帧，末条命令外推到末帧。** 这不是观察到录制首尾都同步标注，更不证明命令提示等于真实动作开始。固定0.09秒、两个雷达采用同一近似 origin 时索引一致，也不证明实际帧周期或时钟相同。没有根据 CSV 时长拟合新周期，也没有把 MD 谱变化自动当作真实动作起点。

已输出每记录时间覆盖/命令索引/外推及每受试者、传感器九类分布；候选五折按人分组，同人的全部 session 和两个雷达同折。独立审核确认角色互斥、来源穷尽和每份记录只出现一次 outer test；五折各角色类别覆盖见独立审计JSON。候选不冒充原作者 fold，不根据测试 F1 反选数据。

官方公開下载HTML可读取，但技术README目录接口返回HTTP401；浏览器初始化两次超时，未取得环境映射。没有使用私有账户认证。原始H5的抽查属性无环境依据。所有记录仍标记 Hospital=unconfirmed；发布版本是否对应论文20人也没有得到确认。

8个 train-source subject01/02 命令转换窗口已绘制 MD、跨bin标准差及标签时间线，带有“近似时间格、命令提示不是实际动作起点”的标注。图中标准差是源值数值统计，不解释为经标定的物理能量。抽样图不能替代人工同步标注。

示例：[01/77GHz窗口](../outputs/split-birnn-preparation-20261014/figures/subject_01-set000-TI_xWR14xx-0.png)、[02/60GHz窗口](../outputs/split-birnn-preparation-20261014/figures/subject_02-set000-TI_xWR68xx-1.png)。

## 参数与计算图

| 模块 | 当前可训练参数 | 论文参数约束 |
| --- | ---: | ---: |
| CNN，尺寸自洽图示解释 | 7,283 | 推导8,301，差−1,018 |
| 三层前向GRU | 4,896 | 4,896 |
| 三层后向GRU | 6,432 | 6,432 |
| 唯一共享head | 153 | 153 |
| Forward含CNN/head | 12,332 | 13,350 |
| Backward含复用head | 6,585 | 6,585 |
| Split唯一总参数 | 18,764 | head只计一次 |
| 普通BiGRU含CNN/head | 20,444 | 21,462 |

CNN另有357个buffer元素，其中包括BN均值/方差及batch counter，不能混入可训练参数凑数。参数审计列出每个层张量形状。Figure4尺寸疑点未获得作者源码澄清；当前版本只保证维度自洽与计算测试，不宣称是作者精确CNN。

真实MD在CUDA下，整段/逐帧前向及cache最大差 `5.96e-8`，任意分块同样 `5.96e-8`，联合/缓存后向最大差0；后向独立损失能更新CNN和两组GRU。CPU测试还覆盖未来扰动因果性、共享head平均、准确平滑分布、忽略标签、首尾/短序列/重叠与recording reset。

## 吞吐量探针

V2使用RTX2080Ti、Torch2.4.1+cu121、FP32、batch8、每长度2次warmup+5次计时，CUDA synchronize。计时含CNN、GRU、损失、backward及Adam，但输入为预先驻留GPU的随机张量，不含完整数据加载和验证开销。

| 模型 | 长度 | 平均单步ms | 峰值allocated GiB |
| --- | ---: | ---: | ---: |
| Split | 100 | 32.372 | 0.466 |
| Split | 512 | 41.956 | 2.318 |
| Split | 1000 | 60.183 | 4.520 |
| BiGRU | 100 | 26.172 | 0.466 |
| BiGRU | 512 | 34.933 | 2.317 |
| BiGRU | 1000 | 55.670 | 4.518 |

单帧前向/缓存后向512帧/普通BiGRU512帧的均值为1.619/0.692/2.523ms，P95为1.677/0.707/2.562ms。这里只证明当前重建模型的计算可行性；CNN不完全相同，框架和实现也不同，不能将它作为论文表2数值复现。计时不包含网络、窗口等待或真实动作反应时间。

V2真实数据仅做一次Adam单步，subject01两份512帧样本，1,024有效标签，loss5.8155、梯度有限、不保存权重。V1和V2各做42个合成探针步骤+1个真实步骤，合计84个合成和2个真实步骤；测试梯度另外计算。**这些不是正式训练，也没有训练后F1。**

由吞吐量可提出 batch8、max_optimizer_steps6400、每128步验证、最少1280步后patience15次检查的候选预算，Split/BiGRU相同，见 `configs/split-birnn-budget-candidate.yaml`。它仍是候选，正式冻结字段关闭：需要确定CNN、数据/近似时间协议并完成验证/断点训练入口。随机采样不把epoch解释成一次全数据遍历。暂不估计完整五折墙钟时间，因为加载、验证和停止规则尚未实测。

## Gate 与交付等级

| Gate | 当前状态 | 依据与剩余项 |
| --- | --- | --- |
| G0数据可信 | 近似协议审计通过；严格复现未通过 | 值/标签算术/来源/fold可重算；Hospital、实际clock和物理onset未知 |
| G1结构正确 | 计算图通过；精确结构未通过 | GRU/head/梯度/等价通过；CNN差1018尚未闭合 |
| G2流式正确 | 当前重建模型回放通过 | 首次/最终/端点/覆盖/尾部/reset与参考一致；未部署实体网络 |
| G3训练可信 | 探针通过；正式冻结待完成 | CUDA真实单步/吞吐量通过；预算与验证入口尚未正式冻结 |

Level1：当前图示重建的计算与调度已验证，但CNN精确结构仍未解决，因此不宣称完整论文模型复现；Level2：Hospital数据协议未确认；Level3：没有正式训练结果，尚不能比较0.825/0.895/0.907。所有formal_training_ready均false。

## 验证与产物

- 本地interview新测试24passed/3skipped；加原data/split回归为51passed/3skipped。三个skip都是h5py相关，服务器已经覆盖。
- 本地全项目pytest收集失败：`tests/test_parrad_convert.py`直接import h5py，但interview未安装。未安装依赖，未把全套测试宣称通过；ruff/black也不可用。
- 服务器V2新增测试27passed；真实HDF5、float64保存、无MD拒绝等均运行。没有声称服务器整个原项目测试通过。
- 独立读取153份完整源MD和NPZ，用与生产代码不同的区间填充方式重建标签，并核验fold与摘要，全部通过。
- V2实际执行源码归档SHA256：`2f3067781137e015fdf8e75da76449c96a923730cb669b063eb415978892e7cd`；未改写V1源码或已执行数据。
- 取回包SHA256：`3af0e5d5ad6bb60cbd3b4d8e3d62fcb8f8ca547f9b0dc6fd94c6dedcd854250c`；44个导出文件摘要在本地再次通过。包只含元数据、日志、探针及8个小诊断窗口，完整153份MD-only NPZ仍在服务器，不是完整本地训练数据包。

本地证据：`outputs/split-birnn-preparation-20261014/`，其中 `server-evidence-v2/dataset/audit.json` 是主数据审计，`evidence-v1/independent-data-audit.json` 是独立重算，`probe-v2.json` 是实测吞吐量，`evidence-v2/cuda-equivalence.json` 是真实CUDA等价性。`figures/`含8幅独立PNG。

源码：`src/continuous_mmwave/split_birnn/`；服务器封存：`/home/xibeidaxue/SEER/.audit-staging/split-birnn-preparation-20261014-v2`。实现准备计划与进度账本已更新。下一阶段优先补Hospital/clock来源和CNN结构依据；若无法追溯原版本，需明确采用可用发布数据与当前CNN解释的方法复现协议，再冻结训练配置。当前不自动启动正式训练。

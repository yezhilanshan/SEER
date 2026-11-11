# Split BiRNN 复现设计与论文核对

日期：2026-10-14。状态：用户审查批准进入实现准备阶段；本版按审查修订，正式训练暂不启动。实施与证据见 `docs/superpowers/plans/2026-10-14-split-birnn-preparation.md` 及进度账本。

## 目标与验收

复现 Werthen-Brabants 等人的《Split BiRNN for real-time activity recognition using radar and deep learning》，Scientific Reports 12, 7436 (2022)，DOI: https://doi.org/10.1038/s41598-022-08240-x 。本地全文为工作区根目录同名 PDF，共 11 页；已提取全文，并视觉检查第 5 页算法和第 6 页 CNN 图。

交付独立的 MD-only 数据入口、Split BiGRU、普通 BiGRU 对照、联合训练、逐帧流式推理、后向窗口修正、五折评估、延迟测量、权重及复现报告。先证明模型和流式算法正确，再进行真实数据训练。成功不以强行达到原文 0.907 为标准，而以方法一致、数据与划分可追溯、指标可独立重算为标准。

原文表 3：普通 BiGRU 的 weighted F1 为 0.895，Split 前向为 0.825，窗口后向修正为 0.907，后两种窗口模型使用 L=512、H=171。这些是论文结果，当前没有本项目的复现结果。新受试者分组协议与原文分组方式不明的五折不能直接视为等价。

## 论文明确规定的设置

| 项目 | 设置 | 原文位置 |
| --- | --- | --- |
| 输入 | 每帧 128 Doppler bins 的已发布 MD，帧周期约 0.09 秒 | 第 2、7、8 页 |
| 标准化 | 每一帧在 128 bins 上独立减均值、除标准差 | 第 8 页 |
| CNN | 逐帧 1D 残差 CNN、BatchNorm、ReLU、max pooling，128 bins 压到 16 features | 第 5-6 页、图 4 |
| 时序 | 三层前向 GRU；另有三层后向 GRU，复用前向中间结果 | 第 4-6 页 |
| 分类头 | 两分支共享 fully connected 分类头；后向结果使用最后一层前向和后向状态的均值 | 第 4 页式 6 |
| 联合训练 | 同时训练两分支；随机录制、起点和长度，长度 100-1000 帧 | 第 7 页 |
| 优化器 | Adam，learning rate 0.001 | 第 7 页 |
| 损失 | CCE + Dice；CCE 的平滑标签真类 0.8，其余每类 0.2/(K-1) | 第 6 页 |
| Dice | 1-(2*sum(y*p)+1)/(sum(y)+sum(p)+1) | 第 6 页式 8-10 |
| 数据子集 | PARrad Hospital；论文描述 20 人、每人四次约 10 分钟录制 | 第 7 页 |
| 评估 | fivefold cross validation、weighted F1 | 第 8-9 页 |
| 窗口 | L 在 8-1024 范围，H 为 L 到 L/6 的六档；表 3 使用 512/171 | 第 8-9 页 |

不要直接调用 PyTorch 默认 label_smoothing=0.2 代替上述标签分布，因为默认的均匀分配包含真类，与正文定义不同。九分类目标为真类 0.8、其他类各 0.025。主复现用 Global Dice：对全部有效帧和类别求和；它等价于 `1-(2*sum(p_true)+1)/(2*N+1)`，没有显式类别平衡效果。正文声称处理类别不平衡，与全局归约存在解释歧义。Per-class Dice 和 CCE-only 仅作命名明确的敏感性实验，不能根据测试结果替换主损失。CCE 平滑，Dice 默认使用硬 one-hot；该区分及两分支损失等权求和属于需要记录的实现解释。

## 九类映射

现有 canonical 的原标签 ID 已在服务器读取核对：

| 新 ID | 论文类别 | 原 14 类 ID |
| --- | --- | --- |
| 0 | Walk | 11、12、13 |
| 1 | Fall | 0 |
| 2 | Stand up | 8、9、10 |
| 3 | Sit down | 5、6 |
| 4 | Get in bed | 1 |
| 5 | Lie in bed | 3 |
| 6 | Roll in bed | 4 |
| 7 | Sit in bed | 7 |
| 8 | Get out bed | 2 |

映射依据原文表 1。实现按命令名称验证映射，不能只假定其他 label_map 版本也具有同一 ID。未知命令报错；忽略标签 -100 保留并不参与 loss/metric。训练采用连续片段，不使用按真实动作边界裁剪的单类片段，也不沿用旧实验的 1 秒边界忽略规则。

## 服务器当前证据

- SSH 已登录 `xibeidaxue@10.15.22.62`，主机 `xibeidaxue-X299-WU8`。
- 原始数据：`/home/xibeidaxue/newdisk2/WJ/PARrad/raw`。
- 实时文件调查发现 subject_01 至 subject_23；77 个最终 `processed_data.h5`，未发现 `.part`。这证明本地文件库存，不代表本次已重新计算全部 H5 摘要或确认官方发布完整性。
- 对全部 77 个 H5 仅读取 MD shape、对其 CSV 读取命令后，确认 153 份传感器记录：TI_xWR14xx 512,881 帧、TI_xWR68xx 520,274 帧，共 1,033,155 帧；CSV 共 6,939 行命令，只有上述 14 种名称。双雷达和同一 session 的帧不能算独立采集时长；这些计数与论文 Hospital 表述并不相同。
- 旧转换目录：`/home/xibeidaxue/newdisk2/WJ/PARrad/canonical/offline_expanded_20261011_v1`，只有 15 人、45 session、89 recording 的旧快照。不能把该快照当成现在全部数据；复现应从当前 77 session 单独生成 MD-only 数据版本。
- 原始 H5 有 `/radar/TI_xWR14xx/mDoppler`、`/radar/TI_xWR68xx/mDoppler`，MD shape 为 `[T,128]`；两个传感器帧数可能不同。不要读取巨大的 `cube` 或为缺失传感器补造数据。
- 抽查 subject_01、10、16、20 的首个 H5：root、radar group 未提供环境属性；subject_01 两传感器及 RD/MD dataset 属性也为空。
- 旧 canonical manifest 的 environment_id 为 `parrad`，无法区分 Hospital/Homelab。
- Python：`/home/xibeidaxue/SEER/.conda/envs/seer/bin/python`；实测 Torch 2.4.1+cu121、NumPy 1.26.4、h5py 3.10.0，CUDA 可用。
- GPU：RTX 2080 Ti，11 GiB；调查时仅桌面进程约 270 MiB，35 摄氏度。正式训练前重新检查其他作业。
- 本地测试按用户要求使用 `C:/User/anaconda/envs/interview/python.exe`。工作区没有 Git 仓库，不创建伪造提交或无意义 worktree。

## 阻碍严格数值复现的缺项

1. **Hospital 的发布目录映射未确认。** 官方说明同时存在 Homelab 和 Hospital，当前发布文件与论文描述的 20 人不一致；不能用 session 数量或 subject 编号猜环境。[官方 PARrad 说明](https://www.imec-int.com/en/79GHz-140GHz-radar-systems/parrad)、[作者实验室数据页](https://sumo.ilabt.imec.be/parrad)。官方网页说明下载含技术 README，但本次尚未读到。
2. **官方 README 尚未取得。** 前阶段外部访问被自动审批拒绝；本阶段用户审查具体要求核实官方 README，已据此读取官方公开下载 HTML。两种公开目录接口均返回 HTTP401，不能把页面可访问当成已经取得技术 README。保留请求证据，不访问私有账户或声称取得环境映射。
3. **原五折分组不明。** 正文未明确按 subject、session、recording 还是片段分组；也未公开具体 fold indices。初始推荐按 subject 五折，两个雷达和同一个人的所有 session 跟随同一折，避免同源片段泄漏。应标为我们采用的协议。
4. **训练预算不明。** 正文给出 Adam 和随机片段，但没有 batch size、epochs、每轮 steps、early stopping。明确记录这些补足参数，验证集只从当前外折训练受试者中划出，不能用外折测试挑选轮数。
5. **CNN 图存在尺寸疑点。** 图 4 第二残差块标注主分支 `(16,3)`、`(16,3)`，投影旁支 `(32,1)`；两者无法直接相加。图中也没有逐一标全压到 16 features 所需的池化位置。推荐可运行解释为通道 4 -> 8 -> 16 -> 32 -> 1，第二旁支输出 16，三个 2 倍池化，总长度 128 -> 16；必须作为补足解释记录，先核验作者代码/补充资料是否可获得。
6. **GRU 第一后向层的公式边界需解释。** 式 4 在零层定义可能表示 x 与自身拼接，但正文参数量支持第一后向层直接吃 16 维 CNN features，上层吃前向/后向上一层各 16 维的拼接。若使用常见双 bias GRU，16 hidden 的三层前向 + 16->9 head 为 5049，第一后向输入 16、后两层输入 32 + head 恰为表 2 的 6585。该推断不能当作取得原实现的证据。
7. **时间同步有限。** 当前 converter 使用首条命令时间作为开始、0.09 秒固定时间格。H5 抽查没有实际帧时间戳；当天已有只读审计也标记 clock_calibrated=false。复现可采用并明确这一标签假设，但不能报告真实身体动作开始后的告警延迟。
8. **CNN 参数量仍须审计。** 原文表 2 为 forward 13350、backward 6585、baseline 21462。常见双 bias GRU 下反推 CNN 为 8301 可训练参数；该推断依赖论文参数统计口径相同，不能当作已取得作者 CNN。当前尺寸自洽解释实测 CNN 7283 可训练参数、357 个 buffer 元素（含 BN batch counter），差 1018；forward 12332、backward 含共享 head 6585、baseline 20444。共享 head 的 153 只在唯一模型总参数中计一次。维度和梯度正确不等于原 CNN 精确重建，G1 的结构一致性仍未完全通过。

禁止按最终准确率是否接近 0.907 选择 Hospital 子集或 CNN 解释。CSV 命令条数不直接等于论文动作实例数，也不能简单乘二推定一致。

## 推荐实现设计

建立独立 `src/continuous_mmwave/split_birnn/` 包，配置写入 `configs/split-birnn-*.yaml`，只新增相关测试和说明，不修改原 SEER 编码器和已执行实验。职责分为数据、模型/损失、流式调度、训练、评估及命令入口。

数据准备只读原始 `mDoppler` 和 CSV，检查最终文件、shape、有限值、类别和文件来源。生成新 MD-only NPZ、manifest、9 类映射、来源清单和候选五折。标准化在线执行，不对原值额外 signed-log；零方差帧有限输出。输入 `[B,T,128]`；主实现同批片段长度一致，不支持训练 padding。训练随机片段以零 GRU 状态开始；这是明确补足选择，正文并未单独规定初始化细节。训练使用批量 GRU 跨完整时间维度计算，不用逐帧 Python 调度训练。

每个 recording 强制保留 subject/session/sensor/recording ID、frame_count、valid_frame_count、transition_count、每类有效帧和连续段数、忽略比例。`-100` 仅忽略监督，不能删除真实帧。Temporal Alignment Audit 输出 CSV 时长、帧覆盖时长、命令索引单调/碰撞、超出首尾范围的命令、未标注区间和末标签外推时长。没有实测 origin 时，首条命令=首帧会人为产生零未标注前缀，必须明确是协议结果而非精确同步证据。双雷达标签索引在假定相同时间格下相同也不证明两雷达同钟。固定 train-source 01/02 的随机命令转换可视化只用于发现候选问题，不将谱变化位置认定为真实动作开始。

Split 模型的前向各层保持独立因果状态。缓存每帧 CNN 16 features 与三层前向各 16 状态，共 64 float；每帧 float32 payload 为 256 bytes，对比 128 bins 的 512 bytes。第一后向层倒序处理 CNN features，上层倒序处理前向/后向上一层拼接。末层前后向平均后进入共享 head。联合训练中后向 loss 保留通向前向缓存和 CNN 的梯度，不误 detach。

流式算法从 recording 开始维护前向状态；第一次累计 L 帧时做后向修正，以后每 H 帧修正最后 L 帧。窗口倒序 GRU 从零后向状态开始，前向状态不在窗口间重置。重叠部分按论文“替换之前预测”的规则保留最后一次修正。分别输出 Forward、First-corrected、Final-corrected 三套 logits/指标，以及每帧首次/最终修正端点、等待时长和覆盖次数。最后不足 H 帧不偷偷增加离线 flush；可另列明确标注的 flush 补充结果。评估包含未被修正的帧，Split 使用原前向预测。未修正帧延迟为缺失值，不能记作零。

时间以帧到达为参考，第一帧为 0，首个 512 帧窗口在第 511 帧到达时发出；论文按 512*0.09=46.08 秒描述窗口覆盖时长，本时间戳约定下端点差为 511*0.09=45.99 秒。稳定更新周期为 171*0.09=15.39 秒；最终修正延迟可能远大于该周期，不能把所有帧最终延迟记作 H*0.09。

普通 BiGRU 对照使用相同 CNN 与三层标准双向 GRU；它的高层同时依赖前后向输出，不误用 Split 模型当对照。比较相同九类、fold、按总优化步数计的训练预算，以及 L=512/H=171 的评价窗口。它重新计算每个窗口，Split 前向跨窗口传状态。forward 是已联合训练 Split 的前向分支，不另换训练模型。Full-sequence BiGRU 仅为明确标注的非实时参考，不是数学保证的性能上界。

普通窗口 BiGRU 没有因果 head，对没有窗口覆盖的帧不补造前向预测。保存 unavailable mask；同覆盖范围对比单独报告，完整 recording 指标把未输出帧按 abstention 计入原标签分母和召回，不能静默删除。首次/最终端点及覆盖日程必须与 Split 相同。

正式主评估固定 L=512、H=171；另外扫描 L={8,16,32,64,128,256,512,1024}、H=round(L/k), k=1..6，并保存真实整数 H。扫描作为性能曲线，不能看外折测试选择新的主设置。除 weighted F1 外报告 macro F1、accuracy、混淆矩阵、每类与每受试者/传感器指标；保存逐帧 logits、标签和时间索引，可独立重算。分别列各折均值/标准差及汇总 OOF，避免混用两种聚合。

延迟测量先 warmup，再 CUDA synchronize；分别测单帧 CNN+前向、缓存后向 512 帧、普通 BiGRU 512 帧，报告 P50/P95/均值/标准差和设备信息。不把纯计算延迟、H*0.09 等待周期和网络传输混为一个指标；实际边云分布部署不属于初始复现必需交付。

## 实施与验证顺序

1. 取得官方 README 或可信环境映射，冻结 Hospital manifest；若公共发布版本无法映射，明确改为“在公开 PARrad 上的方法复现”，不能称为严格 Hospital 数值复现。
2. 编写并验证九类映射、逐帧标准化、grouped folds、来源审计及轻量 MD-only 转换；独立核对 CSV 标签。
3. 模型正确性测试：整段前向与逐帧/分块相同；前向未来扰动不影响过去；缓存后向与联合计算相同；共享 head 与平均严格正确；两分支梯度及精确平滑损失有效。
4. 流式测试：首次修正、重叠替换、非整除 L/H、短序列、尾部未修正、跨 recording reset、64-float 缓存及后向 zero state。与朴素窗口实现逐帧对照。
5. 在 interview 中运行相关 pytest；在服务器 seer 环境跑同一测试与真实 MD/CUDA 单步。记录实际命令、输出和权重摘要。
6. 本阶段只运行随机张量及真实训练来源的单步/吞吐量探针，不进入正式训练。长度 100/512/1000，测 batch=8 候选的时间和 CUDA 峰值显存后再决定正式配置；预算以 `max_optimizer_steps`、`validation_interval_steps`、`early_stopping_patience_checks` 为主，随机片段的 epoch 不能暗示遍历全部数据。这三个字段当前为 null，配置 formal_training_enabled=false。任何正式训练还须四个 Gate 成立，并遵守用户暂不启动正式训练的当前指令。
7. validation 选 checkpoint，完成主评估、窗口曲线和 timing；保存每折 best/last、config、train log、provenance、OOF predictions、独立指标重算与最终报告。

正式运行使用全新 server capsule 与输出目录；数据不覆盖，已有权重不改写。需要修改补足参数时增新版本，不覆盖已经看过的测试结果。全过程不安装 APK，也不安装不必要依赖。

## 审查后的阶段验收

| Gate | 条件 |
| --- | --- |
| G0 数据可信 | 来源/九类/时间覆盖/候选 fold 审计完成，Hospital 和近似时间协议状态明确；不能把近似视作校准 |
| G1 结构正确 | CNN 参数/图解释闭合，GRU 维度与共享 head、前向/缓存后向/梯度等价性通过；本轮 CNN 差异尚待解决 |
| G2 流式正确 | 参考实现一致，首次/最终修正、覆盖次数、重叠/首尾和 reset 正确 |
| G3 训练可信 | 真实单步和吞吐量成功，正式预算与验证协议冻结；本轮只准备，不冻结正式启动许可 |

五折统计同时报告各折 weighted F1 的等权平均及样本标准差 `ddof=1`、汇总 OOF confusion 的 pooled F1、每受试者等权指标。每帧只在其外折测试出现一次；两个传感器、同一个人的所有 session 跟随同一角色。固定随机种子42并输出每折来源清单；不能将 OOF pooled 指标冒充 fold 平均。

报告等级：Level1 方法计算与调度复现（CNN 精确性另列）；Level2 Hospital 数据协议确认；Level3 数值接近论文。接近论文数字不替代 Level1/2 的证据。

推荐：先确认官方 Hospital 映射，再执行上述 subject-grouped 五折方法复现；严格标注所有原文缺项与框架差异。如果无法确认原始 Hospital 对应文件，保留同一实现，在全部可验证公共发布数据上报告方法复现，并把它与论文数值复现区分。

用户已批准实现准备。按独立实施计划执行审计、模型/流式测试和探针，当前不进行正式五折训练。未解决的 Hospital、CNN 和时钟依据分别记录，不以较高 F1 掩盖。

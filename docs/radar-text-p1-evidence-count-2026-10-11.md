# P1：动作证据保留与动作数量校准诊断（2026-10-11）

**冻结同一个Align Encoder后，仅训练轻量读出并加强分类后处理：validation选出的预测数量+段Viterbi在严格482个test窗口上达到131/482=27.18%，高于原B3的119/482=24.69%，差+2.49个百分点。**

当前结果支持：原分类流程的动作数量失控与冗余是完整序列差距的重要来源，不能继续把此前B3优势解释为LLM读出独有的序列建模收益。数量校准也没有解决全部问题；Oracle真数量下仍只有34.02%完整序列正确。H→Z未出现本轮探针下的整体退化，Z→P的有序读取变弱，值得后续检查Projector与有序监督；不能据此断言信息被不可逆删除。

## 1. 协议与不可变性

- 使用P0相同的Align权重、传感器RD/MD输入和描述窗口：12秒、步长6秒、最短8秒、最多3个GT动作段。训练受试者02/03/04，验证05，测试06；2187/702/484窗。主结果排除各2个不足12秒尾窗，为700val/482test，原协议全集另列。所有val/test窗口ID、GT与雷达输入摘要逐一等同P0；没有用不重叠特征数字替代滑窗评价。
- H、Z、P均重新提取：H为完整BiGRU逐帧256维输出，Z为原Adaptive Temporal Pooling的8×256，P为原Projector的8×896。只训练6个探针及一个独立数量头；Encoder/Projector冻结且状态逐位不变，Qwen没有加载或训练，B3复用经输入核验的原预测。
- 全部探针权重与后处理候选仅在完整validation选择，于2026-10-11T13:55:52.389909+08:00锁定后才提取test特征。原始输入66份、源文件133份前后SHA核对通过。
- 动作来自`command_active_until_next_command`标签，持续时间不能等同真实身体动作执行时长；只知环境字段parrad，不推断Homelab/Hospital。类别短语抽取、重叠窗口、双传感器及只有一个测试受试者均限制解释。subject06已经多轮观测，结果是现有协议诊断，不构成新独立受试者泛化证据。

## 2. P1-A：数量、类别/顺序与编辑错误

| 方法 | 序列正确 | 数量正确 | Count MAE | 数量正确条件下Sequence Exact | 平均编辑距离 | 插入/删除/替换 | 歧义窗口 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | 91/482=18.88% | 172/482=35.68% | 1.098 | 52.91% | 2.037 | 476/63/443 | 172 |
| B3 | 119/482=24.69% | 321/482=66.60% | 0.336 | 37.07% | 1.606 | 19/155/600 | 94 |

B1数量正确时类别与顺序表现更好，但这种条件比较筛选了不同窗口，不能作为控制同一困难度后的因果结论。B3数量更准确且插入少，同时删除、替换均明显存在。单位代价Levenshtein按GT→预测计算，确定性优先match/sub/delete/insert；替换不代表已找到对应物理时段，插入/删除分解受最优路径歧义影响。动作Recall为窗口内带重复次数的类别多重集覆盖率，不是时间IoU检测Recall。

| 独对组 | 仅插入 | 仅删除 | 仅替换 | 混合 | 错误路径数量已正确 | 编辑歧义 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1_only (29) | 0 | 7 | 17 | 5 | 17 | 4 |
| B3_only (57) | 29 | 3 | 13 | 12 | 13 | 17 |

已逐条核查全部29个B1独对与57个B3独对窗口的符号/文本记录。29窗中B3有17窗只替换、7窗只删除、5窗删除+替换；57窗中B1有29窗只插入、11窗插入+替换，另有13窗只替换、3窗只删除、1窗删除+替换。因此，“Qwen全是漏动作”或“分类全是多动作”都不成立。
完整记录：[86窗复核表](../outputs/radar-text-p1-20261011-v1/derived/paired-window-review.md)、[CSV](../outputs/radar-text-p1-20261011-v1/derived/paired-window-review.csv)。以下为代表性符号证据，未做人工雷达时段判断：

| sample_id | GT | 错误路径 | 观察 |
| --- | --- | --- | --- |
| e816907bbacd:6000:6134 | 9: stands up from the chair → 11: walks toward the bed → 1: gets into bed | B3: 9: stands up from the chair → 11: walks toward the bed | 省略末端gets into bed |
| 67d5d9de6e50:4200:4334 | 9: stands up from the chair → 13: walks toward the room → 0: falls onto the floor of the room | B3: 9: stands up from the chair → 13: walks toward the room | 省略末端fall |
| 788e53fe3491:2134:2267 | 1: gets into bed → 3: lies in bed → 4: rolls in bed | B3: 7: sits in bed → 8: stands up from the bed → 13: walks toward the room | 数量正确但三个类别全替换 |
| 788e53fe3491:3134:3267 | 12: walks toward the chair → 6: sits down on the chair → 9: stands up from the chair | B1: 12: walks toward the chair → 13: walks toward the room → 11: walks toward the bed → 12: walks toward the chair → 6: sits down on the chair → 10: stands up from the floor of the room → 9: stands up from the chair | 正确序列中插入4个冗余动作 |
| 37fed0eda030:2600:2734 | 4: rolls in bed → 2: gets out of bed | B1: 2: gets out of bed → 4: rolls in bed | 数量正确但Roll/Get-out顺序反转 |
| 67d5d9de6e50:2534:2667 | 9: stands up from the chair → 11: walks toward the bed → 5: sits down on the bed | B1: 9: stands up from the chair → 11: walks toward the bed → 6: sits down on the chair | 数量正确但Sit-bed换成Sit-chair |

## 3. P1-B：H→Z→P轻量读出

H保留全部真实帧，按训练集最大134帧补零后展平34304维；Z展平2048维，P展平7168维。每坐标mean/std仅从训练窗计算，std下限1e-5。线性读出为无激活的两层因子化线性映射（rank≤32）；非线性在相同32维瓶颈加入GELU。它们是有限规格探针，不是最优线性可分性上界。H/Z/P输入层参数不同，尤其H有约110万参数而训练仅2187窗；统一训练预算不保证相同优化难度或最佳泛化。
三任务联合监督：14维Presence BCE（训练集posweight）+3维Count CE（训练集逆频率权重）+3×15个Ordered槽位CE（PAD14，包含PAD损失）。最多3动作，非相邻重复保留，如Roll→Sit→Roll；PAD终止输出，不折叠重复。Count是独立任务输出，与Ordered的PAD长度不强制一致。AdamW lr.001/wd.01、20epoch、batch64、clip1、seed17；每个探针按validation Ordered Exact→Presence Macro-F1→Count Accuracy选权重，平分选最早。

| 特征/读出 | 参数数 | 选epoch | Val Presence F1 | Val Count Acc | Val Ordered Exact | Test Presence F1 | Test Count Acc | Test Ordered Exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H-linear | 1,099,806 | 12 | 49.60% | 62.71% | 20.71% | 48.49% | 71.16% | 101/482=20.95% |
| H-nonlinear | 1,099,806 | 18 | 53.34% | 66.43% | 21.29% | 51.75% | 74.48% | 102/482=21.16% |
| Z-linear | 67,614 | 15 | 54.83% | 65.14% | 24.00% | 53.47% | 75.93% | 118/482=24.48% |
| Z-nonlinear | 67,614 | 15 | 56.94% | 66.43% | 25.86% | 53.77% | 77.59% | 122/482=25.31% |
| P-linear | 231,454 | 20 | 51.30% | 52.71% | 20.57% | 52.20% | 67.01% | 106/482=21.99% |
| P-nonlinear | 231,454 | 19 | 52.46% | 62.14% | 22.14% | 53.83% | 76.97% | 102/482=21.16% |

![H Z P探针对比](../outputs/radar-text-p1-20261011-v1/derived/evidence-probes.png)

- H→Z：Z的Ordered Exact在两种探针下都高于H，没有出现“H好、Z差”的现象，本轮证据不支持优先扩大K来修复池化证据丢失。H并不优秀，但高维探针的优化/泛化差异也可能解释差距，不能据此判定Encoder缺信息。
- Z→P：测试线性Ordered Exact 24.48%→21.99%，非线性25.31%→21.16%；validation也同向。P的Presence F1与Z接近，说明“类别出现”与“有序段序列”应分开诊断。优先检查Projector的有序信息可读性、文本对齐目标与读出优化，不能直接证明信息物理上被删除。
- P探针并未全面优于Qwen，没有得到“H/Z/P都好但Qwen差”的清晰证据。探针与Qwen容量、目标、输出约束不同，当前结果无法单独把问题归因于EOS或LLM；可以后续用固定P进行覆盖/EOS受控实验。

## 4. P1-C：概率、先验与预测数量校准

在原AlignH1逐帧logits上比较概率均值平滑0/.3/.5秒×最短段.6/1/2秒×转移权重0/.25/.5/1，共36组（其中weight0为9组概率基线）。转移与开始先验只统计24个训练完整sensor-record，包含帧自转移并Laplace1；HMM用帧Viterbi。最短段过滤丢弃短预测段、合并连续同类，不强制GT数量。概率均值对类别重编号等变，避免class-ID中值的任意顺序问题。
独立数量头使用H按8个时序bin均值后展平→32维GELU→3类，65,667参数；输入数学上与Z相同，独立指训练任务/头，并非另一种信息源。训练与上述预算相同，仅训练countCE，validation Count Accuracy→负MAE选择epoch7。随后段Viterbi在预测K∈1/2/3约束下划分完整窗口，允许非相邻同类重复、相邻同类禁止，最短段按真实时间戳计算；使用每段类别的log概率和转移分数，不输入GT边界或数量。
固定候选及实际validation选择如下。主方法在probability/HMM/predicted-count三个可部署方案中，按同一validation Sequence Exact→LCS→Recall选择；Oracle从不参与选优。

| 流程 | 概率平滑秒 | 最短段秒 | 转移权重 |
| --- | ---: | ---: | ---: |
| probability | 0.0 | 1.0 | 0.0 |
| hmm | 0.0 | 0.6 | 1.0 |
| count | 0.0 | 1.0 | 1.0 |

验证最优概率窗口为0秒，因此该行改善来自1秒最短段过滤，不能宣称概率平滑本身带来收益。HMM与数量校准改变整个读出流程，其效果不能拆成单独LLM的因果贡献。

| split | 流程 | 完整序列正确 | 数量准确率 | Count MAE | 少/多/等数量窗 | Recall | LCS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 原AlignH1+平滑模板 | 115/700=16.43% | 35.00% | 1.151 | 76/379/245 | 62.31% | 48.73% |
| validation | 原Align+Projector+Qwen | 141/700=20.14% | 56.14% | 0.446 | 231/76/393 | 45.55% | 43.55% |
| validation | 概率/最短段过滤 | 128/700=18.29% | 45.00% | 0.711 | 150/235/315 | 57.60% | 50.40% |
| validation | 训练转移先验HMM | 161/700=23.00% | 54.00% | 0.519 | 147/175/378 | 56.52% | 52.03% |
| validation | 预测数量+段Viterbi | 169/700=24.14% | 68.00% | 0.330 | 130/94/476 | 55.95% | 52.90% |
| test | 原AlignH1+平滑模板 | 91/482=18.88% | 35.68% | 1.098 | 57/253/172 | 65.91% | 52.25% |
| test | 原Align+Projector+Qwen | 119/482=24.69% | 66.60% | 0.336 | 148/13/321 | 46.06% | 45.37% |
| test | 概率/最短段过滤 | 99/482=20.54% | 47.72% | 0.624 | 120/132/230 | 58.71% | 52.83% |
| test | 训练转移先验HMM | 123/482=25.52% | 60.58% | 0.413 | 94/96/292 | 59.13% | 55.09% |
| test | 预测数量+段Viterbi | 131/482=27.18% | 78.63% | 0.216 | 72/31/379 | 57.68% | 55.60% |

![完整序列与数量误差](../outputs/radar-text-p1-20261011-v1/derived/count-calibration.png)

主分类流程由B1的91/482升至131/482（+8.30个百分点），多预测窗从253降至31；对应Recall从65.91%降至57.68%，但仍高于B3的46.06%。数量控制在抑制冗余时也会损失覆盖，不能只看Exact。HMM123/482本身已略高于B3的119/482；这说明结果并不只依赖新增数量头。
逐类实例覆盖与探针Ordered覆盖见[完整14类CSV](../outputs/radar-text-p1-20261011-v1/derived/test-per-class-coverage.csv)。

## 5. 数量分布与事后多数基线

严格val的GT数量1/2/3分别为4/222/474，test为2/90/390。训练多数类别为3；固定预测3在test的数量准确率80.91%，高于数量头78.63%，六个联合探针的Count Accuracy也均低于此多数基线。因此这些总体计数准确率不能被解读为强实例级计数能力，尤其1动作只有2个test窗。应同时查看每类计数Recall、MAE和完整序列。

| split | 数量头Count Acc / MAE | train多数Count Acc | 数量头完整序列 | 固定train多数+同解码器完整序列 |
| --- | ---: | ---: | ---: | ---: |
| validation | 68.00%/0.330 | 67.71% | 169/700=24.14% | 135/700=19.29% |
| test | 78.63%/0.216 | 80.91% | 131/482=27.18% | 116/482=24.07% |

固定多数诊断在查看主结果并发现计数不平衡后补充，明确是事后分析。K只取训练多数，不使用测试GT；使用原已锁定的预测数量解码器参数，没有重新调参、选权重或替换主方法。它用于检查数量头是否超出分布先验，不能包装为预注册正式新测试结果。数量头的test完整序列131/482仍高于固定多数116/482（+3.11个百分点），validation亦同向；这表明总体计数准确率与下游正确序列不等价，也为自适应计数提供了有限支持。

| split | 真数量=1支持/召回 | 真数量=2支持/召回 | 真数量=3支持/召回 |
| --- | ---: | ---: | ---: |
| validation | 4/25.00% | 222/56.76% | 474/73.63% |
| test | 2/0.00% | 90/66.67% | 390/81.79% |

## 6. Oracle Count：仅诊断

| split | Oracle Sequence Exact | Count MAE | Recall | LCS |
| --- | ---: | ---: | ---: | ---: |
| validation | 219/700=31.29% | 0.000 | 57.07% | 56.60% |
| test | 164/482=34.02% | 0.000 | 59.82% | 59.41% |

Oracle使用真实动作段数，所以不能计入正式测试性能。它沿用预测数量流程已选好的参数，未按Oracle结果调参；该固定解码器下164/482比预测数量131/482高33窗（6.85个百分点）。这是剩余计数错误的诊断机会量，不是任意分类/语言方法的理论上限；即使提供真数量，318/482仍错误，类别、顺序、标签与解码均仍有改进空间。

## 7. 原始484窗口补充

| 流程 | 完整序列正确 | Count MAE |
| --- | ---: | ---: |
| 原AlignH1+平滑模板 | 93/484=19.21% | 1.093 |
| 原Align+Projector+Qwen | 121/484=25.00% | 0.335 |
| 概率/最短段过滤 | 101/484=20.87% | 0.622 |
| 训练转移先验HMM | 125/484=25.83% | 0.411 |
| 预测数量+段Viterbi | 133/484=27.48% | 0.215 |
| Oracle真数量（诊断） | 166/484=34.30% | 0.000 |

## 8. 瓶颈定位与下一步

1. 优先保留预测数量+段Viterbi为更强分类对照，同时保留无需数量头的HMM。当前完整序列优势已不能作为直接生成优于强分类读出的证据；比较针对完整流程，不能单独推断LLM无价值或已量化其因果贡献。
2. 数量头受3动作先验影响明显，下一轮先用按数量分层的验证、计数Macro-F1/Recall及新的受试者检查，再考虑复杂计数网络。不要只追求该测试集的总体Count Accuracy。
3. 暂无优先扩大K的探针证据。针对Z→P的有序读取下降，先在冻结Encoder下检查Projector/有序目标与覆盖监督；同预算多随机种子、规格更充分的线性探针可以区分优化与表示因素。P与Qwen不构成干净因果对照。
4. 仍需真实雷达时段人工复核和新的受试者验证，尤其床相关类别、走向目的地混淆与末端动作；文本模板监督不能证明速度、幅度等细粒度自然语言能力。重叠窗/双传感器不是独立事件，本轮不作独立样本显著性声明。

## 9. 复核与产物

- 本地interview相关测试30项、服务器原生相关测试74项通过；新增6项包含重复动作、编辑歧义、类别置换等变、数量约束与小规模穷举最优解一致性。原始测试日志在[归档](../outputs/radar-text-p1-20261011-v1/provenance/native-tests.log)。
- 独立脚本重新计算56份指标汇总、三任务预测指标、模型/候选选择、全部86个独对记录及29个归档文件SHA：[本地复核](../outputs/radar-text-p1-20261011-v1/derived/local-verification.json)。
- 服务器直接检查H→Z池化、原Projector(Z)、训练均值/标准差、所有探针预测及所有校准路径重建：[缓存复核](../outputs/radar-text-p1-20261011-v1/derived/server-cache-audit.json)。Z最大误差0，P最大误差6.68e-06；重建8302个探针窗口输出、1186个校准窗口。解码重建复用生产实现，独立指标与穷举测试另列，不把它冒称另一套独立解码器。
- [固定选择及全部候选](../outputs/radar-text-p1-20261011-v1/experiment/selection.json)、[探针训练历史](../outputs/radar-text-p1-20261011-v1/experiment/probe-training.json)、[P1-A](../outputs/radar-text-p1-20261011-v1/experiment/p1-a.json)、[P1-B](../outputs/radar-text-p1-20261011-v1/experiment/p1-b.json)、[P1-C](../outputs/radar-text-p1-20261011-v1/experiment/p1-c.json)、[模型与输入溯源](../outputs/radar-text-p1-20261011-v1/experiment/provenance.json)。
- [探针CSV](../outputs/radar-text-p1-20261011-v1/derived/probe-results.csv)、[后处理CSV](../outputs/radar-text-p1-20261011-v1/derived/calibration-results.csv)、[计数混淆表](../outputs/radar-text-p1-20261011-v1/derived/independent-count-confusion.csv)。
- [最终只读审查](../outputs/radar-text-p1-20261011-v1/derived/readonly-review.md)通过，无阻断或需修改问题；审查员复跑新增6项测试通过。首次额外缓存校验器因重复读取NPZ整成员导致内存不足，修复为一次加载后在独立v2审计通过，主实验与原归档没有变化：[修复记录](../outputs/radar-text-p1-20261011-v1/derived/cache-audit-repair.json)。
- 服务器原始H/Z/P三份缓存保留在`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/radar_text_p1_20261011_v1/features`，本地只取回小型探针、逐窗预测和结果，不下载原始雷达或Qwen。原结果归档SHA256：`a07a6276db40ad6f4fca9f87657a59fbf2d546bf0dcd2926d63fe7308fb23be0`。

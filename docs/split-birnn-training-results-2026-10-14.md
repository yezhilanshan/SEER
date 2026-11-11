# Split BiRNN 五折训练结果与完成检查（2026-10-14）

**结论：五折 Split BiGRU 与普通 BiGRU 共 10 次正式训练均正常完成；独立指标重算、全部测试记录的权重回放、日志、检查点、采样和 BN 检查通过。** 实验名称为 `split_birnn_reconstructed`，属于已声明 CNN 与近似时间标签下的方法复现，尚不能声称严格复现原论文数值。

双方共有修正输出帧上的五折平均 Weighted F1：Split 最终修正 **80.06% ± 5.33%**，普通 BiGRU **79.30% ± 5.49%**，平均差 **+0.76 个百分点**。Split 在 3/5 折更好、2/5 折更差；不能据此断言稳定或统计显著优于普通 BiGRU。即时前向输出平均只有 **58.23%**，最终修正成绩依赖未来信息。

服务器数据根目录：`/home/xibeidaxue/newdisk2/WJ/PARrad/canonical/split_birnn_preparation_20261014_v1`。训练根目录：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/split_birnn_fivefold_20261014_v3`。本次服务器审计时间为 `2026-10-14T13:33:33.189548+00:00`（北京时间 2026-10-14 21:33）。

## 1. 实际完成状态与验收

每个运行 6400 optimizer steps，共 **64000** 正式训练步；均未触发提前停止。每次有 50 次验证和 100 次 BN 诊断；checkpoint 依据本折验证集 `matched/final` pooled Weighted F1 选择，同分时选更低验证 Combo loss。独立重新扫描全部验证日志，所选步与保存的最佳检查点一致。

- 23 名受试者、77 session、153 雷达记录、1,033,155 个原始有效标签帧。train/validation/test 在每折按 subject 严格互斥，同一 session 的雷达不会跨角色。
- 每个模型的 153 个记录恰好各作一次 outer test。测试覆盖全部五折，未根据测试指标重选模型、标签偏移或窗口参数。
- 独立读取全部 **306** 个测试 NPZ，核对哈希与 canonical 标签/时间戳，重建逐记录及汇总的六套混淆矩阵和 Weighted F1，全部一致；十组保存的验证预测另行独立重算通过。
- 重载十个 `best.pt`，在相同 CUDA/确定性设置下回放全部 **306** 个测试记录；forward/first/final logits 与原始保存结果逐值一致（baseline 无输出位置保留 NaN），修正次数、端点和延迟也一致。所有推理在 `eval()` 下进行，BN 缓冲未变化。
- 十个 `last.pt` 都包含 model、Adam、scheduler、sampler、Python/NumPy/Torch/CUDA RNG、step、最佳状态与 provenance；模型参数及 Adam 状态有限，Adam 各参数步数为 6400。完整恢复的逐值等价证据来自训练前 pilot 的 150→300 步恢复实验，正式运行本身未模拟另一次恢复。
- 64000 步损失和各组梯度均有限；CNN、前向/后向 GRU 与共享 head 每一步都有正的梯度绝对值总和。Split 总损失等于两个 branch Combo loss **之和**，baseline 是一个 Combo loss；总值不可跨模型直接比较。
- frozen 配置、实际源码、canonical 文件与数据/fold manifest 的哈希均通过。下载到本机的 **1298** 个文件逐个核验大小与 SHA256，通过。

训练前相关测试：本机 60 passed / 3 h5py skips；服务器 V3 原生 36 passed。上述完成检查是本次实际运行的读回证据。未宣称整个 SEER 测试套件通过，本机全套收集仍受现有 h5py 缺失影响。

## 2. 固定训练与评价协议

MD-only 128 bins，每帧 population z-score，无 signed-log；重建残差 CNN 输出 16，三层前向 GRU/三层后向 GRU，hidden=16，共享 16→9 head，最终状态平均后分类。CNN trainable=7283，Split 独立 trainable=18764，普通 BiGRU=20444。

Adam LR=0.001，恒定 scheduler，batch=8，每批共同片段长度均匀取 100..1000；recording 均匀抽样、批内不重复，再抽片段起点；训练片段状态从零开始。同折两个模型的 **6400 批长度、记录与起点序列完全相同**。不做类别重采样。

流式评估保持前向状态；后向每个窗口从零开始。`L=512, H=171, frame_interval=0.09s`；完整窗口到齐后修正；first 保留首次修正，final 按最后一次覆盖更新，末尾不额外 flush。Split 未覆盖帧沿用前向输出，baseline 未覆盖帧为 abstention。

`matched` 只比较共同有修正输出的帧，共 1,027,728 帧，覆盖率 **99.4747%**；`full` 使用全部 1,033,155 帧，baseline 无输出计入第十列 abstention。Split full 预测覆盖率为 100%，baseline 为 99.4747%，两者有 **5427** 帧的输出覆盖差。baseline 没有即时前向分类头，其统计文件中的 forward=0 是全 abstention，不能当作一个真实前向模型成绩。

|输出|matched 五折均值 ± 样本 SD (%)|matched 合并 OOF F1 (%)|full 五折均值 ± 样本 SD (%)|
|---|---|---|---|
|Split forward|58.23 ± 5.83|58.56|58.10 ± 5.92|
|Split first correction|75.47 ± 5.70|75.73|75.24 ± 5.83|
|Split final correction|80.06 ± 5.33|80.27|79.81 ± 5.46|
|BiGRU first correction|76.25 ± 5.74|76.53|76.03 ± 5.83|
|BiGRU final|79.30 ± 5.49|79.58|79.08 ± 5.58|

五折均值是五个 fold F1 等权平均，SD 使用 `ddof=1`；合并 OOF 指标先累加混淆矩阵再算 F1，两者不能混用。以下逐类/受试者/传感器和诊断表均用合并 OOF 口径。

首次修正时普通 BiGRU 五折平均为 76.25%，高于 Split 的 75.47%；Split 的平均优势只在本协议的最终修正视图出现，不应概括为任何修正时刻都更好。

|Fold|Split 最佳步|BiGRU 最佳步|Split final (%)|BiGRU final (%)|差值 pp|
|---|---|---|---|---|---|
|0|5888|6144|74.11|71.79|+2.32|
|1|6400|5376|74.57|75.14|-0.58|
|2|5760|6016|83.25|83.55|-0.30|
|3|6400|5760|85.59|83.72|+1.87|
|4|5888|5888|82.78|82.31|+0.47|

![fold-and-class-f1](D:/lab/SEER/outputs/split-birnn-controlled-20261014/figures/fold-and-class-f1.png)

## 3. 训练收敛与 BN

|Fold|模型|前128步平均loss|后128步平均loss|固定样本train特征std|固定样本eval特征std|
|---|---|---|---|---|---|
|0|split|5.5725|3.6066|0.9945|1.1034|
|0|baseline|2.6481|1.4872|1.0437|1.1784|
|1|split|5.6656|3.6522|1.0116|1.0449|
|1|baseline|2.7351|1.5509|0.9402|1.0031|
|2|split|5.7956|3.7908|0.9455|0.9409|
|2|baseline|2.7349|1.5508|1.1864|1.2278|
|3|split|5.6622|3.7689|1.1351|1.3825|
|3|baseline|2.7447|1.5758|1.0520|0.9397|
|4|split|5.5135|3.6494|0.9741|1.0400|
|4|baseline|2.6713|1.5354|0.9624|0.9849|

全部运行的平均训练损失下降。BN running mean/variance 均有限、variance 正、最终计数为 6400；100 次中间诊断同样通过，验证没有更新 BN。特征 train/eval 对比使用 CNN 克隆，正式模型不被诊断修改。二者分布仍有差异，如 fold3 Split std 1.1351→1.3825；“有限且可回放”不能解释为训练/推理统计完全相同。

验证曲线存在波动，个别点明显下降后恢复；最佳验证选点避免直接使用最后一步退化的权重。Split fold1/3 最佳点为第6400步，因此预算内完成不代表已证明充分收敛。本次保持冻结预算，未基于测试结果延长训练。

![learning-curves](D:/lab/SEER/outputs/split-birnn-controlled-20261014/figures/learning-curves.png)

训练期间曾记录 GPU 88°C，使用只针对本实验已核对 PID 的暂停/恢复降温控制器；最终阈值为 80°C 暂停/75°C 恢复，0.5 秒轮询，仍有温度跃升/测量滞后。没有改变优化步、种子、采样或 GPU 全局功率/风扇配置；`elapsed_s` 含冷却等待。检查时训练与控制器进程均已退出，GPU 空闲。

## 4. 逐类结果与后向修正收益

|类别|matched 标签帧|Forward F1 (%)|First F1 (%)|Final F1 (%)|BiGRU final F1 (%)|
|---|---|---|---|---|---|
|Walk|346107|75.49|84.95|86.88|86.73|
|Fall|77385|40.89|70.20|75.82|75.01|
|StandUp|203629|72.07|78.54|81.06|79.16|
|SitDown|123169|51.07|71.95|78.86|77.32|
|GetInBed|62346|41.95|73.98|81.01|77.96|
|LieInBed|45127|32.25|58.02|63.72|65.52|
|RollInBed|61190|42.40|68.65|76.80|77.66|
|SitInBed|46163|9.26|48.56|55.11|66.22|
|GetOutBed|62612|45.28|71.36|82.61|75.02|

Fall、床相关动作的前向识别较弱，未来信息带来的提升较大。Split 最终对 SitInBed 的 F1 为 55.11%，低于 baseline 的 66.22%；LieInBed、RollInBed 也低于 baseline，所以优势并非对所有类别一致。

![confusion-matrices](D:/lab/SEER/outputs/split-birnn-controlled-20261014/figures/confusion-matrices.png)

`improve` 是 forward 错、修正对；`harm` 是 forward 对、修正错；分母为该区域所有原始有效标签帧（包含未修正但沿用前向的帧）。净收益是 **accuracy** 差，不是 F1 差。stable 排除原始 CSV 命令内部边界 ±10 帧；boundary 是其补集，只代表远离/接近命令，不证明物理动作稳定。

|修正/区域|分母帧|改对帧|改错帧|改对率 (%)|改错率 (%)|净accuracy增益 pp|
|---|---|---|---|---|---|---|
|first/all|1033155|207497|49512|20.08|4.79|15.29|
|final/all|1033155|246591|43818|23.87|4.24|19.63|
|final/stable|748653|181700|19650|24.27|2.62|21.65|
|final/boundary|284502|64891|24168|22.81|8.49|14.31|

最终共改对 **246591** 帧，也改错 **43818** 帧。九类净 accuracy 收益均为正，RollInBed 改错率约 9.50%，值得关注；完整逐类 improve/harm 计数见独立审计 JSON。

补做了窗口级逐帧回放：748653 个 stable 帧中，39451 个至少发生一次“先正确、下一窗口变错”；**108** 个发生至少两次这样的正确→错误转换。在 478031 个 forward 原本正确的 stable 帧中，**17811** 个至少两次修正输出错误，19650 个最终仍错误。“重复错误输出”可能是一次改错后持续错误，与“反复正确→错误转换”是两种统计。每记录 harm transition/wrong emission 数组已保存；重新计算的窗口 improve/harm 与原日志逐项一致。

## 5. 标签可靠性诊断（不用于选点）

原始正式标签保持不变，偏移只在诊断副本上进行，移出的端点变 -100，禁止循环回卷。负/正 shift 分别表示标签前移/后移。

|偏移帧|偏移秒|Split full final OOF F1 (%)|BiGRU full final OOF F1 (%)|
|---|---|---|---|
|-10|-0.90|75.86|75.35|
|-6|-0.54|78.60|78.02|
|-3|-0.27|79.79|79.15|
|0|+0.00|80.03|79.36|
|3|+0.27|79.39|78.69|
|6|+0.54|77.97|77.21|
|10|+0.90|75.16|74.31|

|排除命令边界 ±帧|保留帧|Split full final OOF F1 (%)|BiGRU full final OOF F1 (%)|
|---|---|---|---|
|3|937938|82.60|81.87|
|6|856485|84.20|83.41|
|10|748653|85.41|84.53|

不偏移时 Split full final OOF 为 80.03%；±10 帧时降至 75.16%–75.86%。排除 ±10 帧命令边界后为 85.41%，说明错误明显集中于命令附近。诊断没有据此修改标签或声明真实动作同步精度。

最后命令至录制结束的外推尾部共 **3901 帧**，修正覆盖 **1912/3901=49.01%**。full-tail F1：Split 36.84%、baseline 30.09%；仅共同有修正的尾部帧：Split **46.14%**、baseline **49.66%**。非尾部 full final：Split 80.20%、baseline 79.51%。尾部错误混合了未覆盖帧、动作难度和近似标注，不能单凭低 F1 判定标签错误。末命令外推最长约20.23秒的问题仍未被物理同步证据消除。

## 6. 修正等待时间

延迟从被预测帧的到达时间算到覆盖它的窗口最后一帧到达时间，仅统计有修正的 1,027,728 帧。两个模型同一窗口调度，统计相同。

|输出|平均秒|P50秒|P95秒|最大秒|
|---|---|---|---|---|
|first|8.82|8.10|15.84|45.99|
|final|37.17|37.89|45.18|45.99|

Forward 在帧到达后可立即计算；First 平均等待 **8.82s**，Final 平均 **37.17s**。这是 `L512/H171` 与近似0.09s帧间隔造成的未来信息等待，不是 GPU 运算耗时，也不是动作真实起点延迟；没有测量端到端部署延迟。较高 Final F1 不等于在这些动作发生时已即时作出正确判断。

![correction-diagnostics](D:/lab/SEER/outputs/split-birnn-controlled-20261014/figures/correction-diagnostics.png)

## 7. 数据分组、采样和受试者差异

|Fold|train subjects|validation subjects|test subjects|
|---|---|---|---|
|0|01,02,03,04,05,06,09,12,13,14,15,20,21,22|07,08,10,16|11,17,18,19,23|
|1|01,02,03,05,06,09,12,13,14,15,17,20,21,22|11,18,19,23|04,07,08,10,16|
|2|02,03,04,05,07,08,09,10,14,15,16,17,21,22|11,18,19,23|01,06,12,13,20|
|3|01,02,04,06,07,08,09,10,12,13,14,16,17,20,21|11,18,19,23|03,05,15,22|
|4|01,03,04,05,06,07,08,10,12,13,15,16,17,20,22|11,18,19,23|02,09,14,21|

recording 均匀并不等于 subject 均匀，因为每人记录数量不同。每折保存预期 subject/recording 概率与实际抽样计数；本轮最大 subject 概率偏差约 **0.3175 个百分点**，最大 recording 概率偏差约 **0.1344 个百分点**，同折两模型完全相同。实际抽到的 Walk 帧占32.33%–33.86%，未进行类别均衡；完整类别帧比例与抽样帧总数在审计 JSON。

传感器 matched final pooled F1：TI_xWR14xx Split 80.68% / baseline 80.07%；TI_xWR68xx Split 79.87% / baseline 79.09%。受试者如下，不代表二十三次独立随机实验。

|Subject|Split matched final (%)|BiGRU matched final (%)|
|---|---|---|
|subject_01|86.53|85.26|
|subject_02|79.75|80.54|
|subject_03|83.42|83.19|
|subject_04|86.11|87.67|
|subject_05|83.51|81.81|
|subject_06|87.71|89.19|
|subject_07|83.01|77.80|
|subject_08|80.03|80.91|
|subject_09|86.55|83.06|
|subject_10|60.63|63.66|
|subject_11|70.16|60.45|
|subject_12|71.00|72.30|
|subject_13|89.41|88.42|
|subject_14|79.76|78.75|
|subject_15|90.99|88.89|
|subject_16|45.87|49.76|
|subject_17|49.88|46.33|
|subject_18|64.16|59.77|
|subject_19|86.39|87.20|
|subject_20|82.56|83.32|
|subject_21|86.27|86.97|
|subject_22|84.02|79.80|
|subject_23|79.86|79.76|

## 8. 复现范围与产物

[原论文](https://doi.org/10.1038/s41598-022-08240-x)报告 forward/baseline/Split Weighted F1 约0.825/0.895/0.907。此处数据为服务器可用23人集合，尚未确认原论文20人 Hospital 子集；CNN 7283 比推导目标8301少1018个参数；时间轴采用第一命令作为首帧原点与0.09s间隔，实际时钟未校准。本次 batch、预算及验证策略有明确冻结记录，不能把这组不同协议的结果与原文数字作严格差额验收。已达到真实数据五折方法实验，**严格原论文结构/数据/数值复现尚未完成**。

本轮可用证据：

- [独立测试指标及全部诊断](D:/lab/SEER/outputs/split-birnn-controlled-20261014/exported-server-evidence/evidence/formal-independent-audit.json)
- [训练完成检查与306次权重回放](D:/lab/SEER/outputs/split-birnn-controlled-20261014/exported-server-evidence/evidence/post-training-check/finished-training-check.json)
- [本机20个检查点参数、provenance与scheduler核验](D:/lab/SEER/outputs/split-birnn-controlled-20261014/local-checkpoint-verification.json)
- [尾部与采样补充诊断](D:/lab/SEER/outputs/split-birnn-controlled-20261014/exported-server-evidence/evidence/extra-diagnostics.json)
- [正式冻结配置](D:/lab/SEER/outputs/split-birnn-controlled-20261014/exported-server-evidence/formal/frozen-config.json)，[正式冻结五折](D:/lab/SEER/outputs/split-birnn-controlled-20261014/exported-server-evidence/formal/frozen-folds.json)
- [训练前完整恢复证据](D:/lab/SEER/outputs/split-birnn-controlled-20261014/exported-server-evidence/pilot/pilot-proof.json)
- [全部导出文件哈希清单](D:/lab/SEER/outputs/split-birnn-controlled-20261014/exported-server-evidence/finished-artifacts-manifest.json)，[本机下载核验](D:/lab/SEER/outputs/split-birnn-controlled-20261014/export-verification.json)
- [全部权重、日志、预测、源码与审计解压目录](D:/lab/SEER/outputs/split-birnn-controlled-20261014/exported-server-evidence)。2026-10-18核对全部内容哈希后清除了重复下载压缩包，文件映射保留在[清理审计记录](../outputs/workspace-cleanup-20261018-v1/archive-audit-final.json)。

压缩包 SHA256：`7508513ec60b06b9fb7dbdb259d9dea14391c84273d966e7741d5331f491b0c8`，363846396 bytes；解压后1298文件共515750888 bytes已逐个核验。完整原始MD数据保留在服务器，压缩包携带数据manifest与来源哈希。

每折权重保存在本机 `exported-server-evidence/formal/foldN/{split,baseline}/{best,last}.pt`：`best.pt` 用于推理，`last.pt` 用于按完整状态继续训练。示例 [fold0 Split best.pt](D:/lab/SEER/outputs/split-birnn-controlled-20261014/exported-server-evidence/formal/fold0/split/best.pt)；其它折使用同样路径。逐帧 logits/标签/时间戳/修正覆盖/首次及最终延迟在各模型的 `test-predictions/` 与 `validation-predictions/`。

图表同时保存PNG与可导出PDF；本轮没有追加训练、修改冻结源码、调整测试标签或依据测试成绩重选任何设置。

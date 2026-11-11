# HARrad Dynamics与语义联合预训练：启动与进度

2026-10-18更新：全部12组已于04:59结束；本地全量下载、48检查点读回及12组逐点独立复算通过。最终比较和限制见[结果报告](harrad-joint-prediction-results-2026-10-18.md)。以下内容保留为启动时的历史记录。

本轮落实用户授权的Dynamics Decoder A、A-only/B-only/A+B可比实验，并增加当前运动MLP、Persistence及线性趋势对照。正式训练已在服务器后台启动；本文是启动记录，尚无完整三seed结果，不表示已完成联合训练或自主Reset。

## 训练范围

冻结HARrad RD-CNN及Qwen前四层（包括原LoRA）；使用已封存CNN cells缓存和既有语义预热权重。可训练共享RadarTokenAdapter，每2帧生成4×256 tokens。A取tokens均值，经GRU256形成Dynamics状态，再结合当前固定CNN运动向量输出三个Horizon的残差；B保持EventFormer结构，训练Cross-Attention、Gate、投影、初始tokens与12类读出。

| 训练臂 | 可训练部分 | 目标 |
| --- | --- | --- |
| M | 当前运动64→256→192 MLP | 未来运动 |
| A | TokenAdapter、Dynamics GRU及残差头 | 未来运动 |
| B | TokenAdapter、EventFormer新增模块及分类读出 | 当前动作类别 |
| AB | 同一个TokenAdapter及A/B两支 | 两项损失之和 |

运动目标是冻结CNN全局64维特征的双帧均值，使用train-only均值方差标准化，不随共享Adapter训练变化。h=1/3/5对应标称0.1333/0.4/0.6667秒。预测损失为三个Horizon的MSE除以各自train Persistence MSE后按0.1/0.3/0.6加权；语义为CE/log(12)。残差末层零初始化，零步预测严格等于Persistence。

这里联合更新的是CNN之后的共享TokenAdapter，**尚未解冻CNN端到端联合训练**；现有RD-CNN-GRU只复用CNN部分，原分类GRU不充当已训练的Dynamics A。

## 数据与公平性

subject1–6训练31录制190,238帧；subject7验证4录制25,244帧。本轮不读取test信号。每条完整录制按2帧Update；35录制共舍弃22个不足2帧尾包，保留94,955个训练、12,600个验证共同合法起点。

预测起点不因未来跨边界或未标注而删除。当前双帧同属一个已知实例才有语义监督；同类但跨实例也屏蔽CE。真实未标注与完全已知的跨实例混合Update分开统计。

本轮采用官方起点的Oracle前缀初始化，最多32 Update，最后最多16 Update反传；每个优化步重新计算前缀。它不是自主连续Reset训练，也不是整段无限历史训练。官方起点只在已观察到时影响输入，不使用未来类别或边界控制当前输入。

M/A/B/AB共用每seed的固定样本表，seeds42/43/44，各1,000次成功优化更新，共12,000步，batch2；AMP跳步不计成功预算。AdamW lr1e-4、wd0.01，clip1；保存0/300/1000，固定比较final1000。

验证预测、语义面板各512起点，去重共1,005个，各臂使用完全相同的面板；指标按录制先计算再宏平均。主预测h5，分别报告跨实例、已知内部、未来未知与未来混合。只有一个验证受试者，且本轮是固定面板评价，不能当作跨受试者统计结论。旧77.01%来自完整实例末端分类，不能直接与当前Update面板相减。

## 已通过的检查

- 本地9项测试及服务器PyTorch 2.4.1原生9项测试；新增混合/未知回归先失败后通过。
- 独立审查和修复复核：无正式启动阻碍。独立审计另核对源mask与各组支持数、MSE、Skill。
- 真实Qwen最大32 Update、batch2预检：两项损失均传到共享Adapter；冻结Qwen无梯度且digest与语义预热一致；填充扰动误差0、零步Persistence误差0。
- 实测预检峰值已分配显存1,168,506,880字节，约1.09 GiB；AB可训练7,813,585参数。
- 数据契约及三张1000×2样本表下载后本地核验通过；输入模型及缓存SHA均检查。

## 运行与证据

服务器根目录：`/home/xibeidaxue/newdisk2/WJ/HARrad`。

- 正式源码：`experiments/joint_prediction_20261017_v2`，封存180文件。
- 正式输出：`outputs/joint_prediction_20261017_v2`。
- 后台驱动PID：4761；任务顺序为每seed的M、A、B、AB，再到下一个seed。
- 启动脚本：`joint_prediction_bootstrap_v2.py`；原生测试日志与driver.log在源码目录。
- 每任务保存`status.json`、`parameters.json`、0/300/1000检查点、`validation-final.npz`、`independent-audit.json`及`complete.json`；结束时重载最终权重，再次全固定面板评价并要求逐值误差0。
- 总状态`status.json`、`completed-jobs.json`；异常保留`failed.json`。仅有任务/总`complete.json`且审计通过才算完成。

本地证据：`outputs/harrad-joint-prediction-20261017-v2`，包含源码包、manifest、真实预检、数据契约与固定索引。源码ZIP SHA256：`81275e896510184265c56ccaee93401dd330d97ef7f6eb698be264fb70a6ea07`。

v1只完成预检、没有正式训练；其目录和证据保留。v2修复了把已知混合Update误计入unknown_future的统计问题，训练目标、固定采样表与预算不变。

## 首批实际进度

启动后的只读快照已确认PID4761仍运行，无`failed.json`。M-seed42完成1,000次成功更新、1,000次尝试，最终预测重放误差0；10个产物下载SHA核验及本地逐录制指标独立复算通过。保存快照时A-seed42为820/1,000步，随后交付前检查已达到1,000次成功更新、1,000次尝试，进入validation；尚未完成全部12组。

M-seed42固定预测面板的h1/3/5宏Skill为16.48%/26.99%/28.92%。这是首组单seed结果，只确认该目标可学习，不能据此判断联合分支的收益。快照及本地验证位于`native-v2/launch-status-snapshot.json`和`first-completed/local-verification.json`；本段是快照，不代表后台后续实时步数。

完成后需要比较AB相对A的预测、AB相对B的语义及三seed一致性，并独立复算下载预测。若没有稳定互补收益，应如实记录；本轮不自动进入Keep/Birth/Reset、Commit或语言训练。

# 受试者均衡50步开发实验

用户已同意本对话提出的受试者均衡方案，并要求继续执行。该实验只验证fit权重是否改善局部分组退步；不假定已证明过拟合机制。方案在新真实数据实验前锁定。

## 唯一改动与对照

每折从原Stage2 P初始化，Event Former和teacher冻结。natural/probe_eligible、age6/7、其余两个TRAIN subjects的set000/001为fit，set002为dev；本折held_subject和set003、validation/test数值不读。原all-fit teacher特征标准化不变。

对照age-class沿用四个年龄/正负类别桶各质量1/4。新subject-age-class把每个桶在两个fit subjects间等分，八个subject/age/class桶各质量1/8，桶内每endpoint等权。每subject总质量1/2，年龄/类别边际与对照一致。不做Candidate再归一化，不引入dev权重，不重采样。不允许缺组时重分配；任一必需subject/age/class桶为空则停止该折并保留证据。

新权重同时用于warmup和每步临时头的fit，以及P的transition BCE；否则头与P优化不同任务，无法解释对照。检索relation、consistency、anchor及其样本平均方式完全不变。头ridge.001、FP64、LBFGS400/原收敛规则；P AdamW lr1e-5、decay0、clip1；relation1与relation10、consistency.1、anchor1均不变。只训练P，原生产模型和历史输出不覆盖。

held02/03/04各运行两种权重，各自冻结对照、relation1、relation10，共12条50步P轨迹（600实际更新）。seed1701是确定性记录，不算随机重复；三个特征seed沿用3407/7211/12893。固定steps50、milestones5/10/20/50、显式budget-limit100（仅API上限，实际仍50）；不加步，不临时调系数，不解冻主干。

先用本轮原权重对照的20/50步P逐张量、独立dev预测、原语义指标精确核对旧100步轨迹的同预算截点。代码/spec签名、临时头记录之外允许manifest身份不同；数值不一致立即停止新权重实验，不能当作权重收益。新权重baseline的独立dev probe和语义数值应与原P相同；其临时头因新fit权重而不同，这是预期行为。

## 判断与报告

固定50步为主要比较，20步仅辅助轨迹诊断；不以selected step或事后挑一个系数替代固定预算结果。两个系数分别报告每折AUROC/AP、相对原P与相对同预算原权重的差值，并完整报告所有subject/age组、样本与正例数、临时头目标及语义指标。折命名subject_X表示排除该subject；不把其他subjects的dev成绩写成该subject的held测试。

预先声明探索性成功标准（每个系数分别判断）：三折macro AUROC相对原权重50步至少增加0.005，至少两折正且余折非负，macro AP不下降；12个折内subject/age组中，最差相对原P的AUROC差值改善，且至少8组的AUROC相对原权重不下降；三折所有50步原检索/动作语义门槛完整通过。另逐项报告检索下降是否超过原权重，不能把“通过容差”表述为无损。未全部满足记录有限/不稳定收益，不自动更改标准或筛掉失败。

此处0.5个百分点是权重假设的探索门槛，不代替正式工程主AUROC+0.02、AP+0.01和calibration工作点标准。三折fit/dev重叠、上游已见过TRAIN subjects；本实验不是独立未见受试者测试，不能宣布正式Go。若权重均衡无效，再单独设计检索难例保护或其他目标，不与本次改动混合。

## 完整性与执行

默认age-class配置及签名保持兼容；新weighting纳入manifest和checkpoint身份，禁止跨权重恢复。先通过真实合成优化、分桶/漏读/缺组测试与可用完整回归。服务器隔离目录projection-subject-balance-code-v1，输出transition-aware-semantic-primary-subject-balance-v1/{age-class,subject-age-class}/subject_0X-seed-1701；日志同名.logs。只读复用原三折已核验semantic caches。

启动前核对原生产runner/pilot与Stage2/source SHA、GPU、无重复worker、全部新输出不存在。nohup只启动一次，记录PID/时间。断线先查PID/manifest/status，不重复worker。每折结束核验实际AdamW50步、每次3头refit收敛、0/5/10/20/50权重与选中SHA、teacher/控制P冻结、禁读标志；原始JSON/log归档核验服务器/本地SHA后重算指标。失败或缺失不填零，不提高solver预算补过关。

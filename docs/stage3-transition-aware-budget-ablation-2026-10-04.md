# 固定训练预算诊断方案，2026-10-04

用户已授权执行：补齐第三折与原始结果，随后比较20、50、100步，判断训练预算是否限制收益。当前两折结果已知；此文件在新增实验之前锁定配置，不更改此前方案或结果。

## 数据、训练与顺序

1. 下载held03原始JSON/log并核验服务器归档SHA，补齐逐项检索差值。
2. 使用原已验证的20步算法（新隔离代码的默认预算），在新目录完成held04/seed1701；原Stage2起点、fit/dev角色、缓存策略、relation1/10、0/5/10/20检查点均不变。
3. 使用新隔离代码与新输出目录，对held02/03/04各运行一次最多100步的确定性轨迹。原Stage2初始化，不从旧适配权重继续；每条件固定评估0/5/10/20/50/100。20/50/100是同一训练轨迹的预算截点，不计作三个随机试验。不根据dev临时追加、提早终止或更改超参数；原拟合失败停止规则仍生效。

所有折seed固定1701，仅为记录；此确定性流程更换seed不会提供独立随机重复。每折排除自己的held_subject、set003、validation/test数值。两个其他TRAIN subjects的set000/001用于fit、set002用于dev。各折fit/dev有重叠，Stage2见过TRAIN subjects，本实验不是未见受试者测试。

保持primary natural eligible age6/7、四个age/class桶各质量1/4、原all-fit teacher标准化、三个固定feature seeds、FP64临时共享线性头ridge.001/LBFGS400及原收敛规则。每步先更新P、再fit-only重新拟合头。P AdamW lr1e-5、decay0、clip1；原teacher relation、consistency.1、anchor1不变。固定比较relation1/10，报告全部条件，不按新结果选一个系数作唯一结论。原上游、teacher冻结；语义缓存cap4、14类/42模板、角色及依赖SHA核验保持不变。

新程序默认仍限20步。100步必须显式传budget_limit=100，同时设置steps=100、milestones=(5,10,20,50,100)。拒绝非法步数/检查点，并在resume读取数值和重写manifest之前检查完整配置和SHA；禁止把旧20步目录当100步恢复。新程序的前20步在合成真实优化和服务器真实数据上与20步对照逐张量核对，出现差异就调查，不将其归为预算效果。

## 评价和预先声明的判断

主要比较每折、每relation条件的固定步20/50/100独立dev probe AUROC，以及对应原P的paired delta。原聚合不变：age6/7、两个dev subjects等权，三个固定feature seeds取median。另报各年龄/受试者AUROC和AP、样本及正例数、临时头fit objective与dev诊断、每次头拟合状态、完整语义指标及相对本折baseline差值。

检查点选择仍使用原语义门槛和AUROC排序、同分更早；固定预算比较不使用selected score替代各固定截点。不合格或指标未定义的截点仍完整报告，不能算合格收益。缺失数据不填零、不从held补齐。先核验20步复现，再分别对两个系数计算三折等权macro增益；两系数不择优合并。

探索性的“增加预算值得继续”标准：某固定系数100步相对20步三折macro AUROC额外增加至少0.005，至少两折为正、第三折非负，且三折100步均通过完整原语义门槛。另报告50步是否已经达到相同标准。不满足则记为有限/不稳定收益或评估不完整，不自动加步、调学习率、调约束或解冻Former。该0.5个百分点标准只用于判断预算效应，不代替既定正式工程标准（主AUROC +0.02、AP +0.01及calibration recall/FPR等）；此次不读calibration/held，不能宣布正式Go。

若fit目标持续下降而dev无一致改善，优先调查跨受试者泛化；若二者都停滞，调查表达能力和fit-only梯度关系。上述现象仅用于定位假设，不能单凭loss量级判断保护梯度过强。增加转变分支/adapter或修改共享表示均须另定方案，不在本次预算诊断里自动执行。

## 完整性与产物

held04输出：`transition-aware-semantic-primary-third-fold-v1/subject_04-seed-1701`。

预算输出：`transition-aware-semantic-primary-budget-v1/subject_0X-seed-1701`，X为2/3/4。缓存优先只读复用相同角色/依赖已验证的原缓存；held04新缓存先完成校验，再供预算运行复用。

每折冻结对照、relation1、relation10从相同warmup分叉，保存实际AdamW步数、pending阶段、所有头refit报告、checkpoint SHA与选中锁。检查teacher/控制P不变、源manifest和Stage2 SHA不变。detach后台启动前确认没有重复worker，保存PID/启动时间，断网只恢复监控，不能重复启动。

原生产代码/模型与历史输出不覆盖。服务器新隔离代码经测试和一次只读审查后才训练。网络不可用时完成本地方案、代码、测试及部署包，明确记录“未启动”，不伪报后台训练。

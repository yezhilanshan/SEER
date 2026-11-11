# v2.6 冻结前端预测诊断

本入口实现 P0/P1A，运行名为 `command_boundary_proxy_prediction_diagnostic`。使用命令日志代理分段、原 causal MD-GRU 与 Qwen2.5-0.5B EventFormer；没有真实动作实例GT，不训练Reset，所有报告 `formal_go=false`。

首轮已执行完毕，结果与核验见[2026-10-15报告](v26-prediction-pilot-results-2026-10-15.md)。现有YAML指向已经完成的v4输出；下面命令仅说明阶段次序，重复执行会因不可覆盖的产物保护而拒绝。新实验须先固定新协议与新输出目录，不能复用本轮目录。

配置：[parrad-v26-prediction-pilot-v1.yaml](../configs/parrad-v26-prediction-pilot-v1.yaml)。先锁定来源，再按序执行：

```bash
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m continuous_mmwave.v26 --config configs/parrad-v26-prediction-pilot-v1.yaml --mode audit
python -m continuous_mmwave.v26 --config configs/parrad-v26-prediction-pilot-v1.yaml --mode export
python -m continuous_mmwave.v26 --config configs/parrad-v26-prediction-pilot-v1.yaml --mode smoke
python -m continuous_mmwave.v26 --config configs/parrad-v26-prediction-pilot-v1.yaml --mode train
python -m continuous_mmwave.v26 --config configs/parrad-v26-prediction-pilot-v1.yaml --mode evaluate
python -m continuous_mmwave.v26 --config configs/parrad-v26-prediction-pilot-v1.yaml --mode verify
```

必须使用已有兼容环境；无需安装依赖。服务器环境为 `/home/xibeidaxue/newdisk2/WJ/.seer-conda/envs/seer/bin/python`。本机测试用interview Python，pytest临时目录设在workspace。

新输出目录必须不存在，阶段目录不允许覆盖。audit锁定代码/config、嵌套前端配置、Qwen基础权重/tokenizer/config、checkpoint和split；之后变更会拒绝继续。未来缓存/index/归一化与最终头权重分别核验SHA。

## 预测臂与数据

- Persistence：复制当前目标latent，方差按fit recording等权估计。
- Event-free MLP：当前latent，无事件状态。
- GRU/TCN：当前latent和同一代理episode的完整4-token因果前缀。
- Event-conditioned：当前latent和冻结8×896事件状态。

主要样本为每个命令episode前32个合法update，h=1（0.18秒）。TCN感受野恰好32；各神经臂约40万新增可训练参数，冻结EventFormer的额外预训练容量单独报告，不声称总容量相同。

pilot每个训练/验证受试者选择按session_id排序的首个双传感器完整session；subject09的set000只有一个传感器，因此选择set001，选择只依赖元数据。测试雷达数组不访问。

缓存记录不完整update尾帧数量；unknown/mixed/gap不成为有效未来对。maturity仅使用已到达update计数。目标为冻结Radar Tokens的masked mean，归一化统计只从fit导出，按recording等权。

## 输出与验收

`protocol.json`保存来源和结构；`cache-train/`、`cache-validation/`保存冻结F/E、当前/未来索引；`smoke/`与`train/`分别保存一步烟测和固定300成功更新；`evaluation/`保存各臂逐录制预测及受试者宏指标；`verification.json`保存独立NumPy指标重算和权重精确回放检查。

每个神经臂使用同seed/同采样清单，末步权重用于唯一一次validation评价，不按validation选择架构或追加步数。最终评价需要完整五臂、原checkpoint、原recording顺序及完全一致的future索引。训练结果好也不能自动启动Stage2/3。

当前实现的边界：只检查初次encoder update重复一致、最初32个update中的事件前缀重复一致；未验证全录制逐步encoder回放。缓存通过prefix_start保留实际前缀范围，未独立保存command_episode_id。报告当前每维NLL/MSE/variance和batch1 EventFormer CUDA Graph回放P95（不包含输入复制与全在线链）；边界内/跨边界分组、联合NLL、完整在线P50/P95及峰值显存尚未作为本pilot完整验收。

实际启用的加速是固定batch1 CUDA Graph，保持原事件递推与精确顺序回放。其静态输入、侧流预热与复制后replay遵循[PyTorch CUDA Graph说明](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/cuda.rst)，并在服务器Torch2.4.1上实际探测。batch16独立episode版本虽通过CPU因果测试，但native FP16参考误差超过预注册门槛，因此没有用于训练，没有放宽容差；失败目录保留。

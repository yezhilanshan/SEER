# HARrad Stage 1 Event State 未来运动预测实验

依据用户本轮给定的 Stage 1 结构执行，复用冻结旧 RD 前端与 `shared-front-event-warmup-v1/training/best.pt`。

目标为检验 EventFormer 状态能否提供超出当前观测和普通历史模型的未来运动信息。

- [x] 构建严格因果冻结缓存。以 GT 起点初始化 E、年龄和双帧缓冲。保存每个实际双帧更新的 F、E、观察 cutoff 及同事件历史起点。未知区间保持真实流，未来目标有效性只由录制长度决定。
- [x] 固定 256 维目标。冻结 RadarTokenAdapter 输出四个 256 维 tokens，固定平均作为参考运动 latent。输入侧单独训练 1024→256 Motion Projection，初始化为同样的平均。目标不跟随该投影更新，标准化只用 fit 数据。
- [x] 精确未来为原帧 cutoff+3/+9/+15，即标称 0.2/0.6/1.0 秒。未来目标使用对应 cutoff 之前的最后两帧生成冻结雷达 tokens，跨录制拒绝。保留跨事件未来目标。
- [x] A 复制当前参考 latent，预测方差只用 fit 残差拟合。B 包含 GRU 和 TCN，C 为 Event-free，D 为 Event-conditioned。复用 v26 预测头及完整历史实现，分别训练输入 Motion Projection 和预测头，均不更新目标、Encoder、EventFormer、Semantic Reader。
- [x] 普通历史模型从同一 GT 初始化位置读取全部可见更新。TCN 感受野覆盖最大历史。记录各臂参数量，目标预测头容量约 400k，输入投影容量一致。三 seed、各 1000 成功更新、batch16，三 horizon 等权，固定最终检查点评价。
- [x] subject1–6 每人最后一条录制为预测头校准集，其他 25 条 fit。subject7 为 Stage 1 验证，subject8–9 不新增预测评价。冻结来源此前见过全部训练受试者，因此该校准仅对新预测模块成立。
- [x] 指标包括每录制等权 MSE、对角 Gaussian 每维 NLL、相对 Persistence 的 Skill、50/80/90/95% 边际区间覆盖率、方差和标准化残差。分稳态、跨官方起止、同类重复起点、未知观测，允许分组重叠。
- [x] 缓存和 checkpoint 锁定来源 SHA，预测保存均值/方差/目标和真实 cutoff，重载回放与独立复算。
- [x] Go 条件严格检查 D 在三种 horizon 上相对 GRU、TCN 和 C 的三 seed 平均 MSE、NLL 均改善，并公开逐 seed 差值和校准。不满足或证据不完整时保持 No-Go，不启动 Reset Policy。

实现文件为 `src/continuous_mmwave/harrad/event_prediction.py` 和必要的回归测试。服务器使用独立 Stage 1 输出，既有实验作为来源锁。

执行记录。相关本地测试 20 项通过。独立审查核对了固定目标、完整历史、GT 初始化和参数容量，并补齐新模型实例重载后的预测回放、逐 seed 差值与覆盖率汇总。服务器四个可训练模型的 GPU 前向/反向预检通过，测试最大历史 512 时参数量约 144.5–146.1 万。后台 PID383516，来源胶囊为想法三根目录的 `stage1-prediction-launch-v1`，输出为 `stage1-event-prediction-v1`。当前正在生成冻结缓存，已完成前两条录制。优化预算、目标聚合和严格多 horizon 验收为本轮实施约定，后续结果将附完整来源说明。


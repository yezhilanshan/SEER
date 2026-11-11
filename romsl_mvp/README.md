# ROMSL C 与 D 组预备实现

C/D 正式标签入口为 romsl.targets.build_cd_labels。外参和 Doppler 模板当前保持 verified=false，等待作者资料及复核。

## 模块

geometry.load_extrinsics 读取 JSON 或 YAML 中的 R/t 或 T_world_to_radar。外参必须明确 world→radar、世界坐标为米、平移单位、雷达世界原点、来源、核验人和独立验证误差记录。平移支持 m/cm/mm，加载后统一为米。反射矩阵、非正交旋转、来源缺失、原点不一致及未核验状态均返回 BLOCKED_GEOMETRY。

transform_points 对位置应用 R 和 t，transform_velocities 对速度只应用 R。estimate_los 支持任意 [...,3] 关节位置，雷达原点和非有限数据标为无效。project_radial 输出带符号投影。远离雷达为正，接近雷达为负。

正式 C 目标遵循方案书，根速度为带符号径向速度，四肢为相对根速度在根 LOS 上的绝对投影均值。逐关节 LOS 接口可供诊断，正式目标采用共享根 LOS。

doppler.load_doppler 核验零基列号 3/4、单位、符号和证据。支持 m/s、cm/s、mm/s，统一转换为远离雷达为正的 m/s。未知单位或符号会阻塞。verify_sign 使用成对的已知运动观测核验符号，要求同时覆盖接近和远离运动。该接口的参考值应来自可验证的已知运动，人体骨架代理并非反射点 Doppler 真值。

echo_support 使用未归一化原始点云，按照方案固定的 0.1 m/s 阈值计算动态点比例，再计算 0.25+0.75*clip(r/0.25,0,1)。空帧权重为 0.25，mask=false。非有限点不计入分母。

weight_statistics 输出均值、标准差、分位数、接近 1 的比例和有效帧数。诊断默认将标准差小于 0.001 或至少 95% 有效帧权重达到 0.99 标记为 LOW_DIAGNOSTIC_VARIATION。它们是预先声明的退化检测标准。训练权重公式保持固定。CLI 仅在语义闸门通过且实测权重状态为 VARIABLE 时将 D_ready 设为 true。

## 使用

本项目配置文件使用 JSON 语法，属于合法 YAML。读取普通 YAML 需要 PyYAML，绘图需要 matplotlib，合成测试还需要 pytest。数值计算依赖 numpy，既有模型测试依赖 torch。

在项目父目录执行以下命令。

```powershell
python -m pytest -q romsl_mvp/tests
python romsl_mvp/scripts/08_cd_preflight.py --output reports/cd_synthetic --synthetic
python romsl_mvp/scripts/08_cd_preflight.py --output reports/cd_readiness
```

合成模式生成可变权重和全饱和权重两套统计与 PNG，报告标记 synthetic。它保留真实配置的阻塞状态。

作者资料到手后，先填写 configs/extrinsics_e01.yaml 与 configs/doppler_e01.yaml，补齐独立核验证据，再由复核人设置 verified=true。以下命令只统计训练集权重并画图。

```bash
python romsl_mvp/scripts/08_cd_preflight.py --data-root /path/to/MMFi --output /path/to/new_diagnostics --split train
```

需要正式导出时显式增加 --emit-labels，并使用 --exp C 或 --exp D。入口检查所有前置条件，失败返回退出码 2，报告 labels_emitted=false。标签先写入临时目录，全批完成后才发布 labels 目录。默认不读取测试受试者。

已知外参的 4×4 输入使用 T_world_to_radar 字段，底行为 [0,0,0,1]。它与 R/t 两种表示互斥。平移单位由 translation_unit 明确声明。所有坐标系定义和原点信息应同时保留在 source 指向的校准资料中。

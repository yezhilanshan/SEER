# Radar-to-Text first real-data server pilot

用户明确要求连接指定服务器，使用账号下现有 PARrad 和 Qwen 权重。密码仅用于交互认证，不落盘。

1. 已通过 SSH 直连；核实 canonical RD/MD、标签、受试者元数据、GPU 和已有环境。
2. 使用项目已有 `/home/xibeidaxue/SEER/.conda/envs/seer/bin/python`；不修改共享环境。初查 Torch 2.4.1+cu121、Transformers 5.4.0、PEFT 0.18.1、h5py 3.10.0。RTX2080Ti compute capability 7.5，使用 FP32，不以可能允许软件仿真的 bf16 API 返回值作为硬件原生支持证明。
3. 打包独立代码到新 `.audit-staging/radar-text-code-20261010-v1`；SHA 校验源码及配置，不覆盖历史实验。输出新 `PARrad/outputs/radar_text_pilot_20261010_v1`。
4. 使用已锁定 canonical train subjects02/03/04（24视图），validation05（8视图），test06（6视图）。12秒窗口、6秒步长、最短8秒、最多3动作；只从真实class ID构造文本。
5. 先运行原生测试及真实 Qwen+雷达窗口的 forward/backward/generate，核对雷达侧梯度和冻结 LLM；验证可选 LoRA checkpoint 测试。
6. 初始完整子集 pilot 固定为 stage1 3epochs、align 3epochs。stage1 encoder/classifier lr1e-3，batch4；align encoder lr1e-4、projector lr1e-3，batch2、accumulation4；seed17。按validation loss选best；不根据test调参。本轮先验证冻结LLM对齐，不默认追加LoRA训练。
7. 测试通过后启动单个持久 worker，断线不重复启动；先 stage1，再加载 stage1 best 到真实预训练 Qwen align。记录实际批次/损失、PID和产物。
8. 完成后分别报告validation/test和分类模板基线，对模板代理指标与真实描述能力保持区分；取回报告、日志、配置及新雷达侧权重。初始pilot不宣称收敛。

## 预检记录

- SSH直连成功，账号已有环境完整，不安装依赖。真实 canonical RD `[6687,1,93,128]`、MD `[6687,128]`，对齐时间步0.09秒。
- v1 capsule archive SHA49b4ecd5805b879969d1cb61e47d5be7990ca021666c2e472c42a7bec91eb2c3。原生43测试通过、1失败：打包漏带旧测试引用的 `scripts.make_synthetic_parrad_data`。模型/LoRA测试已通过，无训练启动，保留v1。
- v2补齐测试辅助脚本，archive SHAd2a622e8f931094973a735584675fc3fe85bb70c15f297cc5a1d173f7e26fbfd；117源码/配置SHA全部核对，独占解包到`.audit-staging/radar-text-code-20261010-v2`。
- v2原生 **44 passed**（含LoRA和HDF5转换），3.82秒。真实Qwen预检通过：hidden896、24层、LLM可训练参数0；初始caption loss2.60767984，RD/MD/BiGRU/projector梯度norm分别7.3884/4.5157/6.3547/18.8096；峰值GPU约2.61GB；优化器更新0。
- 训练窗口2187（单动作20/双动作582/三动作1585），验证702（单动作4/双动作223/三动作475）。12秒设置下单动作样本很少，本pilot不能给出稳定的单动作泛化结论；不在看test后改窗口。
- 实际Qwen权重SHA fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe（2026-10-10结果核验时修正进度文字转写多出的f；机器生成JSON及权重未改变）。未对齐真实模型的示例输出要求更多radar上下文，说明语言流畅不代表能读取radar tokens。
- 单个持久launcher PID17234于22:30:18+08:00启动，实际 `--output` 使用绝对路径；外层 `Popen(start_new_session=True, stdin=DEVNULL)`，不是依赖交互shell继续存活。
- 独立审查提示通用脚本的相对路径问题和未来预检角色锁定不足。本次实际使用绝对路径，已独立核对三个split的精确身份及SHA，无运行影响。本地helper后续加`.resolve()`、精确split断言、归档独占创建；不修改正在运行的v2 capsule，模型/数据/epoch参数不变。
- 在test预测产生前，预先固定附加的validation-only grounding诊断：均匀取32个验证窗口，以“下一个动作序列不同的窗口”错配雷达，保持目标文本相同，对比teacher-forcing loss；前8对另生成文本。冻结模型、0optimizer更新，不据此追加训练或改本轮参数。生成变化本身不能证明变化方向正确。

## 完成核验

2026-10-10 22:53:06+08:00两个3epoch阶段及validation/test评估完成，stage1 best epoch2、align best epoch3。test121/484完整序列正确，baseline54/484；动作recall46.28%低于baseline55.51%。validation-only错配诊断32对中26对loss改善，前8对生成都改变，但各自radar的参考仅4/8完整正确。26项取回产物及117项训练源码SHA核验通过，原始Qwen/manifest/label-map SHA与预检一致。原始预测复算一致；未追加训练。详见[结果报告](../../radar-text-server-pilot-2026-10-10.md)。

# X-Fi mmWave ResNet18 在 PARrad RD 上的迁移检查

此文记录接入前检查。后续已完成三组真实PARrad冻结骨干实验，结果与替换建议见[RD-only迁移实验报告](rd-only-xfi-transfer-2026-10-11.md)；下文“尚未训练”指本次检查当时的状态。

结论：可以作为 RD 空间/局部片段骨干的预训练初始化候选，不能原样替换当前 RD+MD CNN+BiGRU 完整编码器。需要输入适配、保留外部时序建模，并重新验证/训练雷达到语言接口。当前仅检查权重、官方代码与合成输入，没有训练或测量 PARrad 识别收益，原权重未改。

## 本地权重实测

文件：`D:/lab/SEER/backbone_models/mmwave_ResNet18.pt`，45,087,228 bytes。

SHA256：`03ae1027da2baf2210bfda363497de2913c9ff044423f2158e03a6b4041a6d56`。

- 保存的是完整自定义模型对象，类型路径`backbone_models.mmWave.ResNet.ResNet`，并非裸state_dict。
- 读取使用`weights_only=True`、本地受控空壳类与明确的PyTorch层允许列表；没有执行checkpoint内的未知Python类或切换到任意pickle加载。
- 参数11,248,631；残差块[2,2,2,2]，通道64/128/256/512。
- 首层`Conv2d(17,64,kernel_size=7,stride=2,padding=3,bias=False)`，权重[64,17,7,7]。
- `AdaptiveAvgPool2d(1)`后512维，分类头`Linear(512,55)`；没有BiGRU或雷达到语言Projector。
- 顶层还保存了`softmax`模块，但官方ResNet.forward没有调用它。官方X_Fi的`children()[:-2]`却会包含该模块。迁移应明确选择conv1→bn1→relu→maxpool→layer1..4，避免隐式引入不需要的Softmax。

本地结构与X-Fi XRF55_HAR毫米波预训练分支一致；单个文件没有提供足以核验训练来源/划分/epoch的元数据，因此不声称已核实官方发布文件哈希或原训练精度。

## 官方输入与当前输入的区别

官方[XRF55_Dataset.py](https://github.com/NTUMARS/X-Fi/blob/main/XRF55_HAR/XRF55_Dataset.py)将毫米波数据reshape为[17,256,128]；[X-Fi论文](https://arxiv.org/html/2410.10167v3)实验设置说明XRF55毫米波表示包括Range-Doppler和Range-Angle。原网络与特征抽取见[ResNet.py](https://github.com/NTUMARS/X-Fi/blob/main/XRF55_HAR/backbone_models/mmWave/ResNet.py)、[X_Fi.py](https://github.com/NTUMARS/X-Fi/blob/main/XRF55_HAR/X_Fi.py)。这是模态专用预训练骨干，不是完整多模态X-Fi融合模型。

当前PARrad配置及离线代码使用逐帧RD[B,T,1,93,128]，RD与MD各提取128维，concat为256维后送BiGRU。不能将17通道直接等同于17种传感器、17类动作或17个人体关节。本次读取的数据加载器只显示reshape，没有证明17通道的物理/时间组织，也未确定RD与RA具体拼接顺序；这些不能靠checkpoint形状推断。原始预处理的帧率、轴定义、幅值/对数尺度与归一化仍需核对。

原始网络的卷积及自适应池化可以运行不同空间大小，不必仅为shape一致将PARrad RD强行拉伸成256×128。能运行不代表范围/速度物理坐标或输入分布已对齐，缩放也不能凭空产生RA信息。

## 已完成的形状/数值检查

仅在内存、eval/no_grad、随机合成RD上运行，没有读取真实雷达训练样本：

- 单帧RD[2,1,93,128]复制为17通道，经明确的conv1..layer4得到[2,512,3,4]；空间均值池化后[2,512]。
- 将首层改为1通道，并用原权重沿输入通道**求和**初始化，输出同为[2,512,3,4]，与上述复制输入的输出最大绝对差9.536743e-6，在allclose容差内。
- 求和初始化等价于将同一RD复制17次的卷积；求均值初始化则额外缩小首层响应17倍，不能称为等价迁移。两种都不等价于恢复原17通道语义。
- 单通道、去55类头的骨干参数11,170,240。上述检查只证明张量和数学适配可行，不能证明特征可分性或文本质量。

## 建议的接入方式

```text
RD[B,T,1,93,128]
→ 单通道适配的预训练ResNet18（去55类头，明确选择卷积主干）
→ 每帧512维 → Linear(512,256)
→ BiGRU（每方向128维）
→ 8个有序Tokens → 现有规格Projector → 冻结Qwen
```

这是待验证的RD-only架构建议，没有在本轮修改训练入口。可以先冻结ResNet（BatchNorm也保持eval），用训练受试者拟合小读出器检查可分性；如有收益，再微调输入适配/后段残差层和BiGRU，并重新训练雷达到语言对齐。输出宽度仍为256并不意味着新特征与旧Align Encoder语义相同；旧BiGRU/Projector最多作为待验证的warm start，不能直接视为兼容权重。

另一种适配是将局部17帧RD堆为17通道，以保留原首层形状，再滑动提取片段特征送BiGRU。但在原通道语义/采样间隔未核清前，它只是新设计的适配方案，不能声称复现原预训练输入；切片只能使用当前输入窗口，不能跨split或读入窗口外数据。

## 最小验证方案

保持相同受试者划分、12秒滑动窗口、训练预算和validation选优规则，先比较：

1. RD-only原小CNN+BiGRU。
2. RD-only随机初始化ResNet18+同规格BiGRU。
3. RD-only该预训练ResNet18+同规格BiGRU。

第2/3项隔离预训练收益；第1/2项帮助判断容量变化；原RD+MD保留为额外参考，不能把去掉MD与更换骨干的影响混为一谈。先看14类Macro-F1/床相关召回、有序读出与Count MAE；确认有效后才做冻结Qwen对齐。使用P1的较强分类后处理作对照，不退回未平滑分类模板。

P1已经表明数量控制与读出流程是重要瓶颈，换更大ResNet不是已被验证的解决方案；本次不对性能改善或细粒度语言能力作保证。

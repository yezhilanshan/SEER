# 服务器数据集盘点，2026-10-06

核查时间：服务器北京时间2026-10-06 17:00–17:10，最后确认17:10:50+08:00。按用户要求只读搜索；没有下载新数据、修复下载、转换新录制、训练、安装依赖或更改服务器文件。

## 主要结论

1. PARrad原始目录有28个正式H5（约284.18 GiB），另有一个2.61 GiB的未完成`.part`；加上CSV和sidecar，整个raw目录约286.789 GiB。
2. 相比现有canonical清单，多出subject08/12各四组录制，subject13一组正式H5。它们目前不在canonical train/validation/test中，可考虑作为新增受试者评估候选，先不要混入训练。
3. 全部28个正式H5的文件头均能打开，数据字段都是两个radar sensor的cube/mDoppler/rdn，没有骨架、姿态、RGB或深度数据字段。未重算这些大文件的SHA，也未完整扫描全部数据块，不能据文件头可读宣称每个数据块均无损。
4. 另有一个早期PARrad MD子集和数套pyRadar数据。MD子集与已有PARrad录制重叠；pyRadar有原始bin和样本级mat，需要另行确定动作体系/采样规则，不能直接拼入PARrad训练。
5. 本次搜索范围内没有发现可识别的同步骨架数据集。因此暂时没有证据支持直接开展方案书所列的人体高度/上下肢活动等完整物理监督。

## PARrad：原始文件与未完成下载

根目录：`/home/xibeidaxue/newdisk2/WJ/PARrad/raw`。录制路径规则为`subject_XX/captured_data/setXXX/processed_data.h5`，动作指令文件为同目录`timestamp_speech.csv`。

| subject | 正式H5录制 | 状态 | raw占用GiB（包括小文件和part） |
|---|---|---|---:|
| 02 | set000/001/002/003 | 4个正式H5 | 40.323 |
| 03 | set000/001/002/003 | 4个正式H5 | 40.317 |
| 04 | set000/001/002/003 | 4个正式H5 | 40.304 |
| 05 | set000/001/002/003 | 4个正式H5 | 40.323 |
| 06 | set000/001/002 | 3个正式H5 | 30.241 |
| 08 | set000/001/002/003 | 4个正式H5，未进现有canonical清单 | 41.267 |
| 12 | set000/001/002/003 | 4个正式H5，未进现有canonical清单 | 41.300 |
| 13 | set000 | 1个正式H5；set001只有part；set002空 | 12.714 |
| 15 | 无 | set000/001/002空目录 | 0 |
| 20 | 无 | set000/001/002空目录 | 0 |

“正式H5”仅指文件名已为`.h5`并且本次HDF5元数据可读，并非重新验证了下载端到端SHA。raw共115个文件：28 H5、29 CSV、57 SHA sidecar和1 part。

未完成文件：`/home/xibeidaxue/newdisk2/WJ/PARrad/raw/subject_13/captured_data/set001/processed_data.h5.part`，2,801,795,072 bytes；mtime为2026-09-24 09:01:45。subject13/set000正式H5为10,849,741,736 bytes。未将part作为可用数据，也未启动续传。

## PARrad：格式与标签

28个正式H5全部拥有相同六个Dataset路径：

```text
radar/TI_xWR14xx/cube
radar/TI_xWR14xx/mDoppler
radar/TI_xWR14xx/rdn
radar/TI_xWR68xx/cube
radar/TI_xWR68xx/mDoppler
radar/TI_xWR68xx/rdn
```

抽查字段dtype为float32。示例subject08/set000：cube `[7321,128,1,16,93]`、mDoppler `[7321,128]`、rdn `[7321,93,128]`，两个sensor均7321帧。subject12/set000两个sensor分别6705/6706帧，旧转换逻辑的对齐规则仍需在未来转换时沿用。

抽查subject02和subject08的CSV表头均为`timestamp,command`。这是动作口令时间标签；不能当作同步骨架或精确的人体动作起止真值。本轮没有重做动作标签转换、边界评分或查看官方test数组/预测。

## PARrad：当前canonical划分

目录：`/home/xibeidaxue/newdisk2/WJ/PARrad/canonical/phase1`。共有38个NPZ。

| manifest | subjects | 录制视图数 | sessions |
|---|---|---:|---|
| train.jsonl | 02/03/04 | 24 | 每subject set000–003 × 两sensor |
| validation.jsonl | 05 | 8 | set000–003 × 两sensor |
| test.jsonl | 06 | 6 | set000–002 × 两sensor |

本次只列出了这些manifest的分区元数据、文件名和文件SHA，没有读取validation/test数组或评估分数。subject08/12/13没有canonical NPZ，也没有进入这三份manifest。

当前train manifest SHA256（完整转录）：

```text
train       be1603bd5925dd3905c19b777e4e33aef600cf371a1086f2b41c7bb0c1b6246f
```

当前清单不含08/12/13是已确认事实；“全部历史模型从未使用过它们”还需要绑定对应checkpoint的训练来源清单，不能只凭目前的manifest定论。未来若保留新增受试者评估，应先锁定身份和用途，再做转换；不要先看新受试者的效果并据此决定它属于训练还是测试。

## 其他雷达数据

### 早期MD子集

`/home/xibeidaxue/SEER/data/parrad/hospital_md_77ghz_normal_v1`

- 共4个MD数组：subject04/05 × set000/001，TI_xWR14xx单sensor。
- 有selection_manifest.jsonl、README和四份timestamp_speech.csv；数组为float32 `[frames,128]`，整个目录约0.013 GiB。
- README和selection manifest明确来源是PARrad原H5的mDoppler字节区间提取。不是新增受试者数据，不能重复计为新的独立样本；无骨架、完整RD或完整H5。

### pyRadar系列

| 路径（均在`/home/xibeidaxue/newdisk2/WJ`下） | 文件数与类型 | 占用GiB |
|---|---|---:|
| pyRadar/data | 9个bin | .025 |
| pyRadar/data_10_classes | 210个mat | .028 |
| pyRadar/data_4_classes | 648 mat + 641 bin | 1.769 |
| pyRadar/data_5_classes | 76 mat + 77 bin | .437 |
| pyRadar-main/data | 622 mat + 646 bin | 3.510 |

样本名包含`click2`等手势动作。使用服务器既有anaconda环境只读检查三个Merge/wj.mat头部：

- 10类：allData `[200,32,60]`、allLabels `[1,200]`。
- 4类：allData `[80,32,30]`、allLabels `[1,80]`。
- 5类：allData `[75,32,60]`、allLabels `[1,75]`。

这些是文件/变量元数据，不是完成了类映射、传感器配置、采样时间或数据质量验证。pyRadar与pyRadar-main可能重复；bin和mat也可能是同一采集的原始与处理版本，不能按文件数当独立样本数。没有证据表明它们带有可直接使用的同步骨架监督。

## 其他目录与搜索边界

还看到THUCNews、MOSI、MNIST、CIFAR10、IRSTD-1K/NUDT-SIRST/SIRST3、ucf101等目录/压缩包；没有深入验证其内容或完整性，也没有纳入本次雷达训练建议。

搜索覆盖账号自己的Downloads/下载、SEER/data、newdisk2/WJ、newDisk/xibeidaxue、PycharmProjects、文档/桌面、updata、Matalab等位置。WJ和项目数据递归文件名检索深度最多7级，跳过环境、模型、输出、缓存等非数据目录；未递归检查其他研究人员的磁盘子树，也未把它们计入库存。名称检索MMFi/HuPR/MMBody等及skeleton/pose/joint关键词没有发现可识别的新姿态数据，命中的joint只是Stage2日志名。

这不是“整台服务器一定没有骨架数据”的证明；如果其他下载位于未覆盖目录，需用具体路径继续查。

## 对训练提案的影响

- 新增subject08/12是更有价值的未见受试者评估候选，应优先保留；建议两个seed和阈值先锁定，再一次性评价新增受试者，不先并入训练后仍称未见。
- subject13当前只有set000可作后续候选；其余part/空目录排除，下载完整性另行核验。
- 对Evidence的结论没有改变：目前发现的PARrad文件没有骨架；若没有另一份同步姿态数据，只能做明确标注的雷达统计代理实验，不能宣称完整人体物理监督。
- 现有训练提案仍等待用户批准。此次没有修改其锁定数据划分、启动训练或恢复任何下载。

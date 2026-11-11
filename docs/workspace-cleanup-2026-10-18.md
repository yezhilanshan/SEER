# 本地工作区清理记录（2026-10-18）

范围仅为`D:/lab/SEER`，没有操作服务器。本轮按用户授权清理重复交付包、已中断下载和可再生成测试文件；正式训练权重、检查点、逐点预测、原数据、源码、配置、方案和研究报告继续保留。

## 清理内容

| 类型 | 释放空间 | 依据 |
| --- | --- | --- |
| 27份重复下载压缩包 | 5.40 GiB | 每个普通文件成员均找到实际保留文件并核对SHA256；包内源码ZIP等也有对应副本 |
| 1份中断下载 | 1.04 GiB | `outputs/v261-joint-pilot-20261015-v4/evidence.tar.gz`；原结果报告已注明不是完整交付，本次读取也报截断错误；保留原报告、哈希和compact-evidence |
| 已通过测试的临时fixture及测试权重 | 1.06 GiB | 仅明确的pytest通过/最终测试目录，且文件属于`test_*`测试fixture；避开保留证据所依赖的路径 |
| Python及pytest缓存 | 约4.42 MiB | `__pycache__`和`.pytest_cache` |

共75个明确目标，删除8,060,429,941字节（约7.51 GiB）。清理前约16.21 GiB，清理后约8.71 GiB；实际扫描结果见下方验证文件。

删除前检查了所有目标的最终绝对路径均位于工作区内，并拒绝链接/重解析路径；压缩包删除前再次核对大小和SHA。没有整删`outputs`或`.audit-staging`，也未按“旧实验”一概删除。无法完整匹配的两份独有归档保留。

Split BiRNN结果报告原压缩包链接已改为保留的完整解压目录。其他报告中的历史归档哈希仍保留作追溯；如旧交付检查脚本专门要求已删除的压缩包本体，应使用本次成员映射核对保留文件，或重新获取原始交付包。解压内容相同不保证重新打包后的压缩字节哈希相同。

## 审计与复核

清理审计目录：`outputs/workspace-cleanup-20261018-v1/`。

- `archive-audit-final.json`：归档成员、SHA256和实际保留副本路径。
- `deletion-plan.json`、`validated-targets.json`：逐项删除依据及确认后的绝对路径。
- `cleanup-execution.json`：实际删除目标及字节数。
- `verification.json`：删除后重新核对保留成员文件、HARrad 120份正式任务文件、旧语义诊断关键产物及剩余空间。

可复用工具：`scripts/audit_workspace_cleanup.py`、`scripts/prepare_workspace_cleanup.py`、`scripts/execute_workspace_cleanup.ps1`、`scripts/verify_workspace_cleanup.py`。本次脚本和清单有固定版本防覆盖；不要对已完成的清理目录重复执行删除脚本。

下一步HARrad诊断需要的0/300/1000/final检查点和逐点预测保留。旧PARrad语义曲线、匹配分层、转换分析、报告以及独有失败记录也保留。

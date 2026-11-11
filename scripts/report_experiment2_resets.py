"""Plot the reset randomization distribution and append its interpretation."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def report(evidence,destination):
    rows=json.loads((evidence/'summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,metric in zip(axes,('mse','nll')):
        for i,r in enumerate(rows):
            a=r['metrics'][metric];rep=np.array(a['random_replicates'])
            ax.scatter(i+np.linspace(-.12,.12,10),rep,color='#0072B2',s=23,label='10 random arrangements' if i==0 else None)
            ax.scatter(i,a['gt'],color='#009E73',marker='*',s=130,label='GT reset' if i==0 else None)
            ax.scatter(i,a['no_reset'],color='#D55E00',marker='X',s=65,label='No reset' if i==0 else None)
        ax.set_xticks(range(3),['0.2 s','0.6 s','1.0 s']);ax.set_ylabel(metric.upper()+' (lower is better)')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8,frameon=False)
    fig.suptitle('Experiment 2: reset structure controls',fontsize=13)
    fig.text(.5,.015,'Each point averages 3 training seeds and 4 subject7 recordings; random arrangements are constrained samples.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(evidence/'reset-distribution.png',dpi=180);fig.savefig(evidence/'reset-distribution.pdf')
    original=(evidence/'report.md').read_text(encoding='utf-8')
    interpretation='''
## 结果解释

全部36个模型、36000次更新已完成；训练与独立复算退出码均为0，1202组复核通过。下载证据包已按服务器SHA256校验。

三个预测跨度的GT MSE/NLL均优于全部10份随机排列的三seed均分。GT相对随机集合和No Reset的配对录制区间均不跨0；逐条留出后的平均MSE差值也都保持GT更优。这不是只来自一份特别差的随机排列。10份排列只作描述性分布，不能将10/10解释成严格随机化显著性检验。

Random与No Reset的排序不统一。0.2秒Random MSE更好，配对区间不跨0；0.6和1.0秒MSE差异区间跨0。所有跨度的Random−No Reset NLL区间均跨0。不能宣称普遍存在GT < Random < No Reset的排序。

在更新间隔多重集相同、实际最小间隔相同、实际间隔分布近似匹配的约束下，GT仍更好，支持重置与真实事件对齐具有额外价值；这比单纯限制累计历史长度的解释更充分。但实际间隔并非精确相同（最大逐项2帧），所以没有完全排除时间分布的剩余影响。该设计同时改变状态历史和状态年龄，冻结EventFormer也曾按GT分段预热，不能把收益全部归因于语义事件推理。

本轮GT与其他组在同一12组批次重建状态并匹配重训，以本轮GT为比较基准，不与上一轮GT分数混用。相对于既有GRU参考（MSE 0.75589/0.92940/0.94881），本轮GT MSE 0.76553/0.92015/0.95065仍未在0.2秒和1.0秒超过GRU；原Stage1 No-Go保持，没有启动Reset Policy。

下一步宜先做跨受试者复核，并考虑控制年龄输入、加入固定周期且长度匹配的重置对照，分离事件对齐和年龄/历史长度的作用。当前结果支持继续研究事件对齐的价值，但不足以进入自主Reset训练验收。
'''
    image=(evidence/'reset-distribution.png').resolve().as_posix()
    destination.write_text(original+f'\n![随机重置分布与GT/NoReset对照]({image})\n'+interpretation,encoding='utf-8')


if __name__=='__main__':report(Path(sys.argv[1]),Path(sys.argv[2]))

"""Render frozen-Align readout comparisons and coverage/count diagnostics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pct(x):return 'NA' if x is None else f'{x*100:.2f}%'
def get(rows,m,variant='individual'):return rows[m] if m=='B3' else rows[m][variant]


def main():
    parser=argparse.ArgumentParser()
    for k in ('results','heads','report'):parser.add_argument('--'+k,required=True)
    args=parser.parse_args();root=Path(args.results);raw=root/'experiment';derived=root/'derived';derived.mkdir(exist_ok=True)
    comparison=load(raw/'comparison.json');results=comparison['results'];select=load(raw/'threshold-selection.json')
    train=load(raw/'head-training.json');frames=load(raw/'sliding-frame-classification.json');verify=load(raw/'verification.json')
    old=load(Path(args.heads)/'head-ablation.json')['panels']['original_optimizer'];phrases=load('configs/radar-text-actions-parrad.json')
    audit_path=derived/'local-verification.json'
    audit=load(audit_path) if audit_path.exists() else None
    labels={'B0':'Stage1 + H1 + 模板','B1':'Align + 新H1 + 模板','B2':'Align + 新H4 + 模板','B3':'Align + 原Projector/Qwen'}
    chosen=max(('B1','B2'),key=lambda m:tuple(get(results['validation']['full12'],m)['metrics'][k] for k in ('sequence_exact','order_lcs','action_recall')))
    main_rows=results['test']['full12'];a=get(main_rows,chosen);b=get(main_rows,'B3');gap=(a['metrics']['sequence_exact']-b['metrics']['sequence_exact'])*100
    relation='高于' if gap>0 else '低于' if gap<0 else '等于'
    link='../outputs/'+root.name+'/'
    lines=['# P0：冻结Align Encoder的分类与语言读出公平对比（2026-10-11）','',
           f'**按validation完整窗口选择的分类流程为{chosen}，完整12秒test序列准确率{pct(a["metrics"]["sequence_exact"])}，{relation}同编码器B3的{pct(b["metrics"]["sequence_exact"])}（分类减B3={gap:+.2f}个百分点）。**','',
           '本轮只训练两个Linear(256,14)头。Align Encoder完全冻结；现有8个有序Tokens、Projector和Qwen均未更新。B3复用上轮同滑动窗口输出，逐窗口重新读取RD/MD/时间戳并核验摘要后配对。','',
           '## 1. 固定协议与选择顺序','',
           '- 新训练特征：Align原best epoch3，eval/no_grad重新提取train受试者02/03/04与validation05；同上轮H1/H4非重叠12秒上下文，每个有效原始帧计一次。新特征与旧Align诊断缓存数值复核，训练不复用旧缓存。',
           '- H1/H4沿用original_optimizer完整策略：20epochs、AdamW lr0.001/wd0.01、batch4096、clip1、seed17；均从相同原Stage1分类头初始化。H1不加权/不标准化；H4训练集逆频率权重与mean/std，标准化初始化保持原始logits。新头并非旧H1/H4权重直接挂载。',
           f'- 验证帧Macro-F1选权重，平分取最早epoch：新H1选epoch{train["H1"]["selected_epoch"]}，新H4选epoch{train["H4"]["selected_epoch"]}。选择用subject05，test06不参与。',
           '- 评价重新提取滑动窗口Align特征，名义12秒/stride6秒/最短8秒/最多3个GT run，与原B3完全相同。原702val/484test各有2个不足12秒尾窗；严格700/482为主，原全集补充。没有从非重叠特征切片构造评价特征。',
           '- 三个分类方法均用相同平滑函数、真实时间戳和9组候选：median0/.3/.5秒×minimum0/.3/.6秒，合并相邻同类，不限制GT动作数。主结果各自仅在完整validation上按序列exact→LCS→Recall选择；共同阈值敏感性也报告。整数class-ID中值存在类别顺序任意性限制。',
           f'- 权重与后处理于{select["locked_at"]}锁定，此后才提取test滑动特征；本报告比较方案{chosen}也仅按validation选择。所有训练统计来自train。','',
           '| 方法 | 独立validation阈值 median/min秒 |','| --- | --- |']
    for m in ('B0','B1','B2'):
        th=select['individual'][m];lines.append(f'| {m} {labels[m]} | {th["median_seconds"]}/{th["minimum_seconds"]} |')
    lines += ['',f'共同阈值：median={select["common"]["median_seconds"]}秒，minimum={select["common"]["minimum_seconds"]}秒。','',
              '## 2. 相同完整12秒窗口：独立validation阈值主结果','',
              '| split | 方法 | 完整序列正确 | Precision | Recall | LCS | Count MAE | 预测数−GT均值 | 少/多/等数量窗口 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split in ('validation','test'):
        for m in labels:
            row=get(results[split]['full12'],m);v=row['metrics']
            lines.append(f'| {split} | {m} {labels[m]} | {row["exact_count"]}/{row["windows"]}={pct(v["sequence_exact"])} | {pct(v["action_precision"])} | {pct(v["action_recall"])} | {pct(v["order_lcs"])} | {row["count_mae"]:.3f} | {row["count_bias_pred_minus_gt"]:+.3f} | {row["under_windows"]}/{row["over_windows"]}/{row["equal_count_windows"]} |')
    lines += ['',f'![同编码器对比]({link}derived/align-readout-comparison.png)','',
              'Count MAE=mean|预测实例数−GT实例数|；符号偏差=mean(预测−GT)。负值表示更倾向少输出，正值表示多输出。数量相等不保证类别或顺序正确；少/多计数也不能自动定位具体时段的识别替代。','',
              '## 3. 原484窗口与共同阈值敏感性','',
              '| subset | 方法 | 阈值策略 | 序列准确率 | Recall | LCS | Count MAE | 少/多窗口 |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for subset,variant in [('original_protocol','individual'),('full12','common'),('original_protocol','common'),('full12','unsmoothed')]:
        for m in labels:
            row=get(results['test'][subset],m,variant);v=row['metrics']
            lines.append(f'| {subset} | {m} | {variant if m!="B3" else "固定B3"} | {pct(v["sequence_exact"])} | {pct(v["action_recall"])} | {pct(v["order_lcs"])} | {row["count_mae"]:.3f} | {row["under_windows"]}/{row["over_windows"]} |')
    lines += ['','## 4. Align是否提高线性可读性','',
              '第一张表使用完全相同非重叠12秒训练/评价上下文和训练控制，对照旧Stage1 H1/H4；第二张表是同滑动窗口的逐帧读出。两者上下文不同，不混用指标。Macro-F1仍不能直接等同于生成性能。','',
              '| split | 头 | Stage1非重叠Frame Accuracy / Macro-F1 | Align非重叠Frame Accuracy / Macro-F1 |','| --- | --- | ---: | ---: |']
    for split in ('validation','test'):
        for arm in ('H1','H4'):
            x,y=old[arm]['results'][split],train[arm]['results'][split]
            lines.append(f'| {split} | {arm} | {pct(x["frame_accuracy"])} / {pct(x["macro_f1"])} | {pct(y["frame_accuracy"])} / {pct(y["macro_f1"])} |')
    lines += ['','| split | Stage1 H1滑动Macro-F1 | Align H1滑动Macro-F1 | Align H4滑动Macro-F1 |','| --- | ---: | ---: | ---: |']
    for split in ('validation','test'):
        lines.append(f'| {split} | '+' | '.join(pct(frames[split][m]['macro_f1']) for m in ('B0','B1','B2'))+' |')
    lines += ['','## 5. 逐类Recall与配对错误','',
              '完整12秒test动作实例多重集覆盖；重复同类按次数计。未作可靠时段对齐，故原类未覆盖可能包含分类替代，不能都解释为纯省略。','',
              '| 动作 | GT实例 | Stage1 H1 | Align H1 | Align H4 | Align Qwen |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for i in range(14):
        rows=[get(main_rows,m)['per_class'][i] for m in labels]
        lines.append(f'| {phrases[str(i)]} | {rows[0]["support"]} | '+' | '.join(pct(r['recall']) for r in rows)+' |')
    lines += ['','| 分类流程 | 分类独对 | B3独对 | 两者都对 | 两者都错 |','| --- | ---: | ---: | ---: | ---: |']
    for m in ('B0','B1','B2'):
        r=main_rows['paired_vs_B3'][m]['individual']
        lines.append(f'| {m} | {r["first_only"]} | {r["second_only"]} | {r["both_correct"]} | {r["both_wrong"]} |')
    lines += ['','## 6. 结论与适用范围','']
    for subset,name in [('full12','完整12秒'),('original_protocol','原全部窗口')]:
        for variant in ('individual','common'):
            rows=results['test'][subset];q=get(rows,'B3')['metrics']['sequence_exact']
            reached=[labels[m] for m in ('B1','B2') if get(rows,m,variant)['metrics']['sequence_exact']>=q]
            lines.append(f'- {name}/{variant}：'+('分类达到或超过B3：'+','.join(reached) if reached else 'B3仍高于两个Align分类模板。'))
    stage1=get(main_rows,'B0');align_h1=get(main_rows,'B1')
    old_gap=100*(b['metrics']['sequence_exact']-stage1['metrics']['sequence_exact'])
    new_gap=100*(b['metrics']['sequence_exact']-align_h1['metrics']['sequence_exact'])
    f1_delta=100*(train['H1']['results']['test']['macro_f1']-old['H1']['results']['test']['macro_f1'])
    lines += ['',f'**Align提高了该配方下的类别可读性，但分类模板仍未赶上B3的完整序列准确率。** H1非重叠test Macro-F1由{pct(old["H1"]["results"]["test"]["macro_f1"])}升至{pct(train["H1"]["results"]["test"]["macro_f1"])}（+{f1_delta:.2f}个百分点）；同滑动窗口序列准确率由{pct(stage1["metrics"]["sequence_exact"])}升至{pct(align_h1["metrics"]["sequence_exact"])}，与B3的差距由{old_gap:.2f}缩至{new_gap:.2f}个百分点。这是本控制配方下的描述性差异，不能当作独立事件的显著性或LLM贡献分解。',
              '',f'**两条流程的错误结构不同。** Align H1完整test窗中多输出动作{align_h1["over_windows"]}个窗口、少输出{align_h1["under_windows"]}个；B3分别为{b["over_windows"]}/{b["under_windows"]}。B3的Count MAE更低，但仍存在明显动作覆盖损失。上床/下床/卧床这三类，Align H1覆盖Recall分别为'+ '/'.join(pct(align_h1['per_class'][i]['recall']) for i in (1,2,3))+'，B3为'+'/'.join(pct(b['per_class'][i]['recall']) for i in (1,2,3))+'。这支持优先检查池化、Projector、文本目标和解码中的证据保留，同时继续控制分类序列碎片与多余动作；本轮不能定位为某个单一模块的责任。']
    lines += ['',f'validation选出的{chosen}在同编码器下，Recall={pct(a["metrics"]["action_recall"])}、LCS={pct(a["metrics"]["order_lcs"])}、Count MAE={a["count_mae"]:.3f}；B3对应{pct(b["metrics"]["action_recall"])}、{pct(b["metrics"]["order_lcs"])}、{b["count_mae"]:.3f}。应按这些不同维度判断完整读出流程，不能将序列准确率等同于全面语言质量。',
              '', '冻结同一Align编码器消除了本轮分类/语言比较的编码器权重差异；仍比较“分类头+平滑+模板”与“8Tokens+池化+Projector+Qwen”的完整读出流程，不能单独量化LLM本身因果贡献。特征可读性变化只对本训练配方成立，非所有可能分类上限。本轮未开展完整2×2。',
              '', '限制：已有test subject06曾被观察；只有一名测试受试者，同步双传感器/重叠窗口相关，不能当独立物理事件作简单显著性结论。窗口通过GT≤3run筛选，动作duration是command区间；不代表任意长序列、真实短动作或跨环境泛化。短语匹配有同义表达/否定/词表外文本盲点，Recall是受限动作代理。','',
              '## 7. 核验与产物','',
              f'Align全部参数和buffer逐位不变，梯度为空；Qwen未加载。原输入/权重/结果{verify["original_inputs_unchanged"]}项SHA前后不变，源码{verify["source_files_verified"]}项核验通过。重新提取{verify["fresh_feature_files"]}份特征，窗口输入摘要匹配validation{verify["paired_window_inputs_verified"]["validation"]}/test{verify["paired_window_inputs_verified"]["test"]}。封存v2执行前相关测试本地23 passed、服务器67 passed（各1个既有依赖warning）；最终入口另增加跨pilot路径的Align/B3 SHA前置校验，修复回归后本地24 passed，原v2训练/推理封存包保持不变。',
              '',f'[机器结果]({link}experiment/comparison.json) · [训练曲线/逐帧指标]({link}experiment/head-training.json) · [validation选择]({link}experiment/threshold-selection.json) · [14类CSV]({link}derived/align-readout-per-class.csv) · [test逐窗口预测]({link}experiment/predictions-test.json) · [模型与输入核验]({link}experiment/verification.json) · [新H1权重]({link}experiment/heads/align-H1.pt) · [新H4权重]({link}experiment/heads/align-H4.pt)',
              '', '服务器输出：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/radar_text_align_readout_p0_20261011_v2`。v1预检查失败记录保留，尚未进行参数训练；修复受试者检查范围后v2重新封存执行。原始雷达/Qwen不下载，特征留服务器；新分类头随结果包导出。','']
    if audit:
        assert audit['state']=='passed'
        lines += [f'本地独立审计通过：{audit["raw_files_sha256_verified"]}份原始文件SHA、{audit["classification_sequences_rebuilt"]}条logits/时间戳重建分类序列、{audit["per_window_metric_rows"]}条逐窗序列主指标、{audit["aggregate_summaries_recomputed"]}组汇总和27组validation候选。Count MAE、数量偏差及多/少窗口分解独立重算一致；B3文本与旧输出一致，旧输入摘要与Stage1 H1 logits一致。CSV含14类共560行；审计不替代人工文本事实性评价，也不声称独立重算全部派生字段。',
                  '',f'[本地独立审计]({link}derived/local-verification.json) · [派生产物SHA清单]({link}derived/derived-manifest.json)','']
    Path(args.report).write_text('\n'.join(lines),encoding='utf-8')
    csv_rows=[]
    for split,splits in results.items():
        for subset,methods in splits.items():
            for m in labels:
                for variant in (('individual',) if m=='B3' else ('individual','common','unsmoothed')):
                    for r in get(methods,m,variant)['per_class']:
                        csv_rows.append(dict(split=split,subset=subset,method=m,method_name=labels[m],variant=variant,action=phrases[str(r['class_id'])],**r))
    with (derived/'align-readout-per-class.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    fig,axes=plt.subplots(1,3,figsize=(13,4.3));x=np.arange(4)
    for ax,key,title,scale in zip(axes,['sequence_exact','action_recall','count_mae'],['Sequence exact (%)','Action recall (%)','Count MAE'],[100,100,1]):
        values=[scale*(get(main_rows,m)[key] if key=='count_mae' else get(main_rows,m)['metrics'][key]) for m in labels]
        bars=ax.bar(x,values,color=['#9a9a9a','#2576b8','#45a28d','#e68b32'])
        ax.set(xticks=x,xticklabels=['S1 H1','Align H1','Align H4','Align Qwen'],title=title)
        ax.grid(axis='y',alpha=.2);ax.set_ylim(0,max(values)*1.2)
        for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+max(values)*.025,f'{value:.2f}',ha='center',fontsize=9)
    fig.suptitle('P0: matched full12s test n=482; Stage1 reference and frozen Align readouts');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(derived/f'align-readout-comparison.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(report=args.report,validation_selected=chosen,csv_rows=len(csv_rows))))


if __name__=='__main__':main()

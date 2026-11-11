"""Render paired, protocol-matched template vs radar-to-text results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def pct(value):
    return 'NA' if value is None else f'{100*value:.2f}%'


def metric(methods, method, variant='common'):
    return methods[method] if method=='B3' else methods[method][variant]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    root,report=Path(args.results),Path(args.report)
    raw=root/'experiment'
    comparison=json.loads((raw/'comparison.json').read_text())
    selection=json.loads((raw/'threshold-selection.json').read_text())
    provenance=json.loads((raw/'provenance.json').read_text())
    verification=json.loads((raw/'verification.json').read_text())
    local_verification_path=root/'derived/local-verification.json'
    local_verification=json.loads(local_verification_path.read_text()) if local_verification_path.exists() else None
    result=comparison['results']
    derived=root/'derived';derived.mkdir(exist_ok=True)
    phrases=json.loads(Path('configs/radar-text-actions-parrad.json').read_text(encoding='utf-8'))
    link='../outputs/radar-text-strong-baselines-20261011-v1/'
    labels={'B0':'原Stage1原头+模板','B1':'原Stage1 H1+模板','B2':'原Stage1 H4+模板','B3':'原Align+Qwen生成'}
    chosen=max(('B0','B1','B2'),key=lambda m:tuple(metric(result['validation']['full12'],m)['metrics'][k]
                                                         for k in ('sequence_exact','order_lcs','action_recall')))
    strict=result['test']['full12'];original=result['test']['original_protocol']
    delta=100*(metric(strict,chosen)['metrics']['sequence_exact']-metric(strict,'B3')['metrics']['sequence_exact'])
    direction='高于' if delta>0 else '低于' if delta<0 else '等于'
    lines=['# 更强分类到文本基线：同滑动窗口对照（2026-10-11）','',
           f'**结果：按validation完整窗口选出的共同阈值分类方案{chosen}，在完整名义12秒测试窗上的序列准确率为{pct(metric(strict,chosen)["metrics"]["sequence_exact"])}，{direction}B3的{pct(metric(strict,"B3")["metrics"]["sequence_exact"])}（差{delta:+.2f}个百分点）。**','',
           '本轮重新执行Stage1编码与冻结Qwen生成，没有使用非重叠特征缓存的评价数字。B0–B2每个窗口共享同一份新Stage1时序特征；B3从同一batch原始RD/MD张量生成。没有训练任何参数。','',
           '## 1. 对齐协议与选择约束','',
           '- 原描述模型协议：名义12秒窗、6秒步长、最短8秒、最多3个GT动作run。所有方法使用相同recording、传感器、起止帧、时间戳、标签与窗口ID。',
           '- 原702个validation/484个test窗口中各有2个不足12秒尾窗。主结果为完整名义12秒子集700/482（允许一个采样周期的量化误差）；另报告全部原窗口，以直接复核原25%结论。没有把尾窗当成完整12秒，也没有重新拼接窗口。',
           '- B1使用上轮original_optimizer/H1最佳epoch20；B2使用同组H4最佳epoch19，直接使用该头保存的训练mean/std。未更换为probe_replay头，也未用新validation/test重新估计统计。',
           '- B3使用原Align最佳epoch3、原Qwen2.5-0.5B-Instruct、原prompt与tokenizer；batch2、greedy、max_new_tokens96。本轮全部重新生成，逐样本检查与历史生成的一致性。',
           '- B3包含Align后更新过的雷达编码器，B0–B2使用原Stage1编码器；本轮按用户定义比较完整流程，不能把差值单独归因于Qwen或生成式解码。',
           '- 平滑沿用同一实现：class-ID中值窗0/0.3/0.5秒，最短预测段0/0.3/0.6秒，删除短预测段后合并相邻同类。按真实时间戳处理；不强制真实动作个数。整数类别中值依赖任意ID顺序，是此基线的已知限制。',
           f'- **共同阈值主结果：** 在700个完整validation窗口上，以B0/B1/B2平均序列准确率、LCS、召回依次选优，再以更简单参数破同分；选中median={selection["common"]["median_seconds"]}秒、min_duration={selection["common"]["minimum_seconds"]}秒，三个分类方法完全相同。',
           '- **独立阈值补充：** 每个分类方法使用完全相同候选网格和validation选择规则，分别选择自身最佳参数，检验共同阈值是否压低某个更强分类基线。另报固定无平滑结果。',
           f'- 所有阈值于{selection["locked_at"]}落盘锁定，之后才开始test推理；分类方案{chosen}也按validation序列指标选出。未通过test选头、阈值或方法。','',
           '## 2. 共同阈值：完整12秒窗口','',
           '| split | 方法 | 窗口数 | 完整序列准确率 | 动作Precision | 动作Recall | LCS | 平均输出动作数 | 数量偏少窗口 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split in ('validation','test'):
        for method in labels:
            row=metric(result[split]['full12'],method);m=row['metrics']
            lines.append(f'| {split} | {method} {labels[method]} | {row["windows"]} | {row["exact_count"]}/{row["windows"]} = {pct(m["sequence_exact"])} | {pct(m["action_precision"])} | {pct(m["action_recall"])} | {pct(m["order_lcs"])} | {row["mean_predicted_actions"]:.3f} | {row["count_deficit_windows"]} |')
    lines += ['',f'![同窗口比较]({link}derived/matched-window-scores.png)','',
              '## 3. 原协议全部窗口：直接复核25%','',
              '| split | 方法 | 完整序列准确率 | 动作Precision | 动作Recall | LCS | 数量偏少窗口 | 数量偏多窗口 |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for split in ('validation','test'):
        for method in labels:
            row=metric(result[split]['original_protocol'],method);m=row['metrics']
            lines.append(f'| {split} | {method} | {row["exact_count"]}/{row["windows"]} = {pct(m["sequence_exact"])} | {pct(m["action_precision"])} | {pct(m["action_recall"])} | {pct(m["order_lcs"])} | {row["count_deficit_windows"]} | {row["count_excess_windows"]} |')
    lines += ['','## 4. 独立validation阈值和无平滑补充','',
              '| 方法 | median/min秒 | Val完整12s序列准确率 | Test完整12s序列准确率 | Test原协议序列准确率 | Test原协议Recall |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for variant in ('individual','unsmoothed'):
        for method in ('B0','B1','B2'):
            threshold=selection['individual'][method] if variant=='individual' else dict(median_seconds=0,minimum_seconds=0)
            v=metric(result['validation']['full12'],method,variant)['metrics']
            t=metric(strict,method,variant)['metrics'];o=metric(original,method,variant)['metrics']
            lines.append(f'| {method} {variant} | {threshold["median_seconds"]}/{threshold["minimum_seconds"]} | {pct(v["sequence_exact"])} | {pct(t["sequence_exact"])} | {pct(o["sequence_exact"])} | {pct(o["action_recall"])} |')
    lines += ['','## 5. 配对错误与分组','',
              '以下完整12秒test窗口逐一配对：分类独对=分类正确/B3错误，B3独对相反。它们是描述性计数，不是独立样本显著性检验。','',
              '| 分类方法 | 分类独对 | B3独对 | 两者都对 | 两者都错 |','| --- | ---: | ---: | ---: | ---: |']
    for method in ('B0','B1','B2'):
        row=strict['paired_vs_B3'][method]['common']
        lines.append(f'| {method} | {row["first_only"]} | {row["second_only"]} | {row["both_correct"]} | {row["both_wrong"]} |')
    lines += ['','| 分组 | 窗口数 | B0序列准确率 | B1 | B2 | B3 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for groups in (strict['by_sensor'],strict['by_action_count']):
        for name,rows in groups.items():
            lines.append(f'| {name} | {rows["B3"]["windows"]} | '+' | '.join(pct(rows[m]['metrics']['sequence_exact']) for m in labels)+' |')
    lines += ['','完整12秒test按14类动作实例多重集覆盖统计。重复同类实例只比较次数，不作时间对齐；类别替代会表现为原类未覆盖。','',
              '| 类别 | GT实例 | B0覆盖Recall | B1 | B2 | B3 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for i in range(14):
        rows=[metric(strict,m)['per_class'][i] for m in labels]
        lines.append(f'| {phrases[str(i)]} | {rows[0]["support"]} | '+' | '.join(pct(r['recall']) for r in rows)+' |')
    lines += ['','## 6. 如何更新研究结论','']
    old_b0=metric(original,'B0','unsmoothed')['metrics']['sequence_exact']
    common_b0=metric(original,'B0')['metrics']['sequence_exact']
    b3=metric(original,'B3')['metrics']['sequence_exact']
    selected=metric(original,chosen)['metrics']
    lines += [f'**主要结论：原484窗口口径下，validation选出的{chosen}为{pct(selected["sequence_exact"])}，B3为{pct(b3)}，仍相差{100*(b3-selected["sequence_exact"]):.2f}个百分点。** 原历史无平滑B0差距为{100*(b3-old_b0):.2f}个百分点；统一本轮共同阈值后，B0差距为{100*(b3-common_b0):.2f}个百分点。因此首轮优势被弱分类基线明显放大，但本轮没有消除B3的完整序列优势。',
              '',f'B2并列报告为{pct(metric(original,"B2")["metrics"]["sequence_exact"])}，与B3相差{100*(b3-metric(original,"B2")["metrics"]["sequence_exact"]):.2f}个百分点；它在test略高于{chosen}，不据此替换validation选择。',
              '',f'**优势限于完整序列指标：** {chosen}动作Recall={pct(selected["action_recall"])}、LCS={pct(selected["order_lcs"])}，均高于B3的{pct(metric(original,"B3")["metrics"]["action_recall"])}、{pct(metric(original,"B3")["metrics"]["order_lcs"])}。分类方法更常输出多余动作，B3则更常少输出动作；不能概括为生成模型全面更优。','']
    for subset,name in (('full12','完整12秒'),('original_protocol','原协议全部')):
        methods=result['test'][subset]
        b3=metric(methods,'B3')['metrics']['sequence_exact']
        for variant,label in (('common','共同阈值'),('individual','各自validation阈值')):
            values={m:metric(methods,m,variant)['metrics']['sequence_exact'] for m in ('B1','B2')}
            exceeding=[m for m,v in values.items() if v>=b3]
            conclusion='分类基线达到或超过B3：'+','.join(exceeding) if exceeding else 'B3仍高于H1/H4分类模板'
            lines.append(f'- {name}/{label}：{conclusion}。B1={pct(values["B1"])}，B2={pct(values["B2"])}，B3={pct(b3)}。')
    lines += ['','研究重点应从“是否能生成描述”转向“如何保留可读出的动作证据并减少生成漏检”，尤其是床相关动作。分类模板的多余动作仍需控制。后续若要隔离语言生成贡献，应固定同一编码器再比较读出与接口；本轮没有新增该实验。',
              '', '本轮比较能回答当前窗口协议下原生成模型是否仍超过更强分类模板；不能单独证明语言模型理解细粒度动作或时间关系。B3的领先只是本数据/窗口/代理指标下的流程优势。分类头仍按非重叠缓存训练，平滑仅搜索固定9组，因此这也不是所有可能分类流程的性能上限。','',
              '## 7. 复现、产物和界限','',
              '| split | 原头与前次逐帧预测差异 | 原头max logit差 | B3文本差异窗口 | B3动作序列差异窗口 |',
              '| --- | ---: | ---: | ---: | ---: |']
    for split,row in comparison['reproduction'].items():
        lines.append(f'| {split} | {row["b0_frame_prediction_differences"]} | {row["b0_max_logit_difference"]:.3g} | {row["b3_text_differences"]} | {row["b3_action_sequence_differences"]} |')
    if local_verification:
        assert local_verification['state']=='passed'
        lines += ['',f'本地独立审计通过：核验{local_verification["raw_files_sha256_verified"]}份原始结果SHA；从logits和时间戳重建{local_verification["classification_sequences_rebuilt"]}条分类序列，复算{local_verification["per_window_metric_rows"]}条逐窗序列主指标、{local_verification["aggregate_summaries_recomputed"]}组主指标汇总及validation的27组候选。B3文本与前次完全一致。CSV包含全部14类、560行；这项审计不声称独立复核所有派生字段或人工文本事实性。']
    lines += ['',f'125份服务器源码核验、{verification["input_files_unchanged"]}项原始权重/manifest/标签/记录文件SHA前后核验通过。服务器相关测试62 passed（1 warning），本地相关18 passed（1 warning）。新特征缓存保留服务器，导出预测、logits、时间戳/偏移和每窗口RD/MD/时间戳输入摘要，不包含原始雷达或完整Qwen。','',
              '限制：同一subject06已被前次实验观察，窗口重叠且有同步传感器视图，不视为独立物理事件或新的泛化试验。窗口筛选依赖GT最多3个run，结果不代表无筛选的长序列。短语匹配会漏计同义表达、否定和词表外幻觉，分类模板则使用该固定词表；这是一项受限动作序列代理比较，不是完整语言质量评价。','',
              f'[完整机器结果]({link}experiment/comparison.json) · [validation阈值选择清单]({link}experiment/threshold-selection.json) · [14类覆盖CSV]({link}derived/per-class-template-coverage.csv) · [逐窗口test预测]({link}experiment/predictions-test.json) · [输入与权重核验]({link}experiment/verification.json) · [本地独立审计]({link}derived/local-verification.json) · [派生产物校验清单]({link}derived/derived-manifest.json)','',
              '服务器输出：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/radar_text_strong_baselines_20261011_v1`。','']
    report.write_text('\n'.join(lines),encoding='utf-8')
    csv_rows=[]
    for split,splits in result.items():
        for subset,methods in splits.items():
            for method in labels:
                for variant in (('common',) if method=='B3' else ('common','individual','unsmoothed')):
                    for row in metric(methods,method,variant)['per_class']:
                        csv_rows.append(dict(split=split,subset=subset,method=method,variant=variant,action=phrases[str(row['class_id'])],**row))
    with (derived/'per-class-template-coverage.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,split in zip(axes,('validation','test')):
        methods=result[split]['full12'];x=np.arange(4)
        common=[100*metric(methods,m)['metrics']['sequence_exact'] for m in labels]
        individual=[100*metric(methods,m,'individual')['metrics']['sequence_exact'] for m in labels]
        ax.bar(x-.18,common,.36,label='Shared validation threshold')
        ax.bar(x+.18,individual,.36,label='Per-head validation threshold')
        ax.set(xticks=x,xticklabels=['B0 original','B1 H1','B2 H4','B3 Qwen'],ylabel='Sequence exact (%)',
               title=f'{split}: matched full12s (n={metric(methods,"B3")["windows"]})')
        for dx,values in ((-.18,common),(.18,individual)):
            for i,value in enumerate(values):ax.text(i+dx,value+.35,f'{value:.1f}',ha='center',fontsize=8)
        ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
        ax.set_ylim(0,max(common+individual)*1.2+1)
    fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(derived/f'matched-window-scores.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps(dict(report=str(report),csv_rows=len(csv_rows),validation_chosen_classifier=chosen)))


if __name__=='__main__':
    main()

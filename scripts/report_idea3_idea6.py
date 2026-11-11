"""Create readable tables from the unified existing-model evaluation."""
import csv
import json
from pathlib import Path
import sys

import numpy as np


def report(directory):
    directory=Path(directory)
    data=json.loads((directory/'complete.json').read_text())
    arms=sorted(data['pooled'])
    rows=[]
    for arm in arms:
        for subject,summary in data['by_subject'].items():
            value=summary[arm]
            rows.append(dict(arm=arm,subject=subject,
                conservative_event_f1=value['conservative_event']['f1'],
                conservative_event_macro_f1=value['conservative_event']['macro_f1'],
                historical_event_f1=value['historical_event']['f1'],
                interval_f1=value['interval_only']['f1'],boundary_f1=value['boundary']['f1'],
                boundary_recall=value['boundary']['recall'],fp_min=value['boundary']['fp_min'],
                confirmation_p90_s=value['boundary']['confirmation_p90_s'],
                online_macro_f1=value['record_mean_online_macro_f1'],
                mixed_fp=value['mixed_fp'],unknown_fires=value['unknown_fires']))
    with (directory/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# HARrad 想法三与想法六统一评价', '',
        '共同前端为旧 RD seed42。想法三参评模型为监督起止头加 EventFormer。各模型沿用历史操作点和原生决策频率。',
        'subject7 已用于模型选择，subject8–9 已在此前 SAEF 研究中评估。本表为既有模型复核，单 seed42。', '',
        '|模型|受试者|保守事件F1|历史事件F1|区间F1|边界F1|边界召回|FP/min|确认延迟P90秒|在线Macro-F1|',
        '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        delay=r['confirmation_p90_s']
        fields=[r['arm'],r['subject']]
        fields += [f"{100*r[k]:.2f}%" for k in ('conservative_event_f1','historical_event_f1','interval_f1','boundary_f1','boundary_recall')]
        fields += [f"{r['fp_min']:.2f}", f'{delay:.3f}' if delay is not None else 'N/A',f"{100*r['online_macro_f1']:.2f}%"]
        lines.append('|'+ '|'.join(fields)+'|')
    means={arm:float(np.mean([r['conservative_event_f1'] for r in rows if r['arm']==arm and r['subject']!='subject7'])) for arm in arms}
    native=[a for a in arms if a.endswith('-native') or a=='idea3-start-end']
    common=[a for a in arms if a.endswith('-shared-reader') or a=='idea3-start-end']
    lines += ['', '按 subject8 和 subject9 等权平均的保守事件 F1，现有原生系统最高的行为 '+max(native,key=means.get)+'。',
        '在共用 EventFormer 的比较中，最高的行为 '+max(common,key=means.get)+'。',
        '各行实际误报率和决策频率可能不同。完整预测驱动 Reset、三 seed 机制比较及相同预算操作点仍待完成。',
        'EOF 强制关闭参与区间评价，边界指标排除录制边缘。未知区间负担和逐事件匹配保留在完整 JSON。']
    (directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    checked=0
    for record in data['records']:
        for key in ('conservative','events'):
            for metric in record['metric'][key].values():
                expected=2*metric['tp']/(2*metric['tp']+metric['fp']+metric['fn']) if 2*metric['tp']+metric['fp']+metric['fn'] else 0.
                assert abs(expected-metric['f1'])<1e-12
                checked+=1
    (directory/'report-verification.json').write_text(json.dumps(dict(
        count_arithmetic_checks=checked,heldout_subject_mean_conservative_event_f1=means,
        checks='recomputed F1 from saved counts; full matcher covered by regression tests'),indent=2))


if __name__=='__main__':
    report(sys.argv[1])

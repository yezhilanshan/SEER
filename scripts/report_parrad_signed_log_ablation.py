"""Report real completed validation results only; safe during unfinished training."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pct(x):return f'{100*x:.3f}%'


def build_report(root):
    root=Path(root);policy=read(root/'experiment-policy.json');status=read(root/'status.json') if (root/'status.json').exists() else {'state':'preparing'}
    classes=policy.get('action_phrases',{})
    lines=['# PARrad Signed-Log P0/P1 严格受控消融', '',
        f"状态：**{status['state']}**。开始于 {policy['started']}。只使用 Train/Validation；测试集不参与训练、评价或方案选择。",'',
        '## 实验目的与固定设置','',
        '检验额外 Signed-Log 是否影响 RD-only ResNet18 + BiGRU + 14 类逐帧分类；不改变网络或追求新最优方案。P0 为 Signed-Log + 有效窗口 Z-score；P1 仅移除 Signed-Log，使用官方发布 RD 原值做相同 Z-score。', '',
        '数据与原 Stage1 一致：train01/02/03/04/07/08/09/13/19，validation05/12/16，test06/17/18；窗口12s/stride6s/min8s/max3动作，不改标签、不跨录制，不裁剪、不滤波、不限幅、不训练 MD/LLM。', '',
        '每臂20 epochs，无 early stopping；batch2/累积2，AdamW，encoder lr=0.0003、classifier lr=0.001、weight_decay=.01、clip1，CosineAnnealingLR eta_min=1e-6；相同训练窗口计算的类别权重、FP16 AMP、frame_chunk64、dropout=.1。以 Validation 受试者平均 Macro-F1 选优，平分依次看 pooled Macro-F1、较低未加权 CE。', '',
        '每个 seed 两臂使用同一保存的初始 tensor/RNG 文件，核验整个 state_dict（含 BN buffer）内容 SHA256；每 epoch 采样顺序有独立 RNG 并保存全序列与散列。数值配置保持一致，但不声称 CUDA 位级确定性。AMP 跳步属于运行结果，逐臂记录，不通过改超参数消除。', '',
        'seed17 的主指标绝对差达到预先约定的1个百分点才追加29/43；未达到则不把多帧数量当作独立样本宣称显著。', '',
        '## 代码修改','',
        '`rd_transfer.normalize_rd` 增加默认 signed_log_zscore / 可选 raw_zscore；`rd_scratch` 传递并保存模式，旧 checkpoint 缺失模式时默认 P0。训练入口增加共享初始化和 Validation-only 选项。辅助模块核验参数/采样顺序，按完整录制标签排除转换点±0.3s后再切窗。模型选优仍使用全部有效帧，steady 指标仅作诊断。', '',
        '执行源码版本见每臂 provenance.json 的 source_sha256；具体函数行号见 source-lines.json。单元测试和 Smoke Test 记录见上级启动归档；不能用 Smoke 指标充当正式结果。','']
    completed=[]
    for seed in [17,29,43]:
        pair=root if seed==17 else root/f'seed-{seed}'
        if not pair.exists():continue
        available={arm:read(pair/arm/'results.json')['validation'] for arm in ['P0','P1'] if (pair/arm/'results.json').exists()}
        lines.extend([f'## Seed {seed}',''])
        if set(available)!={'P0','P1'}:
            lines.extend([f"尚未完成配对正式评价，已有结果的组：{list(available)}。以下不填充缺失分数。",'']);continue
        d={arm:read(pair/arm/'validation-diagnostics.json') for arm in available}
        p0,p1=available['P0'],available['P1'];delta=p1['subject_macro_f1']-p0['subject_macro_f1']
        completed.append((seed,available,d))
        metrics=[('主指标：受试者平均 Macro-F1',p0['subject_macro_f1'],p1['subject_macro_f1']),
            ('Pooled Macro-F1',p0['macro_f1'],p1['macro_f1']),('Pooled Accuracy',p0['frame_accuracy'],p1['frame_accuracy']),
            ('Pooled Macro-Precision',p0['macro_precision'],p1['macro_precision']),('Pooled Macro-Recall',p0['macro_recall'],p1['macro_recall']),
            ('Pooled Steady Macro-F1',d['P0']['steady']['macro_f1'],d['P1']['steady']['macro_f1']),
            ('床相关 Macro-F1（1/2/3/4/5/7/8）',d['P0']['bed_macro_f1'],d['P1']['bed_macro_f1'])]
        lines.extend([table(['指标','P0','P1','P1−P0 (百分点)'],[[name,pct(a),pct(b),f'{100*(b-a):+.3f}'] for name,a,b in metrics]),'',
            f"Validation未加权CE：P0={p0['unweighted_ce']:.6f}，P1={p1['unweighted_ce']:.6f}，差={p1['unweighted_ce']-p0['unweighted_ce']:+.6f}。",''])
        for field,title in [('by_subject','逐受试者'),('by_sensor','逐设备')]:
            lines.extend([f'### {title}','',table(['对象','P0 Macro-F1','P1 Macro-F1','差 (百分点)'],
                [[key,pct(d['P0'][field][key]['macro_f1']),pct(d['P1'][field][key]['macro_f1']),
                 f"{100*(d['P1'][field][key]['macro_f1']-d['P0'][field][key]['macro_f1']):+.3f}"] for key in d['P0'][field]]),''])
        lines.extend(['### 14 类评价','',table(['类','动作','P0 P/R/F1','P1 P/R/F1','F1差 (百分点)'],
            [[i,classes.get(str(i),str(i)),' / '.join(pct(p0['per_class'][i][k]) for k in ['precision','recall','f1']),
              ' / '.join(pct(p1['per_class'][i][k]) for k in ['precision','recall','f1']),
              f"{100*(p1['per_class'][i]['f1']-p0['per_class'][i]['f1']):+.3f}"] for i in range(14)]),''])
        for arm in ['P0','P1']:
            history=read(pair/arm/'history.json');selected=read(pair/arm/'selection.json');run=read(pair/arm/'status.json')
            lines.extend([f"{arm} Best Epoch={selected['epoch']}；总耗时={run['elapsed_seconds']/3600:.3f}h；AMP跳步={history[-1]['amp_overflows']}；成功optimizer steps={history[-1]['optimizer_steps']}。",''])
        matrix0,matrix1=p0['confusion_matrix'],p1['confusion_matrix'];changes=[]
        for i in [1,2,3,5,8,9,10]:
            for j in range(14):
                if i!=j and (matrix0[i][j] or matrix1[i][j]):changes.append((matrix1[i][j]-matrix0[i][j],i,j,matrix0[i][j],matrix1[i][j]))
        changes.sort(key=lambda x:abs(x[0]),reverse=True)
        lines.extend(['### 重点类别混淆变化','',table(['GT','预测','P0帧次','P1帧次','差'],[[classes.get(str(i),i),classes.get(str(j),j),a,b,delta] for delta,i,j,a,b in changes[:14]]),
            '', '这里统计的是滑动窗口中的帧次，包含重复观测；混淆减少不等于对该动作的因果机制已确认。',''])
        fairness=read(pair/'fairness.json') if (pair/'fairness.json').exists() else None
        lines.extend([f"公平性核验：{json.dumps(fairness,ensure_ascii=False) if fairness else '配对核验尚未完成'}",''])
        prefix='' if seed==17 else f'seed-{seed}/'
        lines.extend([f'![训练与验证曲线]({prefix}assets/curves.png)',f'![14×14混淆矩阵]({prefix}assets/confusion.png)',''])
    lines.extend(['## 机制解释与最终判断','',
        '此前输入审计发现 Signed-Log 会压缩强值对主体的对比，也会使极少数低值相对主体更突出（约−50.7σ）。本试验只能估计整个该变换的净影响，不能单独把性能差异归因为极端尾部；需要其他独立机制消融才能作因果分解。',''])
    if completed:
        deltas=[a['P1']['subject_macro_f1']-a['P0']['subject_macro_f1'] for _,a,_ in completed]
        lines.append(table(['seed','P0主指标','P1主指标','配对差 (百分点)'],[[seed,pct(a['P0']['subject_macro_f1']),pct(a['P1']['subject_macro_f1']),f"{100*(a['P1']['subject_macro_f1']-a['P0']['subject_macro_f1']):+.3f}"] for seed,a,_ in completed]));lines.append('')
        if len(completed)==3:
            for arm in ['P0','P1']:
                values=[a[arm]['subject_macro_f1'] for _,a,_ in completed]
                lines.append(f'{arm} 主指标 Mean±sample Std：{100*statistics.mean(values):.3f}±{100*statistics.stdev(values):.3f}个百分点。')
            lines.append(f'配对差 Mean±sample Std：{100*statistics.mean(deltas):+.3f}±{100*statistics.stdev(deltas):.3f}个百分点。')
        for field,title in [('by_subject','受试者跨种子方向'),('by_sensor','设备跨种子方向')]:
            keys=completed[0][2]['P0'][field]
            rows=[]
            for key in keys:
                differences=[d['P1'][field][key]['macro_f1']-d['P0'][field][key]['macro_f1'] for _,_,d in completed]
                rows.append([key,f'{100*statistics.mean(differences):+.3f}',f'{sum(v>0 for v in differences)}/{len(differences)}'])
            lines.extend(['',title,'',table(['对象','平均配对差 (百分点)','P1更高的种子数'],rows),''])
        bed_delta=statistics.mean([d['P1']['bed_macro_f1']-d['P0']['bed_macro_f1'] for _,_,d in completed])
        lines.extend([f'床相关 Macro-F1 的已完成种子平均差为 {100*bed_delta:+.3f} 个百分点；具体混淆和逐类变化见上表，不能把总体收益自动解释为床相关动作全部改善。',''])
        lines.extend(['',f"已完成种子中平均配对差为 {100*statistics.mean(deltas):+.3f} 个百分点。逐受试者/设备方向以各表为准，不以相关帧数宣称统计显著。",''])
        if status['state']=='complete':
            mean=statistics.mean(deltas)
            if abs(mean)<.01:
                conclusion='当前未见达到预设幅度的主指标收益，不能确认 Signed-Log 是主要瓶颈；不据此重训双分支、Align 或 LLM。'
            elif mean>0:
                conclusion='删除 Signed-Log 在本组已完成实验的主指标上有收益；需结合种子及对象方向判断稳定性。可将相同成对试验推广到 RD+MD，暂不能直接推断双分支/Align/LLM 都应全面重训。'
            else:
                conclusion='删除 Signed-Log 在本组主指标上下降，现有证据不支持因输入尾部现象直接改用 P1 或全面重训；不排除其他输入表示问题。'
            lines.extend([conclusion,''])
        else:lines.extend(['实验仍未结束，以上均为阶段性真实数据，完整结论暂不下定。',''])
    else:lines.extend(['正式配对训练尚未完成，Macro-F1、逐类改善和是否推广双分支均待结果；没有使用模拟数据。',''])
    lines.extend(['## 结果限制','',
        '仅验证现有命令弱标签与当前窗口协议；标签时钟起点、发令反应时延、环境元数据和不同设备物理速度轴风险保持原样，未在两臂间改动。Test未读取。subject仅3名、设备2种，帧和滑动窗口高度相关；不把相关帧视为独立统计样本。',''])
    (root/'parrad_signed_log_ablation.md').write_text('\n'.join(lines),encoding='utf-8')
    # Use already installed system Matplotlib; no installation in the training env.
    plotter=Path(__file__).with_name('plot_parrad_signed_log_ablation.py')
    if completed and plotter.exists():
        python='/usr/bin/python3' if Path('/usr/bin/python3').exists() else sys.executable
        result=subprocess.run([python,str(plotter),'--root',str(root)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (root/'plot-status.json').write_text(json.dumps(dict(returncode=result.returncode,log=result.stdout),indent=2),encoding='utf-8')
        if result.returncode:
            with (root/'parrad_signed_log_ablation.md').open('a',encoding='utf-8') as output:
                output.write('\n绘图未成功，训练不因此中断；真实数值结果已保存，错误见 plot-status.json，图片需补生成。\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);args=p.parse_args();build_report(args.root)

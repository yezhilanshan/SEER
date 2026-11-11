"""Render the frozen-head comparison, training curves and all-class CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pct(x):
    return 'NA' if x is None else f'{100*x:.2f}%'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    root, report = Path(args.results), Path(args.report)
    data = json.loads((root/'experiment/head-ablation.json').read_text())
    provenance = json.loads((root/'experiment/provenance.json').read_text())
    verification = json.loads((root/'experiment/verification.json').read_text())
    phrases = json.loads(Path('configs/radar-text-actions-parrad.json').read_text(encoding='utf-8'))
    derived = root/'derived'
    derived.mkdir(exist_ok=True)
    base_link = '../outputs/radar-text-head-ablation-20261011-v1/'
    all_rows = []
    for panel, rows in [('reference', {'H0@0': dict(results=data['reference']['results'])}), *data['panels'].items()]:
        for arm, row in rows.items():
            for split in ('validation','test'):
                for subset in ('all_valid','steady'):
                    metric = row['results'][split]
                    if subset == 'steady': metric = metric['steady']
                    for cls in metric['per_class']:
                        all_rows.append(dict(panel=panel, arm=arm, split=split, subset=subset,
                                             action=phrases[str(cls['class_id'])], **cls))
    with (derived/'per-class-head-metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, (panel, rows) in zip(axes, data['panels'].items()):
        for arm, row in rows.items():
            ax.plot([h['epoch'] for h in row['history']], [100*h['validation_macro_f1'] for h in row['history']], label=arm, marker='o', markersize=3)
        ax.axhline(100*data['reference']['results']['validation']['macro_f1'], color='grey', linestyle='--', label='original H0@0')
        ax.set(title=panel, xlabel='Frozen-head epochs', ylabel='Validation Macro-F1 (%)')
        ax.grid(alpha=.2); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    for suffix in ('png','svg'): fig.savefig(derived/f'validation-head-curves.{suffix}', dpi=180)
    plt.close(fig)

    lines = ['# Stage1 冻结分类头受控消融（2026-10-11）','',
             '**结论：冻结已有特征后，重训读出即可明显改善分类。原优化器组中，延长训练的收益较稳定，类权重的收益主要出现在验证受试者，标准化单独收益较小；没有证据支持把原差距主要归为标准化或概率校准问题。组合H4表现最好，但不能分离组合交互。**','',
             '本轮只训练256→14线性头（3598参数），没有实例化/运行CNN、BiGRU或Qwen；原Stage1权重、38份特征缓存均未改变。两组各5个头已完成，所有设置在观察本轮test结果前固定。','',
             '## 1. 两个复现端点','',
             '- **H0@0原分类头：** 直接载入原Stage1最佳epoch2分类头，零更新；train/validation/test逐帧预测、完整14类混淆矩阵及稳态矩阵与前次完全一致。',
             f'- **H4旧探针：** probe_replay组复现已有Stage1 BiGRU探针；选中epoch{data["probe_replay_verification"]["selected_epoch"]}，验证和测试预测差异均0帧，20epoch的验证Macro-F1历史完全一致。最大权重绝对差{data["probe_replay_verification"]["max_head_difference"]:.3g}，仅有微小浮点差异。','',
             '| 原分类头H0@0 | Frame Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | 稳态Macro-F1 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for split in ('train','validation','test'):
        r = data['reference']['results'][split]
        lines.append(f'| {split} | {pct(r["frame_accuracy"])} | {pct(r["macro_precision"])} | {pct(r["macro_recall"])} | {pct(r["macro_f1"])} | {pct(r["steady"]["macro_f1"])} |')
    lines += ['',f'最大logit绝对差：{data["reference"]["max_logit_difference"]}。这里“精确复现”指预测和指标相同；没有把重训后的H0冒充原分类头。','',
              '## 2. 受控设置与解释范围','',
              '| 实验 | 训练预算 | 类别权重 | 特征标准化 |','| --- | ---: | --- | --- |',
              '| H0 | 3 epoch | 无 | 无 |','| H1 | 20 epoch | 无 | 无 |','| H2 | 3 epoch | 有 | 无 |',
              '| H3 | 3 epoch | 无 | 有 |','| H4 | 20 epoch | 有 | 有 |','',
              '- **original_optimizer主分析组：** 全部从原分类头开始，AdamW lr=0.001、weight_decay=0.01、grad_clip=1；原优化器状态未保存，所有头均重置优化器。标准化臂将原参数折叠为 `W_std=W_raw*std; b_std=b_raw+W_raw*mean`，保持初始输出函数相同。',
              '- **probe_replay复现组：** 全部使用旧探针随机初始化seed17，AdamW lr=0.01、weight_decay=0.01、不裁剪。标准化/原始输入下虽使用相同随机权重，初始函数并不相同，因此标准化的解释以主分析组为主。',
              '- **面板内部完全相同：** Stage1 GRU缓存、训练/验证/测试划分、batch4096帧、seed17、打乱顺序前缀、线性结构和评价方式。每epoch40次更新，H0/H1前3epoch训练历史完全一致。类权重`N/(14*n_c)`、均值/标准差只从训练数据计算，std下限1e-5。',
              '- **选优：** 每个头只按验证Macro-F1选epoch，并列取最早；同时报告预算末轮，区分延长训练与更多选优机会。未在test选择超参、epoch或实验臂。',
              '- **与原联合训练的区别：** 原训练batch4重叠窗口、每epoch547次更新，并同时改变编码器且启用dropout；本轮是固定最终编码器、非重叠12秒缓存、每原始帧一次，覆盖与重复采样也不同。因此H0@0→H0短训的提升说明读出可改善，不能单独量化“原3epoch训练不足”的因果贡献。',
              '- **标准化不等于后验概率校准：** 它改变坐标、优化几何和有效权重衰减。本轮附报无权重NLL、Brier和15等宽桶ECE，不把更高F1当成校准改善。',
              '- 受试者train02/03/04、validation05、test06；test已在前次诊断中观察，属于描述性复核，后续新结论需新的受试者验证。','',
              '## 3. H0–H4结果','',f'![验证曲线]({base_link}derived/validation-head-curves.png)','']
    for panel, rows in data['panels'].items():
        lines += [f'### {panel}','',
                  '| 头 | 选中epoch | Val Accuracy | Val Macro-F1 | Val床类F1 | Test Accuracy | Test Macro-F1 | Test床类F1 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm, row in rows.items():
            v,t = row['results']['validation'],row['results']['test']
            lines.append(f'| {arm} | {row["selected_epoch"]} | {pct(v["frame_accuracy"])} | {pct(v["macro_f1"])} | {pct(v["bed_macro_f1"])} | {pct(t["frame_accuracy"])} | {pct(t["macro_f1"])} | {pct(t["bed_macro_f1"])} |')
        lines += ['','| 对照 | Val F1变化（百分点） | Test F1变化（百分点） |','| --- | ---: | ---: |']
        for arm, label in [('H1','延长训练'),('H2','类别权重'),('H3','标准化'),('H4','三者组合')]:
            delta = [100*(rows[arm]['results'][s]['macro_f1']-rows['H0']['results'][s]['macro_f1']) for s in ('validation','test')]
            lines.append(f'| {arm}−H0：{label} | {delta[0]:+.2f} | {delta[1]:+.2f} |')
        lines += ['','| 头 | Val预算末轮F1 | Test预算末轮F1 | Val稳态F1 | Test稳态F1 |','| --- | ---: | ---: | ---: | ---: |']
        for arm,row in rows.items():
            lines.append(f'| {arm} | {pct(row["last_results"]["validation"]["macro_f1"])} | {pct(row["last_results"]["test"]["macro_f1"])} | {pct(row["results"]["validation"]["steady"]["macro_f1"])} | {pct(row["results"]["test"]["steady"]["macro_f1"])} |')
        lines.append('')
    lines += ['主分析组H1选中epoch20，末轮结果与选优相同；probe_replay组H4选中epoch5，继续至20反而下降，不能把“训练更久”解释为“末轮一定更好”。H2验证收益不能稳定迁移到当前测试受试者；主分析组H4的test F1仅比H1高0.14个百分点，组合额外收益很小。H4是组合效应，五组设计不足以估计三因素全部交互，也不能把H4收益拆成三项之和。','',
              '## 4. 概率质量与校准','',
              '| 组 | 头 | Val NLL↓ | Val Brier↓ | Val ECE↓ | Test NLL↓ | Test Brier↓ | Test ECE↓ |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for panel, rows in data['panels'].items():
        for arm,row in rows.items():
            v,t = row['results']['validation'],row['results']['test']
            lines.append(f'| {panel} | {arm} | {v["nll"]:.4f} | {v["brier"]:.4f} | {pct(v["ece_15_equal_width"])} | {t["nll"]:.4f} | {t["brier"]:.4f} | {pct(t["ece_15_equal_width"])} |')
    lines += ['','主分析组标准化H3的F1稍升，但验证ECE比H0更高；组合H4的NLL改善、ECE变差，指标方向并不一致。本轮没有训练温度缩放等校准器，不能声称概率校准是已定位的主因。','',
              '## 5. 全类别读出与短动作边界','',
              '| 动作 | Val原头F1 | Val H0 | Val H1 | Val H2 | Val H3 | Val H4 | Test H4 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    rows = data['panels']['original_optimizer']
    for i in range(14):
        values = [data['reference']['results']['validation']['per_class'][i]['f1']]
        values += [row['results']['validation']['per_class'][i]['f1'] for row in rows.values()]
        values += [rows['H4']['results']['test']['per_class'][i]['f1']]
        lines.append('| '+phrases[str(i)]+' | '+' | '.join(pct(x) for x in values)+' |')
    lines += ['', '床相关类固定为1/2/3/4/5/7/8。短动作定义沿用完整标注run<2秒，仅有validation40帧、test50帧；标签是口令持续到下个口令，并非真实身体执行时长。逐头短动作召回和支持量在机器结果中完整保留，不据此进行可靠短动作排名。','',
              '## 6. 瓶颈判定与下一步','',
              '1. **原分类头不是冻结Stage1特征的可分性上限。** H0@0的验证F1为29.50%，同一最终特征重训3epoch已到43.03%；无需改CNN/BiGRU就能改善读出。原头与持续变化的编码器可能未充分适配，是优先排查方向，但本轮不能分离原训练预算、上下文和采样差异。',
              '2. **先保证冻结读出的训练收敛，再考虑类平衡。** 主分析延长训练在验证/测试均有收益；类权重仅验证改善明显，应监测逐类结果与受试者迁移。标准化可作为优化工具，单独收益较小。',
              '3. **不把分类头改善等同于语言生成改善。** 这些头没有接入Projector/LLM，不能据此声称原生成召回已提升。下一步可固定编码器，以受控读出作感知参考，再检查语言接口的动作覆盖损失。',
              '4. **保持研究边界。** 单种子、少量受试者、重叠来源与双视图使结果不能支持统计显著性结论；不要选择test最好的臂回报为独立泛化验证。后续若比较新策略，需要未使用的受试者。','',
              '## 7. 产物、复现和核验','',
              f'原权重SHA256：`{provenance["checkpoint_sha256"]}`。123份服务器源码与38份缓存SHA核验通过；零更新预测和H4复现门禁通过；服务器相关测试58 passed（1 warning），本地新增及原诊断测试14 passed（1 warning）。v1源码归档遗漏已有测试辅助脚本，失败日志已保留；v2补齐后验证通过，未安装任何依赖。','',
              '服务器输出：`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/radar_text_head_ablation_20261011_v1`。开始11:23:53，完成11:24:10（北京时间），仅小线性层计算。','',
              '取回归档49项文件逐项SHA核验通过。结果归档SHA256：`f00d04b64b137583933909ef24020a52df350011bc01e3017bb41cb87defe0d1`；源码v2归档SHA256：`f5668f7031bb5e18a728723061df3d8e29d614aee4294c2cc023020a1eb8cc08`。独立只读复审核对10个实验臂的数字、曲线和因果边界，无阻塞项。','',
              f'[原始完整指标/训练历史]({base_link}experiment/head-ablation.json) · [全部14类指标CSV]({base_link}derived/per-class-head-metrics.csv) · [不可变输入与源码清单]({base_link}experiment/provenance.json) · [复现门禁记录]({base_link}experiment/verification.json) · [原优化器H4权重]({base_link}experiment/original_optimizer-H4.pt)','',
              '在源码capsule根目录，用现有seer Python运行（output须为新目录）：','',
              '```bash','PYTHONPATH=src python scripts/run_radar_text_head_ablation.py \\',
              '  --diagnostics /home/xibeidaxue/newdisk2/WJ/PARrad/outputs/radar_text_diagnostics_20261011_v1 \\',
              '  --checkpoint /home/xibeidaxue/newdisk2/WJ/PARrad/outputs/radar_text_pilot_20261010_v1/stage1/best.pt \\',
              '  --output /path/to/new/output','```','',
              '权重文件包含head、mean、std和spec。标准化臂推理必须使用保存的训练统计；不能直接将权重套到未标准化特征。完整训练/验证/测试原始预测与本轮结果归档保留，原始源和归档不回写。','']
    report.write_text('\n'.join(lines), encoding='utf-8')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(derived.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    (derived/'manifest.json').write_text(json.dumps(dict(files=hashes,report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()),indent=2))
    print(json.dumps(dict(report=str(report),csv_rows=len(all_rows),verification=verification)))


if __name__ == '__main__':
    main()

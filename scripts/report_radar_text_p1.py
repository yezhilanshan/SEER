"""Render auditable P1 evidence retention and count calibration results."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def pct(x):return 'NA' if x is None else f'{100*x:.2f}%'
def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',required=True);parser.add_argument('--report',required=True)
    args=parser.parse_args();root=Path(args.results);raw=root/'experiment';derived=root/'derived';derived.mkdir(exist_ok=True)
    a=load(raw/'p1-a.json');b=load(raw/'p1-b.json');c=load(raw/'p1-c.json');training=load(raw/'probe-training.json')
    selection=load(raw/'selection.json');review=load(raw/'paired-error-review.json');verify=load(raw/'verification.json')
    audit=load(derived/'local-verification.json');native=load(derived/'server-cache-audit.json');majority=load(derived/'majority-count-diagnostic.json')
    phrases=load('configs/radar-text-actions-parrad.json');link='../outputs/'+root.name+'/'
    def actions(seq):return ' → '.join(f'{i}: {phrases[str(i)]}' for i in seq) or '∅'
    def clean(text):return text.replace('|','\\|').replace('\n',' ')
    train_majority=majority['majority_from_train'];method_names={'B1':'原AlignH1+平滑模板','B3':'原Align+Projector+Qwen','probability':'概率/最短段过滤','hmm':'训练转移先验HMM','predicted-count':'预测数量+段Viterbi','ORACLE-GT-count':'Oracle真数量（诊断）'}
    candidates=('probability','hmm','predicted-count')
    selected=max(candidates,key=lambda n:tuple(c['validation']['full12'][n]['metrics'][k] for k in ('sequence_exact','order_lcs','action_recall')))
    mainrow=c['test']['full12'][selected];q=c['test']['full12']['B3'];base=c['test']['full12']['B1']
    group_stats={};appendix=['# P1-A：全部86个独对窗口的符号与文本复核','',
        '复核范围为GT动作序列、两条路径的生成文本及确定性最小编辑操作；没有逐帧雷达人工复核或可靠时间段对齐。编辑方向GT→预测：插入是多输出，删除是未输出，替换不能直接定位到物理动作时段。多条最优编辑路径时标记歧义，以下分解依赖固定回溯规则。','']
    flat=[]
    for group,rows in review.items():
        other='B3' if group=='B1_only' else 'B1';types=Counter();count_equal=ambiguous=0
        appendix += [f'## {group}（{len(rows)}窗）','']
        for row in rows:
            e=row[other+'_edit'];kind='/'.join(k for k in ('insertions','deletions','substitutions') if e[k]);types[kind]+=1
            count_equal+=len(row[other])==len(row['reference_actions']);ambiguous+=e['ambiguous']
            operations='; '.join(f"{op['op']} GT[{op.get('reference_index','-')}]={op.get('reference','-')} → pred[{op.get('predicted_index','-')}]={op.get('predicted','-')}" for op in e['operations'])
            appendix += [f"### {row['sample_id']} · subject{row['subject']} · {row['sensor']}",'',
                         f"- GT：{actions(row['reference_actions'])}",f"- B1：{actions(row['B1'])}",f"- B3：{actions(row['B3'])}",
                         f"- B1文本：{clean(row['B1_text'])}",f"- B3文本：{clean(row['B3_text'])}",
                         f"- 错误路径{other}：距离{e['distance']}；插入{e['insertions']}，删除{e['deletions']}，替换{e['substitutions']}；编辑歧义={e['ambiguous']}。",f'- 回溯：{operations}','']
            flat.append(dict(group=group,sample_id=row['sample_id'],subject=row['subject'],sensor=row['sensor'],GT=actions(row['reference_actions']),B1=actions(row['B1']),B3=actions(row['B3']),
                        B1_text=row['B1_text'],B3_text=row['B3_text'],error_path=other,edit_distance=e['distance'],insertions=e['insertions'],deletions=e['deletions'],substitutions=e['substitutions'],ambiguous=e['ambiguous'],operations=operations))
        group_stats[group]=dict(windows=len(rows),count_correct=count_equal,ambiguous=ambiguous,operation_patterns=dict(types))
    (derived/'paired-window-review.md').write_text('\n'.join(appendix),encoding='utf-8');write_csv(derived/'paired-window-review.csv',flat)
    (derived/'paired-review-summary.json').write_text(json.dumps(group_stats,indent=2))
    lines=['# P1：动作证据保留与动作数量校准诊断（2026-10-11）','',
       f'**冻结同一个Align Encoder后，仅训练轻量读出并加强分类后处理：validation选出的{method_names[selected]}在严格482个test窗口上达到{mainrow["exact_count"]}/482={pct(mainrow["metrics"]["sequence_exact"])}，高于原B3的{q["exact_count"]}/482={pct(q["metrics"]["sequence_exact"])}，差{100*(mainrow["metrics"]["sequence_exact"]-q["metrics"]["sequence_exact"]):+.2f}个百分点。**','',
       '当前结果支持：原分类流程的动作数量失控与冗余是完整序列差距的重要来源，不能继续把此前B3优势解释为LLM读出独有的序列建模收益。数量校准也没有解决全部问题；Oracle真数量下仍只有34.02%完整序列正确。H→Z未出现本轮探针下的整体退化，Z→P的有序读取变弱，值得后续检查Projector与有序监督；不能据此断言信息被不可逆删除。','',
       '## 1. 协议与不可变性','',
       '- 使用P0相同的Align权重、传感器RD/MD输入和描述窗口：12秒、步长6秒、最短8秒、最多3个GT动作段。训练受试者02/03/04，验证05，测试06；2187/702/484窗。主结果排除各2个不足12秒尾窗，为700val/482test，原协议全集另列。所有val/test窗口ID、GT与雷达输入摘要逐一等同P0；没有用不重叠特征数字替代滑窗评价。',
       '- H、Z、P均重新提取：H为完整BiGRU逐帧256维输出，Z为原Adaptive Temporal Pooling的8×256，P为原Projector的8×896。只训练6个探针及一个独立数量头；Encoder/Projector冻结且状态逐位不变，Qwen没有加载或训练，B3复用经输入核验的原预测。',
       f'- 全部探针权重与后处理候选仅在完整validation选择，于{selection["locked_at"]}锁定后才提取test特征。原始输入{verify["original_inputs_unchanged"]}份、源文件{verify["source_files_verified"]}份前后SHA核对通过。',
       '- 动作来自`command_active_until_next_command`标签，持续时间不能等同真实身体动作执行时长；只知环境字段parrad，不推断Homelab/Hospital。类别短语抽取、重叠窗口、双传感器及只有一个测试受试者均限制解释。subject06已经多轮观测，结果是现有协议诊断，不构成新独立受试者泛化证据。','',
       '## 2. P1-A：数量、类别/顺序与编辑错误','',
       '| 方法 | 序列正确 | 数量正确 | Count MAE | 数量正确条件下Sequence Exact | 平均编辑距离 | 插入/删除/替换 | 歧义窗口 |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,s in a['test']['full12'].items():lines.append(f'| {name} | {s["exact_count"]}/{s["windows"]}={pct(s["metrics"]["sequence_exact"])} | {s["equal_count_windows"]}/{s["windows"]}={pct(s["count_accuracy"])} | {s["count_mae"]:.3f} | {pct(s["sequence_exact_given_correct_count"])} | {s["edit_distance_mean"]:.3f} | {s["insertions"]}/{s["deletions"]}/{s["substitutions"]} | {s["ambiguous_edit_windows"]} |')
    lines += ['',
        'B1数量正确时类别与顺序表现更好，但这种条件比较筛选了不同窗口，不能作为控制同一困难度后的因果结论。B3数量更准确且插入少，同时删除、替换均明显存在。单位代价Levenshtein按GT→预测计算，确定性优先match/sub/delete/insert；替换不代表已找到对应物理时段，插入/删除分解受最优路径歧义影响。动作Recall为窗口内带重复次数的类别多重集覆盖率，不是时间IoU检测Recall。','',
        '| 独对组 | 仅插入 | 仅删除 | 仅替换 | 混合 | 错误路径数量已正确 | 编辑歧义 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for group,g in group_stats.items():
        t=g['operation_patterns'];mixed=sum(v for k,v in t.items() if '/' in k)
        lines.append(f'| {group} ({g["windows"]}) | {t.get("insertions",0)} | {t.get("deletions",0)} | {t.get("substitutions",0)} | {mixed} | {g["count_correct"]} | {g["ambiguous"]} |')
    lines += ['',
        '已逐条核查全部29个B1独对与57个B3独对窗口的符号/文本记录。29窗中B3有17窗只替换、7窗只删除、5窗删除+替换；57窗中B1有29窗只插入、11窗插入+替换，另有13窗只替换、3窗只删除、1窗删除+替换。因此，“Qwen全是漏动作”或“分类全是多动作”都不成立。',
        f'完整记录：[86窗复核表]({link}derived/paired-window-review.md)、[CSV]({link}derived/paired-window-review.csv)。以下为代表性符号证据，未做人工雷达时段判断：','',
        '| sample_id | GT | 错误路径 | 观察 |','| --- | --- | --- | --- |']
    examples=[('e816907bbacd:6000:6134','B1_only','B3','省略末端gets into bed'),('67d5d9de6e50:4200:4334','B1_only','B3','省略末端fall'),
              ('788e53fe3491:2134:2267','B1_only','B3','数量正确但三个类别全替换'),('788e53fe3491:3134:3267','B3_only','B1','正确序列中插入4个冗余动作'),
              ('37fed0eda030:2600:2734','B3_only','B1','数量正确但Roll/Get-out顺序反转'),('67d5d9de6e50:2534:2667','B3_only','B1','数量正确但Sit-bed换成Sit-chair')]
    for sid,group,error,note in examples:
        row=next(v for v in review[group] if v['sample_id'].endswith('/'+sid))
        lines.append(f'| {sid} | {actions(row["reference_actions"])} | {error}: {actions(row[error])} | {note} |')
    lines += ['', '## 3. P1-B：H→Z→P轻量读出','',
        'H保留全部真实帧，按训练集最大134帧补零后展平34304维；Z展平2048维，P展平7168维。每坐标mean/std仅从训练窗计算，std下限1e-5。线性读出为无激活的两层因子化线性映射（rank≤32）；非线性在相同32维瓶颈加入GELU。它们是有限规格探针，不是最优线性可分性上界。H/Z/P输入层参数不同，尤其H有约110万参数而训练仅2187窗；统一训练预算不保证相同优化难度或最佳泛化。',
        '三任务联合监督：14维Presence BCE（训练集posweight）+3维Count CE（训练集逆频率权重）+3×15个Ordered槽位CE（PAD14，包含PAD损失）。最多3动作，非相邻重复保留，如Roll→Sit→Roll；PAD终止输出，不折叠重复。Count是独立任务输出，与Ordered的PAD长度不强制一致。AdamW lr.001/wd.01、20epoch、batch64、clip1、seed17；每个探针按validation Ordered Exact→Presence Macro-F1→Count Accuracy选权重，平分选最早。','',
        '| 特征/读出 | 参数数 | 选epoch | Val Presence F1 | Val Count Acc | Val Ordered Exact | Test Presence F1 | Test Count Acc | Test Ordered Exact |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    probe_csv=[]
    for name,entry in training.items():
        if name=='independent-count':continue
        v=b['validation'][name]['full12'];t=b['test'][name]['full12']
        lines.append(f'| {name} | {entry["parameters"]:,} | {entry["selected_epoch"]} | {pct(v["presence_macro_f1"])} | {pct(v["count_accuracy"])} | {pct(v["ordered"]["metrics"]["sequence_exact"])} | {pct(t["presence_macro_f1"])} | {pct(t["count_accuracy"])} | {t["ordered"]["exact_count"]}/482={pct(t["ordered"]["metrics"]["sequence_exact"])} |')
        for split in ('validation','test'):
            for subset,s in b[split][name].items():probe_csv.append(dict(split=split,subset=subset,probe=name,parameters=entry['parameters'],epoch=entry['selected_epoch'],presence_macro_f1=s['presence_macro_f1'],count_accuracy=s['count_accuracy'],count_mae=s['count_mae'],ordered_exact=s['ordered']['metrics']['sequence_exact'],ordered_count_accuracy=s['ordered']['count_accuracy']))
    write_csv(derived/'probe-results.csv',probe_csv)
    lines += ['',f'![H Z P探针对比]({link}derived/evidence-probes.png)','',
        '- H→Z：Z的Ordered Exact在两种探针下都高于H，没有出现“H好、Z差”的现象，本轮证据不支持优先扩大K来修复池化证据丢失。H并不优秀，但高维探针的优化/泛化差异也可能解释差距，不能据此判定Encoder缺信息。',
        '- Z→P：测试线性Ordered Exact 24.48%→21.99%，非线性25.31%→21.16%；validation也同向。P的Presence F1与Z接近，说明“类别出现”与“有序段序列”应分开诊断。优先检查Projector的有序信息可读性、文本对齐目标与读出优化，不能直接证明信息物理上被删除。',
        '- P探针并未全面优于Qwen，没有得到“H/Z/P都好但Qwen差”的清晰证据。探针与Qwen容量、目标、输出约束不同，当前结果无法单独把问题归因于EOS或LLM；可以后续用固定P进行覆盖/EOS受控实验。','',
        '## 4. P1-C：概率、先验与预测数量校准','',
        '在原AlignH1逐帧logits上比较概率均值平滑0/.3/.5秒×最短段.6/1/2秒×转移权重0/.25/.5/1，共36组（其中weight0为9组概率基线）。转移与开始先验只统计24个训练完整sensor-record，包含帧自转移并Laplace1；HMM用帧Viterbi。最短段过滤丢弃短预测段、合并连续同类，不强制GT数量。概率均值对类别重编号等变，避免class-ID中值的任意顺序问题。',
        '独立数量头使用H按8个时序bin均值后展平→32维GELU→3类，65,667参数；输入数学上与Z相同，独立指训练任务/头，并非另一种信息源。训练与上述预算相同，仅训练countCE，validation Count Accuracy→负MAE选择epoch7。随后段Viterbi在预测K∈1/2/3约束下划分完整窗口，允许非相邻同类重复、相邻同类禁止，最短段按真实时间戳计算；使用每段类别的log概率和转移分数，不输入GT边界或数量。',
        '固定候选及实际validation选择如下。主方法在probability/HMM/predicted-count三个可部署方案中，按同一validation Sequence Exact→LCS→Recall选择；Oracle从不参与选优。','',
        '| 流程 | 概率平滑秒 | 最短段秒 | 转移权重 |','| --- | ---: | ---: | ---: |']
    for name in ('probability','hmm','count'):
        v=selection[name];lines.append(f'| {name} | {v["seconds"]} | {v["minimum"]} | {v["weight"]} |')
    lines += ['',
        '验证最优概率窗口为0秒，因此该行改善来自1秒最短段过滤，不能宣称概率平滑本身带来收益。HMM与数量校准改变整个读出流程，其效果不能拆成单独LLM的因果贡献。','',
        '| split | 流程 | 完整序列正确 | 数量准确率 | Count MAE | 少/多/等数量窗 | Recall | LCS |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    calibration_csv=[]
    for split in ('validation','test'):
        for name,s in c[split]['full12'].items():
            if name=='ORACLE-GT-count':continue
            lines.append(f'| {split} | {method_names[name]} | {s["exact_count"]}/{s["windows"]}={pct(s["metrics"]["sequence_exact"])} | {pct(s["count_accuracy"])} | {s["count_mae"]:.3f} | {s["under_windows"]}/{s["over_windows"]}/{s["equal_count_windows"]} | {pct(s["metrics"]["action_recall"])} | {pct(s["metrics"]["order_lcs"])} |')
        for subset,methods in c[split].items():
            for name,s in methods.items():calibration_csv.append(dict(split=split,subset=subset,method=name,windows=s['windows'],exact_count=s['exact_count'],sequence_exact=s['metrics']['sequence_exact'],count_accuracy=s['count_accuracy'],count_mae=s['count_mae'],count_bias=s['count_bias'],recall=s['metrics']['action_recall'],lcs=s['metrics']['order_lcs'],under=s['under_windows'],over=s['over_windows'],equal=s['equal_count_windows'],edit_distance=s['edit_distance_mean']))
    write_csv(derived/'calibration-results.csv',calibration_csv)
    coverage=[]
    for name,s in c['test']['full12'].items():
        for row in s['per_class']:coverage.append(dict(method=name,action=phrases[str(row['class_id'])],**row))
    for name,s in b['test'].items():
        for row in s['full12']['ordered']['per_class']:coverage.append(dict(method=name,action=phrases[str(row['class_id'])],**row))
    write_csv(derived/'test-per-class-coverage.csv',coverage)
    lines += ['',f'![完整序列与数量误差]({link}derived/count-calibration.png)','',
        '主分类流程由B1的91/482升至131/482（+8.30个百分点），多预测窗从253降至31；对应Recall从65.91%降至57.68%，但仍高于B3的46.06%。数量控制在抑制冗余时也会损失覆盖，不能只看Exact。HMM123/482本身已略高于B3的119/482；这说明结果并不只依赖新增数量头。',
        f'逐类实例覆盖与探针Ordered覆盖见[完整14类CSV]({link}derived/test-per-class-coverage.csv)。','',
        '## 5. 数量分布与事后多数基线','',
        '严格val的GT数量1/2/3分别为4/222/474，test为2/90/390。训练多数类别为3；固定预测3在test的数量准确率80.91%，高于数量头78.63%，六个联合探针的Count Accuracy也均低于此多数基线。因此这些总体计数准确率不能被解读为强实例级计数能力，尤其1动作只有2个test窗。应同时查看每类计数Recall、MAE和完整序列。','',
        '| split | 数量头Count Acc / MAE | train多数Count Acc | 数量头完整序列 | 固定train多数+同解码器完整序列 |','| --- | ---: | ---: | ---: | ---: |']
    for split in ('validation','test'):
        cr=audit['independent_count'][split];s=majority['results'][split]['full12'];p=c[split]['full12']['predicted-count']
        lines.append(f'| {split} | {pct(cr["accuracy"])}/{cr["mae"]:.3f} | {pct(cr["majority_accuracy"])} | {p["exact_count"]}/{p["windows"]}={pct(p["metrics"]["sequence_exact"])} | {s["exact_count"]}/{s["windows"]}={pct(s["metrics"]["sequence_exact"])} |')
    lines += ['',
        '固定多数诊断在查看主结果并发现计数不平衡后补充，明确是事后分析。K只取训练多数，不使用测试GT；使用原已锁定的预测数量解码器参数，没有重新调参、选权重或替换主方法。它用于检查数量头是否超出分布先验，不能包装为预注册正式新测试结果。数量头的test完整序列131/482仍高于固定多数116/482（+3.11个百分点），validation亦同向；这表明总体计数准确率与下游正确序列不等价，也为自适应计数提供了有限支持。','',
        '| split | 真数量=1支持/召回 | 真数量=2支持/召回 | 真数量=3支持/召回 |','| --- | ---: | ---: | ---: |']
    count_csv=[]
    for split in ('validation','test'):
        windows=load(raw/f'windows-{split}.json');pred=load(raw/f'probe-predictions-{split}.json')['independent-count']['count']
        matrix=np.zeros((3,3),int)
        for w,n in zip(windows,pred):
            if w['full_window']:matrix[len(w['actions'])-1,n-1]+=1
        fields=[]
        for i,row in enumerate(matrix):
            fields.append(f'{int(row.sum())}/{pct(row[i]/row.sum())}')
            for j,value in enumerate(row):count_csv.append(dict(split=split,GT_count=i+1,predicted_count=j+1,windows=int(value)))
        lines.append(f'| {split} | '+' | '.join(fields)+' |')
    write_csv(derived/'independent-count-confusion.csv',count_csv)
    lines += ['', '## 6. Oracle Count：仅诊断','',
        '| split | Oracle Sequence Exact | Count MAE | Recall | LCS |','| --- | ---: | ---: | ---: | ---: |']
    for split in ('validation','test'):
        s=c[split]['full12']['ORACLE-GT-count'];lines.append(f'| {split} | {s["exact_count"]}/{s["windows"]}={pct(s["metrics"]["sequence_exact"])} | {s["count_mae"]:.3f} | {pct(s["metrics"]["action_recall"])} | {pct(s["metrics"]["order_lcs"])} |')
    lines += ['',
        'Oracle使用真实动作段数，所以不能计入正式测试性能。它沿用预测数量流程已选好的参数，未按Oracle结果调参；该固定解码器下164/482比预测数量131/482高33窗（6.85个百分点）。这是剩余计数错误的诊断机会量，不是任意分类/语言方法的理论上限；即使提供真数量，318/482仍错误，类别、顺序、标签与解码均仍有改进空间。','',
        '## 7. 原始484窗口补充','',
        '| 流程 | 完整序列正确 | Count MAE |','| --- | ---: | ---: |']
    for name,s in c['test']['original_protocol'].items():lines.append(f'| {method_names[name]} | {s["exact_count"]}/{s["windows"]}={pct(s["metrics"]["sequence_exact"])} | {s["count_mae"]:.3f} |')
    lines += ['', '## 8. 瓶颈定位与下一步','',
        '1. 优先保留预测数量+段Viterbi为更强分类对照，同时保留无需数量头的HMM。当前完整序列优势已不能作为直接生成优于强分类读出的证据；比较针对完整流程，不能单独推断LLM无价值或已量化其因果贡献。',
        '2. 数量头受3动作先验影响明显，下一轮先用按数量分层的验证、计数Macro-F1/Recall及新的受试者检查，再考虑复杂计数网络。不要只追求该测试集的总体Count Accuracy。',
        '3. 暂无优先扩大K的探针证据。针对Z→P的有序读取下降，先在冻结Encoder下检查Projector/有序目标与覆盖监督；同预算多随机种子、规格更充分的线性探针可以区分优化与表示因素。P与Qwen不构成干净因果对照。',
        '4. 仍需真实雷达时段人工复核和新的受试者验证，尤其床相关类别、走向目的地混淆与末端动作；文本模板监督不能证明速度、幅度等细粒度自然语言能力。重叠窗/双传感器不是独立事件，本轮不作独立样本显著性声明。','',
        '## 9. 复核与产物','',
        f'- 本地interview相关测试30项、服务器原生相关测试74项通过；新增6项包含重复动作、编辑歧义、类别置换等变、数量约束与小规模穷举最优解一致性。原始测试日志在[归档]({link}provenance/native-tests.log)。',
        f'- 独立脚本重新计算{audit["summaries_independently_recomputed"]}份指标汇总、三任务预测指标、模型/候选选择、全部86个独对记录及29个归档文件SHA：[本地复核]({link}derived/local-verification.json)。',
        f'- 服务器直接检查H→Z池化、原Projector(Z)、训练均值/标准差、所有探针预测及所有校准路径重建：[缓存复核]({link}derived/server-cache-audit.json)。Z最大误差{native["pool_max_abs_error"]:.3g}，P最大误差{native["projector_max_abs_error"]:.3g}；重建{native["probe_window_predictions_reproduced"]}个探针窗口输出、{native["calibration_windows_reproduced"]}个校准窗口。解码重建复用生产实现，独立指标与穷举测试另列，不把它冒称另一套独立解码器。',
        f'- [固定选择及全部候选]({link}experiment/selection.json)、[探针训练历史]({link}experiment/probe-training.json)、[P1-A]({link}experiment/p1-a.json)、[P1-B]({link}experiment/p1-b.json)、[P1-C]({link}experiment/p1-c.json)、[模型与输入溯源]({link}experiment/provenance.json)。',
        f'- [探针CSV]({link}derived/probe-results.csv)、[后处理CSV]({link}derived/calibration-results.csv)、[计数混淆表]({link}derived/independent-count-confusion.csv)。',
        f'- [最终只读审查]({link}derived/readonly-review.md)通过，无阻断或需修改问题；审查员复跑新增6项测试通过。首次额外缓存校验器因重复读取NPZ整成员导致内存不足，修复为一次加载后在独立v2审计通过，主实验与原归档没有变化：[修复记录]({link}derived/cache-audit-repair.json)。',
        '- 服务器原始H/Z/P三份缓存保留在`/home/xibeidaxue/newdisk2/WJ/PARrad/outputs/radar_text_p1_20261011_v1/features`，本地只取回小型探针、逐窗预测和结果，不下载原始雷达或Qwen。原结果归档SHA256：`a07a6276db40ad6f4fca9f87657a59fbf2d546bf0dcd2926d63fe7308fb23be0`。','']
    report=Path(args.report);report.write_text('\n'.join(lines),encoding='utf-8')
    names=[n for n in training if n!='independent-count'];x=np.arange(len(names));fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,['presence_macro_f1','count_accuracy','ordered'],['Action presence macro F1','Action count accuracy','Ordered action exact']):
        for split,shift,color in [('validation',-.18,'#8baac8'),('test',.18,'#27639a')]:
            ys=[b[split][n]['full12'][key] if key!='ordered' else b[split][n]['full12'][key]['metrics']['sequence_exact'] for n in names]
            ax.bar(x+shift,np.array(ys)*100,.35,label=split,color=color)
        ax.set_xticks(x,names,rotation=45,ha='right');ax.set_title(title);ax.set_ylabel('%');ax.set_ylim(0,100 if key=='count_accuracy' else 65);ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.suptitle('Frozen Align: H / Z / P limited probes; full 12 s windows');fig.tight_layout();fig.savefig(derived/'evidence-probes.png',dpi=170);plt.close(fig)
    methods=['B1','probability','hmm','predicted-count','B3'];fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,['sequence_exact','count_mae'],['Sequence exact (%)','Count MAE']):
        vals=[c['test']['full12'][n]['metrics'][key]*100 if key=='sequence_exact' else c['test']['full12'][n][key] for n in methods]
        bars=ax.bar(methods,vals,color=['#a6afb8','#7da5bd','#4f87a6','#27639a','#ce9853']);ax.set_title(title);ax.tick_params(axis='x',rotation=25);ax.grid(axis='y',alpha=.2)
        for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f'{v:.2f}',ha='center',va='bottom',fontsize=9)
        ax.set_ylim(0,max(vals)*1.2)
    fig.suptitle('Same frozen encoder, full 12 s test windows (n=482); Oracle excluded');fig.tight_layout();fig.savefig(derived/'count-calibration.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(report=str(report),selected=selected,review_windows=len(flat)),ensure_ascii=False))


if __name__=='__main__':main()

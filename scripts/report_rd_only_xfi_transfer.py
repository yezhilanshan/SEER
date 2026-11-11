"""Render audited RD-only transfer results into Chinese report, CSV and figures."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--report',required=True)
    args=p.parse_args();root=Path(args.results);raw=root/'experiment';load=lambda x:json.loads(x.read_text(encoding='utf-8'))
    results=load(raw/'results.json');training=load(raw/'training.json');selection=load(raw/'selection.json');prov=load(raw/'provenance.json')
    assert load(root/'independent-audit.json')['passed'] and load(root/'cache-audit.json')['passed']
    plots=root/'figures';plots.mkdir(exist_ok=True);out=root/'tables';out.mkdir(exist_ok=True)
    phrases={int(k):v for k,v in json.loads(Path('configs/radar-text-actions-parrad.json').read_text()).items()}
    names={'R0':'Stage1 RD-CNN','R1':'Random ResNet18','R2':'X-Fi ResNet18'}
    fig,axes=plt.subplots(1,3,figsize=(19,6))
    for ax,name in zip(axes,names):
        cm=np.array(results['test'][name]['frame']['confusion_matrix']);normalized=cm/np.maximum(1,cm.sum(1,keepdims=True))
        ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues');ax.set_title(names[name]);ax.set_xticks(range(14),range(14));ax.set_yticks(range(14),range(14));ax.set_xlabel('Predicted class');ax.set_ylabel('True class')
        for i in range(14):
            for j in range(14):
                if normalized[i,j]>=.05:ax.text(j,i,f'{normalized[i,j]:.2f}',ha='center',va='center',fontsize=6,color='white' if normalized[i,j]>.5 else 'black')
    fig.tight_layout();fig.savefig(plots/'confusion-matrices.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.6))
    for ax,name in zip(axes,names):
        h=training[name]['history'];ax.plot([r['epoch'] for r in h],[r['validation']['frame']['macro_f1'] for r in h],label='Frame Macro-F1')
        ax.plot([r['epoch'] for r in h],[r['validation']['window']['ordered']['metrics']['sequence_exact'] for r in h],label='Ordered Exact')
        ax.set_title(names[name]);ax.set_xlabel('Epoch');ax.set_ylim(0,1);ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(plots/'validation-curves.png',dpi=180);plt.close(fig)
    rows=[]
    for i in range(14):
        r=dict(class_id=i,action=phrases[i])
        for name in names:
            pc=results['test'][name]['frame']['per_class'][i]
            r.update({name+'_frame_'+k:pc[k] for k in ('support','precision','recall','f1')})
            for method in ('ordered','hmm','predicted_count'):
                r[name+'_'+method+'_recall']=results['test'][name]['sequences'][method]['full12']['per_class'][i]['recall']
        rows.append(r)
    with (out/'per-class.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    pairwise={};pred=load(raw/'predictions-test.json')
    for method in ('ordered','ordered_selected','hmm','predicted_count'):
        a=[r for r in pred['R1'][method] if r['full_window']];b=[r for r in pred['R2'][method] if r['full_window']]
        assert [r['sample_id'] for r in a]==[r['sample_id'] for r in b]
        pairwise[method]={k:[] for k in ('both_correct','R1_only','R2_only','both_wrong')}
        for x,y in zip(a,b):
            first=x['reference_actions']==x['predicted_actions'];second=y['reference_actions']==y['predicted_actions']
            group='both_correct' if first and second else 'R1_only' if first else 'R2_only' if second else 'both_wrong'
            pairwise[method][group].append(dict(sample_id=x['sample_id'],reference=x['reference_actions'],R1=x['predicted_actions'],R2=y['predicted_actions']))
    (root/'paired-R1-R2.json').write_text(json.dumps(pairwise,indent=2),encoding='utf-8')
    pct=lambda x:f'{100*x:.2f}%'
    r1,r2=results['test']['R1'],results['test']['R2'];delta=r2['frame']['macro_f1']-r1['frame']['macro_f1']
    text=[f'# RD-only X-Fi 冻结骨干迁移实验（2026-10-11）','',
        f'三组均已完成20个epoch。X-Fi预训练相对同结构随机ResNet18，test逐帧Macro-F1变化为{100*delta:+.2f}个百分点。'
        f'以下先报告验证集选择，再报告同一严格482测试窗口的结果；本轮没有训练或评估新的Qwen语言路径。','',
        '结论：单通道接入和训练链路可行，但本轮不足以支持直接替换现有雷达编码器。验证集选中原小CNN；X-Fi在测试集部分序列读出有小幅收益，HMM读出却未领先，且验证集逐帧/有序指标低于随机ResNet18。应保留为迁移候选，优先做受控部分解冻或BN适配，再判断是否值得投入语言对齐训练。','',
        '## 协议与可解释范围','',
        '- R0：原Stage1的RD小CNN，冻结；它曾参与RD+MD联合训练，因此只作已有训练参考。R1/R2结构完全相同，分别随机初始化与加载用户提供的X-Fi权重；这对对照用于判断预训练特征在当前预处理下的价值。',
        '- 输入只有RD，每个真实窗口做signed-log/z-score，原始分辨率93×128；MD仅在旧输入一致性摘要中出现。17通道首层权重沿输入通道求和，等价于同一RD复制17份的卷积，不代表复原了X-Fi原17通道语义。',
        '- 空间骨干与BN统计冻结；不含Softmax/55类分类层。新投影→256、两层BiGRU每向128、逐帧14类头、8个顺序池化tokens的32维瓶颈多任务头共同训练。无旧Align/Projector/Qwen参数更新。',
        '- train受试者02/03/04，validation05，test06。原始12秒/6秒滑窗：2187/702/484；主窗口指标限定700/482个完整窗口。逐帧指标保留所有702/484窗口的有效帧，重叠帧会重复计数；另排除标签切换中点±0.3秒报告稳态诊断。',
        '- 每组seed17、20epoch、batch32、AdamW lr0.001/wd0.01、clip1、dropout0.1。特征标准化、类别/存在性/数量权重与转移先验均仅用train；逐帧CE、Presence BCE、Count CE、3槽Ordered CE等权相加。PAD=14，保留非相邻重复动作。',
        '- 主模型：validation逐帧Macro-F1→有序Exact→数量准确率，平局取最早轮；附有序Exact优先模型，避免只以逐帧指标掩盖窗口任务。所有轮次、36组HMM候选和3组预测数量段长候选均记录。全部选择锁定后才提取本轮test特征。',
        '- HMM候选：概率平滑0/0.3/0.5秒，最短段0.6/1/2秒，训练记录转移先验权重0/0.25/0.5/1。数量校准使用模型Count头预测的1/2/3段，未使用真实动作数量。','',
        '## 验证集选择','',
        f'验证集有序Exact在主模型间选择的正式组为 **{selection["formal_arm"]}**；仍完整报告全部组，未根据test改选。','',
        '|组|冻结参数|可训练参数|主轮次|有序优先轮次|Val Frame F1|Val Ordered Exact|','|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        info=training[name];v=results['validation'][name]
        text.append(f'|{name} {names[name]}|{prov["backbone_parameters"][name]:,}|{info["parameters"]:,}|{info["selected"]["frame"]["epoch"]}|{info["selected"]["ordered"]["epoch"]}|{pct(v["frame"]["macro_f1"])}|{pct(v["window"]["ordered"]["metrics"]["sequence_exact"])}|')
    text+=['','## 独立受试者逐帧结果','','|组|Frame Accuracy|Macro Precision|Macro Recall|Macro-F1|稳态F1|床相关F1|','|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        v=results['test'][name];f=v['frame']
        text.append('|'+name+'|'+'|'.join(pct(x) for x in (f['frame_accuracy'],f['macro_precision'],f['macro_recall'],f['macro_f1'],v['frame_steady']['macro_f1'],v['bed_macro_f1']))+'|')
    text+=['','床相关宏平均沿用类别1/2/3/4/5/7/8；完整14类F1/Recall及动作序列覆盖Recall见CSV。','',
        f'![完整14类混淆矩阵]({(plots/"confusion-matrices.png").resolve().as_posix()})','',
        '## 动作序列与数量','','|组|读出方式|Val Exact|Test Exact|正确窗/482|Action Recall|Count MAE|少/多动作窗|','|---|---|---:|---:|---:|---:|---:|---:|']
    labels={'ordered':'主模型有序槽','ordered_selected':'有序优先模型槽','hmm':'主模型HMM模板','predicted_count':'主模型预测数量模板'}
    for name in names:
        for method,label in labels.items():
            v=results['validation'][name]['sequences'][method]['full12'];s=results['test'][name]['sequences'][method]['full12']
            text.append(f'|{name}|{label}|{pct(v["metrics"]["sequence_exact"])}|{pct(s["metrics"]["sequence_exact"])}|{s["exact_count"]}|{pct(s["metrics"]["action_recall"])}|{s["count_mae"]:.3f}|{s["under_windows"]}/{s["over_windows"]}|')
    text+=['','Count MAE按最终输出序列长度计算；独立Count头的准确率/混淆矩阵另存results.json。Ordered槽与Count头不强制一致，仅预测数量模板强制使用预测数量。','',
        '|组|Count头准确率|Count头MAE|Presence Macro-F1|','|---|---:|---:|---:|']
    for name in names:
        v=results['test'][name]['window'];text.append(f'|{name}|{pct(v["count_accuracy"])}|{v["count_mae"]:.3f}|{pct(v["presence_macro_f1"])}|')
    support=load(root/'independent-audit.json')['count_support']['test']
    text +=['',f'test真实1/2/3段支持数为{support}，最大类比例为{pct(max(support)/sum(support))}。数量准确率必须结合这种不平衡解读。','',
        '## 同结构配对比较','','|读出方式|两者都对|仅随机对|仅预训练对|两者都错|','|---|---:|---:|---:|---:|']
    for method,groups in pairwise.items():text.append('|'+labels[method]+'|'+'|'.join(str(len(groups[k])) for k in ('both_correct','R1_only','R2_only','both_wrong'))+'|')
    text+=['','全部配对窗口保存在paired-R1-R2.json。重叠窗口、同动作记录和双传感器存在相关性，不给出独立同分布显著性结论。','',
        '## 结论边界与下一步','',
        '本轮只回答冻结空间特征能否在PARrad RD上被新时序/分类读出利用。随机与预训练的差异包含全部学习到的卷积权重和BN统计；固定BN遇到分辨率、幅度和雷达域差异时可能不适配，负结果不能说明微调后必然无效。','',
        'R0和R1/R2的特征宽度为128/512，输入投影参数不同，且训练来源不同；不能把它们的差异单独解释成网络容量效应。新有序槽是小型监督读出，不是自然语言模型。','',
        '既有P1使用RD+MD与Align Encoder，不能把其Qwen或分类数字当作本轮同编码器对照。是否替换正式Radar-to-Text，需要另行控制RD-only预训练/部分解冻及语言对齐预算，再评估真实Qwen生成；旧Projector不能被假定直接兼容新骨干。','',
        '只有一个随机种子和一个测试受试者。监督仍来自command_active_until_next_command弱标签；床动作过渡、短动作及细粒度事实不能由这些数字完全验证。test06在前轮实验中已多次观察，本轮没有新的未见受试者泛化证据。','',
        '## 复现与核验','',
        '`run_rd_only_xfi_transfer.py --pilot <pilot> --p1 <P1> --xfi <纯张量导出> --output <新目录>`。须用封存源码和source-manifest.json；纯张量导出由package_rd_only_xfi.py安全生成，原.pt不改动。','',
        '- 原始权重SHA256：`'+prov['original_xfi_sha256']+'`。纯张量导出：`'+prov['converted_xfi_sha256']+'`。',
        '- 本地新增7项测试通过；服务器相关25项通过、1项跳过，GPU合成梯度smoke通过。未安装软件或更新共享模型。',
        '- independent-audit.json从逐窗预测和logits独立重算48组序列指标、18份逐帧混淆矩阵，核对验证选择与原P1窗口完全一致。',
        '- cache-audit.json重建27个真实RD窗口的三个骨干特征、384个保存读出输出，核对6份train标准化统计、9份特征缓存SHA；骨干/BN逐张量不变。',
        '- 原始输入和源文件SHA在运行前后核对；本地result-manifest.json封存原始小型产物，约GB级特征缓存保留服务器。原始模型、旧Align/Projector/Qwen与旧结果不修改。','',
        f'![验证训练曲线]({(plots/"validation-curves.png").resolve().as_posix()})','',
        f'[完整产物]({root.resolve().as_posix()}) · [逐类CSV]({(out/"per-class.csv").resolve().as_posix()}) · [独立审计]({(root/"independent-audit.json").resolve().as_posix()})']
    Path(args.report).write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=args.report,test_frame_delta_pp=100*delta,formal_arm=selection['formal_arm'])))

if __name__=='__main__':main()

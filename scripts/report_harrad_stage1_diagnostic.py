"""Build a readable diagnostic report from downloaded metric evidence."""
import json
from pathlib import Path
import sys

import numpy as np


def report(evidence,destination):
    read=lambda name:json.loads((evidence/name).read_text())
    rows=read('record-seed-scene-metrics.json');summaries=read('scene-bootstrap-summary.json')
    lines=['# HARrad Stage 1 Diagnostic 结果','',
        '实验启动于2026-11-05；随机对照与最终复核于2026-11-06完成（北京时间）。','',
        '本轮仅诊断已有 Stage 1 表示与训练预算，保留原 No-Go 判定。没有启动 Reset Policy 训练。', '',
        '25 条 fit 录制用于训练和场景数值阈值，6 条原校准录制不参与本轮验收，4 条 subject7 录制用于诊断。'
        '冻结 Radar Encoder、EventFormer 和 Semantic Reader。原始三个预测器 seed 为42/43/44。'
        'EventFormer 曾使用 subject7 选取检查点，本轮属于开发诊断。', '',
        '以下均先按完整录制等权，再按 seed 等权。差值为 D−GRU，负数表示 D 更好。'
        '置信区间按完整录制配对 Bootstrap，枚举四条录制全部256种重采样，三 seed 共用每次重采样索引。'
        '百分位使用精确离散分布的逆经验 CDF。它不包含跨受试者不确定性，也不将三个 seed 当作三个受试者。', '',
        '## A：分场景表现','',
        '长稳定组要求 GT 实例至少3秒、观测与目标均离两侧边界至少1秒、局部特征变化较小。'
        '快/慢转换由冻结64维逐帧特征的方向投影20%→80%耗时划分（≤3帧/≥6帧）；幅度与单调性另有限制。'
        '这些场景标签可以读取区间前后信息用于离线分组，但不会输入预测模型。短暂扰动是高幅短脉冲后回归的特征代理。'
        '快/慢/扰动没有独立视频或人工标签，不能直接解释为真实转换速度或传感器噪声。分组可重叠。', '',
        '| 场景 | 秒 | 样本/录制 | GRU MSE | D MSE | GRU NLL | D NLL | ΔMSE 95% CI | ΔNLL 95% CI |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    names=dict(all='全部',long_stable='长动作内部稳定候选',fast_proxy='快速变化候选',slow_proxy='缓慢转换候选',
               repeated='相邻同类事件',boundary='边界附近',transient_proxy='短暂扰动候选')
    def ci(value):
        if value is None:return '支持不足'
        return '['+', '.join(f'{x:+.5f}' for x in value)+']'
    for r in summaries:
        if 'mse' not in r:continue
        a,b=r['mse'],r['nll']
        lines.append(f"| {names[r['scene']]} | {r['horizon']:.1f} | {r['sample_support']}/{len(r['files'])} | {a['gru']:.5f} | {a['event']:.5f} | {b['gru']:.5f} | {b['event']:.5f} | {ci(a['paired']['ci95'])} | {ci(b['paired']['ci95'])} |")
    lines+=['',f"![场景配对差值与录制级置信区间]({(evidence/'scene-comparison.png').resolve().as_posix()})",'']
    if (evidence/'scene-support.json').exists():
        support=read('scene-support.json')
        lines+=['支持这些窗口的实例/候选点数如下。同一实例或边界只在每条录制内计一次，仍不能作为独立 Bootstrap 单元。','',
                '| 场景 | 0.2秒支持单位 | 0.6秒支持单位 | 1.0秒支持单位 |','|---|---:|---:|---:|']
        for scene in ('long_stable','fast_proxy','slow_proxy','repeated','boundary','transient_proxy'):
            counts=[sum(r['support_units'] for r in support if r['scene']==scene and r['horizon']==h) for h in (.2,.6,1.)]
            lines.append('| '+names[scene]+' | '+' | '.join(str(n) for n in counts)+' |')
    metas=[read(f'subject7_0{i}-scene-candidates.json') for i in range(1,5)]
    gaps=[r['gap_frames']/15 for m in metas for r in m['repeated_instances']]
    lines+=['',f"快转换候选边界 {sum(len(m['fast']) for m in metas)} 个，慢转换候选边界 {sum(len(m['slow']) for m in metas)} 个，"
        f"相邻同类实例 {len(gaps)} 对，短暂扰动候选 {sum(len(m['transients']) for m in metas)} 个。"
        f"同类实例间未标注间隔为 {min(gaps):.2f}–{max(gaps):.2f} 秒，中位数 {np.median(gaps):.2f} 秒。"
        '这些是候选边界/实例数，不能当作独立统计样本。具体坐标、间隔和逐录制支持保存在 scene-candidates 与指标 JSON。', '',
        '缓慢转换组在0.6秒的平均 MSE 略好，但区间跨0，逐条留出后也会翻转；短期和1秒 MSE 更差。'
        '同类重复组未显示优势。较一致的组内收益来自长动作内部稳定候选。短暂扰动只有两条录制24个起点，结果只作探索。'
        '没有做多重比较校正，任何场景结果都不能用于宣布原 Stage 1 通过。', '',
        '## B：Seed 与录制来源','',
        '| Seed | 秒 | GRU MSE | D MSE | ΔMSE | 录制 Bootstrap 95% CI | ΔNLL |',
        '|---:|---:|---:|---:|---:|---|---:|']
    for seed in (42,43,44):
        for h in (.2,.6,1.):
            subset=[r for r in rows if r['scene']=='all' and r['seed']==seed and r['horizon']==h]
            g=np.mean([r['mse'] for r in subset if r['kind']=='gru']);d=np.mean([r['mse'] for r in subset if r['kind']=='event'])
            detail=next(r for r in summaries if r['scene']=='all' and r['horizon']==h)['by_seed'][str(seed)]
            lines.append(f"| {seed} | {h:.1f} | {g:.5f} | {d:.5f} | {d-g:+.5f} | {ci(detail['mse']['ci95'])} | {detail['nll']['mean']:+.5f} |")
    lines+=['','0.6秒逐条录制的配对差值：','',
            '| 录制 | Seed42 ΔMSE | Seed43 ΔMSE | Seed44 ΔMSE | Seed平均 ΔMSE |','|---|---:|---:|---:|---:|']
    for file in sorted({r['file'] for r in rows}):
        changes=[]
        for seed in (42,43,44):
            subset=[r for r in rows if r['scene']=='all' and r['seed']==seed and r['file']==file and r['horizon']==.6]
            changes.append(next(r['mse'] for r in subset if r['kind']=='event')-next(r['mse'] for r in subset if r['kind']=='gru'))
        lines.append('| '+file+' | '+' | '.join(f'{x:+.5f}' for x in [*changes,np.mean(changes)])+' |')
    mid=next(r for r in summaries if r['scene']=='all' and r['horizon']==.6)
    leave=mid['mse']['paired']['leave_one_out']
    lines+=['',f"0.6秒三 seed 平均 ΔMSE={mid['mse']['paired']['mean']:+.5f}，CI={ci(mid['mse']['paired']['ci95'])}。"
        f"逐条留出范围 [{min(leave):+.5f}, {max(leave):+.5f}]，均未翻转。"
        '因此收益并非仅由一条验证录制造成，但主要来自 seed43；seed42 的平均 MSE 与 NLL 都更差。'
        '验证录制全部属于同一受试者，不能排除收益是 subject7 特有现象。', '',
        '### NLL 与方差','',
        '对角 Gaussian NLL = 0.5 log(2π) + 0.5 log(σ²) + 0.5 error²/σ²。'
        '下表分开给出两个可变项，避免把放大方差导致的误差项下降当作均值预测能力提升。', '',
        '| 模型 | 秒 | MSE | 平均 σ² | 0.5 log σ² | 0.5 error²/σ² | 平均 error²/σ² | 90%覆盖率 | 误差/方差相关 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for kind in ('gru','event'):
        for h in (.2,.6,1.):
            subset=[r for r in rows if r['scene']=='all' and r['kind']==kind and r['horizon']==h]
            average=lambda key:float(np.mean([r[key] for r in subset]))
            lines.append(f"| {'GRU' if kind=='gru' else 'D'} | {h:.1f} | {average('mse'):.5f} | {average('variance'):.5f} | {average('nll_logvar_term'):.5f} | {average('nll_error_term'):.5f} | {average('standardized_error'):.5f} | {average('coverage90'):.2%} | {average('error_variance_correlation'):.4f} |")
    lines+=['','本表 standardized_error 明确采用均方标准化残差，原 Stage 1 核心指标的同名字段采用平均绝对标准化残差，两者不能直接比较。'
            '相关系数先计算每个窗口的坐标平均误差与坐标平均方差的 Pearson 相关，再按录制/seed等权。'
            '逐录制、逐 seed 明细保存在 record-seed-scene-metrics.json。'
            '覆盖率是256维各坐标的边际覆盖，不能解释为整向量联合覆盖。', '', '## C：正确与随机重置','']
    if (evidence/'complete.json').exists():
        checks=read('independent-verification.json')
        assert checks['passed']
        schedules=read('random-reset-schedules.json')
        lines+=['固定所有雷达观测、GT 配对网格、目标和预测起点。仅打乱每条录制的更新段长度顺序，保持更新段数与长度多重集。'
                '年龄历史随随机重置重新开始。此对照隔离 E 的重置结构，仍保留原 GT 观测网格，不能代表完全无 GT 的在线系统。'
                '使用一份固定随机排列，三个 seed 是预测器训练重复，不能替代多份重置排列。', '',
                f"35条录制中有效改变 {sum(r['effectively_changed'] for r in schedules)} 条，"
                f"内部重置重合 {sum(r['internal_reset_overlap'] for r in schedules)}/{sum(r['internal_reset_count'] for r in schedules)}。", '',
                'swap：原 D 权重直接换用随机 E；retrained：随机 E 上从相同初值、相同样本表、相同1000次更新重训。'
                'swap 同时受到输入分布改变影响，以 retrained 作为主要对照。差值为正确−随机，负数表示正确边界更好。', '',
                '| 对照 | 秒 | 正确 MSE | 随机 MSE | ΔMSE 95% CI | 正确 NLL | 随机 NLL | ΔNLL 95% CI |',
                '|---|---:|---:|---:|---|---:|---:|---|']
        for r in read('random-control-bootstrap-summary.json'):
            a,b=r['mse'],r['nll']
            lines.append(f"| {r['condition']} | {r['horizon']:.1f} | {a['correct']:.5f} | {a['random']:.5f} | {ci(a['correct_minus_random']['ci95'])} | {b['correct']:.5f} | {b['random']:.5f} | {ci(b['correct_minus_random']['ci95'])} |")
        lines+=['','随机状态重训的逐 seed 配对结果：','',
                '| Seed | 秒 | 正确−随机 ΔMSE | 95% CI | 正确−随机 ΔNLL | 95% CI |',
                '|---:|---:|---:|---|---:|---|']
        for seed in (42,43,44):
            for r in read('random-control-bootstrap-summary.json'):
                if r['condition']!='retrained':continue
                a,b=(r[k]['by_seed'][str(seed)] for k in ('mse','nll'))
                lines.append(f"| {seed} | {r['horizon']:.1f} | {a['mean']:+.5f} | {ci(a['ci95'])} | {b['mean']:+.5f} | {ci(b['ci95'])} |")
        lines+=['','三个 seed 平均后，正确重置在三个 horizon 的 MSE/NLL 均优于随机重置重训，配对录制区间均未跨0。'
                '逐 seed 的 MSE 平均差值也全部支持正确重置；但 seed43 的0.2秒 NLL 区间跨0，不能说所有逐 seed 指标都已确证。'
                '这为 GT 对齐的重置结构有贡献提供了证据。该干预同时改变重置位置和状态年龄，无法进一步拆解语义对齐、跨事件混合与年龄编码各自的作用。'
                '它也不能证明模型已经学会自主事件切换。', '',
                '结合 A/B，当前尚不支持“优势主要隐藏在慢转换和同类重复事件”的假设。更一致的信号来自稳定动作内部与正确分段。'
                'D 在0.2秒和1.0秒仍未超过 GRU，Stage 1 保持 No-Go。下一轮宜先增加受试者和随机边界排列重复，并加入不重置对照，'
                '检验正确分段的收益能否超过普通连续时序记忆，再决定是否开展专项预测训练。']
        lines+=['',f"独立复算通过 {checks['checks']} 组检查，覆盖保存预测的 NLL/MSE/方差/覆盖率、录制 Bootstrap、"
                '相同目标、权重重载回放、随机缓存哈希与长度多重集。源文件与预测 SHA 保存在 independent-verification.json。'
                '三个随机重训模型均完成1000次更新，六个对照任务均完成四条验证录制。原模型与原预测保留。']
    else:lines+=['随机状态缓存/重训仍在运行，本节尚无验收结论。']
    lines+=['','## 证据与范围','',
            '本地证据：`'+str(evidence.resolve())+'`。服务器原始数据位于 '
            '`/home/xbdx/SEER/experiments/idea3_harrad_v261_20261104_v2/stage1-diagnostic-v1`。', '',
            '全部 CI 都是在固定三个训练 seed、固定一名验证受试者、固定一份随机边界排列的条件下计算。'
            '重叠场景、特征代理和有限录制使这些结果适合定位下一轮实验，不构成 Stage 1 验收或跨受试者优势证明。']
    destination.write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':report(Path(sys.argv[1]),Path(sys.argv[2]))

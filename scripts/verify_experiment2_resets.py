"""Independent saved-output readback and recording-cluster comparisons."""
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np


def read(p):return json.loads(p.read_text())
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def paired(values):
    # Input = recording x training seed; all seeds remain paired within cluster.
    values=np.asarray(values);n=len(values);a=values.mean(1)
    draws=np.array([a[list(ids)].mean() for ids in itertools.product(range(n),repeat=n)])
    return dict(mean=float(values.mean()),ci95=np.quantile(draws,[.025,.975],method='inverted_cdf').tolist(),
                leave_one_out=[float(np.delete(values,i,axis=0).mean()) for i in range(n)],
                by_seed_mean=values.mean(0).tolist())


def prediction_metrics(mu,logvar,target):
    error=np.square(mu.astype(float)-target);variance=np.exp(logvar.astype(float))
    nll=(np.log(2*np.pi)+logvar+error/variance)/2
    assert np.isfinite(error).all() and np.isfinite(nll).all() and np.isfinite(variance).all()
    return [dict(mse=float(error[:,h].mean()),nll=float(nll[:,h].mean()),variance=float(variance[:,h].mean()),
                 coverage90=float((error[:,h]<=1.64485363**2*variance[:,h]).mean())) for h in range(3)]


def verify(root,out):
    source=root/'stage1-event-prediction-v1';schedules=read(out/'schedules.json')
    manifest=read(source/'cache-manifest.json');byfile={r['file']:r for r in manifest}
    features={r['file']:r for r in read(root/'shared-front-event-warmup-v1/feature-manifest.json')}
    checks=0
    assert len(schedules)==35
    assert [r['file'] for r in schedules]==[r['file'] for r in manifest]
    for row in schedules:
        r=byfile[row['file']]
        with np.load(Path(r['path'])/'indices.npz') as x:
            n=len(x['cutoffs']);ticks=np.r_[0,x['cutoffs'][1:]-2,r['frames']]
            starts=np.flatnonzero(x['beginnings']==np.arange(n));lengths=np.diff(np.r_[starts,n])
        actual=np.diff(ticks[np.r_[starts,n]])
        assert row['true_starts']==starts.tolist()
        assert row['true_update_lengths']==lengths.tolist() and row['true_frame_intervals']==actual.tolist()
        edges={v for p in features[r['file']]['instances'] for v in (p['start'],p['stop'])}
        assert len(row['candidates'])==10 and len({tuple(c['starts']) for c in row['candidates']})==10
        for c in row['candidates']:
            new=np.array(c['starts']);assert new[0]==0 and len(new)==len(starts)
            intervals=np.diff(ticks[np.r_[new,n]])
            np.testing.assert_array_equal(np.sort(np.diff(np.r_[new,n])),np.sort(lengths))
            assert intervals.min()==actual.min()
            assert np.max(np.abs(np.sort(intervals)-np.sort(actual)))<=2
            assert not set(ticks[new[1:]])&edges
            np.testing.assert_array_equal(intervals,c['frame_intervals']);checks+=1
    caches=read(out/'cache-manifest.json');assert len(caches)==35*12
    expected_arms=['gt','no_reset']+[f'random{i:02d}' for i in range(10)]
    assert {(r['arm'],r['file']) for r in caches}==set(itertools.product(expected_arms,byfile))
    for r in caches:
        assert digest(Path(r['path']))==r['sha256'];checks+=1
    jobs=read(out/'jobs.json');arms=['gt','no_reset']+[f'random{i:02d}' for i in range(10)]
    assert len(jobs)==36 and {(j['arm'],j['seed']) for j in jobs}==set(itertools.product(arms,(42,43,44)))
    files=[f'subject7_0{i}.hdf5' for i in range(1,5)];values={}
    for j in jobs:
        assert j['steps']==1000 and len(j['records'])==4 and {r['file'] for r in j['records']}==set(files)
        folder=out/f"{j['arm']}-seed{j['seed']}";replay=read(folder/'replay.json')
        assert replay['max_abs_difference']==0 and digest(folder/'final.pt')==replay['sha256']
        for r in j['records']:
            with np.load(folder/(Path(r['file']).stem+'.npz')) as x,np.load(source/f"event-seed{j['seed']}"/(Path(r['file']).stem+'.npz')) as original:
                assert np.array_equal(x['target'],original['target'])
                independent=prediction_metrics(x['mean'],x['logvar'],x['target'])
                for h in range(3):
                    result=independent[h]
                    for k,v in result.items():
                        expected=r['horizons'][h]['coverage']['0.9'] if k=='coverage90' else r['horizons'][h][k]
                        np.testing.assert_allclose(v,expected,rtol=1e-5,atol=1e-6)
                    values[(j['arm'],j['seed'],r['file'],h)]=result;checks+=1
    summary=[]
    for h in range(3):
        row=dict(horizon=(.2,.6,1.)[h],metrics={})
        for metric in ('mse','nll','variance','coverage90'):
            matrices={arm:np.array([[values[(arm,s,f,h)][metric] for s in (42,43,44)] for f in files]) for arm in arms}
            random=np.stack([matrices[a] for a in arms[2:]]);ensemble=random.mean(0)
            spread=random.mean((1,2));gt=matrices['gt'];no=matrices['no_reset']
            row['metrics'][metric]=dict(gt=float(gt.mean()),no_reset=float(no.mean()),random_mean=float(ensemble.mean()),
                random_replicates=spread.tolist(),random_sd=float(spread.std(ddof=1)),random_min=float(spread.min()),random_max=float(spread.max()),
                fraction_random_above_gt=float(np.mean(spread>gt.mean())),
                fraction_random_above_no_reset=float(np.mean(spread>no.mean())),
                gt_minus_random=paired(gt-ensemble),gt_minus_no_reset=paired(gt-no),random_minus_no_reset=paired(ensemble-no))
        summary.append(row)
    result=dict(passed=True,checks=checks,jobs=36,updates=36000,random_replicates=10,training_seeds=[42,43,44],recordings=files,
                subjects=1,original_no_go_unchanged=True,reset_policy_started=False)
    (out/'independent-verification.json').write_text(json.dumps(result,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# 实验二：重复随机边界与 No Reset','',
           'GT、No Reset和10个随机组分别匹配训练三个seed，各1000次更新；冻结来源、观测网格、未来标签与原Stage1相同。'
           '随机排列保留更新间隔多重集、重置次数及实际帧最小间隔；排序后的实际帧间隔最多相差2帧，内部位置不与GT起止点精确重合。'
           'No Reset只在录制开始初始化，年龄按真实已观测帧计算并依原设计封顶。','',
           '| 秒 | GT MSE | Random MSE均值±排列SD | No Reset MSE | GT NLL | Random NLL | No Reset NLL |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        a,b=r['metrics']['mse'],r['metrics']['nll']
        lines.append(f"|{r['horizon']}|{a['gt']:.5f}|{a['random_mean']:.5f} ± {a['random_sd']:.5f}|{a['no_reset']:.5f}|{b['gt']:.5f}|{b['random_mean']:.5f}|{b['no_reset']:.5f}|")
    lines+=['','差值为前者−后者；负数表示前者更好。录制级配对Bootstrap枚举4条录制的256种抽样，seed与排列先在录制内配对平均。', '',
            '| 秒 | 指标 | GT−Random 95% CI | GT−No Reset 95% CI | Random−No Reset 95% CI |','|---|---|---|---|---|']
    for r in summary:
        for k in ('mse','nll'):
            a=r['metrics'][k]
            lines.append('|'+str(r['horizon'])+'|'+k+'|'+'|'.join(str([round(v,6) for v in a[p]['ci95']]) for p in ('gt_minus_random','gt_minus_no_reset','random_minus_no_reset'))+'|')
    lines+=['','| 秒 | GT 方差/90%覆盖率 | Random 方差/90%覆盖率 | No Reset 方差/90%覆盖率 |','|---|---|---|---|']
    for r in summary:
        a,b=r['metrics']['variance'],r['metrics']['coverage90']
        lines.append('|'+str(r['horizon'])+'|'+'|'.join(f"{a[k]:.5f} / {b[k]:.2%}" for k in ('gt','random_mean','no_reset'))+'|')
    lines+=['',f'独立复核通过{checks}组检查，包括350个排列约束、420个状态缓存SHA、36个权重重载以及保存预测的MSE/NLL/方差与相同目标。', '',
            'summary.json保留每份排列得分、离散程度、排序比例、逐seed差值和逐录制留出。排列排序比例仅作描述。'
            '10份排列是每条录制受约束分布下互不重复的抽样，不能当作完全独立的受试者。实际帧间隔分布仅近似匹配（逐项最多2帧），不是精确相同。'
            '区间仅针对subject7四条录制，并条件于固定seed和10份受约束排列，不证明跨受试者泛化。仍保留GT观测网格。'
            '冻结EventFormer曾按GT分段预热并在subject7选取权重，NoReset可能超出其预热状态长度分布。'
            '正确重置同时改变状态历史和年龄历史，不能单独归因于语义推理。原Stage1 NoGo不改变，也不启动Reset Policy。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':verify(Path(sys.argv[1]),Path(sys.argv[2]))

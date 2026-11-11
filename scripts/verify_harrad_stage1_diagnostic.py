"""Read saved predictions independently of the training/diagnostic functions."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np


def load(path):
    return json.loads(path.read_text())


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def check_pairing(summary, values):
    values=np.asarray(values,dtype=float)
    count=len(values)
    assert summary['records']==count
    np.testing.assert_allclose(summary['mean'],values.mean(),atol=1e-10)
    if count<2:
        assert summary['ci95'] is None and summary['leave_one_out'] is None
        return
    per_record=values.mean(axis=1)
    draws=sorted(sum(per_record[i] for i in ids)/count for ids in itertools.product(range(count),repeat=count))
    bounds=[draws[max(0,int(np.ceil(p*len(draws)))-1)] for p in (.025,.975)]
    np.testing.assert_allclose(summary['ci95'],bounds,atol=1e-10)
    leave=[np.concatenate((values[:i],values[i+1:])).mean() for i in range(count)]
    np.testing.assert_allclose(summary['leave_one_out'],leave,atol=1e-10)


def check_cohort(controls,expected):
    assert len(controls)==6
    assert {(r['condition'],r['seed']) for r in controls}==set(itertools.product(('swap','retrained'),(42,43,44)))
    for job in controls:
        assert len(job['records'])==len(expected)
        assert {r['file'] for r in job['records']}==expected
    return len(expected)


def verify(root,output):
    source=root/'stage1-event-prediction-v1'
    checks=0;locks={}
    expected={f'subject7_0{i}.hdf5' for i in range(1,5)}
    features=load(root/'shared-front-event-warmup-v1/feature-manifest.json')
    assert {r['file'] for r in features if r['role']=='validation'}==expected
    config=load(root/'shared-front-event-warmup-v1/config.json')
    for name,info in load(root/'shared-front-event-warmup-v1/input-lock.json')['qwen_files'].items():
        path=Path(config['qwen_path'])/name
        actual=digest(path);assert actual==info['sha256'];locks[str(path)]=actual
    for row in features:
        path=Path(row['artifact_dir'])/'frame_features.npy'
        actual=digest(path)
        assert actual==row['artifacts']['frame_features.npy']['sha256']
        locks[str(path)]=actual
    rows=load(output/'record-seed-scene-metrics.json');supports=[]
    assert {r['file'] for r in rows}==expected
    for kind,seed in itertools.product(('gru','event'),(42,43,44)):
        cohort=[r['file'] for r in rows if r['scene']=='all' and r['horizon']==.2 and r['kind']==kind and r['seed']==seed]
        assert len(cohort)==4 and set(cohort)==expected
    for record in sorted({r['file'] for r in rows}):
        with np.load(output/(Path(record).stem+'-scenes.npz')) as data:
            masks=data['masks'];cutoffs=data['cutoffs']
        names=['all','long_stable','fast_proxy','slow_proxy','repeated','boundary','transient_proxy']
        meta=load(output/(Path(record).stem+'-scene-candidates.json'))
        instances=next(r['instances'] for r in features if r['file']==record)
        for h,offset in enumerate((3,9,15)):
            for g,name in enumerate(names[1:],1):
                chosen=cutoffs[masks[:,h,g]]
                if name=='long_stable':
                    count=sum(any(r['start']<=c-1 and c+offset<=r['stop'] for c in chosen) for r in instances)
                    unit='GT instances'
                else:
                    key,width={'fast_proxy':('fast',8),'slow_proxy':('slow',8),'repeated':('repeated_starts',15),
                               'boundary':('internal_edges',8),'transient_proxy':('transients',3)}[name]
                    count=sum(bool(np.any(np.abs(chosen-point)<=width)) for point in meta[key])
                    unit='candidate points'
                supports.append(dict(file=record,scene=name,horizon=(.2,.6,1.)[h],samples=len(chosen),support_units=count,unit=unit))
        for kind,seed in itertools.product(('gru','event'),(42,43,44)):
            path=source/f'{kind}-seed{seed}'/(Path(record).stem+'.npz')
            locks[str(path)]=digest(path)
            with np.load(path) as data:
                assert np.array_equal(cutoffs,data['observed_cutoff'])
                error=np.square(data['mean'].astype('float64')-data['target'])
                lv=data['logvar'].astype('float64');variance=np.exp(lv)
                nll=(np.log(2*np.pi)+lv+error/variance)/2
                for row in [r for r in rows if r['file']==record and r['kind']==kind and r['seed']==seed]:
                    h=[.2,.6,1.].index(row['horizon']);mask=masks[:,h,names.index(row['scene'])]
                    assert row['n']==int(mask.sum())
                    for key,value in dict(mse=error[mask,h].mean(),nll=nll[mask,h].mean(),
                            variance=variance[mask,h].mean(),standardized_error=(error[mask,h]/variance[mask,h]).mean(),
                            coverage90=(error[mask,h]<=2.705543454*variance[mask,h]).mean()).items():
                        np.testing.assert_allclose(row[key],value,rtol=1e-6,atol=1e-7)
                    checks+=1
    for summary in load(output/'scene-bootstrap-summary.json'):
        if not summary.get('sample_support'):continue
        files=summary['files']
        for key in ('mse','nll'):
            values=[]
            for file in files:
                chosen=[r for r in rows if r['file']==file and r['scene']==summary['scene'] and r['horizon']==summary['horizon']]
                values.append([next(r[key] for r in chosen if r['kind']=='event' and r['seed']==seed)-
                               next(r[key] for r in chosen if r['kind']=='gru' and r['seed']==seed) for seed in (42,43,44)])
            values=np.array(values)
            check_pairing(summary[key]['paired'],values)
            for i,seed in enumerate((42,43,44)):
                check_pairing(summary['by_seed'][str(seed)][key],values[:,i:i+1])
            checks+=1
    controls=load(output/'random-control-results.json')
    cluster_count=check_cohort(controls,expected)
    recomputed={}
    for job in controls:
        folder=output/f"{job['condition']}-seed{job['seed']}"
        replay=load(folder/'checkpoint-replay.json');assert replay['max_abs_difference']==0
        for row in job['records']:
            with np.load(folder/(Path(row['file']).stem+'.npz')) as data, np.load(source/f"event-seed{job['seed']}"/(Path(row['file']).stem+'.npz')) as original:
                assert np.array_equal(data['target'],original['target'])
                error=np.square(data['mean'].astype('float64')-data['target'])
                nll=(np.log(2*np.pi)+data['logvar']+error*np.exp(-data['logvar'].astype('float64')))/2
                for h,metric in enumerate(row['horizons']):
                    recomputed[(job['condition'],job['seed'],row['file'],h)]=dict(mse=float(error[:,h].mean()),nll=float(nll[:,h].mean()))
                    for key,value in dict(mse=error[:,h].mean(),nll=nll[:,h].mean()).items():
                        np.testing.assert_allclose(metric[key],value,rtol=1e-5,atol=1e-6)
                    checks+=1
    control_summaries=load(output/'random-control-bootstrap-summary.json')
    assert len(control_summaries)==6
    for summary in control_summaries:
        h=[.2,.6,1.].index(summary['horizon']);condition=summary['condition']
        assert set(summary['files'])==expected
        for key in ('mse','nll'):
            a=[];b=[]
            for file in summary['files']:
                a.append([next(r[key] for r in rows if r['scene']=='all' and r['kind']=='event' and r['seed']==s and r['file']==file and r['horizon']==summary['horizon']) for s in (42,43,44)])
                b.append([recomputed[(condition,s,file,h)][key] for s in (42,43,44)])
            delta=np.array(a)-b;item=summary[key]
            np.testing.assert_allclose(item['correct'],np.mean(a),rtol=1e-5,atol=1e-6)
            np.testing.assert_allclose(item['random'],np.mean(b),rtol=1e-5,atol=1e-6)
            # Original metric code uses float32 intermediates; tolerate its roundoff only.
            check_pairing_rounded=lambda x,v:np.testing.assert_allclose(
                np.array([x['mean'],*x['ci95'],*x['leave_one_out']]),
                pairing_values(v),rtol=1e-4,atol=2e-6)
            check_pairing_rounded(item['correct_minus_random'],delta)
            for i,s in enumerate((42,43,44)):check_pairing_rounded(item['by_seed'][str(s)],delta[:,i:i+1])
            checks+=1
    schedules=load(output/'random-reset-schedules.json')
    assert len(schedules)==35
    for row in schedules:
        assert row['lengths_preserved']
        cache=next(r for r in load(source/'cache-manifest.json') if r['file']==row['file'])
        with np.load(Path(cache['path'])/'indices.npz') as data:
            length=len(data['cutoffs'])
            true=np.flatnonzero(data['beginnings']==np.arange(length))
            assert row['true_starts']==true.tolist()
            a=np.diff(np.r_[true,length]);b=np.diff(np.r_[row['random_starts'],length])
            np.testing.assert_array_equal(np.sort(a),np.sort(b))
        path=output/'random-cache'/Path(row['file']).stem/'events.npy'
        assert digest(path)==row['sha256']
        original=np.load(Path(cache['path'])/'events.npy',mmap_mode='r')
        random=np.load(path,mmap_mode='r')
        assert original.shape==random.shape
        first_difference=min(next(iter(row['true_starts'][1:]),length),next(iter(row['random_starts'][1:]),length))
        # Causal histories and resets are identical until the first divergent reset.
        np.testing.assert_array_equal(original[:first_difference],random[:first_difference])
        checks+=1
    (output/'scene-support.json').write_text(json.dumps(supports,indent=2))
    write=dict(passed=True,checks=checks,controls=len(controls),recording_clusters=cluster_count,
               subjects=len({f.split('_')[0] for f in expected}),reset_permutations=1,source_sha256=locks,
               checks_description='feature source SHA; saved prediction MSE/NLL/variance/coverage; paired record Bootstrap; target equality; checkpoint replay; reset lengths and cache SHA')
    (output/'independent-verification.json').write_text(json.dumps(write,indent=2))
    print(json.dumps({k:v for k,v in write.items() if k!='source_sha256'}))


def pairing_values(values):
    values=np.asarray(values);n=len(values);means=values.mean(1)
    draws=sorted(sum(means[i] for i in ids)/n for ids in itertools.product(range(n),repeat=n))
    return [values.mean(),*[draws[max(0,int(np.ceil(p*len(draws)))-1)] for p in (.025,.975)],
            *[np.delete(values,i,axis=0).mean() for i in range(n)]]


if __name__=='__main__':verify(Path(sys.argv[1]),Path(sys.argv[2]))

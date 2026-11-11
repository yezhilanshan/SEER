"""Independent frozen-panel metric and target audit; no training scorer imports."""
import math
from pathlib import Path
import numpy as np
from .verification import _metrics, _check
from .preprocess import write_json


def audit_job(path, expected, records, stats):
    with np.load(path,allow_pickle=False) as a:
        arrays={key:a[key] for key in a.files}
    panel=arrays['panel']; truth=arrays['future'].astype('float64'); current=arrays['current'].astype('float64')
    for row,(rid,t,_,_) in enumerate(panel):
        record=records[int(rid)]
        assert record['role']=='validation' and 0<=t<len(record['legal'])
        target=np.stack([(record['motion'][t+h]-stats['mean'])/stats['std'] for h in (1,3,5)])
        np.testing.assert_array_equal(arrays['future'][row],target)
        assert arrays['labels'][row]==record['semantic'][t]
        np.testing.assert_array_equal(arrays['crossing'][row],record['crossing'][t])
        np.testing.assert_array_equal(arrays['future_unknown'][row],[record['unknown'][t+h] for h in (1,3,5)])
        np.testing.assert_array_equal(arrays['future_mixed'][row],[record['mixed'][t+h] for h in (1,3,5)])
    methods={'persistence':np.broadcast_to(current[:,None],truth.shape),
             'linear_trend':current[:,None]+np.array([1,3,5])[None,:,None]*(current-arrays['previous'])[:,None]}
    if 'mean' in arrays: methods['motion']=arrays['mean'].astype('float64')
    summary={}
    selected=panel[:,2].astype(bool)
    for name,prediction in methods.items():
        mse=[]; nll=[]; skills=[]
        for rid in sorted(set(panel[selected,0].tolist())):
            mask=selected&(panel[:,0]==rid); residual=prediction[mask]-truth[mask]
            variance=np.exp(stats['logvar'].astype('float64'))
            actual=np.einsum('nhd,nhd->h',residual,residual)/(mask.sum()*64)
            base=truth[mask]-current[mask,None]
            reference=np.einsum('nhd,nhd->h',base,base)/(mask.sum()*64)
            gaussian=(np.square(residual)/variance+np.log(variance)+math.log(2*math.pi)).sum((0,2))/(2*mask.sum()*64)
            mse.append(actual); nll.append(gaussian); skills.append(np.divide(reference-actual,reference,out=np.zeros(3),where=reference>0))
        computed={'mse':np.mean(mse,0),'nll':np.mean(nll,0),'skill':np.mean(skills,0)}
        target=expected['motion'] if name=='motion' else expected['baselines'][name]
        for key,value in computed.items(): np.testing.assert_allclose(value,target[key],atol=1e-9,rtol=1e-9)
        summary[name]={key:value.tolist() for key,value in computed.items()}
    if 'mean' in arrays:
        for index,horizon in enumerate((1,3,5)):
            groups={'cross_instance':arrays['crossing'][:,index],
                    'known_interior':(~arrays['crossing'][:,index])&(~arrays['future_unknown'][:,index])&(~arrays['future_mixed'][:,index])&(arrays['labels']>=0),
                    'unknown_future':arrays['future_unknown'][:,index],
                    'mixed_future':arrays['future_mixed'][:,index]}
            for name,group in groups.items():
                chosen=selected&group; target=expected['motion_groups'][str(horizon)][name]
                if not chosen.any():
                    assert target is None
                    continue
                mse=[]; skill=[]
                for rid in sorted(set(panel[chosen,0].tolist())):
                    mask=chosen&(panel[:,0]==rid)
                    error=arrays['mean'][mask,index].astype('float64')-truth[mask,index]
                    base=current[mask]-truth[mask,index]
                    actual=np.einsum('nd,nd->',error,error)/error.size
                    reference=np.einsum('nd,nd->',base,base)/base.size
                    mse.append(actual); skill.append((reference-actual)/reference if reference>0 else 0.)
                assert target['n']==int(chosen.sum())
                np.testing.assert_allclose([np.mean(mse),np.mean(skill)],[target['mse'],target['skill']],atol=1e-9,rtol=1e-9)
    if 'logits' in arrays:
        chosen=panel[:,3].astype(bool); computed=[]
        for rid in sorted(set(panel[chosen,0].tolist())):
            mask=chosen&(panel[:,0]==rid)
            metric=_metrics(arrays['logits'][mask].astype('float64'),arrays['labels'][mask])
            _check(metric,expected['semantic']['records'][str(rid)]); computed.append(metric['macro_f1'])
        _check(_metrics(arrays['logits'][chosen].astype('float64'),arrays['labels'][chosen]),expected['semantic']['pooled'])
        np.testing.assert_allclose(np.mean(computed),expected['semantic']['record_macro_f1'],atol=1e-12)
        summary['semantic_record_macro_f1']=float(np.mean(computed))
    result={'passed':True,'origins':len(panel),'metrics':summary}
    write_json(Path(path).parent/'independent-audit.json',result)
    return result

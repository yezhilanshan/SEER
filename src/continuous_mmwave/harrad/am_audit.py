"""Independent saved-logit verification including causal source labels and lengths."""
import numpy as np
from .verification import _metrics,_check


def audit_semantic(arrays,expected,records,bank):
    panel=arrays['panel']; labels=arrays['labels']; scores=arrays['logits'].astype('float64')
    assert np.isfinite(scores).all() and scores.shape==(len(panel),12)
    for index,(rid,t,_,_) in enumerate(panel):
        r=records[int(rid)]; start=max(0,int(r['latest_start'][t]),int(t)-31)
        assert r['role'] in ('train','validation') and labels[index]==r['semantic'][t]>=0
        assert arrays['lengths'][index]==t-start+1
    pooled=_metrics(scores,labels); _check(pooled,expected['pooled']); values=[]; accuracy=[]
    for rid in sorted(set(panel[:,0].tolist())):
        chosen=panel[:,0]==rid; metric=_metrics(scores[chosen],labels[chosen])
        _check(metric,expected['records'][str(rid)]); values.append(metric['macro_f1']); accuracy.append(metric['accuracy'])
    np.testing.assert_allclose([np.mean(values),np.mean(accuracy)],
        [expected['record_macro_f1'],expected['record_accuracy']],atol=1e-12,rtol=1e-12)
    for name,lo,hi in (('u1_4',1,4),('u5_16',5,16),('u17_32',17,32)):
        chosen=(arrays['lengths']>=lo)&(arrays['lengths']<=hi); target=expected['prefix_groups'][name]
        if not chosen.any(): assert target is None; continue
        assert target['n']==int(chosen.sum()); _check(_metrics(scores[chosen],labels[chosen]),target['pooled'])
        metrics=[_metrics(scores[chosen&(panel[:,0]==rid)],labels[chosen&(panel[:,0]==rid)]) for rid in sorted(set(panel[chosen,0].tolist()))]
        np.testing.assert_allclose(np.mean([r['macro_f1'] for r in metrics]),target['record_macro_f1'],atol=1e-12)
    positive=np.einsum('nd,nd->n',arrays['query'].astype('float64'),arrays['reference'].astype('float64')).mean()
    np.testing.assert_allclose(positive,expected['positive_cosine'],atol=1e-6)
    keys=bank['vectors'].detach().cpu().numpy().astype('float64'); bank_labels=bank['labels'].cpu().numpy()
    negative=np.mean([np.einsum('d,nd->n',query,keys[bank_labels!=label]).mean()
                      for query,label in zip(arrays['query'].astype('float64'),labels)])
    np.testing.assert_allclose(negative,expected['negative_cosine'],atol=1e-6)
    return {'passed':True,'n':len(panel),'record_macro_f1':float(np.mean(values))}

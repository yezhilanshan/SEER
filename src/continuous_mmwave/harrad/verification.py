"""Independent audit of saved logits; deliberately imports no training metric code."""
from pathlib import Path
import numpy as np


def _metrics(scores,truth):
    n=len(truth); guesses=np.argmax(scores,axis=1)
    confusion=[[0 for _ in range(12)] for _ in range(12)]
    for actual,predicted in zip(truth.tolist(),guesses.tolist()): confusion[actual][predicted]+=1
    def f1_for(classes,selected):
        values=[]
        for cls in classes:
            tp=sum(a==cls and p==cls for a,p in zip(truth[selected],guesses[selected]))
            actual=int(np.count_nonzero(truth[selected]==cls)); predicted=int(np.count_nonzero(guesses[selected]==cls))
            values.append(2*tp/(actual+predicted) if actual+predicted else 0.)
        return sum(values)/len(values)
    recall=[confusion[i][i]/sum(confusion[i]) if sum(confusion[i]) else 0. for i in range(12)]
    # logaddexp.reduce is independent of the explicit softmax/exp-sum training implementation.
    centered=scores-scores.max(axis=1,keepdims=True)
    losses=np.logaddexp.reduce(centered,axis=1)-centered[np.arange(n),truth]
    events=truth<6; gestures=truth>=6
    return {'n':n,'accuracy':float(np.count_nonzero(guesses==truth)/n),'ce':float(np.mean(losses)),
            'macro_f1':f1_for(list(range(12)),np.ones(n,dtype=bool)),
            'events_macro_f1':f1_for(list(range(6)),events) if events.any() else None,
            'gestures_macro_f1':f1_for(list(range(6,12)),gestures) if gestures.any() else None,
            'confusion':confusion,'per_class_recall':recall}


def _check(actual,expected):
    for key in ('n','accuracy','ce','macro_f1','events_macro_f1','gestures_macro_f1','confusion','per_class_recall'):
        if actual[key] is None: assert expected[key] is None,key
        else: np.testing.assert_allclose(actual[key],expected[key],atol=1e-12,rtol=1e-12,err_msg=key)


def verify_predictions(path: str | Path, expected: dict) -> dict:
    with np.load(path,allow_pickle=False) as arrays:
        scores=np.asarray(arrays['logits'],dtype=np.float64); truth=np.asarray(arrays['labels'],dtype=np.int64)
        subjects=np.asarray(arrays['subjects'])
    assert scores.shape==(len(truth),12) and len(truth)>0 and np.isfinite(scores).all()
    assert np.all((truth>=0)&(truth<12)) and subjects.shape==truth.shape
    result=_metrics(scores,truth); _check(result,expected)
    assert set(subjects.tolist())==set(expected['subjects'])
    result['subjects']={}
    for subject in sorted(set(subjects.tolist())):
        chosen=subjects==subject; calculated=_metrics(scores[chosen],truth[chosen])
        _check(calculated,expected['subjects'][subject]); result['subjects'][subject]=calculated
    return {'passed':True,**result}

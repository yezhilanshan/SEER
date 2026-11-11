"""Independently audit exported events and traces using assignment matching."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def maximum_matches(eligible, quality):
    if not eligible.size:
        return 0
    costs=np.where(eligible,-(sum(eligible.shape)+1)-quality,0.)
    rows,cols=linear_sum_assignment(costs)
    return int(eligible[rows,cols].sum())


def verify(directory):
    directory=Path(directory)
    report=read(directory/'complete.json')
    truth={r['file']:r for r in read(directory/'evaluation-truth.json')}
    checks=0
    for row in report['records']:
        record=truth[row['file']]
        folder=directory/row['arm']
        stem=Path(row['file']).stem
        events=read(folder/(stem+'-events.json'))
        with np.load(folder/(stem+'-trace.npz'),allow_pickle=False) as trace:
            frames=record['frames']; known=np.zeros(frames,bool); labels=np.full(frames,-100,int)
            for instance in record['instances']:
                known[instance['start']:instance['stop']]=True
                labels[instance['start']:instance['stop']]=instance['label']
            fires=np.flatnonzero(trace['fire'])
            assert 0 not in fires
            edges=np.array(sorted({p for t in record['instances'] for p in (t['start'],t['stop']) if 0<p<frames}))
            distance=np.abs(fires[:,None]-edges[None,:])
            valid=known.copy(); valid[1:]|=known[:-1]; valid[0]=False
            for tolerance in (.2,.5,1.):
                metric=row['metric']['boundary'][str(tolerance)]
                assert maximum_matches(distance/15<=tolerance+1e-12,-distance/frames)==metric['tp']
                used={m['predicted'] for m in metric['matches']}
                assert len(used)==metric['tp']
                assert len(edges)-len(used)==metric['fn']
                assert sum(valid[p] for p in fires if p not in used)==metric['fp']
                checks+=1
            probability=trace['probabilities']
            predicted=probability[known].argmax(1)
            matrix=np.zeros((12,12),int)
            np.add.at(matrix,(labels[known],predicted),1)
            denominator=matrix.sum(0)+matrix.sum(1)
            f1=np.divide(2*np.diag(matrix),denominator,out=np.zeros(12),where=denominator>0).mean()
            assert abs(f1-row['metric']['online']['macro_f1'])<1e-12
            if 'published_cutoff' in trace.files:
                assert np.all(trace['published_cutoff']<=np.arange(frames)+1)
            checks+=1
        start=np.array([p['start_frame'] for p in events])[:,None]
        stop=np.array([p['stop_frame'] for p in events])[:,None]
        ts=np.array([t['start'] for t in record['instances']])[None,:]
        te=np.array([t['stop'] for t in record['instances']])[None,:]
        iou=np.maximum(0,np.minimum(stop,te)-np.maximum(start,ts))/(np.maximum(stop,te)-np.minimum(start,ts))
        classes=np.array([p['class_id'] for p in events])[:,None]
        labels=np.array([t['label'] for t in record['instances']])[None,:]
        for threshold in (.3,.5):
            metric=row['metric']['conservative'][str(threshold)]
            assert maximum_matches((iou>=threshold)&(classes==labels),iou)==metric['tp']
            used={m['prediction'] for m in metric['matches']}
            matched_truth={m['truth'] for m in metric['matches']}
            assert len(used)==len(matched_truth)==metric['tp']
            for pair in metric['matches']:
                assert iou[pair['prediction'],pair['truth']]>=threshold
                assert events[pair['prediction']]['class_id']==record['instances'][pair['truth']]['label']
            false=[e for i,e in enumerate(events) if i not in used and known[e['start_frame']:e['stop_frame']].any()]
            assert len(false)==metric['fp']
            assert len(record['instances'])-len(matched_truth)==metric['fn']
            checks+=1
    result=dict(passed=True,record_model_pairs=len(report['records']),checks=checks,
                independent_matcher='SciPy linear assignment; maximum cardinality and quality',
                checks_include='boundary TP/FP/FN, event TP/FP/FN, class/IoU, online Macro-F1 and publication causality')
    (directory/'independent-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    verify(sys.argv[1])

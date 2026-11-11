"""Recompute P1 metrics independently of the training/evaluation implementation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):assert np.isclose(a,b,atol=1e-12,rtol=0),(a,b)


def edits(a,b):
    d=[[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):d[i][0]=i
    for j in range(len(b)+1):d[0][j]=j
    for i,x in enumerate(a,1):
        for j,y in enumerate(b,1):d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(x!=y))
    return d[-1][-1]


def verify(rows,s):
    assert len(rows)==s['windows']
    a=[r['reference_actions'] for r in rows];b=[r['predicted_actions'] for r in rows]
    exact=np.array([x==y for x,y in zip(a,b)]);diff=np.array([len(y)-len(x) for x,y in zip(a,b)])
    assert exact.sum()==s['exact_count']
    for k,v in dict(count_accuracy=(diff==0).mean(),count_mae=np.abs(diff).mean(),count_bias=diff.mean(),
                    under_windows=(diff<0).sum(),over_windows=(diff>0).sum(),equal_count_windows=(diff==0).sum(),
                    sequence_exact_given_correct_count=exact[diff==0].mean() if (diff==0).any() else None,
                    edit_distance_mean=np.mean([edits(x,y) for x,y in zip(a,b)])).items():
        if v is None:assert s[k] is None
        else:close(v,s[k])
    precision=[];recall=[];lcs=[];support=Counter();predicted=Counter();covered=Counter()
    for x,y in zip(a,b):
        hit=sum((Counter(x)&Counter(y)).values());precision.append(hit/max(1,len(y)));recall.append(hit/max(1,len(x)))
        table=np.zeros((len(x)+1,len(y)+1),int)
        for i,label in enumerate(x,1):
            for j,other in enumerate(y,1):table[i,j]=table[i-1,j-1]+1 if label==other else max(table[i-1,j],table[i,j-1])
        lcs.append(table[-1,-1]/max(1,len(x),len(y)));support.update(x);predicted.update(y);covered.update(Counter(x)&Counter(y))
    for k,v in dict(sequence_exact=exact.mean(),action_precision=np.mean(precision),action_recall=np.mean(recall),order_lcs=np.mean(lcs)).items():close(v,s['metrics'][k])
    assert s['insertions']+s['deletions']+s['substitutions']==sum(edits(x,y) for x,y in zip(a,b))
    assert s['insertions']-s['deletions']==diff.sum()
    for r in s['per_class']:
        i=r['class_id'];assert (r['support'],r['predicted'],r['covered'])==(support[i],predicted[i],covered[i])
        if support[i]:close(r['recall'],covered[i]/support[i])
        else:assert r['recall'] is None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',required=True);parser.add_argument('--p0',required=True)
    args=parser.parse_args();root=Path(args.results);raw=root/'experiment';p0=Path(args.p0)/'experiment'
    manifest=load(root/'result-manifest.json')
    for name,item in manifest.items():assert sha(root/name)==item['sha256'] and (root/name).stat().st_size==item['bytes']
    a=load(raw/'p1-a.json');b=load(raw/'p1-b.json');c=load(raw/'p1-c.json');training=load(raw/'probe-training.json');selection=load(raw/'selection.json')
    summaries=0;count_support={};count_results={}
    for name,entry in training.items():
        assert sha(raw/'probes'/f'{name}.pt')==entry['checkpoint_sha256']
        def rank(h):
            v=h['validation']
            return (v['count_accuracy'],-v['count_mae']) if name=='independent-count' else (v['ordered']['metrics']['sequence_exact'],v['presence_macro_f1'],v['count_accuracy'])
        assert max(entry['history'],key=rank)['epoch']==entry['selected_epoch']==selection['probe_epochs'][name]
    def seqrank(x):return tuple(x['validation']['metrics'][k] for k in ('sequence_exact','order_lcs','action_recall'))
    assert max(selection['candidates'],key=seqrank)['config']==selection['hmm']
    assert max([v for v in selection['candidates'] if v['config']['weight']==0],key=seqrank)['config']==selection['probability']
    assert max(selection['count_candidates'],key=seqrank)['config']==selection['count']
    for split in ('validation','test'):
        windows=load(raw/f'windows-{split}.json');before=load(p0/f'windows-{split}.json')
        assert [(w['sample_id'],w['actions'],w['input_sha256']) for w in windows]==[(w['sample_id'],w['actions'],w['input_sha256']) for w in before]
        pred=load(raw/f'probe-predictions-{split}.json');cal=load(raw/f'calibration-predictions-{split}.json');old=load(p0/f'predictions-{split}.json')
        assert cal['B1']==old['B1']['individual'] and cal['B3']==old['B3']
        for name,rows in cal.items():
            assert [r['sample_id'] for r in rows]==[w['sample_id'] for w in windows]
            assert [r['reference_actions'] for r in rows]==[w['actions'] for w in windows]
            if name=='predicted-count':assert [len(r['predicted_actions']) for r in rows]==pred['independent-count']['count']
            if name=='ORACLE-GT-count':assert all(len(r['reference_actions'])==len(r['predicted_actions']) for r in rows)
            for subset in ('full12','original_protocol'):
                subset_rows=[r for r in rows if subset=='original_protocol' or r['full_window']]
                verify(subset_rows,c[split][subset][name]);summaries+=1
                if name in ('B1','B3'):verify(subset_rows,a[split][subset][name]);summaries+=1
        truth=np.zeros((len(windows),14),bool);gt=np.array([len(w['actions']) for w in windows]);full=np.array([w['full_window'] for w in windows])
        for i,w in enumerate(windows):truth[i,list(set(w['actions']))]=True
        count_support[split]={str(i):int((gt[full]==i).sum()) for i in (1,2,3)}
        n=np.array(pred['independent-count']['count']);assert np.array_equal(n,np.array(pred['independent-count']['logits']).argmax(-1)+1)
        count_results[split]=dict(accuracy=float((gt[full]==n[full]).mean()),mae=float(np.abs(gt[full]-n[full]).mean()),
                                majority_accuracy=float((gt[full]==pred['majority-count']).mean()),majority=pred['majority-count'])
        for name,entry in pred.items():
            if name in ('independent-count','majority-count'):continue
            pr=np.array(entry['presence']);n=np.array(entry['count']);assert pr.shape==truth.shape
            for subset in ('full12','original_protocol'):
                mask=full if subset=='full12' else np.ones(len(windows),bool);s=b[split][name][subset]
                tp=(pr[mask]&truth[mask]).sum(0);fp=(pr[mask]&~truth[mask]).sum(0);fn=(~pr[mask]&truth[mask]).sum(0)
                f1=2*tp/np.maximum(1,2*tp+fp+fn);close(f1.mean(),s['presence_macro_f1']);assert np.allclose(f1,s['presence_per_class_f1'],atol=1e-12,rtol=0)
                close((n[mask]==gt[mask]).mean(),s['count_accuracy']);close(np.abs(n[mask]-gt[mask]).mean(),s['count_mae'])
                confusion=np.zeros((3,3),int)
                for x,y in zip(gt[mask],n[mask]):confusion[x-1,y-1]+=1
                assert confusion.tolist()==s['count_confusion']
                verify([r for r,k in zip(entry['ordered'],mask) if k],s['ordered']);summaries+=1
    review=load(raw/'paired-error-review.json');all_ids=set()
    for group,n in (('B1_only',29),('B3_only',57)):
        assert len(review[group])==n
        for row in review[group]:
            assert row['sample_id'] not in all_ids;all_ids.add(row['sample_id']);ref=row['reference_actions']
            assert (ref==row['B1'],ref==row['B3'])==((True,False) if group=='B1_only' else (False,True))
            for name in ('B1','B3'):
                e=row[name+'_edit'];assert edits(ref,row[name])==e['distance']==e['insertions']+e['deletions']+e['substitutions']
                rebuilt=[op['predicted'] for op in e['operations'] if op['op']!='deletion'];assert rebuilt==row[name]
    derived=root/'derived';derived.mkdir(exist_ok=True)
    supplementary=derived/'cache-audit-manifest.json'
    if supplementary.exists():
        for name,item in load(supplementary).items():assert sha(derived/name)==item['sha256'] and (derived/name).stat().st_size==item['bytes']
        majority=load(derived/'majority-count-diagnostic.json');train=load(raw/'windows-train.json')
        assert majority['post_hoc'] and majority['majority_from_train']==Counter(len(w['actions']) for w in train).most_common(1)[0][0]
        assert majority['decoder']==selection['count']
        for split in ('validation','test'):
            rows=load(derived/f'majority-count-predictions-{split}.json');windows=load(raw/f'windows-{split}.json')
            assert [(r['sample_id'],r['reference_actions']) for r in rows]==[(w['sample_id'],w['actions']) for w in windows]
            assert all(len(r['predicted_actions'])==majority['majority_from_train'] for r in rows)
            for subset in ('full12','original_protocol'):
                verify([r for r in rows if subset=='original_protocol' or r['full_window']],majority['results'][split][subset])
    else:raise AssertionError('missing supplementary node audit')
    result=dict(state='passed',archive_files=len(manifest),summaries_independently_recomputed=summaries,paired_review_windows=len(all_ids),
                count_support=count_support,independent_count=count_results,probe_selection_and_calibration_validation_only=True,
                supplementary_majority_metrics_verified=True,supplementary_files_hashed=len(load(supplementary)),
                limits='Independent metric/selection/input/hash audit; production decoder reconstruction and server cache audit reported separately.')
    (derived/'local-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

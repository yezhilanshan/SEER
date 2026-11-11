"""Audit P0 paired readouts with independent smoothing, sequence and count arithmetic."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def extract(text, phrases):
    candidates=[]
    for action,phrase in phrases.items():
        candidates += [(m.start(),m.end(),int(action)) for m in re.finditer(
            r'(?<!\w)'+re.escape(phrase.rstrip('.').lower())+r'(?!\w)',text.lower())]
    accepted=[]
    for item in sorted(candidates,key=lambda x:(-(x[1]-x[0]),x[0])):
        if all(item[0]>=other[1] or item[1]<=other[0] for other in accepted):accepted.append(item)
    return [item[2] for item in sorted(accepted)]


def metrics(row):
    a,b=row['reference_actions'],row['predicted_actions']
    hit=sum((Counter(a)&Counter(b)).values())
    # Full DP table, separate from the evaluator's one-row implementation.
    table=np.zeros((len(a)+1,len(b)+1),dtype=int)
    for i,x in enumerate(a,1):
        for j,y in enumerate(b,1):
            table[i,j]=table[i-1,j-1]+1 if x==y else max(table[i-1,j],table[i,j-1])
    return dict(sequence_exact=float(a==b),action_precision=hit/max(1,len(b)),
                action_recall=hit/max(1,len(a)),order_lcs=int(table[-1,-1])/max(1,len(a),len(b)),
                unmatched=float(not b))


def smooth(pred,times,threshold):
    dt=float(np.median(np.diff(times)))
    assert np.all(np.diff(times)>0)
    med,minimum=threshold['median_seconds'],threshold['minimum_seconds']
    values=pred.copy()
    if med:
        for i,t in enumerate(times):
            neighbors=pred[np.abs(times-t)<=med/2+1e-8]
            values[i]=np.partition(neighbors,len(neighbors)//2)[len(neighbors)//2]
    starts=[0]+[i for i in range(1,len(values)) if values[i]!=values[i-1]]
    stops=starts[1:]+[len(values)]
    result=[]
    for start,stop in zip(starts,stops):
        if times[stop-1]-times[start]+dt+1e-8>=minimum:
            action=int(values[start])
            if not result or result[-1]!=action:result.append(action)
    return result


def verify_summary(rows,summary):
    assert summary['windows']==len(rows)
    assert summary['exact_count']==sum(r['reference_actions']==r['predicted_actions'] for r in rows)
    independent=[metrics(r) for r in rows]
    for key in independent[0]:
        assert np.isclose(np.mean([r[key] for r in independent]),summary['metrics'][key],atol=1e-12,rtol=0),key
    for mode in ('deficit','excess'):
        diff=[len(r['reference_actions'])-len(r['predicted_actions']) for r in rows]
        if mode=='excess':diff=[-v for v in diff]
        assert summary[f'count_{mode}_windows']==sum(v>0 for v in diff)
        assert summary[f'count_{mode}_occurrences']==sum(max(0,v) for v in diff)
    support,predicted,covered=Counter(),Counter(),Counter()
    for row in rows:
        a,b=Counter(row['reference_actions']),Counter(row['predicted_actions'])
        support.update(a);predicted.update(b);covered.update(a&b)
    delta=np.array([len(r['predicted_actions'])-len(r['reference_actions']) for r in rows])
    count=dict(count_mae=float(np.abs(delta).mean()),count_bias_pred_minus_gt=float(delta.mean()),
               mean_under_count=float(np.maximum(-delta,0).mean()),mean_over_count=float(np.maximum(delta,0).mean()),
               under_windows=int((delta<0).sum()),over_windows=int((delta>0).sum()),equal_count_windows=int((delta==0).sum()))
    for key,value in count.items():assert abs(summary[key]-value)<1e-12,key
    for item in summary['per_class']:
        k=item['class_id']
        assert (item['support'],item['predicted'],item['covered'])==(support[k],predicted[k],covered[k])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',required=True)
    parser.add_argument('--pilot',required=True)
    parser.add_argument('--previous',required=True)
    parser.add_argument('--phrases',default='configs/radar-text-actions-parrad.json')
    args=parser.parse_args()
    root,pilot=Path(args.results),Path(args.pilot)
    raw=root/'experiment';phrases=load(Path(args.phrases))
    manifest=load(root/'result-manifest.json')
    for name,item in manifest.items():
        p=root/name
        assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],name
    comparison=load(raw/'comparison.json');selection=load(raw/'threshold-selection.json')
    actual_encoder=load(raw/'provenance.json')['encoder_sha256']
    previous_inputs=load(Path(args.previous)/'provenance.json')['original_input_sha256']
    expected_encoders={v for k,v in previous_inputs.items() if k.replace('\\','/').endswith('/align/best.pt')}
    assert expected_encoders=={actual_encoder}
    assert comparison['selection']==selection and selection['selection_split']=='validation'
    grid=selection['per_method_candidates']
    rank=lambda row:tuple(row['metrics'][key] for key in ('sequence_exact','order_lcs','action_recall'))+(-row['median_seconds'],-row['minimum_seconds'])
    threshold=lambda row:{key:row[key] for key in ('median_seconds','minimum_seconds')}
    assert threshold(max(selection['common_candidates'],key=rank))==selection['common']
    for method in ('B0','B1','B2'):
        assert threshold(max(grid[method],key=rank))==selection['individual'][method]
        assert len(grid[method])==9
    counts={};checked_rows=0;checked_classification_sequences=0;checked_summaries=0
    for split,expected,full_expected in (('validation',702,700),('test',484,482)):
        windows=load(raw/f'windows-{split}.json');predictions=load(raw/f'predictions-{split}.json')
        previous={r['sample_id']:r for r in load(pilot/f'{split}-evaluation.json')['results']}
        ids=[w['sample_id'] for w in windows]
        assert len(ids)==len(set(ids))==expected and set(ids)==set(previous)
        previous_windows=load(Path(args.previous)/f'windows-{split}.json')
        for now,old in zip(windows,previous_windows):
            assert all(now[k]==old[k] for k in ('sample_id','actions','input_sha256','start','stop','full_window'))
        assert sum(w['full_window'] for w in windows)==full_expected
        assert all(re.fullmatch('[a-f0-9]{64}',w['input_sha256']) for w in windows)
        arrays=np.load(raw/f'logits-{split}.npz')
        with np.load(Path(args.previous)/f'logits-{split}.npz') as earlier:
            assert np.array_equal(arrays['B0'],earlier['B1'])
        offsets=arrays['offsets'];assert len(offsets)==expected+1 and offsets[0]==0
        times=[arrays['timestamps'][a:b] for a,b in zip(offsets[:-1],offsets[1:])]
        for i,w in enumerate(windows):
            a,b=offsets[i:i+2];t=times[i]
            assert b-a==w['stop']-w['start'] and np.all(np.isfinite(t))
            labels=arrays['labels'][a:b]
            runs=[int(v) for j,v in enumerate(labels) if v>=0 and (j==0 or v!=labels[j-1])]
            assert runs==w['actions']
            assert bool(w['duration']+w['period']>=12-1e-6)==w['full_window']
        for method in ('B0','B1','B2','B3'):
            variants={'common':predictions[method]} if method=='B3' else predictions[method]
            for variant,rows in variants.items():
                assert [r['sample_id'] for r in rows]==ids
                for i,(w,row) in enumerate(zip(windows,rows)):
                    assert row['reference_actions']==w['actions'] and row['full_window']==w['full_window']
                    assert extract(row['generated'],phrases)==row['predicted_actions']
                    for key,value in metrics(row).items():assert abs(row[key]-value)<1e-12,(method,variant,key)
                    if method=='B3':
                        assert row['generated']==previous[row['sample_id']]['generated']
                    else:
                        th=selection['common'] if variant=='common' else selection['individual'][method] if variant=='individual' else dict(median_seconds=0,minimum_seconds=0)
                        a,b=offsets[i:i+2]
                        assert smooth(arrays[method][a:b].argmax(-1),times[i],th)==row['predicted_actions']
                        checked_classification_sequences+=1
                    checked_rows+=1
                for subset in ('full12','original_protocol'):
                    selected=[r for r in rows if subset=='original_protocol' or r['full_window']]
                    summary=comparison['results'][split][subset][method]
                    verify_summary(selected,summary if method=='B3' else summary[variant]);checked_summaries+=1
            if split=='validation' and method!='B3':
                for candidate in grid[method]:
                    seqs=[]
                    for i,w in enumerate(windows):
                        if not w['full_window']:continue
                        a,b=offsets[i:i+2]
                        seqs.append(dict(reference_actions=w['actions'],predicted_actions=smooth(arrays[method][a:b].argmax(-1),times[i],candidate)))
                    independent=[metrics(r) for r in seqs]
                    for key in ('sequence_exact','order_lcs','action_recall'):
                        assert abs(np.mean([r[key] for r in independent])-candidate['metrics'][key])<1e-12
        counts[split]=dict(windows=expected,complete_windows=full_expected)
    for candidate in selection['common_candidates']:
        for key in ('sequence_exact','order_lcs','action_recall'):
            values=[next(r for r in grid[m] if threshold(r)==threshold(candidate))['metrics'][key] for m in grid]
            assert abs(np.mean(values)-candidate['metrics'][key])<1e-12
    with (root/'derived/align-readout-per-class.csv').open(encoding='utf-8-sig') as stream:
        csv_rows=list(csv.DictReader(stream))
    assert len(csv_rows)==560 and {int(r['class_id']) for r in csv_rows}==set(range(14))
    verification=dict(state='passed',raw_files_sha256_verified=len(manifest),counts=counts,
                      per_window_metric_rows=checked_rows,classification_sequences_rebuilt=checked_classification_sequences,
                      aggregate_summaries_recomputed=checked_summaries,validation_grid_recomputed=27,count_decomposition_verified=True,
                      b3_previous_text_differences=0,align_encoder_sha256=actual_encoder,
                      align_matches_previous_b3=True,csv_rows=len(csv_rows))
    (root/'derived/local-verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification))


if __name__=='__main__':main()

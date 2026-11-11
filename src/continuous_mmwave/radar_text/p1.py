"""P1 sequence evidence diagnostics, light probes and count-controlled decoding."""
from collections import Counter

import numpy as np
import torch
from torch import nn


def edit_alignment(reference, predicted):
    n,m=len(reference),len(predicted)
    cost=np.zeros((n+1,m+1),dtype=int);ways=np.ones_like(cost)
    cost[:,0]=np.arange(n+1);cost[0]=np.arange(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1):
            values=(cost[i-1,j-1]+int(reference[i-1]!=predicted[j-1]),cost[i-1,j]+1,cost[i,j-1]+1)
            best=min(values);cost[i,j]=best
            ways[i,j]=min(2,sum(w for v,w in zip(values,(ways[i-1,j-1],ways[i-1,j],ways[i,j-1])) if v==best))
    i,j=n,m;ops=[]
    while i or j:
        if i and j and cost[i,j]==cost[i-1,j-1]+int(reference[i-1]!=predicted[j-1]):
            op='match' if reference[i-1]==predicted[j-1] else 'substitution'
            ops.append(dict(op=op,reference_index=i-1,predicted_index=j-1,reference=reference[i-1],predicted=predicted[j-1]));i-=1;j-=1
        elif i and cost[i,j]==cost[i-1,j]+1:
            ops.append(dict(op='deletion',reference_index=i-1,reference=reference[i-1]));i-=1
        else:
            ops.append(dict(op='insertion',predicted_index=j-1,predicted=predicted[j-1]));j-=1
    ops.reverse();counts=Counter(o['op'] for o in ops)
    return dict(distance=int(cost[n,m]),ambiguous=bool(ways[n,m]>1),operations=ops,
                insertions=counts['insertion'],deletions=counts['deletion'],substitutions=counts['substitution'])


def sequence_metrics(a,b):
    common=sum((Counter(a)&Counter(b)).values());dp=[0]*(len(b)+1)
    for x in a:
        nxt=[0]
        for j,y in enumerate(b,1):nxt.append(dp[j-1]+1 if x==y else max(dp[j],nxt[-1]))
        dp=nxt
    return dict(sequence_exact=float(a==b),action_precision=common/max(1,len(b)),action_recall=common/max(1,len(a)),
                order_lcs=dp[-1]/max(1,len(a),len(b)))


def sequence_summary(rows):
    if not rows:raise ValueError('empty sequence summary')
    metrics=[sequence_metrics(r['reference_actions'],r['predicted_actions']) for r in rows]
    edits=[edit_alignment(r['reference_actions'],r['predicted_actions']) for r in rows]
    diff=np.array([len(r['predicted_actions'])-len(r['reference_actions']) for r in rows])
    conditional=[m['sequence_exact'] for m,d in zip(metrics,diff) if d==0]
    support,predicted,covered=Counter(),Counter(),Counter()
    for row in rows:
        a,b=Counter(row['reference_actions']),Counter(row['predicted_actions'])
        support.update(a);predicted.update(b);covered.update(a&b)
    return dict(windows=len(rows),exact_count=int(sum(m['sequence_exact'] for m in metrics)),
                metrics={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
                count_accuracy=float(np.mean(diff==0)),count_mae=float(np.abs(diff).mean()),count_bias=float(diff.mean()),
                under_windows=int((diff<0).sum()),over_windows=int((diff>0).sum()),equal_count_windows=int((diff==0).sum()),
                sequence_exact_given_correct_count=float(np.mean(conditional)) if conditional else None,
                edit_distance_mean=float(np.mean([e['distance'] for e in edits])),
                ambiguous_edit_windows=sum(e['ambiguous'] for e in edits),
                **{k:sum(e[k] for e in edits) for k in ('insertions','deletions','substitutions')},
                per_class=[dict(class_id=i,support=support[i],predicted=predicted[i],covered=covered[i],
                     recall=covered[i]/support[i] if support[i] else None) for i in range(14)])


def make_targets(actions):
    presence=torch.zeros(len(actions),14);slots=torch.full((len(actions),3),14,dtype=torch.long)
    for i,a in enumerate(actions):
        if not 1<=len(a)<=3 or any(not 0<=v<14 for v in a):raise ValueError('invalid ordered target')
        presence[i,list(set(a))]=1;slots[i,:len(a)]=torch.tensor(a)
    return dict(presence=presence,count=torch.tensor([len(a)-1 for a in actions]),slots=slots)


def decode_outputs(logits):
    x=logits.detach().cpu();presence=x[:,:14].sigmoid().numpy()>=.5;counts=x[:,14:17].argmax(-1).numpy()+1
    actions=[]
    for slots in x[:,17:].reshape(-1,3,15).argmax(-1).tolist():
        seq=[]
        for action in slots:
            if action==14:break
            seq.append(action)
        actions.append(seq)
    return presence,counts,actions


class EvidenceProbe(nn.Sequential):
    def __init__(self,width,nonlinear=False,outputs=62):
        layers=[nn.Linear(width,32)]
        if nonlinear:layers.append(nn.GELU())
        layers.append(nn.Linear(32,outputs));super().__init__(*layers)


def probability_smooth(probabilities,times,seconds):
    probabilities=np.asarray(probabilities);times=np.asarray(times)
    if seconds<0 or np.any(np.diff(times)<=0):raise ValueError('invalid probability smoothing input')
    if not seconds:return probabilities.copy()
    prefix=np.vstack([np.zeros(probabilities.shape[1]),probabilities.cumsum(0)])
    lo=np.searchsorted(times,times-seconds/2-1e-8);hi=np.searchsorted(times,times+seconds/2+1e-8,side='right')
    return (prefix[hi]-prefix[lo])/(hi-lo)[:,None]


def transition_prior(recordings,classes=14):
    trans=np.ones((classes,classes));start=np.ones(classes)
    for raw in recordings:
        labels=np.asarray(raw)
        if len(labels) and 0<=labels[0]<classes:start[labels[0]]+=1
        a,b=labels[:-1],labels[1:];valid=(a>=0)&(a<classes)&(b>=0)&(b<classes)
        np.add.at(trans,(a[valid],b[valid]),1)
    return np.log(trans/trans.sum(1,keepdims=True)),np.log(start/start.sum())


def hmm_path(logp,transition,start,weight):
    score=logp[0]+weight*start;back=[]
    for emission in logp[1:]:
        values=score[:,None]+weight*transition
        previous=values.argmax(0);score=values[previous,np.arange(len(score))]+emission;back.append(previous)
    action=int(score.argmax());path=[action]
    for previous in reversed(back):action=int(previous[action]);path.append(action)
    return np.array(path[::-1])


def segment_paths(logp,times,minimum,transition,start,max_count=3):
    """Best constant-label segments for each feasible count; adjacent labels differ."""
    logp=np.asarray(logp);times=np.asarray(times);t,c=logp.shape
    dt=float(np.median(np.diff(times)));prefix=np.vstack([np.zeros(c),logp.cumsum(0)])
    if minimum<0 or times[-1]-times[0]+dt+1e-8<minimum:raise ValueError('infeasible minimum duration')
    dp=np.full((max_count+1,t+1,c),-np.inf);begins=np.full_like(dp,-1,dtype=int);previous=np.full_like(begins,-1)
    for end in range(1,t+1):
        if times[end-1]-times[0]+dt+1e-8>=minimum:dp[1,end]=prefix[end]+start
    for count in range(2,max_count+1):
        tr=np.asarray(transition).copy();np.fill_diagonal(tr,-np.inf)
        scores=dp[count-1,:, :,None]+tr[None,:,:]
        prev=scores.argmax(1);best=np.max(scores,axis=1)
        for end in range(1,t+1):
            legal=np.flatnonzero(times[:end]<=times[end-1]+dt-minimum+1e-8)
            if not len(legal):continue
            values=best[legal]-prefix[legal]+prefix[end]
            chosen=values.argmax(0);left=legal[chosen]
            dp[count,end]=values[chosen,np.arange(c)];begins[count,end]=left;previous[count,end]=prev[left,np.arange(c)]
    paths={}
    for count in range(1,max_count+1):
        action=int(dp[count,t].argmax())
        if not np.isfinite(dp[count,t,action]):continue
        result=[];end=t
        for n in range(count,0,-1):
            result.append(action)
            if n>1:left=begins[n,end,action];action=int(previous[n,end,action]);end=int(left)
        paths[count]=result[::-1]
        assert len(paths[count])==count and all(a!=b for a,b in zip(paths[count],paths[count][1:]))
    return paths

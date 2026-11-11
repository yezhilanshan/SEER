"""Episode matching with partial annotations and causal full-recording rollout."""
import numpy as np
import torch
from .boundary import score_events,combine_scores
from .training import classification_metrics


def score_episodes(predicted,instances,known_frames,iou=.5):
    p=np.asarray(predicted,dtype=np.int64).reshape(-1,3)
    truth=sorted(instances,key=lambda r:r['start']); eligible=np.zeros((len(p),len(truth)),dtype=bool)
    for i,(s,e,label) in enumerate(p):
        for j,row in enumerate(truth):
            overlap=max(0,min(int(e),row['stop'])-max(int(s),row['start']))
            union=max(int(e),row['stop'])-min(int(s),row['start'])
            eligible[i,j]=label==row['label'] and overlap/max(union,1)>=iou
    dp=np.zeros((len(p)+1,len(truth)+1),dtype=np.int32)
    for i in range(len(p)-1,-1,-1):
        for j in range(len(truth)-1,-1,-1):
            dp[i,j]=max(dp[i+1,j],dp[i,j+1],int(eligible[i,j])+dp[i+1,j+1])
    i=j=0; matched=set()
    while i<len(p) and j<len(truth):
        if eligible[i,j] and dp[i,j]==1+dp[i+1,j+1]: matched.add(i); i+=1; j+=1
        elif dp[i+1,j]>=dp[i,j+1]: i+=1
        else: j+=1
    false=[i for i in range(len(p)) if i not in matched]
    fp=sum(bool(known_frames[max(0,s):min(e,len(known_frames))].any()) for s,e,_ in p[false])
    tp=len(matched); fn=len(truth)-tp; unknown=len(false)-fp
    return {'tp':tp,'fp':fp,'fn':fn,'unscored_unknown':unknown,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


@torch.no_grad()
def rollout(model,record,threshold=.5,mode='autonomous',limit=None,device='cuda'):
    model.eval(); n=len(record['motion']) if limit is None else min(limit,len(record['motion']))
    state=model.initial(1,device); start=0; events=[]; cuts=[]; probabilities=[]; scores=[]; forecasts=[]
    last_label=0
    for t in range(n):
        forced=None
        if mode!='autonomous':
            value=record['reset']['boundary'][t] if mode=='oracle' else mode=='always_reset'
            forced=torch.tensor([float(value)],device=device)
        previous=state
        state,out=model.step(record['za'][t:t+1].to(device),record['zb'][t:t+1].to(device),
                            torch.as_tensor(record['normalized'][t:t+1],device=device),state,threshold,forced)
        if t and out['decision'].item()==1:
            events.append((start,2*t,last_label)); start=2*t; cuts.append(2*t+2)
        last_label=int(out['logits'].argmax(-1)); probabilities.append(float(out['q']))
        scores.append(out['logits'][0].float().cpu().numpy()); forecasts.append(out['mean'][0].float().cpu().numpy())
    events.append((start,2*n,last_label))
    return {'episodes':np.array(events,dtype=np.int64),'cutoffs':np.array(cuts,dtype=np.int64),
            'effective_boundaries':np.array(cuts,dtype=np.int64)-2,
            'q':np.array(probabilities),'logits':np.array(scores),'mean':np.array(forecasts)}


def evaluate_record(arrays,record,stats):
    n=len(arrays['q']); annotation=record['reset']; known=annotation['known_frames'][:2*n]
    truth=annotation['truth']; truth=truth[(truth//2>0)&(truth//2<n)]
    boundary=score_events(arrays['cutoffs'],truth,known,4)
    boundary['fp_per_known_minute']=boundary['fp']/(known.sum()/900) if known.any() else None
    instances=[r for r in annotation['instances'] if r['stop']<=2*n]
    episodes=score_episodes(arrays['episodes'],instances,known)
    labels=annotation['labels'][:n]; valid=labels>=0
    semantic=classification_metrics(arrays['logits'][valid],labels[valid]) if valid.any() else None
    instance_scores=[]; instance_labels=[]
    for row in instances:
        first=(int(row['start'])+1)//2; last=int(row['stop'])//2
        if first<last:
            instance_scores.append(arrays['logits'][first:last].mean(0)); instance_labels.append(row['label'])
    instance_semantic=classification_metrics(np.array(instance_scores),np.array(instance_labels)) if instance_scores else None
    if instance_semantic is not None:
        instance_semantic.update(total_instances=len(instances),coverage=len(instance_scores)/len(instances))
    t=np.arange(max(0,min(n,len(record['motion'])-5)))
    target=np.stack([record['normalized'][t+h] for h in (1,3,5)],1)
    mse=np.square(arrays['mean'][t]-target).mean((0,2)) if len(t) else np.full(3,np.nan)
    return {'boundary':boundary,'episodes':episodes,'semantic':semantic,'instance_semantic':instance_semantic,'mse':mse.tolist(),
            'skill':(1-mse/stats['persistence_mse']).tolist(),'n_updates':n}

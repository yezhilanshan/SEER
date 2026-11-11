"""Stage2 head fitting and state-carrying, paired Policy-only optimization."""
import hashlib
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .reset_data import iter_stream_schedule
from .reset_model import detached
from .preprocess import write_json,sha256


def source_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if name.startswith(('tokens_a.','dynamics.','semantic.')):
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def working(model):
    return {name:{k:v.detach().cpu().clone() for k,v in getattr(model,name).state_dict().items()}
            for name in ('keep','birth','policy')}


def load_working(model,weights):
    if set(weights)!=set(('keep','birth','policy')): raise ValueError('invalid reset weights')
    for name in weights: getattr(model,name).load_state_dict(weights[name],strict=True)


@torch.no_grad()
def cache_tokens(model,records,stats,device):
    model.eval()
    for r in records:
        r['normalized']=((r['motion']-stats['mean'])/stats['std']).astype('float32')
        za=[]; zb=[]
        for start in range(0,len(r['motion']),256):
            cells=torch.as_tensor(np.array(r['cells'][start:start+256],copy=True),device=device)
            mask=torch.ones(cells.shape[:2],dtype=torch.bool,device=device)
            za.append(model.tokens_a(cells,mask).float().cpu()); zb.append(model.semantic.tokens(cells,mask).float().cpu())
        r['za']=torch.cat(za); r['zb']=torch.cat(zb)


@torch.no_grad()
def _oracle_dynamics(model,r,device):
    if 'oracle_g' in r: return
    g=torch.zeros(1,model.dynamics.hidden_size,device=device); local=g.clone(); gs=[]; ls=[]; after=[]
    for t in range(len(r['motion'])):
        gs.append(g[0].cpu().clone()); ls.append(local[0].cpu().clone())
        x=r['za'][t:t+1].to(device).mean(1)[:,None]
        _,gg=model.dynamics(x,g[None]); g=gg[0]
        if r['reset']['boundary'][t] or t==0: local=torch.zeros_like(local)
        _,ll=model.dynamics(x,local[None]); local=ll[0]; after.append(local[0].cpu().clone())
    r['oracle_g']=torch.stack(gs); r['oracle_l']=torch.stack(ls); r['oracle_after_l']=torch.stack(after)


def _future_loss(mean,target,stats):
    scale=torch.as_tensor(stats['persistence_mse'],device=mean.device,dtype=mean.dtype)
    return ((mean-target).square().mean(-1)/scale*mean.new_tensor([.1,.3,.6])).sum(-1).mean()


def fit_heads(model,records,ids,stats,steps,batch_size,output,device='cuda'):
    output=Path(output); output.mkdir(parents=True,exist_ok=False); before=source_digest(model)
    model.phase('heads').eval(); rng=np.random.default_rng(42); pools={False:[],True:[]}
    for rid in ids:
        r=records[rid]
        if r['role']!='train': raise ValueError('nontrain head sample')
        _oracle_dynamics(model,r,device)
        for born in (False,True):
            t=np.arange(1,len(r['motion'])-5)
            chosen=t[r['reset']['valid'][t]&(r['reset']['boundary'][t]==born)]
            if len(chosen): pools[born].append((rid,chosen))
    if not all(pools.values()): raise ValueError('both keep/birth supports required')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
    history=[]; started=time.time()
    for step in range(1,steps+1):
        sampled=[]
        for born in (False,True):
            for _ in range(batch_size//2):
                rid,positions=pools[born][rng.integers(len(pools[born]))]
                sampled.append((rid,int(rng.choice(positions)),born))
        def gather(key,previous=False):
            vals=[records[rid][key][t-1 if previous else t] for rid,t,_ in sampled]
            return torch.stack([torch.as_tensor(v) for v in vals]).to(device)
        current=gather('normalized'); previous=gather('normalized',True)
        local=gather('oracle_l'); global_=gather('oracle_g'); after=gather('oracle_after_l')
        born=torch.tensor([v for _,_,v in sampled],device=device,dtype=torch.bool)
        keep=model.predict(local,previous)[:,0]; birth=model.predict(global_,previous,model.birth)[:,0]
        target=torch.as_tensor(np.stack([records[rid]['normalized'][t+np.array([1,3,5])] for rid,t,_ in sampled]),device=device)
        keep_nll=.5*(((current-keep).square()/model.variance)+model.variance.log()).mean(-1)
        birth_nll=.5*(((current-birth).square()/model.variance)+model.variance.log()).mean(-1)
        future=_future_loss(model.predict(after,current),target,stats)
        loss=future+.5*(keep_nll[~born].mean()+birth_nll[born].mean())
        opt.zero_grad(set_to_none=True); loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm): raise RuntimeError('nonfinite Stage2A')
        opt.step()
        if step==1 or step%10==0:
            row={'step':step,'attempts':step,'loss':float(loss.detach()),'future':float(future.detach()),'seconds':time.time()-started}
            history.append(row); write_json(output/'status.json',{'phase':'stage2a',**row})
    if before!=source_digest(model): raise RuntimeError('Stage2A changed source')
    torch.save({'working':working(model),'step':steps,'attempts':steps,'optimizer':opt.state_dict()},output/'final.pt')
    write_json(output/'history.json',history)
    report={'successful_steps':steps,'attempts':steps,'source_unchanged':True,'seconds':time.time()-started,'sha256':sha256(output/'final.pt')}
    write_json(output/'complete.json',report); return report


def policy_losses(outputs,record,start,stats,teacher,future_enabled):
    device=outputs[0]['q'].device; t=np.arange(start,start+len(outputs))
    logits=torch.cat([v['policy_logit'] for v in outputs]); q=torch.cat([v['q'] for v in outputs]); zero=q.sum()*0
    target=torch.as_tensor(record['reset']['boundary'][t],device=device,dtype=torch.bool)
    valid=torch.as_tensor(record['reset']['valid'][t],device=device); valid&=torch.as_tensor(t>0,device=device)
    pieces=[]
    for positive in (False,True):
        selected=valid&(target==positive)
        if selected.any(): pieces.append(F.binary_cross_entropy_with_logits(logits[selected],target[selected].float()))
    boundary=torch.stack(pieces).mean() if pieces else zero
    known=torch.as_tensor(record['reset']['labels'][t],device=device)
    semantic=F.cross_entropy(torch.cat([v['logits'] for v in outputs]),known,ignore_index=-100)/math.log(12) if (known>=0).any() else zero
    chosen=t<len(record['motion'])-5
    if chosen.any():
        means=torch.cat([v['mean'] for v in outputs])[torch.as_tensor(chosen,device=device)]
        target_future=torch.as_tensor(np.stack([record['normalized'][t[chosen]+h] for h in (1,3,5)],1),device=device)
        future=_future_loss(means,target_future,stats)
    else: future=zero
    rate=q.mean()
    total=boundary if teacher else boundary+.25*semantic+.05*rate+(future if future_enabled else future.detach())
    return {'total':total,'boundary':boundary,'semantic':semantic,'future':future,'rate':rate}


def fit_policy(model,records,ids,stats,steps,chunk,seed,output,teacher,future_enabled,device='cuda',
               checkpoints=(0,300,1000),evaluate=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    if any(records[i]['role']!='train' for i in ids): raise ValueError('nontrain policy samples')
    model.phase('policy'); before=source_digest(model)
    frozen_heads={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith(('keep.','birth.'))}
    opt=torch.optim.AdamW(model.policy.parameters(),lr=1e-4,weight_decay=.01)
    # Every recording completes a traversal within this many round-robin reads.
    # Teacher-only unknown chunks advance states but cannot spend a BCE update.
    if teacher and not any(records[i]['reset']['valid'][1:].any() for i in ids):
        raise ValueError('teacher needs reliable boundary supervision')
    traversal=len(ids)*max(math.ceil(len(records[i]['motion'])/chunk) for i in ids)
    budget=steps*traversal if teacher else steps
    schedule=iter_stream_schedule(records,ids,budget,chunk,seed); executed=[]
    states={}; positions={}; history=[]; started=time.time(); evaluations={}
    step=0; attempts=0; skipped_unknown=0
    def snapshot(step):
        torch.save({'working':working(model),'step':step,'attempts':attempts,'optimizer':opt.state_dict(),
                    'states':{str(k):{n:v.detach().cpu() for n,v in state.items()} for k,state in states.items()},
                    'positions':positions},output/f'step-{step:04d}.pt')
        if evaluate is not None:
            write_json(output/'status.json',{'phase':'evaluation','step':step,'attempts':attempts})
            evaluations[str(step)]=evaluate(model,step,output)
            write_json(output/'evaluations.json',evaluations)
    if 0 in checkpoints: snapshot(0)
    for rid,start,stop in schedule:
        attempts+=1; executed.append((rid,start,stop))
        r=records[rid]; model.train()
        state=model.initial(1,device) if start==0 else states[rid]
        if start!=positions.get(rid,0): raise RuntimeError('stream position discontinuity')
        outputs=[]
        # Teacher states need no graph; Policy scores still need one for BCE.
        for t in range(start,stop):
            forced=torch.tensor([float(r['reset']['boundary'][t])],device=device) if teacher else None
            state,out=model.step(r['za'][t:t+1].to(device),r['zb'][t:t+1].to(device),
                torch.as_tensor(r['normalized'][t:t+1],device=device),state,forced=forced)
            if teacher: state=detached(state)
            outputs.append(out)
        states[rid]=detached(state); positions[rid]=0 if stop==len(r['motion']) else stop
        reliable=r['reset']['valid'][start:stop] & (np.arange(start,stop)>0)
        if teacher and not reliable.any():
            skipped_unknown+=1
            continue
        losses=policy_losses(outputs,r,start,stats,teacher,future_enabled)
        opt.zero_grad(set_to_none=True); losses['total'].backward()
        norm=torch.nn.utils.clip_grad_norm_(model.policy.parameters(),1.)
        if not torch.isfinite(losses['total']) or not torch.isfinite(norm): raise RuntimeError('nonfinite Policy update')
        opt.step(); step+=1
        if step==1 or step%10==0:
            row={'step':step,'attempts':attempts,'skipped_unknown_chunks':skipped_unknown,'seconds':time.time()-started,**{k:float(v.detach()) for k,v in losses.items()}}
            history.append(row); write_json(output/'status.json',{'phase':'training','total_steps':steps,**row})
        if step in checkpoints: snapshot(step)
        if step%100==0: print(f'{output.name} {step}/{steps}',flush=True)
        if step==steps: break
    np.save(output/'schedule.npy',np.asarray(executed,dtype=np.int64))
    if step!=steps: raise RuntimeError('insufficient reliable teacher chunks')
    if source_digest(model)!=before: raise RuntimeError('Policy changed source')
    if any(not torch.equal(p,frozen_heads[n]) for n,p in model.named_parameters() if n in frozen_heads): raise RuntimeError('Policy changed head')
    torch.save({'working':working(model),'step':steps,'attempts':attempts},output/'final.pt')
    write_json(output/'history.json',history)
    result={'successful_steps':steps,'attempts':attempts,'skipped_unknown_chunks':skipped_unknown,'source_unchanged':True,'heads_unchanged':True,
            'last_positions':{str(k):v for k,v in positions.items()},'seconds':time.time()-started,'evaluations':evaluations}
    write_json(output/'complete.json',result); write_json(output/'status.json',{'phase':'complete','step':steps,'attempts':attempts})
    return result

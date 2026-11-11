"""Frozen source dynamics/semantics with a differentiable shared reset decision."""
import copy
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .joint_model import build_joint
from .event_semantic import build_production


def detached(state): return {key:value.detach() for key,value in state.items()}


class ResetModel(nn.Module):
    def __init__(self,dynamics,semantic):
        super().__init__(); self.tokens_a=dynamics.tokens; self.dynamics=dynamics.dynamics
        self.keep=dynamics.forecast; self.birth=copy.deepcopy(self.keep); self.semantic=semantic
        d=self.dynamics.hidden_size; e=semantic.former.config.llm_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261022)
            self.policy=nn.ModuleDict({'event':nn.Sequential(nn.LayerNorm(e),nn.Linear(e,64),nn.Tanh()),
                'score':nn.Sequential(nn.Linear(2*d+64+64+2,128),nn.GELU(),nn.Linear(128,1))})
        self.register_buffer('variance',torch.ones(64)); self.phase('policy')

    def train(self,mode=True):
        super().train(mode); self.semantic.former.adapter.eval(); return self

    def phase(self,name):
        if name not in ('heads','policy'): raise ValueError('unknown reset phase')
        self.requires_grad_(False)
        for module in ((self.keep,self.birth) if name=='heads' else (self.policy,)): module.requires_grad_(True)
        self._phase=name; return self

    def initial(self,batch,device):
        d=self.dynamics.hidden_size
        return {'G':torch.zeros(batch,d,device=device),'L':torch.zeros(batch,d,device=device),
                'E':self.semantic.former.initial_state(batch),'age':torch.zeros(batch,device=device),
                'previous':torch.zeros(batch,64,device=device),'seen':torch.zeros(batch,dtype=torch.bool,device=device)}

    def predict(self,hidden,current,head=None):
        head=self.keep if head is None else head
        return current[:,None]+head(torch.cat((hidden.float(),current.float()),-1)).reshape(-1,3,64)

    def step(self,za,zb,current,state,threshold=.5,forced=None):
        old=state; keep=self.predict(old['L'],old['previous'])[:,0]
        birth=self.predict(old['G'],old['previous'],self.birth)[:,0]
        evidence=(((current-keep).square()-(current-birth).square())/self.variance).mean(-1)*.5
        event=self.policy['event'](old['E'].float().mean(1))
        feature=torch.cat((old['G'].float(),old['L'].float(),current.float(),event,
                           (old['age']/9).clamp(max=1)[:,None],evidence.clamp(-20,20)[:,None]),-1)
        logit=self.policy['score'](feature).squeeze(-1); q=logit.sigmoid()
        hard=(q>=threshold).to(q.dtype)
        reset=hard+(q-q.detach()) if self.training and torch.is_grad_enabled() else hard
        if forced is not None: reset=forced.to(q)
        reset=torch.where(old['seen'],reset,torch.ones_like(reset))
        with torch.no_grad(): _,g=self.dynamics(za.mean(1).float()[:,None],old['G'][None])
        _,local=self.dynamics(za.mean(1).float()[:,None],((1-reset[:,None])*old['L'])[None])
        base=self.semantic.former.initial_state(len(current))
        incoming=(1-reset[:,None,None])*old['E'].float()+reset[:,None,None]*base.float()
        age=(1-reset)*old['age']+2/15; maturity=(age/1.2).clamp(max=1)
        prompt=self.semantic.prompt(len(current)); mask=torch.ones(zb.shape[:2],dtype=torch.bool,device=zb.device)
        def advance(previous,radar,age_value):
            return self.semantic.former.stream_update(previous,radar,age_value,mask,prompt).event_tokens
        if self.training and incoming.requires_grad:
            event_state=checkpoint(advance,incoming,zb,maturity,use_reentrant=False)
        else: event_state=advance(incoming,zb,maturity)
        state={'G':g[0],'L':local[0],'E':event_state,'age':age,'previous':current,
               'seen':torch.ones_like(old['seen'])}
        return state,{'q':q,'decision':reset,'policy_logit':logit,'keep':keep,'birth':birth,
                      'mean':self.predict(local[0],current),'logits':self.semantic.read(event_state),
                      'evidence':evidence}

    def learned(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}


def build_reset(config,device='cuda'):
    source=config['sources']; a=build_joint(config,'A',42,device)
    a.load_learned(torch.load(source['dynamics']['path'],map_location='cpu',weights_only=False)['learned'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42); b=build_production(config,device)
    b.load_learned(torch.load(config['event_checkpoint'],map_location='cpu',weights_only=False)['learned'])
    from .event_training import frozen_digest
    if frozen_digest(b)!=config['warmup_frozen_digest']: raise ValueError('Reset Qwen source mismatch')
    return ResetModel(a,b).to(device)

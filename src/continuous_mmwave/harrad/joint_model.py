"""Shared radar-token learning with separate local dynamics and semantic states."""
import copy
from contextlib import nullcontext
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .event_semantic import RadarTokenAdapter, build_production


class JointModel(nn.Module):
    def __init__(self, semantic, arm, seed=42, dynamics_dim=256):
        super().__init__()
        if arm not in ('M','A','B','AB'): raise ValueError('unknown joint arm')
        self.arm=arm; self.checkpoint_updates=True; self._prompts={}
        self.tokens=semantic.tokens if semantic is not None and arm!='M' else None
        self.former=semantic.former if arm in ('B','AB') else None
        self.reader=semantic.reader if arm in ('B','AB') else None
        if self.former is not None: self.former.adapter.requires_grad_(False).eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+100)
            dim=self.tokens.project.out_features if self.tokens is not None else 256
            self.dynamics=nn.GRU(dim,dynamics_dim,batch_first=True) if arm in ('A','AB') else None
            torch.manual_seed(seed+200)
            if arm in ('M','A','AB'):
                self.forecast=nn.Sequential(nn.Linear(64 if arm=='M' else dynamics_dim+64,256),nn.GELU(),nn.Linear(256,192))
                nn.init.zeros_(self.forecast[-1].weight); nn.init.zeros_(self.forecast[-1].bias)
            else: self.forecast=None

    def train(self,mode=True):
        super().train(mode)
        if self.former is not None: self.former.adapter.eval()
        return self

    def _prompt(self,batch):
        key=(batch,str(self.former.base_event_tokens.device),str(self.former.adapter.compute_dtype))
        if key not in self._prompts:
            with torch.no_grad(): self._prompts[key]=self.former.build_prompt_cache(batch)
        return self._prompts[key]

    def _roll(self,cells):
        batch,length=cells.shape[:2]
        D=cells.new_zeros(1,batch,self.dynamics.hidden_size) if self.dynamics is not None else None
        E=self.former.initial_state(batch) if self.former is not None else None
        prompt=self._prompt(batch) if self.former is not None else None
        burn=max(0,length-16)
        amp=torch.is_autocast_enabled(cells.device.type)
        dtype=torch.get_autocast_dtype(cells.device.type)
        valid=torch.ones(batch,2,dtype=torch.bool,device=cells.device)
        for index in range(length):
            context=torch.no_grad() if self.training and index<burn else nullcontext()
            with context,torch.autocast(cells.device.type,dtype=dtype,enabled=amp,cache_enabled=False):
                if self.training and index==burn:
                    if D is not None: D=D.detach()
                    if E is not None: E=E.detach()
                Z=self.tokens(cells[:,index],valid)
                if D is not None: _,D=self.dynamics(Z.mean(1)[:,None],D)
                if E is not None:
                    age=cells.new_full((batch,),min((index+1)*2/15/1.2,1.))
                    def update(previous,radar,maturity):
                        return self.former.stream_update(previous,radar,maturity,None,prompt).event_tokens
                    if self.training and torch.is_grad_enabled() and self.checkpoint_updates:
                        E=checkpoint(update,E,Z,age,use_reentrant=False)
                    else: E=update(E,Z,age)
        result={}
        if D is not None: result['D']=D[0]
        if E is not None:
            result['E']=E; result['logits']=self.reader(E.float().mean(1))
        return result

    def forward(self,cells,mask,current):
        if cells.ndim!=5 or cells.shape[2:]!=(2,4,64) or mask.shape!=cells.shape[:2] or current.shape!=(len(cells),64):
            raise ValueError('invalid joint prefix')
        if not mask.any(1).all() or (mask[:,1:]&~mask[:,:-1]).any() or cells.shape[1]>32:
            raise ValueError('nonempty right-padded prefix <=32 required')
        result={}
        if self.arm!='M':
            lengths=mask.long().sum(1); groups=[]
            for length in sorted(lengths.unique().tolist()):
                ids=torch.nonzero(lengths==length,as_tuple=False).flatten()
                groups.append((ids,self._roll(cells[ids,:length])))
            order=torch.cat([ids for ids,_ in groups]).argsort()
            result={key:torch.cat([part[key] for _,part in groups],0)[order] for key in groups[0][1]}
        if self.forecast is not None:
            inputs=current if self.arm=='M' else torch.cat((result['D'],current),-1)
            result['mean']=current[:,None]+self.forecast(inputs).reshape(-1,3,64)
        return result

    def learned_state(self):
        return {name:p.detach().cpu().clone() for name,p in self.named_parameters() if p.requires_grad}

    def load_learned(self,weights):
        if set(weights)!={name for name,p in self.named_parameters() if p.requires_grad}:
            raise ValueError('learned whitelist mismatch')
        with torch.no_grad():
            for name,p in self.named_parameters():
                if name in weights:
                    if p.shape!=weights[name].shape or not torch.isfinite(weights[name]).all(): raise ValueError('invalid learned tensor')
                    p.copy_(weights[name])


def build_joint(config,arm,seed,device='cuda'):
    # The generic base and pretrained HARrad parameters are read-only sources.
    if arm in ('B','AB'):
        # Reproduce the frozen Qwen/LoRA initialization of HARrad warmup for all
        # new seeds; only newly trained components should change across seeds.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            semantic=build_production(config,device)
        saved=torch.load(config['event_checkpoint'],map_location='cpu',weights_only=False)
        semantic.load_learned(saved['learned'])
    elif arm=='A':
        from types import SimpleNamespace
        tokens=RadarTokenAdapter(256)
        saved=torch.load(config['event_checkpoint'],map_location='cpu',weights_only=False)
        tokens.load_state_dict({name[len('tokens.'):]:value for name,value in saved['learned'].items() if name.startswith('tokens.')})
        semantic=SimpleNamespace(tokens=tokens)
    else: semantic=None
    return JointModel(semantic,arm,seed).to(device)

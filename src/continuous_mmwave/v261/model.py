"""Frozen semantic dynamics retain derivatives with respect to shared inputs."""
from __future__ import annotations

import copy
from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..v26.source import state_digest

ARMS = ('M','A','B','AB')


class SharedRadarAdapter(nn.Module):
    def __init__(self, dim: int = 256, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim,bottleneck)
        self.up = nn.Linear(bottleneck,dim)
        nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)

    def forward(self, F: Tensor) -> Tensor:
        return F + self.up(torch.nn.functional.gelu(self.down(self.norm(F))))


class JointPilotModel(nn.Module):
    def __init__(self, frontend: nn.Module, arm: str, seed: int = 42):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('unknown joint arm')
        self.arm,self.seed = arm,seed
        self.former = frontend.event_former.requires_grad_(False).eval()
        self.old_projection = copy.deepcopy(frontend.semantic_projection).requires_grad_(False).eval()
        self.old_classifier = copy.deepcopy(frontend.auxiliary_classifier).requires_grad_(False).eval()
        self.reader = copy.deepcopy(frontend.semantic_projection) if arm in ('B','AB') else None
        self.classifier = copy.deepcopy(frontend.auxiliary_classifier) if arm in ('B','AB') else None
        if self.reader is not None:
            self.reader.requires_grad_(True);self.classifier.requires_grad_(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+100)
            self.shared_adapter = SharedRadarAdapter()
            torch.manual_seed(seed+200)
            self.dynamics = nn.GRU(256,256,batch_first=True) if arm in ('A','AB') else None
            torch.manual_seed(seed+300)
            width = 564 if arm == 'M' else 256
            self.forecast = nn.Sequential(nn.Linear(512,width),nn.GELU(),nn.Linear(width,768)) if arm in ('M','A','AB') else None
            if self.forecast is not None:
                nn.init.zeros_(self.forecast[-1].weight);nn.init.zeros_(self.forecast[-1].bias)
        self.register_buffer('fit_mean',torch.zeros(256))
        self.register_buffer('fit_std',torch.ones(256))
        self.checkpoint_updates = True
        self._prompt_caches = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.former.eval();self.old_projection.eval();self.old_classifier.eval()
        return self

    def set_stats(self, stats: dict):
        self.fit_mean.copy_(stats['mean']);self.fit_std.copy_(stats['std'])

    def _prompt(self, batch: int):
        key = (batch,str(self.fit_mean.device),str(self.former.adapter.compute_dtype))
        if key not in self._prompt_caches:
            with torch.no_grad():
                self._prompt_caches[key] = self.former.build_prompt_cache(batch)
        return self._prompt_caches[key]

    def _roll(self, F: Tensor, teacher: bool = False) -> dict:
        batch,length = F.shape[:2]
        want_B = teacher or self.arm in ('B','AB')
        want_A = not teacher and self.dynamics is not None
        E = self.former.initial_state(batch) if want_B else None
        D = F.new_zeros(1,batch,256) if want_A else None
        prompt = self._prompt(batch) if want_B else None
        burn = max(0,length-16)
        # Autocast can otherwise retain a detached low-precision copy of a
        # trainable weight first used during no-grad burn-in, and reuse it in
        # the gradient tail. Keep the same precision but bypass that cache.
        try:
            amp_enabled=torch.is_autocast_enabled(F.device.type)
            amp_dtype=torch.get_autocast_dtype(F.device.type)
        except TypeError:  # Compatibility with older torch device getters.
            amp_enabled=torch.is_autocast_enabled() if F.is_cuda else torch.is_autocast_cpu_enabled()
            amp_dtype=torch.get_autocast_gpu_dtype() if F.is_cuda else torch.get_autocast_cpu_dtype()

        def update(previous,radar,maturity):
            return self.former.stream_update(previous,radar,maturity,None,prompt).event_tokens

        if not want_B and not want_A:
            Z = self.shared_adapter(F[:,-1])
            return dict(shared=(Z.mean(1)-self.fit_mean)/self.fit_std)
        for j in range(length):
            context = torch.no_grad() if teacher or j < burn else nullcontext()
            with context,torch.autocast(F.device.type,dtype=amp_dtype,enabled=amp_enabled,cache_enabled=False):
                if j == burn:
                    E = None if E is None else E.detach()
                    D = None if D is None else D.detach()
                # Match the adapter output layout at identity: strided teacher
                # inputs can select a different GEMM reduction even in FP32.
                Z = F[:,j].contiguous() if teacher else self.shared_adapter(F[:,j])
                pooled = (Z.mean(1)-self.fit_mean)/self.fit_std
                if want_A:
                    _,D = self.dynamics(pooled[:,None],D)
                if want_B:
                    maturity = Z.new_full((batch,),min((j+1)*.18/1.2,1.))
                    if self.checkpoint_updates and torch.is_grad_enabled() and not teacher:
                        E = checkpoint(update,E,Z,maturity,use_reentrant=False)
                    else:
                        E = update(E,Z,maturity)
        result = dict(shared=pooled)
        if want_A: result['D'] = D[0]
        if want_B:
            result['E'] = E
            result['old_embedding'] = self.old_projection(E)
            result['old_logits'] = self.old_classifier(result['old_embedding'])
            if not teacher:
                result['logits'] = self.classifier(self.reader(E))
        return result

    @staticmethod
    def _validate(F,mask):
        if F.ndim != 4 or F.shape[2:] != (4,256) or mask.shape != F.shape[:2] or not 1 <= F.shape[1] <= 32:
            raise ValueError('prefix requires F[B,L<=32,4,256] and matching mask')
        if not mask.any(1).all() or (mask[:,1:] & ~mask[:,:-1]).any() or not torch.isfinite(F[mask]).all():
            raise ValueError('finite nonempty right-padded prefixes required')

    def _grouped_roll(self,F,mask,teacher=False):
        self._validate(F,mask)
        pieces=[]
        lengths=mask.long().sum(1)
        for length in sorted(lengths.unique().tolist()):
            indices=torch.nonzero(lengths.eq(length),as_tuple=False).flatten()
            pieces.append((indices,self._roll(F[indices,:length],teacher)))
        order=torch.cat([indices for indices,_ in pieces]).argsort()
        return {key:torch.cat([part[key] for _,part in pieces],0)[order] for key in pieces[0][1]}

    def forward_prefix(self,F: Tensor,mask: Tensor,m_ref: Tensor) -> dict:
        result=self._grouped_roll(F,mask.bool())
        if m_ref.shape != (len(F),256) or not torch.isfinite(m_ref).all():
            raise ValueError('finite common reference motion required')
        if self.forecast is not None:
            context=result['D'] if self.dynamics is not None else result['shared']
            delta=self.forecast(torch.cat((context,m_ref),-1)).reshape(-1,3,256)
            result['mean']=m_ref[:,None]+delta
        return result

    @torch.no_grad()
    def teacher_prefix(self,F: Tensor,mask: Tensor) -> dict:
        output=self._grouped_roll(F,mask.bool(),True)
        return dict(embedding=output['old_embedding'],logits=output['old_logits'])

    def learned_state(self) -> dict:
        return {name:p.detach().cpu().clone() for name,p in self.named_parameters() if p.requires_grad}

    def load_learned_state(self,state: dict):
        expected={name for name,p in self.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError('learned checkpoint keys differ from parameter whitelist')
        with torch.no_grad():
            for name,p in self.named_parameters():
                if name in state:
                    if p.shape != state[name].shape or not torch.isfinite(state[name]).all():
                        raise ValueError('invalid learned parameter')
                    p.copy_(state[name])


def trainable_manifest(model: nn.Module) -> dict:
    return dict(parameters={name:dict(shape=list(p.shape),trainable=p.requires_grad,numel=p.numel())
                            for name,p in model.named_parameters()},
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                components={name:state_digest(part) for name,part in
                            (('adapter',model.shared_adapter),('dynamics',model.dynamics),
                             ('forecast',model.forecast),('reader',model.reader),('classifier',model.classifier)) if part is not None})

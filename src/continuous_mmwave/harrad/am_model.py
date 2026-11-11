"""Causal B-side 2x2 experiment; old JointModel and its checkpoints stay intact."""
import copy
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .am_modules import EventModulation
from .joint_model import JointModel
from .event_semantic import build_production
from .event_training import frozen_digest

ARMS=('B0','B_align','B_mod','B_both')


class AlignmentModel(JointModel):
    def __init__(self,semantic,arm='B_both',seed=42):
        if arm not in ARMS: raise ValueError('unknown alignment/modulation arm')
        super().__init__(semantic,'B',seed)
        self.source_qwen_digest=frozen_digest(self)
        self.arm=arm; self.use_alignment=arm in ('B_align','B_both'); self.use_modulation=arm in ('B_mod','B_both')
        self.modulation_enabled=self.use_modulation
        self.reference=copy.deepcopy(self.tokens).requires_grad_(False).eval()
        dim=self.tokens.project.out_features; event_dim=self.former.config.llm_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261021)
            self.modulator=EventModulation(dim,event_dim,min(dim,64))
            self.alignment=nn.Sequential(nn.LayerNorm(event_dim),nn.Linear(event_dim,dim))
        self.modulator.requires_grad_(self.use_modulation); self.alignment.requires_grad_(self.use_alignment)

    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'reference'): self.reference.eval()
        return self

    @torch.no_grad()
    def reference_vectors(self,cells,mask):
        batch,length=cells.shape[:2]
        with torch.autocast(cells.device.type,enabled=False):
            flat=cells.float().reshape(batch*length,2,4,64)
            valid=mask.reshape(-1,1).expand(-1,2)
            z=self.reference(flat,valid).reshape(batch,length,4,-1)
            pooled=z.sum((1,2))/(mask.sum(1)[:,None]*4)
            return F.normalize(F.layer_norm(pooled,(pooled.shape[-1],)),dim=-1)

    def _roll(self,cells):
        batch,length=cells.shape[:2]; state=self.former.initial_state(batch); prompt=self._prompt(batch)
        valid=torch.ones(batch,2,dtype=torch.bool,device=cells.device); burn=max(0,length-16)
        amp=torch.is_autocast_enabled(cells.device.type); dtype=torch.get_autocast_dtype(cells.device.type)
        for index in range(length):
            context=torch.no_grad() if self.training and index<burn else nullcontext()
            with context,torch.autocast(cells.device.type,dtype=dtype,enabled=amp,cache_enabled=False):
                if self.training and burn>0 and index==burn: state=state.detach()
                z=self.tokens(cells[:,index],valid)
                if self.modulation_enabled: z=self.modulator(z,state)
                age=cells.new_full((batch,),min((index+1)*2/15/1.2,1.))
                def advance(previous,radar,maturity):
                    return self.former.stream_update(previous,radar,maturity,None,prompt).event_tokens
                if self.training and torch.is_grad_enabled() and self.checkpoint_updates:
                    state=checkpoint(advance,state,z,age,use_reentrant=False)
                else: state=advance(state,z,age)
        pooled=state.float().mean(1)
        return {'E':state,'logits':self.reader(pooled),'query':F.normalize(self.alignment(pooled),dim=-1)}

    def forward(self,cells,mask,current):
        result=super().forward(cells,mask,current)
        result['reference']=self.reference_vectors(cells,mask)
        return result


def build_alignment(config,arm,seed,device='cuda'):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42); semantic=build_production(config,device)
    saved=torch.load(config['event_checkpoint'],map_location='cpu',weights_only=False)
    semantic.load_learned(saved['learned'])
    model=AlignmentModel(semantic,arm,seed).to(device)
    if model.source_qwen_digest!=config['warmup_frozen_digest']: raise ValueError('frozen Qwen source mismatch')
    return model

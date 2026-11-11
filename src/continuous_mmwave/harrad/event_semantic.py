"""Spatial radar tokens and instance-supervised causal EventFormer pretraining."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class RadarTokenAdapter(nn.Module):
    def __init__(self,radar_dim=256):
        super().__init__(); self.norm=nn.LayerNorm(64); self.project=nn.Linear(64,radar_dim)
        self.space=nn.Parameter(torch.randn(4,radar_dim)*.02)
        self.time=nn.Parameter(torch.randn(2,radar_dim)*.02)
        self.queries=nn.Parameter(torch.randn(4,radar_dim)*.02)
        self.output_norm=nn.LayerNorm(radar_dim)

    def forward(self,cells,frame_mask):
        if cells.shape[1:]!=(2,4,64) or frame_mask.shape!=cells.shape[:2]: raise ValueError('two frames/four spatial cells required')
        cells=torch.where(frame_mask[:,:,None,None],cells,torch.zeros_like(cells))
        values=F.gelu(self.project(self.norm(cells)))+self.space[None,None]+self.time[None,:,None]
        values=values.flatten(1,2); mask=frame_mask[:,:,None].expand(-1,-1,4).flatten(1)
        valid=mask.any(1); safe=mask.clone(); safe[~valid,0]=True
        scores=torch.einsum('qd,bkd->bqk',self.queries,values)/math.sqrt(values.shape[-1])
        scores=scores.masked_fill(~safe[:,None],torch.finfo(scores.dtype).min)
        tokens=self.output_norm(torch.einsum('bqk,bkd->bqd',scores.softmax(-1),values))
        return torch.where(valid[:,None,None],tokens,torch.zeros_like(tokens))


class EventSemanticModel(nn.Module):
    def __init__(self,former,radar_dim=256,detach_updates=8):
        super().__init__(); self.former=former; self.former.adapter.requires_grad_(False).eval()
        self.tokens=RadarTokenAdapter(radar_dim); d=former.config.llm_dim
        self.reader=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.GELU(),nn.Linear(128,12))
        self.detach_updates=detach_updates; self._prompt_caches={}; self.checkpoint_updates=True

    def train(self,mode=True):
        super().train(mode); self.former.adapter.eval(); return self

    def prompt(self,batch):
        key=(batch,str(self.former.base_event_tokens.device),str(self.former.adapter.compute_dtype))
        if key not in self._prompt_caches:
            with torch.no_grad(): self._prompt_caches[key]=self.former.build_prompt_cache(batch)
        return self._prompt_caches[key]

    def advance(self,cells,mask,state,observed_frames):
        active=mask.any(1); observed_frames=observed_frames+mask.long().sum(1)
        radar=self.tokens(cells,mask); radar_mask=active[:,None].expand(-1,4)
        maturity=(observed_frames.float()/15/1.2).clamp(max=1)
        cache=self.prompt(cells.shape[0])
        def step(previous,current,age,valid):
            return self.former.stream_update(previous,current,age,valid,cache).event_tokens
        if self.training and self.checkpoint_updates:
            candidate=checkpoint(step,state,radar,maturity,radar_mask,use_reentrant=False)
        else: candidate=step(state,radar,maturity,radar_mask)
        state=torch.where(active[:,None,None],candidate,state.to(candidate.dtype))
        return state,observed_frames

    def read(self,state): return self.reader(state.float().mean(1))

    def forward(self,cells,mask):
        if cells.ndim!=4 or cells.shape[2:]!=(4,64) or mask.shape!=cells.shape[:2]: raise ValueError('invalid cell sequence')
        if not mask.any(1).all() or (mask[:,1:]&~mask[:,:-1]).any(): raise ValueError('nonempty right-padded mask required')
        batch,length=cells.shape[:2]
        if length%2:
            cells=F.pad(cells,(0,0,0,0,0,1)); mask=F.pad(mask,(0,1),value=False)
        ends=(mask.sum(1)+1)//2; state=self.former.initial_state(batch)
        observed=torch.zeros(batch,dtype=torch.long,device=cells.device)
        final=cells.new_zeros(batch,12); prefix={}
        for index in range(cells.shape[1]//2):
            valid=mask[:,index*2:index*2+2]
            if not valid.any(): continue
            state,observed=self.advance(cells[:,index*2:index*2+2],valid,state,observed)
            k=index+1; ending=ends==k
            if ending.any() or k in (3,6,12):
                logits=self.read(state)
                final=torch.where(ending[:,None],logits,final)
                if k in (3,6,12): prefix[k]={'logits':logits,'valid':observed>=2*k}
            if self.training and k%self.detach_updates==0: state=state.detach()
        return {'final_logits':final,'prefix':prefix,'observed_frames':observed,'state':state}

    def learned_state(self):
        return {k:p.detach().cpu().clone() for k,p in self.named_parameters() if p.requires_grad}

    def load_learned(self,weights):
        names={k for k,p in self.named_parameters() if p.requires_grad}
        if set(weights)!=names: raise ValueError('trainable checkpoint key mismatch')
        self.load_state_dict(weights,strict=False)


def semantic_loss(output,labels,class_weights):
    loss=F.cross_entropy(output['final_logits'],labels,weight=class_weights)
    pieces=[]
    for p in output['prefix'].values():
        valid=p['valid']
        if valid.any(): pieces.append(F.cross_entropy(p['logits'][valid],labels[valid],weight=class_weights))
    return loss+(.25*torch.stack(pieces).mean() if pieces else 0)


def build_production(config,device='cuda'):
    from transformers import AutoTokenizer
    from continuous_mmwave.models.event_former import Qwen2Adapter,StreamingSemanticEventFormer
    from continuous_mmwave.parrad_config import PARradConfig
    model_cfg=PARradConfig(frame_period_s=1/15,frames_per_update=2,radar_dim=256,event_tokens=8,llm_dim=896,
                          llm_layers=4,cross_attn_dim=512,cross_attn_heads=8,cross_attention_init_scale=.001,
                          event_state_initial_update=.1,dropout=0.)
    tokenizer=AutoTokenizer.from_pretrained(config['qwen_path'],local_files_only=True)
    adapter=Qwen2Adapter.from_pretrained(config['qwen_path'],num_layers=4,lora_rank=16,
                                       torch_dtype=torch.float16,local_files_only=True,attn_implementation='eager')
    adapter.requires_grad_(False).eval()
    ids=tokenizer(model_cfg.fixed_event_prompt,return_tensors='pt',add_special_tokens=False)['input_ids'][0]
    former=StreamingSemanticEventFormer(adapter,model_cfg,ids)
    return EventSemanticModel(former,detach_updates=8).to(device)

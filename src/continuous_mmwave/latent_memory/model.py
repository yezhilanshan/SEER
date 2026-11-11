"""Bootstrap then repeatedly retrieve with the current, LLM-evolved latent memory."""
import torch
from torch import nn
from torch.nn import functional as F
from continuous_mmwave.harrad.event_semantic import RadarTokenAdapter


class LatentMemoryModel(nn.Module):
    def __init__(self, backend, lmt_counts=(4,4,4), rounds=2, top_k=4, fps=15., time_encoding='log_time'):
        super().__init__()
        if len(lmt_counts)!=3 or min(lmt_counts)<1 or rounds<0 or top_k<1 or fps<=0 or time_encoding not in ('none','log_time'):
            raise ValueError('invalid latent-memory model settings')
        self.backend=backend; self.rounds=rounds; self.top_k=top_k; self.fps=fps; self.time_encoding=time_encoding
        dim=backend.hidden_size
        self.local_adapter=RadarTokenAdapter().requires_grad_(False).eval()
        self.projector=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,dim),nn.GELU(),nn.Linear(dim,dim))
        self.time_projection=nn.Linear(4,dim,bias=False) if time_encoding=='log_time' else None
        self.initial_lmts=nn.ParameterList([nn.Parameter(torch.randn(n,dim)*.02) for n in lmt_counts])
        self.state_norm=nn.LayerNorm(dim)
        self.settings=dict(lmt_counts=list(lmt_counts),top_k=top_k,fps=fps,time_encoding=time_encoding,dim=dim)

    def train(self, mode=True):
        super().train(mode); self.backend.eval(); self.local_adapter.eval(); return self

    def load_local_adapter(self, path, expected_sha256):
        from continuous_mmwave.harrad.preprocess import sha256
        if not expected_sha256 or sha256(path)!=expected_sha256.lower(): raise ValueError('local adapter checkpoint hash mismatch')
        payload=torch.load(path,map_location='cpu',weights_only=False)
        # Existing HARrad checkpoints export trainable parameters under learned/model; accept exact tokens.* only.
        source=payload.get('learned',payload.get('model',payload.get('learned_state',{})))
        state={k[len('tokens.'):]:v for k,v in source.items() if k.startswith('tokens.')}
        if not state: raise ValueError('checkpoint contains no tokens.* local adapter parameters')
        self.local_adapter.load_state_dict(state,strict=True)
        if not all(torch.isfinite(p).all() for p in self.local_adapter.parameters()): raise ValueError('nonfinite local adapter')

    def _project(self, items, cutoff):
        device=self.initial_lmts[0].device
        cells=torch.stack([x.values for x in items]).to(device)
        masks=torch.stack([x.mask for x in items]).to(device)
        with torch.no_grad(): local=self.local_adapter(cells,masks)
        projected=self.projector(local)
        if self.time_projection is not None:
            times=torch.tensor([[x.start_frame/self.fps,x.end_frame/self.fps,
                                 (cutoff-x.end_frame)/self.fps,(x.end_frame-x.start_frame)/self.fps]
                                for x in items],device=device,dtype=projected.dtype)
            projected=projected+self.time_projection(times.log1p())[:,None]
        return projected

    def forward(self, memory, prompt_ids, rounds=None, fixed_evidence=False):
        rounds=self.rounds if rounds is None else rounds
        if not isinstance(rounds,int) or rounds<0: raise ValueError('nonnegative integer rounds required')
        items=memory.candidates(2)
        if not items: raise ValueError('cannot query empty memory')
        if any(x.end_frame>memory.observed_frames for x in items): raise ValueError('future memory unit')
        projected=self._project(items,memory.observed_frames)
        pools=[[i for i,x in enumerate(items) if x.level<=g] for g in range(3)]
        states=[]
        for group,indexes in enumerate(pools):
            evidence=projected[indexes].flatten(0,1) if indexes else projected.new_empty(0,projected.shape[-1])
            states.append(self.state_norm(self.backend.read(prompt_ids,evidence,self.initial_lmts[group])))
        trace=[]; previous_selection={}
        for step in range(rounds):
            next_states=[]
            for group,indexes in enumerate(pools):
                query=F.normalize(states[group].mean(0),dim=-1)
                keys=F.normalize(projected[indexes].mean(1),dim=-1)
                scores=keys@query
                ranked=torch.argsort(scores,descending=True,stable=True)[:self.top_k].tolist()
                selected=sorted([indexes[i] for i in ranked])
                if fixed_evidence and group in previous_selection: selected=previous_selection[group]
                previous_selection[group]=selected
                evidence=projected[selected].flatten(0,1) if selected else projected.new_empty(0,projected.shape[-1])
                updated=self.state_norm(self.backend.read(prompt_ids,evidence,states[group]))
                trace.append(dict(round=step,group=group,scores=scores.detach().cpu().tolist(),
                                  candidates=[dict(start_frame=items[i].start_frame,end_frame=items[i].end_frame,level=items[i].level) for i in indexes],
                                  selected=[dict(start_frame=items[i].start_frame,end_frame=items[i].end_frame,level=items[i].level) for i in selected],
                                  delta=float((updated-states[group]).detach().norm())))
                next_states.append(updated)
            states=next_states
        return dict(latents=torch.cat(states)[None],trace=trace,cutoff_frame=memory.observed_frames,rounds=rounds)

    def learned_state(self):
        return {n:p.detach().cpu().clone() for n,p in self.named_parameters() if p.requires_grad}

    def load_learned(self, state):
        actual=self.learned_state()
        if set(actual)!=set(state) or any(actual[n].shape!=p.shape or not torch.isfinite(p).all() for n,p in state.items()):
            raise ValueError('learned checkpoint keys/shapes/values mismatch')
        self.load_state_dict(state,strict=False)

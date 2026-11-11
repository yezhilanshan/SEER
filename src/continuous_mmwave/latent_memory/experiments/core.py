"""Composed experimental paths; old model defaults and weights are untouched."""
import torch
from torch.nn import functional as F


def set_phase(model, phase):
    model.requires_grad_(False)
    if phase=='alignment':
        model.projector.requires_grad_(True)
        if model.time_projection is not None: model.time_projection.requires_grad_(True)
    elif phase=='latent':
        model.initial_lmts.requires_grad_(True); model.state_norm.requires_grad_(True)
    else: raise ValueError('unknown phase')
    model.backend.eval(); model.local_adapter.eval()


def new_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters()
            if not n.startswith(('backend.','local_adapter.'))}


def load_new_state(model, state):
    expected=new_state(model)
    if set(expected)!=set(state) or any(expected[n].shape!=v.shape or not torch.isfinite(v).all() for n,v in state.items()):
        raise ValueError('new state keys/shapes/values mismatch')
    model.load_state_dict(state,strict=False)


def read_prefix(model, memory, prompt, arm, rounds=2):
    if arm=='E0':
        tokens=model._project(memory.candidates(2),memory.observed_frames).flatten(0,1)
        return dict(prefix=tokens[None],trace=[],read_calls=0)
    if arm in ('E1','E3','E4'):
        result=model(memory,prompt,rounds=0 if arm=='E1' else rounds,fixed_evidence=arm=='E3')
        return dict(prefix=result['latents'],trace=result['trace'],read_calls=3*(1+result['rounds']))
    if arm!='E2': raise ValueError('unknown experimental arm')
    result=model(memory,prompt,rounds=0)
    items=memory.candidates(2); tokens=model._project(items,memory.observed_frames)
    selected=set(); offset=0
    for group,lmts in enumerate(model.initial_lmts):
        state=result['latents'][0,offset:offset+len(lmts)]; offset+=len(lmts)
        pool=[i for i,item in enumerate(items) if item.level<=group]
        scores=F.normalize(tokens[pool].mean(1),dim=-1)@F.normalize(state.mean(0),dim=-1)
        indices=torch.argsort(scores,descending=True,stable=True)[:model.top_k].tolist()
        selected.update(pool[i] for i in indices)
    selected=sorted(selected)
    return dict(prefix=tokens[selected].flatten(0,1)[None],trace=[],read_calls=3,
                selected=[dict(start_frame=items[i].start_frame,end_frame=items[i].end_frame,level=items[i].level) for i in selected])

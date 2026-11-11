"""Complete causal frozen Event State replay; no optimizer or target input."""
from __future__ import annotations
from contextlib import nullcontext
import hashlib
import numpy as np
import torch
from .targets import prediction_contract


def replay_plan(intervals, observation_valid, command_starts, view):
    if view not in ('oracle','no_reset'): raise ValueError('unknown reset view')
    contract=prediction_contract(intervals,observation_valid,command_starts,(1,))
    intervals=np.asarray(intervals);valid=contract['observation_valid']
    reset=np.zeros(len(valid),bool);age=np.zeros(len(valid),np.int64)
    segment=np.full(len(valid),-1,np.int64)
    last_pure=None; current_start=-1; elapsed=0
    for t in range(len(valid)):
        if not valid[t]: last_pure=None;current_start=-1;elapsed=0;continue
        discontinuity=(t==0 or not valid[t-1] or
                       not np.isclose(intervals[t-1,1],intervals[t,0],atol=1e-7,rtol=0))
        pure=int(contract['proxy_episode_id'][t])
        # Mixed observations stay in the preceding state, never take the next label.
        # The first subsequently unambiguous observation begins a new oracle state.
        oracle_change=(view=='oracle' and pure>=0 and pure!=last_pure)
        if discontinuity or oracle_change:
            reset[t]=True;current_start=t;elapsed=0
            if discontinuity: last_pure=None
        if view=='oracle' and pure>=0: last_pure=pure
        elapsed+=1;age[t]=elapsed;segment[t]=current_start
    return dict(view=view,valid=valid,reset=reset,age=age,segment_start=segment)


@torch.no_grad()
def replay_events(former,tokens,mask,plan,use_graph=True):
    """Batch-one recurrence. Graph shares exactly the sequential stream_update math."""
    device=next(former.parameters()).device
    valid=np.asarray(plan['valid']);indices=np.flatnonzero(valid)
    if len(tokens)!=len(valid) or tuple(mask.shape)!=tuple(tokens.shape[:-1]):
        raise ValueError('matching radar observations required')
    if not len(indices): raise ValueError('no valid observations')
    if not torch.isfinite(tokens[indices]).all(): raise ValueError('finite radar required')
    if not mask[indices].bool().all(): raise ValueError('four complete radar tokens required')
    prompt=former.build_prompt_cache(1)
    autocast=torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else nullcontext()
    graph=None
    if use_graph and device.type=='cuda':
        previous=former.initial_state(1).to(former.adapter.compute_dtype).clone()
        radar=tokens[indices[0]:indices[0]+1].to(device).clone()
        maturity=torch.tensor([.15],device=device)
        with autocast:
            side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3): former.stream_update(previous,radar,maturity,None,prompt)
            torch.cuda.current_stream().wait_stream(side)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result=former.stream_update(previous,radar,maturity,None,prompt).event_tokens
    values=[];state=None
    with autocast:
        for t in indices:
            if graph is not None:
                if plan['reset'][t]:previous.copy_(former.initial_state(1))
                radar.copy_(tokens[t:t+1]);maturity.fill_(min(int(plan['age'][t])*.18/1.2,1.))
                graph.replay();host=result[0].cpu().clone();previous.copy_(result)
            else:
                if plan['reset'][t]:state=former.initial_state(1)
                state=former.stream_update(state,tokens[t:t+1].to(device),
                    min(int(plan['age'][t])*.18/1.2,1.),mask[t:t+1].to(device),prompt).event_tokens
                host=state[0].cpu().clone()
            values.append(host)
    events=torch.stack(values)
    if not torch.isfinite(events).all():raise RuntimeError('nonfinite complete Event State; no clipping/reset allowed')
    return dict(indices=torch.from_numpy(indices),events=events,
                age=torch.from_numpy(plan['age'][indices]),
                segment_start=torch.from_numpy(plan['segment_start'][indices]))


def diagnostic_indices(intervals, observation_valid, command_starts, plan):
    c=prediction_contract(intervals,observation_valid,command_starts,(1,3,5))
    intervals=np.asarray(intervals);n=len(intervals)
    previous=np.zeros(n,bool)
    previous[1:]=(c['observation_valid'][1:] & c['observation_valid'][:-1] &
        np.isclose(intervals[:-1,1],intervals[1:,0],atol=1e-7,rtol=0))
    common=c['common_start'][previous[c['common_start']]]
    # Deterministic derangement: same recording (thus same person/session/sensor),
    # exact observed age, other completed segment, strictly past availability.
    # No class, future target, validation score, or future state is consulted.
    completed={};active={};last_segment=-1;donor=np.full(n,-1,np.int64)
    for t in range(n):
        if not plan['valid'][t]: continue
        segment=int(plan['segment_start'][t]);age=int(plan['age'][t])
        if segment!=last_segment:
            for a,rows in active.items():completed.setdefault(a,[]).extend(rows)
            active={};last_segment=segment
        choices=completed.get(age,[])
        if choices:
            key=f'p1c-causal-age-v1:{t}:{age}'.encode()
            donor[t]=choices[int.from_bytes(hashlib.sha256(key).digest()[:8],'big')%len(choices)]
        active.setdefault(age,[]).append(t)
    result=dict(common_start=common,shuffle_donor=donor,
                ablation_common_start=common[donor[common]>=0],
                trend_previous_valid=previous)
    for h in (1,3,5):
        result[f'future_h{h}']=c['future'][h]
        result[f'deadline_cross_h{h}']=c['proxy_boundary_cross'][h]
        result[f'mixed_current_h{h}']=c['mixed_current'][h]
        result[f'mixed_future_h{h}']=c['mixed_future'][h]
    return result


def linear_trend(motion,selected,horizon):
    selected=torch.as_tensor(selected,dtype=torch.long)
    if horizon not in (1,3,5) or (selected<1).any():raise ValueError('legal trend inputs required')
    return motion[selected]+horizon*(motion[selected]-motion[selected-1])

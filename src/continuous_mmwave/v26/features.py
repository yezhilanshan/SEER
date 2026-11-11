"""Frozen causal F/E cache; command proxies are never called true instances."""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .source import file_sha, resolve_recording, state_digest, write_json
from .targets import build_command_episodes, update_grid, fixed_motion_target, future_indices


@torch.no_grad()
def event_prefix_features(former, tokens, mask, episodes, valid, max_prefix):
    device = next(former.parameters()).device
    cache = former.build_prompt_cache(1)
    state, previous, age = None, None, 0
    values, selected, prefix_starts, ages, latencies = [], [], [], [], []
    autocast = torch.autocast('cuda',dtype=torch.float16) if device.type == 'cuda' else nullcontext()
    with autocast:
        for t in range(len(tokens)):
            if not valid[t]:
                state, previous, age = None, None, 0
                continue
            if previous != int(episodes[t]):
                state = former.initial_state(1)
                previous, age, start = int(episodes[t]), 0, t
            age += 1
            if age > max_prefix:
                continue
            timed = len(latencies) < 100
            if timed and device.type == 'cuda': torch.cuda.synchronize()
            begin = time.perf_counter()
            state = former.stream_update(state,tokens[t:t+1].to(device),min(age*.18/1.2,1.),
                                        mask[t:t+1].to(device),cache).event_tokens
            if timed:
                if device.type == 'cuda': torch.cuda.synchronize()
                latencies.append(time.perf_counter()-begin)
            selected.append(t); prefix_starts.append(start); ages.append(age)
            values.append(state[0].float().cpu())
    if not values:
        raise ValueError('no eligible event prefixes')
    return dict(indices=torch.tensor(selected), events=torch.stack(values),
                prefix_start=torch.tensor(prefix_starts), age=torch.tensor(ages),
                latency_seconds=latencies)


@torch.no_grad()
def batched_event_prefix_features(former,tokens,mask,episodes,valid,max_prefix,batch_size=16):
    """Batch independent episodes, never time within one recurrent trajectory.

    Fixed batch shape keeps scheduling independent of unseen suffix length.
    Padding rows never contribute outputs or alter another row's state.
    """
    if batch_size < 1: raise ValueError('positive event batch size required')
    runs=[]; t=0
    while t < len(tokens):
        if not valid[t]: t+=1; continue
        start=t
        while t < len(tokens) and valid[t] and int(episodes[t])==int(episodes[start]): t+=1
        runs.append((start,min(t-start,max_prefix)))
    device=next(former.parameters()).device
    prompt=former.build_prompt_cache(batch_size)
    values,indices,starts,ages,latencies=[],[],[],[],[]
    autocast=torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else nullcontext()
    with autocast:
        for offset in range(0,len(runs),batch_size):
            group=runs[offset:offset+batch_size]
            state=former.initial_state(batch_size)
            for age in range(max_prefix):
                active=torch.tensor([age < length for _,length in group]+[False]*(batch_size-len(group)),device=device)
                radar=torch.zeros(batch_size,*tokens.shape[1:],device=device)
                radar_mask=torch.zeros(batch_size,*mask.shape[1:],dtype=torch.bool,device=device)
                for row,(start,length) in enumerate(group):
                    source=start+min(age,length-1)
                    radar[row]=tokens[source].to(device); radar_mask[row]=mask[source].to(device)
                timed=len(latencies)<100
                if timed and device.type=='cuda': torch.cuda.synchronize()
                begin=time.perf_counter()
                updated=former.stream_update(state,radar,min((age+1)*.18/1.2,1.),radar_mask,prompt).event_tokens
                state=torch.where(active[:,None,None],updated,state.to(updated.dtype))
                if timed:
                    if device.type=='cuda': torch.cuda.synchronize()
                    latencies.append(time.perf_counter()-begin)
                host=state.float().cpu()
                for row,(start,length) in enumerate(group):
                    if age < length:
                        indices.append(start+age); starts.append(start); ages.append(age+1); values.append(host[row].clone())
    if not values: raise ValueError('no eligible event prefixes')
    order=np.argsort(indices)
    return dict(indices=torch.tensor(np.array(indices)[order]),events=torch.stack(values)[torch.from_numpy(order)],
                prefix_start=torch.tensor(np.array(starts)[order]),age=torch.tensor(np.array(ages)[order]),
                latency_seconds=latencies)


@torch.no_grad()
def graph_event_prefix_features(former,tokens,mask,episodes,valid,max_prefix):
    """Replay the original batch-one math, preserving recurrent order exactly."""
    device=next(former.parameters()).device
    if device.type!='cuda': return event_prefix_features(former,tokens,mask,episodes,valid,max_prefix)
    if not mask.bool().all(): raise ValueError('graph export requires complete valid four-token observations')
    prompt=former.build_prompt_cache(1)
    previous=former.initial_state(1).to(former.adapter.compute_dtype).clone()
    radar=tokens[0:1].to(device).clone()
    maturity=torch.tensor([.15],device=device)
    with torch.autocast('cuda',dtype=torch.float16):
        side=torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3): former.stream_update(previous,radar,maturity,None,prompt)
        torch.cuda.current_stream().wait_stream(side)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result=former.stream_update(previous,radar,maturity,None,prompt).event_tokens
    indices,values,starts,ages,latencies=[],[],[],[],[]
    last,age,start=None,0,0
    for t in range(len(tokens)):
        if not valid[t]: last,age=None,0; continue
        if last != int(episodes[t]):
            previous.copy_(former.initial_state(1))
            last,age,start=int(episodes[t]),0,t
        age+=1
        if age>max_prefix: continue
        radar.copy_(tokens[t:t+1]); maturity.fill_(min(age*.18/1.2,1.))
        timed=len(latencies)<100
        if timed: torch.cuda.synchronize()
        begin=time.perf_counter(); graph.replay()
        host=result.float().cpu()
        if timed: latencies.append(time.perf_counter()-begin)
        previous.copy_(result)
        indices.append(t); starts.append(start); ages.append(age); values.append(host[0].clone())
    if not values: raise ValueError('no eligible event prefixes')
    return dict(indices=torch.tensor(indices),events=torch.stack(values),prefix_start=torch.tensor(starts),
                age=torch.tensor(ages),latency_seconds=latencies)


@torch.no_grad()
def export_prediction_cache(frontend, manifest_path, rows, output_dir, config):
    output = Path(output_dir); output.mkdir(parents=True,exist_ok=False)
    before = state_digest(frontend)
    device = next(frontend.parameters()).device
    encoder = frontend.radar_encoder
    records, audit = [], []
    for row in rows:
        path = resolve_recording(manifest_path,row)
        source_sha = file_sha(path)
        with np.load(path,allow_pickle=False) as arrays:
            md = np.asarray(arrays['md'],dtype=np.float32)
            stamps = np.asarray(arrays['timestamps'],dtype=np.float64)
            labels = np.asarray(arrays['labels'],dtype=np.int64)
        if len(md) != len(stamps) or not np.isfinite(md).all():
            raise ValueError('invalid MD source')
        episodes = build_command_episodes(row,stamps)
        grid = update_grid(stamps,[e['start'] for e in episodes])
        frames = len(grid['intervals'])*2
        grid['valid'] &= (labels[:frames].reshape(-1,2) >= 0).all(1)
        state, tokens, masks = None, [], []
        sensor = config['sensor_names'].index(row['sensor_id'])
        def encode(value, initial):
            return encoder.forward_update(torch.empty(1,2,0,0,0,device=device),value,
                    sensor_index=torch.tensor([sensor],device=device),state=initial,
                    frame_mask=torch.ones(1,2,dtype=torch.bool,device=device))
        for i in range(0,frames,2):
            value = torch.from_numpy(md[i:i+2].copy())[None].to(device)
            encoded = encode(value,state)
            if i == 0:
                repeated = encode(value,None)
                torch.testing.assert_close(encoded.radar_tokens,repeated.radar_tokens,rtol=0,atol=0)
            state = encoded.state
            tokens.append(encoded.radar_tokens[0].float().cpu()); masks.append(encoded.radar_token_mask[0].cpu())
        tokens, masks = torch.stack(tokens), torch.stack(masks)
        event = graph_event_prefix_features(frontend.event_former,tokens,masks,grid['episode'],grid['valid'],config['max_prefix'])
        # Replay the initial visible prefix with no unseen event duration.
        stop = min(32,len(tokens))
        replay = event_prefix_features(frontend.event_former,tokens[:stop],masks[:stop],grid['episode'][:stop],grid['valid'][:stop],config['max_prefix'])
        torch.testing.assert_close(event['events'][:len(replay['events'])],replay['events'],rtol=0,atol=0)
        replay_error=float((event['events'][:len(replay['events'])]-replay['events']).abs().max())
        future = future_indices(grid['intervals'],grid['valid'],(1,))[1]
        eligible = future[event['indices'].numpy()] >= 0
        selected = event['indices'][torch.from_numpy(eligible)]
        if len(selected) == 0: raise ValueError('no valid future observations')
        uid = f"{row['subject_id']}-{row['session_id']}-{row['sensor_id']}"
        payload = dict(uid=uid, role=config['role'], metadata=row, radar=tokens, radar_mask=masks,
            motion=fixed_motion_target(tokens,masks),valid=torch.from_numpy(grid['valid']),
            intervals=torch.from_numpy(grid['intervals']),available_at=torch.from_numpy(grid['intervals'][:,1]),
            selected=selected, future=torch.from_numpy(future[selected.numpy()]),
            event=event['events'][torch.from_numpy(eligible)],prefix_start=event['prefix_start'][torch.from_numpy(eligible)],
            age=event['age'][torch.from_numpy(eligible)],source_sha256=source_sha,
            command_sha256=file_sha(row['source_commands']),trailing_frames=grid['trailing_frames'],
            supervision_source='command_boundary_proxy',formal_go=False)
        destination = output/(uid+'.pt'); torch.save(payload,destination)
        records.append(dict(uid=uid,path=str(destination.resolve()),sha256=file_sha(destination),role=config['role'],
                            subject_id=row['subject_id'],session_id=row['session_id'],sensor_id=row['sensor_id']))
        audit.append(dict(uid=uid, updates=len(tokens), samples=len(selected),trailing_frames=grid['trailing_frames'],
                          event_graph_replay_p95=float(np.quantile(event['latency_seconds'],.95)),event_batch_size=1,
                          replay_max_abs_error=replay_error,zero_step_replay=True))
        print(f'cache {uid}: {len(selected)} prediction samples',flush=True)
    after = state_digest(frontend)
    if after != before: raise RuntimeError('frozen frontend changed during cache export')
    index = dict(schema_version=26,records=records,audit=audit,frontend_sha256=before,frozen_unchanged=True,
                 source_manifest_sha256=file_sha(manifest_path),target_kind='fixed_masked_mean',formal_go=False)
    write_json(output/'index.json',index)
    return index

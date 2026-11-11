"""Read-only full-state magnitude/information-change diagnosis and frozen Gate probes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .long_prediction import state_dynamics
from .source import file_sha,require_sha,write_json,load_frozen_frontend,state_digest,source_digest


def describe(arrays,positions):
    result=dict(updates=len(positions))
    for key in ('rms','step_delta_rms','adjacent_cosine','top_coordinate_energy'):
        value=arrays[key][positions];value=value[np.isfinite(value)]
        result[key]=dict(mean=float(value.mean()),min=float(value.min()),max=float(value.max())) if len(value) else None
    return result


@torch.no_grad()
def diagnose(config,root,oldlock):
    replay=Path(config['replay_root']);manifest=json.loads((replay/'state-manifest.json').read_text())
    output=root/'state-diagnosis';output.mkdir(exist_ok=False);rows=[];gate_cases=[]
    for entry in manifest['records']:
        require_sha(entry['path'],entry['sha256']);record=torch.load(entry['path'],map_location='cpu',weights_only=False)
        event=record['events'];arrays=state_dynamics(event);onset=arrays.pop('first_sustained_high');n=len(event)
        anchor=n if onset is None else onset
        before=np.arange(max(1,anchor-32),anchor);after=np.arange(onset,min(n,onset+32)) if onset is not None else np.array([],int)
        tail=np.arange(max(1,n-32),n);probe=np.unique(np.r_[before,after,tail])
        path=output/f"{entry['role']}-{entry['uid']}-{entry['view']}.npz"
        np.savez_compressed(path,**arrays,probe_indices=probe,probe_events=event[probe].numpy(),age=record['plan']['age'])
        row=dict(uid=entry['uid'],role=entry['role'],view=entry['view'],first_sustained_high=onset,
            before=describe(arrays,before),after=describe(arrays,after),tail=describe(arrays,tail),
            full=describe(arrays,np.arange(1,n)),source_sha256=entry['sha256'],output_sha256=file_sha(path))
        rows.append(row)
        # Keep fixed first four NoReset plateau cases and two normal controls; no outcome-based prediction selection.
        if entry['view']=='no_reset':
            positives=sum(c['plateau'] for c in gate_cases);negatives=sum(not c['plateau'] for c in gate_cases)
            if (onset is not None and positives<4) or (onset is None and negatives<2):
                times=np.unique(np.r_[np.arange(max(1,anchor-4),min(n,anchor+4)),np.arange(n-4,n)])
                gate_cases.append(dict(entry=entry,times=times,plateau=onset is not None))
        print('state diagnosis '+entry['uid']+' '+entry['view']+' onset='+str(onset),flush=True)
    frontend,_,report=load_frozen_frontend(oldlock['frontend_assets']['config_path'],config['stage1'],config['stage2'],oldlock['checkpoints'],config['device'])
    before_hash=state_digest(frontend);gate_rows=[];former=frontend.event_former;device=config['device']
    if before_hash!=config['frontend_sha256']:raise RuntimeError('diagnostic frontend identity differs')
    for case in gate_cases:
        entry=case['entry'];state=torch.load(entry['path'],map_location='cpu',weights_only=False)
        original=torch.load(Path(config['original_root'])/f"cache-{entry['role']}/{entry['uid']}.pt",map_location='cpu',weights_only=False)
        prompt=former.build_prompt_cache(1)
        for t in case['times']:
            previous=former.initial_state(1) if state['plan']['reset'][t] else state['events'][t-1:t].to(device)
            traces=[]
            with torch.autocast('cuda',dtype=torch.float16):
                value=former.stream_update(previous,original['radar'][t:t+1].to(device),
                    min(int(state['plan']['age'][t])*.18/1.2,1.),original['radar_mask'][t:t+1].to(device),prompt,trace=traces.append)
            torch.testing.assert_close(value.event_tokens[0].cpu(),state['events'][t],rtol=0,atol=0)
            trace=traces[0];alpha=trace['alpha'].float().cpu();proposal=trace['E_hat'].float().cpu();events=trace['E'].float().cpu()
            gate_rows.append(dict(uid=entry['uid'],role=entry['role'],update=int(t),plateau=case['plateau'],
                alpha_mean=float(alpha.mean()),alpha_min=float(alpha.min()),alpha_max=float(alpha.max()),
                alpha_lt01=float((alpha<.01).float().mean()),alpha_gt99=float((alpha>.99).float().mean()),
                proposal_rms=float(proposal.square().mean().sqrt()),event_rms=float(events.square().mean().sqrt()),state_exact=True))
    if state_digest(frontend)!=before_hash:raise RuntimeError('frozen model changed during diagnosis')
    write_json(output/'summary.json',dict(records=rows,gate_probes=gate_rows,frontend_unchanged=True,
        frontend_sha256=before_hash,optimizer_steps=0,descriptive_threshold=3.,sustained_updates=32,
        code_sha256=source_digest(),config_sha256=json.loads((root/'protocol.json').read_text())['config_sha256'],
        interpretation='magnitude and change diagnostics only; not evidence of semantic collapse or Reset gate'))
    return dict(status='diagnosed',state_files=len(rows),gate_probes=len(gate_rows),optimizer_steps=0)

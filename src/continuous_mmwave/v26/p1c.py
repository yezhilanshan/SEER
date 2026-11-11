"""P1-C preparation only: immutable complete replay and untrained diagnostics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from .p1 import _load_original, _commands, _digest_array
from .replay import replay_plan, replay_events, diagnostic_indices, linear_trend
from .source import file_sha, require_sha, write_json, state_digest, source_digest, load_frozen_frontend


def perturb_future_radar(tokens,cut):
    if not 0<cut<=len(tokens):raise ValueError('observed prefix required')
    result=tokens.clone()
    for t in range(cut,len(tokens)):result[t]=tokens[(t-cut)%cut]
    return result


@torch.no_grad()
def sequential_check(former, record, plan, result):
    """Initial independent reference + conditional long-tail stream_update reference."""
    device=next(former.parameters()).device;prompt=former.build_prompt_cache(1)
    positions=result['indices'].numpy();lookup={int(t):i for i,t in enumerate(positions)}
    runs=[('initial_independent',positions[:min(64,len(positions))]),
          ('tail_conditional',positions[max(0,len(positions)-32):])]
    report={}
    with torch.autocast('cuda',dtype=torch.float16):
        for name,indices in runs:
            state=None;error=0.
            for j,t in enumerate(indices):
                if plan['reset'][t]:state=former.initial_state(1)
                elif j==0:
                    # This is deliberately declared conditional, not a whole-record independent replay.
                    state=result['events'][lookup[int(t)-1]][None].to(device)
                state=former.stream_update(state,record['radar'][t:t+1].to(device),
                    min(int(plan['age'][t])*.18/1.2,1.),record['radar_mask'][t:t+1].to(device),prompt).event_tokens
                expected=result['events'][lookup[int(t)]]
                torch.testing.assert_close(state[0].cpu(),expected,atol=0,rtol=0)
                error=max(error,float((state[0].float().cpu()-expected.float()).abs().max()))
            report[name]=dict(updates=len(indices),max_abs_error=error)
    return report


def baseline_metrics(record,indices,stats):
    motion=record['motion'];t=indices['common_start'];rows={};arrays={}
    for h in (1,3,5):
        future=indices[f'future_h{h}'][t];target=motion[future]
        masks=dict(all=np.ones(len(t),bool),deadline_cross=indices[f'deadline_cross_h{h}'][t],
            mixed_current=indices[f'mixed_current_h{h}'][t],mixed_future=indices[f'mixed_future_h{h}'][t])
        masks['boundary_support']=masks['deadline_cross']|masks['mixed_current']|masks['mixed_future']
        masks['clean_interior']=~masks['boundary_support']
        rows[str(h)]={}
        for name,mean in [('persistence',motion[t]),('linear_trend',linear_trend(motion,t,h))]:
            mse=((mean-target)/stats['std']).double().square().mean(-1).numpy()
            arrays[f'{name}_mse_h{h}']=mse
            rows[str(h)][name]={key:dict(samples=int(mask.sum()),mse=float(mse[mask].mean()) if mask.any() else None)
                               for key,mask in masks.items()}
    return rows,arrays


def run(config_path):
    config_path=Path(config_path);config=yaml.safe_load(config_path.read_text())
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    original,oldlock=_load_original(config);out=Path(config['output_dir'])
    out.mkdir(parents=True,exist_ok=False);(out/'states').mkdir();(out/'compact').mkdir()
    protected=dict(json.loads(Path(config['p0_snapshot']).read_text())['protected_before'])
    # P1-B outputs are inputs only to provenance; protect their checkpoints and evidence too.
    p1=original.parent/'v26_p1_20261015_v1'
    for path in p1.rglob('*'):
        if path.is_file() and path.name!='evidence.tar.gz':protected[str(path)]=file_sha(path)
    write_json(out/'protocol.json',dict(config_sha256=file_sha(config_path),code_sha256=source_digest(),
        protected=protected,training_executed=False,test_arrays_read=False,formal_go=False,
        oracle_reset='first_pure_update_after_changed_observed_proxy',mixed='continue_previous_state',
        no_reset='one_initialization_per_contiguous_observation_segment',source='complete_frozen_v4_radar'))
    frontend,_,front_report=load_frozen_frontend(oldlock['frontend_assets']['config_path'],config['stage1'],
        config['stage2'],oldlock['checkpoints'],config['device'])
    write_json(out/'frontend.json',front_report);before=state_digest(frontend)
    stats=torch.load(original/'normalization.pt',map_location='cpu',weights_only=False)
    np.savez_compressed(out/'compact/normalization.npz',mean=stats['mean'].numpy(),std=stats['std'].numpy())
    audits=[];files=[]
    for role in ('train','validation'):
        source_index=json.loads((original/f'cache-{role}/index.json').read_text())
        for entry in source_index['records']:
            require_sha(entry['path'],entry['sha256'])
            record=torch.load(entry['path'],map_location='cpu',weights_only=False);uid=record['uid']
            starts=_commands(record);intervals=record['intervals'].numpy()
            observed=(record['radar_mask'].all(-1)&torch.isfinite(record['radar']).all((1,2))&
                      torch.isfinite(record['motion']).all(-1)).numpy()
            plans={v:replay_plan(intervals,observed,starts,v) for v in ('oracle','no_reset')}
            views={};view_reports={};compact=dict(intervals=intervals,command_starts=starts,
                motion=record['motion'].numpy(),old_selected=record['selected'].numpy(),observation_valid=observed)
            for view,plan in plans.items():
                print(f'replay {role} {uid} {view}: {int(observed.sum())} updates',flush=True)
                result=replay_events(frontend.event_former,record['radar'],record['radar_mask'],plan)
                assert np.array_equal(result['indices'].numpy(),np.flatnonzero(plan['valid']))
                full_lookup=torch.full((len(intervals),),-1,dtype=torch.long)
                full_lookup[result['indices']]=torch.arange(len(result['indices']))
                selected=record['selected'];old=result['events'][full_lookup[selected]].float()
                if view=='oracle':
                    torch.testing.assert_close(old,record['event'],atol=0,rtol=0)
                    legacy=dict(compared=len(selected),max_abs_error=float((old-record['event']).abs().max()))
                else:
                    # Compare to old Oracle only while both reset histories coincide.
                    differing=np.flatnonzero(plans['oracle']['reset'] != plan['reset'])
                    stop=int(differing[0]) if len(differing) else len(intervals)
                    shared=selected<stop
                    torch.testing.assert_close(old[shared],record['event'][shared],atol=0,rtol=0)
                    legacy=dict(shared_prefix_compared=int(shared.sum()),shared_prefix_max_abs_error=0.,
                        later_difference_expected=True,later_max_abs_difference=float((old[~shared]-record['event'][~shared]).abs().max()) if (~shared).any() else None)
                reference=sequential_check(frontend.event_former,record,plan,result)
                # Cut just after first mixed update; state must not use unseen command/radar suffixes.
                from .targets import prediction_contract
                c=prediction_contract(intervals,observed,starts,(1,3,5))
                mixed=np.flatnonzero(c['mixed_update']);cut=min(len(intervals),int(mixed[0])+3 if len(mixed) else 64)
                cutoff=intervals[cut-1,1]
                changed=np.concatenate((starts[starts<=cutoff],np.array([cutoff+10.,cutoff+20.])))
                changed_plan=replay_plan(intervals,observed,changed,view)
                for key in ('valid','reset','age','segment_start'):
                    np.testing.assert_array_equal(plan[key][:cut],changed_plan[key][:cut])
                altered=perturb_future_radar(record['radar'][:cut+3],cut)
                # The extra three changed future radar updates are actually run, not merely asserted unused.
                suffix_stop=len(altered)
                short_plan=replay_plan(intervals[:suffix_stop],observed[:suffix_stop],changed,view)
                perturbed=replay_events(frontend.event_former,altered,record['radar_mask'][:suffix_stop],short_plan)
                count=int(observed[:cut].sum())
                torch.testing.assert_close(perturbed['events'][:count],result['events'][:count],atol=0,rtol=0)
                truncated_plan=replay_plan(intervals[:cut],observed[:cut],starts[starts<=cutoff],view)
                truncated=replay_events(frontend.event_former,record['radar'][:cut],record['radar_mask'][:cut],truncated_plan)
                torch.testing.assert_close(truncated['events'],result['events'][:count],atol=0,rtol=0)
                if view=='no_reset':
                    arbitrary=replay_plan(intervals,observed,[0.],view)
                    for key in ('reset','age','segment_start'):np.testing.assert_array_equal(plan[key],arbitrary[key])
                indices=diagnostic_indices(intervals,observed,starts,plan)
                for key,value in {**plan,**indices}.items():
                    if key!='view':compact[f'{view}_{key}']=value
                norms=result['events'].float().square().mean((1,2)).sqrt().numpy()
                compact[f'{view}_state_rms']=norms
                probe=np.unique(np.r_[np.arange(min(40,len(intervals))),mixed[:10],record['selected'].numpy()[::max(1,len(selected)//24)],
                                     np.arange(max(0,len(intervals)-32),len(intervals))])
                probe=probe[plan['valid'][probe]]
                compact[f'{view}_probe_indices']=probe
                compact[f'{view}_probe_events']=result['events'][full_lookup[probe]].numpy()
                path=out/'states'/f'{role}-{uid}-{view}.pt'
                torch.save(dict(**result,uid=uid,role=role,view=view,plan=plan,original_cache_sha256=entry['sha256'],
                    radar_sha256=_digest_array(record['radar']),formal_go=False),path)
                files.append(dict(path=str(path),sha256=file_sha(path),bytes=path.stat().st_size,role=role,uid=uid,view=view))
                view_reports[view]=dict(updates=len(result['indices']),reset_count=int(plan['reset'].sum()),
                    max_age=int(plan['age'].max()),dtype=str(result['events'].dtype),finite=True,
                    state_rms_quantiles=np.quantile(norms,[0,.5,.95,1]).tolist(),legacy=legacy,reference=reference,
                    future_command_radar_prefix_exact=True,cutoff_update=cut-1,
                    common_starts=len(indices['common_start']),ablation_common_starts=len(indices['ablation_common_start']),
                    mixed_covered=int((c['mixed_update']&observed).sum()))
                views[view]=result
                print(f'verified {uid} {view}: legacy={legacy}, age_max={view_reports[view]["max_age"]}',flush=True)
            primary=diagnostic_indices(intervals,observed,starts,plans['oracle'])
            baseline,baseline_arrays=baseline_metrics(record,primary,stats);compact.update(baseline_arrays)
            destination=out/'compact'/f'{role}-{uid}.npz';np.savez_compressed(destination,**compact)
            audits.append(dict(uid=uid,role=role,subject_id=record['metadata']['subject_id'],sensor_id=record['metadata']['sensor_id'],
                total_updates=len(intervals),invalid_observations=int((~observed).sum()),views=view_reports,baselines=baseline))
            write_json(out/'audit-progress.json',audits)
            del views,record
    after=state_digest(frontend)
    if after!=before:raise RuntimeError('frozen frontend changed')
    for path,sha in protected.items():require_sha(path,sha)
    write_json(out/'state-manifest.json',dict(records=files,frontend_sha256=before))
    write_json(out/'verification.json',dict(status='verified',recordings=audits,frontend_unchanged=True,
        protected_unchanged=True,training_executed=False,optimizer_steps=0,test_arrays_read=False,formal_go=False))
    print('COMPLETE: full frozen replay verified; optimizer_steps=0',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    run(parser.parse_args().config)

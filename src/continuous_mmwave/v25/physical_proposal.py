"""Stage3A physical proposals; deliberately no Candidate classifier or boundary locator."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from .checkpoint import file_sha, module_digest, state_digest, code_digest
from .prepare import write_json


def proposal_labels(labels, timestamps):
    labels=np.asarray(labels); timestamps=np.asarray(timestamps,dtype=np.float64)
    if (labels.ndim!=2 or labels.shape[1]!=2 or timestamps.shape!=labels.shape
        or not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps.ravel())>0)):
        raise ValueError('chronological two-frame updates required')
    known=labels.ravel(); frame_time=timestamps.ravel()
    boundaries=frame_time[1:][(known[1:]!=known[:-1])&(known[1:]>=0)&(known[:-1]>=0)]
    time=timestamps[:,-1]
    target=np.exp(-.5*((time[:,None]-boundaries[None,:])/.36)**2).max(1) if len(boundaries) else np.zeros(len(time))
    valid=(labels>=0).all(1)
    valid[1:] &= valid[:-1].copy(); valid[0]=False
    return dict(time=time,boundaries=boundaries,target=target.astype(np.float32),valid=valid)


def fit_mass_weights(rows):
    positive=sum(float(np.asarray(target)[valid].sum(dtype=np.float64)) for target,valid in rows)
    total=sum(int(np.asarray(valid).sum()) for _,valid in rows)
    if not 0<positive<total:
        raise ValueError('fit needs positive and stable Gaussian mass')
    return total/(2*positive),total/(2*(total-positive))


def proposal_metrics(time, score, valid, boundaries, *, threshold=.6, persistence=2, tolerance=.72):
    time=np.asarray(time,dtype=np.float64);score=np.asarray(score);valid=np.asarray(valid,dtype=bool)
    boundaries=np.asarray(boundaries,dtype=np.float64)
    if (time.ndim!=1 or not len(time) or score.shape!=time.shape or valid.shape!=time.shape
        or not np.isfinite(time).all() or not np.isfinite(score).all() or not np.isfinite(boundaries).all()
        or not np.all(np.diff(time)>0) or ((score<0)|(score>1)).any()):
        raise ValueError('finite chronological Proposal stream required')
    # Nominal exposure for first/disconnected scored update; actual intervals
    # only when both adjacent updates are scored and cadence is contiguous.
    delta=np.diff(time);period=float(np.median(delta)) if len(delta) else .18
    contiguous=np.r_[False,delta<=1.5*period]
    duration=float(sum((time[i]-time[i-1] if i and valid[i-1] and contiguous[i] else period)
                       for i in range(len(time)) if valid[i]))
    proposed=[];streak=0;armed=True
    for i,(t,s,ok) in enumerate(zip(time,score,valid)):
        if not contiguous[i]:
            streak=0;armed=True
        if not ok or s<threshold:
            streak=0;armed=True;continue
        streak+=1
        if streak>=persistence and armed:
            proposed.append(float(t));armed=False
    eligible=[float(b) for b in boundaries if np.any(valid&(np.abs(time-b)<=tolerance+1e-9))]
    used=set();matches=[]
    for t in proposed:
        candidates=[j for j,b in enumerate(eligible) if j not in used and abs(t-b)<=tolerance+1e-9]
        if candidates:
            j=candidates[0];used.add(j);matches.append(dict(proposal=t,boundary=eligible[j],latency_s=t-eligible[j]))
    # Independent post-boundary metric prevents an early alarm hiding a miss.
    post_used=set()
    for t in proposed:
        candidates=[j for j,b in enumerate(eligible) if j not in post_used and -1e-9<=t-b<=tolerance+1e-9]
        if candidates: post_used.add(candidates[0])
    latency=[r['latency_s'] for r in matches]; false=len(proposed)-len(matches)
    return dict(threshold=threshold,persistence=persistence,tolerance_s=tolerance,proposal_times=proposed,
        matches=matches,gt_boundaries=len(boundaries),eligible_boundaries=len(eligible),
        excluded_boundaries=len(boundaries)-len(eligible),matched=len(matches),missed=len(eligible)-len(matches),
        proposals=len(proposed),false_proposals=false,duration_s=duration,
        eligible_updates=int(valid.sum()),high_updates=int((valid&(score>=threshold)).sum()),
        recall=None if not eligible else len(matches)/len(eligible),
        post_boundary_recall=None if not eligible else len(post_used)/len(eligible),post_matched=len(post_used),
        false_per_minute=None if duration==0 else false*60/duration,
        density_per_minute=None if duration==0 else len(proposed)*60/duration,
        high_update_fraction=None if not valid.any() else float((score[valid]>=threshold).mean()),
        latency_median_s=None if not latency else float(np.median(latency)),
        latency_p95_s=None if not latency else float(np.percentile(latency,95)),
        early_fraction=None if not latency else float(np.mean(np.asarray(latency)<-1e-9)))


class ProposalBatches:
    """Fit-only continuous TBPTT streams, balanced across subject/sensor cells."""
    def __init__(self,index,*,device):
        self.device=device;self.records={};self.cells=defaultdict(list);self.turn=0
        self.cursors={};self.states={};self.draws=Counter();self.valid_draws=Counter();self.record_draws=Counter()
        for row in index['records']:
            if row['role']!='fit':continue
            if file_sha(row['path'])!=row['sha256']:raise ValueError('encoder source SHA changed')
            record=torch.load(row['path'],map_location='cpu',weights_only=False)
            labels=proposal_labels(record['labels'].numpy(),record['timestamps'].numpy())
            masks=record['radar_mask'].bool().numpy()
            labels['valid'][1:] &= (masks[1:]&masks[:-1]).any(1)
            if not labels['valid'].any():raise ValueError('no eligible fit transitions')
            record.update(proposal=labels);self.records[row['uid']]=record
            self.cells[(row['metadata']['subject_id'],row['metadata']['sensor_id'])].append(row['uid'])
        self.cell_keys=sorted(self.cells)
        self.positive_weight,self.negative_weight=fit_mass_weights(
            [(r['proposal']['target'],r['proposal']['valid']) for r in self.records.values()])
        self.last=[]

    def __call__(self,stage):
        losses=[];self.last=[]
        for _ in range(16):
            turn=self.turn;self.turn+=1;cell=self.cell_keys[turn%len(self.cell_keys)]
            sources=self.cells[cell];uid=sources[(turn//len(self.cell_keys))%len(sources)]
            row=self.records[uid];start=self.cursors.get(uid,0);end=min(start+64,len(row['radar']))
            target=row['proposal']['target'][start:end];valid=row['proposal']['valid'][start:end]
            batch=dict(tokens=row['radar'][start:end][None].to(self.device),mask=row['radar_mask'][start:end][None].to(self.device),
                target=torch.from_numpy(target)[None].to(self.device),valid=torch.from_numpy(valid)[None].to(self.device),
                class_weights=(self.positive_weight,self.negative_weight))
            if start:
                batch.update(initial_state=self.states[uid],previous_tokens=row['radar'][start-1:start].to(self.device),
                    previous_mask=row['radar_mask'][start-1:start].to(self.device))
            result=stage.physical_stream_step(batch)
            losses.append(result['loss']);self.states[uid]=result['next_state']
            self.cursors[uid]=0 if end==len(row['radar']) else end
            self.draws[cell]+=1;self.valid_draws[cell]+=result['valid_transitions'];self.record_draws[uid]+=1
            self.last.append(dict(uid=uid,start=start,end=end,valid_updates=result['valid_transitions']))
        return torch.stack(losses).mean()

    def coverage(self):
        return dict(chunks=self.turn,positive_weight=self.positive_weight,negative_weight=self.negative_weight,
            cells=[dict(subject_id=s,sensor_id=c,chunks=self.draws[(s,c)],valid_updates=self.valid_draws[(s,c)]) for s,c in self.cell_keys],
            record_draws=dict(self.record_draws))


def aggregate_metrics(rows):
    fields=('gt_boundaries','eligible_boundaries','excluded_boundaries','matched','missed','proposals',
            'false_proposals','duration_s','eligible_updates','high_updates','post_matched')
    result={key:sum(r[key] for r in rows) for key in fields}
    n=result['eligible_boundaries'];d=result['duration_s'];u=result['eligible_updates']
    latency=[m['latency_s'] for r in rows for m in r['matches']]
    result.update(recall=None if not n else result['matched']/n,post_boundary_recall=None if not n else result['post_matched']/n,
        false_per_minute=None if not d else result['false_proposals']*60/d,
        density_per_minute=None if not d else result['proposals']*60/d,
        high_update_fraction=None if not u else result['high_updates']/u,
        latency_median_s=None if not latency else float(np.median(latency)),
        latency_p95_s=None if not latency else float(np.percentile(latency,95)),
        early_fraction=None if not latency else float(np.mean(np.asarray(latency)<-1e-9)))
    return result


@torch.inference_mode()
def evaluate_proposals(stage,index,output,step):
    output=Path(output);stage.eval();device=next(stage.parameters()).device;raw=[];arrays={}
    for number,row in enumerate(r for r in index['records'] if r['role']=='dev'):
        if file_sha(row['path'])!=row['sha256']:raise ValueError('dev source SHA changed')
        record=torch.load(row['path'],map_location='cpu',weights_only=False)
        label=proposal_labels(record['labels'].numpy(),record['timestamps'].numpy())
        radar=record['radar'].to(device);mask=record['radar_mask'].to(device);state=None;local=[];transition=[]
        for i in range(len(radar)):
            value=0.
            if i:
                common=mask[i-1:i]&mask[i:i+1]
                if common.any():value=float(stage.heads.local(radar[i-1:i],radar[i:i+1],common).sigmoid()[0])
                else:label['valid'][i]=False
            r,state=stage.heads.radar(radar[i:i+1],mask[i:i+1],state,stage.update_period_s)
            local.append(value);transition.append(float(r.sigmoid()[0]))
        l=np.array(local);r=np.array(transition);f=1-(1-l)*(1-r)
        prefix=f'r{number:02d}'
        arrays.update({prefix+'_'+k:v for k,v in {**label,'local':l,'radar':r,'fusion':f}.items()})
        raw.append(dict(uid=row['uid'],subject_id=row['metadata']['subject_id'],sensor_id=row['metadata']['sensor_id'],
            array_prefix=prefix,paths={path:{str(t):proposal_metrics(label['time'],score,label['valid'],label['boundaries'],threshold=t)
                for t in (.3,.4,.5,.6,.7,.8,.9)} for path,score in (('local',l),('radar',r),('fusion',f))}))
    if not raw:raise ValueError('missing development Proposal streams')
    groups={'all':raw}
    for subject in sorted({r['subject_id'] for r in raw}):groups['subject/'+subject]=[r for r in raw if r['subject_id']==subject]
    for sensor in sorted({r['sensor_id'] for r in raw}):groups['sensor/'+sensor]=[r for r in raw if r['sensor_id']==sensor]
    for subject,sensor in sorted({(r['subject_id'],r['sensor_id']) for r in raw}):
        groups[f'subject_sensor/{subject}/{sensor}']=[r for r in raw if (r['subject_id'],r['sensor_id'])==(subject,sensor)]
    scores={name:{path:{str(t):aggregate_metrics([r['paths'][path][str(t)] for r in rows])
                       for t in (.3,.4,.5,.6,.7,.8,.9)} for path in ('local','radar','fusion')} for name,rows in groups.items()}
    archive=output/f'proposal-step{step}-arrays.npz';np.savez_compressed(archive,**arrays)
    report=dict(stage='Stage3A-Physical-Proposal-Learning',step=step,primary=dict(path='fusion',threshold=.6,persistence=2,tolerance_s=.72),
        detector='one-per-contiguous-high-excursion; no Candidate suppression or GT resets',
        annotation='known adjacent-frame class change',groups=scores,records=raw,arrays_sha256=file_sha(archive),
        formal_go=False,final_boundary_evaluation=False)
    write_json(output/f'proposal-step{step}.json',report)
    return report


def prior_code_digest(root):
    root=Path(root);folder=root/'src/continuous_mmwave/v25';digest=hashlib.sha256()
    for path in sorted(folder.glob('*.py')):digest.update(path.name.encode());digest.update(path.read_bytes())
    for name in ('models/event_former.py','models/semantic.py','models/transition.py','models/memory.py',
                 'parrad_memory.py','stages/stage2.py','stages/parrad_stage4.py','train.py'):
        digest.update(name.encode());digest.update((folder.parent/name).read_bytes())
    return digest.hexdigest()


def load_handoff(config):
    old=Path(config['prior_code']);manifest_path=old/'source-manifest.json'
    if file_sha(manifest_path)!=config['prior_manifest_sha256']:raise ValueError('prior source manifest changed')
    for name,digest in json.loads(manifest_path.read_text()).items():
        if file_sha(old/name)!=digest:raise ValueError('prior source file changed: '+name)
    path=config['evidence_checkpoint_path']
    if file_sha(path)!=config['evidence_checkpoint_sha256']:raise ValueError('Evidence500 checkpoint SHA mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (payload['phase']!='evidence' or payload['step']!=500
        or payload['optimizer_steps']!=dict(physical=0,semantic=0,evidence=500)
        or payload['code_digest']!=prior_code_digest(old)
        or payload['source']['architecture']!='v25-age-free-positive-coordinate-v1'
        or payload['source']['encoder_index_sha256']!=config['encoder_index_sha256']
        or not payload['source']['stop_gradient']):
        raise ValueError('explicit prior Evidence500 provenance mismatch')
    return payload


def main():
    import argparse
    import yaml
    from .compatible_evidence import validate_source_roles
    from .gate_diagnostic import write_status
    from .stage3 import V25Stage3
    from .train_heads import train_fixed_head
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();config=yaml.safe_load(Path(args.config).read_text());output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    status=dict(phase='source-check',stage='Stage3A-Physical-Proposal-Learning',actual_optimizer_steps=0,budget=1000,
                holdout_access=False,official_data_access=False,formal_go=False)
    def update(**kwargs):
        status.update(kwargs);write_status(output/'status.json',status);print(json.dumps(status),flush=True)
    try:
        if config['seed']!=42 or config['actual_optimizer_steps']!=1000 or not config['sealed_holdouts']:
            raise ValueError('locked Proposal1000 budget/seed required')
        index_path=Path(config['native_preparation'])/'encoder/index.json'
        if file_sha(index_path)!=config['encoder_index_sha256']:raise ValueError('encoder index changed')
        index=json.loads(index_path.read_text());validate_source_roles(index)
        prior=load_handoff(config)
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        stage=V25Stage3().to('cuda');stage.load_state_dict(prior['stage_state'],strict=True)
        stage.optimizer_steps=dict(prior['optimizer_steps']);stage.set_phase('physical')
        frozen={k:v.cpu().clone() for k,v in stage.state_dict().items() if not k.startswith(('heads.local.','heads.radar.'))}
        source=dict(config_sha256=file_sha(args.config),encoder_index_sha256=file_sha(index_path),
            evidence_checkpoint_sha256=config['evidence_checkpoint_sha256'],prior_manifest_sha256=config['prior_manifest_sha256'],
            prior_code_digest=prior['code_digest'],code_digest=code_digest(),prior_source=prior['source'],
            frontend_optimizer_steps=0,gate_optimizer_steps=0,holdout_access=False,official_data_access=False)
        write_json(output/'source.json',source)
        torch.save(dict(stage_state=stage.state_dict(),optimizer_steps=dict(stage.optimizer_steps),source=source),output/'physical-step0.pt')
        update(phase='fit-stream-preparation');batches=ProposalBatches(index,device='cuda')
        write_json(output/'sampling-step0.json',batches.coverage())
        update(phase='proposal-evaluation',evaluation_checkpoint=0)
        evaluate_proposals(stage,index,output,0)
        chunk_log=(output/'chunks.jsonl').open('x')
        def on_step(row):
            chunk_log.write(json.dumps(dict(step=row['step'],chunks=batches.last))+'\n');chunk_log.flush()
            write_status(output/'status.json',{**status,'phase':'proposal-training','actual_optimizer_steps':row['step']})
            if row['step'] in (100,500,1000):
                update(phase='proposal-evaluation',actual_optimizer_steps=row['step'],evaluation_checkpoint=row['step'])
                evaluate_proposals(stage,index,output,row['step']);stage.train()
                update(phase='proposal-training',actual_optimizer_steps=row['step'])
        update(phase='proposal-training')
        try:trained=train_fixed_head(stage,batches,phase='physical',steps=1000,output=output/'training',source=source,on_step=on_step)
        finally:chunk_log.close()
        if any(not torch.equal(stage.state_dict()[k].cpu(),v) for k,v in frozen.items()):raise RuntimeError('Evidence or semantic changed')
        write_json(output/'sampling.json',batches.coverage())
        if len(batches.cell_keys)!=10 or any(batches.draws[c]!=1600 for c in batches.cell_keys):raise RuntimeError('fit chunk balance failed')
        selected=trained['checkpoints'][-1];saved=torch.load(selected['path'],map_location='cpu',weights_only=False)
        if saved['optimizer_steps']!=dict(physical=1000,semantic=0,evidence=500) or state_digest(saved['stage_state'])!=module_digest(stage):
            raise RuntimeError('final saved head mismatch')
        write_json(output/'result-readback-checks-v1.json',dict(actual_optimizer_steps=1000,selected_checkpoint=selected,
            optimizer_steps=stage.optimizer_steps,source=source,frozen_nonphysical_heads_unchanged=True,
            frontend_not_loaded_or_updated=True,automatic_event_retrieval_not_evaluated=True,
            selected_by='fixed-final1000-no-dev-selection',formal_go=False,holdout_access=False,official_data_access=False,
            evaluation_checkpoints=[0,100,500,1000],next_action='report-only-no-automatic-next-training'))
        update(phase='completed',actual_optimizer_steps=1000)
    except BaseException as exc:
        if 'batches' in locals():write_json(output/'failure-sampling.json',batches.coverage())
        if 'stage' in locals():status['actual_optimizer_steps']=stage.optimizer_steps['physical']
        update(phase='failed',error=repr(exc));raise


if __name__=='__main__':main()

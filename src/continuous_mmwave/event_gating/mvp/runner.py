"""Fixed-budget natural-recording SAEF experiment; calibration never uses validation."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from ..data import write_json,sha256,locked_path
from .model import MVPConfig,ActionHead,EvidenceNet,rollout,training_feature_inputs
from .data import load_records,split_fit_calibration
from .metrics import evaluate_rollout
from .reporting import write_reports


def weight_digest(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_checkpoint(path,device='cpu'):
    saved=torch.load(path,map_location=device,weights_only=True)
    if saved.get('format')!='saef-mvp-v1': raise ValueError('not a SAEF-MVP checkpoint')
    return saved


def modules(saved,device):
    config=MVPConfig(**saved['config'])
    action=ActionHead(config).to(device).eval(); action.load_state_dict(saved['action'])
    local=EvidenceNet(config,False).to(device).eval(); local.load_state_dict(saved['local'])
    state=EvidenceNet(config,True).to(device).eval(); state.load_state_dict(saved['state'])
    return config,action,local,state


def train_head(head,records,steps,lr,seed,device,*,evidence=False,config=None,
               closed_parameters=None,action=None):
    if type(steps) is not int or steps<1: raise ValueError('positive integer update budget required')
    records=[r for r in records if np.any(r['boundary_mask'] if evidence else np.asarray(r['labels'])>=0)]
    if not records:
        raise ValueError('no boundary supervision' if evidence else 'no semantic supervision')
    rng=np.random.default_rng(seed); optimizer=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=.01)
    history=[]
    positive=sum(float((r['boundary_target']*r['boundary_mask']).sum()) for r in records)
    total=sum(int(r['boundary_mask'].sum()) for r in records)
    if evidence and (positive<=0 or total<=positive): raise ValueError('boundary supervision needs positive and negative support')
    weight=torch.tensor((total-positive)/max(positive,1e-8),device=device)
    for step in range(steps):
        record=records[int(rng.integers(len(records)))]
        x=torch.as_tensor(np.array(record['x'],copy=True),dtype=torch.float32,device=device)
        optimizer.zero_grad(set_to_none=True); head.train()
        if evidence:
            if closed_parameters is None:
                inputs=training_feature_inputs(x,config,boundaries=record['boundaries'])
            else:
                head.eval()
                result=rollout(x,action,head,config,**closed_parameters)
                inputs=training_feature_inputs(x,config,fire=result['trace']['fire'])
                head.train()
            mask=torch.as_tensor(record['boundary_mask'],device=device).bool()
            target=torch.as_tensor(record['boundary_target'],dtype=torch.float32,device=device)
            logits=head.input_logits(inputs).reshape(-1)
            if not mask.any(): raise ValueError('record has no supervised boundary updates')
            loss=F.binary_cross_entropy_with_logits(logits[mask],target[mask],pos_weight=weight)
        else:
            target=torch.as_tensor(record['labels'],device=device).long(); mask=target>=0
            if not mask.any(): raise ValueError('record has no semantic supervision')
            loss=F.cross_entropy(head(x)[mask],target[mask])
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite supervised loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
        optimizer.step(); history.append(float(loss.detach()))
    head.eval()
    return dict(successful_updates=steps,loss=history,optimizer=optimizer.state_dict(),
                eligible_files=[r['file'] for r in records])


def require_calibration_boundaries(records):
    # Every internal annotated endpoint has a scored adjacent frame.
    if not any(0 < instance[edge] < record['frames']
               for record in records for instance in record['instances'] for edge in ('start','stop')):
        raise ValueError('calibration requires annotated internal boundaries in training recordings')


def calibrate(records,action,evidence,config,mechanism,thresholds,taus,device):
    require_calibration_boundaries(records)
    choices=[]
    for tau in (taus if mechanism=='lif' else [1.]):
        for threshold in thresholds:
            scores=[]
            for record in records:
                x=torch.as_tensor(np.array(record['x'],copy=True),dtype=torch.float32,device=device)
                result=rollout(x,action,evidence,config,mechanism=mechanism,threshold=float(threshold),tau=float(tau))
                scores.append(evaluate_rollout(result,record,period=config.frame_period_s)['boundary']['0.5'])
            # Equal-record weighting prevents long recordings from defining the operating point.
            supported=[r for r in scores if r['tp']+r['fn']>0]
            f1=float(np.mean([r['f1'] for r in supported]))
            recall=float(np.mean([r['recall'] for r in supported]))
            fp=float(np.mean([r['false_fires_per_minute'] for r in scores if r['valid_duration_s']>0]))
            choices.append(dict(mechanism=mechanism,threshold=float(threshold),tau=float(tau),
                                f1=f1,recall=recall,fp_per_min=fp,
                                unscored_unknown=sum(r['unscored_unknown'] for r in scores)))
    best=max(choices,key=lambda r:(r['f1'],r['recall'],-r['fp_per_min'],-r['tau'],-r['threshold']))
    return {k:best[k] for k in ('mechanism','threshold','tau')},choices


def export_record(output,record,result,metrics):
    name=Path(record['file']).stem
    np.savez_compressed(output/(name+'-trace.npz'),**result['trace'])
    write_json(output/(name+'-events.json'),result['events'])
    write_json(output/(name+'-metrics.json'),metrics)
    # Self-contained SVG diagnostic: continuous signals and exact frame-aligned markers.
    tr=result['trace']; evidence=np.asarray(tr['evidence']).reshape(-1); voltage=np.asarray(tr['voltage']).reshape(-1)
    n=len(evidence); width=1000; height=260; scale=max(3.,float(voltage.max(initial=0)))
    def path(values):
        return ' '.join(f'{20+i*960/max(1,n-1):.2f},{220-float(v)*180/scale:.2f}' for i,v in enumerate(values))
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="260">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="20" y="18">Blue: evidence; orange: voltage; red: predicted fire; green: official boundary</text>',
           f'<polyline fill="none" stroke="blue" points="{path(evidence)}"/>',
           f'<polyline fill="none" stroke="orange" points="{path(voltage)}"/>']
    for points,color in [(np.flatnonzero(tr['fire']),'red'),(record['boundaries'],'green')]:
        for i in points:
            x=20+int(i)*960/max(1,n-1); lines.append(f'<line x1="{x}" x2="{x}" y1="30" y2="220" stroke="{color}" opacity=".5"/>')
    period=metrics['frame_period_s']
    lines.append(f'<text x="20" y="250">0 to {n*period:.2f} nominal seconds; no boundary backtracking</text></svg>')
    (output/(name+'-timeline.svg')).write_text('\n'.join(lines),encoding='utf-8')


def run_experiment(settings,records,output,device='cpu',synthetic_only=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    try:
        return _run(settings,records,output,device,synthetic_only)
    except Exception as error:
        write_json(output/'failed.json',dict(error=type(error).__name__,message=str(error)))
        raise


def _run(settings,records,output,device,synthetic_only):
    source_hashes=verify_provenance(records,synthetic_only)
    config=MVPConfig(**settings.get('mvp',{})); cfg=settings.get('training',{}); seed=int(cfg.get('seed',42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(int(cfg.get('threads',1)))
    fit,cal,val=split_fit_calibration(records)
    if not fit or not cal or not val: raise ValueError('fit/calibration/validation must all be nonempty')
    require_calibration_boundaries(cal)
    for r in records:
        if r['x'].shape!=(r['frames'],config.feature_dim): raise ValueError('feature dimensions disagree with MVP configuration')
        if not np.isclose(r.get('frame_period_s',config.frame_period_s),config.frame_period_s,rtol=0,atol=1e-12):
            raise ValueError('record time grid differs from MVP configuration')
    protocol=dict(settings=settings,config=asdict(config),fit_files=[r['file'] for r in fit],
                  calibration_files=[r['file'] for r in cal],validation_files=[r['file'] for r in val],
                  synthetic_only=synthetic_only,source_hashes=source_hashes,record_identity=record_identity(records,config),
                  selection='training-record calibration only; validation never selects checkpoints or parameters')
    write_json(output/'protocol.json',protocol)
    action=ActionHead(config).to(device)
    action_log=train_head(action,fit,int(cfg.get('action_steps',500)),float(cfg.get('lr',.001)),seed,device)
    action.requires_grad_(False).eval()
    semantic={}
    with torch.no_grad():
        for role,group in [('fit',fit),('calibration',cal),('validation',val)]:
            confusion=np.zeros((config.num_classes,config.num_classes),np.int64)
            for record in group:
                x=torch.as_tensor(np.array(record['x'],copy=True),dtype=torch.float32,device=device)
                pred=action(x).argmax(-1).cpu().numpy(); y=record['labels']; valid=y>=0
                np.add.at(confusion,(y[valid],pred[valid]),1)
            tp=confusion.diagonal(); den=confusion.sum(0)+confusion.sum(1)
            f1=np.divide(2*tp,den,out=np.zeros(config.num_classes),where=den>0)
            semantic[role]=dict(macro_f1=float(f1.mean()),support=int(confusion.sum()),confusion=confusion.tolist())
    write_json(output/'action-audit.json',semantic)
    local=EvidenceNet(config,False).to(device); state=EvidenceNet(config,True).to(device)
    state.load_state_dict(copy.deepcopy(local.state_dict()))
    local_log=train_head(local,fit,int(cfg.get('evidence_steps',500)),float(cfg.get('lr',.001)),seed+1,device,evidence=True,config=config)
    state_log=train_head(state,fit,int(cfg.get('evidence_steps',500)),float(cfg.get('lr',.001)),seed+1,device,evidence=True,config=config)
    calibration=settings.get('calibration',{}); thresholds=calibration.get('thresholds',[.25,.5,.75,1.,1.5,2.])
    taus=calibration.get('taus',[.5,1.,2.])
    if not thresholds or not taus or any(not np.isfinite(v) or v<=0 for v in thresholds+taus):
        raise ValueError('nonempty positive finite calibration grids required')
    parameters={}; grids={}
    for arm,mechanism,head in [('B0','direct',local),('B1','if',local),('B2','lif',local),('B3','lif',state)]:
        parameters[arm],grids[arm]=calibrate(cal,action,head,config,mechanism,thresholds,taus,device)
    closed_steps=int(cfg.get('closed_loop_steps',0))
    if closed_steps<0: raise ValueError('closed_loop_steps must be nonnegative')
    closed_logs={}
    if closed_steps:
        for name,head,arm in [('local',local,'B2'),('state',state,'B3')]:
            closed_logs[name]=train_head(head,fit,closed_steps,float(cfg.get('lr',.001)),seed+2,device,
                                         evidence=True,config=config,closed_parameters=parameters[arm],action=action)
        for arm,mechanism,head in [('B0','direct',local),('B1','if',local),('B2','lif',local),('B3','lif',state)]:
            parameters[arm],grids[arm]=calibrate(cal,action,head,config,mechanism,thresholds,taus,device)
    saved=dict(format='saef-mvp-v1',config=asdict(config),action=action.state_dict(),local=local.state_dict(),state=state.state_dict(),
               calibration=parameters,protocol=protocol,training=dict(action=action_log,local=local_log,state=state_log,closed=closed_logs))
    torch.save(saved,output/'model.pt'); write_json(output/'calibration.json',dict(parameters=parameters,grid=grids))
    restored=modules(load_checkpoint(output/'model.pt'),device)
    arms={}; replay_ok=True
    for arm in ('B0','B1','B2','B3'):
        folder=output/arm; folder.mkdir(); head=state if arm=='B3' else local; restored_head=restored[3] if arm=='B3' else restored[2]
        rows=[]
        for record in val:
            x=torch.as_tensor(np.array(record['x'],copy=True),dtype=torch.float32,device=device)
            result=rollout(x,action,head,config,**parameters[arm])
            replay=rollout(x,restored[1],restored_head,restored[0],**parameters[arm])
            for key in result['trace']:
                if not np.array_equal(result['trace'][key],replay['trace'][key]): replay_ok=False
            metrics=evaluate_rollout(result,record,period=config.frame_period_s)
            export_record(folder,record,result,metrics); rows.append(dict(file=record['file'],metrics=metrics))
        arms[arm]=dict(parameters=parameters[arm],evidence_sha256=weight_digest(head),records=rows)
    if not replay_ok: raise RuntimeError('checkpoint replay differs')
    # Oracle vs autonomous is diagnostic on fit only; oracle never appears in calibration/validation.
    diagnostic=[]
    for record in fit:
        x=torch.as_tensor(np.array(record['x'],copy=True),dtype=torch.float32,device=device)
        free=rollout(x,action,state,config,**parameters['B3'])
        oracle=rollout(x,action,state,config,teacher_boundaries=record['boundaries'],**parameters['B3'])
        diagnostic.append(dict(file=record['file'],mean_abs_evidence_gap=float(np.abs(free['trace']['evidence']-oracle['trace']['evidence']).mean())))
    write_json(output/'oracle-closed-loop-diagnostic.json',diagnostic)
    sensitivity={}
    for arm,head in [('B2',local),('B3',state)]:
        sensitivity[arm]=[]
        for tau in taus:
            selected=max([p for p in grids[arm] if p['tau']==tau],key=lambda p:(p['f1'],p['recall'],-p['fp_per_min'],-p['threshold']))
            values=[]
            for record in val:
                x=torch.as_tensor(np.array(record['x'],copy=True),dtype=torch.float32,device=device)
                r=rollout(x,action,head,config,mechanism='lif',threshold=selected['threshold'],tau=tau)
                values.append(evaluate_rollout(r,record,period=config.frame_period_s))
            sensitivity[arm].append(dict(tau=tau,threshold_selected_on_training_calibration=selected['threshold'],records=values))
    write_json(output/'tau-sensitivity.json',sensitivity)
    verify_provenance(records,synthetic_only)
    result=dict(arms=arms,synthetic_only=synthetic_only,real_harrad_training=not synthetic_only,
                checkpoint_replay_passed=True,fit_files=protocol['fit_files'],calibration_files=protocol['calibration_files'],
                validation_files=protocol['validation_files'],test_signal_values_read=False)
    write_reports(output,arms,synthetic_only)
    write_json(output/'complete.json',result)
    return result


def verify_provenance(records,synthetic_only):
    hashes={}
    for record in records:
        if synthetic_only:
            hashes[record['file']]=None
            continue
        p=record.get('provenance',{})
        if (p.get('feature_kind')!='continuous_hidden' or p.get('test_signal_values_read') is not False
                or p.get('native_grid')!='full_1_to_1' or not p.get('source_locks')):
            raise ValueError('verified full-stream feature provenance required')
        for lock in p['source_locks'].values(): locked_path(lock)
        locked_path(p['continuous_hidden'])
        hashes[record['file']]=p['continuous_hidden']['sha256']
    return hashes


def record_identity(records,config):
    metadata=[dict(file=r['file'],subject=r['subject'],role=r['role'],frames=r['frames'],
                   instances=r['instances'],frame_period_s=r.get('frame_period_s',config.frame_period_s))
              for r in sorted(records,key=lambda r:r['file'])]
    return hashlib.sha256(json.dumps(metadata,sort_keys=True,allow_nan=False).encode()).hexdigest()


def audit_records(records,settings):
    config=MVPConfig(**settings.get('mvp',{}))
    proof=verify_provenance(records,False)
    fit,cal,val=split_fit_calibration(records)
    require_calibration_boundaries(cal)
    rows=[]
    for r in records:
        period=r.get('frame_period_s',config.frame_period_s)
        if not np.isclose(period,config.frame_period_s,rtol=0,atol=1e-12):
            raise ValueError('record time grid differs from MVP configuration')
        if r['x'].shape!=(r['frames'],config.feature_dim):
            raise ValueError('feature dimensions disagree with MVP configuration')
        rows.append(dict(file=r['file'],subject=r['subject'],role=r['role'],
                         frames=r['frames'],feature_dim=r['x'].shape[1],
                         frame_period_s=period,duration_s=r['frames']*period,
                         first_source_frame=0,last_source_frame=r['frames']-1,
                         known_frames=int(r['known_frames'].sum()),
                         unknown_frames=int((~r['known_frames']).sum()),
                         supervised_boundary_frames=int(r['boundary_mask'].sum()),
                         boundaries=r['boundaries'],interval_convention='[start,stop)',
                         supervision=r.get('supervision'),
                         continuous_hidden_sha256=proof[r['file']]))
    return dict(recordings=len(records),frames=sum(r['frames'] for r in records),
                fit=[r['file'] for r in fit],calibration=[r['file'] for r in cal],validation=[r['file'] for r in val],
                records=rows,source_hashes=proof,source_locks=records[0]['provenance']['source_locks'],
                record_identity=record_identity(records,config),
                time_basis='nominal',test_signal_values_read=False)


def evaluate_checkpoint(checkpoint,records,output,device='cpu'):
    saved=load_checkpoint(checkpoint,device); proof=verify_provenance(records,saved['protocol']['synthetic_only'])
    if proof!=saved['protocol']['source_hashes']: raise ValueError('evaluation source provenance differs from training')
    config,action,local,state=modules(saved,device)
    if record_identity(records,config)!=saved['protocol'].get('record_identity'):
        raise ValueError('evaluation recording annotation/split/time identity differs from training')
    _,_,validation=split_fit_calibration(records)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    arms={}
    for arm in ('B0','B1','B2','B3'):
        head=state if arm=='B3' else local; folder=output/arm; folder.mkdir(); rows=[]
        for record in validation:
            x=torch.as_tensor(np.array(record['x'],copy=True),dtype=torch.float32,device=device)
            result=rollout(x,action,head,config,**saved['calibration'][arm])
            metrics=evaluate_rollout(result,record,period=config.frame_period_s)
            export_record(folder,record,result,metrics); rows.append(dict(file=record['file'],metrics=metrics))
        arms[arm]=dict(parameters=saved['calibration'][arm],evidence_sha256=weight_digest(head),records=rows)
    result=dict(arms=arms,checkpoint_sha256=sha256(checkpoint),optimizer_updates=0,
                synthetic_only=saved['protocol']['synthetic_only'],test_signal_values_read=False)
    write_reports(output,arms,saved['protocol']['synthetic_only'])
    verify_provenance(records,saved['protocol']['synthetic_only'])
    write_json(output/'complete.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description='HARrad SAEF-MVP natural continuous supervision')
    sub=parser.add_subparsers(dest='command',required=True)
    for cmd in ('audit','run','evaluate','smoke'):
        p=sub.add_parser(cmd); p.add_argument('--output-dir',required=True)
        if cmd!='smoke': p.add_argument('--config',required=True)
        if cmd=='evaluate': p.add_argument('--checkpoint',required=True)
        p.add_argument('--device',default='cpu')
    args=parser.parse_args(argv)
    if args.command=='smoke':
        rows=[]
        for i,(subject,role) in enumerate([('subject1','train'),('subject1','train'),('subject7','validation')]):
            x=np.zeros((32,4),np.float32); x[:16,0]=1; x[16:,1]=1
            labels=np.array([0]*12+[-100]*4+[1]*12+[-100]*4)
            points=[12,16,28]; target=np.exp(-.5*((np.arange(32)[:,None]-points)/3)**2).max(1).astype(np.float32)
            rows.append(dict(file=f'{subject}_{i}',subject=subject,role=role,frames=32,x=x,labels=labels,
                             boundary_target=target,boundary_mask=np.ones(32,bool),boundaries=points,known_frames=labels>=0,
                             instances=[dict(start=0,stop=12,label=0,instance_id=0),dict(start=16,stop=28,label=1,instance_id=1)]))
        settings=dict(mvp=dict(feature_dim=4,hidden_dim=8,num_classes=3),training=dict(action_steps=2,evidence_steps=2),
                      calibration=dict(thresholds=[.5,1.],taus=[1.]))
        result=run_experiment(settings,rows,args.output_dir,args.device,True)
    else:
        settings=json.loads(Path(args.config).read_text(encoding='utf-8-sig')); records=load_records(settings)
        if args.command=='audit':
            out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=False)
            result=audit_records(records,settings)
            write_json(out/'audit.json',result)
        elif args.command=='evaluate': result=evaluate_checkpoint(args.checkpoint,records,args.output_dir,args.device)
        else: result=run_experiment(settings,records,args.output_dir,args.device)
    print(json.dumps(dict(command=args.command,output_dir=args.output_dir,synthetic_only=result.get('synthetic_only',False))))
    return 0

"""Source-isolated stage training, replay, CLI and explicitly synthetic plumbing smoke."""
import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from .config import GateConfig
from .data import (load_prepared, locked_path, prepare, read_sequence, sha256, write_json)
from .metrics import score_recording, sequence_metrics
from .model import EventGate
from .streaming import EventStream


def _seed(seed, threads):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(checkpoint, device='cpu'):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('architecture') != 'harrad-event-gating-v1' or saved.get('format_version') != 1:
        raise ValueError('not an event-gating checkpoint; legacy model files are not interchangeable')
    model = EventGate(GateConfig(**saved['gate_config']), saved['mode']).to(device)
    model.load_state_dict(saved['model'], strict=True)
    model.controller.temperature = saved.get('runtime_temperature', model.config.temperature)
    model.eval()
    return model, saved


@torch.no_grad()
def predict_events(model, features, identity='recording', chunk_frames=64, minimum_segment_frames=1):
    stream = EventStream(model, identity, minimum_segment_frames)
    rows = []
    for start in range(0, len(features), chunk_frames):
        rows.extend(stream.push(features[start:start+chunk_frames]))
    rows.extend(stream.finish())
    return rows


def _summary(rows):
    keys = ('sequence_exact','count_mae','edit_distance','oversegmentation_ratio')
    if not rows:
        raise ValueError('no evaluable constructed sequences')
    support=sum(r['adjacent_same_class_instance_support'] for r in rows)
    recovered=sum(r['adjacent_same_class_instances_matched'] for r in rows)
    return dict(samples=len(rows), **{k:float(np.mean([r[k] for r in rows])) for k in keys},
                adjacent_same_class_instance_support=support,adjacent_same_class_instances_matched=recovered,
                adjacent_same_class_instance_recall=recovered/support if support else None)


def select_best_snapshot(saved):
    candidates=[r for r in (saved,saved.get('best_snapshot')) if r and r.get('metrics') is not None]
    if not candidates:
        return None
    def rank(row):
        return (row['metrics']['sequence_exact'],-row['metrics']['count_mae'],
                -row['decoder_minimum_frames'],-row['epoch'])
    return copy.deepcopy({k:v for k,v in max(candidates,key=rank).items() if k!='best_snapshot'})


@torch.no_grad()
def evaluate_sequences(model, payload, root, role='validation', minimum_segment_frames=1):
    model.eval()
    rows = []
    device = next(model.parameters()).device
    for row in payload['sequences']:
        if row['role'] != role:
            continue
        features, targets, _ = read_sequence(root, row, device)
        events = predict_events(model, features, row['sample_id'], minimum_segment_frames=minimum_segment_frames)
        predicted = [event['class_id'] for event in events]
        rows.append(dict(sample_id=row['sample_id'], subject=row['subject'], reference=targets,
                         predicted=predicted, **sequence_metrics(predicted, targets)))
    return _summary(rows), rows


def _check_data_model(payload, config):
    if (payload['feature_dim'] != config.feature_dim or payload['num_classes'] != config.num_classes
            or not math.isclose(payload['frame_period_s'], config.frame_period_s, rel_tol=1e-8)):
        raise ValueError('gate dimensions/time grid differ from prepared data')


def _teacher_target(teacher, features):
    events = predict_events(teacher, features)
    locations = [r['stop_frame'] for r in events[:-1]]
    times = torch.arange(len(features), device=features.device, dtype=features.dtype)
    target = torch.zeros_like(times)
    sigma = .25/teacher.config.frame_period_s
    for boundary in locations:
        target = torch.maximum(target, torch.exp(-.5*((times-boundary)/sigma).square()))
    return target, len(locations)


def train(settings, prepared, output, *, stage='cif', checkpoint=None, device='cpu', resume=False):
    if stage not in ('cif','lif-distill','lif','frame-baseline'):
        raise ValueError('unknown stage')
    if stage in ('lif-distill','lif') and checkpoint is None:
        raise ValueError(f'{stage} requires a preceding stage checkpoint')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _train(settings, Path(prepared), output, stage, checkpoint, device, resume)
    except Exception as error:
        write_json(output/'failed.json', dict(error=type(error).__name__,message=str(error),stage=stage))
        raise


def _train(settings, prepared, output, stage, checkpoint, device, resume):
    payload = load_prepared(prepared)
    actual_locks=payload.get('source_locks',{})
    if not payload.get('synthetic_only',False) and not actual_locks:
        raise ValueError('prepared data lacks front-end provenance')
    if settings.get('locks') is not None and any(settings['locks'].get(k,{}).get('sha256')!=v['sha256']
                                                for k,v in actual_locks.items()):
        raise ValueError('training configuration provenance differs from prepared front-end')
    settings=copy.deepcopy(settings)
    if actual_locks:
        settings['locks']=copy.deepcopy(actual_locks)
    config = GateConfig(**settings.get('gate',{}))
    _check_data_model(payload, config)
    cfg = settings.get('training',{})
    seed = int(cfg.get('seed',42))
    _seed(seed,int(cfg.get('threads',4)))
    mode = 'lif' if stage.startswith('lif') else stage
    model = EventGate(config,mode).to(device)
    prepared_sha = sha256(prepared)
    previous = None
    teacher = None
    if checkpoint:
        loaded, previous = load_model(checkpoint,device)
        if previous['gate_config'] != config.to_dict() or previous['prepared_sha256'] != prepared_sha:
            raise ValueError('checkpoint architecture or prepared-data provenance mismatch')
        if resume:
            if previous['stage'] != stage:
                raise ValueError('resume requires a checkpoint from the same stage')
        elif stage=='lif-distill' and previous['mode']!='cif':
            raise ValueError('lif-distill must initialize from CIF')
        elif stage=='lif' and previous['mode']!='lif':
            raise ValueError('lif must initialize from a distilled LIF checkpoint')
        elif stage in ('cif','frame-baseline') and previous['mode']!=mode:
            raise ValueError('incompatible initialization checkpoint')
        model.load_state_dict(previous['model'])
        if stage=='lif-distill':
            if resume:
                teacher, _ = load_model(previous['teacher_checkpoint'],device)
                if sha256(previous['teacher_checkpoint']) != previous['teacher_sha256']:
                    raise ValueError('CIF teacher checkpoint changed')
            else:
                teacher = loaded
            teacher.requires_grad_(False).eval()
    for name, parameter in model.named_parameters():
        if mode=='cif':
            parameter.requires_grad_(name.startswith(('driver.','classifier.')))
        elif mode=='frame-baseline':
            parameter.requires_grad_(name.startswith('classifier.'))
        elif stage=='lif-distill':
            parameter.requires_grad_(not name.startswith('controller.'))
    base_lr = float(cfg.get('lr',.001))
    controller = [p for n,p in model.named_parameters() if p.requires_grad and n.startswith('controller.')]
    regular = [p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('controller.')]
    groups = [dict(params=regular,lr=base_lr)]
    if controller:
        groups.append(dict(params=controller,lr=.1*base_lr))
    optimizer = torch.optim.AdamW(groups,weight_decay=float(cfg.get('weight_decay',.01)))
    epoch_key = {'cif':'cif_epochs','lif-distill':'lif_distill_epochs','lif':'lif_epochs','frame-baseline':'baseline_epochs'}[stage]
    epochs = int(cfg.get(epoch_key, 1))
    batch_size = int(cfg.get('batch_size',8))
    if epochs<1 or batch_size<1 or base_lr<=0:
        raise ValueError('positive training epochs, batch size and learning rate required')
    train_rows = [r for r in payload['sequences'] if r['role']=='train']
    if not train_rows:
        raise ValueError('empty training split')
    start_epoch, steps, exposures = 1,0,0
    best_rank, best_metrics, best_epoch, best_decoder = None,None,None,1
    best_saved = None
    history = []
    if resume:
        if previous is None:
            raise ValueError('resume requires a checkpoint')
        optimizer.load_state_dict(previous['optimizer'])
        start_epoch = previous['epoch']+1
        steps,exposures = previous['optimizer_steps'], previous['exposures']
        torch.set_rng_state(previous['torch_rng'])
        random.setstate(previous['python_rng'])
        if torch.cuda.is_available() and previous.get('cuda_rng'):
            torch.cuda.set_rng_state_all(previous['cuda_rng'])
        if start_epoch>epochs:
            raise ValueError('resume epoch budget is already exhausted')
        carried = select_best_snapshot(previous)
        if carried is not None:
            best_saved = carried
            best_metrics,best_epoch,best_decoder = carried['metrics'],carried['epoch'],carried['decoder_minimum_frames']
            best_rank = (best_metrics['sequence_exact'],-best_metrics['count_mae'],-best_decoder)
            _save(output/'best.pt',best_saved)
    teacher_path = (previous['teacher_checkpoint'] if resume else str(Path(checkpoint).resolve())) if teacher else None
    teacher_sha = sha256(teacher_path) if teacher_path else None
    source = dict(stage=stage,mode=mode,format_version=1,architecture='harrad-event-gating-v1',gate_config=config.to_dict(),
                  settings=settings,prepared_sha256=prepared_sha, dataset=payload['dataset'],
                  synthetic_only=payload.get('synthetic_only',False),
                  training_subjects=sorted({r['subject'] for r in train_rows}),
                  selection_subjects=sorted({r['subject'] for r in payload['sequences'] if r['role']=='validation'}),
                  teacher_checkpoint=teacher_path,teacher_sha256=teacher_sha,
                  initialized_from_sha256=sha256(checkpoint) if checkpoint else None)
    write_json(output/'protocol.json',{**source,'time_basis':'nominal','test_signal_values_read':False})
    def checkpoint_payload(epoch, metrics, decoder):
        return {**source,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                    'optimizer_steps':steps,'exposures':exposures,'metrics':metrics,'decoder_minimum_frames':decoder,
                    'runtime_temperature':model.controller.temperature,'torch_rng':torch.get_rng_state(),
                    'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    def save(path, epoch, metrics, decoder):
        _save(path,{**checkpoint_payload(epoch,metrics,decoder),'best_snapshot':best_saved})
    save(output/'initial.pt',start_epoch-1,None,1)
    if teacher:
        (output/'teacher-targets').mkdir()
    started = time.monotonic()
    for epoch in range(start_epoch,epochs+1):
        model.train()
        total_loss, teacher_mismatch, hard_total, soft_total, radius_sum = 0.,0,0,0.,0.
        if mode=='lif':
            total_epochs = int(cfg.get('lif_distill_epochs',8))+int(cfg.get('lif_epochs',12))
            elapsed = epoch-1+(int(cfg.get('lif_distill_epochs',8)) if stage=='lif' else 0)
            model.controller.temperature = config.temperature+(.1-config.temperature)*min(1.,elapsed/max(1,total_epochs-1))
        order = np.random.default_rng(seed+epoch).permutation(len(train_rows))
        for begin in range(0,len(order),batch_size):
            optimizer.zero_grad(set_to_none=True)
            indices = order[begin:begin+batch_size]
            for index in indices:
                row = train_rows[int(index)]
                features,targets,labels = read_sequence(prepared.parent,row,device)
                teacher_target = None
                if teacher:
                    cache = output/'teacher-targets'/(row['sample_id']+'.npz')
                    if cache.exists():
                        with np.load(cache,allow_pickle=False) as data:
                            teacher_target=torch.from_numpy(data['target']).to(device)
                            teacher_count=int(data['count'])
                    else:
                        teacher_target,teacher_count = _teacher_target(teacher,features)
                        np.savez(cache,target=teacher_target.cpu().numpy(),count=teacher_count)
                    teacher_mismatch += int(teacher_count!=len(targets)-1)
                if mode=='frame-baseline':
                    loss = F.cross_entropy(model.classifier(features),labels)
                else:
                    result = model.training_loss(features,targets,teacher=teacher_target,
                                                 distill_weight=1. if teacher else 0.)
                    loss=result['loss']; hard_total+=result['hard_count']; soft_total+=float(result['soft_count'].detach())
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'nonfinite loss for {row["sample_id"]}')
                (loss/len(indices)).backward()
                total_loss += float(loss.detach()); exposures+=1
                radius_sum+=float(2*torch.sigmoid(model.radius_logit.detach()))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg.get('grad_clip',1.)),error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise FloatingPointError('nonfinite gradient')
            optimizer.step(); steps+=1
        candidates = cfg.get('baseline_minimum_frames',[1,4,8]) if mode=='frame-baseline' else [1]
        choices=[]
        for decoder in candidates:
            metrics, predictions=evaluate_sequences(model,payload,prepared.parent,minimum_segment_frames=int(decoder))
            rank=(metrics['sequence_exact'],-metrics['count_mae'],-int(decoder))
            choices.append((rank,int(decoder),metrics,predictions))
        rank,decoder,metrics,predictions=max(choices,key=lambda p:p[0])
        history.append(dict(epoch=epoch,loss=total_loss/len(train_rows),metrics=metrics,optimizer_steps=steps,
                            hard_firings=hard_total,soft_firings=soft_total,teacher_count_mismatches=teacher_mismatch,
                            radius_mean=radius_sum/len(train_rows),temperature=model.controller.temperature,
                            decoder_minimum_frames=decoder,elapsed_s=time.monotonic()-started))
        write_json(output/'history.json',history)
        write_json(output/f'validation-epoch-{epoch:03d}.json',predictions)
        if best_rank is None or rank>best_rank:
            best_rank,best_metrics,best_epoch,best_decoder=rank,metrics,epoch,decoder
            best_saved=copy.deepcopy(checkpoint_payload(epoch,metrics,decoder))
            save(output/'best.pt',epoch,metrics,decoder)
        save(output/f'epoch-{epoch:03d}.pt',epoch,metrics,decoder)
        save(output/'last.pt',epoch,metrics,decoder)
    save(output/'final.pt',epochs,history[-1]['metrics'],history[-1]['decoder_minimum_frames'])
    reloaded,saved=load_model(output/'best.pt',device)
    replay, predictions=evaluate_sequences(reloaded,payload,prepared.parent,minimum_segment_frames=best_decoder)
    if replay!=best_metrics:
        raise RuntimeError('best checkpoint readback changed metrics')
    write_json(output/'best-replay.json',predictions)
    if teacher:
        write_json(output/'teacher-cache-lock.json',{p.name:sha256(p) for p in (output/'teacher-targets').glob('*.npz')})
    if sha256(prepared)!=prepared_sha:
        raise ValueError('prepared manifest changed during training')
    load_prepared(prepared)  # verify sequence bytes still match the initial input snapshot
    result=dict(stage=stage,optimizer_steps=steps,exposures=exposures,best_epoch=best_epoch,best_metrics=best_metrics,
                best_decoder_minimum_frames=best_decoder,checkpoint_replay_passed=True,
                synthetic_only=payload.get('synthetic_only',False),test_signal_values_read=False)
    write_json(output/'complete.json',result)
    return result


def evaluate(checkpoint,prepared,output,device='cpu'):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    try:
        payload=load_prepared(prepared,verify_natural=True)
        model,saved=load_model(checkpoint,device)
        if saved['prepared_sha256']!=sha256(prepared):
            raise ValueError('evaluation prepared-data provenance mismatch')
        _check_data_model(payload,model.config)
        minimum=saved['decoder_minimum_frames']
        sequence,details=evaluate_sequences(model,payload,Path(prepared).parent,minimum_segment_frames=minimum)
        write_json(output/'sequence-predictions.json',details)
        natural=[]
        with (output/'natural-predictions.jsonl').open('x',encoding='utf-8') as handle:
            for record in payload['natural']:
                if record['role']!='validation':
                    continue
                features=np.load(record['path'],mmap_mode='r',allow_pickle=False)
                stream=EventStream(model,record['file'],minimum)
                events=[]
                for start in range(0,len(features),64):
                    block=torch.from_numpy(np.array(features[start:start+64],dtype=np.float32)).to(device)
                    rows=stream.push(block); events.extend(rows)
                    for row in rows: handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                rows=stream.finish(); events.extend(rows)
                for row in rows: handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                natural.append(dict(file=record['file'],tolerances={str(t):score_recording(events,record['instances'],record['frames'],
                                                                         period=model.config.frame_period_s,tolerance=t) for t in (2,4,8)}))
        write_json(output/'natural-metrics.json',natural)
        result=dict(sequence=sequence,natural_recordings=len(natural),scope='constructed sequences / partial-label natural diagnostics',
                    time_basis='nominal',synthetic_only=payload.get('synthetic_only',False),test_signal_values_read=False)
        write_json(output/'complete.json',result)
        return result
    except Exception as error:
        write_json(output/'failed.json',dict(error=type(error).__name__,message=str(error)))
        raise


def infer(checkpoint,input_path,output,*,device='cpu',settings=None,raw_rd=False,prepared=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    try:
        return _infer(checkpoint,input_path,output,device=device,settings=settings,raw_rd=raw_rd,prepared=prepared)
    except Exception as error:
        write_json(output/'failed.json',dict(error=type(error).__name__,message=str(error)))
        raise


def _infer(checkpoint,input_path,output,*,device='cpu',settings=None,raw_rd=False,prepared=None):
    model,saved=load_model(checkpoint,device)
    array=np.load(input_path,mmap_mode='r',allow_pickle=False)
    if array.ndim<2 or not len(array):
        raise ValueError('empty or invalid inference input')
    if not raw_rd and not saved.get('synthetic_only',False):
        if prepared is None or sha256(prepared)!=saved['prepared_sha256']:
            raise ValueError('feature inference provenance requires the matching prepared manifest')
        payload=load_prepared(prepared)
        _check_data_model(payload,model.config)
        matching=[r for r in payload['natural'] if Path(r['path']).resolve()==Path(input_path).resolve()]
        if len(matching)!=1:
            raise ValueError('input provenance is not a verified continuous_hidden recording')
        row=matching[0]
        locked_path(dict(path=str(input_path),sha256=row['sha256']))
        if array.shape!=(row['frames'],model.config.feature_dim):
            raise ValueError('feature provenance shape mismatch')
    frontend,carry,mean,scale=None,None,None,None
    if raw_rd:
        from continuous_mmwave.harrad.model import SegmentClassifier
        from continuous_mmwave.harrad.streaming import FrozenRDStream
        if not settings:
            raise ValueError('RD mode requires source-locked HARrad configuration')
        expected_locks=saved['settings'].get('locks',{})
        if any(settings['locks'][key]['sha256']!=expected_locks.get(key,{}).get('sha256')
               for key in ('radar_checkpoint','radar_normalizers')):
            raise ValueError('inference front-end provenance differs from training')
        path=locked_path(settings['locks']['radar_checkpoint'])
        norms=locked_path(settings['locks']['radar_normalizers'])
        backbone=SegmentClassifier('rd').eval()
        backbone.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
        frontend=FrozenRDStream(backbone).to(device)
        with np.load(norms,allow_pickle=False) as values:
            mean=np.array(values['rd_mean'],dtype=np.float32); scale=np.array(values['rd_scale'],dtype=np.float32)
        if mean.shape!=(40,64) or scale.shape!=(40,64) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale<=0).any():
            raise ValueError('invalid RD normalizers')
    stream=EventStream(model,str(input_path),saved['decoder_minimum_frames'])
    count=0
    with (output/'events.jsonl').open('x',encoding='utf-8') as handle, torch.no_grad():
        for start in range(0,len(array),64):
            block=np.array(array[start:start+64],dtype=np.float32,copy=True)
            if raw_rd:
                block=np.clip((block-mean)/scale,-8,8)
                result=frontend.forward_chunk(torch.from_numpy(block)[None].to(device),state=carry)
                carry=result['state']; features=result['hidden'][0]
            else:
                features=torch.from_numpy(block).to(device)
            for row in stream.push(features):
                handle.write(json.dumps(row,ensure_ascii=False)+'\n'); count+=1
        for row in stream.finish():
            handle.write(json.dumps(row,ensure_ascii=False)+'\n'); count+=1
    result=dict(events=count,frames=len(array),input_sha256=sha256(input_path),checkpoint_sha256=sha256(checkpoint),
                input_kind='preprocessed_rd_40x64' if raw_rd else 'synthetic_hidden' if saved.get('synthetic_only',False)
                else 'continuous_hidden',time_basis='nominal')
    write_json(output/'complete.json',result)
    return result


def smoke(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    root=output/'data'; root.mkdir()
    rng=np.random.default_rng(31)
    sequences=[]; natural=[]
    for subject,role in [('subject1','train'),('subject7','validation')]:
        segments=[0,0,1,2]
        values=np.concatenate([np.tile(np.eye(4,dtype=np.float32)[label],(10,1))+.02*rng.normal(size=(10,4)) for label in segments]).astype(np.float32)
        path=root/(subject+'.npy'); np.save(path,values)
        instances=[dict(start=i*10,stop=(i+1)*10,label=label,instance_id=i) for i,label in enumerate(segments)]
        natural.append(dict(file=subject,subject=subject,role=role,frames=len(values),instances=instances,path=str(path.resolve()),sha256=sha256(path)))
        p=root/(subject+'.npz')
        np.savez(p,features=values,frame_labels=np.repeat(segments,10),targets=np.array(segments),source_frames=np.arange(len(values)))
        sequences.append(dict(sample_id=subject,file=subject,subject=subject,role=role,frames=len(values),instances=instances,
                              targets=segments,path=p.name,sha256=sha256(p)))
    settings=dict(gate=dict(feature_dim=4,hidden_dim=8,num_classes=3,recent_tau_s=.2,buffer_frames=16),
                  training=dict(seed=31,threads=1,cif_epochs=2,lif_distill_epochs=1,lif_epochs=1,baseline_epochs=1,batch_size=1,baseline_minimum_frames=[1,2]))
    payload=dict(format_version=1,dataset='synthetic_plumbing_only',synthetic_only=True,feature_dim=4,num_classes=3,
                 frame_period_s=1/15,sequences=sequences,natural=natural,test_signal_values_read=False)
    prepared=root/'prepared.json'; write_json(prepared,payload)
    summaries=[]
    for stage in ['cif','lif-distill','lif','frame-baseline']:
        checkpoint=output/'cif'/'best.pt' if stage=='lif-distill' else output/'lif-distill'/'best.pt' if stage=='lif' else None
        summaries.append(train(settings,prepared,output/stage,stage=stage,checkpoint=checkpoint))
        evaluate(output/stage/'best.pt',prepared,output/(stage+'-evaluation'))
    result=dict(synthetic_only=True,real_harrad_training=False,stages=[r['stage'] for r in summaries],summaries=summaries)
    write_json(output/'complete.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description='Independent HARrad CIF/LIF event-gating experiment')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','train','evaluate','infer','smoke'):
        p=sub.add_parser(name)
        p.add_argument('--output-dir',required=True)
        if name!='smoke': p.add_argument('--device',default='cpu')
        if name in ('prepare','train'): p.add_argument('--config',required=True)
        if name=='train':
            p.add_argument('--prepared',required=True)
            p.add_argument('--stage',choices=['cif','lif-distill','lif','frame-baseline'],default='cif')
            p.add_argument('--checkpoint'); p.add_argument('--resume',action='store_true')
        if name=='evaluate':
            p.add_argument('--prepared',required=True); p.add_argument('--checkpoint',required=True)
        if name=='infer':
            p.add_argument('--input',required=True); p.add_argument('--checkpoint',required=True)
            p.add_argument('--rd',action='store_true',help='input is preprocessed RD [T,40,64], not raw ADC')
            p.add_argument('--config')
            p.add_argument('--prepared',help='required provenance manifest for real continuous_hidden input')
    args=parser.parse_args(argv)
    try:
        settings=json.loads(Path(args.config).read_text(encoding='utf-8-sig')) if getattr(args,'config',None) else None
        if args.command=='prepare':
            payload=prepare(settings,args.output_dir,args.device)
            result=dict(sequences=len(payload['sequences']),recordings=len(payload['natural']))
        elif args.command=='train':
            result=train(settings,args.prepared,args.output_dir,stage=args.stage,checkpoint=args.checkpoint,device=args.device,resume=args.resume)
        elif args.command=='evaluate':
            result=evaluate(args.checkpoint,args.prepared,args.output_dir,args.device)
        elif args.command=='infer':
            result=infer(args.checkpoint,args.input,args.output_dir,device=args.device,settings=settings,raw_rd=args.rd,
                         prepared=args.prepared)
        else:
            result=smoke(args.output_dir)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return 0
    except Exception as error:
        print(f'{type(error).__name__}: {error}',file=sys.stderr)
        return 2

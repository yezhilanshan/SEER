"""HARrad frozen-event Stage 1 with exact future-frame targets."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .event_semantic import build_production
from .preprocess import sha256, write_json
from .training import seed_everything
from ..v26.long_prediction import MatchedResidualPredictor, matched_architecture


HORIZONS = (3, 9, 15)


def sample_indices(cutoffs, frames):
    cutoffs = np.asarray(cutoffs, dtype=np.int64)
    if np.any(cutoffs < 2) or np.any(np.diff(cutoffs) <= 0):
        raise ValueError('strictly increasing observed cutoffs required')
    return np.flatnonzero(cutoffs + max(HORIZONS) <= frames)


def future_cutoffs(current):
    return np.asarray(current)[..., None] + np.asarray(HORIZONS)


class PredictionModel(nn.Module):
    def __init__(self, kind, maximum_history):
        super().__init__()
        self.motion = nn.Linear(1024, 256)
        with torch.no_grad():
            self.motion.weight.copy_(torch.eye(256).repeat(1, 4) / 4)
            self.motion.bias.zero_()
        self.heads = nn.ModuleList([
            MatchedResidualPredictor(kind, matched_architecture(kind, maximum_history))
            for _ in HORIZONS])

    def forward(self, radar, event, prefix, mask):
        current = self.motion(radar.flatten(1))
        predictions = [head(current, prefix, event, mask) for head in self.heads]
        return torch.stack([p[0] for p in predictions], 1), torch.stack([p[1] for p in predictions], 1)


def load(path):
    return json.loads(Path(path).read_text())


def prepare(root, output):
    warmup = root / 'shared-front-event-warmup-v1'
    config = load(warmup / 'config.json')
    finished = load(warmup / 'training/complete.json')
    source = warmup / 'training/best.pt'
    if sha256(source) != finished['best_sha256']:
        raise ValueError('event checkpoint changed')
    if sha256(config['radar_checkpoint']) != config['radar_checkpoint_sha256']:
        raise ValueError('radar checkpoint changed')
    for name, info in load(warmup / 'input-lock.json')['qwen_files'].items():
        if sha256(Path(config['qwen_path']) / name) != info['sha256']:
            raise ValueError('Qwen source changed')
    model = build_production(config)
    saved = torch.load(source, map_location='cpu', weights_only=False)
    model.load_learned(saved['learned'])
    model.requires_grad_(False).eval()
    manifest = load(warmup / 'feature-manifest.json')
    calibration = {max(r['file'] for r in manifest if r['subject'] == f'subject{i}') for i in range(1, 7)}
    destination = output / 'cache'
    destination.mkdir()
    result = []
    with torch.inference_mode():
        for record in manifest:
            if record['role'] not in ('train', 'validation'):
                raise ValueError('Stage 1 train/validation sources only')
            path = Path(record['cells_path'])
            if sha256(path) != record['artifacts']['cells.npy']['sha256']:
                raise ValueError('frozen radar cells changed')
            cells = np.load(path, mmap_mode='r')
            frames = len(cells)
            folder = destination / path.parent.name
            folder.mkdir()
            # Tokens at cutoff c use exactly frames [c-2,c). This includes odd cutoffs.
            all_tokens = np.empty((frames + 1, 4, 256), np.float32)
            all_tokens[:2] = np.nan
            for first in range(2, frames + 1, 256):
                ends = np.arange(first, min(first + 256, frames + 1))
                pairs = np.stack([cells[c-2:c] for c in ends])
                values = model.tokens(torch.as_tensor(pairs, device='cuda'),
                    torch.ones(len(ends), 2, dtype=torch.bool, device='cuda'))
                all_tokens[ends] = values.cpu().numpy()
            starts = sorted({0} | {r['start'] for r in record['instances']})
            cutoffs, beginnings = [], []
            for begin, end in zip(starts, starts[1:] + [frames]):
                cutoffs.extend(range(begin + 2, end + 1, 2))
                beginnings.extend([len(beginnings)] * len(range(begin + 2, end + 1, 2)))
            cutoffs = np.asarray(cutoffs, np.int64)
            beginnings = np.asarray(beginnings, np.int64)
            states = np.lib.format.open_memmap(folder/'events.npy', mode='w+', dtype='float32',
                                               shape=(len(cutoffs), 8, 896))
            event = None
            prompt = model.prompt(1)
            for index, cutoff in enumerate(cutoffs):
                if beginnings[index] == index:
                    event = model.former.initial_state(1)
                radar = torch.as_tensor(all_tokens[cutoff:cutoff+1], device='cuda')
                age = torch.tensor([min((index-beginnings[index]+1)*2/15/1.2, 1.)], device='cuda')
                event = model.former.stream_update(event, radar, age, None, prompt).event_tokens
                states[index] = event[0].float().cpu().numpy()
            states.flush()
            selected = sample_indices(cutoffs, frames)
            labels = np.full(frames, -1, np.int64)
            instance_ids = labels.copy()
            repeat = []
            for index, instance in enumerate(record['instances']):
                s, e = instance['start'], instance['stop']
                labels[s:e] = instance['label']
                instance_ids[s:e] = index
                if index and record['instances'][index-1]['label'] == instance['label']:
                    repeat.append(s)
            edges = np.array(sorted({p for r in record['instances'] for p in (r['start'], r['stop'])}))
            target = all_tokens.mean(1)
            np.save(folder/'radar.npy', all_tokens[cutoffs])
            np.save(folder/'reference.npy', target[cutoffs])
            np.save(folder/'future.npy', target[future_cutoffs(cutoffs[selected])])
            groups = np.zeros((len(selected), 3, 4), bool)
            for i, cutoff in enumerate(cutoffs[selected]):
                for h, offset in enumerate(HORIZONS):
                    interval = instance_ids[cutoff-1:cutoff+offset]
                    groups[i, h] = (np.all(interval == interval[0]) and interval[0] >= 0,
                        np.any((edges >= cutoff) & (edges < cutoff+offset)),
                        any(cutoff <= p < cutoff+offset for p in repeat), np.any(interval < 0))
            np.savez(folder/'indices.npz', cutoffs=cutoffs, beginnings=beginnings, selected=selected, groups=groups)
            role = 'validation' if record['role'] == 'validation' else 'calibration' if record['file'] in calibration else 'fit'
            row = dict(file=record['file'], role=role, path=str(folder), frames=frames,
                       samples=len(selected), maximum_history=int((np.arange(len(cutoffs))-beginnings+1).max()),
                       files={p.name:sha256(p) for p in folder.iterdir()})
            result.append(row)
            write_json(output/'cache-manifest.json', result)
            write_json(output/'status.json', dict(phase='frozen_cache', records=len(result), total=35, last=record['file']))
    if sha256(source) != finished['best_sha256']:
        raise ValueError('event source changed during caching')
    write_json(output/'source-lock.json', dict(event_checkpoint=str(source), event_sha256=finished['best_sha256'],
        radar_sha256=config['radar_checkpoint_sha256'], target='fixed mean of frozen radar tokens; input projection separately trained',
        horizons_frames=HORIZONS, horizons_s=[.2,.6,1.], teacher='official instance starts; same history for GRU/TCN',
        frozen_modules_unchanged=True, test_signal_values_read=False))
    return result


def records_from(manifest):
    result = []
    for row in manifest:
        folder = Path(row['path'])
        for name, expected in row['files'].items():
            if sha256(folder/name) != expected:
                raise ValueError('Stage 1 cache changed')
        record = dict(row)
        for key in ('radar','reference','future','events'):
            record[key] = np.load(folder/(key+'.npy'), mmap_mode='r')
        with np.load(folder/'indices.npz') as data:
            record.update({k:data[k].copy() for k in data.files})
        result.append(record)
    return result


def batch(records, samples, mean, scale, device):
    prefixes, radar, event, target = [], [], [], []
    for rid, sid in samples:
        record = records[rid]
        position = record['selected'][sid]
        start = record['beginnings'][position]
        prefixes.append((np.array(record['radar'][start:position+1])-mean)/scale)
        radar.append((record['radar'][position]-mean)/scale)
        event.append(record['events'][position])
        target.append((record['future'][sid]-mean)/scale)
    length = max(map(len, prefixes))
    prefix = np.zeros((len(samples), length, 4, 256), np.float32)
    mask = np.zeros((len(samples), length), bool)
    for i, value in enumerate(prefixes):
        prefix[i, :len(value)] = value
        mask[i, :len(value)] = True
    return tuple(torch.as_tensor(np.array(x),device=device) for x in (radar,event,prefix,mask,target))


def metrics(mu, logvar, target, persistence, variance, groups):
    error = np.square(mu-target)
    nll = .5*(np.log(2*np.pi)+logvar+error*np.exp(-logvar))
    z = np.abs(mu-target)/np.exp(.5*logvar)
    result = []
    for h in range(3):
        item = dict(mse=float(error[:,h].mean()), nll=float(nll[:,h].mean()),
                    skill=float(1-error[:,h].mean()/np.square(persistence[:,h]-target[:,h]).mean()),
                    variance=float(np.exp(logvar[:,h]).mean()), standardized_error=float(z[:,h].mean()),
                    coverage={str(level):float((z[:,h]<=q).mean()) for level,q in
                              ((.5,.67448975),(.8,1.28155157),(.9,1.64485363),(.95,1.95996398))})
        item['groups'] = {name:dict(n=int(groups[:,h,g].sum()),
            mse=float(error[groups[:,h,g],h].mean()) if groups[:,h,g].any() else None,
            nll=float(nll[groups[:,h,g],h].mean()) if groups[:,h,g].any() else None)
            for g,name in enumerate(('steady','transition','repeated','unknown'))}
        result.append(item)
    return result


def run(root, output, steps, seed_list):
    output.mkdir(parents=True, exist_ok=False)
    seed_everything(42,4)
    manifest = prepare(root, output)
    records = records_from(manifest)
    fit = [i for i,r in enumerate(records) if r['role']=='fit']
    # Fixed reference normalization is independent of all trainable projections.
    values = np.concatenate([r['reference'][r['selected']] for r in records if r['role']=='fit'])
    mean = values.mean(0, dtype=np.float64).astype('float32')
    scale = np.maximum(values.std(0, dtype=np.float64),1e-3).astype('float32')
    errors = np.concatenate([((r['future']-r['reference'][r['selected'],None])/scale)**2 for r in records if r['role']=='fit'])
    variance = np.clip(errors.mean(0),np.exp(-6.),np.exp(2.))
    np.savez(output/'target-stats.npz', mean=mean, scale=scale, persistence_variance=variance)
    maximum_history = max(r['maximum_history'] for r in records)
    jobs = []
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        schedule = [[(int(rid),int(rng.integers(records[rid]['samples']))) for rid in rng.choice(fit,16)] for _ in range(steps)]
        for kind in ('persistence','gru','tcn','event_free','event'):
            directory = output/f'{kind}-seed{seed}'
            directory.mkdir()
            model = None
            if kind!='persistence':
                seed_everything(seed,4)
                model = PredictionModel(kind,maximum_history).cuda()
                optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
                for step, samples in enumerate(schedule,1):
                    model.train()
                    radar,event,prefix,mask,target = batch(records,samples,mean,scale,'cuda')
                    mu,lv = model(radar,event,prefix,mask)
                    loss = (.5*(np.log(2*np.pi)+lv+(mu-target).square()*(-lv).exp())).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(loss) or not torch.isfinite(norm):
                        raise RuntimeError('nonfinite predictor update')
                    optimizer.step()
                    if step==1 or step%20==0:
                        write_json(output/'status.json',dict(phase='prediction_training',kind=kind,seed=seed,step=step,steps=steps,loss=float(loss.detach())))
                model.eval()
                validation_id = next(i for i,r in enumerate(records) if r['role']=='validation')
                probe = batch(records,[(validation_id,i) for i in range(min(16,records[validation_id]['samples']))],mean,scale,'cuda')
                with torch.inference_mode():
                    expected = tuple(x.cpu().clone() for x in model(*probe[:4]))
                torch.save(dict(model=model.state_dict(),kind=kind,seed=seed,steps=steps,
                                maximum_history=maximum_history),directory/'final.pt')
                del model
                model = PredictionModel(kind,maximum_history).cuda()
                model.load_state_dict(torch.load(directory/'final.pt',map_location='cuda',weights_only=False)['model'])
                model.eval()
                with torch.inference_mode():
                    actual = model(*probe[:4])
                replay_gap = max(float((a.cpu()-b).abs().max()) for a,b in zip(actual,expected))
                if replay_gap != 0:
                    raise RuntimeError(f'predictor checkpoint replay mismatch: {replay_gap}')
                write_json(directory/'checkpoint-replay.json',dict(passed=True,maximum_absolute_error=replay_gap))
            reports = []
            for rid, record in enumerate(records):
                if record['role']=='fit':
                    continue
                write_json(output/'status.json',dict(phase='prediction_evaluation',kind=kind,seed=seed,file=record['file']))
                means,logvars = [],[]
                with torch.inference_mode():
                    for start in range(0,record['samples'],16):
                        ids = list(range(start,min(start+16,record['samples'])))
                        if model is None:
                            current=(record['reference'][record['selected'][ids]]-mean)/scale
                            mu=np.repeat(current[:,None],3,1)
                            lv=np.broadcast_to(np.log(variance),mu.shape)
                        else:
                            radar,event,prefix,mask,_=batch(records,[(rid,i) for i in ids],mean,scale,'cuda')
                            m,l=model(radar,event,prefix,mask)
                            mu,lv=m.cpu().numpy(),l.cpu().numpy()
                        means.append(mu);logvars.append(lv)
                mu,lv=np.concatenate(means),np.concatenate(logvars)
                target=(record['future']-mean)/scale
                persistence=np.repeat(((record['reference'][record['selected']]-mean)/scale)[:,None],3,1)
                report=metrics(mu,lv,target,persistence,variance,record['groups'])
                reports.append(dict(file=record['file'],role=record['role'],horizons=report))
                np.savez_compressed(directory/(Path(record['file']).stem+'.npz'),mean=mu,logvar=lv,target=target,
                    persistence=persistence,groups=record['groups'],observed_cutoff=record['cutoffs'][record['selected']],
                    target_cutoff=future_cutoffs(record['cutoffs'][record['selected']]))
            job=dict(kind=kind,seed=seed,steps=steps if model else 0,records=reports,
                     parameters=sum(p.numel() for p in model.parameters()) if model else 0)
            write_json(directory/'complete.json',job)
            jobs.append(job)
            write_json(output/'jobs.json',jobs)
            del model
            torch.cuda.empty_cache()
    aggregate = {}
    for kind in ('persistence','gru','tcn','event_free','event'):
        group = [j for j in jobs if j['kind']==kind]
        aggregate[kind] = [{key:float(np.mean([r['horizons'][h][key] for j in group for r in j['records'] if r['role']=='validation']))
                            for key in ('mse','nll','skill')} for h in range(3)]
        for h in range(3):
            aggregate[kind][h]['coverage'] = {str(level):float(np.mean([
                r['horizons'][h]['coverage'][str(level)] for j in group for r in j['records'] if r['role']=='validation']))
                for level in (.5,.8,.9,.95)}
    paired = {}
    for seed in seed_list:
        by_kind = {j['kind']:j for j in jobs if j['seed']==seed}
        paired[str(seed)] = {other:[{key:float(np.mean([
            r['horizons'][h][key] for r in by_kind['event']['records'] if r['role']=='validation']))
            - float(np.mean([r['horizons'][h][key] for r in by_kind[other]['records'] if r['role']=='validation']))
            for key in ('mse','nll')} for h in range(3)] for other in ('gru','tcn','event_free')}
    passed = all(aggregate['event'][h][key]<aggregate[other][h][key]
                 for h in range(3) for key in ('mse','nll') for other in ('gru','tcn','event_free'))
    write_json(output/'complete.json',dict(aggregate=aggregate,stage1_prediction_advantage=passed,
        paired_event_minus_baseline=paired,reset_policy_started=False,seeds=seed_list,steps=steps,
        horizons_s=[.2,.6,1.],test_signal_values_read=False))
    write_json(output/'status.json',dict(phase='complete',prediction_advantage=passed))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1000)
    args=parser.parse_args()
    run(args.root,args.output,args.steps,[42,43,44])

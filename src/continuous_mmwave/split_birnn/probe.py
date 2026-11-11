"""Preparation-only synthetic throughput and one real CUDA optimization step."""
from __future__ import annotations
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from .model import SplitBiGRU, BiGRUBaseline, parameter_audit
from .loss import combo_loss
from .audit import write_json


def sample_batch(rows: list[dict], rng: np.random.Generator, *, batch_size: int, length: int):
    if batch_size <= 0 or length <= 0:
        raise ValueError('positive batch size and length required')
    arrays=[]
    for row in rows:
        with np.load(row['path']) as data:
            if len(data['md'])>=length:
                arrays.append((row,data['md'].copy(),data['labels'].copy()))
    if not arrays:
        raise ValueError('no recordings long enough for shared snippet length')
    selected = rng.choice(len(arrays),batch_size,replace=batch_size>len(arrays))
    x,y,sources=[],[],[]
    for i in selected:
        row,md,labels=arrays[i]
        start=int(rng.integers(0,len(md)-length+1))
        x.append(md[start:start+length].astype(np.float32)); y.append(labels[start:start+length])
        sources.append({'recording_id':row['recording_id'],'start_frame':start,'frames':length})
    return torch.from_numpy(np.stack(x)),torch.from_numpy(np.stack(y)),sources


def step(net,optimizer,x,y):
    if not (y != -100).any():
        raise ValueError('preparation optimization probe requires supervised frames')
    optimizer.zero_grad(set_to_none=True)
    if isinstance(net,SplitBiGRU):
        out=net.forward_sequence(x)
        loss=combo_loss(out['forward'],y)[0]+combo_loss(out['corrected'],y)[0]
    else:
        loss=combo_loss(net(x),y)[0]
    if not torch.isfinite(loss):
        raise ValueError('nonfinite loss')
    loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):
        raise ValueError('missing/nonfinite gradient')
    optimizer.step()
    return float(loss.detach())


def run_probe(output: Path, manifest: Path, *, batch_size=8, repeats=5):
    if not torch.cuda.is_available():
        raise RuntimeError('this native throughput probe requires CUDA')
    if batch_size < 1 or repeats < 1:
        raise ValueError('positive batch size and repeats required')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(1)
    device=torch.device('cuda')
    result={'scope':'preparation_not_formal_training','formal_optimizer_steps':0,
            'random_initialization':True,'trained_checkpoint_loaded':False,
            'seed':42,'torch':torch.__version__,'platform':platform.platform(),
            'device':torch.cuda.get_device_name(), 'parameter_audit':parameter_audit(SplitBiGRU()),
            'batch_size':batch_size,'warmup_steps':2,'timed_steps':repeats,'training_throughput':[]}
    for cls in (SplitBiGRU,BiGRUBaseline):
        for length in (100,512,1000):
            net=cls().to(device).train(); optimizer=torch.optim.Adam(net.parameters(),lr=.001)
            x=torch.randn(batch_size,length,128,device=device)
            y=torch.randint(0,9,(batch_size,length),device=device)
            for _ in range(2): step(net,optimizer,x,y)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            elapsed=[]
            for _ in range(repeats):
                torch.cuda.synchronize(); start=time.perf_counter()
                loss=step(net,optimizer,x,y)
                torch.cuda.synchronize(); elapsed.append((time.perf_counter()-start)*1000)
            row={'model':cls.__name__,'frames':length,'mean_ms':float(np.mean(elapsed)),
                 'p50_ms':float(np.median(elapsed)),'p95_ms':float(np.percentile(elapsed,95)),
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                 'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'last_synthetic_loss':loss,
                 'valid_frames_per_second':batch_size*length/(np.mean(elapsed)/1000)}
            result['training_throughput'].append(row); print(json.dumps(row),flush=True)
            del net,optimizer,x,y; torch.cuda.empty_cache()
    rows=[json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    # Existing SEER training subject; do not inspect outer holdout performance to pick settings.
    rows=[{**r,'path':str((Path(manifest).parent/r['path']).resolve())} for r in rows if r['subject_id']=='subject_01']
    x,y,sources=sample_batch(rows,np.random.default_rng(42),batch_size=2,length=512)
    net=SplitBiGRU().to(device).train(); optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    loss=step(net,optimizer,x.to(device),y.to(device)); torch.cuda.synchronize()
    result['real_data_single_step']={'loss':loss,'sources':sources,'gradient_finite':True,
                                    'optimizer_steps':1,'checkpoint_saved':False,'labels_valid':int((y!=-100).sum())}
    net.eval(); single=torch.randn(1,1,128,device=device); stream=torch.randn(1,512,128,device=device)
    with torch.no_grad():
        _,cache,_=net.forward_chunk(stream)
        baseline=BiGRUBaseline().to(device).eval()
        functions={'forward_one_frame':lambda:net.forward_chunk(single),
                   'backward_cached_512':lambda:net.correct(cache),
                   'baseline_window_512':lambda:baseline(stream)}
        timing={}
        for name,fn in functions.items():
            for _ in range(10): fn()
            times=[]
            for _ in range(30):
                torch.cuda.synchronize(); start=time.perf_counter(); fn(); torch.cuda.synchronize()
                times.append((time.perf_counter()-start)*1000)
            timing[name]={'mean_ms':float(np.mean(times)),'std_ms':float(np.std(times,ddof=1)),
                          'p50_ms':float(np.median(times)),'p95_ms':float(np.percentile(times,95))}
    result['compute_latency_not_network_or_action_delay']=timing
    result['latency_model_state']='Split after one real preparation step; baseline randomly initialized'
    result['synthetic_optimizer_steps']=6*(2+repeats)
    result['training_budget_frozen']=False
    result['formal_training_ready']=False
    write_json(output/'probe.json',result)
    print('PROBE_DONE',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args();run_probe(a.output,a.manifest,batch_size=a.batch_size,repeats=a.repeats)

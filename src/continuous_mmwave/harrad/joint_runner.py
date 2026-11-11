"""HARrad fixed-contract joint-pretraining capsule runner and real CUDA checks."""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .joint_data import make_record,fit_stats,fixed_design,collate_origins
from .joint_model import build_joint
from .joint_training import fit_job,joint_losses
from .event_training import frozen_digest
from .training import seed_everything
from .preprocess import sha256,write_json


def prepare(config,output):
    output=Path(output); output.mkdir(exist_ok=False)
    for row in config['locks'].values():
        if sha256(row['path'])!=row['sha256']: raise ValueError('input source lock changed')
    metadata=json.loads(Path(config['locks']['continuous_manifest']['path']).read_text())
    canonical=json.loads(Path(config['locks']['canonical_manifest']['path']).read_text())
    if len(metadata)!=35 or len(canonical)!=35: raise ValueError('unexpected recording count')
    original={r['file']:r for r in canonical}
    records=[]; manifest=[]
    # Validate every role/subject before opening any signal array.
    for r in metadata:
        if r['role'] not in ('train','validation'): raise ValueError('test signals forbidden')
        expected='validation' if r['subject']=='subject7' else 'train'
        if r['subject'] not in {f'subject{i}' for i in range(1,8)} or r['role']!=expected: raise ValueError('split mismatch')
        if any(r[k]!=original[r['file']][k] for k in ('frames','role','subject','instances')): raise ValueError('canonical metadata mismatch')
    for r in metadata:
        cell_path=Path(r['cells_path']); vector_path=Path(r['artifact_dir'])/'frame_features.npy'
        for path in (cell_path,vector_path):
            if sha256(path)!=r['artifacts'][path.name]['sha256']: raise ValueError('frozen CNN feature changed')
        record=make_record(r,np.load(cell_path,mmap_mode='r'),np.load(vector_path,mmap_mode='r'))
        records.append(record)
        manifest.append({k:record[k] for k in ('file','role','subject','frames','dropped_tail_frames')})
        manifest[-1].update(updates=len(record['motion']),common_origins=len(record['legal']),
                            semantic_origins=int((record['semantic'][record['legal']]>=0).sum()),
                            cells_path=str(cell_path),frame_features_path=str(vector_path),
                            source_hashes={p.name:r['artifacts'][p.name]['sha256'] for p in (cell_path,vector_path)})
    for role,n,frames in [('train',31,190238),('validation',4,25244)]:
        chosen=[r for r in records if r['role']==role]
        if len(chosen)!=n or sum(r['frames'] for r in chosen)!=frames: raise ValueError('incomplete coverage')
    stats=fit_stats(records); design=fixed_design(records,config['seeds'],config['training']['steps'],config['evaluation_per_record'])
    np.savez(output/'target-stats.npz',**stats); np.savez_compressed(output/'design.npz',**design)
    write_json(output/'manifest.json',manifest)
    write_json(output/'contract.json',{'horizons_updates':[1,3,5],'horizons_seconds':[2/15,6/15,10/15],
               'target_dim':64,'oracle_initialization':True,'max_context_updates':32,'gradient_updates':16,
               'validation_unique_origins':len(design['validation']),
               'motion_panel_n':int(design['validation'][:,2].sum()),'semantic_panel_n':int(design['validation'][:,3].sum()),
               'target_stats_sha256':sha256(output/'target-stats.npz'),'design_sha256':sha256(output/'design.npz'),
               'test_signal_values_read':False})
    return records,stats,design


def preflight(config,records,stats,output):
    output=Path(output); output.mkdir(exist_ok=False); seed_everything(42,4)
    origins=[]
    for rid,r in enumerate(records):
        if r['role']!='train': continue
        for t in r['legal']:
            length=min(int(t-r['latest_start'][t]+1),32)
            if length==32 and (len(origins)>0 or r['semantic'][t]>=0):
                origins.append((rid,int(t)))
                if len(origins)==2: break
        if len(origins)==2: break
    if len(origins)!=2: raise RuntimeError('no real maximum-context stress batch available')
    b={k:v.cuda() for k,v in collate_origins(records,origins,stats).items()}
    model=build_joint(config,'AB',42); before=frozen_digest(model)
    if before!=config['warmup_frozen_digest']: raise RuntimeError('frozen Qwen differs from warmup')
    torch.cuda.reset_peak_memory_stats(); started=time.time()
    model.eval()
    with torch.no_grad():
        initial=model(b['cells'],b['mask'],b['current'])
        zero_gap=float((initial['mean']-b['current'][:,None]).abs().max())
        if zero_gap!=0: raise RuntimeError('zero forecast is not Persistence')
    # Temporarily open the zero output layer for the input-gradient test; this
    # throwaway model is never resumed by a formal job.
    with torch.no_grad(): model.forecast[-1].weight.normal_(std=.001)
    gradients={}
    for objective in ('motion','semantic'):
        model.train(); model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
            losses=joint_losses(model(b['cells'],b['mask'],b['current']),b,stats)
            loss=losses[objective]
        (loss*256).backward()
        nonempty=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not nonempty or not all(torch.isfinite(g).all() for g in nonempty): raise RuntimeError('nonfinite real joint gradient')
        grad=float(model.tokens.project.weight.grad.abs().max())
        if grad<=0: raise RuntimeError('task gradient misses shared TokenAdapter')
        gradients[objective]={'loss':float(loss.detach()),'shared_projection_grad_max_scaled':grad}
    if any(p.grad is not None for p in model.former.adapter.parameters()): raise RuntimeError('Qwen received gradient')
    model.eval()
    with torch.no_grad():
        mask=b['mask'].clone(); mask[0,6:]=False
        a=model(b['cells'],mask,b['current'])
        changed=b['cells'].clone(); changed[0,6:]*=-100
        c=model(changed,mask,b['current'])
        pad_gap=max(float((a[k][0]-c[k][0]).abs().max()) for k in ('mean','logits'))
        if pad_gap!=0: raise RuntimeError('real padding/causal prefix mismatch')
    unchanged=before==frozen_digest(model)
    if not unchanged: raise RuntimeError('frozen parameters changed in preflight')
    peak=torch.cuda.max_memory_allocated()
    if peak>9*1024**3: raise RuntimeError('joint stress preflight exceeds 9GiB allocated')
    report={'passed':True,'origins':origins,'max_context':32,'batch_size':2,'zero_forecast_max_abs':zero_gap,
            'gradients':gradients,'frozen_unchanged':True,'frozen_digest_matches_warmup':True,
            'padding_future_max_abs':pad_gap,'peak_cuda_allocated':peak,'seconds':time.time()-started,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    write_json(output/'complete.json',report)
    del model; gc.collect(); torch.cuda.empty_cache()
    return report


def run(config,output,prepare_only=False):
    output=Path(output)
    if prepare_only:
        output.mkdir(parents=True,exist_ok=False)
    elif not output.exists():
        output.mkdir(parents=True,exist_ok=False)
    elif not (output/'preflight/complete.json').exists() or (output/'jobs').exists():
        raise RuntimeError('refusing existing/incomplete run')
    started=time.time()
    try:
        write_json(output/'config.json',config)
        if config['training']['steps']!=1000 or config['seeds']!=[42,43,44] or config['arms']!=['M','A','B','AB']:
            raise ValueError('fixed training budget changed')
        if (output/'data').exists():
            # Rebuild records from locked sources, leaving the immutable prepared
            # stats/design in place. Only the formal launch uses this path.
            import tempfile
            with tempfile.TemporaryDirectory(prefix='harrad_joint_') as directory:
                records,stats,design=prepare(config,Path(directory)/'data')
                for name in ('target-stats.npz','design.npz'):
                    # npz containers have timestamps; compare arrays, not zip bytes.
                    with np.load(output/'data'/name) as a,np.load(Path(directory)/'data'/name) as b:
                        if set(a.files)!=set(b.files): raise RuntimeError('prepared design keys changed')
                        for key in a.files: np.testing.assert_array_equal(a[key],b[key])
        else: records,stats,design=prepare(config,output/'data')
        if not (output/'preflight/complete.json').exists():
            write_json(output/'status.json',{'phase':'real_qwen_preflight','updated_unix':time.time()})
            preflight(config,records,stats,output/'preflight')
        if prepare_only:
            write_json(output/'status.json',{'phase':'prepared','updated_unix':time.time()}); return
        (output/'jobs').mkdir(exist_ok=False)
        completed={}
        for seed in config['seeds']:
            for arm in config['arms']:
                seed_everything(seed,4); name=f'{arm}-seed{seed}'
                write_json(output/'status.json',{'phase':'training','current_job':name,'completed_jobs':len(completed),
                           'total_jobs':12,'updated_unix':time.time()})
                model=build_joint(config,arm,seed)
                result=fit_job(model,records,stats,design[f'schedule_{seed}'],design['validation'],config['training'],output/'jobs'/name)
                completed[name]=result; write_json(output/'completed-jobs.json',completed)
                del model; gc.collect(); torch.cuda.empty_cache()
        for row in config['locks'].values():
            if sha256(row['path'])!=row['sha256']: raise RuntimeError('original model/source changed')
        report={'phase':'complete','jobs':completed,'successful_steps':sum(r['successful_steps'] for r in completed.values()),
                'attempts':sum(r['attempts'] for r in completed.values()),'test_signal_values_read':False,
                'seconds_formal':time.time()-started,'completed_unix':time.time()}
        write_json(output/'complete.json',report); write_json(output/'status.json',{'phase':'complete','updated_unix':time.time()})
    except BaseException as exc:
        failed={'phase':'failed','error':repr(exc),'traceback':traceback.format_exc(),'updated_unix':time.time()}
        write_json(output/'failed.json',failed); write_json(output/'status.json',failed); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True)
    p.add_argument('--prepare-only',action='store_true'); args=p.parse_args()
    run(json.loads(Path(args.config).read_text()),args.output,args.prepare_only)

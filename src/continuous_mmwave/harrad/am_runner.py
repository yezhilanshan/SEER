"""Source-locked preparation, native preflight and persistent semantic 2x2 run."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time
import traceback
import numpy as np
import torch
from .am_data import select_bank,diagnostic_panels,build_bank
from .am_model import ARMS,build_alignment
from .am_training import fit_alignment,alignment_losses
from .joint_runner import prepare
from .joint_model import build_joint
from .joint_training import _device_batch
from .joint_data import collate_origins
from .event_training import frozen_digest
from .training import seed_everything
from .preprocess import write_json,sha256


def validate_config(config):
    if config['arms']!=list(ARMS): raise ValueError('fixed 2x2 arms changed')
    c=config['training']
    if config['seeds']!=[42,43,44] or c['steps']!=1000 or c['checkpoints']!=[0,300,1000]:
        raise ValueError('fixed semantic experiment budget changed')
    if config['bank_per_class']!=32 or c['temperature']!=.1 or c['alignment_weight']!=.1:
        raise ValueError('fixed reference-bank objective changed')
    if c['evaluation_batch_size']!=2 or c['max_batch_attempts']!=3:
        raise ValueError('fixed evaluation/retry settings changed')


def preflight_binding(config,manifest):
    encoded=json.dumps(config,sort_keys=True,separators=(',',':')).encode()
    return {'config_sha256':hashlib.sha256(encoded).hexdigest(),'source_manifest_sha256':sha256(manifest)}


def validate_preflight_binding(config,proof,manifest):
    for key,value in preflight_binding(config,manifest).items():
        if proof.get(key)!=value: raise ValueError(f'AM preflight {key} changed')


def source_manifest():
    return Path(__file__).resolve().parents[3]/'source-manifest.json'


def prepare_design(config,path):
    records,stats,design=prepare(config,path)
    source=config['source_design']
    if sha256(source['path'])!=source['sha256']: raise ValueError('original joint design lock changed')
    with np.load(source['path']) as old:
        for key,value in design.items(): np.testing.assert_array_equal(value,old[key])
    panels=diagnostic_panels(records,design); rows=select_bank(records,config['bank_per_class'])
    if len(panels['train'])!=512 or len(panels['validation'])!=512 or rows.shape!=(384,3):
        raise ValueError('incomplete fixed panels or class-covered bank')
    np.savez_compressed(Path(path)/'am-design.npz',**panels,bank_origins=rows)
    return records,stats,design,panels,rows


def real_preflight(config,records,stats,rows,path):
    path=Path(path); path.mkdir(exist_ok=False); seed_everything(42,4)
    origins=[]
    for rid,r in enumerate(records):
        if r['role']!='train': continue
        for t in r['legal']:
            if r['semantic'][t]>=0 and t-int(r['latest_start'][t])+1>=32:
                origins.append((rid,int(t)))
                if len(origins)==2: break
        if len(origins)==2: break
    if len(origins)!=2: raise RuntimeError('no fully labelled maximum-context stress batch')
    batch=_device_batch(collate_origins(records,origins,stats),'cuda')
    baseline=build_joint(config,'B',42).eval()
    with torch.no_grad(): expected=baseline(batch['cells'],batch['mask'],batch['current'])['logits'].cpu()
    del baseline; gc.collect(); torch.cuda.empty_cache()
    gaps={}; gradients={}; bank=None; peak=0; started=time.time()
    for arm in ARMS:
        model=build_alignment(config,arm,42).eval(); before=frozen_digest(model)
        torch.cuda.reset_peak_memory_stats()
        if bank is None: bank=build_bank(model,records,stats,rows,'cuda')
        with torch.no_grad(): result=model(batch['cells'],batch['mask'],batch['current'])
        gap=float((result['logits'].cpu()-expected).abs().max()); gaps[arm]=gap
        if gap!=0: raise RuntimeError(f'{arm} zero-step original B mismatch {gap}')
        if arm=='B_both':
            for objective in ('semantic','contrastive'):
                model.train(); model.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False):
                    out=model(batch['cells'],batch['mask'],batch['current'])
                    losses=alignment_losses(model,out,batch,bank,config['training'])
                (losses[objective]*256).backward()
                grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads): raise RuntimeError('nonfinite AM gradients')
                token=float(model.tokens.project.weight.grad.abs().max())
                if token<=0: raise RuntimeError('objective misses student TokenAdapter')
                row={'loss':float(losses[objective].detach()),'student_token_gradient_scaled':token}
                if objective=='semantic':
                    row['modulation_affine_gradient_scaled']=float(model.modulator.affine[-1].weight.grad.abs().max())
                    if row['modulation_affine_gradient_scaled']<=0: raise RuntimeError('CE misses modulation output layer')
                gradients[objective]=row
            if any(p.grad is not None for p in model.reference.parameters()): raise RuntimeError('reference teacher received gradient')
            if any(p.grad is not None for p in model.former.adapter.parameters()): raise RuntimeError('Qwen received gradient')
            model.eval()
            with torch.no_grad():
                mask=batch['mask'].clone(); mask[0,6:]=False
                a=model(batch['cells'],mask,batch['current']); altered=batch['cells'].clone(); altered[0,6:]*=-100
                b=model(altered,mask,batch['current'])
                pad=max(float((a[k]-b[k]).abs().max()) for k in ('logits','query','reference'))
                if pad!=0: raise RuntimeError('AM padding/future mismatch')
        if before!=frozen_digest(model): raise RuntimeError('AM frozen parameters changed')
        peak=max(peak,torch.cuda.max_memory_allocated())
        del model; gc.collect(); torch.cuda.empty_cache()
    if peak>9*1024**3: raise RuntimeError('AM real preflight exceeds 9GiB allocated')
    vectors=bank['vectors'].cpu().numpy(); labels=bank['labels'].cpu().numpy()
    np.savez_compressed(path/'reference-bank.npz',vectors=vectors,labels=labels,origins=rows)
    report={'passed':True,**preflight_binding(config,source_manifest()),'origins':origins,'max_context':32,'batch_size':2,'zero_step_logit_gaps':gaps,
            'gradients':gradients,'padding_future_max_abs':pad,'frozen_unchanged':True,
            'peak_cuda_allocated':peak,'seconds':time.time()-started,'bank_size':len(rows),
            'bank_vectors_sha256':hashlib.sha256(vectors.tobytes()).hexdigest(),
            'reference_bank_sha256':sha256(path/'reference-bank.npz')}
    write_json(path/'complete.json',report); return report


def run(config,output,prepare_only=False):
    validate_config(config); output=Path(output); started=time.time()
    if not output.exists(): output.mkdir(parents=True,exist_ok=False)
    elif prepare_only or not (output/'preflight/complete.json').exists() or (output/'jobs').exists():
        raise RuntimeError('refusing duplicate/incomplete AM run')
    # Reject drift before touching the prepared run's configuration or status.
    if (output/'preflight/complete.json').exists():
        proof=json.loads((output/'preflight/complete.json').read_text())
        validate_preflight_binding(config,proof,source_manifest())
        saved=json.loads((output/'config.json').read_text())
        if saved!=config: raise ValueError('AM saved config changed')
    try:
        if not (output/'config.json').exists(): write_json(output/'config.json',config)
        if (output/'data').exists():
            with tempfile.TemporaryDirectory(prefix='harrad_am_') as directory:
                records,stats,design,panels,rows=prepare_design(config,Path(directory)/'data')
                for name in ('design.npz','target-stats.npz','am-design.npz'):
                    with np.load(output/'data'/name) as old,np.load(Path(directory)/'data'/name) as fresh:
                        if set(old.files)!=set(fresh.files): raise RuntimeError('AM prepared keys changed')
                        for key in old.files: np.testing.assert_array_equal(old[key],fresh[key])
        else: records,stats,design,panels,rows=prepare_design(config,output/'data')
        if not (output/'preflight/complete.json').exists():
            write_json(output/'status.json',{'phase':'preflight','updated_unix':time.time()})
            real_preflight(config,records,stats,rows,output/'preflight')
        if prepare_only:
            write_json(output/'status.json',{'phase':'prepared','updated_unix':time.time()}); return
        proof=json.loads((output/'preflight/complete.json').read_text()); bankfile=output/'preflight/reference-bank.npz'
        if not proof['passed'] or sha256(bankfile)!=proof['reference_bank_sha256']: raise RuntimeError('AM bank/preflight changed')
        with np.load(bankfile) as arrays:
            np.testing.assert_array_equal(arrays['origins'],rows)
            bank={'vectors':torch.from_numpy(arrays['vectors'].copy()).cuda(),
                  'labels':torch.from_numpy(arrays['labels'].copy()).cuda(),'origins':rows}
        (output/'jobs').mkdir(exist_ok=False); completed={}
        for seed in config['seeds']:
            for arm in config['arms']:
                name=f'{arm}-seed{seed}'; seed_everything(seed,4)
                write_json(output/'status.json',{'phase':'training','current_job':name,'completed_jobs':len(completed),
                           'total_jobs':12,'updated_unix':time.time()})
                model=build_alignment(config,arm,seed)
                completed[name]=fit_alignment(model,records,stats,design[f'schedule_{seed}'],panels,bank,config['training'],output/'jobs'/name)
                write_json(output/'completed-jobs.json',completed)
                del model; gc.collect(); torch.cuda.empty_cache()
        for row in config['locks'].values():
            if sha256(row['path'])!=row['sha256']: raise RuntimeError('original source changed after AM training')
        result={'phase':'complete','successful_steps':sum(v['successful_steps'] for v in completed.values()),
                'attempts':sum(v['attempts'] for v in completed.values()),'jobs':completed,'test_signal_values_read':False,
                'seconds_formal':time.time()-started,'completed_unix':time.time()}
        write_json(output/'complete.json',result); write_json(output/'status.json',{'phase':'complete','updated_unix':time.time()})
    except BaseException as exc:
        failed={'phase':'failed','error':repr(exc),'traceback':traceback.format_exc(),'updated_unix':time.time()}
        write_json(output/'failed.json',failed); write_json(output/'status.json',failed); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--prepare-only',action='store_true'); args=parser.parse_args()
    run(json.loads(Path(args.config).read_text()),args.output,prepare_only=args.prepare_only)

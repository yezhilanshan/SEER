"""Durable step1 streaming verification -> step2 frozen-backbone semantic warmup."""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from .event_semantic import build_production,semantic_loss
from .event_data import prepare_features,event_rows,EventDataset,event_collate
from .event_training import frozen_digest,overfit_semantic,fit_semantic,evaluate_continuous
from .preprocess import sha256,write_json
from .training import seed_everything


def real_preflight(model,rows,output):
    train=EventDataset([r for r in rows if r['role']=='train'][:2]); batch=event_collate([train[0],train[1]])
    cells=batch['cells'][:,:13].cuda().requires_grad_(); mask=batch['mask'][:,:13].cuda(); labels=batch['labels'].cuda()
    frozen=frozen_digest(model)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=256); model.train()
    with torch.autocast('cuda',dtype=torch.float16): loss=semantic_loss(model(cells,mask),labels,torch.ones(12,device='cuda'))
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    finite=bool(gradients) and all(torch.isfinite(g).all() for g in gradients)
    token_grad=float(model.tokens.project.weight.grad.abs().max())
    input_grad=float(cells.grad.abs().max()) if cells.grad is not None else 0.
    if not finite or token_grad<=0 or input_grad<=0: raise RuntimeError('real Qwen gradient preflight failed')
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    scaler.step(opt); scaler.update()
    if frozen!=frozen_digest(model): raise RuntimeError('frozen Qwen changed in preflight')
    model.eval()
    with torch.no_grad():
        a=model(cells.detach(),mask)
        changed=cells.detach().clone(); changed[:,6:]*=-3
        b=model(changed,mask)
        gap=float((a['prefix'][3]['logits']-b['prefix'][3]['logits']).abs().max())
        if gap!=0: raise RuntimeError(f'real causal prefix changed by future: {gap}')
        padded=model(torch.nn.functional.pad(cells.detach(),(0,0,0,0,0,4)),torch.nn.functional.pad(mask,(0,4),value=False))
        padgap=float((a['final_logits']-padded['final_logits']).abs().max())
        if padgap!=0: raise RuntimeError(f'real padding changed state: {padgap}')
    result={'passed':True,'loss':float(loss.detach()),'token_projection_grad_max':token_grad,'input_grad_max_scaled':input_grad,
            'finite_gradients':finite,'frozen_base_unchanged':True,'future_prefix_max_abs':gap,'padding_max_abs':padgap,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'frozen_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad)}
    write_json(Path(output)/'real-qwen-preflight.json',result)
    return result


def run(config,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False); started=time.time()
    try:
        write_json(output/'config.json',config)
        seed_everything(42,4)
        for key in ('radar_checkpoint','normalizers'):
            path=Path(config[key] if key in config else Path(config['baseline_output'])/'normalizers.npz')
            if sha256(path)!=config[key+'_sha256']: raise ValueError(f'{key} changed')
        if sha256(Path(config['baseline_output'])/'canonical-manifest.json')!=config['baseline_manifest_sha256']:
            raise ValueError('baseline manifest changed')
        expected={'tbptt_updates':8,'prefix_updates':[3,6,12],'prefix_loss_weight':.25,'frames_per_update':2,
                  'freeze_radar':True,'freeze_qwen_including_lora':True,'text_generation_loss':False}
        if any(config['training'][k]!=v for k,v in expected.items()): raise ValueError('unsupported warmup contract change')
        qwen=Path(config['qwen_path'])
        llm_locks={p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in qwen.iterdir() if p.is_file()}
        write_json(output/'input-lock.json',{'qwen_files':llm_locks,'radar_checkpoint':config['radar_checkpoint'],
                  'radar_checkpoint_sha256':config['radar_checkpoint_sha256'],'normalizers_sha256':config['normalizers_sha256'],
                  'scope':'train/validation only; official instance reset; classification warmup, no autonomous Reset'})
        records=prepare_features(config,output)
        rows=event_rows(records)
        seed_everything(42,4); model=build_production(config)
        real_preflight(model,rows,output); del model; gc.collect(); torch.cuda.empty_cache()
        seed_everything(1701,4); model=build_production(config)
        write_json(output/'status.json',{'phase':'semantic_overfit_gate','updated_unix':time.time()})
        overfit_semantic(model,rows,config['training'],output/'gate','cuda')
        del model; gc.collect(); torch.cuda.empty_cache()
        seed_everything(config['training']['seed'],4); model=build_production(config)
        write_json(output/'status.json',{'phase':'semantic_training','updated_unix':time.time()})
        result=fit_semantic(model,rows,config['training'],output/'training','cuda')
        write_json(output/'status.json',{'phase':'continuous_validation_diagnostic','updated_unix':time.time()})
        continuous=evaluate_continuous(model,records,output,'cuda')
        if sha256(config['radar_checkpoint'])!=config['radar_checkpoint_sha256']: raise RuntimeError('source radar weights changed')
        result={'phase':'complete','training':result,'continuous_validation':continuous,'seconds':time.time()-started,
                'completed_unix':time.time(),'test_signal_values_read':False}
        write_json(output/'complete.json',result); write_json(output/'status.json',{'phase':'complete','updated_unix':time.time()})
    except BaseException as exc:
        failure={'phase':'failed','error':repr(exc),'traceback':traceback.format_exc(),'seconds':time.time()-started,'updated_unix':time.time()}
        write_json(output/'failed.json',failure); write_json(output/'status.json',failure); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); run(json.loads(Path(args.config).read_text()),args.output)

"""Read-only numerical diagnosis; never perform an optimizer update."""
import random
from pathlib import Path
import torch
import yaml
from continuous_mmwave.parrad_stage4_runner import load_frontend, QueryCompressor, examples_for, objective

settings=yaml.safe_load(Path('configs/parrad-stage4-memory-pilot-v2.yaml').read_text())['stage4']
torch.manual_seed(42); torch.set_num_threads(1)
runtime,tokenizer,_=load_frontend(settings,torch.device('cuda'))
payload=torch.load(Path(settings['output_dir'])/'events.pt',map_location='cpu',weights_only=False)
model=QueryCompressor(896,**settings['compressor']).cuda().train()
fit=examples_for(payload['events'],'fit'); order=list(range(len(fit))); random.Random(42).shuffle(order)
example=fit[order[0]]
rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state_all()
for scale in (65536.,256.,1.):
    torch.set_rng_state(rng); torch.cuda.set_rng_state_all(cuda_rng); model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.float16):
        metrics=objective(model,example,runtime.controller.event_former.adapter,
                          runtime.controller.semantic_projection,tokenizer,'cuda',1.)
    (metrics['loss']*scale).backward()
    failed=[name for name,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    norm=float(torch.linalg.vector_norm(torch.stack([p.grad.float().norm()/scale for p in model.parameters() if p.grad is not None])))
    print(dict(scale=scale,loss=float(metrics['loss']),level=example[0],nonfinite_parameters=failed,norm=norm),flush=True)

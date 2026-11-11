"""Independent checkpoint readback and fixed post-training generation controls."""
import json
from pathlib import Path
import torch
import yaml
from continuous_mmwave.parrad_stage4_runner import (sha,atomic_json,load_frontend,QueryCompressor,
    examples_for,compress_example,training_summary,SUMMARY_PROMPT,upstream_digest)
from continuous_mmwave.stages.parrad_stage4 import generate_from_memory

torch.set_num_threads(1)
settings=yaml.safe_load(Path('configs/parrad-stage4-memory-pilot-v2.yaml').read_text())['stage4']
root=Path(settings['output_dir'])
rows=[json.loads(line) for line in (root/'training.jsonl').read_text().splitlines()]
assert [r['step'] for r in rows]==list(range(1,101))
result=json.loads((root/'training-result.json').read_text()); checks=[]
for step in (20,50,100):
    path=root/f'compressor-step{step}.pt'
    state=torch.load(path,map_location='cpu',weights_only=False)
    optimizer_steps=sorted({int(s['step']) for s in state['optimizer_state']['state'].values()})
    assert state['step']==step and optimizer_steps==[step]
    assert all(torch.isfinite(t).all() for t in state['model_state'].values())
    assert sha(path)==result['checkpoints'][str(step)]
    checks.append(dict(step=step,optimizer_steps=optimizer_steps,sha256=sha(path),finite=True))
runtime,tokenizer,_=load_frontend(settings,torch.device('cuda'))
assert upstream_digest(runtime)==result['frozen_upstream_digest']==state['frozen_upstream_digest']
payload=torch.load(root/'events.pt',map_location='cpu',weights_only=False)
assert state['cache_sha256']==sha(root/'events.pt')
model=QueryCompressor(**state['model_settings']).cuda().eval(); model.load_state_dict(state['model_state'])
torch.save(dict(model_settings=state['model_settings'],model_state=state['model_state'],
                step=100,provenance=state['provenance']),root/'compressor-inference.pt')
example=examples_for(payload['events'],'dev')[0]
adapter=runtime.controller.event_former.adapter
controls=[]
with torch.no_grad():
    summary,source=compress_example(model,example,'cuda')
    for name,context in [('summary_plain',summary),('raw_plain',source),('empty_plain',source[:,:0])]:
        ids=tokenizer(SUMMARY_PROMPT,return_tensors='pt',add_special_tokens=False)['input_ids'].cuda()
        embedded=torch.cat((context.to(adapter.compute_dtype),adapter.embed_input_ids(ids)),1)
        generated=adapter.model.generate(inputs_embeds=embedded,attention_mask=torch.ones(embedded.shape[:2],device='cuda',dtype=torch.long),
                                         do_sample=False,max_new_tokens=96,eos_token_id=tokenizer.eos_token_id,
                                         pad_token_id=tokenizer.pad_token_id,use_cache=True)
        controls.append(dict(condition=name,answer=tokenizer.decode(generated[0],skip_special_tokens=True)))
    controls.append(dict(condition='summary_chat',answer=generate_from_memory(adapter,tokenizer,summary,
                          'Summarize the events in chronological order.')))
atomic_json(root/'generation-controls.json',dict(posthoc_diagnostic=True,checkpoint_step=100,
    source_ids=[e['uid'] for e in example[1]],reference=training_summary(example[1]),conditions=controls,
    description='First development L1 group, fixed four conditions, no model update or selection'))
atomic_json(root/'readback-checks.json',dict(status='passed',actual_updates=len(rows),checkpoints=checks,
    training_level_counts={str(level):sum(r['level']==level for r in rows) for level in (1,2)},
    frozen_upstream_unchanged=True,inference_sha256=sha(root/'compressor-inference.pt'),
    cache_sha256=sha(root/'events.pt'),training_status=json.loads((root/'status-train.json').read_text())['status'],
    replay_status=json.loads((root/'status-replay.json').read_text())['status']))
print(json.dumps(dict(readback='passed',generation_controls=controls)),flush=True)

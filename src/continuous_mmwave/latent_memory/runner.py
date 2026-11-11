"""Independent prepare/probe/train/evaluate/replay CLI. No training occurs implicitly."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from continuous_mmwave.harrad.preprocess import sha256,write_json
from continuous_mmwave.stages.parrad_stage4 import chat_prompt_ids
from .config import LatentMemoryConfig
from .data import load_records,query_cutoffs,prefix_reference,runtime_record,PrefixExamples
from .frontend import ContinuousMemoryStream
from .evaluation import parse_description,sequence_metrics,state_metrics,aggregate
from .training import frozen_digest,load_checkpoint,train_steps


def build_model(config, device):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from .llm_backend import FrozenCausalBackend
    from .model import LatentMemoryModel
    torch.manual_seed(config.seed)
    tokenizer=AutoTokenizer.from_pretrained(config.llm_path,local_files_only=True)
    if tokenizer.eos_token_id is None: raise ValueError('tokenizer needs EOS')
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    llm=AutoModelForCausalLM.from_pretrained(config.llm_path,local_files_only=True,
                torch_dtype=torch.float16 if device=='cuda' else torch.float32,attn_implementation='eager')
    backend=FrozenCausalBackend(llm)
    backend.checkpoint_reads=config.gradient_checkpointing
    model=LatentMemoryModel(backend,config.lmt_counts,config.rounds,config.top_k,config.fps,config.time_encoding)
    if config.adapter_checkpoint: model.load_local_adapter(config.adapter_checkpoint,config.adapter_sha256)
    model=model.to(device)
    return model,tokenizer,chat_prompt_ids(tokenizer,config.fixed_instruction,device)


def source_manifest(config, records):
    package=Path(__file__).resolve().parent.parent
    llm_files={str(p.relative_to(config.llm_path)):sha256(p) for p in sorted(Path(config.llm_path).rglob('*')) if p.is_file()}
    if not llm_files: raise ValueError('missing local LLM files')
    return dict(schema=1,config=config.to_dict(),manifest_sha256=sha256(config.manifest_path),
                cells_sha256={r['file']:r['artifacts']['cells.npy']['sha256'] for r in records},
                adapter_sha256=sha256(config.adapter_checkpoint) if config.adapter_checkpoint else None,
                llm_files=llm_files,code_sha256={str(p.relative_to(package)):sha256(p) for p in sorted(package.rglob('*.py'))})


def probe(model, tokenizer, prompt, row, config, output):
    from .data import rebuild_memory
    before=frozen_digest(model)
    cutoff=min(config.query_frames,row['frames'])
    memory=rebuild_memory(runtime_record(row),cutoff,config)
    result=model(memory,prompt)
    text=prefix_reference(row['instances'],cutoff)['text']
    answer=tokenizer.encode(text,add_special_tokens=False)+[tokenizer.eos_token_id]
    loss=model.backend.answer_loss(result['latents'],prompt,torch.tensor([answer],device=prompt.device))
    loss.backward()
    norms={n:float(p.grad.norm()) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
    if (not norms or not all(np.isfinite(v) for v in norms.values())
        or not any(v>0 for n,v in norms.items() if n.startswith('projector.'))
        or any(p.grad is None or not p.grad.abs().sum()>0 for p in model.initial_lmts)):
        raise RuntimeError('missing/nonfinite input gradient')
    model.zero_grad(set_to_none=True)
    # Artificial complete history exercises long-memory isolation even on a short real prefix.
    stream=ContinuousMemoryStream(capacities=(2,2,2),group_size=2)
    stream.push(torch.arange(32.).view(32,1,1).expand(-1,4,64).clone()/32)
    isolated=stream.snapshot()
    with torch.no_grad():
        a=model(isolated,prompt)['latents'][:,:len(model.initial_lmts[0])].clone()
        for item in isolated.levels[1]+isolated.levels[2]: item.values.add_(10)
        b=model(isolated,prompt)['latents'][:,:len(model.initial_lmts[0])]
        if not torch.equal(a,b): raise RuntimeError('short-group information leak')
        generation=model.backend.generate(result['latents'].detach(),prompt,tokenizer,config.max_new_tokens)
        saved=model.learned_state(); torch.save(saved,output/'learned-readback.pt')
        for p in model.parameters():
            if p.requires_grad: p.zero_()
        model.load_learned(torch.load(output/'learned-readback.pt',weights_only=True,map_location='cpu'))
        replay=model(memory,prompt)
        gap=float((replay['latents']-result['latents']).abs().max())
        if gap!=0: raise RuntimeError(f'learned state replay mismatch {gap}')
    unchanged=frozen_digest(model)==before
    if not unchanged: raise RuntimeError('frozen parameter change')
    report=dict(loss=float(loss.detach()),input_gradient_nonzero=True,gradient_norms=norms,
                frozen_unchanged=unchanged,group_isolation_passed=True,learned_reload_max_abs=gap,
                cutoff_frame=cutoff,trace=result['trace'],generation=generation,
                checkpoint_reads=config.gradient_checkpointing,
                scope='engineering probe; no optimizer update; untrained new latent-memory path')
    write_json(output/'probe.json',report)
    return report


@torch.no_grad()
def replay_records(model, tokenizer, prompt, records, config, output, *, rounds=None, max_queries=None, fixed_evidence=False):
    rows=[]; model.eval()
    if model.backend.device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    for row in records:
        runtime=runtime_record(row)
        values=np.load(runtime['cells_path'],mmap_mode='r',allow_pickle=False)
        stream=ContinuousMemoryStream(config.capacities,config.group_size); cursor=0
        for cutoff in query_cutoffs(runtime['frames'],config.query_frames):
            stream.push(torch.from_numpy(np.array(values[cursor:cutoff],copy=True))); cursor=cutoff
            if model.backend.device.type=='cuda': torch.cuda.synchronize()
            started=time.perf_counter()
            result=model(stream.snapshot(),prompt,rounds=rounds,fixed_evidence=fixed_evidence)
            generated=model.backend.generate(result['latents'],prompt,tokenizer,config.max_new_tokens)
            if model.backend.device.type=='cuda': torch.cuda.synchronize()
            reference=prefix_reference(row['instances'],cutoff)
            parsed=parse_description(generated['text'])
            prediction=dict(file=runtime['file'],cutoff_frame=cutoff,cutoff_seconds=cutoff/config.fps,
                            reference=reference,predicted_sequence=parsed,**generated,
                            metrics=sequence_metrics(reference['sequence'],parsed),trace=result['trace'],
                            state_metrics=state_metrics(reference['sequence'],reference['ongoing'],generated['text']),
                            rounds=result['rounds'],query_seconds=time.perf_counter()-started,
                            llm_group_read_calls=3*(1+result['rounds']))
            with (output/'predictions.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(prediction,ensure_ascii=False)+'\n')
            rows.append(prediction)
            print(f"replay {runtime['file']} cutoff={cutoff} seconds={prediction['query_seconds']:.3f}",flush=True)
            if max_queries is not None and len(rows)>=max_queries: return rows
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','probe','train','evaluate','replay'])
    parser.add_argument('--config',required=True); parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--steps',type=int); parser.add_argument('--resume'); parser.add_argument('--checkpoint')
    parser.add_argument('--role',choices=['train','validation'],default='validation')
    parser.add_argument('--max-recordings',type=int); parser.add_argument('--max-queries',type=int)
    parser.add_argument('--rounds',type=int); parser.add_argument('--fixed-evidence',action='store_true')
    parser.add_argument('--allow-untrained',action='store_true')
    args=parser.parse_args(argv)
    if args.mode=='train' and (args.steps is None or args.steps<1): parser.error('train requires explicit --steps > 0')
    if args.mode in ('replay','evaluate') and not args.checkpoint and not args.allow_untrained:
        parser.error('evaluate/replay require --checkpoint or explicit --allow-untrained')
    if any(x is not None and x<=0 for x in (args.max_recordings,args.max_queries)): parser.error('limits must be positive')
    if args.rounds is not None and args.rounds<0: parser.error('rounds must be nonnegative')
    if args.mode=='train' and (args.max_recordings or args.max_queries or args.rounds is not None):
        parser.error('training uses the complete configured train query panel; change config rather than silently subset')
    config=LatentMemoryConfig.from_dict(json.loads(Path(args.config).read_text(encoding='utf-8-sig')))
    output=Path(args.output)
    if output.exists(): raise FileExistsError(f'refuse existing output: {output}')
    records=load_records(config.manifest_path,config.manifest_sha256)
    if args.mode=='prepare':
        output.mkdir(parents=True)
        write_json(output/'summary.json',dict(recordings=len(records),frames=sum(r['frames'] for r in records),
                    config=config.to_dict(),manifest_sha256=sha256(config.manifest_path),test_signals_read=False))
        return
    source=source_manifest(config,records)
    model,tokenizer,prompt=build_model(config,args.device)
    if args.mode=='train':
        selected=[r for r in records if r['role']=='train']
        examples=PrefixExamples(selected,tokenizer,prompt,config)
        train_steps(model,examples,output,steps=args.steps,source=source,seed=config.seed,
                    learning_rate=config.learning_rate,weight_decay=config.weight_decay,grad_clip=config.grad_clip,
                    checkpoint_every=config.checkpoint_every,resume=args.resume)
        return
    output.mkdir(parents=True); write_json(output/'source.json',source)
    try:
        if args.mode=='probe':
            row=next(r for r in records if r['role']=='train')
            report=probe(model,tokenizer,prompt,row,config,output)
            write_json(output/'status.json',dict(status='complete',mode='probe',optimizer_updates=0))
            return report
        saved=load_checkpoint(args.checkpoint,model,source) if args.checkpoint else None
        selected=[r for r in records if r['role']==args.role]
        if args.max_recordings: selected=selected[:args.max_recordings]
        if not selected: raise ValueError('no recordings for requested role')
        rows=replay_records(model,tokenizer,prompt,selected,config,output,rounds=args.rounds,
                            max_queries=args.max_queries,fixed_evidence=args.fixed_evidence)
        summary=aggregate(rows)
        summary.update(checkpoint_step=None if saved is None else saved['step'],untrained=saved is None,
                       rounds=config.rounds if args.rounds is None else args.rounds,fixed_evidence=args.fixed_evidence,
                       peak_cuda_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0)
        write_json(output/'summary.json',summary); write_json(output/'status.json',dict(status='complete',mode=args.mode))
        return summary
    except Exception as error:
        write_json(output/'status.json',dict(status='failed',mode=args.mode,error=repr(error))); raise

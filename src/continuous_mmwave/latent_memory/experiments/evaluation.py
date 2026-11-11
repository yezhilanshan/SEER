"""Evidence controls and text diagnostics share one frozen LM and output grammar."""
import hashlib
import json
from pathlib import Path
import time
import torch
from continuous_mmwave.stages.parrad_stage4 import chat_prompt_ids
from ..data import PHRASES
from .episodes import read_cells,memory_view,render_reference
from .grammar import generate
from .core import read_prefix
from .metrics import score,aggregate


def prepare_examples(episodes, view='raw'):
    result=[]
    for e in episodes:
        cells=read_cells(e)
        result.append({**e,'cells':cells,'memory':memory_view(cells,view)})
    return result


def donor_map(examples):
    result={}
    for target in examples:
        choices=[e for e in examples if e['seconds']==target['seconds'] and e['uid']!=target['uid']
                 and e['reference']['sequence']!=target['reference']['sequence']]
        if not choices: raise ValueError(f"no matched-length different-sequence donor: {target['uid']}")
        result[target['uid']]=min(choices,key=lambda e:hashlib.sha256((target['uid']+e['uid']).encode()).hexdigest())['uid']
    return result


def _append(path,row):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')


@torch.no_grad()
def evaluate(model,examples,prompt,tokenizer,grammar,path,*,arm='E0',rounds=2,control='correct',
             donors=None,view='raw',max_new_tokens=512):
    rows=[]; by_uid={e['uid']:e for e in examples}; model.eval()
    for index,e in enumerate(examples):
        actual=e
        if control=='swapped': actual=by_uid[donors[e['uid']]]
        memory=actual['memory']
        if control=='zero': memory=memory_view(torch.zeros_like(e['cells']),view)
        if control not in ('correct','swapped','zero'): raise ValueError('unknown control')
        if model.backend.device.type=='cuda': torch.cuda.synchronize()
        start=time.perf_counter(); reading=read_prefix(model,memory,prompt,arm,rounds)
        generated=generate(model,reading['prefix'],prompt,tokenizer,grammar,max_new_tokens)
        if model.backend.device.type=='cuda': torch.cuda.synchronize()
        row=dict(uid=e['uid'],file=e['file'],seconds=e['seconds'],arm=arm,rounds=rounds,control=control,
                 actual_evidence_uid=actual['uid'],reference=e['reference'],**generated,
                 metrics=score(e['reference'],generated['text'],generated['truncated']),
                 metrics_vs_actual_evidence=score(actual['reference'],generated['text'],generated['truncated']),
                 trace=reading['trace'],selected=reading.get('selected'),read_calls=reading['read_calls'],
                 prefix_tokens=reading['prefix'].shape[1],
                 prefix_sha256=hashlib.sha256(reading['prefix'].float().cpu().contiguous().numpy().tobytes()).hexdigest(),
                 query_seconds=time.perf_counter()-start)
        _append(path,row); rows.append(row)
        if index%8==0: print(f'evaluate {arm} {control} {index+1}/{len(examples)}',flush=True)
    return rows,aggregate(rows)


@torch.no_grad()
def text_oracle(model,examples,tokenizer,grammar,instruction,path,max_new_tokens=512):
    rows=[]
    for index,e in enumerate(examples):
        original=e['reference']
        cases=[('T0',original)]
        if len(set(original['sequence']))>=2:
            sequence=list(reversed(original['sequence']))
            cases.append(('T-order',dict(sequence=sequence,ongoing=[False]*len(sequence))))
            a,b=list(dict.fromkeys(original['sequence']))[:2]
            cases.append(('T-alternating',dict(sequence=[a,b,a,b],ongoing=[False]*4)))
        if original['sequence']:
            sequence=[original['sequence'][0]]+original['sequence']
            cases.append(('T-count',dict(sequence=sequence,ongoing=[False]*len(sequence))))
        for kind,reference in cases:
            evidence=[PHRASES[y]+(' (ongoing)' if active else '')
                      for y,active in zip(reference['sequence'],reference['ongoing'])]
            query=instruction+'\nOrdered evidence entries (each entry is a separate instance): '+json.dumps(evidence)
            prompt=chat_prompt_ids(tokenizer,query,model.backend.device)
            prefix=torch.empty(1,0,model.backend.hidden_size,device=model.backend.device)
            generated=generate(model,prefix,prompt,tokenizer,grammar,max_new_tokens)
            row=dict(uid=e['uid']+':'+kind,file=e['file'],kind=kind,reference=reference,evidence=evidence,
                     **generated,metrics=score(reference,generated['text'],generated['truncated']))
            _append(path,row); rows.append(row)
        print(f'text_oracle {index+1}/{len(examples)}',flush=True)
    summaries={kind:aggregate([r for r in rows if r['kind']==kind]) for kind in ('T0','T-order','T-count','T-alternating')}
    return rows,summaries

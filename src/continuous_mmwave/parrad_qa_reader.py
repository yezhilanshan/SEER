"""Sealed independent multi-event reader; single-event teacher is preserved."""
from pathlib import Path

import torch

from continuous_mmwave.parrad_stage4_runner import sha


class QAReaders:
    def __init__(self, single, multi):
        self.single, self.multi = single, multi
        for reader in (single, multi):
            reader.eval(); reader.requires_grad_(False); reader.zero_grad(set_to_none=True)

    def for_size(self, size):
        if size not in (1, 2, 4): raise ValueError('QA supports only 1/2/4 events')
        return self.single if size == 1 else self.multi


def reader_for(teacher, task):
    return teacher.for_size(task['size']) if isinstance(teacher, QAReaders) else teacher


def reader_digest(reader):
    # Same streaming tensor digest convention, but for one module only.
    import hashlib
    digest = hashlib.sha256()
    for name, value in reader.state_dict().items():
        digest.update(f'{name}:{tuple(value.shape)}:{value.dtype}'.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def retrieval_metrics(scores, labels):
    labels=torch.as_tensor(labels,device=scores.device)
    if scores.ndim!=2 or scores.shape[1]!=len(labels) or not torch.isfinite(scores).all():
        raise ValueError('invalid retrieval score matrix')
    classes={}
    for label,row in enumerate(scores):
        relevant=labels==label; count=int(relevant.sum())
        if not count: classes[str(label)]=None; continue
        ranking=torch.argsort(row,descending=True,stable=True)
        hits=relevant[ranking].float(); precision=hits.cumsum(0)/torch.arange(1,len(hits)+1,device=scores.device)
        classes[str(label)]=dict(positives=count,AP=float((precision*hits).sum()/count),
            recall_at_1=float(hits[:1].sum()/count),recall_at_5=float(hits[:5].sum()/count),
            top1_hit=bool(hits[0]),ranked_indices=ranking.cpu().tolist())
    present=[v for v in classes.values() if v is not None]
    return dict(classes=classes,mAP=None if not present else sum(v['AP'] for v in present)/len(present),
        macro_recall_at_1=None if not present else sum(v['recall_at_1'] for v in present)/len(present),
        macro_recall_at_5=None if not present else sum(v['recall_at_5'] for v in present)/len(present))


@torch.no_grad()
def retrieval_snapshot(runtime,tokenizer,events,device):
    from continuous_mmwave.parrad_stage4_qa import ACTIVITIES
    from continuous_mmwave.stages.parrad_stage4 import encode_query
    teacher=runtime.controller.event_former.adapter
    projection=runtime.controller.semantic_projection
    selected=[e for e in events.values() if e['cohort']=='dev']
    # Original raw L0 tokens and original text encoder/P only. No new QA/fusion.
    keys=torch.stack([projection(e['tokens'][None].to(device))[0] for e in selected])
    queries=torch.stack([encode_query(teacher,projection,tokenizer,q,device) for q in ACTIVITIES])
    scores=queries@keys.T; labels=[int(e['label']) for e in selected]
    return dict(scope='original P; cached dev99 L0 events; 14 fixed literal activity queries',
        event_uids=[e['uid'] for e in selected],labels=labels,queries=list(ACTIVITIES),
        query_keys=queries.cpu().tolist(),event_keys=keys.cpu().tolist(),scores=scores.cpu().tolist(),
        metrics=retrieval_metrics(scores,labels))


def load_multi_reader(settings, single, protected, original_digest, device):
    path = Path(settings['qa_reader'])
    if sha(path) != settings['qa_reader_sha256']: raise ValueError('QA reader SHA mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    source = saved['source']; config = saved['reader_settings']
    if (saved['step'] != 500 or source['original_upstream_digest'] != original_digest
        or source['protected_sha256'] != {str(p):h for p,h in protected.items()}):
        raise ValueError('QA reader source mismatch')
    model_dir = Path(source['base_model'])
    actual_files = {str(p.relative_to(model_dir)):sha(p) for p in sorted(model_dir.rglob('*')) if p.is_file()}
    if not actual_files or actual_files != source['base_files']: raise ValueError('QA reader base source mismatch')
    from transformers import AutoModelForCausalLM
    from peft import set_peft_model_state_dict
    from continuous_mmwave.parrad_qa_text_decoder import create_reader, adapter_state, frozen_base_digest
    lm = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32)
    if (lm.config.model_type != 'qwen2' or lm.config.num_hidden_layers != config['layers']
        or lm.config.hidden_size != config['dim'] or config['dim'] != single.hidden_size):
        raise ValueError('QA reader architecture mismatch')
    reader = create_reader(lm, rank=config['rank']).to(device)
    expected = adapter_state(reader)
    if (set(expected) != set(saved['adapter_state']) or any(expected[k].shape != v.shape
        or not torch.isfinite(v).all() for k,v in saved['adapter_state'].items())):
        raise ValueError('QA reader adapter key/shape/value mismatch')
    set_peft_model_state_dict(reader.model, saved['adapter_state'])
    if not all(torch.equal(v, adapter_state(reader)[k]) for k,v in saved['adapter_state'].items()):
        raise ValueError('QA reader LoRA readback mismatch')
    if frozen_base_digest(reader) != source['new_frozen_base_digest']: raise ValueError('QA reader base digest mismatch')
    if {p.data_ptr() for p in reader.parameters()} & {p.data_ptr() for p in single.parameters()}:
        raise RuntimeError('QA readers share parameter storage')
    reader.eval(); reader.requires_grad_(False); reader.zero_grad(set_to_none=True)
    return reader, dict(path=str(path), sha256=sha(path), updates=500, reader_settings=config,
        source=source, frozen_digest=reader_digest(reader))

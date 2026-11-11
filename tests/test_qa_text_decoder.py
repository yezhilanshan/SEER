import importlib

import pytest
import torch

from continuous_mmwave.parrad_event_boundary_tasks import build_boundary_tasks
from tests.test_short_context_qa import events


def runner():
    name='continuous_mmwave.parrad_qa_text_decoder'
    assert importlib.util.find_spec(name), 'independent text QA decoder runner is missing'
    return importlib.import_module(name)


def tiny():
    transformers=pytest.importorskip('transformers')
    pytest.importorskip('peft')
    cfg=transformers.Qwen2Config(vocab_size=128,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        max_position_embeddings=128)
    return transformers.Qwen2ForCausalLM(cfg)


def test_new_decoder_updates_only_own_lora_then_frozen_reader_passes_prefix_gradients():
    r=runner(); original=tiny(); original.requires_grad_(False)
    old={k:v.clone() for k,v in original.state_dict().items()}
    reader=r.create_reader(tiny(),rank=2)
    assert not {p.data_ptr() for p in reader.parameters()} & {p.data_ptr() for p in original.parameters()}
    names=[n for n,p in reader.named_parameters() if p.requires_grad]
    assert names and all('lora_' in n and ('q_proj' in n or 'v_proj' in n) for n in names)
    assert any('layers.0.' in n for n in names) and any('layers.1.' in n for n in names)
    digest=r.frozen_base_digest(reader)
    ids=torch.tensor([[3,4,5,6,7,8]])
    attention=torch.ones_like(ids); target=torch.zeros_like(ids,dtype=torch.bool); target[:,-2:]=True
    optimizer=torch.optim.AdamW([p for p in reader.parameters() if p.requires_grad],lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        r.answer_loss(reader,ids,attention,target).backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in reader.parameters() if p.requires_grad)
        optimizer.step()
    assert r.frozen_base_digest(reader)==digest
    assert all(torch.equal(v,old[k]) for k,v in original.state_dict().items())
    reader.requires_grad_(False); reader.eval(); reader.zero_grad(set_to_none=True)
    prefix=torch.randn(1,16,32,requires_grad=True)
    reader.caption_logits(prefix,ids,attention).square().mean().backward()
    assert prefix.grad.abs().sum()>0 and all(p.grad is None for p in reader.parameters())


def test_text_batches_keep_locked_budget_and_fit_only_balanced_opposite_sources():
    r=runner(); rows=events(); taskset=build_boundary_tasks(rows,42)
    batches=r.text_batches(taskset,{e['uid']:e for e in rows},42)
    assert len(batches)==500
    assert batches==r.text_batches(taskset,{e['uid']:e for e in rows},42)
    assert sum(b[0]['size']==2 for b in batches)==250
    for batch in batches:
        assert [t['answer'] for t in batch[:2]]==['Yes','No']
        assert all(t['cohort']=='fit' and t['size']>1 for t in batch)
        assert batch[0]['query']!=batch[1]['query'] # Different actual text evidence.
        assert all('supplied memory' not in t['query'] and 'Evidence:' in t['query'] for t in batch)


def test_answer_loss_aligns_next_token_and_excludes_prompt():
    r=runner()
    from types import SimpleNamespace
    logits=torch.randn(1,5,12,requires_grad=True)
    reader=SimpleNamespace(compute_dtype=torch.float32,hidden_size=4,
        caption_logits=lambda context,ids,mask:logits)
    ids=torch.tensor([[1,2,3,4,5]]); mask=torch.ones_like(ids); target=torch.zeros_like(ids,dtype=torch.bool)
    target[:,-2:]=True
    loss=r.answer_loss(reader,ids,mask,target)
    expected=torch.nn.functional.cross_entropy(logits[:,2:4].reshape(-1,12),ids[:,3:].reshape(-1))
    assert torch.equal(loss,expected)
    loss.backward(); assert logits.grad[:,:2].abs().sum()==0 and logits.grad[:,4:].abs().sum()==0
    with pytest.raises(ValueError): r.answer_loss(reader,ids,mask,torch.zeros_like(target))


def test_adapter_checkpoint_roundtrip_does_not_export_frozen_backbone(tmp_path):
    r=runner(); torch.manual_seed(42); reader=r.create_reader(tiny(),rank=2)
    optimizer=torch.optim.AdamW([p for p in reader.parameters() if p.requires_grad],lr=.01)
    ids=torch.tensor([[3,4,5,6]]); mask=torch.ones_like(ids); target=torch.ones_like(ids,dtype=torch.bool)
    r.answer_loss(reader,ids,mask,target).backward(); optimizer.step()
    path=tmp_path/'checkpoint.pt'; source={'test':'native tiny'}
    r.save_checkpoint(path,reader,optimizer,1,source)
    saved=torch.load(path,weights_only=False)
    assert saved['adapter_state'] and all('lora_' in n for n in saved['adapter_state'])
    assert r.verify_checkpoint(path,reader,source)['actual_steps']==1
    corrupted=dict(saved,optimizer_names=['wrong']*len(saved['optimizer_names']))
    damaged=tmp_path/'damaged.pt'; torch.save(corrupted,damaged)
    with pytest.raises(RuntimeError,match='parameter'): r.verify_checkpoint(damaged,reader,source)
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    restored=r.create_reader(tiny(),rank=2)
    set_peft_model_state_dict(restored.model,saved['adapter_state'])
    assert all(torch.equal(v,get_peft_model_state_dict(restored.model)[k]) for k,v in saved['adapter_state'].items())


def test_independent_verifier_archives_early_load_failure_without_source(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys
    root=tmp_path/'run'; root.mkdir()
    status=dict(status='failed',phase='load',reason='test missing base model',actual_updates=0,
                settings={},code_sha256={})
    (root/'status.json').write_text(json.dumps(status))
    logs=root.with_name(root.name+'.logs'); logs.mkdir(); (logs/'worker.log').write_text('original failure')
    script=Path(__file__).resolve().parents[1]/'.audit-staging/verify-text-decoder.py'
    result=subprocess.run([sys.executable,str(script),'--root',str(root)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    report=json.loads((root/'independent-readback.json').read_text())
    assert report['status']=='failed' and report['original_failure']==status['reason']
    assert not report['source_checks_performed'] and 'source.json' in report['missing_artifacts']
    assert (root.parent/'stage4-text-decoder-results-v1.tar.gz').exists()

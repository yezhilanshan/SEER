"""Real tiny Qwen tests: group isolation, evolved queries, and frozen-backbone gradients."""
import torch
import pytest


def make_model(seed=9):
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from continuous_mmwave.latent_memory.llm_backend import FrozenCausalBackend
    from continuous_mmwave.latent_memory.model import LatentMemoryModel
    torch.set_num_threads(1); torch.manual_seed(seed)
    config=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
                      num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=1024,
                      bos_token_id=1,eos_token_id=2,pad_token_id=0,attention_dropout=0.)
    config._attn_implementation='eager'
    backend=FrozenCausalBackend(Qwen2ForCausalLM(config))
    return LatentMemoryModel(backend,lmt_counts=(2,2,2),rounds=2,top_k=2,fps=15.)


def make_memory():
    from continuous_mmwave.latent_memory.frontend import ContinuousMemoryStream
    stream=ContinuousMemoryStream(capacities=(2,2,2),group_size=2)
    torch.manual_seed(3); stream.push(torch.randn(32,4,64))
    return stream.snapshot()


def test_bootstrap_and_multiround_evolution_train_inputs_with_frozen_llm():
    model=make_model(); memory=make_memory(); prompt=torch.tensor([[1,4,5]])
    before={n:p.detach().clone() for n,p in model.backend.named_parameters()}
    output=model(memory,prompt)
    assert output['latents'].shape==(1,6,32)
    assert len(output['trace'])==6
    assert {r['round'] for r in output['trace']}=={0,1}
    loss=model.backend.answer_loss(output['latents'],prompt,torch.tensor([[7,8,2]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(z.grad is not None and z.grad.abs().sum()>0 for z in model.initial_lmts)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.projector.parameters())
    assert all(p.grad is None and torch.equal(p,before[n]) for n,p in model.backend.named_parameters())
    assert all(p.grad is None for p in model.local_adapter.parameters())
    model.train(); assert not model.backend.training and not model.local_adapter.training
    zero=model(memory,prompt,rounds=0)
    assert zero['trace']==[] and not torch.allclose(zero['latents'],output['latents'])


def test_short_lmts_cannot_read_long_memory_directly_or_indirectly():
    model=make_model().eval(); memory=make_memory(); prompt=torch.tensor([[1,4]])
    original=model(memory,prompt)['latents'][:,:2].detach()
    assert memory.levels[2]
    for item in memory.levels[1]+memory.levels[2]: item.values.add_(500)
    changed=model(memory,prompt)['latents'][:,:2].detach()
    torch.testing.assert_close(original,changed,rtol=0,atol=0)
    for trace in model(memory,prompt)['trace']:
        assert all(x['level']<=trace['group'] for x in trace['selected'])


def test_each_round_scores_current_lmts_and_records_reproducible_provenance():
    model=make_model().eval(); prompt=torch.tensor([[1,6]]); memory=make_memory()
    a=model(memory,prompt); b=model(memory,prompt)
    assert a['trace']==b['trace']
    # Scores from round 1 must be recalculated from the evolved query, even if top-k is unchanged.
    assert any(a['trace'][g]['scores']!=a['trace'][g+3]['scores'] for g in range(3))
    for row in a['trace']:
        starts=[x['start_frame'] for x in row['selected']]
        assert starts==sorted(starts)
        assert all(x['end_frame']<=32 for x in row['selected'])


def test_single_frame_empty_levels_and_context_overflow_are_explicit():
    from continuous_mmwave.latent_memory.frontend import ContinuousMemoryStream
    model=make_model(); stream=ContinuousMemoryStream(); stream.push(torch.ones(1,4,64))
    out=model(stream.snapshot(),torch.tensor([[1,3]])); assert torch.isfinite(out['latents']).all()
    with pytest.raises(ValueError,match='context'):
        model.backend.read(torch.ones(1,1020,dtype=torch.long),torch.ones(4,32),torch.ones(2,32))


def test_checkpointed_reads_preserve_frozen_eval_gradients():
    model=make_model(); memory=make_memory(); prompt=torch.tensor([[1,4]])
    model.backend.checkpoint_reads=True
    saved_a=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda x:(saved_a.append(x.numel()) or x),lambda x:x):
        a=model(memory,prompt)['latents']; a.square().sum().backward()
    grads={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
    model.zero_grad(set_to_none=True); model.backend.checkpoint_reads=False
    saved_b=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda x:(saved_b.append(x.numel()) or x),lambda x:x):
        b=model(memory,prompt)['latents']; b.square().sum().backward()
    assert sum(saved_a)<sum(saved_b)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    for n,p in model.named_parameters():
        if n in grads: torch.testing.assert_close(p.grad,grads[n],rtol=0,atol=0)


def test_checkpointed_answer_loss_matches_native_shift_and_saves_activations():
    model=make_model(); backend=model.backend
    prompt=torch.tensor([[1,4,5]]); answer=torch.tensor([[7,8,2]])
    prefix=torch.randn(1,6,32,requires_grad=True)
    hidden=torch.cat((prefix,backend.embed(prompt),backend.embed(answer)),1)
    native=backend.lm(inputs_embeds=hidden,use_cache=False).logits
    expected=torch.nn.functional.cross_entropy(native[:,8:11].reshape(-1,64),answer.reshape(-1))
    backend.checkpoint_reads=True; small=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda x:(small.append(x.numel()) or x),lambda x:x):
        actual=backend.answer_loss(prefix,prompt,answer)
    torch.testing.assert_close(actual,expected)
    backend.checkpoint_reads=False; large=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda x:(large.append(x.numel()) or x),lambda x:x):
        backend.answer_loss(prefix,prompt,answer)
    assert sum(small)<sum(large)

import torch
import pytest
from test_latent_memory_model import make_model


class ByteTokenizer:
    eos_token_id=2
    pad_token_id=0
    def encode(self,text,add_special_tokens=False): return [b+3 for b in text.encode()]
    def decode(self,ids,skip_special_tokens=True): return bytes(i-3 for i in ids if i>=3).decode()


def test_grammar_preserves_unbounded_repeats_and_only_constrains_syntax():
    from continuous_mmwave.latent_memory.experiments.grammar import DescriptionGrammar
    tokenizer=ByteTokenizer(); grammar=DescriptionGrammar(tokenizer)
    for text in ['The person stood up, then stood up, then sat down (ongoing).',
                 'No annotated action has been observed yet.']:
        ids=tokenizer.encode(text)+[2]; prefix=[]
        for token in ids:
            assert token in grammar.allowed(prefix)
            prefix.append(token)
        assert grammar.complete(prefix)
    prefix=tokenizer.encode('The person stood up, then ')
    assert tokenizer.encode('swiped left')[0] in grammar.allowed(prefix)
    assert not grammar.complete(tokenizer.encode('The person stood up'))
    with pytest.raises(ValueError): grammar.allowed(tokenizer.encode('The person never '))


def test_new_metrics_penalize_invalid_outputs_and_use_declared_denominator():
    from continuous_mmwave.latent_memory.experiments.metrics import score,aggregate
    reference=dict(sequence=[0,0,1],ongoing=[False,False,False])
    result=score(reference,'The person stood up, then sat down.')
    assert result['edit_score']==pytest.approx(2/3)
    assert result['total_count_mae']==1 and result['class_count_l1']==1
    assert score(reference,'The person never stood up.')['edit_score']==0
    assert score(reference,'The person stood up.',truncated=True)['edit_score']==0
    rows=[dict(file='A',metrics={'edit_score':1.,'sequence_em':1.,'state_em':1.,'total_count_mae':0.,'class_count_l1':0.,'parseable':True,'truncated':False})]*9
    rows += [dict(file='B',metrics={'edit_score':0.,'sequence_em':0.,'state_em':0.,'total_count_mae':1.,'class_count_l1':1.,'parseable':True,'truncated':False})]
    assert aggregate(rows)['record_macro']['edit_score']==.5
    assert aggregate(rows)['record_macro']['state_em']==.5
    partial=score(dict(sequence=[0,1],ongoing=[False,False]),'The person stood up, then sat',truncated=True)
    assert partial['total_count_mae']==1 and partial['class_count_l1']==1
    assert partial['edit_score']==0 and partial['sequence_em']==0


def test_shared_weight_e0_e4_paths_and_first_round_identity():
    from continuous_mmwave.latent_memory.experiments.episodes import raw_memory
    from continuous_mmwave.latent_memory.experiments.core import read_prefix,set_phase
    model=make_model(); memory=raw_memory(torch.randn(240,4,64)); prompt=torch.tensor([[1,4]])
    assert read_prefix(model,memory,prompt,'E0')['prefix'].shape==(1,480,32)
    assert read_prefix(model,memory,prompt,'E1')['prefix'].shape==(1,6,32)
    rag=read_prefix(model,memory,prompt,'E2'); assert rag['prefix'].shape[1]<=24
    assert len({x['start_frame'] for x in rag['selected']})==len(rag['selected'])
    a=read_prefix(model,memory,prompt,'E3',rounds=1)
    b=read_prefix(model,memory,prompt,'E4',rounds=1)
    torch.testing.assert_close(a['prefix'],b['prefix'],rtol=0,atol=0)
    assert a['trace']==b['trace']
    set_phase(model,'alignment')
    assert all(not p.requires_grad for p in model.initial_lmts)
    assert any(p.requires_grad for p in model.projector.parameters())
    set_phase(model,'latent')
    assert all(not p.requires_grad for p in model.projector.parameters())
    assert all(p.requires_grad for p in model.initial_lmts)
    assert not any(p.requires_grad for p in model.backend.parameters())


def test_constrained_hf_generation_starts_from_empty_generated_ids():
    from continuous_mmwave.latent_memory.experiments.grammar import DescriptionGrammar,generate
    model=make_model(); model.backend.lm.resize_token_embeddings(259)
    tokenizer=ByteTokenizer(); grammar=DescriptionGrammar(tokenizer)
    output=generate(model,torch.zeros(1,3,32),torch.tensor([[1,4]]),tokenizer,grammar,max_new_tokens=8)
    assert 0<len(output['token_ids'])<=8
    assert output['token_ids'][0] in grammar.allowed([])
    grammar.allowed(output['token_ids'])

import json
import torch
import pytest
from test_latent_memory_model import make_model
from test_g0g2_core import ByteTokenizer


def fixture():
    from continuous_mmwave.latent_memory.experiments.episodes import raw_memory
    model=make_model(); model.backend.lm.resize_token_embeddings(259)
    reference=dict(sequence=[0],ongoing=[False],text='The person stood up.')
    examples=[dict(uid='fixed:0:60',file='fixed',seconds=4,reference=reference,memory=raw_memory(torch.ones(60,4,64)))]
    return model,examples


def test_stage_training_freezes_llm_and_exact_resume(tmp_path):
    from continuous_mmwave.latent_memory.experiments.training import train_stage,load_stage
    from continuous_mmwave.latent_memory.experiments.grammar import DescriptionGrammar
    from continuous_mmwave.latent_memory.experiments.core import new_state
    grammar=DescriptionGrammar(ByteTokenizer()); prompt=torch.tensor([[1,4]])
    model,examples=fixture(); source={'panel':'fixed'}
    ckpt=train_stage(model,examples,prompt,grammar,tmp_path/'a',steps=1,phase='alignment',source=source)
    assert all(p.grad is None for p in model.backend.parameters())
    resumed,_=fixture(); train_stage(resumed,examples,prompt,grammar,tmp_path/'b',steps=2,phase='alignment',source=source,resume=ckpt)
    direct,_=fixture(); train_stage(direct,examples,prompt,grammar,tmp_path/'c',steps=2,phase='alignment',source=source)
    for n,p in new_state(direct).items(): torch.testing.assert_close(p,new_state(resumed)[n],rtol=0,atol=0)
    with pytest.raises(ValueError,match='source'): load_stage(ckpt,resumed,{'panel':'wrong'})


def test_gates_never_launch_hme_on_unfit_or_prior_only_outputs():
    from continuous_mmwave.latent_memory.experiments.workflow import pilot_gate
    summary={'sequence_em':1.,'edit_score':1.,'state_em':1.}
    result=pilot_gate(oracle_exact=True,correct=summary,null=summary,swapped=summary,expected_cases=32,actual_cases=32)
    assert not result['passed'] and not result['g2_allowed']
    result=pilot_gate(oracle_exact=True,correct=summary,null={'edit_score':.2},swapped={'edit_score':.4},expected_cases=32,actual_cases=32)
    assert result['passed']
    assert not pilot_gate(False,summary,{'edit_score':0},{'edit_score':0},32,32)['passed']


def test_oracle_with_wrong_ongoing_flags_cannot_pass():
    from continuous_mmwave.latent_memory.experiments.workflow import oracle_pass
    assert not oracle_pass([dict(metrics={'sequence_em':1.,'state_em':0.})])
    assert oracle_pass([dict(metrics={'sequence_em':1.,'state_em':1.})])


def test_latent_modes_are_crossed_with_every_duration_instead_of_confounded():
    from continuous_mmwave.latent_memory.experiments.training import scheduled_mode
    schedule=[scheduled_mode(step,'latent',[4,8,16]) for step in range(1,19)]
    assert len(set(schedule))==18
    assert {s[0] for s in schedule}=={4,8,16}

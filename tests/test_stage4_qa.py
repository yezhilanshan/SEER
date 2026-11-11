import torch
from continuous_mmwave.stages.parrad_stage4 import chat_prompt_ids, chat_supervision
from continuous_mmwave.parrad_stage4_qa import (activity_answer, grade_answer, presence_pairs,
    render_context, qa_loss, task_query)
from continuous_mmwave.models.memory import QueryCompressor
from continuous_mmwave.models.semantic import SemanticProjection
from continuous_mmwave.models.event_former import TinyLLMAdapter


class Tokenizer:
    eos_token_id=99
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert len(messages)==1 and messages[0]['role']=='user'
        return 'user:'+messages[0]['content']+'assistant:'
    def encode(self, text, add_special_tokens=False): return [ord(c)%90+1 for c in text]
    def __call__(self,text,add_special_tokens=False,return_tensors='pt'):
        ids=torch.tensor([self.encode(text)])
        return dict(input_ids=ids,attention_mask=torch.ones_like(ids))


def test_chat_teacher_forcing_prefix_exactly_matches_generation_and_masks_roles():
    tokenizer=Tokenizer()
    ids,attention,targets=chat_supervision(tokenizer,'question','Yes','cpu')
    prefix=chat_prompt_ids(tokenizer,'question','cpu')
    torch.testing.assert_close(ids[:,:prefix.shape[1]],prefix)
    assert not targets[:,:prefix.shape[1]].any()
    assert targets[:,prefix.shape[1]:].all() and ids[0,-1]==99
    assert attention.all()


def test_presence_and_sequence_labels_use_selected_evidence_only():
    evidence=[{'label':11},{'label':1},{'label':7}]
    assert activity_answer('presence',evidence,1)=='Yes'
    assert activity_answer('presence',evidence,2)=='No'
    assert activity_answer('first',evidence)==activity_answer('last',evidence[:1])
    assert ';' in activity_answer('summary',evidence)
    assert 'getting into bed' in task_query('presence',1)
    assert grade_answer('Yes.','Yes','presence')
    assert not grade_answer('Maybe yes','Yes','presence')
    assert not grade_answer('No. Yes.','Yes','presence')


def test_opposite_memory_pairs_keep_same_question():
    rows=[dict(uid='a',kind='presence',query='q',answer='Yes'),
          dict(uid='b',kind='presence',query='q',answer='No'),
          dict(uid='c',kind='presence',query='other',answer='No')]
    assert presence_pairs(rows)==[('a','b')]


def test_mixed_context_gradient_reaches_both_levels_not_frozen_l0():
    torch.manual_seed(4)
    model=QueryCompressor(16,internal_dim=16,heads=4,dropout=0)
    projection=SemanticProjection(16,8)
    adapter=TinyLLMAdapter(128,16,2,4)
    for p in projection.parameters(): p.requires_grad_(False)
    events={str(i):dict(uid=str(i),start=float(i),recording='a',tokens=torch.randn(8,16),label=i%14) for i in range(24)}
    selected=[dict(level=2,source_ids=[str(i) for i in range(16)]),
              dict(level=1,source_ids=[str(i) for i in range(16,20)]),
              *[dict(level=0,source_ids=[str(i)]) for i in range(20,24)]]
    context,retention=render_context(model,projection,selected,events,'cpu')
    assert context.shape==(1,40,16)
    torch.testing.assert_close(context[0,-8:],events['23']['tokens'])
    ids,attention,target=chat_supervision(Tokenizer(),'q','Yes','cpu')
    loss=qa_loss(adapter,context,ids,attention,target,retention,2.)
    loss['loss'].backward()
    assert model.level_embedding.weight.grad[1].abs().sum()>0
    assert model.level_embedding.weight.grad[2].abs().sum()>0
    assert all(p.grad is None for p in projection.parameters())
    assert all(p.grad is None for p in adapter.parameters())

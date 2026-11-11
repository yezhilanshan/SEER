import pytest
import torch
from continuous_mmwave.models.memory import QueryCompressor
from continuous_mmwave.parrad_memory import EventMemory, MemoryUnit
from continuous_mmwave.stages.parrad_stage4 import summary_objective
from continuous_mmwave.models.event_former import TinyLLMAdapter
from continuous_mmwave.models.semantic import SemanticProjection


def compressor():
    return QueryCompressor(16, internal_dim=16, heads=4, dropout=0)


def test_compressor_masks_and_shared_recursive_gradients():
    torch.manual_seed(7)
    model = compressor()
    source = torch.randn(1, 3, 8, 16)
    mask = torch.ones(1, 3, 8, dtype=torch.bool)
    mask[:, 2] = False
    changed = source.clone(); changed[:, 2] = 999
    times = torch.tensor([[0., 1., 2.]])
    a = model(source, mask, times, level=1)
    b = model(changed, mask, times, level=1)
    assert a.shape == (1, 4, 16)
    torch.testing.assert_close(a, b)
    higher = model(a[:, None].expand(-1, 2, -1, -1), torch.ones(1, 2, 4, dtype=torch.bool), times[:, :2], level=2)
    higher.square().mean().backward()
    assert model.queries.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='valid'):
        model(source, torch.zeros_like(mask), times, level=1)


def test_frozen_llm_and_projection_pass_gradient_only_to_compressor():
    model = compressor()
    adapter = TinyLLMAdapter(32, 16, 2, 4)
    projection = SemanticProjection(16, 8)
    for parameter in projection.parameters(): parameter.requires_grad_(False)
    source = torch.randn(1, 2, 8, 16)
    summary = model(source, torch.ones(1, 2, 8, dtype=torch.bool), torch.zeros(1, 2), level=1)
    ids = torch.tensor([[1, 2, 3, 4, 0]])
    target_mask = torch.tensor([[False, False, True, True, False]])
    metrics = summary_objective(adapter, projection, summary, source.flatten(1, 2), ids, ids.ne(0), target_mask)
    metrics['loss'].backward()
    assert model.output_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in adapter.parameters())
    assert all(p.grad is None for p in projection.parameters())
    changed = ids.clone(); changed[0, -1] = 9
    again = summary_objective(adapter, projection, summary.detach(), source.flatten(1, 2), changed, ids.ne(0), target_mask)
    torch.testing.assert_close(metrics['summary_loss'], again['summary_loss'])


def test_memory_promotes_orders_and_roundtrips():
    model = compressor().eval(); projection = SemanticProjection(16, 8).eval()
    memory = EventMemory(model, projection, l0_capacity=2, l1_capacity=2, l2_capacity=2, group_size=2)
    for i in range(9):
        tokens = torch.randn(8, 16)
        memory.add(MemoryUnit(str(i), tokens, float(i), float(i + 1), float(i+1), 0.8,
                              projection(tokens[None])[0].detach()))
    assert [len(memory.levels[i]) for i in range(3)] == [1, 2, 1]
    context, selected = memory.context(torch.randn(8), now=10)
    assert [u.level for u in selected] == [2, 1, 1, 0]
    assert context.shape == (1, 20, 16)
    restored = EventMemory(model, projection, l0_capacity=2, l1_capacity=2, l2_capacity=2, group_size=2)
    restored.restore(memory.snapshot())
    torch.testing.assert_close(context, restored.context(torch.randn(8), now=10)[0])
    assert all(not u.tokens.requires_grad for level in memory.levels for u in level)


def test_provisional_replace_caption_does_not_merge_and_overlap_rejected():
    model = compressor().eval(); projection = SemanticProjection(16, 8).eval()
    memory = EventMemory(model, projection)
    tokens = torch.randn(8, 16); key = projection(tokens[None])[0].detach()
    memory.add(MemoryUnit('a', tokens, 0, 1, 1, .9, key, status='provisional'))
    memory.refine('a', tokens + 1, key, caption='changed caption')
    assert memory.levels[0][0].status == 'captioned'
    memory.add(MemoryUnit('b', tokens, 1, 2, 2, .9, key, caption='changed caption'))
    assert len(memory.levels[0]) == 2
    with pytest.raises(ValueError, match='chronological'):
        memory.add(MemoryUnit('c', tokens, .5, 2, 2, .9, key))


def test_semantic_merge_stays_chronological_through_l0_and_l1_promotion():
    model=compressor().eval(); projection=SemanticProjection(16,8).eval()
    memory=EventMemory(model,projection,l0_capacity=2,l1_capacity=2,l2_capacity=2,
                       group_size=2,merge_cosine=.99,merge_gap_s=.1)
    tokens=torch.randn(8,16); key=torch.ones(8)
    memory.add(MemoryUnit('a',tokens,0,1,1,.9,-key))
    memory.add(MemoryUnit('b',tokens,1,2,2,.9,key))
    memory.add(MemoryUnit('c',tokens,2,3,3,.8,key))
    assert len(memory.levels[0])==2 and len(memory.levels[1])==0
    assert memory.levels[0][-1].level==0 and memory.levels[0][-1].count==2
    assert memory.levels[0][-1].confidence==.8
    memory.add(MemoryUnit('d',tokens,4,5,5,.7,key))
    assert len(memory.levels[1])==1
    assert memory.levels[1][0].start==0 and memory.levels[1][0].end==3

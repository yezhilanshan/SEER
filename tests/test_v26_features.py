import importlib
import importlib.util
import torch


def test_event_prefix_never_uses_future_end_or_tokens():
    assert importlib.util.find_spec('continuous_mmwave.v26.features') is not None, 'causal features missing'
    module = importlib.import_module('continuous_mmwave.v26.features')
    from continuous_mmwave.config import ModelConfig
    from continuous_mmwave.models.event_former import StreamingSemanticEventFormer, TinyLLMAdapter
    config = ModelConfig(llm_dim=16, llm_heads=4, cross_attn_dim=16, cross_attn_heads=4, radar_dim=8, llm_layers=2, dropout=0.)
    former = StreamingSemanticEventFormer(TinyLLMAdapter(32,16,2,4), config, torch.tensor([1,2])).eval()
    former.requires_grad_(False)
    tokens = torch.randn(5,4,8)
    mask = torch.ones(5,4,dtype=torch.bool)
    original = module.event_prefix_features(former,tokens,mask,[0]*5,[True]*5,32)
    changed = tokens.clone(); changed[3:] += 100
    changed = module.event_prefix_features(former,changed,mask,[0,0,0,1,1],[True]*5,32)
    torch.testing.assert_close(original['events'][:3],changed['events'][:3],rtol=0,atol=0)
    assert not original['events'].requires_grad


def test_batched_event_prefix_matches_sequential_and_remains_causal():
    module=importlib.import_module('continuous_mmwave.v26.features')
    assert hasattr(module,'batched_event_prefix_features'), 'batched independent prefix export missing'
    from continuous_mmwave.config import ModelConfig
    from continuous_mmwave.models.event_former import StreamingSemanticEventFormer,TinyLLMAdapter
    config=ModelConfig(llm_dim=16,llm_heads=4,cross_attn_dim=16,cross_attn_heads=4,radar_dim=8,llm_layers=2,dropout=0.)
    former=StreamingSemanticEventFormer(TinyLLMAdapter(32,16,2,4),config,torch.tensor([1,2])).eval().requires_grad_(False)
    radar=torch.randn(9,4,8); mask=torch.ones(9,4,dtype=torch.bool)
    episodes=[0,0,0,1,1,1,1,2,2]; valid=[True,True,False,True,True,True,True,True,True]
    reference=module.event_prefix_features(former,radar,mask,episodes,valid,3)
    batch=module.batched_event_prefix_features(former,radar,mask,episodes,valid,3,4)
    assert torch.equal(reference['indices'],batch['indices']) and torch.equal(reference['prefix_start'],batch['prefix_start'])
    torch.testing.assert_close(reference['events'],batch['events'],rtol=1e-5,atol=1e-6)
    changed=radar.clone(); changed[4:]+=100
    after=module.batched_event_prefix_features(former,changed,mask,episodes,valid,3,4)
    torch.testing.assert_close(batch['events'][:3],after['events'][:3],rtol=0,atol=0)


def test_cuda_graph_export_cpu_fallback_is_exact():
    module=importlib.import_module('continuous_mmwave.v26.features')
    assert hasattr(module,'graph_event_prefix_features'), 'batch1 graph export missing'
    from continuous_mmwave.config import ModelConfig
    from continuous_mmwave.models.event_former import StreamingSemanticEventFormer,TinyLLMAdapter
    config=ModelConfig(llm_dim=16,llm_heads=4,cross_attn_dim=16,cross_attn_heads=4,radar_dim=8,llm_layers=2,dropout=0.)
    former=StreamingSemanticEventFormer(TinyLLMAdapter(32,16,2,4),config,torch.tensor([1,2])).eval().requires_grad_(False)
    radar=torch.randn(5,4,8); mask=torch.ones(5,4,dtype=torch.bool)
    direct=module.event_prefix_features(former,radar,mask,[0,0,0,1,1],[True]*5,3)
    graph=module.graph_event_prefix_features(former,radar,mask,[0,0,0,1,1],[True]*5,3)
    torch.testing.assert_close(direct['events'],graph['events'],rtol=0,atol=0)

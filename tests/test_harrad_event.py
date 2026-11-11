from types import SimpleNamespace
import torch


def tiny_model():
    from continuous_mmwave.models.event_former import TinyLLMAdapter,StreamingSemanticEventFormer
    from continuous_mmwave.config import FIXED_EVENT_PROMPT
    from continuous_mmwave.harrad.event_semantic import EventSemanticModel
    torch.set_num_threads(1); torch.manual_seed(17)
    cfg=SimpleNamespace(llm_dim=16,llm_layers=2,event_tokens=2,radar_dim=16,radar_tokens_per_update=4,
                        cross_attn_dim=16,cross_attn_heads=4,dropout=0.,cross_attention_init_scale=.01,
                        event_state_initial_update=.1,fixed_event_prompt=FIXED_EVENT_PROMPT)
    adapter=TinyLLMAdapter(vocab_size=32,hidden_size=16,layers=2,heads=4)
    former=StreamingSemanticEventFormer(adapter,cfg,torch.tensor([1,2,3]))
    return EventSemanticModel(former,radar_dim=16,detach_updates=8)


def test_adapter_padding_and_prefix_causality():
    model=tiny_model().eval()
    x=torch.randn(2,25,4,64); mask=torch.ones(2,25,dtype=torch.bool); mask[0,9:]=False
    result=model(x,mask)
    changed=x.clone(); changed[:,8:]=torch.randn_like(changed[:,8:])*100
    newer=model(changed,mask)
    torch.testing.assert_close(result['prefix'][3]['logits'],newer['prefix'][3]['logits'])
    shorter=model(x[:,:6],mask[:,:6])
    torch.testing.assert_close(result['prefix'][3]['logits'],shorter['final_logits'])
    # Adding padded future frames must not change final prediction or observed age.
    padded=model(torch.cat([x,torch.zeros(2,4,4,64)],1),torch.cat([mask,torch.zeros(2,4,dtype=torch.bool)],1))
    torch.testing.assert_close(result['final_logits'],padded['final_logits'])
    assert result['observed_frames'].tolist()==[9,25]


def test_event_loss_reaches_token_adapter_and_keeps_base_frozen():
    from continuous_mmwave.harrad.event_semantic import semantic_loss
    model=tiny_model().train(); before={k:p.detach().clone() for k,p in model.former.adapter.named_parameters()}
    x=torch.randn(2,10,4,64); mask=torch.ones(2,10,dtype=torch.bool)
    output=model(x,mask); loss=semantic_loss(output,torch.tensor([0,1]),torch.ones(12))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.tokens.project.weight.grad.abs().max()>0
    assert model.former.cross_attention[0].alpha.grad is not None
    assert all(p.grad is None and not p.requires_grad for p in model.former.adapter.parameters())
    assert all(torch.equal(before[k],p) for k,p in model.former.adapter.named_parameters())


def test_adapter_outputs_distinct_tokens_and_masks_invalid_cells():
    from continuous_mmwave.harrad.event_semantic import RadarTokenAdapter
    adapter=RadarTokenAdapter(16)
    x=torch.randn(2,2,4,64); mask=torch.tensor([[1,0],[1,1]],dtype=torch.bool)
    a=adapter(x,mask); changed=x.clone(); changed[0,1]=10000
    torch.testing.assert_close(a,adapter(changed,mask))
    assert a.shape==(2,4,16) and not torch.allclose(a[:,0],a[:,1])

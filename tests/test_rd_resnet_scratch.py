import torch
from torch.nn import functional as F
import pytest


def test_scratch_backbone_is_single_channel_trainable_and_updates():
    from continuous_mmwave.radar_text.rd_scratch import RDResNetPretrainer
    torch.set_num_threads(2);torch.manual_seed(17)
    model=RDResNetPretrainer(frame_chunk=8,amp=False).train()
    assert model.encoder.backbone.conv1.in_channels==1
    assert all(p.requires_grad for p in model.parameters())
    before=model.encoder.backbone.conv1.weight.detach().clone()
    bn_before=model.encoder.backbone.bn1.running_mean.clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    x=torch.randn(2,3,1,32,32);lengths=torch.tensor([3,2])
    logits=model(x,lengths);labels=torch.tensor([[0,1,2],[3,4,-100]])
    assert logits.shape==(2,3,14)
    F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before,model.encoder.backbone.conv1.weight)
    assert not torch.equal(bn_before,model.encoder.backbone.bn1.running_mean)


def test_encoder_ignores_padding_and_emits_eight_ordered_tokens():
    from continuous_mmwave.radar_text.rd_scratch import RDResNetPretrainer
    torch.set_num_threads(2);torch.manual_seed(17)
    model=RDResNetPretrainer(frame_chunk=8,amp=False).eval()
    x=torch.randn(2,4,1,32,32);lengths=torch.tensor([4,2]);changed=x.clone();changed[1,2:]=float('nan')
    with torch.no_grad():
        h,z=model.encoder(x,lengths);h2,z2=model.encoder(changed,lengths)
    assert h.shape==(2,4,256) and z.shape==(2,8,256)
    torch.testing.assert_close(h,h2);torch.testing.assert_close(z,z2)
    assert torch.count_nonzero(h[1,2:])==0
    with pytest.raises(ValueError):model(x,torch.tensor([4,0]))


def test_accumulation_scales_final_partial_group_correctly():
    from continuous_mmwave.radar_text.rd_scratch import accumulation_divisor
    assert [accumulation_divisor(i,5,2) for i in range(5)]==[2,2,2,2,1]


def test_checkpoint_roundtrip_preserves_logits_and_subject_metadata(tmp_path):
    from continuous_mmwave.radar_text.rd_scratch import RDResNetPretrainer,save_model,load_model
    torch.set_num_threads(2);torch.manual_seed(17)
    model=RDResNetPretrainer(frame_chunk=8,amp=False).eval()
    x=torch.randn(1,2,1,32,32);lengths=torch.tensor([2]);p=tmp_path/'best.pt'
    save_model(p,model,epoch=3,subjects={'train':['subject_01'],'validation':['subject_05']})
    other,meta=load_model(p,device='cpu')
    assert meta['epoch']==3 and meta['subjects']['train']==['subject_01']
    with torch.no_grad():torch.testing.assert_close(model(x,lengths),other(x,lengths))


def test_amp_overflow_skips_update_and_backs_off_then_recovers():
    from continuous_mmwave.radar_text.rd_scratch import optimizer_update
    model=torch.nn.Linear(2,1);optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    scaler=torch.amp.GradScaler('cpu',init_scale=8.)
    original={k:v.clone() for k,v in model.state_dict().items()}
    scaler.scale(model(torch.ones(1,2)).sum()).backward()
    for p in model.parameters():p.grad.fill_(float('inf'))
    updated,_=optimizer_update(model,optimizer,scaler,1.)
    assert not updated and scaler.get_scale()==4.
    assert all(torch.equal(v,original[k]) for k,v in model.state_dict().items())
    optimizer.zero_grad();scaler.scale(model(torch.ones(1,2)).sum()).backward()
    updated,norm=optimizer_update(model,optimizer,scaler,1.)
    assert updated and torch.isfinite(norm)
    assert not torch.equal(model.weight,original['weight'])


def test_non_amp_nonfinite_gradients_fail_before_weights_change():
    from continuous_mmwave.radar_text.rd_scratch import optimizer_update
    model=torch.nn.Linear(2,1);optimizer=torch.optim.SGD(model.parameters(),lr=.1);scaler=torch.amp.GradScaler('cpu',enabled=False)
    original=model.weight.detach().clone();model(torch.ones(1,2)).sum().backward()
    model.weight.grad.fill_(float('inf'))
    with pytest.raises(ValueError,match='nonfinite'):optimizer_update(model,optimizer,scaler,1.)
    assert torch.equal(original,model.weight)

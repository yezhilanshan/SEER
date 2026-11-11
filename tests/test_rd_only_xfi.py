import pytest
import torch
from torch import nn


def test_sum_collapse_matches_repeating_single_rd_without_changing_original():
    from continuous_mmwave.radar_text.rd_transfer import collapse_input_conv
    torch.manual_seed(4);conv=nn.Conv2d(17,4,7,2,3,bias=False).double();before=conv.weight.detach().clone()
    one=collapse_input_conv(conv);x=torch.randn(2,1,19,23,dtype=torch.float64)
    torch.testing.assert_close(one(x),conv(x.repeat(1,17,1,1)),rtol=1e-10,atol=1e-10)
    assert torch.equal(conv.weight,before) and one.in_channels==1


def test_random_backbone_single_batch_shape_bn_frozen_no_softmax():
    from continuous_mmwave.radar_text.rd_transfer import make_backbone
    model=make_backbone('random-resnet',seed=17);model.train()
    assert all(not m.training for m in model.modules())
    assert not any(isinstance(m,(nn.Softmax,nn.Linear)) for m in model.modules())
    before={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():value=model(torch.randn(1,1,93,128))
    assert value.shape==(1,512) and torch.isfinite(value).all()
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    assert all(not p.requires_grad for p in model.parameters())


def test_pretrained_plain_tensor_checkpoint_strict_load_and_single_channel(tmp_path):
    from continuous_mmwave.radar_text.rd_transfer import RDResNet18,make_backbone
    torch.manual_seed(3);original=RDResNet18(channels=17);state=original.state_dict();state['fc.weight']=torch.zeros(55,512);state['fc.bias']=torch.zeros(55)
    path=tmp_path/'state.pt';torch.save(dict(state_dict=state),path);model=make_backbone('xfi-resnet',path)
    torch.testing.assert_close(model.network.conv1.weight,state['conv1.weight'].sum(1,keepdim=True))
    assert torch.equal(model.network.layer4[1].bn2.running_var,state['layer4.1.bn2.running_var'])
    state['unexpected.weight']=torch.ones(1);torch.save(dict(state_dict=state),path)
    with pytest.raises((ValueError,RuntimeError)):make_backbone('xfi-resnet',path)


def test_plain_checkpoint_on_older_torch_without_safe_globals(tmp_path,monkeypatch):
    from continuous_mmwave.radar_text.rd_transfer import RDResNet18,load_xfi_state
    state=RDResNet18().state_dict();state['fc.weight']=torch.zeros(55,512);state['fc.bias']=torch.zeros(55)
    path=tmp_path/'export.pt';torch.save(dict(state_dict=state,original_sha256='test'),path)
    monkeypatch.delattr(torch.serialization,'safe_globals',raising=False)
    assert torch.equal(load_xfi_state(path)['conv1.weight'],state['conv1.weight'])


def test_rd_normalization_ignores_padding_and_rejects_nonfinite_real_frames():
    from continuous_mmwave.radar_text.rd_transfer import normalize_rd
    rd=torch.randn(2,5,1,11,13);lengths=torch.tensor([3,5]);rd[0,3:]=float('nan')
    norm=normalize_rd(rd,lengths);one=normalize_rd(rd[:1,:3],torch.tensor([3]))
    torch.testing.assert_close(norm[0,:3],one[0]);assert not norm[0,3:].any()
    rd[0,0,0,0,0]=float('nan')
    with pytest.raises(ValueError):normalize_rd(rd,lengths)
    with pytest.raises(ValueError):normalize_rd(torch.zeros(2,5,2,11,13),lengths)
    with pytest.raises(ValueError):normalize_rd(torch.zeros(2,5,1,11,13),torch.tensor([0,5]))


def test_temporal_readout_padding_invariant_and_all_heads_receive_gradients():
    from continuous_mmwave.radar_text.rd_transfer import TemporalReadout
    torch.manual_seed(7);m=TemporalReadout(12,dropout=0).eval();x=torch.randn(2,7,12);lengths=torch.tensor([4,7])
    hidden,tokens,frame,window=m(x,lengths);xx=x.clone();xx[0,4:]=999
    other=m(xx,lengths)
    assert hidden.shape==(2,7,256) and tokens.shape==(2,8,256) and frame.shape==(2,7,14) and window.shape==(2,62)
    torch.testing.assert_close(tokens,other[1]);torch.testing.assert_close(window,other[3])
    torch.testing.assert_close(m(x[:1,:4],torch.tensor([4]))[1],tokens[:1])
    (frame.square().mean()+window.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    with pytest.raises(ValueError):m(x,torch.tensor([8,7]))


def test_real_full_object_checkpoint_safely_imports_without_original_python_module():
    from pathlib import Path
    from continuous_mmwave.radar_text.rd_transfer import load_xfi_state
    path=Path('backbone_models/mmwave_ResNet18.pt')
    if not path.exists():pytest.skip('original full-object checkpoint is local only')
    state=load_xfi_state(path)
    assert state['conv1.weight'].shape==(64,17,7,7) and state['fc.weight'].shape==(55,512)

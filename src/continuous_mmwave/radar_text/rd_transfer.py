"""RD-only frozen frame backbones and trainable offline temporal readouts."""
from pathlib import Path
import pickle

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

from .model import FrameCNN


class _StoredResNet(nn.Module):
    """Controlled pickle storage container; never execute checkpoint forwards."""


class _StoredBasicBlock(nn.Module):
    pass


def load_xfi_state(path):
    safe=[(_StoredResNet,'backbone_models.mmWave.ResNet.ResNet'),
          (_StoredBasicBlock,'backbone_models.mmWave.ResNet.BasicBlock'),
          nn.Conv2d,nn.BatchNorm2d,nn.Linear,nn.Softmax,nn.Sequential,
          nn.AdaptiveAvgPool2d,nn.ReLU,nn.MaxPool2d,set]
    try:
        value=torch.load(Path(path),map_location='cpu',weights_only=True)
    except pickle.UnpicklingError:
        if not hasattr(torch.serialization,'safe_globals'):
            raise ValueError('export a tensor-only checkpoint using a newer torch; unsafe loading is disabled')
        with torch.serialization.safe_globals(safe):value=torch.load(Path(path),map_location='cpu',weights_only=True)
    state=value.state_dict() if isinstance(value,_StoredResNet) else value.get('state_dict',value)
    if not isinstance(state,dict) or not all(isinstance(v,torch.Tensor) for v in state.values()):
        raise ValueError('expected tensor-only X-Fi state')
    if state['conv1.weight'].shape!=(64,17,7,7) or state['fc.weight'].shape!=(55,512):
        raise ValueError('not the expected XRF55 ResNet18 checkpoint')
    return state


def collapse_input_conv(conv):
    if conv.groups!=1 or conv.in_channels!=17:raise ValueError('expected17channel ungrouped convolution')
    new=nn.Conv2d(1,conv.out_channels,conv.kernel_size,conv.stride,conv.padding,conv.dilation,
                  bias=conv.bias is not None,padding_mode=conv.padding_mode).to(conv.weight)
    with torch.no_grad():
        new.weight.copy_(conv.weight.sum(1,keepdim=True))
        if conv.bias is not None:new.bias.copy_(conv.bias)
    return new


class BasicBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__();self.conv1=nn.Conv2d(cin,cout,3,stride,1,bias=False);self.bn1=nn.BatchNorm2d(cout)
        self.relu=nn.ReLU();self.conv2=nn.Conv2d(cout,cout,3,1,1,bias=False);self.bn2=nn.BatchNorm2d(cout)
        self.downsample=nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout)) if cin!=cout or stride!=1 else None

    def forward(self,x):
        identity=x if self.downsample is None else self.downsample(x)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))+identity)


class RDResNet18(nn.Module):
    def __init__(self,channels=17):
        super().__init__();self.conv1=nn.Conv2d(channels,64,7,2,3,bias=False);self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU();self.maxpool=nn.MaxPool2d(3,2,1)
        self.layer1=nn.Sequential(BasicBlock(64,64),BasicBlock(64,64))
        self.layer2=nn.Sequential(BasicBlock(64,128,2),BasicBlock(128,128))
        self.layer3=nn.Sequential(BasicBlock(128,256,2),BasicBlock(256,256))
        self.layer4=nn.Sequential(BasicBlock(256,512,2),BasicBlock(512,512))
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x=self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return F.adaptive_avg_pool2d(x,1).flatten(1)


class FrozenRDBackbone(nn.Module):
    def __init__(self,network,width):
        super().__init__();self.network=network;self.width=width;self.requires_grad_(False);self.train(False)

    def train(self,mode=True):
        return super().train(False)

    def forward(self,x):
        if x.ndim!=4 or x.shape[1]!=1:raise ValueError('RD frames must be [N,1,R,D]')
        return self.network(x)


def make_backbone(kind,checkpoint=None,seed=17):
    torch.manual_seed(seed)
    if kind=='stage1-cnn':
        if checkpoint is None:raise ValueError('stage1 CNN needs its state')
        model=FrameCNN(1,128);model.load_state_dict(checkpoint,strict=True)
        return FrozenRDBackbone(model,128)
    if kind not in ('random-resnet','xfi-resnet'):raise ValueError('unknown RD backbone')
    model=RDResNet18()
    if kind=='xfi-resnet':
        if checkpoint is None:raise ValueError('X-Fi checkpoint required')
        state=load_xfi_state(checkpoint);backbone={k:v for k,v in state.items() if k not in ('fc.weight','fc.bias')}
        model.load_state_dict(backbone,strict=True)
    model.conv1=collapse_input_conv(model.conv1)
    return FrozenRDBackbone(model,512)


def _lengths(lengths,b,t):
    if lengths.shape!=(b,) or lengths.dtype not in (torch.int32,torch.int64) or (lengths<=0).any() or (lengths>t).any():
        raise ValueError('invalid sequence lengths')


def normalize_rd(rd,lengths,normalization='signed_log_zscore'):
    if normalization not in ('signed_log_zscore','raw_zscore'):raise ValueError('invalid RD normalization')
    if rd.ndim!=5 or rd.shape[2]!=1 or min(rd.shape[3:])<1:raise ValueError('expected RD[B,T,1,R,D]')
    _lengths(lengths,*rd.shape[:2]);out=torch.zeros_like(rd,dtype=torch.float32)
    for i,n in enumerate(lengths.tolist()):
        x=rd[i,:n].float()
        if not torch.isfinite(x).all():raise ValueError('nonfinite valid RD')
        if normalization=='signed_log_zscore':x=x.sign()*x.abs().log1p()
        out[i,:n]=(x-x.mean())/x.std(unbiased=False).clamp_min(1e-5)
    return out


class TemporalReadout(nn.Module):
    def __init__(self,width,dropout=.1):
        super().__init__();self.input=nn.Linear(width,256)
        self.gru=nn.GRU(256,128,2,batch_first=True,bidirectional=True,dropout=dropout)
        self.frame=nn.Linear(256,14);self.window=nn.Sequential(nn.Linear(8*256,32),nn.GELU(),nn.Linear(32,62))

    def forward(self,features,lengths):
        if features.ndim!=3 or features.shape[-1]!=self.input.in_features:raise ValueError('invalid temporal feature shape')
        b,t=features.shape[:2];_lengths(lengths,b,t)
        x=self.input(features);packed=pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False)
        h,_=self.gru(packed);h,_=pad_packed_sequence(h,batch_first=True,total_length=t)
        z=torch.stack([F.adaptive_avg_pool1d(h[i,:n].T,8).T for i,n in enumerate(lengths.tolist())])
        return h,z,self.frame(h),self.window(z.flatten(1))

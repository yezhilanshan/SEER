"""Trainable RD-only ResNet18/BiGRU pretraining, distinct from frozen transfer."""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from .rd_transfer import RDResNet18,normalize_rd,_lengths


class RDResNetEncoder(nn.Module):
    def __init__(self,frame_chunk=64,amp=True,dropout=.1,normalization='signed_log_zscore'):
        super().__init__()
        if frame_chunk<1:raise ValueError('frame_chunk must be positive')
        if normalization not in ('signed_log_zscore','raw_zscore'):raise ValueError('invalid RD normalization')
        self.normalization=normalization
        self.frame_chunk,self.amp,self.dropout=frame_chunk,amp,dropout
        self.backbone=RDResNet18(channels=1)
        self.projection=nn.Linear(512,256)
        self.gru=nn.GRU(256,128,2,batch_first=True,bidirectional=True,dropout=dropout)

    def forward(self,rd,lengths):
        rd=normalize_rd(rd,lengths,self.normalization);b,t=rd.shape[:2];_lengths(lengths,b,t)
        mask=torch.arange(t,device=rd.device)[None]<lengths.to(rd.device)[:,None]
        with torch.autocast(device_type=rd.device.type,dtype=torch.float16,enabled=self.amp and rd.is_cuda):
            features=torch.cat([self.backbone(x) for x in rd[mask].split(self.frame_chunk)])
        # GRU and classification remain FP32 on the server's Turing GPU.
        projected=self.projection(features.float());sequence=projected.new_zeros(b,t,256);sequence[mask]=projected
        packed=pack_padded_sequence(sequence,lengths.cpu(),batch_first=True,enforce_sorted=False)
        hidden,_=self.gru(packed);hidden,_=pad_packed_sequence(hidden,batch_first=True,total_length=t)
        tokens=torch.stack([F.adaptive_avg_pool1d(hidden[i,:n].T,8).T for i,n in enumerate(lengths.tolist())])
        return hidden,tokens


class RDResNetPretrainer(nn.Module):
    def __init__(self,frame_chunk=64,amp=True,dropout=.1,normalization='signed_log_zscore'):
        super().__init__();self.encoder=RDResNetEncoder(frame_chunk,amp,dropout,normalization);self.classifier=nn.Linear(256,14)

    def forward(self,rd,lengths):return self.classifier(self.encoder(rd,lengths)[0])


def accumulation_divisor(index,total,accumulation):
    if not 0<=index<total or accumulation<1:raise ValueError('invalid accumulation group')
    return min(accumulation,total-(index//accumulation)*accumulation)


def optimizer_update(model,optimizer,scaler,max_norm):
    """Let AMP reject overflows; count only actual updates, without touching bad grads."""
    scaler.unscale_(optimizer)
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads:raise ValueError('no gradients to update')
    # FP64 reduction avoids norm overflow when individual FP32 elements are finite.
    norm=torch.stack([g.detach().norm(dtype=torch.float64) for g in grads]).norm()
    if torch.isfinite(norm):
        coefficient=(max_norm/(norm+1e-6)).clamp(max=1.)
        for grad in grads:grad.mul_(coefficient.to(grad))
    elif not scaler.is_enabled():raise ValueError('nonfinite gradients without AMP')
    previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
    return scaler.get_scale()>=previous_scale,norm


def save_model(path,model,**metadata):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    payload=dict(format='rd_resnet18_scratch_v1',model_config=dict(frame_chunk=model.encoder.frame_chunk,amp=model.encoder.amp,dropout=model.encoder.dropout,normalization=model.encoder.normalization),
                 encoder=model.encoder.state_dict(),classifier=model.classifier.state_dict(),**metadata)
    torch.save(payload,temporary);temporary.replace(path)


def load_model(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['format']!='rd_resnet18_scratch_v1':raise ValueError('wrong architecture checkpoint')
    model=RDResNetPretrainer(**payload['model_config']);model.encoder.load_state_dict(payload['encoder']);model.classifier.load_state_dict(payload['classifier'])
    return model.to(device).eval(),payload

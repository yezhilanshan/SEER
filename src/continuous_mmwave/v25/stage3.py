"""Trainable v2.5 phases and fixed-budget optimizer accounting."""
from __future__ import annotations

import math
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F

from .models import V25Heads
from .training import semantic_loss,evidence_loss,age_matched_weights


def validate_feature_cache(payload,frontend_digest,*,split='train'):
    if payload.get('schema_version') != 25 or payload.get('architecture') != 'v25-age-free-tanh':
        raise ValueError('v2.5 age-free cache required; rebuild legacy event-state caches')
    if payload.get('frontend_digest') != frontend_digest:
        raise ValueError('v2.5 feature cache source mismatch')
    if payload.get('split') != split:
        raise ValueError('feature cache split mismatch; do not train on development/test')
    return payload


class V25Stage3(nn.Module):
    def __init__(self,radar_dim=256,event_dim=896,semantic_dim=128,evidence_dim=4,update_period_s=.18,tbptt_steps=32):
        super().__init__()
        if update_period_s <= 0 or tbptt_steps <= 0:
            raise ValueError('positive training timing constants required')
        self.heads = V25Heads(radar_dim,event_dim,semantic_dim,evidence_dim)
        self.update_period_s,self.tbptt_steps = update_period_s,tbptt_steps
        self.optimizer_steps = dict(physical=0,semantic=0,evidence=0)
        self.set_phase('physical')

    def set_phase(self,phase):
        prefixes = {'physical':('local.','radar.'),'semantic':('semantic.',),'oracle':('semantic.',),'evidence':('evidence.',)}
        if phase not in prefixes:
            raise ValueError('v2.5 phases are physical/oracle/semantic/evidence; no Verify training')
        self.phase = phase
        for name,p in self.heads.named_parameters():
            p.requires_grad_(name.startswith(prefixes[phase]))

    def stage2_evidence_loss(self,tokens,targets,mask):
        # Intentionally keeps the event-token graph; joint Stage2 supervision.
        return evidence_loss(self.heads.evidence(tokens),targets,mask)

    @staticmethod
    def _bce(logits,target,weight=None,class_weights=None):
        if target.shape != logits.shape or not torch.isfinite(target).all() or ((target < 0)|(target > 1)).any():
            raise ValueError('invalid physical targets')
        losses = F.binary_cross_entropy_with_logits(logits,target.to(logits),reduction='none')
        if class_weights is not None:
            a,b=class_weights
            if not all(math.isfinite(x) and x>0 for x in (a,b)):
                raise ValueError('positive finite physical component weights required')
            target=target.to(logits)
            losses=-(a*target*F.logsigmoid(logits)+b*(1-target)*F.logsigmoid(-logits))
            mass=a*target+b*(1-target)
            if weight is not None:
                if weight.shape!=losses.shape or not torch.isfinite(weight).all() or (weight<0).any() or weight.sum()<=0:
                    raise ValueError('invalid physical sample weights')
                losses=losses*weight;mass=mass*weight
            return losses.sum()/mass.sum()
        if weight is None:
            return losses.mean()
        if weight.shape != losses.shape or not torch.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
            raise ValueError('invalid physical sample weights')
        return (losses*weight).sum()/weight.sum()

    def physical_step(self,batch):
        if 'tokens' not in batch:
            previous,current,mask = (batch[k] for k in ('previous','current','mask'))
            local = self.heads.local(previous,current,mask)
            _,state = self.heads.radar(previous,mask,None,self.update_period_s)
            radar,_ = self.heads.radar(current,mask,state,self.update_period_s)
            valid = torch.ones_like(local,dtype=torch.bool)
        else:
            tokens,mask = batch['tokens'],batch['mask']
            if tokens.ndim != 4 or tokens.shape[1] < 2 or mask.shape != tokens.shape[:3]:
                raise ValueError('physical sequence requires [B,U,R,D], U>=2')
            _,state = self.heads.radar(tokens[:,0],mask[:,0],None,self.update_period_s)
            locals_,radars = [],[]
            for i in range(1,tokens.shape[1]):
                common = mask[:,i-1].bool() & mask[:,i].bool()
                eligible = common.any(-1)
                local_i = tokens.new_zeros((tokens.shape[0],))
                if eligible.any():
                    local_i[eligible] = self.heads.local(tokens[eligible,i-1],tokens[eligible,i],common[eligible])
                if i % self.tbptt_steps == 0:
                    state = replace(state,fast=state.fast.detach(),mean=state.mean.detach(),m2=state.m2.detach())
                radar_i,state = self.heads.radar(tokens[:,i],mask[:,i],state,self.update_period_s)
                locals_.append(local_i); radars.append(radar_i)
            local,radar = torch.stack(locals_,1),torch.stack(radars,1)
            valid = batch.get('valid',torch.ones_like(local,dtype=torch.bool)).bool()
            common_valid = (mask[:,:-1].bool() & mask[:,1:].bool()).any(-1)
            valid = valid & common_valid
        if not valid.any():
            raise ValueError('physical loss needs valid transitions')
        weight = batch.get('weight')
        weight = None if weight is None else weight[valid]
        local_loss = self._bce(local[valid],batch['local_target'][valid],weight)
        radar_loss = self._bce(radar[valid],batch['radar_target'][valid],weight)
        return dict(loss=local_loss+radar_loss,local_loss=local_loss,radar_loss=radar_loss)

    def physical_stream_step(self,batch):
        tokens,mask=batch['tokens'],batch['mask']
        if tokens.ndim!=4 or tokens.shape[1]<1 or mask.shape!=tokens.shape[:3]:
            raise ValueError('physical stream chunk needs [B,U,R,D]')
        state=batch.get('initial_state')
        previous=batch.get('previous_tokens'); previous_mask=batch.get('previous_mask')
        locals_,radars,valids=[],[],[]
        for i in range(tokens.shape[1]):
            if state is not None and i and i%self.tbptt_steps==0:
                state=replace(state,fast=state.fast.detach(),mean=state.mean.detach(),m2=state.m2.detach())
            current=tokens[:,i]; current_mask=mask[:,i]
            eligible=torch.zeros(len(tokens),dtype=torch.bool,device=tokens.device)
            local=current.new_zeros(len(tokens))
            if previous is not None:
                common=previous_mask.bool()&current_mask.bool()
                eligible=common.any(-1)
                if eligible.any():
                    local[eligible]=self.heads.local(previous[eligible],current[eligible],common[eligible])
            radar,state=self.heads.radar(current,current_mask,state,self.update_period_s)
            locals_.append(local); radars.append(radar); valids.append(eligible)
            previous,previous_mask=current,current_mask
        valid=torch.stack(valids,1)&batch['valid'].bool()
        if not valid.any():
            raise ValueError('no valid continuous transitions')
        weight=batch.get('weight')
        weight=None if weight is None else weight[valid]
        local_loss=self._bce(torch.stack(locals_,1)[valid],batch['target'][valid],weight,batch.get('class_weights'))
        radar_loss=self._bce(torch.stack(radars,1)[valid],batch['target'][valid],weight,batch.get('class_weights'))
        next_state=replace(state,fast=state.fast.detach(),mean=state.mean.detach(),m2=state.m2.detach())
        return dict(loss=local_loss+radar_loss,local_loss=local_loss,radar_loss=radar_loss,
                    next_state=next_state,valid_transitions=int(valid.sum()))

    def semantic_step(self,batch):
        closed = self.heads.semantic(batch['active'],batch['candidate'])
        closed_weights = age_matched_weights(batch['age_bucket'],batch['target'])
        oracle,targets,weights = None,None,None
        if 'oracle_active' in batch:
            oracle = self.heads.semantic(batch['oracle_active'],batch['oracle_candidate'])
            targets = batch['oracle_target']
            weights = age_matched_weights(batch['oracle_age_bucket'],targets)
        loss = semantic_loss(closed,batch['target'],oracle,targets,oracle_weight=batch.get('oracle_weight',.3),
                             closed_weights=closed_weights,oracle_weights=weights)
        return dict(loss=loss,closed_logits=closed)

    def oracle_step(self,batch):
        logits=self.heads.semantic(batch['active'],batch['candidate'])
        weights=age_matched_weights(batch['age_bucket'],batch['target'])
        return dict(loss=semantic_loss(logits,batch['target'],closed_weights=weights),oracle_logits=logits)

    def evidence_step(self,batch):
        predicted = self.heads.evidence(batch['tokens'])
        return dict(loss=evidence_loss(predicted,batch['target'],batch['mask']))


def fixed_budget_train(module,batches,*,phase,steps,lr=1e-3):
    if not isinstance(steps,int) or steps <= 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('positive fixed training budget/lr required')
    rows = list(batches)
    if not rows:
        raise ValueError('training batch list is empty')
    module.set_phase(phase)
    module.train()
    parameters = [p for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=lr)
    device = next(module.parameters()).device
    records = []
    for index in range(steps):
        row = {k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in rows[index%len(rows)].items()}
        if row.get('split','train') != 'train':
            raise ValueError('fixed-budget training accepts train batches only')
        optimizer.zero_grad(set_to_none=True)
        result = getattr(module,f'{phase}_step')(row)
        loss = result['loss']
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss; optimizer update not performed')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm) or not any(p.grad is not None for p in parameters):
            raise RuntimeError('invalid gradient; optimizer update not performed')
        optimizer.step()
        module.optimizer_steps['semantic' if phase=='oracle' else phase] += 1
        records.append(dict(step=index+1,loss=float(loss.detach()),gradient_norm=float(norm)))
    # Assert Adam's independent step counters, not just the Python loop length.
    counts = [int(state['step']) for state in optimizer.state.values() if 'step' in state]
    if not counts or any(count != steps for count in counts):
        raise RuntimeError('AdamW actual step count mismatch')
    return dict(phase=phase,actual_optimizer_steps=steps,records=records,optimizer_state=optimizer.state_dict())

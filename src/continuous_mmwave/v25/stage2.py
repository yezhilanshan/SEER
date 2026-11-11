"""Isolated gate migration with physical supervision and original-P retention."""
from __future__ import annotations

import math
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

from .training import evidence_loss


class Stage2Migration(nn.Module):
    """Reuse the original frontend; train NEW gate and evidence head only.

    Build/load a separate frontend instance before using this wrapper. Never
    apply it to a live legacy runtime. The teacher P, decoder, radar encoder
    and original EventFormer parameters remain frozen. Retention is a loss,
    not a promise that newly produced event tokens keep retrieval accuracy.
    """
    def __init__(self,frontend,evidence_projection,*,text_weight=1.,align_weight=1.,
                 consistency_weight=1.,evidence_weight=1.,retain_weight=1.,tbptt_steps=32):
        super().__init__()
        if getattr(frontend.event_former.state_gate,'architecture',None) != 'v25-age-free-tanh':
            raise ValueError('explicitly migrated v2.5 gate required')
        weights = (text_weight,align_weight,consistency_weight,evidence_weight,retain_weight)
        if not all(math.isfinite(v) and v >= 0 for v in weights):
            raise ValueError('nonnegative finite Stage2 weights required')
        self.frontend,self.evidence_projection = frontend,evidence_projection
        self.weights = weights
        if not isinstance(tbptt_steps,int) or tbptt_steps <= 0:
            raise ValueError('positive TBPTT interval required')
        self.tbptt_steps = tbptt_steps
        frontend.requires_grad_(False)
        frontend.event_former.state_gate.requires_grad_(True)
        evidence_projection.requires_grad_(True)
        self.optimizer_steps = 0

    def objective(self,output,evidence_targets,evidence_mask,teacher_stream_key,*,teacher_commit_key,
                  commit_targets,commit_mask):
        for teacher,key in ((teacher_stream_key,output.stream_key),(teacher_commit_key,output.commit_key)):
            if teacher.shape != key.shape or not torch.isfinite(teacher).all():
                raise ValueError('independent original teacher retrieval keys must match [B,D]')
        physical = (evidence_loss(self.evidence_projection(output.stream_event_tokens),evidence_targets,evidence_mask)
                    +evidence_loss(self.evidence_projection(output.commit_event_tokens),commit_targets,commit_mask))/2
        retain = ((1-F.cosine_similarity(output.stream_key,teacher_stream_key.detach().to(output.stream_key),-1)).mean()
                  +(1-F.cosine_similarity(output.commit_key,teacher_commit_key.detach().to(output.commit_key),-1)).mean())/2
        text,align,consistency,evidence,retention = self.weights
        total = (text*output.losses.text+align*output.losses.align+consistency*output.losses.consistency
                 +evidence*physical+retention*retain)
        return dict(loss=total,text_loss=output.losses.text,alignment_loss=output.losses.align,
                    consistency_loss=output.losses.consistency,evidence_loss=physical,retention_loss=retain)

    def token_step(self,batch):
        """Train from frozen-encoder radar cache; require independent teacher keys.

        Inputs: radar[B,U,4,D], radar_mask[B,U,4], caption_ids/mask, labels,
        independent stream/commit targets and teachers. initial_state is current
        gate state from the preceding consecutive chunk, never legacy state.
        """
        from ..stages.stage2 import supervised_symmetric_alignment_loss,causal_caption_cross_entropy
        self.frontend.eval()
        self.frontend.event_former.state_gate.train()
        former,p = self.frontend.event_former,self.frontend.semantic_projection
        radar,mask = batch['radar'],batch['radar_mask']
        if radar.ndim != 4 or radar.shape[1] == 0 or mask.shape != radar.shape[:3] or not mask.bool().any(-1).all():
            raise ValueError('Stage2 migration needs valid complete radar updates [B,U,R,D]')
        state = batch.get('initial_state')
        if state is None:
            state = former.initial_state(len(radar))
        else:
            if state.shape != (len(radar),former.config.event_tokens,former.config.llm_dim):
                raise ValueError('continuation state shape mismatch')
            state = state.detach()
        cache = former.build_prompt_cache(len(radar))
        for index in range(radar.shape[1]):
            if index and index % self.tbptt_steps == 0:
                state = state.detach()
            state = former.stream_update(state,radar[:,index],1.,mask[:,index],cache).event_tokens
        commit_radar = batch.get('commit_radar',radar.flatten(1,2))
        commit_mask = batch.get('commit_radar_mask',mask.flatten(1,2))
        commit = former.commit_refine(commit_radar,commit_mask,cache).event_tokens
        stream_key,commit_key = p(state),p(commit)
        ids,caption_mask = batch['caption_ids'],batch['caption_mask']
        text_key = p(former.adapter.encode_text(ids,caption_mask),caption_mask)
        alignment = (supervised_symmetric_alignment_loss(stream_key,text_key,batch['labels'],.07)
                     +supervised_symmetric_alignment_loss(commit_key,text_key,batch['labels'],.07))/2
        logits = former.adapter.caption_logits(commit,ids[:,:-1],caption_mask[:,:-1])
        text = causal_caption_cross_entropy(logits,former.config.event_tokens,ids,caption_mask)
        consistency = (1-F.cosine_similarity(stream_key,commit_key,-1)).mean()
        output = SimpleNamespace(stream_event_tokens=state,commit_event_tokens=commit,stream_key=stream_key,
            commit_key=commit_key,losses=SimpleNamespace(text=text,align=alignment,consistency=consistency))
        result = self.objective(output,batch['stream_target'],batch['stream_target_mask'],batch['teacher_stream_key'],
            teacher_commit_key=batch['teacher_commit_key'],commit_targets=batch['commit_target'],
            commit_mask=batch['commit_target_mask'])
        result['next_state'] = state.detach()
        return result

    def training_step(self,batch,caption_ids,caption_mask,evidence_targets,evidence_mask,teacher_stream_key,*,
                      teacher_commit_key,commit_targets,commit_mask):
        self.frontend.eval()  # Frozen decoder dropout is disabled; gradients still flow.
        self.frontend.event_former.state_gate.train()
        output = self.frontend.training_step(batch,caption_ids,caption_mask)
        return self.objective(output,evidence_targets,evidence_mask,teacher_stream_key,
            teacher_commit_key=teacher_commit_key,commit_targets=commit_targets,commit_mask=commit_mask)

"""Stage 2: streaming/commit radar-language alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

from continuous_mmwave.data import RadarBatch
from continuous_mmwave.models.event_former import StreamingSemanticEventFormer
from continuous_mmwave.models.radar_encoder import StreamingRadarEncoder
from continuous_mmwave.models.semantic import SemanticProjection


def symmetric_alignment_loss(events: Tensor, text: Tensor, temperature: float) -> Tensor:
    """事件/文本的对称 InfoNCE：同一事件的两个方向的交叉熵取平均。

    对角线为正样本对，因此同一 batch 内的其他样本充当负例，要求 batch_size>=2。
    """
    if events.shape != text.shape or events.ndim != 2:
        raise ValueError("event/text keys must share [B,D]")
    if events.shape[0] < 2:
        raise ValueError("semantic alignment requires at least two paired samples")
    events, text = functional.normalize(events, dim=-1), functional.normalize(text, dim=-1)
    logits = events @ text.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    # 两个方向：event->text 与 text->event，避免非对称的表示坍塌。
    return (functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.T, labels)) / 2


def supervised_symmetric_alignment_loss(
    events: Tensor,
    text: Tensor,
    labels: Tensor,
    temperature: float,
) -> Tensor:
    """Symmetric multi-positive InfoNCE where equal action labels are positives."""
    if events.shape != text.shape or events.ndim != 2:
        raise ValueError("event/text keys must share [B,D]")
    if labels.shape != (events.shape[0],):
        raise ValueError("labels must be [B]")
    if events.shape[0] < 2:
        raise ValueError("semantic alignment requires at least two paired samples")
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    events, text = functional.normalize(events, dim=-1), functional.normalize(text, dim=-1)
    logits = events @ text.T / temperature
    positives = labels[:, None].eq(labels[None, :])

    def direction_loss(scores: Tensor, mask: Tensor) -> Tensor:
        positive_scores = scores.masked_fill(~mask, -torch.inf)
        return (torch.logsumexp(scores, dim=1) - torch.logsumexp(positive_scores, dim=1)).mean()

    return (direction_loss(logits, positives) + direction_loss(logits.T, positives.T)) / 2


def cosine_consistency_loss(stream: Tensor, commit: Tensor) -> Tensor:
    """流式事件状态与 commit 重建结果的一致性损失（1 - cosine）。"""
    return (1 - functional.cosine_similarity(stream, commit, dim=-1)).mean()


def causal_caption_cross_entropy(logits: Tensor, event_count: int, ids: Tensor, mask: Tensor) -> Tensor:
    """caption 的因果语言建模损失。

    ``logits`` 为 [事件 token | caption token] 的拼接输出，预测第 i 个 caption 词
    需使用位置 i-1（即事件末位或上一个词），因此从 ``event_count - 1`` 处切片。
    被 padding 的位置置为 -100，由 cross_entropy 自动忽略。
    """
    prediction = logits[:, event_count - 1 : event_count - 1 + ids.shape[1]]
    if prediction.shape[1] != ids.shape[1]:
        raise ValueError("caption logits are incomplete")
    labels = ids.masked_fill(~mask.bool(), -100)
    return functional.cross_entropy(prediction.flatten(0, 1), labels.flatten())


@dataclass(frozen=True)
class Stage2Loss:
    """Stage 2 仅有的三项损失，权重由配置提供。"""

    text: Tensor  # 事件->caption 的因果 LM 损失
    align: Tensor  # 事件/文本对称 InfoNCE
    consistency: Tensor  # stream 与 commit 的 cosine 一致性

    def total(self, align_weight: float, consistency_weight: float) -> Tensor:
        return self.text + align_weight * self.align + consistency_weight * self.consistency


@dataclass(frozen=True)
class Stage2Output:
    """训练步的完整中间结果，便于日志记录与后续诊断。"""

    losses: Stage2Loss
    stream_event_tokens: Tensor  # [B,E,llm_dim]，流式结束状态
    commit_event_tokens: Tensor  # [B,E,llm_dim]，commit 重建结果
    stream_key: Tensor  # [B,semantic_dim]，归一化后的流式语义 key
    commit_key: Tensor
    text_key: Tensor
    stream_state_norms: Tensor  # [B,U]，逐步状态范数（用于监控漂移/坍塌）
    stream_class_logits: Tensor | None = None
    commit_class_logits: Tensor | None = None


class Stage2Module(nn.Module):
    """Stage 2：共享 event former 的流式/commit 雷达-语言对齐。

    Radar Encoder 从 Stage 1 继续以独立小学习率微调，其余模块正常学习。
    """

    def __init__(
        self,
        radar_encoder: StreamingRadarEncoder,
        event_former: StreamingSemanticEventFormer,
        semantic_projection: SemanticProjection,
        align_weight: float = 1,
        consistency_weight: float = 1,
        contrastive_temperature: float = 0.07,
        checkpoint_streaming: bool = True,
    ) -> None:
        super().__init__()
        self.radar_encoder, self.event_former = radar_encoder, event_former
        self.semantic_projection = semantic_projection
        # 三项损失的权重由训练脚本的配置提供，此处只保存不参与前向计算。
        self.align_weight, self.consistency_weight = align_weight, consistency_weight
        self.contrastive_temperature, self.checkpoint_streaming = contrastive_temperature, checkpoint_streaming

    def _stream_view(self, tokens: Tensor, token_mask: Tensor, prompt_cache: Any) -> tuple[Tensor, Tensor]:
        """按时间顺序把 radar token 逐个 update 喂给事件状态，模拟真实流式推理。

        返回最终事件状态与每一步的状态范数。纯填充的 update（变长序列尾部）
        被跳过，状态保持上一步的值，以保证与短序列结果一致。
        """
        size = self.event_former.config.radar_tokens_per_update
        # tokens 的形状为(B, updates * size, 256)
        updates = tokens.shape[1] // size
        state = self.event_former.initial_state(tokens.shape[0])
        # 一个 update 的 4 个 token 共享有效性：任一有效即参与更新。
        update_mask = token_mask.reshape(tokens.shape[0], updates, size).any(-1)
        total = update_mask.sum(1).clamp_min(1)
        # completed 记录当前样本已经处理了多少个有效 update，除以总数后得到 0→1 的进度值
        completed = torch.zeros_like(total)
        norms = []
        for index in range(updates):
            valid = update_mask[:, index]
            completed += valid
            radar = tokens[:, index * size : (index + 1) * size]
            mask = token_mask[:, index * size : (index + 1) * size]
            # maturity = 已完成 update 数 / 总 update 数，随事件推进从 0 增长到 1。
            maturity = completed.to(tokens.dtype) / total.to(tokens.dtype)
            if self.training and self.checkpoint_streaming:
                # 流式循环会产生 U 份计算图，用 activation checkpoint 换取显存
                # 训练状态或用户显式开启checkpoint_streaming，前向传播时不存储激活值，反向传播时重新计算

                def update(
                    current: Tensor, current_radar: Tensor, current_maturity: Tensor, current_mask: Tensor
                ) -> Tensor:
                    return self.event_former.stream_update(
                        current, current_radar, current_maturity, current_mask, prompt_cache
                    ).event_tokens

                proposal = checkpoint(update, state, radar, maturity, mask, use_reentrant=False)
            else:
                # 其他情况（评估/测试）不需要反向传播，本身就不需要存储中间激活值
                proposal = self.event_former.stream_update(state, radar, maturity, mask, prompt_cache).event_tokens
            # 无效 update 不改变状态（等价于跳过），保证变长批内的一致性。
            state = torch.where(valid[:, None, None], proposal, state)
            # 使用 L2范数 计算每个样本在推进过程中“能量状态”随时间的变化
            norms.append(torch.linalg.vector_norm(state, dim=(-2, -1)))
        return state, torch.stack(norms, 1)

    def training_step(self, batch: RadarBatch, caption_ids: Tensor, caption_mask: Tensor) -> Stage2Output:
        """一次 Stage 2 前向：同时计算流式与 commit 两条路径的三项损失。"""
        radar = self.radar_encoder.forward_sequence(batch.rd, batch.points, batch.point_mask, batch.frame_mask)
        # 固定 Prompt 的 K/V 在 stream 与 commit 两路重复使用，只需构建一次。
        cache = self.event_former.build_prompt_cache(radar.radar_tokens.shape[0])
        # 按 update 逐个喂、逐步演化状态，模拟真实推理时事件尚未结束、只能看到当前为止数据的场景
        stream, norms = self._stream_view(radar.radar_tokens, radar.radar_token_mask, cache)
        # 一次性看完整段 radar token 做精修，模拟事件结束后对全局的总结
        commit = self.event_former.commit_refine(radar.radar_tokens, radar.radar_token_mask, cache).event_tokens
        # 事件与文本共享同一投影，映射到同一归一化语义空间。
        stream_key, commit_key = self.semantic_projection(stream), self.semantic_projection(commit)
        text_hidden = self.event_former.adapter.encode_text(caption_ids, caption_mask)
        text_key = self.semantic_projection(text_hidden, caption_mask)
        # 流式与 commit 分别与文本对齐，取平均后一起参与优化。
        # 对比损失：让事件特征靠近相应的文本描述，远离不匹配的
        align = (
            symmetric_alignment_loss(stream_key, text_key, self.contrastive_temperature)
            + symmetric_alignment_loss(commit_key, text_key, self.contrastive_temperature)
        ) / 2
        # 一致性损失：要求 stream 和 commit 在投影后的方向尽可能一致，避免流式理解和完整理解相差太远
        consistency = cosine_consistency_loss(stream_key, commit_key)
        # 语言建模用 commit 事件作前缀；输入 caption 去掉末位以错开预测目标
        # 标准 causal LM 的错位技巧——输入第 0~n-2 个 token，预测第 1~n-1 个
        caption_logits = self.event_former.adapter.caption_logits(commit, caption_ids[:, :-1], caption_mask[:, :-1])
        text = causal_caption_cross_entropy(
            caption_logits, self.event_former.config.event_tokens, caption_ids, caption_mask
        )
        return Stage2Output(
            Stage2Loss(text, align, consistency), stream, commit, stream_key, commit_key, text_key, norms
        )

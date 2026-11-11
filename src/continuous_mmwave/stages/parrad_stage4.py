"""Differentiable objectives through the completely frozen semantic/LLM teacher."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def summary_objective(adapter, projection, summary: Tensor, source: Tensor,
                      ids: Tensor, attention_mask: Tensor, target_mask: Tensor,
                      retain_weight: float = 1., source_mask: Tensor | None = None) -> dict[str, Tensor]:
    if target_mask.shape != ids.shape or attention_mask.shape != ids.shape:
        raise ValueError('caption masks must match ids')
    if any(p.requires_grad for p in adapter.parameters()) or any(p.requires_grad for p in projection.parameters()):
        raise ValueError('Stage4 teacher and projection must be frozen')
    # All prefix positions are excluded. The previous logit predicts each target,
    # including the first caption token after the instruction prompt.
    logits = adapter.caption_logits(summary.to(adapter.compute_dtype), ids, attention_mask)
    offset = summary.shape[1]
    predicted = logits[:, offset - 1: offset + ids.shape[1] - 1]
    valid = target_mask.bool() & attention_mask.bool()
    if not valid.any():
        raise ValueError('summary objective needs target tokens')
    caption_loss = F.cross_entropy(predicted[valid].float(), ids[valid])
    with torch.no_grad():
        source_key = projection(source.detach(), source_mask)
    summary_key = projection(summary)
    retain_loss = (1 - F.cosine_similarity(summary_key, source_key, dim=-1)).mean()
    return dict(loss=caption_loss + retain_weight * retain_loss,
                summary_loss=caption_loss, retain_loss=retain_loss)


def encode_query(adapter, projection, tokenizer, query: str, device) -> Tensor:
    text = tokenizer(query, add_special_tokens=False, return_tensors='pt')
    ids, mask = text['input_ids'].to(device), text['attention_mask'].to(device)
    with torch.no_grad():
        return projection(adapter.encode_text(ids, mask), mask)[0]


def chat_prompt_ids(tokenizer, query: str, device) -> Tensor:
    instruction = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': query}], tokenize=False, add_generation_prompt=True)
    return tokenizer(instruction, add_special_tokens=False, return_tensors='pt')['input_ids'].to(device)


def chat_supervision(tokenizer, query: str, answer: str, device):
    prefix = chat_prompt_ids(tokenizer, query, device)
    target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    if not answer.strip() or prefix.shape[1] + len(target) > 512:
        raise ValueError('empty or excessively long assistant answer')
    ids = torch.cat((prefix, torch.tensor([target], device=device, dtype=torch.long)), 1)
    targets = torch.zeros_like(ids, dtype=torch.bool); targets[:, prefix.shape[1]:] = True
    return ids, torch.ones_like(ids, dtype=torch.bool), targets


@torch.no_grad()
def generate_from_memory(adapter, tokenizer, context: Tensor, query: str, *, max_new_tokens: int = 96) -> str:
    if not hasattr(adapter, 'model'):
        raise TypeError('Full LLM generation needs a native causal-LM adapter')
    ids = chat_prompt_ids(tokenizer, query, context.device)
    embedded = torch.cat((context.to(adapter.compute_dtype), adapter.embed_input_ids(ids)), dim=1)
    mask = torch.ones(embedded.shape[:2], device=embedded.device, dtype=torch.long)
    was_training = adapter.model.training
    adapter.model.eval()
    try:
        output = adapter.model.generate(inputs_embeds=embedded, attention_mask=mask,
                                        do_sample=False, max_new_tokens=max_new_tokens,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id, use_cache=True)
    finally:
        adapter.model.train(was_training)
    return tokenizer.decode(output[0], skip_special_tokens=True)

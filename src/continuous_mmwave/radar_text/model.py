"""Dual CNN + offline BiGRU + ordered pooling + soft radar prefix for a causal LM."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import RadarTextConfig


PROMPT = (
    "Describe the person's activities in this radar recording in temporal order. "
    "Only describe activities supported by the radar.\n<RADAR_INPUT>"
)


def md_time_patches(md, lengths, context):
    """[B,T,D] -> [B,T,1,D,context]; repeat only true sequence edges."""
    b, t, d = md.shape
    offsets = torch.arange(context, device=md.device) - context // 2
    indices = torch.arange(t, device=md.device)[None, :, None] + offsets[None, None]
    indices = indices.expand(b, -1, -1).clamp(min=0)
    indices = torch.minimum(indices, lengths.to(md.device)[:, None, None] - 1)
    patches = md[torch.arange(b, device=md.device)[:, None, None], indices]
    return patches.permute(0, 1, 3, 2).unsqueeze(2)


class FrameCNN(nn.Sequential):
    def __init__(self, channels, feature_dim):
        layers = []
        for out_channels in (32, 64, 128):
            layers.extend([nn.Conv2d(channels, out_channels, 3, stride=2, padding=1),
                           nn.GroupNorm(8, out_channels), nn.GELU()])
            channels = out_channels
        layers.extend([nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, feature_dim)])
        super().__init__(*layers)


class RadarEncoder(nn.Module):
    def __init__(self, config: RadarTextConfig):
        super().__init__()
        self.config = config
        self.rd_cnn = FrameCNN(config.rd_channels, config.feature_dim)
        self.md_cnn = FrameCNN(1, config.feature_dim)
        self.gru = nn.GRU(config.feature_dim * 2, config.gru_hidden, config.gru_layers,
                          batch_first=True, bidirectional=True,
                          dropout=config.dropout if config.gru_layers > 1 else 0)

    def forward(self, rd, md, lengths):
        if rd.ndim != 5 or md.ndim != 3 or rd.shape[:2] != md.shape[:2]:
            raise ValueError("expected aligned RD [B,T,C,H,W] and MD [B,T,D]")
        b, t = md.shape[:2]
        lengths = lengths.to(device=rd.device, dtype=torch.long)
        if lengths.shape != (b,) or (lengths <= 0).any() or (lengths > t).any():
            raise ValueError("invalid sequence lengths")
        if rd.shape[2] != self.config.rd_channels or min(*rd.shape[3:], md.shape[2]) < 1:
            raise ValueError("RD/MD spatial dimensions do not match encoder")
        valid = torch.arange(t, device=rd.device)[None] < lengths[:, None]
        rd, md = rd.float().clone(), md.float().clone()
        # Normalization is local to each real window; no statistics from validation/test.
        for i, length in enumerate(lengths.tolist()):
            for tensor in (rd, md):
                x = tensor[i, :length]
                if not torch.isfinite(x).all():
                    raise ValueError("radar values must be finite")
                if self.config.normalization == "signed_log_zscore":
                    x = x.sign() * x.abs().log1p()
                    tensor[i, :length] = (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-5)
        rd_features = self.rd_cnn(rd[valid])
        patches = md_time_patches(md, lengths, self.config.md_context)
        md_features = self.md_cnn(patches[valid])
        fused = rd_features.new_zeros(b, t, self.config.feature_dim * 2)
        fused[valid] = torch.cat([rd_features, md_features], dim=-1)
        packed = pack_padded_sequence(fused, lengths.cpu(), batch_first=True, enforce_sorted=False)
        hidden, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True, total_length=t)
        tokens = torch.stack([
            F.adaptive_avg_pool1d(hidden[i, :length].T, self.config.tokens).T
            for i, length in enumerate(lengths.tolist())
        ])
        return hidden, tokens


class RadarPretrainer(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.encoder = RadarEncoder(config)
        self.classifier = nn.Linear(config.gru_hidden * 2, num_classes)

    def forward(self, rd, md, lengths):
        hidden, _ = self.encoder(rd, md, lengths)
        return self.classifier(hidden)


class RadarTextModel(nn.Module):
    def __init__(self, config, llm, tokenizer, *, lora_rank=0, lora_alpha=32,
                 lora_dropout=0.05, prompt=PROMPT):
        super().__init__()
        self.config, self.tokenizer, self.prompt = config, tokenizer, prompt
        llm.requires_grad_(False)
        if lora_rank:
            try:
                from peft import LoraConfig, TaskType, get_peft_model
            except ImportError as error:
                raise RuntimeError("LoRA requires peft in the selected Python environment") from error
            llm = get_peft_model(llm, LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=lora_rank, lora_alpha=lora_alpha,
                lora_dropout=lora_dropout, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ))
        self.llm, self.lora_rank = llm, lora_rank
        self.lora_alpha, self.lora_dropout = lora_alpha, lora_dropout
        self.encoder = RadarEncoder(config)
        width = llm.get_input_embeddings().weight.shape[1]
        self.projector = nn.Sequential(nn.Linear(config.gru_hidden * 2, width), nn.GELU(), nn.Linear(width, width))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        rendered = tokenizer.apply_chat_template(self._messages(), tokenize=False, add_generation_prompt=True)
        if rendered.count("<RADAR_INPUT>") != 1:
            raise ValueError("prompt must have exactly one <RADAR_INPUT> marker")
        before, after = rendered.split("<RADAR_INPUT>")
        self.before_ids = tokenizer.encode(before, add_special_tokens=False)
        self.after_ids = tokenizer.encode(after, add_special_tokens=False)
        self.rendered_prompt = rendered
        self.train()

    def _messages(self):
        return [{"role": "system", "content": "You describe human activities from radar evidence."},
                {"role": "user", "content": self.prompt}]

    def train(self, mode=True):
        super().train(mode)
        if not self.lora_rank:
            self.llm.eval()
        return self

    def answer_ids(self, caption):
        if not caption.strip():
            raise ValueError("caption cannot be empty")
        rendered = self.tokenizer.apply_chat_template(
            self._messages() + [{"role": "assistant", "content": caption}], tokenize=False,
        )
        if not rendered.startswith(self.rendered_prompt):
            raise ValueError("chat template assistant prefix does not match generation prefix")
        ids = self.tokenizer.encode(rendered[len(self.rendered_prompt):], add_special_tokens=False)
        if not ids or len(ids) > self.config.max_answer_tokens:
            raise ValueError("caption exceeds max_answer_tokens or has no answer tokens")
        return ids

    def prepare_inputs(self, rd, md, lengths, captions=None):
        if captions is not None and len(captions) != rd.shape[0]:
            raise ValueError("one caption required per radar window")
        _, tokens = self.encoder(rd, md, lengths)
        embedding = self.llm.get_input_embeddings()
        radar = self.projector(tokens).to(embedding.weight.dtype)
        device = radar.device
        before = embedding(torch.tensor(self.before_ids, device=device))
        after = embedding(torch.tensor(self.after_ids, device=device))
        sequences, labels = [], []
        for i in range(len(radar)):
            prefix = torch.cat([before, radar[i], after])
            target = [] if captions is None else self.answer_ids(captions[i])
            target_ids = torch.tensor(target, device=device, dtype=torch.long)
            sequences.append(torch.cat([prefix, embedding(target_ids)]))
            labels.append(torch.cat([torch.full((len(prefix),), -100, device=device), target_ids]))
        max_length = max(map(len, sequences))
        inputs = radar.new_zeros(len(sequences), max_length, radar.shape[-1])
        mask = torch.zeros(len(sequences), max_length, device=device, dtype=torch.long)
        targets = torch.full_like(mask, -100)
        for i, (sequence, label) in enumerate(zip(sequences, labels)):
            # Decoder generation reads the last position; inference uses left padding.
            start = max_length - len(sequence) if captions is None else 0
            inputs[i, start:start + len(sequence)] = sequence
            mask[i, start:start + len(sequence)] = 1
            targets[i, start:start + len(sequence)] = label
        result = dict(inputs_embeds=inputs, attention_mask=mask,
                      position_ids=(mask.cumsum(-1) - 1).clamp_min(0))
        if captions is not None:
            result["labels"] = targets
        return result

    def forward(self, rd, md, lengths, captions):
        # Freezing parameters preserves gradients to radar inputs; torch.no_grad would not.
        return self.llm(**self.prepare_inputs(rd, md, lengths, captions), use_cache=False)

    @torch.no_grad()
    def generate(self, rd, md, lengths, *, max_new_tokens=96):
        if self.training:
            raise ValueError("call model.eval() before generation")
        inputs = self.prepare_inputs(rd, md, lengths)
        # GenerationMixin grows attention_mask at each cached step and derives positions.
        # A user-supplied static position_ids tensor would repeat the final RoPE position.
        inputs.pop("position_ids")
        ids = self.llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=self.tokenizer.pad_token_id,
                                eos_token_id=self.tokenizer.eos_token_id, use_cache=True)
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

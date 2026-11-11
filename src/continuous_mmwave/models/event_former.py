"""Streaming/commit Semantic Event Former from v2.2."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from continuous_mmwave.config import FIXED_EVENT_PROMPT, ModelConfig


def _additive(allowed: Tensor, dtype: torch.dtype) -> Tensor:
    """把 bool 可见性矩阵转成加法注意力掩码：可见为 0，不可见为 dtype 最小值。"""
    return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(~allowed, torch.finfo(dtype).min)


def build_prefix_event_mask(prompt_length: int, event_tokens: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """构建 ``[固定 Prompt | Event Tokens]`` 的注意力掩码。

    - Prompt 内部：因果（只能看左侧的 prompt token）。
    - Event tokens：双向可见，且可看整段 Prompt（Prompt 已完整，无需因果限制）。
    返回形状为 [1,1,L,L]，可直接作为 additive attn_mask 使用。
    """
    total = prompt_length + event_tokens
    allowed = torch.zeros((total, total), dtype=torch.bool, device=device)
    allowed[:prompt_length, :prompt_length] = torch.ones(
        (prompt_length, prompt_length), dtype=torch.bool, device=device
    ).tril()
    allowed[prompt_length:, :] = True
    return _additive(allowed, dtype).view(1, 1, total, total)


def build_causal_padding_mask(mask: Tensor, dtype: torch.dtype) -> Tensor:
    """因果 + padding 的组合掩码，形状 [B,1,L,L]。"""
    if mask.ndim != 2:
        raise ValueError("attention mask must be [B,L]")
    length = mask.shape[1]
    causal = torch.ones((length, length), dtype=torch.bool, device=mask.device).tril()
    # 第 i 个 query 只能看 j<=i 且 mask[j] 有效的 key。
    allowed = causal.unsqueeze(0) & mask.bool().unsqueeze(1)
    return _additive(allowed, dtype).unsqueeze(1)


class BottleneckRadarCrossAttention(nn.Module):
    """LLM 隐层与 radar token（默认 256 维）间的显式 bottleneck cross-attention。

    不直接使用 LLM 的注意力头，而是先降到 ``bottleneck_dim(512)`` 做多头注意力，
    再由 ``output`` 投回 LLM 维度，并以 ``tanh(alpha)`` 作为残差门控。
    """

    def __init__(
        self,
        llm_dim: int,
        radar_dim: int,
        bottleneck_dim: int,
        heads: int,
        dropout: float,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if bottleneck_dim % heads:
            raise ValueError("bottleneck dimension must be divisible by heads")
        if not 0 <= init_scale < 1:
            raise ValueError("cross-attention init scale must be in [0,1)")
        self.heads, self.head_dim = heads, bottleneck_dim // heads
        self.query = nn.Linear(llm_dim, bottleneck_dim)
        self.key = nn.Linear(radar_dim, bottleneck_dim)
        self.value = nn.Linear(radar_dim, bottleneck_dim)
        self.output = nn.Linear(bottleneck_dim, llm_dim)
        self.dropout = nn.Dropout(dropout)
        # 参数保留 tanh 约束，但按“有效残差比例”配置，避免调用方依赖 raw alpha。
        self.alpha = nn.Parameter(torch.tensor(math.atanh(init_scale), dtype=torch.float32))

    def _heads(self, tensor: Tensor) -> Tensor:
        """[B,L,D] -> [B,heads,L,head_dim]。"""
        return tensor.reshape(tensor.shape[0], tensor.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def forward(self, events: Tensor, radar: Tensor, radar_mask: Tensor | None = None) -> Tensor:
        """events [B,E,llm_dim]，radar [B,R,radar_dim] -> [B,E,llm_dim]。"""
        output_dtype = events.dtype
        events = events.to(self.query.weight.dtype)
        radar = radar.to(self.key.weight.dtype)
        q, k, v = self._heads(self.query(events)), self._heads(self.key(radar)), self._heads(self.value(radar))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        invalid_all = None
        if radar_mask is not None:
            if radar_mask.shape != radar.shape[:2]:
                raise ValueError("radar_mask must be [B,L]")
            # 与 QueryPooling 同理：先保证每行至少有一个可用 key，再将无效行整体置零。
            # 处理某些样本的 radar token 全部无效
            safe = radar_mask.bool().clone()
            invalid_all = ~safe.any(1)
            safe[invalid_all, 0] = True
            scores = scores.masked_fill(~safe[:, None, None, :], torch.finfo(scores.dtype).min)
        context = (self.dropout(scores.softmax(-1)) @ v).transpose(1, 2).reshape(events.shape[0], events.shape[1], -1)
        if invalid_all is not None:
            context = torch.where(invalid_all[:, None, None], torch.zeros_like(context), context)
        return (events + torch.tanh(self.alpha) * self.output(context)).to(output_dtype)


class EventStateGate(nn.Module):
    """Token-wise convex gate between the previous state and a new proposal."""

    def __init__(self, dim: int, initial_update: float = 0.1) -> None:
        super().__init__()
        if not 0 < initial_update < 1:
            raise ValueError("initial_update must be in (0,1)")
        self.norm = nn.LayerNorm(dim)
        self.network = nn.Sequential(nn.Linear(dim * 2 + 1, dim), nn.GELU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, math.log(initial_update / (1 - initial_update)))

    def forward(self, previous: Tensor, proposal: Tensor, maturity: float | Tensor) -> tuple[Tensor, Tensor]:
        """previous/proposal [B,E,D]，maturity 为标量或 [B] -> (新状态, 门控值)。"""
        batch, tokens = previous.shape[:2]
        maturity = torch.as_tensor(maturity, dtype=previous.dtype, device=previous.device)
        if maturity.ndim == 0:
            maturity = maturity.expand(batch)
        if maturity.shape != (batch,):
            raise ValueError("maturity must be scalar or [B]")
        # maturity 原本是[B]，需要扩展成 [B, E, 1]
        feature = maturity[:, None, None].expand(batch, tokens, 1)
        delta = proposal - previous
        gates = torch.sigmoid(self.network(torch.cat((self.norm(previous), self.norm(delta), feature), -1)))
        return torch.lerp(previous, proposal, gates), gates


class NativeLLMAdapter(nn.Module):
    """冻结 LLM 前 N 层的统一接口，使 Event Former 不依赖具体模型实现。

    只暴露逐层自注意力/前馈与文本编码能力；实现方负责保证权重冻结。
    子类必须提供 ``hidden_size`` / ``layer_count`` / ``vocab_size``。
    """

    hidden_size: int
    layer_count: int
    vocab_size: int

    @property
    def compute_dtype(self) -> torch.dtype:
        raise NotImplementedError

    @property
    def supports_prompt_cache(self) -> bool:
        """是否支持固定 Prompt 的 K/V 缓存（Qwen2 支持，Tiny 不支持）。"""
        return False

    # token id -> 向量
    def embed_input_ids(self, ids: Tensor) -> Tensor:
        raise NotImplementedError

    # 预计算位置信息，Qwen使用的位置编码是 RoPE
    def prepare_position_context(self, hidden: Tensor) -> Any:
        raise NotImplementedError

    # 第 layer 层的自注意力（带残差）
    def self_attention(self, layer: int, hidden: Tensor, mask: Tensor, context: Any) -> Tensor:
        raise NotImplementedError

    # 第 layer 层的前馈层（带残差）
    def feed_forward(self, layer: int, hidden: Tensor) -> Tensor:
        raise NotImplementedError

    # 固定文本描述也只能使用前 N 层，这样和雷达特征经过相同层得到的语义空间基本一致
    def encode_text(self, ids: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError

    # 事件 token 作前缀写自然语言描述 -> 生成损失函数要使用
    def caption_logits(self, events: Tensor, ids: Tensor, mask: Tensor) -> Tensor:
        raise NotImplementedError

    # 构造文本prompt的 KV cache
    def build_prompt_cache(self, prompt_ids: Tensor, batch: int) -> Any:
        return None

    # 克隆 prompt 的 KV cache
    def clone_prompt_cache(self, cache: Any) -> Any:
        return cache

    # 带有 prompt cache的自注意力模块
    def self_attention_with_prompt_cache(self, layer: int, hidden: Tensor, cache: Any, prompt_length: int) -> Tensor:
        raise NotImplementedError


class _TinyLayer(nn.Module):
    """与 Qwen 第结构对齐的极小层，仅用于 CPU 冒烟测试。"""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.heads = heads


class TinyLLMAdapter(NativeLLMAdapter):
    """随机初始化的微型“LLM”，用于在无 GPU/无权重时验证接口、梯度与 checkpoint。"""

    def __init__(self, vocab_size: int, hidden_size: int, layers: int, heads: int) -> None:
        super().__init__()
        self.hidden_size, self.layer_count, self.vocab_size = hidden_size, layers, vocab_size
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([_TinyLayer(hidden_size, heads) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        # 权重共享，与 Qwen2 的 tie_word_embeddings 行为一致。
        self.lm_head.weight = self.token_embedding.weight
        for parameter in self.parameters():
            parameter.requires_grad = False

    @property
    def compute_dtype(self) -> torch.dtype:
        """全模型使用与词嵌入一致的精度（Qwen 路径下即 bf16）。"""
        return self.token_embedding.weight.dtype

    def embed_input_ids(self, ids: Tensor) -> Tensor:
        return self.token_embedding(ids)

    def prepare_position_context(self, hidden: Tensor) -> None:
        """Tiny 模型不用位置编码，返回 None 以满足接口。"""
        return None

    def self_attention(self, layer: int, hidden: Tensor, mask: Tensor, context: None = None) -> Tensor:
        block = self.layers[layer]
        length = hidden.shape[1]
        # MultiheadAttention 要求 [B*heads,L,L] 的逐头掩码，而此处 mask 为 [B,1,L,L]。
        expanded = mask.expand(hidden.shape[0], block.heads, length, length).reshape(-1, length, length)
        attended, _ = block.self_attn(
            block.input_layernorm(hidden),
            block.input_layernorm(hidden),
            block.input_layernorm(hidden),
            attn_mask=expanded,
            need_weights=False,
        )
        return hidden + attended

    def feed_forward(self, layer: int, hidden: Tensor) -> Tensor:
        block = self.layers[layer]
        return hidden + block.mlp(block.post_attention_layernorm(hidden))

    def _run(self, hidden: Tensor, mask: Tensor) -> Tensor:
        """前向遍历全部层（测试用，不做 activation checkpoint）。"""
        context = self.prepare_position_context(hidden)
        for layer in range(self.layer_count):
            hidden = self.feed_forward(layer, self.self_attention(layer, hidden, mask, context))
        return hidden

    def encode_text(self, ids: Tensor, mask: Tensor) -> Tensor:
        """caption 文本经过前 N 层后的隐层表示 [B,L,hidden]。"""
        hidden = self.embed_input_ids(ids)
        return self._run(hidden, build_causal_padding_mask(mask, hidden.dtype))

    def caption_logits(self, events: Tensor, ids: Tensor, mask: Tensor) -> Tensor:
        """把事件 token 拼在 caption 前做因果语言建模，返回全部位置的 logits。"""
        hidden = torch.cat((events, self.embed_input_ids(ids)), 1)
        full_mask = torch.cat((torch.ones(events.shape[:2], dtype=torch.bool, device=events.device), mask.bool()), 1)
        return self.lm_head(self.final_norm(self._run(hidden, build_causal_padding_mask(full_mask, hidden.dtype))))


class Gemma3Adapter(NativeLLMAdapter):
    """Gemma 3 text adapter with q/v LoRA on the selected layer prefix.

    Gemma 3 applies four RMSNorm operations per decoder block and alternates
    local/global RoPE.  The block is split here so radar cross-attention can
    remain between native self-attention and the native feed-forward path.
    """

    def __init__(self, causal_lm: nn.Module, num_layers: int = 4, lora_rank: int = 16) -> None:
        super().__init__()
        if getattr(causal_lm.config, "model_type", None) not in {"gemma3", "gemma3_text"}:
            raise ValueError("Gemma 3 adapter requires a Gemma 3 text causal LM")
        if not 0 < num_layers <= causal_lm.config.num_hidden_layers:
            raise ValueError("invalid Gemma 3 layer prefix")

        from peft import LoraConfig, get_peft_model

        causal_lm.config.use_cache = False
        causal_lm.gradient_checkpointing_enable()
        for parameter in causal_lm.parameters():
            parameter.requires_grad = False
        self.model = get_peft_model(
            causal_lm,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                lora_dropout=0,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                layers_to_transform=list(range(num_layers)),
                layers_pattern="layers",
            ),
        )
        self.layer_count = num_layers
        self.hidden_size = int(causal_lm.config.hidden_size)
        self.vocab_size = int(causal_lm.config.vocab_size)

    @classmethod
    def from_pretrained(
        cls, name: str, num_layers: int = 4, lora_rank: int = 16, **kwargs: Any
    ) -> Gemma3Adapter:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        return cls(model, num_layers, lora_rank)

    @property
    def base(self) -> nn.Module:
        """Return the causal LM below the PEFT wrapper."""
        return self.model.get_base_model()

    @property
    def core(self) -> nn.Module:
        """Gemma3TextModel containing embeddings, decoder layers and RoPE."""
        return self.base.model

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.core.embed_tokens.weight.dtype

    def embed_input_ids(self, ids: Tensor) -> Tensor:
        return self.core.embed_tokens(ids)

    def prepare_position_context(self, hidden: Tensor) -> tuple[Tensor, Any, Any]:
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        return (
            positions,
            self.core.rotary_emb(hidden, positions),
            self.core.rotary_emb_local(hidden, positions),
        )

    def self_attention(self, layer: int, hidden: Tensor, mask: Tensor, context: tuple[Tensor, Any, Any]) -> Tensor:
        block = self.core.layers[layer]
        positions, global_rope, local_rope = context
        if block.self_attn.is_sliding:
            query = positions[0, :, None]
            key = positions[0, None, :]
            local_allowed = key > query - int(self.core.config.sliding_window)
            local_mask = _additive(local_allowed, hidden.dtype).view(1, 1, hidden.shape[1], hidden.shape[1])
            mask = torch.minimum(mask, local_mask)
        attended = block.self_attn(
            hidden_states=block.input_layernorm(hidden),
            position_embeddings=local_rope if block.self_attn.is_sliding else global_rope,
            attention_mask=mask,
            use_cache=False,
        )
        if isinstance(attended, tuple):
            attended = attended[0]
        return hidden + block.post_attention_layernorm(attended)

    def feed_forward(self, layer: int, hidden: Tensor) -> Tensor:
        block = self.core.layers[layer]
        transformed = block.mlp(block.pre_feedforward_layernorm(hidden))
        return hidden + block.post_feedforward_layernorm(transformed)

    def encode_text(self, ids: Tensor, mask: Tensor) -> Tensor:
        hidden = self.embed_input_ids(ids)
        additive = build_causal_padding_mask(mask, hidden.dtype)
        context = self.prepare_position_context(hidden)
        for layer in range(self.layer_count):
            hidden = self.feed_forward(layer, self.self_attention(layer, hidden, additive, context))
        return hidden

    def caption_logits(self, events: Tensor, ids: Tensor, mask: Tensor) -> Tensor:
        hidden = torch.cat((events, self.embed_input_ids(ids)), 1)
        attention = torch.cat((torch.ones(events.shape[:2], dtype=mask.dtype, device=events.device), mask), 1)
        return self.model(inputs_embeds=hidden, attention_mask=attention, use_cache=False).logits


def _cloneable_cache_prefix(cache: Any, layer_count: int) -> list[tuple[Tensor, Tensor, Any]]:
    """Clone only the initialized Qwen cache prefix, excluding empty model tail layers."""
    data: list[tuple[Tensor, Tensor, Any]] = []
    for index, (key, value, sliding) in enumerate(cache):
        if index >= layer_count:
            break
        if key is None or value is None:
            raise ValueError(f"Qwen prompt cache layer {index} is uninitialized")
        data.append((key.clone(), value.clone(), sliding))
    if len(data) != layer_count:
        raise ValueError(f"Qwen prompt cache has {len(data)} initialized layers; expected {layer_count}")
    return data


class Qwen2Adapter(NativeLLMAdapter):
    """生产用适配器：冻结 Qwen2/Qwen2.5，仅在前 ``num_layers`` 层的 q/v 上训练 LoRA。
    由于要在 LLM 的前 num_layers 层的 self-attention和 FFN之间插入 radar cross-attention，
    因此需要手动拆解出前 N 层的 attn/mlp 逐层调用，以便在层间插入 radar cross-attention，
    而不使用 transformers 的整模型 forward。
    """

    def __init__(self, causal_lm: nn.Module, num_layers: int = 4, lora_rank: int = 16) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model

        if not 0 < num_layers <= causal_lm.config.num_hidden_layers:
            raise ValueError("invalid Qwen layer prefix")
        # use_cache 与 gradient checkpointing 在手动逐层 forward 下无意义，关闭即可。
        causal_lm.config.use_cache = False
        causal_lm.gradient_checkpointing_enable()
        # 先冻结全部参数，再由 PEFT 打开 LoRA 适配器的梯度。
        for parameter in causal_lm.parameters():
            parameter.requires_grad = False
        self.model = get_peft_model(
            causal_lm,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                lora_dropout=0,
                bias="none",
                # 方案书要求：仅 q/v 投影适用 LoRA，且只作用于前 N 层。
                target_modules=["q_proj", "v_proj"],
                layers_to_transform=list(range(num_layers)),
                layers_pattern="layers",
            ),
        )
        # Some PEFT releases keep LoRA tensors in the base model's FP16 dtype.
        # GradScaler intentionally rejects unscaling FP16 parameter gradients, so
        # keep only the small trainable adapter tensors in FP32 while the frozen
        # Qwen backbone remains in its configured low-precision dtype.
        for parameter in self.model.parameters():
            if parameter.requires_grad and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
        self.layer_count = num_layers
        self.hidden_size = int(causal_lm.config.hidden_size)
        self.vocab_size = int(causal_lm.config.vocab_size)

    @classmethod
    def from_pretrained(cls, name: str, num_layers: int = 4, lora_rank: int = 16, **kwargs: Any) -> Qwen2Adapter:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        if getattr(model.config, "model_type", None) != "qwen2":
            raise ValueError("production adapter requires Qwen2/Qwen2.5")
        return cls(model, num_layers, lora_rank)

    @property
    def base(self) -> nn.Module:
        """剥掉 PEFT 包装后的原始 CausalLM。"""
        return self.model.get_base_model()

    @property
    def core(self) -> nn.Module:
        """Qwen2Model 本体，提供 embed_tokens / layers / rotary_emb。"""
        return self.base.model

    @property
    def compute_dtype(self) -> torch.dtype:
        """bfloat16（配置决定），Event Former 的旁路模块会跟随此精度。"""
        return self.core.embed_tokens.weight.dtype

    @property
    def supports_prompt_cache(self) -> bool:
        return True

    def embed_input_ids(self, ids: Tensor) -> Tensor:
        return self.core.embed_tokens(ids)

    def prepare_position_context(self, hidden: Tensor) -> tuple[Tensor, Any]:
        """一次性算好位置 id 与 RoPE 嵌入，供各层复用。"""
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        return positions, self.core.rotary_emb(hidden, positions)

    def _attention(
        self, layer: int, hidden: Tensor, mask: Tensor, context: tuple[Tensor, Any], cache: Any = None
    ) -> Tensor:
        """带残差的单层自注意力，可选择性写入/读取 KV cache。"""
        block = self.core.layers[layer]
        positions, rotary = context
        attended = block.self_attn(
            hidden_states=block.input_layernorm(hidden),
            attention_mask=mask,
            position_ids=positions,
            position_embeddings=rotary,
            past_key_values=cache,
        )
        # 部分 transformers 版本返回 (hidden, attn_weights) 而非仅 hidden。
        if isinstance(attended, tuple):
            attended = attended[0]
        return hidden + attended

    def self_attention(self, layer: int, hidden: Tensor, mask: Tensor, context: tuple[Tensor, Any]) -> Tensor:
        return self._attention(layer, hidden, mask, context)

    def feed_forward(self, layer: int, hidden: Tensor) -> Tensor:
        block = self.core.layers[layer]
        return hidden + block.mlp(block.post_attention_layernorm(hidden))

    def encode_text(self, ids: Tensor, mask: Tensor) -> Tensor:
        """caption 经过前 N 层（含 LoRA）后的隐层，用于语义对齐。"""
        hidden = self.embed_input_ids(ids)
        additive = build_causal_padding_mask(mask, hidden.dtype)
        context = self.prepare_position_context(hidden)
        for layer in range(self.layer_count):
            hidden = self.feed_forward(layer, self.self_attention(layer, hidden, additive, context))
        return hidden

    def caption_logits(self, events: Tensor, ids: Tensor, mask: Tensor) -> Tensor:
        """事件 token 作为前缀、caption 作为续写，交给完整 CausalLM 得到词表 logits。"""
        hidden = torch.cat((events, self.embed_input_ids(ids)), 1)
        attention = torch.cat((torch.ones(events.shape[:2], dtype=mask.dtype, device=events.device), mask), 1)
        return self.model(inputs_embeds=hidden, attention_mask=attention).logits

    def build_prompt_cache(self, prompt_ids: Tensor, batch: int) -> Any:
        """预计算固定 Event Prompt 在前 N 层的 K/V（prompt 每步不变时只需算一次）。

        返回值在整个训练步内被反复 clone，不会在流式迭代中被原地改写。
        """
        from transformers.cache_utils import DynamicCache

        ids = prompt_ids.unsqueeze(0).expand(batch, -1)
        hidden = self.embed_input_ids(ids)
        mask = build_causal_padding_mask(torch.ones_like(ids, dtype=torch.bool), hidden.dtype)
        context = self.prepare_position_context(hidden)
        cache = DynamicCache(config=self.core.config)
        for layer in range(self.layer_count):
            # 逐层把 K/V 写入 cache，返回的 hidden 在此处无用。
            hidden = self.feed_forward(layer, self._attention(layer, hidden, mask, context, cache))
        return cache

    def clone_prompt_cache(self, cache: Any) -> Any:
        """深拷 K/V，避免不同 update（或 stream/commit 两路）互相干扰。"""
        from transformers.cache_utils import DynamicCache

        data = _cloneable_cache_prefix(cache, self.layer_count)
        return DynamicCache(ddp_cache_data=data, config=self.core.config)

    def self_attention_with_prompt_cache(self, layer: int, hidden: Tensor, cache: Any, prompt_length: int) -> Tensor:
        """Event token 只作为 query， attends 到缓存的 Prompt K/V（无需再拼 Prompt）。"""
        # 位置 id 紧接在 Prompt 之后，与拼接 Prompt 的实现保持等价。
        positions = torch.arange(prompt_length, prompt_length + hidden.shape[1], device=hidden.device).unsqueeze(0)
        rotary = self.core.rotary_emb(hidden, positions)
        # Prompt 已完整可见，因此只需全零（不遮蔽）掩码。
        mask = hidden.new_zeros((hidden.shape[0], 1, hidden.shape[1], prompt_length + hidden.shape[1]))
        return self._attention(layer, hidden, mask, (positions, rotary), cache)


@dataclass(frozen=True)
class EventFormerOutput:
    event_tokens: Tensor  # [B,E,llm_dim]
    gates: Tensor  # [B,E,1]，状态门控值，可用于监控更新强度
    state_norm: Tensor  # 事件状态的 L2 范数，用于监控是否崩溃/爆炸
    fixed_prompt_text: str


def _relative_time(length: int, dim: int, reference: Tensor) -> Tensor:
    """Commit 路径的相对时间编码：把 [0,1] 均匀铺到整段事件上。

    与流式路径不同，commit 一次性看到完整事件序列，因此用归一化相对时间
    而非全局 update 序号（后者在长度可变时不可比）。
    """
    positions = torch.linspace(0, 1, length, dtype=reference.dtype, device=reference.device).unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, dim, 2, dtype=reference.dtype, device=reference.device) * (-math.log(10_000) / max(dim, 1))
    )
    result = reference.new_zeros((length, dim))
    result[:, 0::2] = torch.sin(positions * frequency)
    result[:, 1::2] = torch.cos(positions * frequency[: result[:, 1::2].shape[1]])
    return result


class StreamingSemanticEventFormer(nn.Module):
    """把 radar token 逐步写入固定长度事件记忆（Streaming），并可一次性重建（Commit）。

    两条路径共享同一套参数（事件 token 参数、cross-attention、输出投影、状态门控
    与固定的 Event Prompt），区别仅在于输入 radar token 的数量与时间编码方式。
    """

    def __init__(
        self,
        adapter: NativeLLMAdapter,
        config: ModelConfig,
        fixed_prompt_ids: Tensor,
        fixed_prompt_text: str = FIXED_EVENT_PROMPT,
    ) -> None:
        super().__init__()
        if adapter.hidden_size != config.llm_dim or adapter.layer_count < config.llm_layers:
            raise ValueError("adapter dimensions/layers do not match")
        if fixed_prompt_text != config.fixed_event_prompt:
            raise ValueError("Streaming Path must use the fixed Event Prompt")
        if fixed_prompt_ids.ndim != 1 or not fixed_prompt_ids.numel():
            raise ValueError("fixed prompt ids must be non-empty [L]")
        self.adapter, self.config, self.fixed_prompt_text = adapter, config, fixed_prompt_text
        # Prompt ids 作为 buffer 保存：会随 checkpoint 一起存/取，但不是可训练参数。
        self.register_buffer("fixed_prompt_ids", fixed_prompt_ids.long().clone())
        # 事件记忆的初始值，所有流式 update 均从这里出发。
        # Trainable Event Former parameters stay in FP32. CUDA autocast uses
        # low-precision kernels during forward while GradScaler accumulates safe
        # FP32 parameter gradients; the frozen LLM remains FP16/BF16.
        self.base_event_tokens = nn.Parameter(torch.empty(config.event_tokens, config.llm_dim))
        # 每层一个 cross-attention，与 LLM 前 N 层一一对应。
        self.cross_attention = nn.ModuleList(
            [
                BottleneckRadarCrossAttention(
                    config.llm_dim,
                    config.radar_dim,
                    config.cross_attn_dim,
                    config.cross_attn_heads,
                    config.dropout,
                    config.cross_attention_init_scale,
                )
                for _ in range(config.llm_layers)
            ]
        )
        self.output_projection = nn.Linear(config.llm_dim, config.llm_dim)
        self.state_gate = EventStateGate(config.llm_dim, config.event_state_initial_update)
        nn.init.normal_(self.base_event_tokens, std=0.02)
        # Wout 恒等初始化：未训练时 proposal 直接等于 LLM 输出，不做额外变换。
        nn.init.eye_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def initial_state(self, batch: int) -> Tensor:
        """流式推理的起始事件状态 [B,E,llm_dim]。"""
        return self.base_event_tokens.unsqueeze(0).expand(batch, -1, -1)

    def build_prompt_cache(self, batch: int) -> Any:
        """预先算好固定 Prompt 的 K/V；后端不支持缓存时返回 None（走拼接路径）。"""
        return (
            self.adapter.build_prompt_cache(self.fixed_prompt_ids, batch)
            if self.adapter.supports_prompt_cache
            else None
        )

    def _update(
        self, previous: Tensor, radar: Tensor, maturity: float | Tensor, radar_mask: Tensor | None, prompt_cache: Any,
        trace: Callable[[dict[str, Tensor]], None] | None = None,
    ) -> EventFormerOutput:
        """Streaming 与 Commit 共用的核心步骤：让当前事件状态吸收一批 radar token。"""
        previous, radar = previous.to(self.adapter.compute_dtype), radar.to(self.adapter.compute_dtype)
        prompt_length = self.fixed_prompt_ids.shape[0]
        if prompt_cache is not None:
            # 快路径：Prompt K/V 已缓存，每个 update 只需 clone 一份避免污染。
            cache, events = self.adapter.clone_prompt_cache(prompt_cache), previous
            for layer in range(self.config.llm_layers):
                events = self.adapter.self_attention_with_prompt_cache(layer, events, cache, prompt_length)
                events = self.cross_attention[layer](events, radar, radar_mask)
                events = self.adapter.feed_forward(layer, events)
        else:
            # 慢路径：把 Prompt 与事件 token 拼接后走带前缀掩码的前向。
            prompt = self.adapter.embed_input_ids(self.fixed_prompt_ids.unsqueeze(0).expand(previous.shape[0], -1))
            hidden = torch.cat((prompt, previous), 1)
            mask = build_prefix_event_mask(prompt_length, self.config.event_tokens, hidden.dtype, hidden.device)
            context = self.adapter.prepare_position_context(hidden)
            for layer in range(self.config.llm_layers):
                hidden = self.adapter.self_attention(layer, hidden, mask, context)
                # 只有事件 token 参与 radar cross-attention，Prompt 保持不动。
                events = self.cross_attention[layer](hidden[:, prompt_length:], radar, radar_mask)
                hidden = torch.cat((hidden[:, :prompt_length], events), 1)
                hidden = self.adapter.feed_forward(layer, hidden)
            events = hidden[:, prompt_length:]
        # Native decoder blocks already return ``input + residual``.  Feeding
        # that complete residual stream into the outer recurrent interpolation
        # would count ``previous`` twice: every update would add a nearly
        # constant Transformer residual and the state norm would grow linearly.
        # Treat only this update's residual content as the candidate state; the
        # convex gate below is solely responsible for retaining ``previous``.
        residual_content = events.to(self.output_projection.weight.dtype) - previous.to(
            self.output_projection.weight.dtype
        )
        proposal = self.output_projection(residual_content)
        # 门控插值得到新状态；maturity 越高表示事件越接近完成。
        tokens, gates = self.state_gate(previous.to(proposal.dtype), proposal, maturity)
        tokens = tokens.to(self.adapter.compute_dtype)
        if trace is not None:
            trace({
                "E_prev": previous.detach().clone(),
                "H4": events.detach().clone(),
                "R": residual_content.detach().clone(),
                "E_hat": proposal.detach().clone(),
                "alpha": gates.detach().clone(),
                "E": tokens.detach().clone(),
            })
        return EventFormerOutput(tokens, gates, torch.linalg.vector_norm(tokens, dim=(-2, -1)), self.fixed_prompt_text)

    def stream_update(
        self,
        previous: Tensor,
        radar: Tensor,
        maturity: float | Tensor,
        radar_mask: Tensor | None = None,
        prompt_cache: Any = None,
        *,
        trace: Callable[[dict[str, Tensor]], None] | None = None,
    ) -> EventFormerOutput:
        """单次流式 update，输入必须是恰好一个 update 的 4 个 radar token。"""
        if radar.ndim != 3 or radar.shape[1] != self.config.radar_tokens_per_update:
            raise ValueError("streaming update requires exactly four radar tokens")
        return self._update(previous, radar, maturity, radar_mask, prompt_cache, trace)

    def _prepare_commit(self, radar: Tensor, radar_mask: Tensor | None) -> tuple[Tensor, Tensor]:
        """把整段事件的 radar token 整理成 commit 可用的定长批。

        - 剔除填充 token，并把长度修剪到 ``max_commit_radar_tokens``（超长时沿时间维平均池化）；
        - 加入 [0,1] 相对时间编码；
        - 重新 padding 到该 batch 内最大长度，并返回对应的 [B,L] mask。
        """
        if radar.ndim != 3 or radar.shape[1] == 0:
            raise ValueError("commit radar must be non-empty [B,L,D]")
        if radar_mask is None:
            radar_mask = torch.ones(radar.shape[:2], dtype=torch.bool, device=radar.device)
        events = []
        for index in range(radar.shape[0]):
            event = radar[index, radar_mask[index].bool()]
            if not event.shape[0]:
                raise ValueError("every event needs valid radar tokens")
            if event.shape[0] > self.config.max_commit_radar_tokens:
                # 平均池化而非截断，避免丢弃事件尾部的语义。
                event = (
                    functional.adaptive_avg_pool1d(event.T.unsqueeze(0), self.config.max_commit_radar_tokens)
                    .squeeze(0)
                    .T
                )
            events.append(event + _relative_time(event.shape[0], event.shape[1], event))
        length = max(event.shape[0] for event in events)
        prepared = radar.new_zeros((radar.shape[0], length, radar.shape[2]))
        mask = torch.zeros((radar.shape[0], length), dtype=torch.bool, device=radar.device)
        for index, event in enumerate(events):
            prepared[index, : event.shape[0]], mask[index, : event.shape[0]] = event, True
        return prepared, mask

    def prepare_commit_radar_tokens(self, radar: Tensor, radar_mask: Tensor | None = None) -> Tensor:
        """只取整理后的 radar token（供测试/调试使用）。"""
        return self._prepare_commit(radar, radar_mask)[0]

    def commit_refine(
        self, radar: Tensor, radar_mask: Tensor | None = None, prompt_cache: Any = None
    ) -> EventFormerOutput:
        """Commit 路径：从初始状态出发，用整段事件一次性重建事件语义。"""
        prepared, mask = self._prepare_commit(radar, radar_mask)
        # maturity 固定为 1.0：commit 时事件已经完整。
        return self._update(self.initial_state(radar.shape[0]), prepared, 1.0, mask, prompt_cache)

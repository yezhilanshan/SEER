from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from continuous_mmwave.models.event_former import Gemma3Adapter, Qwen2Adapter
from continuous_mmwave.config import ModelConfig
from continuous_mmwave.models.event_former import TinyLLMAdapter
from continuous_mmwave.train import ExperimentConfig, TinyTokenizer, _build_llm_backend


class _IdentityPeftModel(nn.Module):
    """Keep the real Gemma model while replacing only the unavailable PEFT wrapper."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.wrapped = model

    def get_base_model(self) -> nn.Module:
        return self.wrapped

    def forward(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)


def _install_identity_peft(monkeypatch: pytest.MonkeyPatch) -> list[SimpleNamespace]:
    captured: list[SimpleNamespace] = []
    module = ModuleType("peft")
    module.__spec__ = ModuleSpec("peft", loader=None)

    def lora_config(**kwargs) -> SimpleNamespace:
        config = SimpleNamespace(**kwargs)
        captured.append(config)
        return config

    module.LoraConfig = lora_config
    module.get_peft_model = lambda model, _config: _IdentityPeftModel(model)
    monkeypatch.setitem(sys.modules, "peft", module)
    return captured


def _tiny_gemma3(sliding_window: int = 16):
    transformers = pytest.importorskip("transformers")
    config = transformers.Gemma3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=sliding_window,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return transformers.Gemma3ForCausalLM(config).eval()


def test_gemma3_adapter_matches_native_prefix_and_targets_only_qv(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_identity_peft(monkeypatch)
    model = _tiny_gemma3()
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        expected = model.model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states[2]

    adapter = Gemma3Adapter(model, num_layers=2, lora_rank=4)
    with torch.no_grad():
        actual = adapter.encode_text(ids, mask)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert len(captured) == 1
    assert captured[0].target_modules == ["q_proj", "v_proj"]
    assert captured[0].layers_to_transform == [0, 1]
    assert captured[0].layers_pattern == "layers"


def test_gemma3_adapter_caption_logits_cover_event_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_identity_peft(monkeypatch)
    adapter = Gemma3Adapter(_tiny_gemma3(), num_layers=2, lora_rank=4)
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids, dtype=torch.bool)

    logits = adapter.caption_logits(torch.randn(1, 2, 32), ids, mask)

    assert logits.shape == (1, 5, 64)


def test_gemma3_adapter_honors_local_sliding_window(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_identity_peft(monkeypatch)
    model = _tiny_gemma3(sliding_window=4)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    with torch.no_grad():
        expected = model.model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states[1]
        actual = Gemma3Adapter(model, num_layers=1, lora_rank=4).encode_text(ids, mask)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_gemma3_adapter_rejects_non_gemma_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_identity_peft(monkeypatch)
    transformers = pytest.importorskip("transformers")
    model = transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    )

    with pytest.raises(ValueError, match="Gemma 3"):
        Gemma3Adapter(model, num_layers=2, lora_rank=4)


class _TokenizerStub:
    pad_token_id = None
    pad_token = None
    eos_token = "<eos>"

    def __call__(self, _text: str, **_kwargs):
        return {"input_ids": torch.tensor([[7, 8, 9]])}


def _experiment(backend: str) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            radar_dim=16,
            fusion_heads=4,
            llm_dim=640,
            llm_layers=4,
            llm_heads=4,
            cross_attn_dim=32,
            cross_attn_heads=4,
        ),
        data={},
        training={},
        llm={"backend": backend, "name_or_path": "google/gemma-3-270m-it"},
        stage={},
    )


def test_gemma3_backend_builds_adapter_tokenizer_and_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = pytest.importorskip("transformers")
    expected_adapter = TinyLLMAdapter(64, 640, 4, 4)
    tokenizer = _TokenizerStub()
    monkeypatch.setattr(Gemma3Adapter, "from_pretrained", lambda *_args, **_kwargs: expected_adapter)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: tokenizer)

    adapter, actual_tokenizer, prompt_ids = _build_llm_backend(_experiment("gemma3"))

    assert adapter is expected_adapter
    assert actual_tokenizer is tokenizer
    assert tokenizer.pad_token == tokenizer.eos_token
    assert prompt_ids.tolist() == [7, 8, 9]


def test_unknown_llm_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported LLM backend"):
        _build_llm_backend(_experiment("gemam3"))


def test_tiny_backend_remains_available() -> None:
    config = _experiment("tiny")
    config.llm.update({"vocab_size": 64, "prompt_ids": [1, 2, 3]})

    adapter, tokenizer, prompt_ids = _build_llm_backend(config)

    assert isinstance(adapter, TinyLLMAdapter)
    assert isinstance(tokenizer, TinyTokenizer)
    assert prompt_ids.tolist() == [1, 2, 3]


def test_qwen_backend_defaults_to_0_5b_model(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = pytest.importorskip("transformers")
    expected_adapter = TinyLLMAdapter(64, 640, 4, 4)
    tokenizer = _TokenizerStub()
    loaded_names: list[str] = []

    def load_adapter(name: str, *_args, **_kwargs):
        loaded_names.append(name)
        return expected_adapter

    config = _experiment("qwen2")
    config.llm.pop("name_or_path")
    monkeypatch.setattr(Qwen2Adapter, "from_pretrained", load_adapter)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: tokenizer)

    adapter, actual_tokenizer, _prompt_ids = _build_llm_backend(config)

    assert adapter is expected_adapter
    assert actual_tokenizer is tokenizer
    assert loaded_names == ["Qwen/Qwen2.5-0.5B-Instruct"]


def test_qwen_backend_resolves_configured_float16_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    transformers = pytest.importorskip("transformers")
    expected_adapter = TinyLLMAdapter(64, 640, 4, 4)
    tokenizer = _TokenizerStub()
    captured: dict[str, object] = {}

    def load_adapter(_name: str, *_args, **kwargs):
        captured.update(kwargs)
        return expected_adapter

    config = _experiment("qwen2")
    config.llm["torch_dtype"] = "float16"
    monkeypatch.setattr(Qwen2Adapter, "from_pretrained", load_adapter)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: tokenizer)

    _build_llm_backend(config)

    assert captured["torch_dtype"] is torch.float16

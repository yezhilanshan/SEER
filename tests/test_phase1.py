from __future__ import annotations

import math

import pytest
import torch

from continuous_mmwave.checkpoint import load_stage1_radar_state, save_stage1_radar_export
from continuous_mmwave.config import FIXED_EVENT_PROMPT, ModelConfig
from continuous_mmwave.data import RadarBatch, RadarSample, radar_collate
from continuous_mmwave.models.event_former import (
    BottleneckRadarCrossAttention,
    EventStateGate,
    Qwen2Adapter,
    _cloneable_cache_prefix,
    StreamingSemanticEventFormer,
    TinyLLMAdapter,
    build_prefix_event_mask,
)
from continuous_mmwave.models.pose import mpjpe_loss, velocity_loss
from continuous_mmwave.models.radar_encoder import StatefulCausalTCN, StreamingRadarEncoder
from continuous_mmwave.models.semantic import SemanticProjection
from continuous_mmwave.stages.stage1 import Stage1Module
from continuous_mmwave.stages.stage2 import Stage2Loss, Stage2Module, symmetric_alignment_loss
from continuous_mmwave.train import load_experiment_config, train_stage1_batch, train_stage2_batch


def cfg(**overrides) -> ModelConfig:
    values = {
        "num_joints": 5,
        "radar_dim": 16,
        "fusion_heads": 4,
        "llm_dim": 32,
        "llm_layers": 4,
        "llm_heads": 4,
        "cross_attn_dim": 8,
        "cross_attn_heads": 2,
        "semantic_dim": 8,
        "max_commit_radar_tokens": 16,
        "dropout": 0.0,
    }
    values.update(overrides)
    return ModelConfig(**values)


def test_configured_cross_attention_scale_opens_radar_gradient() -> None:
    config = cfg(cross_attention_init_scale=1e-3)
    module = StreamingSemanticEventFormer(
        TinyLLMAdapter(64, 32, 4, 4), config, torch.tensor([1, 2, 3])
    )
    radar = torch.randn(2, 4, config.radar_dim, requires_grad=True)

    result = module.stream_update(module.initial_state(2), radar, 0.5)
    result.event_tokens.square().mean().backward()

    assert radar.grad is not None
    assert torch.isfinite(radar.grad).all()
    assert torch.count_nonzero(radar.grad) > 0
    assert torch.tanh(module.cross_attention[0].alpha).item() == pytest.approx(1e-3)


def test_state_gate_is_a_convex_update_with_safe_initial_fraction() -> None:
    gate = EventStateGate(8, initial_update=0.1)
    previous = torch.randn(2, 3, 8)
    proposal = torch.randn(2, 3, 8)

    state, values = gate(previous, proposal, 0.5)

    assert torch.all((0 <= values) & (values <= 1))
    torch.testing.assert_close(values, torch.full_like(values, 0.1))
    torch.testing.assert_close(state, torch.lerp(previous, proposal, values))


def test_parrad_stage2_config_keeps_radar_path_open() -> None:
    config = load_experiment_config("configs/parrad-md-gru-stage2.yaml")

    assert config.model.cross_attention_init_scale > 0


def batch(skeleton: bool = False) -> RadarBatch:
    return RadarBatch(
        torch.randn(2, 4, 2, 64, 64),
        torch.randn(2, 4, 6, 6),
        torch.ones(2, 4, 6, dtype=torch.bool),
        torch.randn(2, 4, 5, 3) if skeleton else None,
        ("person sits", "person stands"),
        ("a", "b"),
        torch.ones(2, 4, dtype=torch.bool),
    )


def stage2() -> Stage2Module:
    config = cfg()
    adapter = TinyLLMAdapter(64, 32, 4, 4)
    return Stage2Module(
        StreamingRadarEncoder(config),
        StreamingSemanticEventFormer(adapter, config, torch.tensor([1, 2, 3])),
        SemanticProjection(32, 8),
        contrastive_temperature=0.1,
    )


def test_qwen_0_5b_defaults() -> None:
    config = ModelConfig()
    assert config.frames_per_update == 4
    assert config.radar_tokens_per_update == 4
    assert (config.radar_dim, config.llm_dim, config.llm_heads, config.event_tokens) == (256, 896, 14, 8)


def test_stage2_config_targets_qwen_0_5b() -> None:
    config = load_experiment_config("configs/stage2.yaml")

    assert (config.model.llm_dim, config.model.llm_heads) == (896, 14)
    assert config.llm["name_or_path"] == "/home/xibeidaxue/SEER/Qwen2.5-0.5B-Instruct"
    assert config.llm["torch_dtype"] == "float16"


def test_collate_variable_time_and_points() -> None:
    result = radar_collate(
        [
            RadarSample(torch.randn(3, 2, 64, 64), torch.randn(3, 2, 6)),
            RadarSample(torch.randn(8, 2, 64, 64), torch.randn(8, 5, 6)),
        ]
    )
    assert result.rd.shape == (2, 8, 2, 64, 64)
    assert result.points.shape == (2, 8, 5, 6)
    assert result.frame_mask[0].tolist() == [True, True, True, False, False, False, False, False]


def test_tcn_chunked_equals_full_sequence() -> None:
    torch.manual_seed(1)
    model = StatefulCausalTCN(16, 3, (1, 2, 4, 8), 0).eval()
    sequence = torch.randn(2, 12, 16)
    full, _ = model(sequence)
    first, state = model(sequence[:, :4])
    second, state = model(sequence[:, 4:8], state)
    third, _ = model(sequence[:, 8:], state)
    torch.testing.assert_close(torch.cat((first, second, third), 1), full)


def test_encoder_four_tokens_and_large_update_index() -> None:
    encoder = StreamingRadarEncoder(cfg()).eval()
    item = batch()
    output = encoder.forward_update(item.rd, item.points, item.point_mask)
    assert output.radar_tokens.shape == (2, 4, 16)
    state = type(output.state)(output.state.rd_tcn, output.state.point_tcn, 100_000)
    later = encoder.forward_update(item.rd, item.points, item.point_mask, state)
    assert torch.isfinite(later.radar_tokens).all()


def test_pose_losses_ignore_padding() -> None:
    prediction = torch.zeros(1, 2, 1, 3)
    target = prediction.clone()
    target[:, 1] = 99
    mask = torch.tensor([[True, False]])
    assert mpjpe_loss(prediction, target, mask) == 0
    assert velocity_loss(prediction, target, mask) == 0


def test_stage1_optimizer_step_and_pose_free_export(tmp_path) -> None:
    module = Stage1Module(StreamingRadarEncoder(cfg()), 5)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    metrics = train_stage1_batch(module, batch(True), optimizer, 1)
    assert all(math.isfinite(value) for value in metrics.values())
    path = tmp_path / "radar.pt"
    save_stage1_radar_export(path, module, cfg().to_dict())
    payload = load_stage1_radar_state(path, StreamingRadarEncoder(cfg()))
    assert all("pose_head" not in name for name in payload["radar_encoder_state"])


def test_prefix_mask_prompt_causal_events_bidirectional() -> None:
    allowed = build_prefix_event_mask(2, 3, torch.float32, torch.device("cpu"))[0, 0] == 0
    assert allowed.tolist() == [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [True, True, True, True, True],
        [True, True, True, True, True],
        [True, True, True, True, True],
    ]


def test_cross_attention_zero_gate_is_identity_and_mask_safe() -> None:
    cross = BottleneckRadarCrossAttention(32, 16, 8, 2, 0)
    events, radar = torch.randn(2, 8, 32), torch.randn(2, 4, 16)
    torch.testing.assert_close(cross(events, radar), events)
    cross.alpha.data.fill_(1)
    output = cross(events, radar, torch.tensor([[True] * 4, [False] * 4]))
    assert torch.isfinite(output).all()


def test_event_former_stream_and_commit_contracts() -> None:
    config = cfg()
    former = StreamingSemanticEventFormer(
        TinyLLMAdapter(64, 32, 4, 4), config, torch.tensor([1, 2, 3]), FIXED_EVENT_PROMPT
    )
    stream = former.stream_update(former.initial_state(2), torch.randn(2, 4, 16), 0.5)
    commit = former.commit_refine(torch.randn(2, 33, 16))
    assert stream.event_tokens.shape == commit.event_tokens.shape == (2, 8, 32)
    assert former.prepare_commit_radar_tokens(torch.zeros(2, 33, 16)).shape == (2, 16, 16)


def test_event_former_does_not_accumulate_transformer_identity_residual() -> None:
    class ConstantResidualAdapter(TinyLLMAdapter):
        def self_attention(self, layer, hidden, mask, context=None):
            return hidden + torch.ones_like(hidden)

        def feed_forward(self, layer, hidden):
            return hidden + torch.ones_like(hidden)

    config = cfg(llm_layers=1, cross_attention_init_scale=0.0, event_state_initial_update=0.1)
    former = StreamingSemanticEventFormer(
        ConstantResidualAdapter(64, 32, 1, 4), config, torch.tensor([1, 2, 3]), FIXED_EVENT_PROMPT
    )
    state = former.initial_state(1)
    radar = torch.zeros(1, 4, config.radar_dim)
    norms = []

    for _ in range(80):
        output = former.stream_update(state, radar, 0.5)
        state = output.event_tokens
        norms.append(output.state_norm.item())

    # The adapter returns ``previous + constant``.  The outer recurrent gate
    # must blend toward that update content, not add the identity residual on
    # every call.  A double residual grows linearly and exceeds this bound.
    assert norms[-1] < 40.0
    assert norms[-1] - norms[-2] < 0.01


def test_semantic_projection_shared_normalized_space() -> None:
    projection = SemanticProjection(32, 8)
    keys = projection(torch.randn(3, 8, 32))
    torch.testing.assert_close(torch.linalg.vector_norm(keys, dim=-1), torch.ones(3))


def test_infonce_requires_negatives_and_rewards_matches() -> None:
    with pytest.raises(ValueError, match="at least two"):
        symmetric_alignment_loss(torch.ones(1, 2), torch.ones(1, 2), 0.1)
    keys = torch.eye(2)
    assert symmetric_alignment_loss(keys, keys, 0.1) < symmetric_alignment_loss(keys, keys.flip(0), 0.1)


def test_stage2_has_exact_three_losses() -> None:
    losses = Stage2Loss(torch.tensor(2.0), torch.tensor(3.0), torch.tensor(5.0))
    assert losses.total(0.1, 0.2).item() == pytest.approx(3.3)


def test_stage2_forward_backward_updates_radar_and_projection() -> None:
    module = stage2()
    optimizer = torch.optim.AdamW((p for p in module.parameters() if p.requires_grad), lr=1e-3)
    radar_before = module.radar_encoder.rd_pool.queries.detach().clone()
    metrics = {}
    for _ in range(2):
        metrics = train_stage2_batch(
            module,
            batch(),
            torch.tensor([[4, 5, 6], [7, 8, 9]]),
            torch.ones(2, 3, dtype=torch.bool),
            optimizer,
            1,
        )
    assert all(math.isfinite(value) for value in metrics.values())
    assert not torch.equal(radar_before, module.radar_encoder.rd_pool.queries)


def test_variable_duration_stage2_ignores_padded_update() -> None:
    variable = radar_collate(
        [
            RadarSample(torch.randn(4, 2, 64, 64), torch.randn(4, 3, 6), caption="a"),
            RadarSample(torch.randn(8, 2, 64, 64), torch.randn(8, 3, 6), caption="b"),
        ]
    )
    output = stage2().training_step(variable, torch.tensor([[4, 5], [6, 7]]), torch.ones(2, 2, dtype=torch.bool))
    torch.testing.assert_close(output.stream_state_norms[0, 0], output.stream_state_norms[0, 1])


def test_qwen_front_four_layers_qv_lora_and_bfloat16() -> None:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    qwen_config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
    )
    adapter = Qwen2Adapter(transformers.Qwen2ForCausalLM(qwen_config).to(torch.bfloat16), 4, 4)
    names = [name for name, parameter in adapter.named_parameters() if parameter.requires_grad]
    assert names and all("lora_" in name and ("q_proj" in name or "v_proj" in name) for name in names)
    assert all(parameter.dtype == torch.float32 for parameter in adapter.parameters() if parameter.requires_grad)
    former = StreamingSemanticEventFormer(adapter, cfg(), torch.tensor([1, 2, 3]))
    assert all(parameter.dtype == torch.float32 for parameter in former.parameters() if parameter.requires_grad)
    cache = former.build_prompt_cache(2)
    output = former.stream_update(former.initial_state(2), torch.randn(2, 4, 16), 0.5, prompt_cache=cache)
    assert output.event_tokens.dtype == torch.bfloat16


def test_qwen_cache_clone_ignores_uninitialized_trailing_layers() -> None:
    key = torch.randn(2, 2, 3, 4)
    value = torch.randn(2, 2, 3, 4)
    cache = [(key, value, None), (key + 1, value + 1, None), (None, None, None)]

    cloned = _cloneable_cache_prefix(cache, layer_count=2)

    assert len(cloned) == 2
    assert all(layer[0] is not None and layer[1] is not None for layer in cloned)
    assert cloned[0][0] is not key
    torch.testing.assert_close(cloned[0][0], key)

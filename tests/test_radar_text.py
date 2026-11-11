from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from continuous_mmwave.radar_text.config import RadarTextConfig
from continuous_mmwave.radar_text.data import OfflineCaptionDataset, caption_from_actions, collapse_actions
from continuous_mmwave.radar_text.metrics import score_caption
from continuous_mmwave.radar_text.model import RadarEncoder, RadarTextModel, md_time_patches


@pytest.fixture(autouse=True)
def small_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_targets_preserve_repeated_actions_and_reject_unknown_gaps():
    assert collapse_actions([0, 0, 1, 1, 0]) == [0, 1, 0]
    with pytest.raises(ValueError, match="unknown"):
        collapse_actions([0, -100, 1])
    assert caption_from_actions([0, 1, 0], {0: "walks", 1: "sits down"}) == (
        "The person walks, then sits down, and then walks."
    )


def test_md_patches_are_local_and_ignore_padding():
    md = torch.arange(10.0).reshape(2, 5, 1)
    patches = md_time_patches(md, torch.tensor([3, 5]), 3)
    assert patches.shape == (2, 5, 1, 1, 3)
    assert patches[0, 0, 0, 0].tolist() == [0, 0, 1]
    assert patches[0, 2, 0, 0].tolist() == [1, 2, 2]


def test_bidirectional_encoder_is_invariant_to_batch_padding():
    torch.manual_seed(3)
    encoder = RadarEncoder(RadarTextConfig(dropout=0)).eval()
    rd, md = torch.randn(2, 7, 1, 16, 16), torch.randn(2, 7, 16)
    lengths = torch.tensor([4, 7])
    with torch.no_grad():
        tokens = encoder(rd, md, lengths)[1]
        alone = encoder(rd[:1, :4], md[:1, :4], lengths[:1])[1]
        rd[0, 4:], md[0, 4:] = 1e5, -1e5
        changed = encoder(rd, md, lengths)[1]
    assert tokens.shape == (2, 8, 256)
    torch.testing.assert_close(tokens[:1], alone, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(tokens, changed)


def test_caption_dataset_uses_seconds_and_skips_unlabelled_windows(tmp_path):
    rd = np.zeros((20, 1, 93, 128), dtype=np.float32)
    np.savez(tmp_path / "x.npz", rd=rd, md=np.zeros((20, 128), dtype=np.float32),
             labels=np.array([0] * 5 + [1] * 5 + [-100] * 10), timestamps=np.arange(20) * 0.5)
    row = dict(path="x.npz", subject_id="s1", session_id="a", sensor_id="60", environment_id="lab")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row), encoding="utf-8")
    dataset = OfflineCaptionDataset(manifest, {0: "walks", 1: "sits down"},
                                    window_seconds=5, stride_seconds=5, min_seconds=4)
    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["rd"].shape[0] == 10
    assert sample["actions"] == [0, 1]
    assert sample["caption"] == "The person walks, then sits down."


def tiny_model():
    from continuous_mmwave.radar_text.runner import make_tiny_model
    return make_tiny_model(RadarTextConfig(dropout=0))


def test_frozen_qwen_receives_radar_gradients_and_only_scores_answers():
    model = tiny_model()
    rd, md, lengths = torch.randn(2, 5, 1, 16, 16), torch.randn(2, 5, 16), torch.tensor([3, 5])
    captions = ["The person walks.", "The person sits down, then walks."]
    packed = model.prepare_inputs(rd, md, lengths, captions)
    assert packed["inputs_embeds"].shape[:2] == packed["labels"].shape
    assert torch.all(packed["labels"][packed["attention_mask"] == 0] == -100)
    assert (packed["labels"] != -100).sum().item() == sum(len(model.answer_ids(c)) for c in captions)
    loss = model(rd, md, lengths, captions).loss
    loss.backward()
    assert torch.isfinite(loss)
    for branch in (model.encoder.rd_cnn, model.encoder.md_cnn, model.encoder.gru, model.projector):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.llm.parameters())
    model.eval()
    assert len(model.generate(rd[:1], md[:1], lengths[:1], max_new_tokens=3)) == 1


def test_checkpoint_roundtrip_and_architecture_mismatch(tmp_path):
    from continuous_mmwave.radar_text.runner import save_checkpoint, load_weights
    model = tiny_model().eval()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, epoch=1, stage="align", settings={"tiny": True})
    restored = tiny_model().eval()
    load_weights(path, restored)
    rd, md, lengths = torch.randn(1, 4, 1, 16, 16), torch.randn(1, 4, 16), torch.tensor([4])
    torch.testing.assert_close(model(rd, md, lengths, ["walks"]).logits,
                               restored(rd, md, lengths, ["walks"]).logits)
    incompatible = tiny_model()
    incompatible.config = RadarTextConfig(tokens=4)
    with pytest.raises(ValueError, match="config"):
        load_weights(path, incompatible)


def test_order_metrics_penalize_swaps_and_extra_actions():
    phrases = {0: "walks", 1: "sits down", 2: "lies down"}
    correct = score_caption("The person walks, then sits down.", [0, 1], phrases)
    swapped = score_caption("The person sits down, then walks.", [0, 1], phrases)
    extra = score_caption("The person walks, sits down, then lies down.", [0, 1], phrases)
    assert correct["sequence_exact"] == 1
    assert swapped["sequence_exact"] == 0 and swapped["order_lcs"] < 1
    assert extra["action_precision"] < 1
    assert score_caption("unrelated text", [0], phrases)["action_recall"] == 0


def test_pretraining_subjects_survive_warm_start_and_reject_holdout(tmp_path):
    from continuous_mmwave.radar_text.model import RadarPretrainer
    from continuous_mmwave.radar_text.runner import save_checkpoint, load_weights, check_holdout_subjects
    first = RadarPretrainer(RadarTextConfig(dropout=0), 3)
    first.training_subjects = {"s1", "s2"}
    first.selection_subjects = {"s9"}
    path = tmp_path / "stage1.pt"
    save_checkpoint(path, first, epoch=1, stage="stage1", settings={})
    second = RadarPretrainer(RadarTextConfig(dropout=0), 3)
    load_weights(path, second)
    second.training_subjects.add("s3")
    final = tmp_path / "second.pt"
    save_checkpoint(final, second, epoch=1, stage="stage1", settings={})
    saved = torch.load(final, weights_only=True)
    assert saved["training_subjects"] == ["s1", "s2", "s3"]
    assert saved["selection_subjects"] == ["s9"]
    with pytest.raises(ValueError, match="overlap"):
        check_holdout_subjects(saved, [{"subject_id": "s1"}])
    check_holdout_subjects(saved, [{"subject_id": "s4"}])
    check_holdout_subjects(saved, [{"subject_id": "s9"}])
    with pytest.raises(ValueError, match="overlap"):
        check_holdout_subjects(saved, [{"subject_id": "s9"}], test=True)


def test_infer_accepts_npz_without_labels_or_timestamps(tmp_path):
    from continuous_mmwave.radar_text.runner import infer, save_checkpoint
    path = tmp_path / "caption.pt"
    save_checkpoint(path, tiny_model(), epoch=1, stage="align", settings={"tiny": True})
    np.savez(tmp_path / "input.npz", rd=np.zeros((3, 1, 16, 16), dtype="float32"),
             md=np.zeros((3, 16), dtype="float32"))
    assert isinstance(infer(path, tmp_path / "input.npz", max_new_tokens=2), str)


def test_generation_advances_rope_positions():
    model = tiny_model().eval()
    positions = []
    def record(module, args, kwargs):
        positions.append(kwargs["position_ids"].detach().clone())
    hook = model.llm.model.register_forward_pre_hook(record, with_kwargs=True)
    try:
        model.generate(torch.randn(1, 4, 1, 16, 16), torch.randn(1, 4, 16), torch.tensor([4]), max_new_tokens=4)
    finally:
        hook.remove()
    assert len(positions) >= 2
    for previous, current in zip(positions, positions[1:]):
        assert current[0, -1] == previous[0, -1] + 1


def test_multiple_recordings_in_a_session_have_unique_caption_ids(tmp_path):
    rows = []
    for name in ("first.npz", "second.npz"):
        np.savez(tmp_path / name, rd=np.zeros((10, 1, 93, 128), dtype="float32"),
                 md=np.zeros((10, 128), dtype="float32"), labels=np.zeros(10, dtype="int64"),
                 timestamps=np.arange(10) * 0.5)
        rows.append(dict(path=name, subject_id="s1", session_id="a", sensor_id="60", environment_id="lab"))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    dataset = OfflineCaptionDataset(manifest, {0: "walks"}, window_seconds=5, stride_seconds=5, min_seconds=4)
    assert dataset[0]["sample_id"] != dataset[1]["sample_id"]


def test_training_sampler_keeps_recordings_together_without_losing_windows():
    from types import SimpleNamespace
    from continuous_mmwave.radar_text.data import RecordingWindowSampler
    dataset = SimpleNamespace(windows=[(i // 3, 0, 1, str(i), [0]) for i in range(12)])
    sampler = RecordingWindowSampler(dataset)
    order = list(sampler)
    assert sorted(order) == list(range(12))
    groups = [dataset.windows[index][0] for index in order]
    assert len(collapse_actions(groups)) == 4


def test_lora_gradient_and_adapter_checkpoint(tmp_path):
    pytest.importorskip("peft")
    from continuous_mmwave.radar_text.runner import load_weights, save_checkpoint
    base = tiny_model()
    model = RadarTextModel(base.config, base.llm, base.tokenizer, lora_rank=2, lora_alpha=4, lora_dropout=0)
    rd, md, lengths = torch.randn(1, 4, 1, 16, 16), torch.randn(1, 4, 16), torch.tensor([4])
    model(rd, md, lengths, ["The person walks."]).loss.backward()
    assert any("lora_" in name and p.grad is not None and p.grad.abs().sum() > 0
               for name, p in model.llm.named_parameters())
    assert all("lora_" in name for name, p in model.llm.named_parameters() if p.requires_grad)
    path = tmp_path / "lora.pt"
    save_checkpoint(path, model, epoch=1, stage="lora", settings={})
    base = tiny_model()
    restored = RadarTextModel(base.config, base.llm, base.tokenizer,
                              lora_rank=2, lora_alpha=4, lora_dropout=0)
    load_weights(path, restored)
    model.eval()
    restored.eval()
    torch.testing.assert_close(model(rd, md, lengths, ["walks"]).logits,
                               restored(rd, md, lengths, ["walks"]).logits)

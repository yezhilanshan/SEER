from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from scripts.make_synthetic_parrad_data import generate


class FakeGradScaler:
    def __init__(self, scale_before: float, scale_after: float, *, skip: bool) -> None:
        self.current_scale = scale_before
        self.scale_after = scale_after
        self.skip = skip

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.skip:
            optimizer.step()

    def update(self) -> None:
        self.current_scale = self.scale_after

    def get_scale(self) -> float:
        return self.current_scale


@pytest.mark.parametrize(
    ("scale_after", "skip", "expected_step"),
    [(32768.0, True, False), (65536.0, False, True)],
)
def test_amp_step_reports_whether_optimizer_ran(
    scale_after: float, skip: bool, expected_step: bool
) -> None:
    from continuous_mmwave.train import _step

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    before = parameter.detach().clone()
    scaler = FakeGradScaler(65536.0, scale_after, skip=skip)

    stepped = _step(parameter.square(), optimizer, [parameter], 1.0, scaler)

    assert stepped is expected_step
    assert bool(parameter.detach() != before) is expected_step


@pytest.mark.parametrize(
    ("epoch", "warmup_epochs", "expected_phase", "expected_text_weight"),
    [
        (0, 3, "semantic_warmup", 0.0),
        (2, 3, "semantic_warmup", 0.0),
        (3, 3, "joint", 0.05),
        (4, 3, "joint", 0.1),
        (8, 3, "joint", 0.1),
    ],
)
def test_stage2_phase_and_caption_weight_schedule(
    epoch: int,
    warmup_epochs: int,
    expected_phase: str,
    expected_text_weight: float,
) -> None:
    from continuous_mmwave.train import _parrad_stage2_phase, _parrad_stage2_text_weight

    assert _parrad_stage2_phase(epoch, warmup_epochs) == expected_phase
    assert _parrad_stage2_text_weight(epoch, warmup_epochs, 2, 0.1) == pytest.approx(
        expected_text_weight
    )


def test_stage2_warmup_validation_prefers_action_macro_f1() -> None:
    from continuous_mmwave.train import _validation_score

    metrics = {
        "phase": "semantic_warmup",
        "auxiliary_macro_f1": 0.4,
        "auxiliary_accuracy": 0.5,
        "auxiliary_classification_loss": 1.2,
        "loss": 9.0,
        "stream_commit_cosine": 0.0,
        "consistency": 1.0,
    }

    assert _validation_score("stage2", True, metrics) == pytest.approx((0.4, 0.5, -1.2))


def _write_config(
    path: Path,
    manifest: Path,
    output: Path,
    *,
    validation_manifest: Path | None = None,
    test_manifest: Path | None = None,
    class_weights: str | None = None,
    label_boundary_ignore_seconds: float | None = None,
    window_frames: int = 2,
    train_window_mode: str | None = None,
    streaming_finetune_epochs: int | None = None,
    streaming_sequence_frames: int | None = None,
    streaming_chunk_frames: int | None = None,
    frame_classification_weight: float | None = None,
    token_classification_weight: float | None = None,
) -> Path:
    payload = {
        "model": {
            "frame_period_s": 0.09,
            "frames_per_update": 2,
            "rd_channels": 1,
            "rd_height": 93,
            "rd_width": 128,
            "md_bins": 128,
            "radar_dim": 16,
            "motion_dim": 8,
            "radar_tokens_per_update": 4,
            "sensor_vocab_size": 2,
            "use_sensor_embedding": True,
            "fusion_heads": 4,
            "rd_tcn_dilations": [1, 2],
            "md_tcn_dilations": [1, 2],
            "dropout": 0.0,
        },
        "data": {
            "train_manifest": str(manifest),
            "window_frames": window_frames,
            "window_stride": window_frames,
        },
        "training": {
            "seed": 7,
            "device": "cpu",
            "output_dir": str(output),
            "batch_size": 2,
            "num_workers": 0,
            "epochs": 1,
            "learning_rate": 0.001,
            "streaming_learning_rate": 0.0003,
            "warmup_ratio": 0.05,
            "min_lr_ratio": 0.1,
            "early_stopping_patience": 4,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "amp": False,
        },
        "llm": {},
        "stage": {"variant": "parrad", "num_classes": 3},
    }
    if validation_manifest is not None:
        payload["data"]["validation_manifest"] = str(validation_manifest)
    if test_manifest is not None:
        payload["data"]["test_manifest"] = str(test_manifest)
    if class_weights is not None:
        payload["stage"]["class_weights"] = class_weights
    if label_boundary_ignore_seconds is not None:
        payload["data"]["label_boundary_ignore_seconds"] = label_boundary_ignore_seconds
    if train_window_mode is not None:
        payload["data"]["train_window_mode"] = train_window_mode
    if streaming_sequence_frames is not None:
        payload["data"]["streaming_sequence_frames"] = streaming_sequence_frames
    if streaming_chunk_frames is not None:
        payload["data"]["streaming_chunk_frames"] = streaming_chunk_frames
    if streaming_finetune_epochs is not None:
        payload["training"]["streaming_finetune_epochs"] = streaming_finetune_epochs
    if frame_classification_weight is not None:
        payload["stage"]["frame_classification_weight"] = frame_classification_weight
    if token_classification_weight is not None:
        payload["stage"]["token_classification_weight"] = token_classification_weight
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_parrad_stage1_config_selects_parrad_model(tmp_path: Path) -> None:
    from continuous_mmwave.parrad_config import PARradConfig
    from continuous_mmwave.train import load_experiment_config

    manifest = generate(tmp_path / "data")
    config_path = _write_config(tmp_path / "config.yaml", manifest, tmp_path / "output")

    config = load_experiment_config(config_path)

    assert isinstance(config.model, PARradConfig)
    assert config.model.frames_per_update == 2


def test_parrad_stage1_training_writes_metrics_and_encoder_only_checkpoint(tmp_path: Path) -> None:
    from continuous_mmwave.train import run_training

    manifest = generate(tmp_path / "data")
    output = tmp_path / "output"
    config_path = _write_config(tmp_path / "config.yaml", manifest, output)

    result = run_training("stage1", config_path, max_steps=1)

    assert result == output
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(metrics) == 1
    assert {
        "epoch",
        "step",
        "loss",
        "classification_loss",
        "token_classification_loss",
        "temporal_smoothness_loss",
        "temporal_smoothness_weight",
        "frame_classification_weight",
        "token_classification_weight",
        "accuracy",
        "token_accuracy",
        "supervised_tokens",
        "learning_rate",
    } == set(metrics[0])
    assert 0.0 <= metrics[0]["accuracy"] <= 1.0
    checkpoint = torch.load(output / "radar_encoder.pt", map_location="cpu", weights_only=False)
    assert checkpoint["stage"] == "stage1-radar-export"
    assert checkpoint["model_config"]["frames_per_update"] == 2
    assert checkpoint["radar_encoder_state"]
    assert all("classifier" not in name for name in checkpoint["radar_encoder_state"])


def test_stage1_token_supervision_config_trains_and_reports_token_path(tmp_path: Path) -> None:
    from continuous_mmwave.train import run_training

    manifest = _write_evaluation_split(
        tmp_path / "data", "train-subject", np.array([0, 0, 1, 1], dtype=np.int64)
    )
    output = tmp_path / "output"
    config_path = _write_config(
        tmp_path / "config.yaml",
        manifest,
        output,
        frame_classification_weight=0.5,
        token_classification_weight=1.0,
    )

    run_training("stage1", config_path, max_steps=1)

    metrics = json.loads((output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert metrics["frame_classification_weight"] == 0.5
    assert metrics["token_classification_weight"] == 1.0
    assert metrics["supervised_tokens"] == 2.0
    assert np.isfinite(metrics["token_classification_loss"])
    assert 0.0 <= metrics["token_accuracy"] <= 1.0


def test_stage1_token_warmup_loads_export_and_freezes_backbone(tmp_path: Path) -> None:
    from continuous_mmwave.checkpoint import save_stage1_radar_export
    from continuous_mmwave.models.parrad_encoder import PARradCausalMDGRUEncoder
    from continuous_mmwave.stages.parrad_stage1 import PARradStage1Module
    from continuous_mmwave.train import _build_parrad_stage1, load_experiment_config

    manifest = generate(tmp_path / "data")
    config_path = _write_config(tmp_path / "config.yaml", manifest, tmp_path / "output")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["model"].update({"encoder_type": "causal_md_gru", "md_gru_dim": 16})
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_experiment_config(config_path)
    source = PARradStage1Module(PARradCausalMDGRUEncoder(config.model), num_classes=3)
    with torch.no_grad():
        source.encoder.gru.gru.weight_ih_l0.fill_(0.125)
        source.encoder.token_resampler.queries.fill_(0.25)
    checkpoint = tmp_path / "radar_encoder.pt"
    save_stage1_radar_export(checkpoint, source, config.model.to_dict())
    payload["stage"].update(
        {
            "initial_radar_checkpoint": str(checkpoint),
            "freeze_radar_backbone": True,
            "token_classification_weight": 1.0,
        }
    )
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    module = _build_parrad_stage1(load_experiment_config(config_path))

    torch.testing.assert_close(
        module.encoder.gru.gru.weight_ih_l0,
        torch.full_like(module.encoder.gru.gru.weight_ih_l0, 0.125),
    )
    torch.testing.assert_close(
        module.encoder.token_resampler.queries,
        torch.full_like(module.encoder.token_resampler.queries, 0.25),
    )
    assert not module.encoder.gru.gru.weight_ih_l0.requires_grad
    assert not module.encoder.frame_projection.weight.requires_grad
    assert module.encoder.token_resampler.queries.requires_grad
    assert module.encoder.token_projection.weight.requires_grad
    assert module.classifier.weight.requires_grad
    assert module.token_classifier.weight.requires_grad


def _write_evaluation_split(root: Path, subject: str, labels: np.ndarray | None = None) -> Path:
    import numpy as np

    root.mkdir(parents=True, exist_ok=True)
    recording = root / f"{subject}.npz"
    rng = np.random.default_rng(sum(map(ord, subject)))
    label_values = np.array([0, 1, 2, 0], dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64)
    frames = len(label_values)
    np.savez(
        recording,
        rd=rng.normal(size=(frames, 1, 93, 128)).astype(np.float32),
        md=rng.normal(size=(frames, 128)).astype(np.float32),
        labels=label_values,
        timestamps=np.arange(frames, dtype=np.float64) * 0.09,
    )
    manifest = root / f"{subject}.jsonl"
    row = {
        "path": recording.name,
        "subject_id": subject,
        "session_id": "set000",
        "sensor_id": "60ghz",
        "environment_id": "test",
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def test_parrad_stage1_validation_selects_best_and_evaluates_test_split(tmp_path: Path) -> None:
    from continuous_mmwave.train import run_training

    train = _write_evaluation_split(tmp_path / "data", "train-subject")
    validation = _write_evaluation_split(tmp_path / "data", "validation-subject")
    test = _write_evaluation_split(tmp_path / "data", "test-subject")
    output = tmp_path / "output"
    config_path = _write_config(
        tmp_path / "config.yaml",
        train,
        output,
        validation_manifest=validation,
        test_manifest=test,
        class_weights="balanced",
    )

    run_training("stage1", config_path)

    validation_rows = [
        json.loads(line) for line in (output / "validation_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(validation_rows) == 1
    assert {"epoch", "step", "loss", "accuracy", "frames"} <= set(validation_rows[0])
    assert validation_rows[0]["frames"] == 4.0
    assert (output / "best.pt").is_file()
    test_metrics = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
    assert test_metrics["checkpoint"] == "best.pt"
    assert test_metrics["frames"] == 4.0
    weights = json.loads((output / "class_weights.json").read_text(encoding="utf-8"))
    assert weights == [2 / 3, 4 / 3, 4 / 3]


def test_stage1_token_supervision_validation_reports_token_metrics(tmp_path: Path) -> None:
    from continuous_mmwave.train import run_training

    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    train = _write_evaluation_split(tmp_path / "data", "train-subject", labels)
    validation = _write_evaluation_split(tmp_path / "data", "validation-subject", labels)
    output = tmp_path / "output"
    config_path = _write_config(
        tmp_path / "config.yaml",
        train,
        output,
        validation_manifest=validation,
        class_weights="balanced",
        token_classification_weight=1.0,
    )

    run_training("stage1", config_path)

    metrics = json.loads(
        (output / "validation_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert metrics["token_classification_weight"] == 1.0
    assert metrics["supervised_tokens"] == 3.0
    assert np.isfinite(metrics["token_classification_loss"])
    assert 0.0 <= metrics["token_accuracy"] <= 1.0
    assert 0.0 <= metrics["token_macro_f1"] <= 1.0


def test_weighted_evaluation_loss_is_invariant_to_batch_size(tmp_path: Path) -> None:
    import numpy as np

    from continuous_mmwave.data_parrad import PARradManifestDataset, PARradWindowDataset, parrad_collate
    from continuous_mmwave.models.parrad_encoder import PARradRadarEncoder
    from continuous_mmwave.stages.parrad_stage1 import PARradStage1Module
    from continuous_mmwave.train import evaluate_parrad_stage1, load_experiment_config

    manifest = _write_evaluation_split(tmp_path / "data", "evaluation-subject")
    with np.load(tmp_path / "data" / "evaluation-subject.npz", allow_pickle=False) as arrays:
        rd, md, timestamps = arrays["rd"], arrays["md"], arrays["timestamps"]
    np.savez(
        tmp_path / "data" / "evaluation-subject.npz",
        rd=np.concatenate((rd, rd[:2]), axis=0),
        md=np.concatenate((md, md[:2]), axis=0),
        labels=np.array([0, 0, 0, 1, 2, 2], dtype=np.int64),
        timestamps=np.arange(6, dtype=np.float64) * 0.09,
    )
    config_path = _write_config(tmp_path / "config.yaml", manifest, tmp_path / "output")
    config = load_experiment_config(config_path)
    recordings = PARradManifestDataset(manifest, config.model, {"60ghz": 0})
    windows = PARradWindowDataset(recordings, window_frames=2, window_stride=2)
    collate = partial(parrad_collate, frames_per_update=2)
    torch.manual_seed(5)
    module = PARradStage1Module(
        PARradRadarEncoder(config.model),
        num_classes=3,
        class_weights=torch.tensor([1.0, 2.0, 3.0]),
    )

    separate = evaluate_parrad_stage1(
        module, DataLoader(windows, batch_size=1, collate_fn=collate), torch.device("cpu")
    )
    combined = evaluate_parrad_stage1(
        module, DataLoader(windows, batch_size=3, collate_fn=collate), torch.device("cpu")
    )

    assert separate["loss"] == pytest.approx(combined["loss"], abs=1e-7)
    assert separate["accuracy"] == pytest.approx(combined["accuracy"], abs=1e-7)


def test_training_applies_configured_boundary_mask_to_evaluation_splits(tmp_path: Path) -> None:
    from continuous_mmwave.train import run_training

    labels = np.array([0, 0, 1, 1, 1, 1], dtype=np.int64)
    train = _write_evaluation_split(tmp_path / "data", "train-subject", labels)
    validation = _write_evaluation_split(tmp_path / "data", "validation-subject", labels)
    test = _write_evaluation_split(tmp_path / "data", "test-subject", labels)
    output = tmp_path / "output"
    config_path = _write_config(
        tmp_path / "config.yaml",
        train,
        output,
        validation_manifest=validation,
        test_manifest=test,
        label_boundary_ignore_seconds=0.09,
    )

    run_training("stage1", config_path)

    validation_row = json.loads((output / "validation_metrics.jsonl").read_text(encoding="utf-8"))
    test_metrics = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
    assert validation_row["frames"] == 4.0
    assert validation_row["raw_frames"] == 6.0
    assert test_metrics["frames"] == 4.0
    assert test_metrics["raw_frames"] == 6.0


def _tiny_stream_batch(labels: torch.Tensor):
    from continuous_mmwave.data_parrad import PARradBatch

    frames = labels.shape[1]
    return PARradBatch(
        rd=torch.randn(labels.shape[0], frames, 1, 16, 16),
        md=torch.randn(labels.shape[0], frames, 16),
        labels=labels,
        timestamps=torch.arange(frames, dtype=torch.float64).repeat(labels.shape[0], 1) * 0.09,
        frame_mask=torch.ones_like(labels, dtype=torch.bool),
        sensor_index=torch.zeros(labels.shape[0], dtype=torch.long),
        metadata=tuple(
            {"subject_id": f"s{index}", "session_id": "a", "sensor_id": "60ghz", "environment_id": "test"}
            for index in range(labels.shape[0])
        ),
    )


def _tiny_stage1(num_classes: int = 3, token_classification_weight: float = 0.0):
    from continuous_mmwave.models.parrad_encoder import PARradRadarEncoder
    from continuous_mmwave.parrad_config import PARradConfig
    from continuous_mmwave.stages.parrad_stage1 import PARradStage1Module

    config = PARradConfig(
        rd_height=16,
        rd_width=16,
        md_bins=16,
        radar_dim=16,
        motion_dim=8,
        fusion_heads=4,
        sensor_vocab_size=1,
        dropout=0.0,
    )
    return PARradStage1Module(
        PARradRadarEncoder(config),
        num_classes=num_classes,
        token_classification_weight=token_classification_weight,
    )


def test_streaming_batch_reports_token_supervision() -> None:
    from continuous_mmwave.train import train_parrad_streaming_batch

    module = _tiny_stage1(num_classes=2, token_classification_weight=1.0)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    batch = _tiny_stream_batch(torch.tensor([[0, 0, 1, 1]]))

    metrics = train_parrad_streaming_batch(
        module,
        batch,
        optimizer,
        clip=1.0,
        chunk_frames=2,
        device=torch.device("cpu"),
    )

    assert metrics["supervised_tokens"] == 2
    assert np.isfinite(metrics["token_classification_loss"])
    assert 0.0 <= metrics["token_accuracy"] <= 1.0


def test_streaming_batch_carries_ignored_chunks_without_optimizing_them() -> None:
    from continuous_mmwave.train import train_parrad_streaming_batch

    module = _tiny_stage1(num_classes=2)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    batch = _tiny_stream_batch(torch.tensor([[-100, -100, 0, 1]]))

    metrics = train_parrad_streaming_batch(
        module,
        batch,
        optimizer,
        clip=1.0,
        chunk_frames=2,
        device=torch.device("cpu"),
    )

    assert metrics["chunks"] == 2
    assert metrics["optimizer_steps"] == 1
    assert np.isfinite(metrics["loss"])
    assert np.isfinite(metrics["classification_loss"])
    assert np.isfinite(metrics["temporal_smoothness_loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_streaming_training_carries_detached_state_across_sequence_boundaries() -> None:
    from continuous_mmwave.train import train_parrad_streaming_batch

    module = _tiny_stage1(num_classes=2)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    first_batch = _tiny_stream_batch(torch.tensor([[0, 0, 1, 1]]))
    second_batch = _tiny_stream_batch(torch.tensor([[1, 1, 0, 0]]))

    _, first_state = train_parrad_streaming_batch(
        module,
        first_batch,
        optimizer,
        clip=1.0,
        chunk_frames=2,
        device=torch.device("cpu"),
        return_state=True,
    )
    _, second_state = train_parrad_streaming_batch(
        module,
        second_batch,
        optimizer,
        clip=1.0,
        chunk_frames=2,
        device=torch.device("cpu"),
        initial_state=first_state,
        return_state=True,
    )

    assert first_state.update_index == 2
    assert second_state.update_index == 4
    assert all(cache.grad_fn is None for cache in second_state.rd_tcn.caches)
    assert all(cache.grad_fn is None for cache in second_state.md_tcn.caches)


def test_recording_sequences_mark_only_recording_group_boundaries() -> None:
    from continuous_mmwave.data_parrad import PARradSample
    from continuous_mmwave.train import _iter_parrad_recording_sequences

    def sample(frames: int, subject: str) -> PARradSample:
        return PARradSample(
            rd=torch.randn(frames, 1, 16, 16),
            md=torch.randn(frames, 16),
            labels=torch.zeros(frames, dtype=torch.long),
            timestamps=torch.arange(frames, dtype=torch.float64) * 0.09,
            sensor_index=0,
            subject_id=subject,
            session_id="a",
            sensor_id="60ghz",
            environment_id="test",
        )

    batches = list(
        _iter_parrad_recording_sequences(
            (sample(6, "s1"), sample(3, "s2")),
            sequence_frames=4,
            recording_batch_size=1,
            boundary_ignore_frames=0,
            order=(0, 1),
        )
    )

    assert [new_group for new_group, _ in batches] == [True, False, True]
    assert [int(batch.frame_mask.sum()) for _, batch in batches] == [4, 2, 3]


def test_warmup_cosine_schedule_reaches_peak_then_decays_to_floor() -> None:
    from continuous_mmwave.train import WarmupCosineSchedule

    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1e-3)
    schedule = WarmupCosineSchedule(
        optimizer,
        base_lr=1e-3,
        total_steps=6,
        warmup_ratio=2 / 6,
        min_lr_ratio=0.1,
    )

    learning_rates = [schedule.step() for _ in range(6)]

    assert learning_rates[:2] == pytest.approx([5e-4, 1e-3])
    assert learning_rates[2:] == sorted(learning_rates[2:], reverse=True)
    assert learning_rates[-1] == pytest.approx(1e-4)


def test_phase_early_stopping_counts_non_improving_validations() -> None:
    from continuous_mmwave.train import PhaseEarlyStopping

    stopper = PhaseEarlyStopping(patience=2)

    assert stopper.update(accuracy=0.5, loss=1.0) == (True, False)
    assert stopper.update(accuracy=0.4, loss=0.9) == (False, False)
    assert stopper.update(accuracy=0.5, loss=1.0) == (False, True)
    assert stopper.update(accuracy=0.6, loss=1.2) == (True, False)


def test_phase_early_stopping_prefers_macro_f1_for_parrad() -> None:
    from continuous_mmwave.train import PhaseEarlyStopping

    stopper = PhaseEarlyStopping(patience=2)

    assert stopper.update(accuracy=0.8, loss=1.0, macro_f1=0.2) == (True, False)
    assert stopper.update(accuracy=0.9, loss=0.8, macro_f1=0.1) == (False, False)
    assert stopper.update(accuracy=0.7, loss=1.2, macro_f1=0.3) == (True, False)


def test_sequence_diagnostics_report_confusion_groups_and_boundary_quality() -> None:
    from continuous_mmwave.train import _sequence_diagnostics

    diagnostics = _sequence_diagnostics(
        predictions=(torch.tensor([0, 0, 1, 1, 2, 2]), torch.tensor([0, 1, 1, 2])),
        targets=(torch.tensor([0, 0, 1, 1, 1, 1]), torch.tensor([0, 0, 1, 2])),
        metadata=(
            {"subject_id": "s1", "session_id": "a", "sensor_id": "A"},
            {"subject_id": "s2", "session_id": "b", "sensor_id": "B"},
        ),
        num_classes=3,
        boundary_tolerance_frames=1,
    )

    assert diagnostics["confusion_matrix"] == [[3, 1, 0], [0, 3, 2], [0, 0, 1]]
    assert diagnostics["per_class_metrics"]["0"] == pytest.approx(
        {"precision": 1.0, "recall": 0.75, "f1": 6 / 7, "support": 4.0}
    )
    assert diagnostics["per_class_metrics"]["1"] == pytest.approx(
        {"precision": 0.75, "recall": 0.6, "f1": 2 / 3, "support": 5.0}
    )
    assert diagnostics["temporal"] == pytest.approx(
        {
            "true_boundaries": 3.0,
            "predicted_boundaries": 4.0,
            "matched_boundaries": 3.0,
            "boundary_precision": 0.75,
            "boundary_recall": 1.0,
            "boundary_f1": 6 / 7,
            "boundary_mae_frames": 1 / 3,
            "switch_excess_ratio": 4 / 3,
            "within_segment_switches": 2.0,
            "within_segment_switch_rate": 0.4,
        }
    )
    assert diagnostics["groups"]["subject"]["s1"]["accuracy"] == pytest.approx(2 / 3)
    assert diagnostics["groups"]["subject"]["s2"]["accuracy"] == pytest.approx(3 / 4)
    assert diagnostics["groups"]["sensor"]["A"]["frames"] == 6.0
    assert set(diagnostics["groups"]["recording"]) == {"s1/a/A", "s2/b/B"}


def test_sequence_diagnostics_detect_boundaries_separated_by_ignored_frames() -> None:
    from continuous_mmwave.train import _sequence_diagnostics

    diagnostics = _sequence_diagnostics(
        predictions=(torch.tensor([0, 0, 0, 1, 1, 1]),),
        targets=(torch.tensor([0, 0, -100, -100, 1, 1]),),
        metadata=({"subject_id": "s1", "session_id": "a", "sensor_id": "A"},),
        num_classes=2,
        boundary_tolerance_frames=1,
    )

    assert diagnostics["temporal"] == pytest.approx(
        {
            "true_boundaries": 1.0,
            "predicted_boundaries": 1.0,
            "matched_boundaries": 1.0,
            "boundary_precision": 1.0,
            "boundary_recall": 1.0,
            "boundary_f1": 1.0,
            "boundary_mae_frames": 1.0,
            "switch_excess_ratio": 1.0,
            "within_segment_switches": 0.0,
            "within_segment_switch_rate": 0.0,
        }
    )


def test_continuous_evaluation_is_chunk_size_invariant_and_counts_only_supervision() -> None:
    from continuous_mmwave.data_parrad import PARradSample
    from continuous_mmwave.train import evaluate_parrad_stage1_streaming

    torch.manual_seed(17)
    module = _tiny_stage1().eval()
    sample = PARradSample(
        rd=torch.randn(6, 1, 16, 16),
        md=torch.randn(6, 16),
        labels=torch.tensor([0, 0, 1, 1, 1, 1]),
        timestamps=torch.arange(6, dtype=torch.float64) * 0.09,
        sensor_index=0,
        subject_id="s1",
        session_id="a",
        sensor_id="60ghz",
        environment_id="test",
    )

    by_update = evaluate_parrad_stage1_streaming(
        module,
        (sample,),
        torch.device("cpu"),
        chunk_frames=2,
        recording_batch_size=1,
        boundary_ignore_frames=1,
    )
    by_two_updates = evaluate_parrad_stage1_streaming(
        module,
        (sample,),
        torch.device("cpu"),
        chunk_frames=4,
        recording_batch_size=1,
        boundary_ignore_frames=1,
    )

    assert by_update["frames"] == by_two_updates["frames"] == 4.0
    assert by_update["loss"] == pytest.approx(by_two_updates["loss"], abs=1e-6)
    assert by_update["accuracy"] == pytest.approx(by_two_updates["accuracy"], abs=1e-7)
    assert by_update["raw_frames"] == by_two_updates["raw_frames"] == 6.0
    assert by_update["segments"] == by_two_updates["segments"] == 2.0
    for key in (
        "macro_recall",
        "macro_f1",
        "raw_accuracy",
        "raw_macro_recall",
        "raw_macro_f1",
        "segment_accuracy",
    ):
        assert by_update[key] == pytest.approx(by_two_updates[key], abs=1e-7)
    assert set(by_update["per_class_recall"]) == {"0", "1", "2"}
    assert set(by_update["sensor_accuracy"]) == {"60ghz"}


def test_two_phase_training_uses_label_runs_then_streaming_and_evaluates_recordings(tmp_path: Path) -> None:
    from continuous_mmwave.parrad_diagnostics import run_parrad_diagnostics
    from continuous_mmwave.train import run_training

    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64)
    train = _write_evaluation_split(tmp_path / "data", "train-subject", labels)
    validation = _write_evaluation_split(tmp_path / "data", "validation-subject", labels)
    test = _write_evaluation_split(tmp_path / "data", "test-subject", labels)
    output = tmp_path / "output"
    config_path = _write_config(
        tmp_path / "config.yaml",
        train,
        output,
        validation_manifest=validation,
        test_manifest=test,
        label_boundary_ignore_seconds=0.09,
        window_frames=4,
        train_window_mode="label_run",
        streaming_finetune_epochs=1,
        streaming_sequence_frames=8,
        streaming_chunk_frames=4,
    )

    run_training("stage1", config_path)

    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    validation_rows = [
        json.loads(line) for line in (output / "validation_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    test_metrics = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
    assert len(metrics) == 3
    assert [row["phase"] for row in metrics] == ["label_run", "label_run", "streaming_finetune"]
    assert all(row["learning_rate"] > 0 for row in metrics)
    assert len(validation_rows) == 2
    assert [row["phase"] for row in validation_rows] == ["label_run", "streaming_finetune"]
    assert [row["frames"] for row in validation_rows] == [6.0, 6.0]
    assert [row["raw_frames"] for row in validation_rows] == [8.0, 8.0]
    assert test_metrics["frames"] == 6.0
    assert test_metrics["raw_frames"] == 8.0
    assert (output / "best_label_run.pt").exists()
    assert (output / "best_streaming.pt").exists()

    diagnostics_path = run_parrad_diagnostics(
        config_path,
        output / "best.pt",
        output / "diagnostics.json",
    )
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert set(diagnostics["splits"]) == {"train", "validation", "test"}
    assert diagnostics["splits"]["train"]["raw_frames"] == 8.0
    assert len(diagnostics["splits"]["test"]["confusion_matrix"]) == 3
    assert set(diagnostics["splits"]["test"]["groups"]["subject"]) == {"test-subject"}


def test_balanced_weights_count_only_frames_returned_by_label_run_windows(tmp_path: Path) -> None:
    from continuous_mmwave.data_parrad import PARradManifestDataset, PARradWindowDataset
    from continuous_mmwave.train import _balanced_class_weights, load_experiment_config

    labels = np.array([0] * 6 + [1] * 4 + [2] * 3, dtype=np.int64)
    manifest = _write_evaluation_split(tmp_path / "data", "train-subject", labels)
    config = load_experiment_config(
        _write_config(tmp_path / "config.yaml", manifest, tmp_path / "output", window_frames=4)
    )
    recordings = PARradManifestDataset(manifest, config.model, {"60ghz": 0})
    windows = PARradWindowDataset(
        recordings,
        window_frames=4,
        label_boundary_ignore_frames=1,
        window_mode="label_run",
    )

    _, weights = _balanced_class_weights(windows, num_classes=3)

    assert weights == pytest.approx([0.6, 1.5, 1.5])


def test_parrad_stage2_cli_loads_md_gru_export_and_runs_one_step(tmp_path: Path) -> None:
    from continuous_mmwave.train import run_training

    manifest = generate(tmp_path / "data")
    stage1_output = tmp_path / "stage1"
    stage1_config = _write_config(tmp_path / "stage1.yaml", manifest, stage1_output)
    payload = yaml.safe_load(stage1_config.read_text(encoding="utf-8"))
    payload["model"].update(
        {
            "encoder_type": "causal_md_gru",
            "md_gru_dim": 16,
            "event_tokens": 8,
            "llm_dim": 32,
            "llm_layers": 4,
            "llm_heads": 4,
            "cross_attn_dim": 8,
            "cross_attn_heads": 2,
            "semantic_dim": 8,
            "lora_rank": 4,
            "max_commit_radar_tokens": 32,
        }
    )
    stage1_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    run_training("stage1", stage1_config, max_steps=1)

    stage2_output = tmp_path / "stage2"
    payload["data"].update(
        {
            "train_window_mode": "label_run",
            "min_label_run_frames": 1,
        }
    )
    payload["training"].update(
        {
            "output_dir": str(stage2_output),
            "learning_rate": 0.001,
            "radar_learning_rate": 0.0001,
        }
    )
    payload["llm"] = {"backend": "tiny", "vocab_size": 64, "prompt_ids": [1, 2, 3]}
    payload["stage"] = {
        "variant": "parrad",
        "num_classes": 3,
        "stage1_checkpoint": str(stage1_output / "radar_encoder.pt"),
        "align_weight": 1.0,
        "consistency_weight": 1.0,
        "contrastive_temperature": 0.1,
        "checkpoint_streaming": False,
        "semantic_warmup_epochs": 1,
        "auxiliary_weight": 0.5,
        "text_weight_final": 0.1,
        "text_weight_ramp_epochs": 2,
    }
    stage2_config = tmp_path / "stage2.yaml"
    stage2_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = run_training("stage2", stage2_config, max_steps=1)

    assert result == stage2_output
    metrics = [json.loads(line) for line in (stage2_output / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 1
    assert {
        "epoch",
        "step",
        "loss",
        "text",
        "align",
        "consistency",
        "event_state_norm",
        "auxiliary_classification_loss",
        "auxiliary_accuracy",
        "optimizer_step_skipped",
        "phase",
    } == set(metrics[0])
    assert metrics[0]["optimizer_step_skipped"] == 0.0
    assert metrics[0]["text"] == 0.0
    assert metrics[0]["align"] == 0.0
    assert metrics[0]["phase"] == "semantic_warmup"
    trainable = json.loads((stage2_output / "trainable_parameters.json").read_text(encoding="utf-8"))
    assert all(not name.startswith("radar_encoder.") for name in trainable["trainable_names"])
    assert all(np.isfinite(value) for key, value in metrics[0].items() if key != "phase")
    checkpoint = torch.load(stage2_output / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["stage"] == "stage2"
    assert [group["lr"] for group in checkpoint["optimizer_state"]["param_groups"]] == pytest.approx(
        [0.0001, 0.001]
    )


def test_parrad_stage2_validation_selects_best_and_evaluates_test_split(tmp_path: Path) -> None:
    from continuous_mmwave.train import run_training

    train = _write_evaluation_split(
        tmp_path / "data", "train-subject", np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    )
    validation = _write_evaluation_split(
        tmp_path / "data", "validation-subject", np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    )
    test = _write_evaluation_split(
        tmp_path / "data", "test-subject", np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    )
    stage1_output = tmp_path / "stage1"
    stage1_config = _write_config(tmp_path / "stage1.yaml", train, stage1_output)
    payload = yaml.safe_load(stage1_config.read_text(encoding="utf-8"))
    payload["model"].update(
        {
            "encoder_type": "causal_md_gru",
            "md_gru_dim": 16,
            "event_tokens": 8,
            "llm_dim": 32,
            "llm_layers": 4,
            "llm_heads": 4,
            "cross_attn_dim": 8,
            "cross_attn_heads": 2,
            "semantic_dim": 8,
            "lora_rank": 4,
            "max_commit_radar_tokens": 32,
        }
    )
    stage1_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    run_training("stage1", stage1_config, max_steps=1)

    stage2_output = tmp_path / "stage2"
    payload["data"].update(
        {
            "validation_manifest": str(validation),
            "test_manifest": str(test),
            "train_window_mode": "label_run",
            "min_label_run_frames": 1,
        }
    )
    payload["training"].update(
        {
            "output_dir": str(stage2_output),
            "evaluation_batch_size": 2,
            "epochs": 2,
            "learning_rate": 0.001,
            "radar_learning_rate": 0.0001,
        }
    )
    payload["llm"] = {"backend": "tiny", "vocab_size": 64, "prompt_ids": [1, 2, 3]}
    payload["stage"] = {
        "variant": "parrad",
        "num_classes": 3,
        "stage1_checkpoint": str(stage1_output / "radar_encoder.pt"),
        "align_weight": 1.0,
        "consistency_weight": 1.0,
        "contrastive_temperature": 0.1,
        "checkpoint_streaming": False,
        "semantic_warmup_epochs": 1,
        "auxiliary_weight": 0.5,
        "text_weight_final": 0.1,
        "text_weight_ramp_epochs": 2,
    }
    stage2_config = tmp_path / "stage2.yaml"
    stage2_config.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    run_training("stage2", stage2_config)

    validation_rows = [
        json.loads(line)
        for line in (stage2_output / "validation_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    test_metrics = json.loads((stage2_output / "test_metrics.json").read_text(encoding="utf-8"))
    expected = {
        "loss",
        "text",
        "align",
        "consistency",
        "stream_commit_cosine",
        "event_state_norm",
        "auxiliary_classification_loss",
        "auxiliary_accuracy",
        "auxiliary_macro_f1",
        "samples",
    }
    assert expected <= validation_rows[0].keys()
    assert expected <= test_metrics.keys()
    assert validation_rows[0]["samples"] == 3.0
    assert test_metrics["samples"] == 3.0
    assert (stage2_output / "best.pt").exists()
    assert (stage2_output / "best_semantic_warmup.pt").exists()
    best = torch.load(stage2_output / "best.pt", map_location="cpu", weights_only=False)
    assert best["trainable_only"] is True
    assert best["epoch"] == 1
    assert all(np.isfinite(validation_rows[0][key]) for key in expected)

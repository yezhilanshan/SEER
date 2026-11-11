from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
import pytest

from continuous_mmwave.parrad_stage3_data import (
    CacheFingerprint,
    Stage3CacheRecord,
    write_stage3_cache,
)
from continuous_mmwave.parrad_stage3_runner import _binary_metrics, _score, run_parrad_stage3
from continuous_mmwave.parrad_stage3_calibrate import (
    coarse_threshold_grid,
    fixed_threshold_grid,
    load_validation_recordings,
    proposal_grid_shards,
    refinement_threshold_grid,
    smoke_threshold_grid,
    timeout_reject_grid,
    timeout_grid_shards,
)
from continuous_mmwave.train import main as cli_main
from continuous_mmwave.parrad_stage3_diagnostics import (
    ControllerThresholds,
    ThresholdGrid,
    ValidationSweepResult,
    calibrate_thresholds,
    evaluate_test_once,
    iter_threshold_candidates,
    match_boundaries_seconds,
    merge_validation_sweeps,
    run_validation_sweep,
)


def _record(offset: float) -> Stage3CacheRecord:
    return Stage3CacheRecord(
        radar_tokens=(np.arange(4 * 4 * 8).reshape(4, 4, 8) / 100 + offset).astype(np.float16),
        radar_mask=np.ones((4, 4), dtype=np.bool_),
        frame_labels=np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int64),
        timestamps=(np.arange(8).reshape(4, 2) * 0.09 + offset).astype(np.float64),
        oracle_keys=np.array(
            [[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0], [0.1, 0.9, 0, 0]],
            dtype=np.float16,
        ),
        oracle_maturity=np.array([0.5, 1, 0.5, 1], dtype=np.float32),
        oracle_run_id=np.array([0, 0, 1, 1], dtype=np.int64),
    )


def tiny_stage3_config(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "stage2.pt"
    checkpoint.write_bytes(b"tiny-stage2")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    fingerprint = CacheFingerprint(checkpoint_hash, "config", "manifest")
    cache_dir = tmp_path / "cache"
    rows = []
    for split_index, split in enumerate(("train", "validation")):
        split_root = cache_dir / split
        for item_index in range(2):
            key = f"{split}-{item_index}"
            write_stage3_cache(
                split_root,
                key,
                _record(split_index + item_index / 10),
                fingerprint,
                {
                    "subject_id": f"{split}-subject-{item_index}",
                    "session_id": f"{split}-session-{item_index}",
                    "sensor_id": "60ghz",
                    "environment_id": "lab",
                },
            )
            rows.append({"split": split, "key": key, "path": str(split_root)})
    cache_dir.mkdir(exist_ok=True)
    (cache_dir / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fingerprint": fingerprint.__dict__,
                "records": rows,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = {
        "model": {
            "radar_dim": 8,
            "semantic_dim": 4,
            "frames_per_update": 2,
            "frame_period_s": 0.09,
        },
        "training": {"device": "cpu"},
        "stage3": {
            "cache_dir": str(cache_dir),
            "stage2_checkpoint": str(checkpoint),
            "output_dir": str(tmp_path / "outputs"),
            "seed": 42,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "epochs_physical": 1,
            "epochs_semantic": 1,
            "epochs_verify": 1,
            "early_stopping_patience": 1,
            "prototype_dim": 4,
        },
    }
    path = tmp_path / "stage3.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_stage3_runner_trains_phases_in_order_and_selects_validation_best(tmp_path: Path) -> None:
    result = run_parrad_stage3(tiny_stage3_config(tmp_path), max_steps_per_phase=1)

    assert [row["phase"] for row in result.phase_summaries] == [
        "physical",
        "semantic",
        "verify",
    ]
    assert (result.output_dir / "best_physical.pt").exists()
    assert (result.output_dir / "best_semantic.pt").exists()
    assert (result.output_dir / "best_verify.pt").exists()
    physical_validation = json.loads(
        (result.output_dir / "validation-physical.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert {
        "local_pr_auc",
        "radar_pr_auc",
        "recall_at_fpr_10",
    } <= physical_validation.keys()


def test_binary_metrics_reports_recall_at_ten_percent_false_positive_rate() -> None:
    metrics = _binary_metrics(
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.tensor([0.9, 0.4, 0.8, 0.1]),
    )

    assert metrics["recall_at_fpr_10"] == pytest.approx(0.5)


def test_verify_checkpoint_score_enforces_false_confirm_gate() -> None:
    high_ranking_but_unsafe = {
        "pr_auc": 0.999,
        "auroc": 0.999,
        "balanced_accuracy": 0.62,
        "false_confirm_rate": 0.76,
        "loss": 0.7,
    }
    slightly_lower_but_safe = {
        "pr_auc": 0.97,
        "auroc": 0.989,
        "balanced_accuracy": 0.92,
        "false_confirm_rate": 0.045,
        "loss": 0.5,
    }

    assert _score("verify", slightly_lower_but_safe) > _score(
        "verify", high_ranking_but_unsafe
    )


def test_physical_checkpoint_score_prioritizes_recall_under_false_positive_budget() -> None:
    higher_pr_but_low_recall = {
        "pr_auc": 0.30,
        "recall_at_fpr_10": 0.20,
        "loss": 0.7,
    }
    slightly_lower_pr_but_useful_recall = {
        "pr_auc": 0.28,
        "recall_at_fpr_10": 0.55,
        "loss": 0.8,
    }

    assert _score("physical", slightly_lower_pr_but_useful_recall) > _score(
        "physical", higher_pr_but_low_recall
    )


def test_stage3_checkpoint_contains_heads_only_and_cache_fingerprint(tmp_path: Path) -> None:
    result = run_parrad_stage3(tiny_stage3_config(tmp_path), max_steps_per_phase=1)

    payload = torch.load(result.final_checkpoint, map_location="cpu", weights_only=False)

    assert payload["schema_version"] == 1
    assert payload["cache_fingerprint"] == result.cache_fingerprint
    assert payload["phase"] == "verify"
    assert payload["model_state"]
    assert all(
        name.startswith(("local_change", "radar_transition", "semantic_difference", "verify"))
        for name in payload["model_state"]
    )


def test_boundary_matching_is_one_to_one_in_seconds() -> None:
    result = match_boundaries_seconds(
        predicted=[1.1, 1.2, 3.4], truth=[1.0, 3.0], tolerance_s=0.5
    )

    assert result.true_positives == 2
    assert result.false_positives == 1
    assert result.false_negatives == 0
    assert result.mean_absolute_error_s == pytest.approx(0.25)


def test_calibration_uses_validation_only_and_freezes_thresholds(tmp_path: Path) -> None:
    first = ControllerThresholds(0.4, 0.7, 0.2, 8, "candidate-a")
    second = ControllerThresholds(0.5, 0.8, 0.2, 10, "candidate-b")
    results = [
        ValidationSweepResult(first, 0.6, 0.7, 0.3, 0.8),
        ValidationSweepResult(second, 0.7, 0.8, 0.2, 1.0),
    ]
    grid = ThresholdGrid((0.4, 0.5), (0.7, 0.8), (0.2,), (8, 10))
    threshold_path = tmp_path / "thresholds.json"

    chosen = calibrate_thresholds(results, grid, threshold_path)
    before = threshold_path.read_bytes()
    metrics = evaluate_test_once(lambda thresholds: {"proposal": thresholds.proposal}, chosen, tmp_path / "tested.json")

    assert chosen == second
    assert metrics == {"proposal": 0.5}
    assert threshold_path.read_bytes() == before
    with pytest.raises(RuntimeError, match="already evaluated"):
        evaluate_test_once(lambda thresholds: {}, chosen, tmp_path / "tested.json")


def test_threshold_candidates_cover_proposal_persistence_axis() -> None:
    grid = ThresholdGrid(
        proposal=(0.4,),
        verify_confirm=(0.7,),
        verify_reject=(0.2,),
        candidate_timeout_updates=(10,),
        proposal_persistence=(1, 2),
    )

    candidates = tuple(iter_threshold_candidates(grid, "checkpoint-sha"))

    assert [candidate.proposal_persistence for candidate in candidates] == [1, 2]
    assert len({candidate.fingerprint for candidate in candidates}) == 2


def test_validation_sweep_resumes_and_selects_checkpoint_with_persistence(tmp_path: Path) -> None:
    first_checkpoint = tmp_path / "best_verify.pt"
    second_checkpoint = tmp_path / "last_verify.pt"
    first_checkpoint.write_bytes(b"best")
    second_checkpoint.write_bytes(b"last")
    grid = ThresholdGrid(
        proposal=(0.4,),
        verify_confirm=(0.7,),
        verify_reject=(0.2,),
        candidate_timeout_updates=(10,),
        proposal_persistence=(1, 2),
    )
    calls: list[tuple[str, int]] = []

    def evaluator(checkpoint: Path, thresholds: ControllerThresholds) -> dict[str, object]:
        calls.append((checkpoint.name, thresholds.proposal_persistence))
        preferred = checkpoint == second_checkpoint and thresholds.proposal_persistence == 2
        return {
            "boundary_036": {"f1": 0.8 if preferred else 0.6},
            "boundary_072": {"f1": 0.9 if preferred else 0.7},
            "over_segmentation": 0.1 if preferred else 0.3,
            "median_confirmation_latency_s": 0.9,
        }

    chosen = run_validation_sweep(
        {"best": first_checkpoint, "last": second_checkpoint},
        grid,
        evaluator,
        tmp_path / "calibration",
        validation_fingerprint="validation-full",
    )
    first_call_count = len(calls)
    resumed = run_validation_sweep(
        {"best": first_checkpoint, "last": second_checkpoint},
        grid,
        evaluator,
        tmp_path / "calibration",
        validation_fingerprint="validation-full",
    )
    artifact = json.loads(
        (tmp_path / "calibration" / "validation_thresholds.json").read_text(encoding="utf-8")
    )

    assert chosen == resumed
    assert chosen.proposal_persistence == 2
    assert len(calls) == first_call_count == 4
    assert artifact["status"] == "calibrated_validation"
    assert artifact["test_evaluation_allowed"] is False
    assert artifact["validation_context"]["validation_fingerprint"] == "validation-full"
    assert len(artifact["validation_context_sha256"]) == 64
    assert Path(artifact["selected_checkpoint"]["path"]).name == "last_verify.pt"

    with pytest.raises(ValueError, match="sweep context does not match"):
        run_validation_sweep(
            {"best": first_checkpoint, "last": second_checkpoint},
            grid,
            evaluator,
            tmp_path / "calibration",
            validation_fingerprint="validation-subset",
        )


def test_validation_sweep_rejects_overwritten_checkpoint_on_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best_verify.pt"
    checkpoint.write_bytes(b"first-version")
    grid = ThresholdGrid((0.4,), (0.7,), (0.2,), (10,), (1,))

    def evaluator(_checkpoint: Path, _thresholds: ControllerThresholds) -> dict[str, object]:
        return {
            "boundary_036": {"f1": 0.6},
            "boundary_072": {"f1": 0.7},
            "over_segmentation": 0.2,
            "median_confirmation_latency_s": 0.9,
        }

    run_validation_sweep(
        {"best": checkpoint},
        grid,
        evaluator,
        tmp_path / "calibration",
        validation_fingerprint="validation-full",
    )
    checkpoint.write_bytes(b"second-version")

    with pytest.raises(ValueError, match="sweep context does not match"):
        run_validation_sweep(
            {"best": checkpoint},
            grid,
            evaluator,
            tmp_path / "calibration",
            validation_fingerprint="validation-full",
        )


def test_merge_validation_sweeps_requires_complete_shards_and_selects_globally(
    tmp_path: Path,
) -> None:
    best_checkpoint = tmp_path / "best.pt"
    last_checkpoint = tmp_path / "last.pt"
    best_checkpoint.write_bytes(b"best")
    last_checkpoint.write_bytes(b"last")
    grid = ThresholdGrid((0.4,), (0.7,), (0.2,), (10,), (1, 2))

    def evaluator(checkpoint: Path, thresholds: ControllerThresholds) -> dict[str, object]:
        preferred = checkpoint == last_checkpoint and thresholds.proposal_persistence == 2
        return {
            "boundary_036": {"f1": 0.8 if preferred else 0.6},
            "boundary_072": {"f1": 0.9 if preferred else 0.7},
            "over_segmentation": 0.1 if preferred else 0.3,
            "median_confirmation_latency_s": 0.9,
        }

    best_output = tmp_path / "best-sweep"
    last_output = tmp_path / "last-sweep"
    run_validation_sweep(
        {"best": best_checkpoint},
        grid,
        evaluator,
        best_output,
        validation_fingerprint="validation-full",
    )
    run_validation_sweep(
        {"last": last_checkpoint},
        grid,
        evaluator,
        last_output,
        validation_fingerprint="validation-full",
    )

    chosen = merge_validation_sweeps(
        [best_output, last_output], tmp_path / "merged"
    )
    artifact = json.loads(
        (tmp_path / "merged" / "validation_thresholds.json").read_text(encoding="utf-8")
    )

    assert chosen.proposal_persistence == 2
    assert artifact["selected_checkpoint"]["name"] == "last"
    assert set(artifact["validation_context"]["checkpoints"]) == {"best", "last"}


def test_fixed_threshold_grid_matches_frozen_validation_design() -> None:
    grid = fixed_threshold_grid()

    assert grid.proposal == (0.3, 0.4, 0.5, 0.6, 0.7)
    assert grid.proposal_persistence == (1, 2)
    assert grid.verify_confirm == (0.5, 0.6, 0.7, 0.8)
    assert grid.verify_reject == (0.2, 0.3, 0.4)
    assert grid.candidate_timeout_updates == (10,)
    assert len(tuple(iter_threshold_candidates(grid, "checkpoint"))) == 120

    slower_updates = fixed_threshold_grid(update_period_s=0.25, reject_timeout_s=1.8)
    assert slower_updates.candidate_timeout_updates == (8,)


def test_coarse_grid_is_partitioned_into_three_complete_proposal_shards() -> None:
    grid = coarse_threshold_grid()
    shards = proposal_grid_shards(grid)

    assert grid.proposal == (0.4, 0.5, 0.6)
    assert grid.proposal_persistence == (1, 2)
    assert grid.verify_confirm == (0.45, 0.5)
    assert grid.verify_reject == (0.2, 0.3)
    assert len(tuple(iter_threshold_candidates(grid, "checkpoint"))) == 24
    assert len(shards) == 3
    assert [len(tuple(iter_threshold_candidates(shard, "checkpoint"))) for shard in shards] == [8, 8, 8]
    assert {
        candidate.fingerprint
        for shard in shards
        for candidate in iter_threshold_candidates(shard, "checkpoint")
    } == {
        candidate.fingerprint
        for candidate in iter_threshold_candidates(grid, "checkpoint")
    }


def test_smoke_grid_uses_a_reachable_verify_threshold() -> None:
    grid = smoke_threshold_grid()

    assert grid.verify_confirm == (0.45,)
    assert len(tuple(iter_threshold_candidates(grid, "checkpoint"))) == 1


def test_refinement_grid_targets_the_coarse_verify_cliff() -> None:
    selected = ControllerThresholds(0.5, 0.45, 0.2, 10, "winner", 2)

    grid = refinement_threshold_grid(selected)

    assert grid.proposal == (0.45, 0.5, 0.55)
    assert grid.proposal_persistence == (2,)
    assert grid.verify_confirm == (0.45, 0.46, 0.47, 0.48, 0.49, 0.5)
    assert grid.verify_reject == (0.2,)
    assert grid.candidate_timeout_updates == (10,)
    assert len(tuple(iter_threshold_candidates(grid, "checkpoint"))) == 18


def test_timeout_reject_grid_isolates_candidate_lifecycle_axes() -> None:
    selected = ControllerThresholds(0.55, 0.45, 0.2, 10, "winner", 2)

    grid = timeout_reject_grid(selected)
    shards = timeout_grid_shards(grid)

    assert grid.proposal == (0.55,)
    assert grid.proposal_persistence == (2,)
    assert grid.verify_confirm == (0.45, 0.46, 0.47)
    assert grid.verify_reject == (0.2, 0.4)
    assert grid.candidate_timeout_updates == (6, 8, 10)
    assert len(tuple(iter_threshold_candidates(grid, "checkpoint"))) == 18
    assert [shard.candidate_timeout_updates for shard in shards] == [(6,), (8,), (10,)]
    assert {
        candidate.fingerprint
        for shard in shards
        for candidate in iter_threshold_candidates(shard, "checkpoint")
    } == {
        candidate.fingerprint
        for candidate in iter_threshold_candidates(grid, "checkpoint")
    }


def test_merge_validation_sweeps_accepts_complete_partitioned_grid(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "winner.pt"
    checkpoint.write_bytes(b"winner")
    target_grid = ThresholdGrid((0.4, 0.5), (0.7,), (0.2,), (10,), (1,))
    sources: list[Path] = []

    def evaluator(_checkpoint: Path, thresholds: ControllerThresholds) -> dict[str, object]:
        preferred = thresholds.proposal == 0.5
        return {
            "boundary_036": {"f1": 0.8 if preferred else 0.6},
            "boundary_072": {"f1": 0.9 if preferred else 0.7},
            "over_segmentation": 0.1 if preferred else 0.3,
            "median_confirmation_latency_s": 0.9,
        }

    for index, shard_grid in enumerate(proposal_grid_shards(target_grid)):
        source = tmp_path / f"shard-{index}"
        run_validation_sweep(
            {"winner": checkpoint},
            shard_grid,
            evaluator,
            source,
            validation_fingerprint="validation-full",
        )
        sources.append(source)

    chosen = merge_validation_sweeps(
        sources,
        tmp_path / "merged-partitions",
        target_grid=target_grid,
    )

    assert chosen.proposal == 0.5


def test_validation_recording_loader_does_not_touch_test_manifest(tmp_path: Path) -> None:
    arrays = {
        "rd": np.zeros((2, 1, 93, 128), dtype=np.float32),
        "md": np.zeros((2, 128), dtype=np.float32),
        "labels": np.array([0, 1], dtype=np.int64),
        "timestamps": np.array([0.0, 0.09], dtype=np.float64),
    }
    train_npz = tmp_path / "train.npz"
    validation_npz = tmp_path / "validation.npz"
    np.savez(train_npz, **arrays)
    np.savez(validation_npz, **arrays)
    metadata = {
        "subject_id": "subject_01",
        "session_id": "session_01",
        "sensor_id": "60ghz",
        "environment_id": "lab",
    }
    train_manifest = tmp_path / "train.jsonl"
    validation_manifest = tmp_path / "validation.jsonl"
    train_manifest.write_text(
        json.dumps({"path": str(train_npz), **metadata}) + "\n", encoding="utf-8"
    )
    validation_manifest.write_text(
        json.dumps({"path": str(validation_npz), **metadata}) + "\n", encoding="utf-8"
    )
    config = tmp_path / "stage3.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"encoder_type": "causal_md_gru"},
                "data": {
                    "train_manifest": str(train_manifest),
                    "validation_manifest": str(validation_manifest),
                    "test_manifest": str(tmp_path / "must-not-be-read.jsonl"),
                },
                "stage": {"variant": "parrad", "num_classes": 14},
            }
        ),
        encoding="utf-8",
    )

    recordings = load_validation_recordings(config)

    assert len(recordings) == 1
    assert recordings[0].subject_id == "subject_01"


def test_stage3_cli_trains_heads_and_prepares_closed_loop_calibration(tmp_path: Path) -> None:
    cli_main(
        [
            "stage3",
            "--config",
            str(tiny_stage3_config(tmp_path)),
            "--max-steps",
            "1",
        ]
    )

    output = tmp_path / "outputs"
    assert (output / "best_verify.pt").exists()
    threshold_artifact = json.loads((output / "validation_thresholds.json").read_text(encoding="utf-8"))
    assert threshold_artifact["status"] == "pending_closed_loop_validation"
    assert threshold_artifact["test_evaluation_allowed"] is False

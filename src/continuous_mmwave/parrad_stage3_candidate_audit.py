"""Read-only, one-episode-per-proposal audit of online Stage 3 Candidates."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from continuous_mmwave.parrad_stage3_data import Stage3CacheItem, extract_event_runs
from continuous_mmwave.parrad_stage3_diagnostics import ControllerThresholds
from continuous_mmwave.parrad_stage3_hard_negative import _apply_thresholds


@dataclass(frozen=True)
class CandidateUpdate:
    timestamp: float
    difference_probability: float
    stability: float
    maturity: float
    verify_probability: float
    decision: str


@dataclass(frozen=True)
class CandidateEpisode:
    split: str
    recording_key: str
    sensor_id: str
    proposal_time: float
    terminal_time: float
    terminal_reason: str
    updates: tuple[CandidateUpdate, ...]
    nearest_boundary_time: float | None
    nearest_boundary_distance_s: float | None
    proximity_target: bool
    gt_run_at_proposal: int | None
    gt_run_at_terminal: int | None


@dataclass(frozen=True)
class CandidateFeatureRow:
    split: str
    recording_key: str
    sensor_id: str
    proposal_time: float
    inputs: tuple[float, float, float]
    target: bool


def candidate_feature_rows(
    episodes: Sequence[CandidateEpisode],
    *,
    position: str,
    boundary_tolerance_s: float,
) -> tuple[CandidateFeatureRow, ...]:
    """Select one first, last, or mean feature vector per scored Candidate."""

    if position not in {"first", "last", "mean"}:
        raise ValueError("position must be first, last, or mean")
    if not math.isfinite(boundary_tolerance_s) or boundary_tolerance_s <= 0:
        raise ValueError("boundary tolerance must be finite and positive")
    rows: list[CandidateFeatureRow] = []
    for episode in episodes:
        if not episode.updates:
            continue
        if position == "mean":
            count = len(episode.updates)
            values = tuple(
                sum(getattr(update, name) for update in episode.updates) / count
                for name in ("difference_probability", "stability", "maturity")
            )
        else:
            update = episode.updates[0 if position == "first" else -1]
            values = (update.difference_probability, update.stability, update.maturity)
        rows.append(CandidateFeatureRow(
            episode.split, episode.recording_key, episode.sensor_id, episode.proposal_time,
            values,
            episode.nearest_boundary_distance_s is not None
            and episode.nearest_boundary_distance_s <= boundary_tolerance_s,
        ))
    return tuple(rows)


def score_binary(targets: Sequence[bool], scores: Sequence[float]) -> dict[str, float | int | None]:
    """Tie-aware AUROC and average precision; undefined metrics remain null."""

    if len(targets) != len(scores):
        raise ValueError("targets and scores must have equal length")
    if any(not math.isfinite(float(score)) for score in scores):
        raise ValueError("scores must be finite")
    count = len(targets)
    positives = sum(bool(target) for target in targets)
    negatives = count - positives
    result: dict[str, float | int | None] = {
        "count": count,
        "positives": positives,
        "positive_prevalence": positives / count if count else None,
        "auroc": None,
        "average_precision": None,
    }
    if not positives:
        return result
    if not negatives:
        result["average_precision"] = 1.0
        return result
    pairs = sorted(zip(scores, targets, strict=True), key=lambda pair: pair[0], reverse=True)
    true_positives = false_positives = 0
    area = average_precision = 0.0
    index = 0
    while index < count:
        score = pairs[index][0]
        next_index = index
        group_positives = group_negatives = 0
        while next_index < count and pairs[next_index][0] == score:
            if pairs[next_index][1]:
                group_positives += 1
            else:
                group_negatives += 1
            next_index += 1
        previous_tpr = true_positives / positives
        previous_fpr = false_positives / negatives
        true_positives += group_positives
        false_positives += group_negatives
        tpr = true_positives / positives
        fpr = false_positives / negatives
        area += (fpr - previous_fpr) * (tpr + previous_tpr) / 2
        average_precision += (tpr - previous_tpr) * (true_positives / (true_positives + false_positives))
        index = next_index
    result["auroc"] = area
    result["average_precision"] = average_precision
    return result


def fit_candidate_probe(
    train_rows: Sequence[CandidateFeatureRow],
    validation_rows: Sequence[CandidateFeatureRow],
    *,
    hidden_dim: int,
    epochs: int = 20,
    seed: int = 42,
    learning_rate: float = 1e-3,
) -> dict[str, Any]:
    """Fit a tiny train-only diagnostic head and score validation once."""

    if not train_rows or not validation_rows:
        raise ValueError("probe requires nonempty train and validation Candidates")
    if hidden_dim < 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("invalid diagnostic probe settings")
    if any(row.split != "train" for row in train_rows):
        raise ValueError("probe fitting accepts only train rows")
    if any(row.split != "validation" for row in validation_rows):
        raise ValueError("probe evaluation accepts only validation rows")
    train_targets = [row.target for row in train_rows]
    if len(set(train_targets)) < 2:
        raise ValueError("probe training requires both Candidate classes")
    train_inputs = torch.tensor([row.inputs for row in train_rows], dtype=torch.float32)
    validation_inputs = torch.tensor([row.inputs for row in validation_rows], dtype=torch.float32)
    if not torch.isfinite(train_inputs).all() or not torch.isfinite(validation_inputs).all():
        raise ValueError("probe inputs must be finite")
    mean = train_inputs.mean(dim=0)
    std = train_inputs.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_inputs = (train_inputs - mean) / std
    validation_inputs = (validation_inputs - mean) / std
    train_labels = torch.tensor(train_targets, dtype=torch.float32)[:, None]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = (
            torch.nn.Sequential(torch.nn.Linear(3, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, 1))
            if hidden_dim
            else torch.nn.Linear(3, 1)
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(train_inputs), train_labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(validation_inputs)).reshape(-1).tolist()
    targets = [row.target for row in validation_rows]
    sensor_metrics = {
        sensor: score_binary(
            [row.target for row in validation_rows if row.sensor_id == sensor],
            [score for row, score in zip(validation_rows, scores, strict=True) if row.sensor_id == sensor],
        )
        for sensor in sorted({row.sensor_id for row in validation_rows})
    }
    return {
        "architecture": "linear" if hidden_dim == 0 else f"mlp_3_{hidden_dim}_1",
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "train_mean": mean.tolist(),
        "train_std": std.tolist(),
        "validation_metrics": score_binary(targets, scores),
        "validation_sensor_metrics": sensor_metrics,
        "validation_scores": scores,
    }


def _bootstrap_recordings(
    rows: Sequence[CandidateFeatureRow], scores: Sequence[float], *, repetitions: int = 200
) -> dict[str, list[float] | int]:
    """Resample recording identities, keeping their correlated Candidates together."""

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        sensor_suffix = f"__{row.sensor_id}"
        session_key = (
            row.recording_key[: -len(sensor_suffix)]
            if row.recording_key.endswith(sensor_suffix)
            else row.recording_key
        )
        groups.setdefault(session_key, []).append(index)
    keys = sorted(groups)
    rng = random.Random(42)
    aucs: list[float] = []
    aps: list[float] = []
    for _ in range(repetitions):
        sampled = [index for _ in keys for index in groups[rng.choice(keys)]]
        metrics = score_binary([rows[index].target for index in sampled], [scores[index] for index in sampled])
        if metrics["auroc"] is not None:
            aucs.append(float(metrics["auroc"]))
        if metrics["average_precision"] is not None:
            aps.append(float(metrics["average_precision"]))

    def interval(values: list[float]) -> list[float]:
        if not values:
            return []
        ordered = sorted(values)
        return [ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]]

    return {
        "recordings": len(keys),
        "valid_auc_repetitions": len(aucs),
        "auroc_95pct_interval": interval(aucs),
        "average_precision_95pct_interval": interval(aps),
    }


def feature_distribution(rows: Sequence[CandidateFeatureRow]) -> dict[str, Any]:
    """Describe each frozen feature by class overall and within each sensor."""

    names = ("difference_probability", "stability", "maturity")

    def group(items: Sequence[CandidateFeatureRow]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for class_name, target in (("positive", True), ("negative", False)):
            selected = [row for row in items if row.target is target]
            fields: dict[str, Any] = {"count": len(selected)}
            for index, name in enumerate(names):
                values = sorted(row.inputs[index] for row in selected)
                fields[name] = (
                    {
                        "median": statistics.median(values),
                        "q25": values[int(0.25 * (len(values) - 1))],
                        "q75": values[int(0.75 * (len(values) - 1))],
                    }
                    if values else None
                )
            result[class_name] = fields
        return result

    return {
        "overall": group(rows),
        "by_sensor": {
            sensor: group([row for row in rows if row.sensor_id == sensor])
            for sensor in sorted({row.sensor_id for row in rows})
        },
    }


@torch.inference_mode()
def score_frozen_verify(
    rows: Sequence[CandidateFeatureRow], verify_head: torch.nn.Module
) -> dict[str, Any]:
    """Score the existing Verify unchanged on one feature vector per Candidate."""

    if not rows:
        raise ValueError("frozen Verify requires scored Candidates")
    parameter = next(verify_head.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    inputs = torch.tensor([row.inputs for row in rows], dtype=torch.float32, device=device)
    if not torch.isfinite(inputs).all():
        raise ValueError("frozen Verify inputs must be finite")
    scores = torch.sigmoid(verify_head(inputs)).reshape(-1).detach().cpu().tolist()
    if len(scores) != len(rows):
        raise ValueError("frozen Verify emitted an unexpected number of scores")
    return {
        "validation_metrics": score_binary([row.target for row in rows], scores),
        "validation_sensor_metrics": {
            sensor: score_binary(
                [row.target for row in rows if row.sensor_id == sensor],
                [score for row, score in zip(rows, scores, strict=True) if row.sensor_id == sensor],
            )
            for sensor in sorted({row.sensor_id for row in rows})
        },
        "validation_scores": scores,
    }


def candidate_diagnostic(
    train_episodes: Sequence[CandidateEpisode],
    validation_episodes: Sequence[CandidateEpisode],
    *,
    frozen_verify: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Predeclared two-tolerance, three-snapshot train/validation probe matrix."""

    if any(episode.split != "train" for episode in train_episodes):
        raise ValueError("train episodes have a mismatched split")
    if any(episode.split != "validation" for episode in validation_episodes):
        raise ValueError("validation episodes have a mismatched split")

    def episode_summary(items: Sequence[CandidateEpisode]) -> dict[str, Any]:
        durations = sorted(episode.terminal_time - episode.proposal_time for episode in items)
        update_counts = sorted(len(episode.updates) for episode in items)
        return {
            "count": len(items),
            "by_sensor": dict(sorted(Counter(episode.sensor_id for episode in items).items())),
            "terminal_reasons": dict(sorted(Counter(episode.terminal_reason for episode in items).items())),
            "without_scored_update": sum(not episode.updates for episode in items),
            "duration_s_median": statistics.median(durations) if durations else None,
            "duration_s_p90": durations[int(0.9 * (len(durations) - 1))] if durations else None,
            "updates_median": statistics.median(update_counts) if update_counts else None,
            "updates_p90": update_counts[int(0.9 * (len(update_counts) - 1))] if update_counts else None,
            "gt_run_changed_count": sum(
                episode.gt_run_at_proposal is not None
                and episode.gt_run_at_terminal is not None
                and episode.gt_run_at_proposal != episode.gt_run_at_terminal
                for episode in items
            ),
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "source_split": ["train", "validation"],
        "test_evaluation_allowed": False,
        "episodes": {
            "train": episode_summary(train_episodes),
            "validation": episode_summary(validation_episodes),
        },
        "tolerances": {},
    }
    for tolerance in (0.36, 0.72):
        positions: dict[str, Any] = {}
        for position in ("first", "last", "mean"):
            train_rows = candidate_feature_rows(
                train_episodes, position=position, boundary_tolerance_s=tolerance
            )
            validation_rows = candidate_feature_rows(
                validation_episodes, position=position, boundary_tolerance_s=tolerance
            )
            feature_names = ("difference_probability", "stability", "maturity")
            univariate = {
                name: score_binary(
                    [row.target for row in validation_rows],
                    [row.inputs[index] for row in validation_rows],
                )
                for index, name in enumerate(feature_names)
            }
            probes: dict[str, Any] = {}
            for name, hidden_dim in (("linear", 0), ("mlp", 16)):
                result = fit_candidate_probe(train_rows, validation_rows, hidden_dim=hidden_dim)
                scores = result.pop("validation_scores")
                result["recording_bootstrap"] = _bootstrap_recordings(validation_rows, scores)
                probes[name] = result
            if frozen_verify is not None:
                result = score_frozen_verify(validation_rows, frozen_verify)
                scores = result.pop("validation_scores")
                result["recording_bootstrap"] = _bootstrap_recordings(validation_rows, scores)
                probes["frozen_verify"] = result
            positions[position] = {
                "train_candidates": len(train_rows),
                "validation_candidates": len(validation_rows),
                "validation_positive_prevalence": score_binary(
                    [row.target for row in validation_rows], [0.0] * len(validation_rows)
                )["positive_prevalence"],
                "univariate": univariate,
                "feature_distribution": feature_distribution(validation_rows),
                "probes": probes,
            }
        report["tolerances"][str(tolerance)] = positions
    return report


def _truth(item: Stage3CacheItem) -> tuple[tuple[float, ...], tuple[tuple[float, float], ...]]:
    labels = torch.from_numpy(item.record.frame_labels.reshape(-1).astype("int64"))
    times = torch.from_numpy(item.record.timestamps.reshape(-1).astype("float64"))
    runs = extract_event_runs(labels, times)
    return (
        tuple(float(run.boundary_from_previous) for run in runs if run.boundary_from_previous is not None),
        tuple((run.start_time, run.stop_time) for run in runs),
    )


def _run_at(time: float, runs: tuple[tuple[float, float], ...]) -> int | None:
    return next((index for index, (start, stop) in enumerate(runs) if start <= time <= stop), None)


def _finish(
    item: Stage3CacheItem,
    proposal_time: float,
    terminal_time: float,
    reason: str,
    updates: list[CandidateUpdate],
    boundaries: tuple[float, ...],
    runs: tuple[tuple[float, float], ...],
    tolerance: float,
) -> CandidateEpisode:
    nearest = min(boundaries, key=lambda boundary: abs(boundary - proposal_time)) if boundaries else None
    distance = abs(nearest - proposal_time) if nearest is not None else None
    return CandidateEpisode(
        split=item.split,
        recording_key=item.key,
        sensor_id=str(item.metadata.get("sensor_id", "")),
        proposal_time=proposal_time,
        terminal_time=terminal_time,
        terminal_reason=reason,
        updates=tuple(updates),
        nearest_boundary_time=nearest,
        nearest_boundary_distance_s=distance,
        proximity_target=distance is not None and distance <= tolerance,
        gt_run_at_proposal=_run_at(proposal_time, runs),
        gt_run_at_terminal=_run_at(terminal_time, runs),
    )


@torch.inference_mode()
def collect_candidate_episodes(
    runtime: Any,
    records: Sequence[Stage3CacheItem],
    thresholds: ControllerThresholds,
    *,
    boundary_tolerance_s: float = 0.72,
) -> tuple[CandidateEpisode, ...]:
    """Replay only the supplied train/validation records; never fit or change weights."""

    if not math.isfinite(boundary_tolerance_s) or boundary_tolerance_s <= 0:
        raise ValueError("boundary tolerance must be finite and positive")
    if any(item.split not in {"train", "validation"} for item in records):
        raise ValueError("Candidate audit accepts only train or validation records")
    _apply_thresholds(runtime, thresholds)
    controller = runtime.controller
    event_former = getattr(controller, "event_former", None)
    parameter = (
        next(event_former.parameters(), None)
        if event_former is not None and hasattr(event_former, "parameters")
        else None
    )
    device = parameter.device if parameter is not None else torch.device("cpu")
    episodes: list[CandidateEpisode] = []
    for item in records:
        controller.reset()
        boundaries, runs = _truth(item)
        proposal_time: float | None = None
        updates: list[CandidateUpdate] = []
        for index in range(len(item.record.radar_tokens)):
            tokens = torch.from_numpy(item.record.radar_tokens[index].astype("float32"))
            mask = torch.from_numpy(item.record.radar_mask[index].astype("bool"))
            timestamp = float(item.record.timestamps[index].reshape(-1)[0])
            output = controller.step_tokens(
                tokens.unsqueeze(0).to(device), mask.unsqueeze(0).to(device), timestamp, item.metadata
            )
            if output.decision == "proposal":
                if proposal_time is not None:
                    raise ValueError("controller proposed before the previous Candidate ended")
                proposal_time = timestamp
                updates = []
            if output.candidate_proposal_time is not None:
                if proposal_time is None:
                    raise ValueError("Candidate update has no preceding proposal")
                if not math.isclose(float(output.candidate_proposal_time), proposal_time, abs_tol=1e-6):
                    raise ValueError("Candidate proposal timestamp changed during its lifetime")
                values = (
                    output.difference_probability, output.stability,
                    output.candidate_maturity, output.verify_probability,
                )
                if any(value is None or not math.isfinite(float(value)) for value in values):
                    raise ValueError("Candidate update contains missing or nonfinite features")
                updates.append(CandidateUpdate(timestamp, *(float(value) for value in values), output.decision))
            if output.decision in {"confirm", "reject"}:
                if proposal_time is None:
                    raise ValueError("controller terminated a Candidate without a proposal")
                episodes.append(_finish(
                    item, proposal_time, timestamp, output.decision, updates,
                    boundaries, runs, boundary_tolerance_s,
                ))
                proposal_time = None
                updates = []
        if proposal_time is not None:
            terminal_time = updates[-1].timestamp if updates else proposal_time
            episodes.append(_finish(
                item, proposal_time, terminal_time, "recording_end", updates,
                boundaries, runs, boundary_tolerance_s,
            ))
    return tuple(episodes)


def main() -> None:
    """Run the frozen-checkpoint audit without loading the test cache."""

    from continuous_mmwave.parrad_stage3_calibrate import build_runtime
    from continuous_mmwave.parrad_stage3_hard_negative import _file_sha256, _load_thresholds, _load_yaml
    from continuous_mmwave.parrad_stage3_runner import _load_cache_items

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--selected-thresholds", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    thresholds = _load_thresholds(args.selected_thresholds, args.checkpoint)
    raw = _load_yaml(args.config)
    stage3 = dict(raw.get("stage3", {}))
    records, cache_fingerprint = _load_cache_items(
        Path(str(stage3["cache_dir"])),
        Path(str(stage3["stage2_checkpoint"])),
        splits=("train", "validation"),
    )
    runtime = build_runtime(args.config, args.checkpoint, torch.device(args.device))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    episodes: dict[str, tuple[CandidateEpisode, ...]] = {}
    for split in ("train", "validation"):
        episodes[split] = collect_candidate_episodes(runtime, records[split], thresholds)
        path = args.output_dir / f"candidate-episodes-{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for episode in episodes[split]:
                handle.write(json.dumps(asdict(episode), sort_keys=True) + "\n")
    report = candidate_diagnostic(
        episodes["train"], episodes["validation"], frozen_verify=runtime.controller.verify
    )
    report.update({
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "threshold_artifact_path": str(args.selected_thresholds),
        "threshold_artifact_sha256": _file_sha256(args.selected_thresholds),
        "cache_fingerprint": cache_fingerprint,
        "thresholds": asdict(thresholds),
        "recordings": {split: len(records[split]) for split in ("train", "validation")},
    })
    (args.output_dir / "candidate-diagnostic.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "episodes": report["episodes"]}, sort_keys=True))


if __name__ == "__main__":
    main()

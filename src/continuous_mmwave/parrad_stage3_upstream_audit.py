"""Frozen, validation-only audit of Stage 3 proposal and semantic inputs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from continuous_mmwave.parrad_stage3_data import extract_event_runs
from continuous_mmwave.parrad_stage3_candidate_audit import _truth, score_binary
from continuous_mmwave.parrad_stage3_hard_negative import _apply_thresholds


@dataclass(frozen=True)
class StepObservation:
    output: Any
    local_probability: float | None
    radar_probability: float
    radar_calls: int
    semantic_active: Tensor | None
    semantic_candidate: Tensor | None
    semantic_maturity: float | None
    semantic_logit: float | None


class ControllerStepTap:
    """Observe head inputs/outputs without changing the controller or its weights."""

    def __init__(self, controller: Any) -> None:
        self.controller = controller
        self.handles: list[Any] = []
        self._clear()

    def _clear(self) -> None:
        self.local_logits: list[float] = []
        self.radar_logits: list[float] = []
        self.semantic_calls: list[tuple[Tensor, Tensor, float, float]] = []

    def __enter__(self) -> ControllerStepTap:
        self.handles = [
            self.controller.local_change.register_forward_hook(self._local),
            self.controller.radar_transition.register_forward_hook(self._radar),
            self.controller.semantic_difference.register_forward_hook(self._semantic),
        ]
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _local(self, _module: Any, _inputs: tuple[Any, ...], output: Tensor) -> None:
        self.local_logits.append(float(output.reshape(-1)[0]))

    def _radar(self, _module: Any, _inputs: tuple[Any, ...], output: tuple[Tensor, Any]) -> None:
        self.radar_logits.append(float(output[0].reshape(-1)[0]))

    def _semantic(self, _module: Any, inputs: tuple[Tensor, Tensor, Tensor], output: Tensor) -> None:
        self.semantic_calls.append((
            inputs[0].detach().clone(), inputs[1].detach().clone(),
            float(inputs[2].reshape(-1)[0]), float(output.reshape(-1)[0]),
        ))

    def step(self, tokens: Tensor, mask: Tensor, timestamp: float, metadata: Any = None) -> StepObservation:
        self._clear()
        output = self.controller.step_tokens(tokens, mask, timestamp, metadata)
        if len(self.local_logits) > 1 or len(self.radar_logits) not in {1, 2} or len(self.semantic_calls) > 1:
            raise RuntimeError("unexpected Stage 3 head call count during one controller step")
        local = float(torch.sigmoid(torch.tensor(self.local_logits[0]))) if self.local_logits else None
        radar = float(torch.sigmoid(torch.tensor(self.radar_logits[0])))
        fused = 1 - (1 - (local or 0.0)) * (1 - radar)
        if not math.isclose(fused, output.proposal_probability, abs_tol=1e-5):
            raise RuntimeError("observed physical head scores do not reproduce controller proposal")
        semantic = self.semantic_calls[0] if self.semantic_calls else None
        if (semantic is None) != (output.difference_probability is None):
            raise RuntimeError("semantic head calls do not match controller output")
        if semantic is not None and not math.isclose(
            float(torch.sigmoid(torch.tensor(semantic[3]))), output.difference_probability, abs_tol=1e-5
        ):
            raise RuntimeError("observed semantic logit does not reproduce controller probability")
        return StepObservation(
            output, local, radar, len(self.radar_logits),
            None if semantic is None else semantic[0],
            None if semantic is None else semantic[1],
            None if semantic is None else semantic[2],
            None if semantic is None else semantic[3],
        )


def proposal_attribution(local: float, radar: float, threshold: float) -> str:
    """Describe component threshold crossing at one observed proposal, not a causal ablation."""

    if not all(math.isfinite(value) and 0 <= value <= 1 for value in (local, radar, threshold)):
        raise ValueError("proposal probabilities and threshold must be finite in [0,1]")
    if local >= threshold and radar >= threshold:
        return "both"
    if local >= threshold:
        return "local_only"
    if radar >= threshold:
        return "radar_only"
    if 1 - (1 - local) * (1 - radar) >= threshold:
        return "fused_only"
    return "below_threshold"


def normalized_cosine_difference(active: Tensor, candidate: Tensor) -> float | None:
    """Fixed, unlearned [0,1] mapping of online key cosine distance.

    Zero-norm or nonfinite keys have no defined cosine and remain missing rather
    than being assigned a misleading neutral score.
    """

    if active.shape != candidate.shape or active.ndim != 2 or active.shape[0] != 1:
        raise ValueError("online semantic keys must share [1,D] shape")
    if not torch.isfinite(active).all() or not torch.isfinite(candidate).all():
        return None
    if float(torch.linalg.vector_norm(active)) <= 1e-8 or float(torch.linalg.vector_norm(candidate)) <= 1e-8:
        return None
    cosine = float(functional.cosine_similarity(active.float(), candidate.float(), dim=-1)[0])
    return min(1.0, max(0.0, (1.0 - cosine) / 2.0))


def semantic_cosine_age_diagnostic(semantic: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank online Candidates by age without fitting or tuning any parameter."""

    bins = (
        ("lt_0p4", lambda age: age < 0.4),
        ("0p4_to_0p8", lambda age: 0.4 <= age < 0.8),
        ("0p8_to_1p2", lambda age: 0.8 <= age <= 1.2),
        ("gt_1p2", lambda age: age > 1.2),
    )
    observed_candidates = {str(row["candidate_id"]) for row in semantic}

    def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "candidate_count": len(rows),
            "retention_fraction": len(rows) / len(observed_candidates) if observed_candidates else None,
            "by_sensor_count": dict(sorted(Counter(str(row["sensor_id"]) for row in rows).items())),
        }
        for target_name, target_fn in (
            ("eventual_switch", lambda row: row.get("eventual_gt_run_switch")),
            ("contemporaneous_switch", lambda row: (
                int(row["reference_run_id"]) != int(row["current_run_id"])
                if int(row["reference_run_id"]) >= 0 and int(row["current_run_id"]) >= 0 else None
            )),
        ):
            selected = [(row, target_fn(row)) for row in rows]
            selected = [(row, target) for row, target in selected if target is not None]
            targets = [bool(target) for _, target in selected]
            result[target_name] = {
                "cosine": score_binary(targets, [float(row["online_cosine_probability"]) for row, _ in selected]),
                "trained_head": score_binary(targets, [float(row["online_probability"]) for row, _ in selected]),
                "cosine_by_sensor": {
                    sensor: score_binary(
                        [bool(target) for row, target in selected if row["sensor_id"] == sensor],
                        [float(row["online_cosine_probability"]) for row, _ in selected if row["sensor_id"] == sensor],
                    )
                    for sensor in sorted({row["sensor_id"] for row, _ in selected})
                },
            }
        return result

    valid: list[dict[str, Any]] = []
    for row in semantic:
        age = float(row["age_s"])
        if not math.isfinite(age) or age < 0:
            raise ValueError("Candidate age must be nonnegative and finite")
        if row.get("online_cosine_probability") is not None:
            valid.append(row)
    by_age: dict[str, dict[str, dict[str, Any]]] = {}
    for row in valid:
        age_label = f"{float(row['age_s']):.2f}"
        by_age.setdefault(age_label, {}).setdefault(str(row["candidate_id"]), row)
    by_bin: dict[str, dict[str, dict[str, Any]]] = {name: {} for name, _ in bins}
    for row in valid:
        for name, predicate in bins:
            if predicate(float(row["age_s"])):
                by_bin[name].setdefault(str(row["candidate_id"]), row)
                break
    return {
        "schema_version": 1,
        "source_split": "validation",
        "test_evaluation_allowed": False,
        "baseline": "p_diff = clamp((1 - cosine(z_A,z_C))/2, 0, 1); no fitted normalization",
        "sample_rule": "earliest scored update per Candidate per age bin; exact age uses one update per Candidate",
        "total_observed_candidates": len(observed_candidates),
        "missing_or_invalid_cosine_rows": len(semantic) - len(valid),
        "age_bins": {name: describe(list(by_bin[name].values())) for name, _ in bins},
        "exact_ages": {
            age: describe(list(rows.values()))
            for age, rows in sorted(by_age.items(), key=lambda pair: float(pair[0]))
        },
        "caveat": (
            "Late-age Candidates are survivors of earlier controller decisions; age-bin AUROC is not a causal "
            "maturation curve. Eventual GT-run switch is a diagnostic terminal proxy, while contemporaneous "
            "GT-run switch can change within one Candidate. Neither is adjudicated semantic validity."
        ),
    }


@torch.inference_mode()
def semantic_counterfactuals(
    head: Any,
    *,
    online_reference: Tensor,
    online_candidate: Tensor,
    oracle_reference: Tensor,
    oracle_candidate: Tensor,
    online_maturity: float,
    oracle_maturity: float,
) -> dict[str, float]:
    """Swap one key or maturity at a time; comparisons are diagnostic, not valid trajectories."""

    keys = (online_reference, online_candidate, oracle_reference, oracle_candidate)
    if any(key.ndim != 2 or key.shape != keys[0].shape or not torch.isfinite(key).all() for key in keys):
        raise ValueError("semantic keys must be finite and share [1,D] shape")
    if keys[0].shape[0] != 1 or not all(math.isfinite(value) and 0 <= value <= 1 for value in (online_maturity, oracle_maturity)):
        raise ValueError("semantic audit requires batch one and valid maturities")
    active = torch.cat((online_reference, oracle_reference, online_reference, oracle_reference, oracle_reference))
    candidate = torch.cat((online_candidate, oracle_candidate, oracle_candidate, online_candidate, oracle_candidate))
    maturity = torch.tensor(
        [online_maturity] * 4 + [oracle_maturity], dtype=active.dtype, device=active.device
    )
    logits = head(active, candidate, maturity).reshape(-1)
    if logits.numel() != 5 or not torch.isfinite(logits).all():
        raise ValueError("semantic head returned nonfinite or wrongly shaped logits")
    names = (
        "online_logit", "oracle_pair_online_maturity_logit",
        "online_reference_oracle_candidate_logit", "oracle_reference_online_candidate_logit",
        "oracle_pair_oracle_maturity_logit",
    )
    result = {name: float(value) for name, value in zip(names, logits, strict=True)}
    for name in names:
        result[name.replace("_logit", "_probability")] = float(torch.sigmoid(torch.tensor(result[name])))
    result["reference_cosine_distance"] = float(1 - functional.cosine_similarity(online_reference, oracle_reference)[0])
    result["candidate_cosine_distance"] = float(1 - functional.cosine_similarity(online_candidate, oracle_candidate)[0])
    return result


def _sensor_scores(rows: list[dict[str, Any]], target_name: str, score_name: str) -> dict[str, Any]:
    return {
        sensor: score_binary(
            [bool(row[target_name]) for row in rows if row["sensor_id"] == sensor],
            [float(row[score_name]) for row in rows if row["sensor_id"] == sensor],
        )
        for sensor in sorted({row["sensor_id"] for row in rows})
    }


def summarize_upstream_rows(
    physical: list[dict[str, Any]], semantic: list[dict[str, Any]], *, proposal_threshold: float
) -> dict[str, Any]:
    """Separate update-level physical timing from one-row-per-Candidate semantic ranking."""

    eligible = [row for row in physical if row["eligible"] and row["boundary_distance_s"] is not None]
    proposals = [row for row in eligible if row["decision"] == "proposal"]
    far = [row for row in proposals if row["boundary_distance_s"] > 1.44]
    physical_scores: dict[str, Any] = {}
    for name in ("local_probability", "radar_probability", "proposal_probability"):
        selected = [row for row in eligible if row[name] is not None]
        targets = [row["boundary_distance_s"] <= 0.72 for row in selected]
        values = [float(row[name]) for row in selected]
        annotated = [{**row, "near_boundary": target} for row, target in zip(selected, targets, strict=True)]
        physical_scores[name] = {
            "overall": score_binary(targets, values),
            "by_sensor": _sensor_scores(annotated, "near_boundary", name),
        }

    score_names = (
        "online_probability", "oracle_pair_online_maturity_probability",
        "online_reference_oracle_candidate_probability",
        "oracle_reference_online_candidate_probability", "oracle_pair_oracle_maturity_probability",
    )
    semantic_results: dict[str, Any] = {}
    for position in ("first", "last"):
        chosen: dict[str, dict[str, Any]] = {}
        for row in semantic:
            if row.get("eventual_gt_run_switch") is None or any(row.get(name) is None for name in score_names):
                continue
            key = str(row["candidate_id"])
            if position == "last" or key not in chosen:
                chosen[key] = row
        rows = list(chosen.values())
        targets = [bool(row["eventual_gt_run_switch"]) for row in rows]
        result: dict[str, Any] = {
            "candidate_count": len(rows),
            "positive_prevalence": sum(targets) / len(rows) if rows else None,
        }
        for name in score_names:
            result[name] = {
                **score_binary(targets, [float(row[name]) for row in rows]),
                "by_sensor": _sensor_scores(rows, "eventual_gt_run_switch", name),
            }
        for name in ("reference_cosine_distance", "candidate_cosine_distance"):
            result[name] = {
                label: (statistics.median(float(row[name]) for row in rows if bool(row["eventual_gt_run_switch"]) is target)
                        if any(bool(row["eventual_gt_run_switch"]) is target for row in rows) else None)
                for label, target in (("switch", True), ("no_switch", False))
            }
        semantic_results[position] = result
    return {
        "schema_version": 1,
        "source_split": "validation",
        "test_evaluation_allowed": False,
        "proposal_threshold": proposal_threshold,
        "physical": {
            "eligible_updates": len(eligible),
            "proposals": len(proposals),
            "near_boundary_0p72s_scores": physical_scores,
            "far_proposals_gt_1p44s": {
                "count": len(far),
                "by_sensor": dict(sorted(Counter(row["sensor_id"] for row in far).items())),
                "attribution": dict(sorted(Counter(row["attribution"] for row in far).items())),
                "median_component_probability": {
                    name: statistics.median(float(row[name]) for row in far) if far else None
                    for name in ("local_probability", "radar_probability", "proposal_probability")
                },
            },
        },
        "semantic": semantic_results,
        "caveat": (
            "Oracle keys are GT-run-reset counterfactuals, not deployable online states; "
            "GT-run-switch is a diagnostic proxy, not adjudicated Candidate validity. "
            "Proposal attribution is threshold arithmetic on observed states, not a causal ablation."
        ),
    }


@torch.inference_mode()
def adjacent_oracle_pair_diagnostic(records: tuple[Any, ...], head: Any) -> dict[str, Any]:
    """Compare direct adjacent GT runs with same-run controls at matched maturity."""

    if not records or any(item.split != "validation" for item in records):
        raise ValueError("adjacent oracle audit accepts only nonempty validation records")
    parameter = next(head.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    rows: dict[str, list[dict[str, Any]]] = {"nascent": [], "mature": []}
    for item in records:
        record = item.record
        labels = torch.from_numpy(record.frame_labels.reshape(-1).astype("int64"))
        times = torch.from_numpy(record.timestamps.reshape(-1).astype("float64"))
        runs = extract_event_runs(labels, times)
        keys = torch.from_numpy(record.oracle_keys.astype("float32")).to(device)
        for run_id, run in enumerate(runs):
            if run_id == 0 or run.boundary_from_previous is None:
                continue
            previous = [index for index, value in enumerate(record.oracle_run_id) if int(value) == run_id - 1]
            current = [index for index, value in enumerate(record.oracle_run_id) if int(value) == run_id]
            if len(previous) < 2 or not current:
                continue
            for position, negative_index, positive_index in (
                ("nascent", previous[0], current[0]),
                ("mature", previous[-2], current[-1]),
            ):
                active = keys[previous[-1] : previous[-1] + 1].expand(2, -1)
                candidate = torch.stack((keys[negative_index], keys[positive_index]))
                maturity_value = float(record.oracle_maturity[positive_index])
                maturity = torch.full((2,), maturity_value, device=device, dtype=keys.dtype)
                logits = head(active, candidate, maturity).reshape(-1)
                if logits.numel() != 2 or not torch.isfinite(logits).all():
                    raise ValueError("semantic head returned invalid adjacent pair scores")
                scores = torch.sigmoid(logits)
                distance = 1 - functional.cosine_similarity(active, candidate, dim=-1)
                rows[position].append({
                    "recording_key": item.key,
                    "sensor_id": str(item.metadata.get("sensor_id", "")),
                    "negative_probability": float(scores[0]),
                    "positive_probability": float(scores[1]),
                    "negative_cosine_distance": float(distance[0]),
                    "positive_cosine_distance": float(distance[1]),
                    "maturity": maturity_value,
                })

    def describe(selected: list[dict[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"pairs": 0, "paired_win_rate": None, "auroc": None}
        targets = [False] * len(selected) + [True] * len(selected)
        scores = [row["negative_probability"] for row in selected] + [row["positive_probability"] for row in selected]
        distances = [row["negative_cosine_distance"] for row in selected] + [row["positive_cosine_distance"] for row in selected]
        return {
            "pairs": len(selected),
            "paired_win_rate": sum(row["positive_probability"] > row["negative_probability"] for row in selected) / len(selected),
            "paired_tie_rate": sum(row["positive_probability"] == row["negative_probability"] for row in selected) / len(selected),
            "score": score_binary(targets, scores),
            "cosine_distance": score_binary(targets, distances),
            "median_probability": {
                "same_run": statistics.median(row["negative_probability"] for row in selected),
                "adjacent_run": statistics.median(row["positive_probability"] for row in selected),
            },
        }

    return {
        "schema_version": 1,
        "source_split": "validation",
        "test_evaluation_allowed": False,
        **{
            position: {
                **describe(items),
                "by_sensor": {
                    sensor: describe([row for row in items if row["sensor_id"] == sensor])
                    for sensor in sorted({row["sensor_id"] for row in items})
                },
            }
            for position, items in rows.items()
        },
        "caveat": "Direct GT-run oracle pairs use matched maturity and do not represent online Candidate selection.",
    }


@torch.inference_mode()
def audit_validation_records(
    runtime: Any, records: tuple[Any, ...], thresholds: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay only explicitly supplied validation cache records with frozen heads."""

    if not records or any(item.split != "validation" for item in records):
        raise ValueError("upstream audit accepts only nonempty validation records")
    _apply_thresholds(runtime, thresholds)
    controller = runtime.controller
    parameter = next(controller.semantic_projection.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    physical: list[dict[str, Any]] = []
    semantic: list[dict[str, Any]] = []
    with ControllerStepTap(controller) as tap:
        for item in records:
            controller.reset()
            boundaries, _ = _truth(item)
            proposal_index: int | None = None
            proposal_timestamp: float | None = None
            candidate_start = 0
            oracle_keys = torch.from_numpy(item.record.oracle_keys.astype("float32")).to(device)
            run_ids = item.record.oracle_run_id
            sensor = str(item.metadata.get("sensor_id", ""))
            for index in range(len(item.record.radar_tokens)):
                timestamp = float(item.record.timestamps[index].reshape(-1)[0])
                tokens = torch.from_numpy(item.record.radar_tokens[index].astype("float32")).unsqueeze(0).to(device)
                mask = torch.from_numpy(item.record.radar_mask[index].astype("bool")).unsqueeze(0).to(device)
                eligible = controller.active is not None and controller.candidate is None
                observed = tap.step(tokens, mask, timestamp, item.metadata)
                output = observed.output
                distance = min((abs(timestamp - boundary) for boundary in boundaries), default=None)
                attribution = (
                    proposal_attribution(observed.local_probability or 0.0, observed.radar_probability, thresholds.proposal)
                    if output.decision == "proposal" else None
                )
                physical.append({
                    "recording_key": item.key, "sensor_id": sensor, "update_index": index,
                    "timestamp": timestamp, "eligible": eligible, "boundary_distance_s": distance,
                    "local_probability": observed.local_probability,
                    "radar_probability": observed.radar_probability,
                    "proposal_probability": float(output.proposal_probability),
                    "decision": output.decision, "attribution": attribution,
                })
                if output.decision == "proposal":
                    proposal_index = index
                    proposal_timestamp = timestamp
                    candidate_start = len(semantic)
                if observed.semantic_active is not None:
                    if (proposal_index is None or proposal_timestamp is None
                            or observed.semantic_candidate is None or observed.semantic_maturity is None):
                        raise RuntimeError("semantic score was emitted without an active Candidate")
                    reference_index = proposal_index - 1
                    reference_run = int(run_ids[reference_index]) if reference_index >= 0 else -1
                    current_run = int(run_ids[index])
                    row: dict[str, Any] = {
                        "candidate_id": f"{item.key}:{proposal_index}", "recording_key": item.key,
                        "sensor_id": sensor, "proposal_index": proposal_index,
                        "update_index": index, "age_updates": index - proposal_index + 1,
                        "age_s": timestamp - proposal_timestamp,
                        "timestamp": timestamp, "boundary_distance_s": distance,
                        "reference_run_id": reference_run, "current_run_id": current_run,
                        "online_logit": observed.semantic_logit,
                        "online_probability": float(output.difference_probability),
                        "online_cosine_probability": normalized_cosine_difference(
                            observed.semantic_active, observed.semantic_candidate
                        ),
                        "online_maturity": observed.semantic_maturity,
                        "oracle_maturity": float(item.record.oracle_maturity[index]),
                        "eventual_gt_run_switch": None,
                    }
                    if reference_run >= 0 and current_run >= 0:
                        compared = semantic_counterfactuals(
                            controller.semantic_difference,
                            online_reference=observed.semantic_active,
                            online_candidate=observed.semantic_candidate,
                            oracle_reference=oracle_keys[reference_index : reference_index + 1],
                            oracle_candidate=oracle_keys[index : index + 1],
                            online_maturity=observed.semantic_maturity,
                            oracle_maturity=float(item.record.oracle_maturity[index]),
                        )
                        if not math.isclose(compared["online_logit"], observed.semantic_logit, abs_tol=1e-4):
                            raise RuntimeError("counterfactual online pair does not reproduce observed semantic logit")
                        row.update(compared)
                    semantic.append(row)
                if output.decision in {"confirm", "reject"} and proposal_index is not None:
                    reference_run = int(run_ids[proposal_index - 1]) if proposal_index > 0 else -1
                    terminal_run = int(run_ids[index])
                    target = reference_run != terminal_run if reference_run >= 0 and terminal_run >= 0 else None
                    for row in semantic[candidate_start:]:
                        row["eventual_gt_run_switch"] = target
                    proposal_index = None
                    proposal_timestamp = None
            if proposal_index is not None:
                reference_run = int(run_ids[proposal_index - 1]) if proposal_index > 0 else -1
                terminal_run = int(run_ids[-1])
                target = reference_run != terminal_run if reference_run >= 0 and terminal_run >= 0 else None
                for row in semantic[candidate_start:]:
                    row["eventual_gt_run_switch"] = target
    return physical, semantic


def main() -> None:
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
    stage3 = dict(_load_yaml(args.config).get("stage3", {}))
    records, fingerprint = _load_cache_items(
        Path(str(stage3["cache_dir"])), Path(str(stage3["stage2_checkpoint"])), splits=("validation",)
    )
    runtime = build_runtime(args.config, args.checkpoint, torch.device(args.device))
    physical, semantic = audit_validation_records(runtime, records["validation"], thresholds)
    report = summarize_upstream_rows(physical, semantic, proposal_threshold=thresholds.proposal)
    report["semantic_cosine_age"] = semantic_cosine_age_diagnostic(semantic)
    report.update({
        "recordings": len(records["validation"]), "cache_fingerprint": fingerprint,
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": _file_sha256(args.checkpoint),
        "threshold_artifact": str(args.selected_thresholds),
        "threshold_artifact_sha256": _file_sha256(args.selected_thresholds),
    })
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (("physical-updates.jsonl", physical), ("semantic-candidates.jsonl", semantic)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (args.output_dir / "upstream-summary.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "recordings": report["recordings"]}, sort_keys=True))


if __name__ == "__main__":
    main()

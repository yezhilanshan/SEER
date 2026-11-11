"""Throwaway, validation-only Stage 3 Candidate forced-horizon probe.

This module is diagnostic; it never changes production controller semantics.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from continuous_mmwave.parrad_stage3_candidate_audit import _truth, score_binary
from continuous_mmwave.parrad_stage3_hard_negative import _apply_thresholds
from continuous_mmwave.parrad_stage3_upstream_audit import ControllerStepTap, normalized_cosine_difference


SHADOW_FEATURES = (
    "d_active_candidate", "d_active_shadow", "d_candidate_shadow",
    "candidate_vs_active_drift", "candidate_shadow_residual_change",
)


def candidate_shadow_features(
    active: torch.Tensor, candidate: torch.Tensor, shadow: torch.Tensor,
    initial_candidate: torch.Tensor, initial_shadow: torch.Tensor,
    *, previous: dict[str, float | None] | None, delta_s: float | None,
) -> dict[str, float | None]:
    """Causal, non-learned features of a single forked Candidate/Shadow pair."""

    ac = normalized_cosine_difference(active, candidate)
    a_s = normalized_cosine_difference(active, shadow)
    cs = normalized_cosine_difference(candidate, shadow)
    vectors = (candidate, shadow, initial_candidate, initial_shadow)
    residual: float | None = None
    if all(bool(torch.isfinite(value).all()) and float(torch.linalg.vector_norm(value)) > 0 for value in vectors):
        normalized = [torch.nn.functional.normalize(value, dim=-1) for value in vectors]
        residual = float(torch.linalg.vector_norm(
            (normalized[0] - normalized[1]) - (normalized[2] - normalized[3])
        ) / 4.0)
    result: dict[str, float | None] = {
        "d_active_candidate": ac,
        "d_active_shadow": a_s,
        "d_candidate_shadow": cs,
        "candidate_vs_active_drift": ac - a_s if ac is not None and a_s is not None else None,
        "candidate_shadow_residual_change": residual,
    }
    for name in SHADOW_FEATURES:
        old = previous.get(name) if previous is not None else None
        value = result[name]
        result[f"slope_{name}"] = (
            (value - old) / delta_s
            if value is not None and old is not None and delta_s is not None and delta_s > 0 else None
        )
    return result


def clean_proposal_label(
    boundaries: list[float] | tuple[float, ...], *, proposal_time: float,
    horizon_s: float, tolerance_s: float,
) -> bool | None:
    """Keep isolated near-Proposal boundaries and boundary-free trajectories only."""

    in_window = [value for value in boundaries if proposal_time - tolerance_s <= value <= proposal_time + horizon_s]
    if not in_window:
        return False
    if len(in_window) == 1 and abs(in_window[0] - proposal_time) <= tolerance_s:
        return True
    return None


def fork_candidate_controller(source: Any, *, horizon_updates: int) -> Any:
    """Copy a just-proposed Candidate and suppress termination in the copy only."""

    if horizon_updates < 1 or source.candidate is None or source.reference is None:
        raise ValueError("forced-horizon fork requires a Candidate at its Proposal")
    if source.candidate.age_updates != 1:
        raise ValueError("forced-horizon fork must occur on the Proposal update")
    forced = copy.copy(source)
    forced.decision_trace = list(source.decision_trace)
    forced.confirm_persistence = max(source.confirm_persistence, horizon_updates + 1)
    forced.reject_persistence = max(source.reject_persistence, horizon_updates + 1)
    return forced


def validate_baseline_context(
    baseline: dict[str, Any], *, checkpoint_sha256: str,
    threshold_sha256: str, cache_fingerprint: str,
) -> None:
    """Refuse to compare trajectories from different models or validation caches."""

    if baseline.get("source_split") != "validation" or baseline.get("test_evaluation_allowed") is not False:
        raise ValueError("baseline audit is not validation-only")
    for label, field, expected in (
        ("checkpoint", "checkpoint_sha256", checkpoint_sha256),
        ("threshold", "threshold_artifact_sha256", threshold_sha256),
        ("cache", "cache_fingerprint", cache_fingerprint),
    ):
        if baseline.get(field) != expected:
            raise ValueError(f"baseline {label} context mismatch")


def audit_forced_horizon(
    runtime: Any, records: tuple[Any, ...], thresholds: Any, baseline_audit_dir: Path,
    *, horizon_updates: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Replay natural decisions unchanged and inspect independent forced branches."""

    if not records or any(item.split != "validation" for item in records):
        raise ValueError("forced-horizon probe accepts only nonempty validation records")
    if horizon_updates < 1:
        raise ValueError("horizon_updates must be positive")
    summary = json.loads((baseline_audit_dir / "upstream-summary.json").read_text(encoding="utf-8"))
    if summary.get("source_split") != "validation" or summary.get("test_evaluation_allowed") is not False:
        raise ValueError("baseline audit must be validation-only")

    def load_rows(name: str) -> list[dict[str, Any]]:
        with (baseline_audit_dir / name).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    physical = {
        (str(row["recording_key"]), int(row["update_index"])): row
        for row in load_rows("physical-updates.jsonl")
    }
    semantic = {
        (str(row["candidate_id"]), int(row["update_index"])): row
        for row in load_rows("semantic-candidates.jsonl")
    }
    if len(physical) != sum(len(item.record.radar_tokens) for item in records):
        raise ValueError("baseline physical rows do not cover supplied validation records")
    _apply_thresholds(runtime, thresholds)
    controller = runtime.controller
    parameter = next(controller.semantic_projection.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    forced_rows: list[dict[str, Any]] = []
    matched_semantic: set[tuple[str, int]] = set()
    proposals = truncated = checked = 0
    for item in records:
        controller.reset()
        record = item.record
        boundaries, _ = _truth(item)
        run_ids = record.oracle_run_id
        oracle_keys = torch.from_numpy(record.oracle_keys.astype("float32")).to(device)

        def inputs(index: int) -> tuple[torch.Tensor, torch.Tensor, float]:
            return (
                torch.from_numpy(record.radar_tokens[index].astype("float32")).unsqueeze(0).to(device),
                torch.from_numpy(record.radar_mask[index].astype("bool")).unsqueeze(0).to(device),
                float(record.timestamps[index].reshape(-1)[0]),
            )

        for proposal_index in range(len(record.radar_tokens)):
            tokens, mask, timestamp = inputs(proposal_index)
            output = controller.step_tokens(tokens, mask, timestamp, item.metadata)
            expected = physical.get((item.key, proposal_index))
            if expected is None or output.decision != expected["decision"] or not math.isclose(
                output.proposal_probability, float(expected["proposal_probability"]), abs_tol=1e-5
            ):
                raise RuntimeError(f"natural controller diverged at {item.key}:{proposal_index}")
            checked += 1
            if output.decision != "proposal":
                continue
            proposals += 1
            candidate_id = f"{item.key}:{proposal_index}"
            reference_index = proposal_index - 1
            reference_run = int(run_ids[reference_index]) if reference_index >= 0 else -1
            boundary_distance = min((abs(timestamp - boundary) for boundary in boundaries), default=None)
            near_boundary = boundary_distance is not None and boundary_distance <= 0.72
            probe = fork_candidate_controller(controller, horizon_updates=horizon_updates)
            if probe.reference is None or probe.candidate is None or probe.shadow is None:
                raise RuntimeError("fork lost Candidate/Shadow state")
            active_key = probe.reference.semantic_key
            initial_candidate = probe.candidate.key_history[0]
            with torch.inference_mode():
                initial_shadow = probe.semantic_projection(probe.shadow.event_tokens).detach().clone()
            previous_features = candidate_shadow_features(
                active_key, initial_candidate, initial_shadow,
                initial_candidate, initial_shadow, previous=None, delta_s=None,
            )
            previous_time = timestamp
            clean_target = clean_proposal_label(
                boundaries, proposal_time=timestamp,
                horizon_s=horizon_updates * probe.update_period_s, tolerance_s=0.72,
            )
            last_index = min(proposal_index + horizon_updates, len(record.radar_tokens) - 1)
            if last_index < proposal_index + horizon_updates:
                truncated += 1
            with ControllerStepTap(probe) as tap:
                for index in range(proposal_index + 1, last_index + 1):
                    future_tokens, future_mask, future_time = inputs(index)
                    observed = tap.step(future_tokens, future_mask, future_time, item.metadata)
                    if (observed.output.decision != "none" or observed.semantic_active is None
                            or observed.semantic_candidate is None):
                        raise RuntimeError("forced Candidate terminated or omitted a semantic score")
                    if probe.shadow is None:
                        raise RuntimeError("forced Candidate lost Shadow state")
                    with torch.inference_mode():
                        shadow_key = probe.semantic_projection(probe.shadow.event_tokens).detach().clone()
                    features = candidate_shadow_features(
                        observed.semantic_active, observed.semantic_candidate, shadow_key,
                        initial_candidate, initial_shadow,
                        previous=previous_features, delta_s=future_time - previous_time,
                    )
                    previous_features, previous_time = features, future_time
                    current_run = int(run_ids[index])
                    row: dict[str, Any] = {
                        "candidate_id": candidate_id, "recording_key": item.key,
                        "sensor_id": str(item.metadata.get("sensor_id", "")),
                        "proposal_index": proposal_index, "proposal_time": timestamp,
                        "update_index": index, "timestamp": future_time,
                        "age_updates": index - proposal_index + 1,
                        "age_s": future_time - timestamp,
                        "proposal_boundary_distance_s": boundary_distance,
                        "proposal_near_boundary_0p72s": near_boundary,
                        "clean_proposal_target": clean_target,
                        "reference_run_id": reference_run, "current_run_id": current_run,
                        "current_run_switch": (
                            reference_run != current_run if reference_run >= 0 and current_run >= 0 else None
                        ),
                        "online_cosine_probability": normalized_cosine_difference(
                            observed.semantic_active, observed.semantic_candidate
                        ),
                        "online_head_probability": float(observed.output.difference_probability),
                        "naturally_observed": (candidate_id, index) in semantic,
                        "online_reference_to_oracle_probability": (
                            normalized_cosine_difference(
                                observed.semantic_active, oracle_keys[reference_index : reference_index + 1]
                            ) if reference_index >= 0 else None
                        ),
                        "online_candidate_to_oracle_probability": normalized_cosine_difference(
                            observed.semantic_candidate, oracle_keys[index : index + 1]
                        ),
                        **features,
                    }
                    old = semantic.get((candidate_id, index))
                    if old is not None:
                        old_cosine = old.get("online_cosine_probability")
                        new_cosine = row["online_cosine_probability"]
                        cosine_matches = (
                            old_cosine is None and new_cosine is None
                        ) or (
                            old_cosine is not None and new_cosine is not None
                            and math.isclose(float(new_cosine), float(old_cosine), abs_tol=1e-5)
                        )
                        if (not math.isclose(row["online_head_probability"], float(old["online_probability"]), abs_tol=1e-5)
                                or not cosine_matches):
                            raise RuntimeError(f"forced branch diverged before natural termination at {candidate_id}:{index}")
                        matched_semantic.add((candidate_id, index))
                    forced_rows.append(row)
    if matched_semantic != set(semantic):
        raise RuntimeError("forced branch did not reproduce every natural Candidate score")
    return forced_rows, {
        "baseline_decisions_checked": checked,
        "baseline_semantic_scores_checked": len(matched_semantic),
        "forced_candidates": proposals,
        "truncated_candidates": truncated,
    }


def summarize_forced_rows(
    rows: list[dict[str, Any]], *, horizon_updates: int, update_period_s: float
) -> dict[str, Any]:
    """Show age ranking on all observed and full-horizon matched Candidates."""

    if horizon_updates < 1 or not math.isfinite(update_period_s) or update_period_s <= 0:
        raise ValueError("forced horizon and update period must be positive")
    candidate_ids = {str(row["candidate_id"]) for row in rows}
    full_horizon = {
        str(row["candidate_id"]) for row in rows if int(row["age_updates"]) == horizon_updates + 1
    }

    def describe(selected: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "candidate_count": len(selected),
            "naturally_observed_candidates": sum(bool(row["naturally_observed"]) for row in selected),
        }
        for name, field in (
            ("proposal_near_boundary", "proposal_near_boundary_0p72s"),
            ("current_run_switch", "current_run_switch"),
        ):
            eligible = [row for row in selected if row[field] is not None]
            scores: dict[str, Any] = {}
            for label, key in (
                ("cosine", "online_cosine_probability"),
                ("trained_head", "online_head_probability"),
            ):
                valid = [row for row in eligible if row[key] is not None]
                scores[label] = score_binary(
                    [bool(row[field]) for row in valid], [float(row[key]) for row in valid]
                )
            scores["cosine_by_sensor"] = {
                sensor: score_binary(
                    [bool(row[field]) for row in eligible if row["sensor_id"] == sensor and row["online_cosine_probability"] is not None],
                    [float(row["online_cosine_probability"]) for row in eligible if row["sensor_id"] == sensor and row["online_cosine_probability"] is not None],
                )
                for sensor in sorted({str(row["sensor_id"]) for row in eligible})
            }
            result[name] = scores
        feature_names = SHADOW_FEATURES + tuple(f"slope_{name}" for name in SHADOW_FEATURES)
        clean = [row for row in selected if row.get("clean_proposal_target") is not None]

        def clean_scores(group: list[dict[str, Any]]) -> dict[str, Any]:
            output: dict[str, Any] = {
                "count": len(group), "positives": sum(row["clean_proposal_target"] is True for row in group),
            }
            for feature in feature_names:
                valid = [row for row in group if row.get(feature) is not None]
                output[feature] = score_binary(
                    [bool(row["clean_proposal_target"]) for row in valid],
                    [float(row[feature]) for row in valid],
                )
            return output

        clean_result = clean_scores(clean)
        clean_result["by_sensor"] = {
            sensor: clean_scores([row for row in clean if str(row["sensor_id"]) == sensor])
            for sensor in sorted({str(row["sensor_id"]) for row in selected})
        }
        clean_result["by_recording"] = {
            recording: clean_scores([row for row in clean if str(row["recording_key"]) == recording])
            for recording in sorted({str(row["recording_key"]) for row in selected if "recording_key" in row})
        }
        result["clean_proposal"] = clean_result
        result["proposal_near_boundary"]["shadow_features"] = {
            feature: score_binary(
                [bool(row["proposal_near_boundary_0p72s"]) for row in selected if row.get(feature) is not None],
                [float(row[feature]) for row in selected if row.get(feature) is not None],
            )
            for feature in feature_names
        }
        result["median_online_to_oracle_cosine_probability"] = {
            label: (
                statistics.median(float(row[key]) for row in selected if row[key] is not None)
                if any(row[key] is not None for row in selected) else None
            )
            for label, key in (
                ("reference", "online_reference_to_oracle_probability"),
                ("candidate", "online_candidate_to_oracle_probability"),
            )
        }
        return result

    def curve(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            str(age_updates): {
                "nominal_age_s": round((age_updates - 1) * update_period_s, 6),
                **describe([row for row in selected if int(row["age_updates"]) == age_updates]),
            }
            for age_updates in range(2, horizon_updates + 2)
        }

    return {
        "schema_version": 1,
        "source_split": "validation",
        "test_evaluation_allowed": False,
        "total_candidates": len(candidate_ids),
        "forced_horizon_updates": horizon_updates,
        "forced_horizon_s": horizon_updates * update_period_s,
        "by_age_updates": curve(rows),
        "matched_full_horizon": {
            "candidate_count": len(full_horizon),
            "by_age_updates": curve([row for row in rows if str(row["candidate_id"]) in full_horizon]),
        },
        "caveat": (
            "Counterfactual Candidates ignore confirm and reject after a natural Proposal; they do not model "
            "new Proposals or downstream Active state changes. Proposal proximity is a timing proxy, not "
            "adjudicated semantic validity. Long forced branches can cross multiple GT runs."
        ),
    }


def main() -> None:
    from continuous_mmwave.parrad_stage3_calibrate import build_runtime
    from continuous_mmwave.parrad_stage3_hard_negative import _file_sha256, _load_thresholds, _load_yaml
    from continuous_mmwave.parrad_stage3_runner import _load_cache_items

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selected-thresholds", type=Path, required=True)
    parser.add_argument("--baseline-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon-updates", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    thresholds = _load_thresholds(args.selected_thresholds, args.checkpoint)
    stage3 = dict(_load_yaml(args.config).get("stage3", {}))
    records, fingerprint = _load_cache_items(
        Path(str(stage3["cache_dir"])), Path(str(stage3["stage2_checkpoint"])),
        splits=("validation",),
    )
    baseline = json.loads((args.baseline_audit_dir / "upstream-summary.json").read_text(encoding="utf-8"))
    validate_baseline_context(
        baseline,
        checkpoint_sha256=_file_sha256(args.checkpoint),
        threshold_sha256=_file_sha256(args.selected_thresholds),
        cache_fingerprint=fingerprint,
    )
    runtime = build_runtime(args.config, args.checkpoint, torch.device(args.device))
    rows, integrity = audit_forced_horizon(
        runtime, records["validation"], thresholds, args.baseline_audit_dir,
        horizon_updates=args.horizon_updates,
    )
    report = summarize_forced_rows(
        rows, horizon_updates=args.horizon_updates,
        update_period_s=runtime.controller.update_period_s,
    )
    report.update({
        "diagnostic_only": True,
        "recordings": len(records["validation"]),
        "cache_fingerprint": fingerprint,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _file_sha256(args.checkpoint),
        "threshold_artifact": str(args.selected_thresholds),
        "threshold_artifact_sha256": _file_sha256(args.selected_thresholds),
        "baseline_audit_dir": str(args.baseline_audit_dir),
        "baseline_summary_sha256": _file_sha256(args.baseline_audit_dir / "upstream-summary.json"),
        "probe_code_sha256": _file_sha256(Path(__file__)),
        "integrity": integrity,
    })
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "forced-candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (args.output_dir / "forced-summary.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "integrity": integrity}, sort_keys=True))


if __name__ == "__main__":
    main()

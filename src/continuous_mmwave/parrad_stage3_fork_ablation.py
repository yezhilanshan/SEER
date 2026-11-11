"""Validation-only, frozen-weight 2x2 Fork initialization/gating ablation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from continuous_mmwave.parrad_stage3_candidate_audit import _truth, score_binary
from continuous_mmwave.parrad_stage3_forced_horizon_probe import (
    clean_proposal_label, fork_candidate_controller, validate_baseline_context,
)
from continuous_mmwave.parrad_stage3_hard_negative import _apply_thresholds
from continuous_mmwave.parrad_stage3_shadow_hard_pair import summarize_same_recording_hard_pairs
from continuous_mmwave.parrad_stage3_upstream_audit import normalized_cosine_difference


ARM_NAMES = ("pp", "pf", "ap", "af")
SCORE_FIELDS = (
    "a_to_pp", "a_to_pf", "a_to_ap", "a_to_af", "one_minus_a_to_af",
    "pp_to_af", "ap_to_af", "pp_to_pf", "pp_to_ap", "pf_to_af",
)
LAYER_FEATURE_PATHS = {
    **{f"{arm}_gate_mean": ("gate_stats", arm, "mean") for arm in ARM_NAMES},
    **{
        f"{pair}_{metric}": ("pair_geometry", pair, metric)
        for pair in ("pp_pf", "ap_af", "pp_ap", "pf_af", "pp_af")
        for metric in ("event_l2", "event_cosine", "key_l2", "key_cosine", "gate_abs_mean")
    },
    "radar_change_l2": ("radar_change", "l2"),
    "radar_change_cosine": ("radar_change", "cosine"),
    "physical_proposal_probability": ("physical_proposal_probability",),
}
REPRODUCTION_TOLERANCE = 1e-5


class ForkTrajectories:
    """Four states: P_base/Active initialization x progressive/full maturity."""

    def __init__(
        self, event_former: Any, projection: Any, base: torch.Tensor,
        active: torch.Tensor, prompt_cache: Any,
    ) -> None:
        if base.shape != active.shape or base.shape[0] != 1:
            raise ValueError("Fork initial states must share batch-one shape")
        self.event_former = event_former
        self.projection = projection
        self.prompt_cache = prompt_cache
        with torch.inference_mode():
            self.initial_events = torch.cat((base, base, active, active), dim=0).clone()
            self.initial_keys = torch.cat(
                [projection(self.initial_events[index : index + 1]) for index in range(4)], dim=0
            ).clone()
            self.events = self.initial_events.clone()
            self.keys = self.initial_keys.clone()
            self.gates: torch.Tensor | None = None

    @torch.inference_mode()
    def step(self, radar: torch.Tensor, mask: torch.Tensor, *, progressive_maturity: float) -> None:
        if radar.shape[0] != 1 or mask.shape != radar.shape[:2] or not 0 < progressive_maturity <= 1:
            raise ValueError("Fork step requires batch-one radar and maturity in (0,1]")
        maturities = (progressive_maturity, 1.0, progressive_maturity, 1.0)
        outputs = [
            self.event_former.stream_update(
                self.events[index : index + 1], radar, maturity, mask, self.prompt_cache,
            )
            for index, maturity in enumerate(maturities)
        ]
        updated = [output.event_tokens for output in outputs]
        self.events = torch.cat(updated, dim=0)
        self.keys = torch.cat([self.projection(state) for state in updated], dim=0)
        gates = [getattr(output, "gates", None) for output in outputs]
        self.gates = torch.cat(gates, dim=0) if all(gate is not None for gate in gates) else None
        if not bool(torch.isfinite(self.events).all()) or not bool(torch.isfinite(self.keys).all()):
            raise RuntimeError("nonfinite Fork trajectory")
        if self.gates is not None and not bool(torch.isfinite(self.gates).all()):
            raise RuntimeError("nonfinite Fork gates")

    def measure_gate(self, name: str) -> dict[str, float]:
        if self.gates is None:
            raise RuntimeError("Event Former did not expose gate activations")
        gate = self.gates[ARM_NAMES.index(name)]
        return {"mean": float(gate.float().mean()), "min": float(gate.min()), "max": float(gate.max())}

    def pair_geometry(self, left: str, right: str) -> dict[str, float | None]:
        left_index, right_index = ARM_NAMES.index(left), ARM_NAMES.index(right)
        event_left, event_right = self.events[left_index : left_index + 1], self.events[right_index : right_index + 1]
        key_left, key_right = self.keys[left_index : left_index + 1], self.keys[right_index : right_index + 1]
        event_left_flat, event_right_flat = event_left.reshape(1, -1), event_right.reshape(1, -1)
        result: dict[str, float | None] = {
            "event_l2": float(torch.linalg.vector_norm((event_left - event_right).float())),
            "event_cosine": normalized_cosine_difference(event_left_flat, event_right_flat),
            "key_l2": float(torch.linalg.vector_norm((key_left - key_right).float())),
            "key_cosine": normalized_cosine_difference(key_left, key_right),
        }
        result["gate_abs_mean"] = (
            float((self.gates[left_index] - self.gates[right_index]).float().abs().mean())
            if self.gates is not None else None
        )
        return result

    def measure(self, name: str) -> dict[str, float]:
        index = ARM_NAMES.index(name)
        event, key = self.events[index], self.keys[index]
        return {
            "state_norm": float(torch.linalg.vector_norm(event)),
            "key_norm": float(torch.linalg.vector_norm(key)),
            "delta_event": float(torch.linalg.vector_norm(event - self.initial_events[index])),
            "delta_key": float(torch.linalg.vector_norm(key - self.initial_keys[index])),
        }

    def reproduction_errors(
        self, *, candidate_events: torch.Tensor, candidate_key: torch.Tensor,
        shadow_events: torch.Tensor, shadow_key: torch.Tensor, tolerance: float,
    ) -> dict[str, float]:
        if tolerance <= 0:
            raise ValueError("reproduction tolerance must be positive")
        errors = {
            "candidate_event": float((self.events[0:1] - candidate_events).abs().max()),
            "candidate_key": float((self.keys[0:1] - candidate_key).abs().max()),
            "shadow_event": float((self.events[3:4] - shadow_events).abs().max()),
            "shadow_key": float((self.keys[3:4] - shadow_key).abs().max()),
        }
        if any(not math.isfinite(error) or error > tolerance for error in errors.values()):
            raise RuntimeError(f"Fork reproduction error exceeds {tolerance}: {errors}")
        return errors


def audit_fork_ablation(
    runtime: Any, records: tuple[Any, ...], thresholds: Any,
    baseline_audit_dir: Path, baseline_forced_dir: Path, *, horizon_updates: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not records or any(item.split != "validation" for item in records):
        raise ValueError("Fork ablation accepts only nonempty validation records")
    if horizon_updates < 1:
        raise ValueError("horizon_updates must be positive")
    baseline = json.loads((baseline_audit_dir / "upstream-summary.json").read_text(encoding="utf-8"))
    old_summary = json.loads((baseline_forced_dir / "forced-summary.json").read_text(encoding="utf-8"))
    if (baseline.get("source_split") != "validation" or baseline.get("test_evaluation_allowed") is not False
            or old_summary.get("source_split") != "validation"
            or old_summary.get("test_evaluation_allowed") is not False
            or old_summary.get("diagnostic_only") is not True
            or int(old_summary.get("forced_horizon_updates", -1)) != horizon_updates):
        raise ValueError("baseline audits must match validation-only Fork horizon")

    def load_rows(path: Path) -> list[dict[str, Any]]:
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    physical = {
        (str(row["recording_key"]), int(row["update_index"])): row
        for row in load_rows(baseline_audit_dir / "physical-updates.jsonl")
    }
    old_forced = {
        (str(row["candidate_id"]), int(row["update_index"])): row
        for row in load_rows(baseline_forced_dir / "forced-candidates.jsonl")
    }
    if len(physical) != sum(len(item.record.radar_tokens) for item in records):
        raise ValueError("natural baseline does not cover validation records")
    _apply_thresholds(runtime, thresholds)
    controller = runtime.controller
    parameter = next(controller.semantic_projection.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    prompt_cache1 = controller.prompt_cache
    rows: list[dict[str, Any]] = []
    checked_forced: set[tuple[str, int]] = set()
    max_errors = {name: 0.0 for name in ("candidate_event", "candidate_key", "shadow_event", "shadow_key")}
    proposals = truncated = checked_decisions = 0

    for item in records:
        controller.reset()
        record = item.record
        boundaries, _ = _truth(item)

        def inputs(index: int) -> tuple[torch.Tensor, torch.Tensor, float]:
            return (
                torch.from_numpy(record.radar_tokens[index].astype("float32")).unsqueeze(0).to(device),
                torch.from_numpy(record.radar_mask[index].astype("bool")).unsqueeze(0).to(device),
                float(record.timestamps[index].reshape(-1)[0]),
            )

        for proposal_index in range(len(record.radar_tokens)):
            tokens, mask, timestamp = inputs(proposal_index)
            output = controller.step_tokens(tokens, mask, timestamp, item.metadata)
            old = physical.get((item.key, proposal_index))
            if old is None or output.decision != old["decision"] or not math.isclose(
                output.proposal_probability, float(old["proposal_probability"]), abs_tol=1e-5,
            ):
                raise RuntimeError(f"natural controller diverged at {item.key}:{proposal_index}")
            checked_decisions += 1
            if output.decision != "proposal":
                continue
            proposals += 1
            candidate_id = f"{item.key}:{proposal_index}"
            probe = fork_candidate_controller(controller, horizon_updates=horizon_updates)
            if probe.reference is None or probe.candidate is None or probe.shadow is None:
                raise RuntimeError("Fork lost Proposal state")
            branches = ForkTrajectories(
                probe.event_former, probe.semantic_projection,
                probe.event_former.initial_state(1).clone(), probe.reference.event_tokens.clone(),
                prompt_cache1,
            )
            clean_target = clean_proposal_label(
                boundaries, proposal_time=timestamp,
                horizon_s=horizon_updates * probe.update_period_s, tolerance_s=0.72,
            )
            last_index = min(proposal_index + horizon_updates, len(record.radar_tokens) - 1)
            if last_index < proposal_index + horizon_updates:
                truncated += 1

            def radar_change(index: int) -> dict[str, float | None]:
                if index == 0:
                    return {"l2": None, "cosine": None}
                common = record.radar_mask[index].astype("bool") & record.radar_mask[index - 1].astype("bool")
                if not bool(common.any()):
                    return {"l2": None, "cosine": None}
                previous = torch.from_numpy(record.radar_tokens[index - 1, common].astype("float32")).reshape(1, -1)
                current = torch.from_numpy(record.radar_tokens[index, common].astype("float32")).reshape(1, -1)
                return {
                    "l2": float(torch.linalg.vector_norm(current - previous)),
                    "cosine": normalized_cosine_difference(previous, current),
                }

            def record_branch(index: int, time: float) -> None:
                if probe.reference is None or probe.candidate is None or probe.shadow is None:
                    raise RuntimeError("forced Fork ended unexpectedly")
                with torch.inference_mode():
                    shadow_key = probe.semantic_projection(probe.shadow.event_tokens).detach().clone()
                errors = branches.reproduction_errors(
                    candidate_events=probe.candidate.event_tokens,
                    candidate_key=probe.candidate.key_history[-1],
                    shadow_events=probe.shadow.event_tokens,
                    shadow_key=shadow_key,
                    tolerance=REPRODUCTION_TOLERANCE,
                )
                for name, value in errors.items():
                    max_errors[name] = max(max_errors[name], value)
                keys = {name: branches.keys[position : position + 1]
                        for position, name in enumerate(ARM_NAMES)}
                active_key = probe.reference.semantic_key
                scores = {f"a_to_{name}": normalized_cosine_difference(active_key, key)
                          for name, key in keys.items()}
                scores.update({
                    "pp_to_af": normalized_cosine_difference(keys["pp"], keys["af"]),
                    "ap_to_af": normalized_cosine_difference(keys["ap"], keys["af"]),
                    "pp_to_pf": normalized_cosine_difference(keys["pp"], keys["pf"]),
                    "pp_to_ap": normalized_cosine_difference(keys["pp"], keys["ap"]),
                    "pf_to_af": normalized_cosine_difference(keys["pf"], keys["af"]),
                })
                scores["one_minus_a_to_af"] = (
                    1.0 - scores["a_to_af"] if scores["a_to_af"] is not None else None
                )
                row: dict[str, Any] = {
                    "candidate_id": candidate_id, "recording_key": item.key,
                    "sensor_id": str(item.metadata.get("sensor_id", "")),
                    "session_key": item.key.rsplit("__", 1)[0] if "__" in item.key else item.key,
                    "proposal_index": proposal_index, "proposal_time": timestamp,
                    "update_index": index, "timestamp": time,
                    "age_updates": index - proposal_index + 1,
                    "age_s": time - timestamp,
                    "clean_proposal_target": clean_target,
                    "reproduction_max_abs_error": errors,
                    "arms": {name: branches.measure(name) for name in ARM_NAMES},
                    "gate_stats": {name: branches.measure_gate(name) for name in ARM_NAMES},
                    "pair_geometry": {
                        f"{left}_{right}": branches.pair_geometry(left, right)
                        for left, right in (("pp", "pf"), ("ap", "af"), ("pp", "ap"),
                                            ("pf", "af"), ("pp", "af"))
                    },
                    "radar_change": radar_change(index),
                    "physical_proposal_probability": float(output.proposal_probability),
                    **scores,
                }
                if index > proposal_index:
                    previous = old_forced.get((candidate_id, index))
                    if previous is None:
                        raise RuntimeError(f"missing prior forced row {candidate_id}:{index}")
                    old_ac = previous.get("d_active_candidate")
                    old_cs = previous.get("d_candidate_shadow")
                    current_ac = normalized_cosine_difference(active_key, probe.candidate.key_history[-1])
                    current_cs = normalized_cosine_difference(probe.candidate.key_history[-1], shadow_key)
                    for label, expected, actual in (
                        ("active-candidate", old_ac, current_ac),
                        ("candidate-shadow", old_cs, current_cs),
                    ):
                        if (expected is None) != (actual is None) or (
                            expected is not None and not math.isclose(float(expected), float(actual), abs_tol=1e-5)
                        ):
                            raise RuntimeError(f"prior forced {label} diverged at {candidate_id}:{index}")
                    checked_forced.add((candidate_id, index))
                rows.append(row)

            branches.step(tokens, mask, progressive_maturity=min(probe.update_period_s / probe.maturity_seconds, 1.0))
            record_branch(proposal_index, timestamp)
            for index in range(proposal_index + 1, last_index + 1):
                future_tokens, future_mask, future_time = inputs(index)
                forced_output = probe.step_tokens(future_tokens, future_mask, future_time, item.metadata)
                if forced_output.decision != "none" or probe.candidate is None:
                    raise RuntimeError("forced Candidate terminated before horizon")
                maturity = min(probe.candidate.age_updates * probe.update_period_s / probe.maturity_seconds, 1.0)
                branches.step(future_tokens, future_mask, progressive_maturity=maturity)
                record_branch(index, future_time)
    if checked_forced != set(old_forced):
        raise RuntimeError("Fork ablation did not cover every prior forced row")
    return rows, {
        "baseline_decisions_checked": checked_decisions,
        "baseline_forced_rows_checked": len(checked_forced),
        "forced_candidates": proposals,
        "truncated_candidates": truncated,
        "max_abs_reproduction_error": max_errors,
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
    }


def summarize_fork_rows(
    rows: list[dict[str, Any]], *, horizon_updates: int, update_period_s: float,
) -> dict[str, Any]:
    if horizon_updates < 1 or not math.isfinite(update_period_s) or update_period_s <= 0:
        raise ValueError("Fork summary requires a positive horizon and update period")
    full_ids = {str(row["candidate_id"]) for row in rows if int(row["age_updates"]) == horizon_updates + 1}

    def score_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for feature in SCORE_FIELDS:
            valid = [row for row in group if row.get(feature) is not None]
            result[feature] = score_binary(
                [bool(row["clean_proposal_target"]) for row in valid],
                [float(row[feature]) for row in valid],
            )
        return result

    def describe(group: list[dict[str, Any]]) -> dict[str, Any]:
        clean = [row for row in group if row["clean_proposal_target"] is not None]
        result: dict[str, Any] = {
            "candidate_count": len(group),
            "clean_count": len(clean),
            "clean_positives": sum(row["clean_proposal_target"] is True for row in clean),
            "scores": score_group(clean),
        }
        for key, output_name in (("sensor_id", "by_sensor"), ("recording_key", "by_recording"),
                                 ("session_key", "by_session")):
            result[output_name] = {
                name: score_group([row for row in clean if str(row[key]) == name])
                for name in sorted({str(row[key]) for row in group})
            }
        drift: dict[str, Any] = {}
        for arm in ARM_NAMES:
            drift[arm] = {}
            for feature in ("delta_event", "delta_key", "state_norm", "key_norm"):
                values = [float(row["arms"][arm][feature]) for row in group]
                if any(not math.isfinite(value) for value in values):
                    raise ValueError(f"nonfinite Fork {arm} {feature}")
                drift[arm][feature] = {
                    "median": statistics.median(values) if values else None,
                    "positive_median": statistics.median(
                        float(row["arms"][arm][feature]) for row in clean
                        if row["clean_proposal_target"] is True
                    ) if any(row["clean_proposal_target"] is True for row in clean) else None,
                    "negative_median": statistics.median(
                        float(row["arms"][arm][feature]) for row in clean
                        if row["clean_proposal_target"] is False
                    ) if any(row["clean_proposal_target"] is False for row in clean) else None,
                }
        result["arm_medians"] = drift
        return result

    def age_curve(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            str(age): {
                "nominal_age_s": round((age - 1) * update_period_s, 6),
                **describe([row for row in selected if int(row["age_updates"]) == age]),
            }
            for age in range(1, horizon_updates + 2)
        }

    full_rows = [row for row in rows if str(row["candidate_id"]) in full_ids]
    return {
        "schema_version": 2,
        "source_split": "validation",
        "test_evaluation_allowed": False,
        "diagnostic_only": True,
        "arm_names": {"pp": "P_base+progressive", "pf": "P_base+full",
                      "ap": "Active+progressive", "af": "Active+full"},
        "score_direction": "Higher raw cosine distance means stronger transition; only one_minus_a_to_af is reversed, predeclared for stability checking.",
        "total_candidates": len({str(row["candidate_id"]) for row in rows}),
        "by_age_updates": age_curve(rows),
        "matched_full_horizon": {
            "candidate_count": len(full_ids),
            "by_age_updates": age_curve(full_rows),
        },
        "same_recording_hard_pairs": summarize_same_recording_hard_pairs(
            full_rows, horizon_updates=horizon_updates, features=SCORE_FIELDS,
        ),
        "caveat": (
            "Forced branches are counterfactual after each natural Proposal; they do not generate new downstream Proposals. "
            "Clean labels use GT boundary timing only. Existing validation sessions were used to discover the inverse "
            "Active-Shadow direction, so session analysis is descriptive rather than independent confirmation."
        ),
    }


def summarize_fork_layers(
    rows: list[dict[str, Any]], *, horizon_updates: int, update_period_s: float,
) -> dict[str, Any]:
    """Score predeclared input, gate, Event State and Key measurements on one matched cohort."""
    if horizon_updates < 1 or not math.isfinite(update_period_s) or update_period_s <= 0:
        raise ValueError("layer summary requires a positive horizon and update period")
    full_ids = {str(row["candidate_id"]) for row in rows if int(row["age_updates"]) == horizon_updates + 1}

    def score_group(group: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, path in LAYER_FEATURE_PATHS.items():
            selected = []
            for row in group:
                value: Any = row
                for part in path:
                    value = value[part]
                if value is not None:
                    selected.append((bool(row["clean_proposal_target"]), float(value)))
            result[name] = score_binary(
                [target for target, _ in selected], [value for _, value in selected],
            )
        return result

    full_rows = [row for row in rows if str(row["candidate_id"]) in full_ids]
    by_age: dict[str, Any] = {}
    for age in range(1, horizon_updates + 2):
        age_rows = [row for row in full_rows if int(row["age_updates"]) == age]
        clean = [row for row in age_rows if row["clean_proposal_target"] is not None]
        by_age[str(age)] = {
            "nominal_age_s": round((age - 1) * update_period_s, 6),
            "candidate_count": len(age_rows),
            "clean_count": len(clean),
            "clean_positives": sum(row["clean_proposal_target"] is True for row in clean),
            "scores": score_group(clean),
            "by_session": {
                key: score_group([row for row in clean if str(row["session_key"]) == key])
                for key in sorted({str(row["session_key"]) for row in age_rows})
            },
            "by_sensor": {
                key: score_group([row for row in clean if str(row["sensor_id"]) == key])
                for key in sorted({str(row["sensor_id"]) for row in age_rows})
            },
        }
    return {
        "source_split": "validation", "test_evaluation_allowed": False, "diagnostic_only": True,
        "matched_full_horizon": len(full_ids), "feature_paths": LAYER_FEATURE_PATHS,
        "by_age_updates": by_age,
    }


def write_fork_outputs(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    if (summary.get("source_split") != "validation"
            or summary.get("test_evaluation_allowed") is not False
            or summary.get("diagnostic_only") is not True):
        raise ValueError("Fork output must remain validation-only diagnostic data")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    serialized_rows = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    serialized_summary = json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "fork-rows.jsonl").write_text(serialized_rows, encoding="utf-8")
    (output_dir / "fork-summary.json").write_text(serialized_summary, encoding="utf-8")


def main() -> None:
    from continuous_mmwave.parrad_stage3_calibrate import build_runtime
    from continuous_mmwave.parrad_stage3_hard_negative import _file_sha256, _load_thresholds, _load_yaml
    from continuous_mmwave.parrad_stage3_runner import _load_cache_items

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selected-thresholds", type=Path, required=True)
    parser.add_argument("--baseline-audit-dir", type=Path, required=True)
    parser.add_argument("--baseline-forced-dir", type=Path, required=True)
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
    checkpoint_sha = _file_sha256(args.checkpoint)
    threshold_sha = _file_sha256(args.selected_thresholds)
    baseline_path = args.baseline_audit_dir / "upstream-summary.json"
    old_path = args.baseline_forced_dir / "forced-summary.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    old_forced = json.loads(old_path.read_text(encoding="utf-8"))
    for context in (baseline, old_forced):
        validate_baseline_context(
            context, checkpoint_sha256=checkpoint_sha,
            threshold_sha256=threshold_sha, cache_fingerprint=fingerprint,
        )
    if old_forced.get("baseline_summary_sha256") != _file_sha256(baseline_path):
        raise ValueError("prior forced probe has a different natural baseline")
    runtime = build_runtime(args.config, args.checkpoint, torch.device(args.device))
    rows, integrity = audit_fork_ablation(
        runtime, records["validation"], thresholds,
        args.baseline_audit_dir, args.baseline_forced_dir,
        horizon_updates=args.horizon_updates,
    )
    report = summarize_fork_rows(
        rows, horizon_updates=args.horizon_updates,
        update_period_s=runtime.controller.update_period_s,
    )
    report["layer_audit"] = summarize_fork_layers(
        rows, horizon_updates=args.horizon_updates,
        update_period_s=runtime.controller.update_period_s,
    )
    report.update({
        "recordings": len(records["validation"]),
        "cache_fingerprint": fingerprint,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "threshold_artifact": str(args.selected_thresholds),
        "threshold_artifact_sha256": threshold_sha,
        "baseline_audit_dir": str(args.baseline_audit_dir),
        "baseline_summary_sha256": _file_sha256(baseline_path),
        "baseline_forced_dir": str(args.baseline_forced_dir),
        "baseline_forced_summary_sha256": _file_sha256(old_path),
        "probe_code_sha256": _file_sha256(Path(__file__)),
        "integrity": integrity,
    })
    write_fork_outputs(rows, report, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "integrity": integrity}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Validation-only, no-learning same-recording Candidate/Shadow hard-pair audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from continuous_mmwave.parrad_stage3_forced_horizon_probe import SHADOW_FEATURES


def validate_shadow_source(summary: dict[str, Any]) -> None:
    if (summary.get("source_split") != "validation"
            or summary.get("test_evaluation_allowed") is not False
            or summary.get("diagnostic_only") is not True):
        raise ValueError("hard-pair audit requires a validation-only diagnostic source")


def summarize_same_recording_hard_pairs(
    rows: list[dict[str, Any]], *, horizon_updates: int,
    features: tuple[str, ...] = SHADOW_FEATURES + tuple(f"slope_{name}" for name in SHADOW_FEATURES),
) -> dict[str, Any]:
    """Pair each clean positive with its closest-in-time clean negative in the same recording."""

    if horizon_updates < 1:
        raise ValueError("horizon_updates must be positive")
    full_ids = {str(row["candidate_id"]) for row in rows if int(row["age_updates"]) == horizon_updates + 1}

    def pair_metrics(pairs: list[tuple[dict[str, Any], dict[str, Any], float]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "pair_count": len(pairs),
            "median_proposal_gap_s": statistics.median(gap for _, _, gap in pairs) if pairs else None,
        }
        for feature in features:
            valid = [(positive, negative) for positive, negative, _ in pairs
                     if positive.get(feature) is not None and negative.get(feature) is not None]
            differences = [float(positive[feature]) - float(negative[feature]) for positive, negative in valid]
            if any(not math.isfinite(value) for value in differences):
                raise ValueError(f"nonfinite hard-pair feature: {feature}")
            result[feature] = {
                "count": len(differences),
                "ties": sum(value == 0 for value in differences),
                "pair_win_rate": (
                    sum(1.0 if value > 0 else 0.5 if value == 0 else 0.0 for value in differences)
                    / len(differences) if differences else None
                ),
                "median_positive_minus_negative": statistics.median(differences) if differences else None,
            }
        return result

    by_age: dict[str, Any] = {}
    for age in range(2, horizon_updates + 2):
        selected = [row for row in rows if str(row["candidate_id"]) in full_ids
                    and int(row["age_updates"]) == age]
        recordings = sorted({str(row["recording_key"]) for row in selected})
        pairs: list[tuple[dict[str, Any], dict[str, Any], float]] = []
        for recording in recordings:
            group = [row for row in selected if str(row["recording_key"]) == recording]
            positives = [row for row in group if row.get("clean_proposal_target") is True]
            negatives = [row for row in group if row.get("clean_proposal_target") is False]
            for positive in positives:
                if not negatives:
                    continue
                negative = min(
                    negatives,
                    key=lambda row: (abs(float(row["proposal_time"]) - float(positive["proposal_time"])),
                                     float(row["proposal_time"]), str(row["candidate_id"])),
                )
                pairs.append((positive, negative,
                              abs(float(negative["proposal_time"]) - float(positive["proposal_time"]))))
        age_result = pair_metrics(pairs)
        age_result["by_recording"] = {
            name: pair_metrics([pair for pair in pairs if str(pair[0]["recording_key"]) == name])
            for name in recordings
        }
        age_result["by_sensor"] = {
            name: pair_metrics([pair for pair in pairs if str(pair[0]["sensor_id"]) == name])
            for name in sorted({str(row["sensor_id"]) for row in selected})
        }
        for feature in features:
            values = [metrics[feature]["pair_win_rate"] for metrics in age_result["by_recording"].values()
                      if metrics[feature]["pair_win_rate"] is not None]
            age_result[feature]["recording_macro_pair_win_rate"] = statistics.mean(values) if values else None
            age_result[feature]["recordings_above_half"] = sum(value > 0.5 for value in values)
        by_age[str(age)] = age_result
    return {
        "schema_version": 1,
        "source_split": "validation",
        "test_evaluation_allowed": False,
        "diagnostic_only": True,
        "full_horizon_candidate_count": len(full_ids),
        "by_age_updates": by_age,
        "method": (
            "For each full-horizon clean positive, select the temporally nearest clean negative from the "
            "same recording at the same age, with replacement; score > 0.5 means the feature ranks the "
            "positive higher. Matching uses GT only offline and is not a deployable controller rule."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary_path = args.source_dir / "forced-summary.json"
    rows_path = args.source_dir / "forced-candidates.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    validate_shadow_source(summary)
    with rows_path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    result = summarize_same_recording_hard_pairs(
        rows, horizon_updates=int(summary["forced_horizon_updates"]),
    )
    result.update({
        "source_dir": str(args.source_dir),
        "source_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "source_rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "threshold_artifact_sha256": summary["threshold_artifact_sha256"],
        "integrity": summary["integrity"],
    })
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "full_candidates": result["full_horizon_candidate_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

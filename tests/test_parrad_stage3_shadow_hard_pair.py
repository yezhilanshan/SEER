from __future__ import annotations

import importlib
import json
import sys

import pytest


def test_hard_pair_uses_nearest_negative_in_same_recording() -> None:
    diagnostic = importlib.import_module("continuous_mmwave.parrad_stage3_shadow_hard_pair")
    rows = [
        dict(candidate_id="positive-1", recording_key="rec-1", sensor_id="14xx", proposal_time=1.0,
             age_updates=3, clean_proposal_target=True, d_candidate_shadow=0.9),
        dict(candidate_id="near-negative", recording_key="rec-1", sensor_id="14xx", proposal_time=2.0,
             age_updates=3, clean_proposal_target=False, d_candidate_shadow=0.1),
        dict(candidate_id="far-negative", recording_key="rec-1", sensor_id="14xx", proposal_time=10.0,
             age_updates=3, clean_proposal_target=False, d_candidate_shadow=0.95),
        dict(candidate_id="positive-2", recording_key="rec-2", sensor_id="68xx", proposal_time=3.0,
             age_updates=3, clean_proposal_target=True, d_candidate_shadow=0.2),
        dict(candidate_id="negative-2", recording_key="rec-2", sensor_id="68xx", proposal_time=4.0,
             age_updates=3, clean_proposal_target=False, d_candidate_shadow=0.3),
    ]

    result = diagnostic.summarize_same_recording_hard_pairs(
        rows, horizon_updates=2, features=("d_candidate_shadow",),
    )["by_age_updates"]["3"]

    assert result["pair_count"] == 2
    assert result["d_candidate_shadow"]["pair_win_rate"] == pytest.approx(0.5)
    assert result["by_recording"]["rec-1"]["d_candidate_shadow"]["pair_win_rate"] == 1.0
    assert result["by_recording"]["rec-2"]["d_candidate_shadow"]["pair_win_rate"] == 0.0
    assert result["median_proposal_gap_s"] == pytest.approx(1.0)


def test_hard_pair_requires_validation_only_source() -> None:
    diagnostic = importlib.import_module("continuous_mmwave.parrad_stage3_shadow_hard_pair")
    with pytest.raises(ValueError, match="validation-only"):
        diagnostic.validate_shadow_source({"source_split": "test", "test_evaluation_allowed": False})


def test_cli_reports_candidate_count_not_number_of_age_bins(tmp_path, monkeypatch, capsys) -> None:
    diagnostic = importlib.import_module("continuous_mmwave.parrad_stage3_shadow_hard_pair")
    source = tmp_path / "source"
    source.mkdir()
    (source / "forced-summary.json").write_text(json.dumps({
        "source_split": "validation", "test_evaluation_allowed": False,
        "diagnostic_only": True, "forced_horizon_updates": 2,
        "checkpoint_sha256": "checkpoint", "threshold_artifact_sha256": "threshold",
        "integrity": {},
    }), encoding="utf-8")
    rows = [
        dict(candidate_id=str(index), recording_key="rec", sensor_id="14xx",
             proposal_time=float(index), age_updates=3,
             clean_proposal_target=(index == 0), d_candidate_shadow=float(index))
        for index in range(3)
    ]
    (source / "forced-candidates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    output = tmp_path / "pairs.json"
    monkeypatch.setattr(sys, "argv", ["shadow-hard-pair", "--source-dir", str(source), "--output", str(output)])

    diagnostic.main()

    assert json.loads(capsys.readouterr().out)["full_candidates"] == 3
    assert json.loads(output.read_text(encoding="utf-8"))["full_horizon_candidate_count"] == 3

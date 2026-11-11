from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def candidate_rows(key="subject_02__set000__TI_xWR14xx", candidate="a", terminal=7):
    subject, session, sensor = key.split("__")
    rows = []
    for age in range(1, 14):
        for cohort in ("natural", "forced"):
            if cohort == "natural" and age > terminal:
                continue
            rows.append({
                "source_split": "train", "recording_key": key, "subject_id": subject,
                "session_id": session, "sensor_id": sensor, "candidate_id": candidate,
                "cohort": cohort, "age_updates": age, "age_seconds": (age - 1) * .18,
                "current_update": 20 + age - 1, "proposal_update": 20,
                "active_ref_update": 19, "active_ref_state_id": "active:19",
                "update_target": age >= 3, "update_exclusion": None,
                "proposal_target": False, "proposal_exclusion": None,
                "decision": "confirm" if cohort == "natural" and age == terminal else "candidate",
            })
    candidates = [{"candidate_id": candidate, "proposal_update": 20, "active_ref_update": 19,
                   "active_ref_state_id": "active:19", "terminal_age": terminal, "terminal_decision": "confirm"}]
    return rows, candidates


def synthetic_source(path: Path):
    from continuous_mmwave import parrad_stage3_frozen_layers as extraction
    from continuous_mmwave.parrad_stage3_linear_probe import PROBE_SETTINGS
    path.mkdir()
    records, completed = [], []
    for subject in extraction.AUDIT_SUBJECTS:
        for session in extraction.AUDIT_SESSIONS:
            for sensor in extraction.AUDIT_SENSORS:
                key = "__".join((subject, session, sensor))
                directory = path / key
                directory.mkdir()
                rows, candidates = [], []
                for positive in (False, True):
                    extra, cs = candidate_rows(key, key + (":negative" if not positive else ":positive"))
                    for row in extra:
                        row["update_target"] = positive
                        row["proposal_target"] = positive
                    rows.extend(extra)
                    candidates.extend(cs)
                arrays = {}
                for source, dim in (("F", 4), ("E", 8), ("z", 2)):
                    refs = np.ones((len(rows), dim), np.float32)
                    vals = np.asarray([1 if r["update_target"] else -1 for r in rows], np.float32)
                    current = refs + vals[:, None] * np.asarray([1] + [0] * (dim-1), np.float32)
                    arrays["current_" + source], arrays["reference_" + source] = current, refs
                np.savez_compressed(directory / "layers.npz", **arrays)
                integrity = {"max_event_error": 0, "max_key_error": 0, "checked_steps": 1,
                             "natural_updates": 100, "natural_decisions": {"confirm": 2}}
                for name, data in (("rows", rows), ("candidates", candidates), ("integrity", integrity)):
                    (directory / (name + ".json")).write_text(json.dumps(data))
                records.append({"key": key, "metadata": {"subject_id": subject, "session_id": session, "sensor_id": sensor}, "updates": 100})
                completed.append({"key": key, "integrity": integrity, "artifact_sha256": {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ("rows.json", "candidates.json", "layers.npz", "integrity.json")}})
    config, checkpoint, stage2, thresholds = [path / name for name in ("config.yaml", "stage3.pt", "stage2.pt", "thresholds.json")]
    for file in (config, checkpoint, stage2, thresholds):
        file.write_text(file.name)
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    code = Path(extraction.__file__).parent
    manifest = {
        "source_split": "train", "status": "complete", "diagnostic_only": True, "test_evaluation_allowed": False,
        "audit_kind": "frozen_short_horizon_sequential", "trace_profile": "continuous", "ages": list(range(1,14)),
        "sequence_settings": extraction.SEQUENCE_SETTINGS, "probe_settings": PROBE_SETTINGS,
        "records": records, "completed_records": completed,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "stage2_checkpoint": str(stage2), "stage2_checkpoint_sha256": sha(stage2),
        "config": str(config), "config_sha256": sha(config),
        "threshold_artifact": str(thresholds), "threshold_artifact_sha256": sha(thresholds),
        "extractor_sha256": sha(Path(extraction.__file__)),
        "event_former_sha256": sha(code / "models/event_former.py"),
        "probe_code_sha256": sha(code / "parrad_stage3_linear_probe.py"),
        "sequence_code_sha256": {name: sha(code / name) for name in (
            "parrad_stage3_sequence_data.py", "parrad_stage3_sequence_features.py", "parrad_stage3_sequence_probe.py")},
        "cache_fingerprint": {"synthetic": True}, "layer_dimensions": {"F": 4, "E": 8, "z": 2},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_history_selection_is_past_only_candidate_local_and_cohort_local():
    from continuous_mmwave.parrad_stage3_sequence_data import select_endpoints
    rows, candidates = candidate_rows()
    other, cs = candidate_rows(candidate="b")
    rows.extend(other)
    selected, counts = select_endpoints(rows, candidates + cs, cohort="natural", endpoint_kind="fixed_age",
                                       age=7, history_lengths=(1,3), task="update")
    assert len(selected) == 2
    for endpoint in selected:
        assert [rows[i]["age_updates"] for i in endpoint.history_rows] == [5,6,7]
        assert {rows[i]["cohort"] for i in endpoint.history_rows} == {"natural"}
        assert {rows[i]["candidate_id"] for i in endpoint.history_rows} == {endpoint.candidate_id}
        assert endpoint.endpoint_row == endpoint.history_rows[-1]


def test_shared_endpoint_rows_and_no_eight_update_padding():
    from continuous_mmwave.parrad_stage3_sequence_data import select_endpoints
    rows, candidates = candidate_rows(terminal=8)
    def pick(age, histories, values=rows):
        return select_endpoints(values, candidates, cohort="natural", endpoint_kind="fixed_age", age=age, history_lengths=histories, task="proposal")
    assert len(pick(7, (1,3))[0]) == 1
    assert pick(7, (1,3,8))[0] == []
    full = pick(8, (1,3,8))[0][0]
    assert full.metadata["proposal_target"] is False
    assert full.endpoint_row == pick(8, (1,))[0][0].endpoint_row
    missing = [r for r in rows if not (r["cohort"] == "natural" and r["age_updates"] == 6)]
    assert pick(8, (1,3,8), missing)[1]["incomplete_history"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "reference", "update"])
def test_duplicate_age_and_changed_reference_abort(mutation):
    from continuous_mmwave.parrad_stage3_sequence_data import select_endpoints
    rows, candidates = candidate_rows()
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "reference":
        rows[2]["active_ref_state_id"] = "wrong"
    else:
        rows[2]["current_update"] += 1
    with pytest.raises(ValueError):
        select_endpoints(rows, candidates, cohort="natural", endpoint_kind="fixed_age", age=7, history_lengths=(1,3), task="update")


def test_terminal_selection_keeps_final_trace_and_excludes_recording_end():
    from continuous_mmwave.parrad_stage3_sequence_data import select_endpoints
    rows, candidates = candidate_rows()
    endpoints, _ = select_endpoints(rows, candidates, cohort="natural", endpoint_kind="terminal", age=3, history_lengths=(1,3), task="update")
    assert rows[endpoints[0].endpoint_row]["decision"] == "confirm"
    candidates[0]["terminal_decision"], candidates[0]["terminal_age"] = "recording_end", None
    assert select_endpoints(rows, candidates, cohort="natural", endpoint_kind="terminal", age=3, history_lengths=(1,3), task="update")[0] == []


@pytest.mark.parametrize("field,value", [("source_split","validation"), ("status","extracting"), ("test_evaluation_allowed",True)])
def test_sequence_source_rejects_invalid_split_hashes_and_partial_outputs_before_arrays(tmp_path, monkeypatch, field, value):
    from continuous_mmwave.parrad_stage3_sequence_data import load_sequence_source
    path = tmp_path / "source"
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({field: value}))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("arrays must remain unopened"))
    with pytest.raises(ValueError):
        load_sequence_source(path)


def test_smoke_overlap_requires_identical_decisions_references_and_tensors(tmp_path):
    from continuous_mmwave.parrad_stage3_sequence_data import verify_smoke_overlap
    previous,smoke=tmp_path/"previous",tmp_path/"smoke"
    synthetic_source(previous); manifest=synthetic_source(smoke)
    manifest["records"]=manifest["records"][:1]
    manifest["completed_records"]=manifest["completed_records"][:1]
    (smoke/"manifest.json").write_text(json.dumps(manifest))
    result=verify_smoke_overlap(previous,smoke)
    assert result["status"]=="accepted" and result["max_absolute_error"]==0
    key=manifest["records"][0]["key"]
    file=smoke/key/"layers.npz"
    with np.load(file) as values:
        arrays={name:values[name] for name in values.files}
    arrays["current_F"][0,0]+=.1
    np.savez_compressed(file,**arrays)
    with pytest.raises(ValueError): verify_smoke_overlap(previous,smoke)

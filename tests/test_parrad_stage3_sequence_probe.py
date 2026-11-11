from __future__ import annotations
import json
import numpy as np
import pytest


def test_grid_respects_eight_update_timing_and_separates_terminal_cohorts():
    from continuous_mmwave.parrad_stage3_sequence_probe import experiment_grid
    grid=experiment_grid()
    assert len(grid)==22
    for group in grid:
        if 8 in group["histories"]:
            assert group["age"]>=8
        if group["endpoint_kind"]=="terminal":
            assert group["cohort"]=="natural"
    assert {(g["age"],g["histories"]) for g in grid if g["cohort"]=="natural" and g["endpoint_kind"]=="fixed_age"} == {(4,(1,3)),(6,(1,3)),(7,(1,3)),(8,(1,3,8))}


def test_pair_deltas_require_identical_endpoint_ids_and_targets():
    from continuous_mmwave.parrad_stage3_sequence_probe import paired_delta
    left={"endpoint_ids":["a","b"],"targets":[False,True],"metrics":{"auroc":.6,"average_precision":.6,"balanced_accuracy":.5}}
    right={**left,"metrics":{"auroc":.8,"average_precision":.7,"balanced_accuracy":.7}}
    assert paired_delta(left,right)["auroc"]==pytest.approx(.2)
    with pytest.raises(ValueError):
        paired_delta(left,{**right,"endpoint_ids":["b","a"]})
    with pytest.raises(ValueError):
        paired_delta(left,{**right,"targets":[True,False]})


def test_terminal_age_baseline_and_single_class_strata_are_explicit():
    from continuous_mmwave.parrad_stage3_sequence_probe import stratified_metrics
    rows=[{"recording_key":"a","sensor_id":"s","session_id":"r","age_updates":4},
          {"recording_key":"b","sensor_id":"s","session_id":"r","age_updates":7}]
    result=stratified_metrics(rows,[False,True],[.1,.9])
    assert result["by_age"]["4"]["auroc"] is None
    assert result["conditional_recording"]["pairs"]==0
    assert result["conditional_recording"]["auroc"] is None


def small_grid():
    return ({"name":"natural-terminal-min3-update","cohort":"natural","endpoint_kind":"terminal", "age":3,
             "histories":(1,3),"task":"update"},)


def test_small_synthetic_sequence_evaluation_is_subject_disjoint(tmp_path,monkeypatch):
    import continuous_mmwave.parrad_stage3_sequence_probe as probe
    from test_parrad_stage3_sequence_data import synthetic_source
    source=tmp_path/"source"; synthetic_source(source)
    monkeypatch.setattr(probe,"experiment_grid",small_grid)
    output=tmp_path/"output"
    summary=probe.evaluate(source,output)
    records=[json.loads(s) for s in (output/"sequence-results.jsonl").read_text().splitlines()]
    fitted=[r for r in records if r["protocol"]=="relational"]
    assert len(fitted)==90  # 5 sources × 2 histories × 3 subjects × 3 seeds
    assert all(r["parameter_count"]==131 for r in fitted)
    assert all(r["held_count"]==16 and r["train_count"]==32 for r in fitted)
    assert all(r["metrics"]["auroc"]>.9 for r in fitted)
    assert any(r["protocol"]=="age_only" for r in records)
    assert summary["status"]=="complete" and summary["test_evaluation_allowed"] is False
    saved=output/fitted[0]["artifact"]
    with np.load(saved,allow_pickle=False) as values:
        assert values["coefficients"].shape[0] == 131
        assert set(values["held_subjects"])=={values["held_subjects"][0]}
    before=(output/"sequence-summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        probe.evaluate(source,output)
    assert (output/"sequence-summary.json").read_bytes()==before


def test_failed_or_nonconverged_fits_are_not_hidden(tmp_path,monkeypatch):
    import continuous_mmwave.parrad_stage3_sequence_probe as probe
    from test_parrad_stage3_sequence_data import synthetic_source
    source=tmp_path/"source"; synthetic_source(source)
    monkeypatch.setattr(probe,"experiment_grid",small_grid)
    def fail(*args,**kwargs):
        raise RuntimeError("injected solver failure")
    monkeypatch.setattr(probe,"fit_probe",fail)
    with pytest.raises(RuntimeError,match="solver failure"):
        probe.evaluate(source,tmp_path/"failed")
    assert json.loads((tmp_path/"failed"/"sequence-manifest.json").read_text())["status"]=="failed"
    assert not (tmp_path/"failed"/"sequence-summary.json").exists()


def test_empty_training_fold_does_not_score_uninitialized_geometry(tmp_path,monkeypatch):
    import continuous_mmwave.parrad_stage3_sequence_probe as probe
    from test_parrad_stage3_sequence_data import synthetic_source
    from continuous_mmwave.parrad_stage3_sequence_data import sha256
    source=tmp_path/"source"; manifest=synthetic_source(source)
    for result in manifest["completed_records"]:
        if result["key"].startswith(("subject_02", "subject_03")):
            for name in ("rows.json","candidates.json"):
                file=source/result["key"]/name; file.write_text("[]")
                result["artifact_sha256"][name]=sha256(file)
    (source/"manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(probe,"experiment_grid",lambda:({**small_grid()[0],"endpoint_kind":"fixed_age","age":7},))
    output=tmp_path/"output"; probe.evaluate(source,output)
    records=[json.loads(s) for s in (output/"sequence-results.jsonl").read_text().splitlines()]
    controls=[r for r in records if r["held_subject"]=="subject_04" and r["protocol"].startswith("fixed_")]
    assert len(controls)==20
    assert all(r["status"]=="undefined" and r["metrics"]["auroc"] is None for r in controls)


def test_nonconvergence_is_enumerated_without_silent_retry(tmp_path,monkeypatch):
    import continuous_mmwave.parrad_stage3_sequence_probe as probe
    from test_parrad_stage3_sequence_data import synthetic_source
    source=tmp_path/"source"; synthetic_source(source)
    monkeypatch.setattr(probe,"experiment_grid",small_grid)
    original=probe.fit_probe
    def nonconverged(*args,**kwargs):
        result=original(*args,**kwargs); result["converged"]=False
        return result
    monkeypatch.setattr(probe,"fit_probe",nonconverged)
    summary=probe.evaluate(source,tmp_path/"output")
    assert len(summary["nonconverged_fits"])==123
    assert all(not p["both_converged"] for p in summary["paired"])


@pytest.mark.parametrize("mutation", ["code","artifact","checkpoint","cohort"])
def test_source_checks_changed_hashes_and_cohort_before_arrays(tmp_path,monkeypatch,mutation):
    from continuous_mmwave.parrad_stage3_sequence_data import load_sequence_source
    from test_parrad_stage3_sequence_data import synthetic_source
    source=tmp_path/"source"; manifest=synthetic_source(source)
    if mutation=="code": manifest["probe_code_sha256"]="wrong"
    elif mutation=="artifact": (source/manifest["records"][0]["key"]/"rows.json").write_text("[]")
    elif mutation=="checkpoint": (source/"stage3.pt").write_text("changed")
    else: manifest["records"][0]["key"]="validation"
    (source/"manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(np,"load",lambda *a,**k:pytest.fail("must reject before loading features"))
    with pytest.raises(ValueError): load_sequence_source(source)

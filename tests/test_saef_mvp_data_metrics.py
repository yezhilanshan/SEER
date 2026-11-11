import json
import numpy as np
import pytest
from continuous_mmwave.event_gating.data import sha256


def assets(tmp_path):
    locks = {}
    for name in ('radar_checkpoint', 'radar_normalizers'):
        p = tmp_path/name
        p.write_bytes(name.encode())
        locks[name] = dict(path=str(p), sha256=sha256(p))
    rows = []
    for subject, role, file in [('subject1', 'train', 'a'), ('subject1', 'train', 'b'), ('subject7', 'validation', 'c')]:
        root = tmp_path/file
        root.mkdir()
        p = root/'continuous_hidden.npy'
        np.save(p, np.arange(160*128, dtype=np.float32).reshape(160, 128))
        rows.append(dict(file=file, subject=subject, role=role, frames=160, artifact_dir=str(root),
                         continuous_path=str(p), artifacts={p.name:dict(sha256=sha256(p))},
                         instances=[dict(start=20, stop=60, label=2, instance_id=0),
                                    dict(start=100, stop=140, label=2, instance_id=1)]))
    verification = dict(passed=True, source_checkpoint_sha256=locks['radar_checkpoint']['sha256'],
                        source_normalizers_sha256=locks['radar_normalizers']['sha256'],
                        recordings=3, frames=480, test_signal_values_read=False)
    for name, content in [('feature_manifest', rows), ('feature_verification', verification)]:
        p = tmp_path/(name+'.json')
        p.write_text(json.dumps(content))
        locks[name] = dict(path=str(p), sha256=sha256(p))
    return dict(locks=locks), rows


def rewrite(settings, name, content):
    from pathlib import Path
    p = Path(settings['locks'][name]['path'])
    p.write_text(json.dumps(content))
    settings['locks'][name]['sha256'] = sha256(p)


def test_full_native_stream_partial_targets_and_split(tmp_path):
    from continuous_mmwave.event_gating.mvp.data import load_records, split_fit_calibration
    settings, _ = assets(tmp_path)
    rows = load_records(settings)
    r = rows[0]
    assert r['x'].shape == (160, 128)
    assert r['source_frames'].tolist() == list(range(160))
    assert r['labels'][0] == -100 and r['labels'][25] == 2
    assert r['boundaries'] == [20, 60, 100, 140]
    assert r['boundary_target'][20] == 1
    assert r['boundary_target'][23] == pytest.approx(np.exp(-.5))
    assert r['boundary_mask'][19] and r['weak_transition_frames'][19]
    assert not r['boundary_mask'][80] and r['boundary_mask'][40]
    assert r['time_basis'] == 'nominal' and r['frame_period_s'] == 1/15
    fit, cal, val = split_fit_calibration(rows)
    assert [r['file'] for r in fit] == ['a']
    assert [r['file'] for r in cal] == ['b']
    assert [r['file'] for r in val] == ['c']
    with pytest.raises(ValueError, match='two|2'):
        split_fit_calibration(rows[1:])


@pytest.mark.parametrize('change,match', [('test','test'), ('path','conflict'), ('grid','frame'), ('source','provenance')])
def test_rejects_bad_source_before_signal_loading(tmp_path, change, match):
    from continuous_mmwave.event_gating.mvp.data import load_records
    settings, rows = assets(tmp_path)
    if change == 'test': rows[0]['role'] = 'test'
    if change == 'path': rows[0]['continuous_path'] = 'oracle_hidden.npy'
    if change == 'grid': rows[0]['source_frames'] = list(range(0, 320, 2))
    if change == 'source': rows[0]['source_checkpoint_sha256'] = 'bad'
    rewrite(settings, 'feature_manifest', rows)
    with pytest.raises(ValueError, match=match): load_records(settings)


def test_cache_hash_shape_and_finite_validation(tmp_path):
    from continuous_mmwave.event_gating.mvp.data import load_records
    settings, rows = assets(tmp_path)
    p = rows[0]['continuous_path']
    np.save(p, np.full((160,128), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match='SHA'): load_records(settings)
    rows[0]['artifacts']['continuous_hidden.npy']['sha256'] = sha256(p)
    rewrite(settings, 'feature_manifest', rows)
    with pytest.raises(ValueError, match='nonfinite'): load_records(settings)


def metric_record(instances, frames=100):
    return dict(frames=frames, instances=instances)


def result(fires, events=(), frames=100):
    fire = np.zeros(frames, bool)
    fire[fires] = True
    return dict(events=list(events), trace=dict(frame=np.arange(frames), fire=fire))


def test_boundary_union_matching_delay_and_unknown_denominators():
    from continuous_mmwave.event_gating.mvp.metrics import evaluate_rollout
    record = metric_record([dict(start=10, stop=40, label=1), dict(start=40, stop=70, label=1)])
    m = evaluate_rollout(result([12,30,40,85]), record, period=.1)
    b = m['boundary']['0.2']
    assert (b['tp'], b['fp'], b['fn']) == (2,1,1)
    assert b['unscored_unknown'] == 1
    # Valid boundary region includes stop=70 because its preceding frame is known.
    assert b['valid_duration_s'] == pytest.approx(6.1)
    assert b['unknown_duration_s'] == pytest.approx(3.9)
    assert b['false_fires_per_minute'] == pytest.approx(60/6.1)
    assert b['unknown_fires_per_minute'] == pytest.approx(60/3.9)
    assert b['signed_delay_mean_s'] == pytest.approx(.1)
    assert b['missed_ratio'] == pytest.approx(1/3)
    assert m['oversegmentation']['fires_inside_instances'] == 2


def test_event_iou_full_intervals_and_unknown_mixed_policy():
    from continuous_mmwave.event_gating.mvp.metrics import evaluate_rollout
    record = metric_record([dict(start=20, stop=40, label=1)])
    events = [dict(start_frame=0, stop_frame=50, class_id=1),
              dict(start_frame=50, stop_frame=100, class_id=2)]
    m = evaluate_rollout(result([], events), record, period=.1)
    assert m['events']['0.3']['tp'] == 1
    assert m['events']['0.5']['tp'] == 0
    assert m['events']['0.5']['unscored_mixed'] == 1
    assert m['events']['0.5']['unscored_unknown'] == 1
    assert m['events']['0.3']['matches'][0]['iou'] == pytest.approx(.4)


def test_edges_excluded_and_one_fire_cannot_match_two_gt():
    from continuous_mmwave.event_gating.mvp.metrics import evaluate_rollout
    r = metric_record([dict(start=0, stop=49, label=0), dict(start=51, stop=100, label=1)])
    m = evaluate_rollout(result([50]), r, period=.1)
    assert m['boundary']['0.2']['tp'] == 1
    assert m['boundary']['0.2']['fn'] == 1
    assert m['boundary']['0.2']['excluded_truth_at_recording_edges'] == 2


def test_recording_edges_do_not_become_negative_boundary_supervision(tmp_path):
    from continuous_mmwave.event_gating.mvp.data import load_records
    settings, rows = assets(tmp_path)
    rows[0]['instances'] = [dict(start=0, stop=160, label=2, instance_id=0)]
    rewrite(settings, 'feature_manifest', rows)
    row = load_records(settings)[0]
    assert row['boundaries'] == []
    assert not row['boundary_mask'][0] and not row['boundary_mask'][-1]
    assert row['boundary_mask'][80]


def test_early_correct_boundary_fire_is_not_extra_segmentation():
    from continuous_mmwave.event_gating.mvp.metrics import evaluate_rollout
    r = metric_record([dict(start=20, stop=70, label=0)])
    m = evaluate_rollout(result([68]), r, period=.1)
    assert m['oversegmentation']['fires_inside_instances'] == 1
    assert m['oversegmentation']['extra_fires_inside_instances'] == 0
    assert m['oversegmentation']['affected_instances'] == 0


def test_event_metrics_retain_class_support_for_semantic_consistency():
    from continuous_mmwave.event_gating.mvp.metrics import evaluate_rollout
    r=metric_record([dict(start=10,stop=30,label=2),dict(start=40,stop=70,label=1)])
    events=[dict(start_frame=10,stop_frame=30,class_id=2),
            dict(start_frame=40,stop_frame=70,class_id=0)]
    m=evaluate_rollout(result([],events),r,period=.1)
    assert m['events']['0.5']['per_class']['2']['tp']==1
    assert m['events']['0.5']['per_class']['1']['fn']==1
    assert m['events']['0.5']['per_class']['0']['fp']==1


def test_loader_rejects_source_period_conflict_before_overwriting_metadata(tmp_path):
    from continuous_mmwave.event_gating.mvp.data import load_records
    settings,rows=assets(tmp_path)
    rows[0]['frame_period_s']=.2
    rewrite(settings,'feature_manifest',rows)
    with pytest.raises(ValueError,match='time grid'):
        load_records(settings)

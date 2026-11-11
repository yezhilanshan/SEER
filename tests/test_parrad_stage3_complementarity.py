"""Catch leakage, unequal capacity/cohorts, and incomplete pilot provenance."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from test_parrad_stage3_sequence_data import synthetic_source


def test_grid_only_uses_clean_update_task_and_natural_decision_ages():
    from continuous_mmwave.parrad_stage3_complementarity import experiment_grid
    groups = experiment_grid()
    assert all(g['task'] == 'update' for g in groups)
    assert {g['age'] for g in groups if g['cohort'] == 'natural' and g['endpoint_kind'] == 'fixed_age'} == {4,5,6,7,8}
    assert all(g['histories'] == (1,3) for g in groups if g['age'] < 8)
    assert all(g['histories'] == (1,3,8) for g in groups if g['age'] >= 8)


@pytest.mark.parametrize('history', [1,3,8])
@pytest.mark.parametrize('sources', [('F',),('E',),('z',),('F','E'),('F','z')])
def test_cached_joint_projection_matches_single_original_projection(history, sources):
    from continuous_mmwave.parrad_stage3_complementarity import pack_block
    from continuous_mmwave.parrad_stage3_sequence_features import pack_history_block, features_from_pack
    rng = np.random.default_rng(41)
    current = {s:rng.normal(size=(3,history,d)) for s,d in zip(sources,(4,7))}
    reference = {s:np.repeat(rng.normal(size=(3,1,c.shape[-1])),history,axis=1) for s,c in current.items()}
    scales = {s:2. for s in sources}
    for seed in (3407,7211,12893):
        actual = features_from_pack(pack_block(current,reference,sources,seed),scales)
        expected = features_from_pack(pack_history_block(current,reference,sources=sources,seed=seed),scales)
        assert actual.shape == (3,130)
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)


def test_source_scales_cannot_fit_calibration_or_held_features():
    from continuous_mmwave.parrad_stage3_complementarity import source_scales
    pack = {'sources':('F',),'history':3,'blocks':{'F':{'residual_norm2':np.array([12.,48.,1e12,1e14])}}}
    assert source_scales(pack,np.array([True,True,False,False])) == {'F':np.sqrt(10.)}


def row(subject, session, i):
    return {'source_split':'train','subject_id':subject,'session_id':session,'sensor_id':'TI_xWR14xx',
            'recording_key':f'{subject}__{session}__TI_xWR14xx','candidate_id':str(i),
            'cohort':'natural','current_update':i,'age_updates':7}


def test_fit_artifact_keeps_all_roles_model_thresholds_and_candidate_labels(tmp_path):
    from continuous_mmwave.parrad_stage3_complementarity import fit_record
    rows = [row(s,t,i) for s,t in [('subject_02','set000'),('subject_03','set001'),
                                ('subject_02','set003'),('subject_03','set003'),('subject_04','set000')]
            for i in (0,1)]
    y = np.array([False,True]*5)
    x = np.zeros((10,130)); x[:,0] = np.where(y,1.,-1.)
    descriptor = {'group':'natural-fixed_age-7','source':'F','history':1,'seed':3407,'held_subject':'subject_04'}
    (tmp_path/'fits').mkdir()
    result = fit_record(tmp_path,'001',descriptor,rows,y,x,{'F':1.})
    assert result['parameter_count'] == 131 and result['status'] == 'fitted'
    assert result['metrics']['positives'] == 1 and result['metrics']['negatives'] == 1
    assert result['metrics']['prevalence'] == .5
    for op in result['operating_points'].values():
        assert op['calibration']['false_positive_rate'] == 0
    with np.load(tmp_path/result['artifact'],allow_pickle=False) as a:
        assert len(a['fit_endpoint_ids']) == 4 and len(a['calibration_endpoint_ids']) == 4 and len(a['held_endpoint_ids']) == 2
        assert all('set003' not in identifier for identifier in a['fit_endpoint_ids'])
        assert set(a['fit_endpoint_ids']).isdisjoint(a['calibration_endpoint_ids'])
        assert a['state_weight'].shape == (1,130)
        np.testing.assert_array_equal(a['held_targets'], [False,True])
        assert a['thresholds'].shape == (2,)


def test_missing_calibration_negatives_are_undefined_not_held_calibrated(tmp_path):
    from continuous_mmwave.parrad_stage3_complementarity import fit_record
    rows = [row('subject_02','set000',0),row('subject_03','set000',1),
            row('subject_02','set003',2),row('subject_04','set000',3),row('subject_04','set000',4)]
    y = np.array([False,True,True,False,True]); x = np.tile(np.arange(5.)[:,None],(1,130))
    (tmp_path/'fits').mkdir()
    d = {'group':'g','source':'E','history':1,'seed':3407,'held_subject':'subject_04'}
    r = fit_record(tmp_path,'001',d,rows,y,x,{'E':1.})
    assert r['status'] == 'fitted' and r['metrics']['auroc'] is not None
    assert all(p['reason'] == 'no_calibration_negatives' and p['held']['recall'] is None for p in r['operating_points'].values())


def test_paired_deltas_reject_different_calibration_or_candidate_cohorts():
    from continuous_mmwave.parrad_stage3_complementarity import paired_delta
    base = {'endpoint_fingerprint':'a','role_fingerprint':'b','metrics':{'auroc':.6},
            'operating_points':{'0.05':{'held':{'recall':.2,'false_positive_rate':.05}}}}
    other = copy.deepcopy(base); other['metrics']['auroc'] = .7; other['operating_points']['0.05']['held']['recall'] = .3
    assert paired_delta(base,other)['auroc'] == pytest.approx(.1)
    assert paired_delta(base,other)['recall@0.05'] == pytest.approx(.1)
    other['role_fingerprint'] = 'changed'
    with pytest.raises(ValueError): paired_delta(base,other)


def test_selection_prepares_shared_candidate_cohort_without_metadata_features(tmp_path):
    from continuous_mmwave.parrad_stage3_complementarity import select_group, prepare
    from continuous_mmwave.parrad_stage3_sequence_data import load_sequence_source
    source = tmp_path/'source'; synthetic_source(source); checked = load_sequence_source(source)
    g = {'name':'g','cohort':'natural','endpoint_kind':'fixed_age','age':7,'histories':(1,3),'task':'update'}
    endpoints,_ = select_group(checked,g)
    packs = prepare(source,endpoints,1)
    for e in endpoints:
        e.metadata['age_seconds'] = 9999
        e.metadata['decision'] = 'reject'
        e.metadata['verify_score'] = 1e8
    again = prepare(source,endpoints,1)
    for key in packs:
        for s in packs[key]['sources']:
            np.testing.assert_array_equal(packs[key]['blocks'][s]['projection'],again[key]['blocks'][s]['projection'])
    assert len(endpoints) == 48
    assert select_group(checked,{**g,'age':8,'histories':(1,3,8)})[0] == []


def test_evaluator_refuses_existing_output_and_nontrain_source(tmp_path):
    from continuous_mmwave.parrad_stage3_complementarity_runner import evaluate
    source = tmp_path/'source'; source.mkdir(); (source/'manifest.json').write_text(json.dumps({'source_split':'validation'}))
    out = tmp_path/'out'; out.mkdir()
    with pytest.raises(FileExistsError): evaluate(source,out)
    fresh = tmp_path/'fresh'
    with pytest.raises(ValueError): evaluate(source,fresh)
    assert not fresh.exists()


def test_full_synthetic_pilot_saves_complete_matrix_with_sparse_cohorts_marked(tmp_path):
    from continuous_mmwave.parrad_stage3_complementarity_runner import evaluate
    source=tmp_path/'source'; synthetic_source(source)
    output=tmp_path/'pilot'
    result=evaluate(source,output)
    assert result['status']=='complete' and result['record_count']==1350
    assert result['fitted_count']==1080 and len(result['undefined_records'])==270
    assert len(list((output/'fits').glob('*.npz')))==1350
    manifest=json.loads((output/'complementarity-manifest.json').read_text())
    assert manifest['source_split']=='train' and manifest['test_evaluation_allowed'] is False
    assert len(manifest['completed_groups'])==13 and manifest['active_fit'] is None
    assert manifest['upstream_provenance']['checkpoint_sha256']
    assert manifest['code_sha256']['parrad_controller.py']
    assert all(p['role_fingerprint'] and p['endpoint_fingerprint'] for p in result['paired'])
    assert all(r['status']=='fitted' or r['reason']=='no_held_endpoints'
               for r in map(json.loads,(output/'complementarity-results.jsonl').read_text().splitlines()))


def test_recorded_optimizer_settings_control_actual_fit(tmp_path,monkeypatch):
    from continuous_mmwave import parrad_stage3_complementarity as pilot
    monkeypatch.setitem(pilot.SETTINGS,'max_iter',1)
    rows=[row('subject_02','set000',0),row('subject_03','set001',1),row('subject_04','set000',2)]
    y=np.array([False,True,False]); x=np.tile(np.array([-2.,1.,0.])[:,None],(1,130))
    (tmp_path/'fits').mkdir()
    d={'group':'g','source':'F','history':1,'seed':3407,'held_subject':'subject_04'}
    record=pilot.fit_record(tmp_path,'001',d,rows,y,x,{'F':1.})
    assert record['iterations']==1


def test_metric_implementation_is_hash_bound():
    from continuous_mmwave.parrad_stage3_complementarity_runner import code_hashes
    from continuous_mmwave.parrad_stage3_sequence_data import sha256
    from continuous_mmwave import parrad_stage3_candidate_audit
    assert code_hashes()['parrad_stage3_candidate_audit.py']==sha256(Path(parrad_stage3_candidate_audit.__file__))

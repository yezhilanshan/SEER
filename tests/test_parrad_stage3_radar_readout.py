from __future__ import annotations

import importlib
import json
import numpy as np
import pytest


def module():
    return importlib.import_module('continuous_mmwave.parrad_stage3_radar_readout')


def row(subject='subject_02', session='set000', sensor='TI_xWR14xx', candidate='a', age=3):
    return {'source_split':'train', 'subject_id':subject, 'session_id':session,
            'sensor_id':sensor, 'recording_key':'__'.join((subject,session,sensor)),
            'candidate_id':candidate, 'current_update':20+age-1,
            'age_updates':age, 'age_seconds':(age-1)*.18}


def test_fold_keeps_both_sensors_of_calibration_session_out_of_fitting():
    m=module()
    rows=[row(s,t,v) for s in ('subject_02','subject_03','subject_04')
          for t in ('set000','set001','set002','set003')
          for v in ('TI_xWR14xx','TI_xWR68xx')]
    fit,cal,held=m.fold_masks(rows,'subject_04')
    assert (fit.sum(),cal.sum(),held.sum())==(12,4,8)
    assert set(np.flatnonzero(fit)) & set(np.flatnonzero(cal))==set()
    assert all(rows[i]['session_id']=='set003' for i in np.flatnonzero(cal))
    assert all(rows[i]['subject_id']=='subject_04' for i in np.flatnonzero(held))
    with pytest.raises(ValueError): m.fold_masks([row(subject='subject_05')],'subject_04')
    with pytest.raises(ValueError): m.fold_masks([{**row(),'source_split':'validation'}],'subject_04')


def test_candidate_weights_do_not_reward_long_lifetimes_or_collide_across_recordings():
    rows=[row(candidate='a',age=3),row(candidate='a',age=4),row(candidate='b'),
          row(subject='subject_03',candidate='a')]
    assert module().candidate_weights(rows).tolist()==[.5,.5,1.,1.]
    with pytest.raises(ValueError): module().candidate_weights([row(),row()])


def test_features_use_current_frozen_reference_and_past_only_stability():
    # Adjacent cosines 0 and 1, giving mean stability .5. Future and GT are absent.
    c=np.array([[[1.,0.],[0.,1.],[0.,2.]]])
    a=np.repeat(np.array([[[1.,1.]]]),3,axis=1)
    m=module()
    delta=m.features(c,a,np.array([.36]),'delta_logistic',3407,1.)
    assert delta.tolist()==[[-1.,1.,.36]]
    full=m.features(c,a,np.array([.36]),'relation_logistic',3407,1.)
    assert full.tolist()==[[0.,2.,1.,1.,.36]]
    mlp=m.features(c,a,np.array([.36]),'relation_mlp',3407,1.)
    assert mlp.tolist()==[[0.,2.,1.,1.,-1.,1.,.36]]
    stable=m.features(c,a,np.array([.36]),'stability_mlp',3407,1.)
    assert stable.tolist()==[[0.,2.,1.,1.,-1.,1.,.36,.5]]
    assert m.features(c,a,np.array([.36]),'projected_logistic',3407,1.).shape==(1,130)
    with pytest.raises(ValueError): m.features(c,a,np.array([.36]),'unknown',3407,1.)
    bad=a.copy(); bad[:,0,0]=10
    with pytest.raises(ValueError): m.features(c,bad,np.array([.36]),'stability_mlp',3407,1.)


def test_calibration_ties_cannot_exceed_target_fpr():
    m=module()
    # Five negatives; tied .8 scores must both be rejected at target .2.
    y=np.array([0,0,0,0,0,1],bool); scores=np.array([.1,.2,.3,.8,.8,.9])
    threshold=m.calibrate_threshold(y,scores,.2)
    assert threshold>.8
    metrics=m.operating_metrics(y,scores,threshold)
    assert metrics['false_positive_rate']==0
    assert metrics['recall']==1
    with pytest.raises(ValueError): m.calibrate_threshold(np.ones(3,bool),np.ones(3),.05)


@pytest.mark.parametrize('kind',['logistic','mlp'])
def test_fit_uses_training_only_transforms_and_preserves_saved_predictions(kind):
    m=module()
    x=np.array([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]])
    y=np.array([0.,0.,1.,1.]); w=np.array([.5,.5,1.,1.])
    probe=m.fit_readout(x,y,w,np.array([[-100.,0.],[100.,0.]]),kind=kind,seed=3407,
                        settings={**m.SETTINGS,'mlp_epochs':30})
    assert probe['mean'].tolist()==pytest.approx([.5,0.])
    assert probe['scores'][0]<.5<probe['scores'][1]
    assert np.isfinite(probe['losses']).all()
    again=m.predict_readout(probe,np.array([[-100.,0.],[100.,0.]]))
    np.testing.assert_allclose(again,probe['scores'],atol=1e-7)
    duplicate=m.fit_readout(x,y,w,np.array([[999.,0.]]),kind=kind,seed=3407,
                            settings={**m.SETTINGS,'mlp_epochs':30})
    np.testing.assert_array_equal(duplicate['mean'],probe['mean'])
    for key in probe['state']: np.testing.assert_array_equal(duplicate['state'][key],probe['state'][key])


def test_small_real_pipeline_is_frozen_train_only_and_saves_split_ids(tmp_path,monkeypatch):
    m=module()
    from test_parrad_stage3_sequence_data import synthetic_source
    source=tmp_path/'source'; synthetic_source(source)
    monkeypatch.setattr(m,'MODEL_SPECS',(('projected_logistic','logistic'),('stability_mlp','mlp')))
    monkeypatch.setitem(m.SETTINGS,'seeds',[3407])
    monkeypatch.setitem(m.SETTINGS,'mlp_epochs',4)
    out=tmp_path/'pilot'; summary=m.evaluate(source,out)
    assert summary['status']=='complete' and summary['source_split']=='train'
    assert summary['test_evaluation_allowed'] is False
    results=[json.loads(v) for v in (out/'readout-results.jsonl').read_text().splitlines()]
    assert len(results)==24  # two heads × three folds × four common endpoint groups
    assert all(r['metrics']['auroc']>.9 for r in results)
    manifest=json.loads((out/'readout-manifest.json').read_text())
    assert len(manifest['fits'])==6
    for fit in manifest['fits']:
        with np.load(out/fit['artifact'],allow_pickle=False) as a:
            assert len(a['fit_endpoint_ids'])==120
            assert len(a['calibration_endpoint_ids'])==40
            assert len(a['held_endpoint_ids'])==80
            assert not set(a['fit_endpoint_ids']) & set(a['calibration_endpoint_ids'])
            assert not set(a['fit_endpoint_ids']) & set(a['held_endpoint_ids'])
            assert all('set003' in k for k in a['calibration_endpoint_ids'])
    baseline=out/'readout-summary.json'; before=baseline.read_bytes()
    with pytest.raises(FileExistsError): m.evaluate(source,out)
    assert baseline.read_bytes()==before


def test_reject_partial_or_validation_source_before_creating_output(tmp_path):
    m=module(); source=tmp_path/'source'; source.mkdir()
    (source/'manifest.json').write_text(json.dumps({'source_split':'validation','status':'complete'}))
    out=tmp_path/'output'
    with pytest.raises(ValueError): m.evaluate(source,out)
    assert not out.exists()


def test_failed_fit_identifies_attempt_without_fabricating_completion(tmp_path,monkeypatch):
    m=module()
    from test_parrad_stage3_sequence_data import synthetic_source
    source=tmp_path/'source'; synthetic_source(source)
    monkeypatch.setattr(m,'MODEL_SPECS',(('projected_logistic','logistic'),))
    def fail(*args,**kwargs): raise RuntimeError('solver failure')
    monkeypatch.setattr(m,'fit_readout',fail)
    out=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='solver failure'): m.evaluate(source,out)
    manifest=json.loads((out/'readout-manifest.json').read_text())
    assert manifest['status']=='failed'
    assert manifest['active_fit']=={'fit_id':'001','model':'projected_logistic','seed':3407,'held_subject':'subject_02'}
    assert manifest['fits']==[]
    assert not (out/'readout-summary.json').exists()

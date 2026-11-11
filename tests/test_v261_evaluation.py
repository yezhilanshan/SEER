import importlib
import numpy as np
import pytest
import torch


def api():return importlib.import_module('continuous_mmwave.v261.evaluation')


def test_macro_skill_averages_person_ratios_and_fixed_class_f1():
    rows=[dict(subject='a',uid='a1',horizon=5,group='all',samples=2,mse=1.,persistence_mse=2.,nll=0.),
          dict(subject='b',uid='b1',horizon=5,group='all',samples=200,mse=10.,persistence_mse=10.,nll=1.)]
    result=api().aggregate_motion(rows)
    assert result['subject_macro']['skill']==pytest.approx(.25)
    classification=api().classification_metrics(np.array([0,0,-100]),np.array([[2.]+[0.]*13]*3))
    assert classification['samples']==2
    assert classification['macro_f1']==pytest.approx(1/14)
    empty=api().classification_metrics(np.array([-100]),np.zeros((1,14)))
    assert empty['macro_f1'] is None


def test_array_verifier_rejects_target_index_and_logits_tampering():
    original=dict(selected=np.array([1,2]),future=np.array([[2,4,6],[3,5,7]]),
                  current=np.zeros((2,256)),target=np.ones((2,3,256)),semantic=np.array([-100,0]),
                  mean=np.zeros((2,3,256)),logits=np.zeros((2,14)))
    api().verify_prediction_array(original,original,'AB')
    for key in ('selected','future','target','semantic'):
        changed={k:v.copy() for k,v in original.items()};changed[key].flat[0]+=1
        with pytest.raises(ValueError,match='mismatch'):
            api().verify_prediction_array(changed,original,'AB')
    changed={k:v.copy() for k,v in original.items()};changed['logits'][0,0]=np.nan
    with pytest.raises(ValueError,match='finite'):
        api().verify_prediction_array(changed,original,'AB')


def test_standardized_probe_does_not_update_features():
    from test_v261_contract import fixture
    from continuous_mmwave.v261.contract import build_joint_record,fit_motion_contract
    cache,labels,physical=fixture()
    record=build_joint_record(cache,labels,physical,'train')
    stats=fit_motion_contract([record],dict(mean=torch.zeros(256),std=torch.ones(256)))
    features={record.uid:cache['radar'].mean(1).clone()}
    original=features[record.uid].clone()
    result=api().fit_shared_probes(features,[record],stats,seed=42,steps=200)
    assert result['actual_optimizer_steps']=={'motion':200,'semantic':200}
    assert torch.equal(features[record.uid],original)
    api().validate_probe_checkpoint(result,42)
    result['seed']=43
    with pytest.raises(ValueError,match='probe'):
        api().validate_probe_checkpoint(result,42)

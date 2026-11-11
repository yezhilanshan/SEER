import numpy as np
import pytest
import torch
from pathlib import Path
import yaml
from continuous_mmwave.v25.physical_proposal import proposal_labels, proposal_metrics, fit_mass_weights
from continuous_mmwave.v25.stage3 import V25Stage3


def test_locked_source_hashes_are_valid_sha256():
    config=yaml.safe_load((Path(__file__).parents[1]/'configs/parrad-v25-physical-proposal-v1.yaml').read_text())
    for key,value in config.items():
        if key.endswith('sha256'):
            assert len(value)==64 and all(c in '0123456789abcdef' for c in value),key


def test_gaussian_uses_completion_time_includes_mixed_boundary():
    labels=np.array([[0,0],[0,1],[1,1],[-1,-1],[2,2],[2,2]])
    times=np.arange(12).reshape(6,2)*.09
    row=proposal_labels(labels,times)
    assert np.array_equal(row['time'],times[:,-1])
    assert np.array_equal(row['boundaries'],[.27])
    assert row['valid'].tolist()==[False,True,True,False,False,True]
    assert row['target'][1]==pytest.approx(1.)


def test_duplicate_proposals_are_false_and_missing_latency_is_null():
    times=np.arange(8)*.18
    score=np.array([0,.9,.9,0,.9,.9,0,0])
    out=proposal_metrics(times,score,np.ones(8,bool),np.array([.36]),threshold=.6)
    assert out['proposal_times']==pytest.approx([.36,.90])
    assert out['matched']==1 and out['false_proposals']==1
    assert out['recall']==1 and out['latency_median_s']==0
    assert out['density_per_minute']==pytest.approx(2*60/(8*.18))
    empty=proposal_metrics(times,np.zeros(8),np.ones(8,bool),np.array([.36]),threshold=.6)
    assert empty['recall']==0 and empty['latency_median_s'] is None
    unknown=proposal_metrics(times,score,np.zeros(8,bool),np.array([.36]),threshold=.6)
    assert unknown['recall'] is None and unknown['false_per_minute'] is None


def test_gaps_reset_streak_and_do_not_count_gap_duration():
    times=np.array([0,.18,10.,10.18])
    out=proposal_metrics(times,np.ones(4),np.array([True,False,False,True]),np.array([]),threshold=.6)
    assert out['proposal_times']==[]
    assert out['duration_s']==pytest.approx(.36)


def test_soft_weights_balance_mass():
    target=np.array([0.,.1,.5,1.]); valid=np.ones(4,bool)
    a,b=fit_mass_weights([(target,valid)])
    assert (target*a).sum()==pytest.approx(((1-target)*b).sum())


def test_component_bce_balances_soft_labels_without_cross_terms():
    target=torch.tensor([0.,.1,.5,1.]); logits=torch.tensor([-2.,-.5,1.,3.])
    a,b=fit_mass_weights([(target.numpy(),np.ones(4,bool))])
    expected=-(a*target*torch.nn.functional.logsigmoid(logits)+
               b*(1-target)*torch.nn.functional.logsigmoid(-logits)).sum()/(a*target+b*(1-target)).sum()
    actual=V25Stage3._bce(logits,target,class_weights=(a,b))
    assert actual==pytest.approx(float(expected))


def test_streaming_bce_honors_weights():
    torch.manual_seed(3)
    stage=V25Stage3(radar_dim=4,event_dim=8,semantic_dim=4)
    stage.eval()
    batch=dict(tokens=torch.randn(1,3,2,4),mask=torch.ones(1,3,2,dtype=torch.bool),
               target=torch.tensor([[0.,1.,0.]]),valid=torch.ones(1,3,dtype=torch.bool))
    baseline=stage.physical_stream_step(batch)['loss']
    weighted=stage.physical_stream_step({**batch,'weight':torch.tensor([[1.,0.,1.]])})['loss']
    assert not torch.isclose(baseline,weighted)

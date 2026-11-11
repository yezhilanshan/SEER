import importlib.util
from pathlib import Path
import numpy as np
import pytest
import json
from continuous_mmwave.v25.physical_proposal import proposal_metrics

path=Path(__file__).parents[1]/'.audit-staging/proposal_zero_diagnostic.py'
spec=importlib.util.spec_from_file_location('zero_diagnostic',path)
diag=importlib.util.module_from_spec(spec);spec.loader.exec_module(diag)


def test_ranking_perfect_reverse_and_ties():
    y=np.array([1.,1.,0.,0.])
    good=diag.ranking(y,np.array([.9,.8,.2,.1]))
    assert good['auc']==1 and good['ap']==1
    bad=diag.ranking(y,np.array([.1,.2,.8,.9]))
    assert bad['auc']==0 and bad['ap']==pytest.approx(5/12)
    tied=diag.ranking(y,np.ones(4))
    assert tied['auc']==.5 and tied['ap']==.5


def test_fractional_ranking_has_own_prevalence_and_tie_baseline():
    y=np.array([.8,.2,.1])
    r=diag.ranking(y,np.ones(3))
    assert r['auc']==pytest.approx(.5)
    assert r['ap']==pytest.approx(y.mean())


def test_high_segment_spans_boundaries_and_trigger_suppresses():
    t=np.arange(30)*.18;v=np.ones(30,bool);s=np.ones(30)*.8;b=np.array([1.,2.,3.])
    r=diag.segments(t,s,v,b,.6)
    assert r['segment_count']==1 and r['max_s']==pytest.approx(5.4)
    assert r['multi_boundary_segments']==1 and r['boundaries_in_multi_segments']==3
    d=diag.event_diagnostic(t,s,v,b,.6)
    assert d['recall']==0 and d['pair_coverage_recall']==1
    assert d['misses']['disarmed_emission_outside_window']==3


def test_invalid_gap_rearms_and_breaks_segment():
    t=np.arange(8)*.18;v=np.array([1,1,1,0,1,1,1,1],bool)
    r=diag.segments(t,np.ones(8),v,np.array([]),.6)
    assert r['segment_count']==2 and r['max_s']==pytest.approx(.72)


def test_or_fusion_lifts_two_medium_heads_above_trigger():
    r=diag.fusion_summary(np.array([.4,.7]),np.array([.4,.2]),np.ones(2,bool))
    assert r['medium_heads_fused_high_fraction']==.5


def test_full_diagnosis_reproduces_original_trigger_and_writes_artifacts(tmp_path):
    run=tmp_path/'run';run.mkdir();t=.09+np.arange(30)*.18;b=np.array([1.3,2.5,3.7])
    valid=np.ones(30,bool);valid[0]=False
    y=np.exp(-.5*((t[:,None]-b)/.36)**2).max(1).astype(np.float32)
    for step in (500,1000):
        local=.15+.7*y if step==500 else np.full(30,.4)
        radar=.1+.6*y if step==500 else np.full(30,.4)
        scores={'local':local,'radar':radar,'fusion':1-(1-local)*(1-radar)}
        array_path=run/f'proposal-step{step}-arrays.npz'
        np.savez_compressed(array_path,**{'r00_'+k:v for k,v in {'time':t,'boundaries':b,'valid':valid,'target':y,**scores}.items()})
        meta=dict(uid='record1',subject_id='subject1',sensor_id='sensor1',array_prefix='r00',
            paths={path:{'0.6':proposal_metrics(t,s,valid,b)} for path,s in scores.items()})
        (run/f'proposal-step{step}.json').write_text(json.dumps(dict(arrays_sha256=diag.sha(array_path),records=[meta])))
    output=tmp_path/'result';result=diag.analyze(run,output)
    assert result['actual_optimizer_updates']==0 and result['model_forwards']==0
    assert (output/'pr-curves.npz').is_file() and (output/'threshold-sweep.csv').is_file()
    r=result['checkpoints']['1000']['all']['paths']['fusion']['event_sweep']['0.6']
    assert r['recall']==0 and r['pair_coverage_recall']==1

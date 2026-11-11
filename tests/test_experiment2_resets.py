import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('experiment2',Path(__file__).parents[1]/'scripts/experiment2_resets.py')
experiment=importlib.util.module_from_spec(spec);spec.loader.exec_module(experiment)


def test_independent_unique_permutations_preserve_intervals_and_avoid_gt_edges():
    lengths=np.array([3,5,7,9,11,13]);starts=np.r_[0,np.cumsum(lengths)[:-1]]
    beginnings=np.repeat(starts,lengths);cutoffs=np.arange(1,len(beginnings)+1)*2
    edges=np.r_[starts*2,starts*2+1]
    result=experiment.generate(beginnings,cutoffs,int(cutoffs[-1]),edges)
    assert result==experiment.generate(beginnings,cutoffs,int(cutoffs[-1]),edges)
    assert len({tuple(c['starts']) for c in result['candidates']})==10
    for row in result['candidates']:
        assert sorted(row['update_lengths'])==sorted(lengths)
        assert sorted(row['frame_intervals'])==sorted(lengths*2)
        assert not set(np.array(row['starts'][1:])*2)&set(edges)


def test_nonuniform_observation_clock_uses_actual_frame_minimum():
    lengths=np.array([3,5,7,9,11,13]);extra=np.array([1,0,1,1,0,1])
    starts=np.r_[0,np.cumsum(lengths)[:-1]];raw=np.r_[0,np.cumsum(2*lengths+extra)[:-1]]
    cutoffs=np.concatenate([b+2*np.arange(1,l+1) for b,l in zip(raw,lengths)])
    beginnings=np.repeat(starts,lengths);frames=int((2*lengths+extra).sum())
    result=experiment.generate(beginnings,cutoffs,frames,raw)
    for row in result['candidates']:
        assert min(row['frame_intervals'])==min(result['true_frame_intervals'])
        assert max(abs(np.sort(row['frame_intervals'])-np.sort(result['true_frame_intervals'])))<=2


def test_record_bootstrap_keeps_training_seeds_paired():
    path=Path(__file__).parents[1]/'scripts/verify_experiment2_resets.py'
    spec2=importlib.util.spec_from_file_location('verify_experiment2',path)
    verifier=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(verifier)
    result=verifier.paired([[1,2,3],[-3,-2,-1]])
    assert result['mean']==0 and result['ci95']==[-2,2]
    assert result['leave_one_out']==[-2,2]


def test_independent_coverage_readback_uses_saved_coordinate_variance():
    path=Path(__file__).parents[1]/'scripts/verify_experiment2_resets.py'
    spec2=importlib.util.spec_from_file_location('verify_coverage',path)
    verifier=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(verifier)
    mu=np.zeros((3,3,256));lv=np.zeros_like(mu)
    target=np.broadcast_to(np.arange(3)[:,None,None],mu.shape)
    result=verifier.prediction_metrics(mu,lv,target)
    for h in result:
        assert h['coverage90']==2/3 and h['variance']==1
        np.testing.assert_allclose(h['mse'],5/3)

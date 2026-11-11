import numpy as np
import pytest


def test_edit_alignment_distinguishes_repetition_and_operations():
    from continuous_mmwave.radar_text.p1 import edit_alignment
    r=edit_alignment([4,7,4],[7,4])
    assert r['distance']==1 and r['deletions']==1 and r['insertions']==r['substitutions']==0
    r=edit_alignment([1,2],[1,3,2])
    assert r['insertions']==1 and r['distance']==1
    assert edit_alignment([1],[2])['substitutions']==1
    assert edit_alignment([1,2],[2,1])['ambiguous']


def test_ordered_targets_keep_repeated_runs_and_padding():
    from continuous_mmwave.radar_text.p1 import make_targets,decode_outputs
    import torch
    a=make_targets([[4,7,4],[7,4]])
    assert a['presence'][0].tolist()==a['presence'][1].tolist()
    assert a['slots'].tolist()==[[4,7,4],[7,4,14]]
    assert a['count'].tolist()==[2,1]
    logits=torch.full((2,62),-5.)
    for i,row in enumerate(a['slots']):
        for j,k in enumerate(row):logits[i,17+j*15+k]=5.
    assert decode_outputs(logits)[2]==[[4,7,4],[7,4]]


def test_probability_smoothing_is_class_permutation_equivariant():
    from continuous_mmwave.radar_text.p1 import probability_smooth
    p=np.array([[.9,.1],[.2,.8],[.3,.7]])
    t=np.array([0.,.1,.2])
    a=probability_smooth(p,t,.3)
    np.testing.assert_allclose(a[:,::-1],probability_smooth(p[:,::-1],t,.3))
    np.testing.assert_allclose(a.sum(1),1)


def test_segment_viterbi_returns_real_count_and_preserves_repeated_nonadjacent_labels():
    from continuous_mmwave.radar_text.p1 import segment_paths
    probabilities=np.full((9,3),.01)
    probabilities[:3,0]=.98;probabilities[3:6,1]=.98;probabilities[6:,0]=.98
    paths=segment_paths(np.log(probabilities),np.arange(9)*.1,.2,np.zeros((3,3)),np.zeros(3),3)
    assert paths[3]==[0,1,0]
    assert len(paths[2])==2 and paths[2][0]!=paths[2][1]
    assert len(paths[1])==1
    with pytest.raises(ValueError):segment_paths(np.log(probabilities),np.arange(9)*.1,1.,np.zeros((3,3)),np.zeros(3),3)


def test_hmm_transition_prior_and_count_metrics_are_not_gt_count_constraints():
    from continuous_mmwave.radar_text.p1 import transition_prior,hmm_path,sequence_summary
    trans,start=transition_prior([np.array([0,0,1,1]),np.array([1,1,0])],classes=2)
    np.testing.assert_allclose(np.exp(trans).sum(1),1)
    p=np.log(np.array([[.9,.1],[.8,.2],[.1,.9]]))
    assert hmm_path(p,trans,start,0).tolist()==[0,0,1]
    r=sequence_summary([dict(reference_actions=[0,1],predicted_actions=[0,1,0])])
    assert r['count_accuracy']==0 and r['count_mae']==1 and r['insertions']==1


def test_segment_dynamic_program_matches_exhaustive_small_paths():
    from continuous_mmwave.radar_text.p1 import segment_paths
    from itertools import combinations,product
    rng=np.random.default_rng(4);logp=rng.normal(size=(6,3));transition=rng.normal(size=(3,3));start=rng.normal(size=3)
    result=segment_paths(logp,np.arange(6)*.1,.2,transition,start)
    for n in (1,2,3):
        best=(-np.inf,None)
        for cuts in combinations(range(1,6),n-1):
            edges=(0,)+cuts+(6,)
            if any(b-a<2 for a,b in zip(edges,edges[1:])):continue
            for labels in product(range(3),repeat=n):
                if any(a==b for a,b in zip(labels,labels[1:])):continue
                value=start[labels[0]]+sum(logp[a:b,k].sum() for a,b,k in zip(edges,edges[1:],labels))+sum(transition[a,b] for a,b in zip(labels,labels[1:]))
                if value>best[0]:best=(value,list(labels))
        assert result[n]==best[1]

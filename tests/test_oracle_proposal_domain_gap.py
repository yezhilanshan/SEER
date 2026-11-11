import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest


def core():
    p=Path(__file__).parents[1]/'.audit-staging/oracle_proposal_gap_core.py'
    assert p.exists(), 'domain diagnostic core missing'
    sys.path.insert(0,str(p.parent))
    s=importlib.util.spec_from_file_location('gap_core',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


def test_weighted_auc_and_ap_ties_against_explicit_pairs():
    c=core();y=np.array([1,0,1,0]);s=np.array([.8,.8,.3,.1]);w=np.array([2.,1.,1.,3.])
    result=c.metrics(y,s,w)
    expected=sum(w[i]*w[j]*(float(s[i]>s[j])+.5*float(s[i]==s[j])) for i in [0,2] for j in [1,3])/12
    assert result['auroc']==pytest.approx(expected)
    assert result['ap']==pytest.approx(((2/3)*2+(3/4)*1)/3)
    assert c.metrics([1,1],[.2,.3])['auroc'] is None
    assert c.metrics([],[])['ap'] is None


def test_balancing_changes_ap_but_not_roc():
    c=core();y=np.array([0,0,0,1]);s=np.array([.1,.2,.9,.8])
    a=c.metrics(y,s);b=c.metrics(y,s,c.class_weights(y))
    assert a['auroc']==pytest.approx(b['auroc'])
    assert b['ap']>a['ap'] and b['positive_mass']==pytest.approx(.5)


def test_excluded_top_scoring_sample_cannot_make_weighted_ap_nan():
    c=core();m=c.metrics([0,1,0],[1.,.8,.2],[0.,.5,.5])
    assert m['auroc']==1. and m['ap']==1.


def test_matched_score_distributions_use_same_support_and_mass_as_metrics():
    core()
    from analyze_oracle_proposal_gap import point
    items=[dict(target=y,score=s,swapped=s,cosine=s) for y,s in [(0,.9),(0,.1),(0,.3),(1,.6)]]
    p=point(items,[0.,.25,.25,.5])
    assert p['same_score']['n']==2
    assert p['same_score']['mean']==pytest.approx(.2)
    assert p['different_score']['mean']==pytest.approx(.6)


def test_matching_excludes_missing_strata_and_never_uses_scores():
    c=core();a=[(0,0,0)]*4+[(0,0,1)]*2+[(0,1,0)]
    b=[(0,0,0)]*2+[(0,0,1)]*3+[(1,0,1)]
    weights,info=c.match_weights([a,b])
    assert info['common_strata']==2
    assert info['kept_counts']==[6,5]
    assert weights[0][-1]==0 and weights[1][-1]==0
    for w,keys in zip(weights,[a,b]):
        assert w.sum()==pytest.approx(1)
        assert sum(x for x,k in zip(w,keys) if k[1]!=k[2])==pytest.approx(.5)
    # A missing positive class must not become a fabricated AUC=0.
    _,info=c.match_weights([[(0,0,0)],[(0,0,0)]])
    assert not info['both_classes']


def test_uniform_anchors_are_fixed_causal_and_complete():
    c=core();assert c.anchors(28).tolist()==[18,21]
    assert c.anchors(24).tolist()==[]
    for j in c.anchors(100):
        r,q=c.spans(int(j));assert r==(j-18,j-6) and q==(j,j+7)
        assert r[1]-1<j and q[1]-1==j+6


def test_occurrences_distinguish_repeated_same_class():
    c=core();labels=np.repeat([0,0,1,1,0,0],2).reshape(6,2)
    ids=c.occurrences(labels)
    assert ids.tolist()==[0,0,1,1,2,2]
    assert ids[0]!=ids[4] and labels[0,0]==labels[4,0]


def test_operator_reset_and_prefix_causality():
    import torch
    c=core()
    class Former:
        def initial_state(self,n):return torch.zeros(n,1)
        def build_prompt_cache(self,n):return None
        def stream_update(self,s,x,*args):
            class R:pass
            r=R();r.event_tokens=s+x[:,0];return r
    x=torch.arange(2*12,dtype=torch.float32).reshape(2,12,1,1)
    mask=torch.ones(2,12,1,dtype=torch.bool)
    a=c.window_keys(Former(),lambda s:s,x,mask)
    assert torch.equal(torch.as_tensor(a),torch.cumsum(x[:,:,0],1))
    changed=x.clone();changed[:,7:]+=100
    b=c.window_keys(Former(),lambda s:s,changed,mask)
    np.testing.assert_array_equal(a[:,:7],b[:,:7])
    np.testing.assert_array_equal(a,c.window_keys(Former(),lambda s:s,x,mask))

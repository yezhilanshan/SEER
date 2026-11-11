from pathlib import Path
import sys,numpy as np,pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'.audit-staging'))
import proposal_supervision_diagnostic as d
from proposal_causal_triggers import emit

def config(lag=1,radius=.72):return dict(family='peak',high=.6,lookahead_updates=lag,nms_s=radius)
def stream(scores):return np.arange(len(scores))*.18,np.array(scores),np.ones(len(scores),bool)

def test_trace_emits_same_causal_events_and_counts_delay():
    t,s,v=stream([.1,.7,.9,.7,.1]);c=config(2)
    x=d.trace_peak(t,s,v,c);assert x['emissions']==emit(t,s,v,c)
    assert x['emissions'][0]['time_s']==pytest.approx(.72)
    assert x['emissions'][0]['peak_time_s']==pytest.approx(.36)
    assert d.classify(t,s,v,np.array([.72]),c)[0]['category']=='matched'

def test_no_neighbor_and_plateau_are_distinguished():
    t,s,v=stream([.1,.1,.9,.1,.1]);x=d.classify(t,s,v,np.array([.36]),config())
    assert x[0]['category']=='neighbor_constraint'
    t,s,v=stream([.8]*10);x=d.classify(t,s,v,np.array([.72]),config())
    assert x[0]['category']=='peak_shape'

def test_nms_and_match_conflict_distinguished():
    t,s,v=stream([.1,.7,.9,.7,.8,.7,.1]);tr=d.trace_peak(t,s,v,config())
    assert tr['emissions']==emit(t,s,v,config()) and int(tr['stages']['peak'].sum())==2 and int(tr['stages']['accepted'].sum())==1
    t,s,v=stream([.1,.7,.9,.7,.1]);x=d.classify(t,s,v,np.array([.36,.54]),config())
    assert [r['category'] for r in x]==['matched','match_conflict']

def test_masked_and_zero_norm_tokens_remain_missing_without_mutation():
    x=np.array([[[1.,0.],[99.,99.]],[[0.,1.],[-99.,-99.]],[[0.,0.],[0.,0.]]]);mask=np.array([[True,False],[True,False],[True,False]])
    original=x.copy();value,valid=d.token_features(x,mask)
    assert not valid[0].any() and valid[1].all() and not valid[2].any()
    assert value[1,0]==pytest.approx(1.) and value[1,1]==pytest.approx(np.sqrt(2.))
    assert np.isnan(value[2]).all() and np.array_equal(original,x)

def test_chunk_mass_accounts_for_valid_length_and_weighted_denominator():
    y=np.array([0.,.2,1.,.5]);v=np.array([False,True,True,False]);r=d.chunk_mass(y,v,0,4,3.,.6)
    assert r['valid_updates']==2 and r['gaussian_mass']==pytest.approx(1.2)
    assert r['weighted_denominator']==pytest.approx(4.08) and r['unweighted_denominator']==2
    assert r['weighted_positive_share']==pytest.approx(3.6/4.08)
    with pytest.raises(ValueError):d.chunk_mass(y,v,0,1,3.,.6)

def test_alignment_missing_and_overlapping_boundaries_flagged():
    t=np.arange(30)*.18;s=np.zeros(30);s[12]=.9;v=np.ones(30,bool)
    x=d.alignment(t,s,v,np.array([2.16,2.34]),.5)
    assert all(r['overlapping_window'] for r in x) and x[0]['offset_s']==pytest.approx(0.)
    assert d.alignment(t,s,v,np.array([2.16]),1.)[0]['offset_s'] is None

def test_tail_high_without_lookahead_is_context_failure_not_fake_event():
    t,s,v=stream([.1,.1,.1,.7,.9]);c=config(2)
    x=d.classify(t,s,v,np.array([.72]),c)[0]
    assert x['category']=='invalid_context' and x['tail_unconfirmable_high_count']==2
    assert d.trace_peak(t,s,v,c)['emissions']==[]

def test_missing_alignment_windows_have_no_rate_and_partial_support_excludes_missing():
    t,s,v=stream([.1,.8,.1]);boundaries=np.array([.18,20.])
    no_threshold=d.alignment(t,s,v,boundaries,None)
    missing=d.alignment(t,s,np.zeros_like(v),boundaries,.5)
    assert d.alignment_rate(no_threshold)['peak_rate'] is None
    assert d.alignment_rate(missing)['peak_rate'] is None
    x=d.alignment_rate(d.alignment(t,s,v,boundaries,.5))
    assert x['evaluable']==1 and x['missing_observation']==1 and x['peak_rate']==1.

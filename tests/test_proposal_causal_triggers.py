from pathlib import Path
import importlib.util
import numpy as np

spec=importlib.util.spec_from_file_location('causal_triggers',Path(__file__).parents[1]/'.audit-staging/proposal_causal_triggers.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def c(family,**kwargs):return dict(family=family,high=.6,**kwargs)
def emit(scores,config,valid=None):
    return m.emit(np.arange(len(scores))*.18,np.array(scores),np.ones(len(scores),bool) if valid is None else valid,config)

def test_oneshot_and_hysteresis_have_different_release_conditions():
    scores=[.1,.8,.8,.5,.8,.8,.3,.3,.8,.8]
    assert np.allclose([x['time_s'] for x in emit(scores,c('one-shot'))],[.36,.90,1.62])
    assert np.allclose([x['time_s'] for x in emit(scores,c('hysteresis',low=.4,rearm_updates=2))],[.36,1.62])

def test_peak_decision_is_delayed_not_backdated_and_tail_not_flushed():
    scores=[.1,.7,.9,.7,.1]
    e=emit(scores,c('peak',lookahead_updates=2,nms_s=.18))
    assert len(e)==1 and np.isclose(e[0]['time_s'],.72)
    assert np.isclose(e[0]['peak_time_s'],.36) and np.isclose(e[0]['lookahead_s'],.36)
    assert emit(scores[:-1],c('peak',lookahead_updates=2,nms_s=.18))==[]

def test_peak_ties_choose_first_and_nms_never_replaces_issued_peak():
    scores=[.1,.7,.7,.1,.8,.9,.8,.1]
    e=emit(scores,c('peak',lookahead_updates=1,nms_s=.72))
    assert len(e)==2 # equal .72 centre separation is allowed
    assert np.isclose(e[0]['peak_time_s'],.18)
    blocked=emit(scores,c('peak',lookahead_updates=1,nms_s=.9))
    assert len(blocked)==1 and np.isclose(blocked[0]['peak_time_s'],.18)

def test_prefix_invariance_for_all_trigger_families():
    scores=[.1,.7,.9,.8,.2,.1,.8,.9,.7,.1,.8,.9]
    for config in [c('one-shot'),c('hysteresis',low=.4,rearm_updates=2),c('peak',lookahead_updates=1,nms_s=.36),c('peak',lookahead_updates=2,nms_s=.36)]:
        full=emit(scores,config)
        for n in range(1,len(scores)+1):
            expected=[x for x in full if x['time_s']<=(n-1)*.18+1e-9]
            assert emit(scores[:n],config)==expected

def test_gap_resets_armed_history_and_peak_nms():
    scores=[.1,.7,.9,.7,.1,.7,.9,.7,.1]
    valid=np.ones(len(scores),bool);valid[4]=False
    assert len(emit(scores,c('peak',lookahead_updates=1,nms_s=2.),valid))==2
    valid=np.array([True,True,False,True,True])
    assert len(emit([.8]*5,c('hysteresis',low=.2,rearm_updates=2),valid))==2

def test_match_uses_emission_time_and_preserves_signed_latency():
    t=np.arange(9)*.18
    e=[dict(time_s=.90,peak_time_s=.18,lookahead_s=.72)]
    r=m.measure(t,np.ones(9,bool),np.array([.09]),e)
    assert r['matched']==0 and r['false_proposals']==1
    r=m.measure(t,np.ones(9,bool),np.array([.99]),e)
    assert r['matched']==1 and np.isclose(r['latency_median_s'],-.09)

def test_calibration_cap_has_no_feasible_and_deterministic_tie():
    grid=[dict(config={'family':'one-shot','high':.6},recall=.9,false_per_minute=9.,density_per_minute=10.,latency_p95_s=.3),
          dict(config={'family':'one-shot','high':.8},recall=.5,false_per_minute=3.,density_per_minute=5.,latency_p95_s=.3),
          dict(config={'family':'one-shot','high':.7},recall=.5,false_per_minute=3.,density_per_minute=5.,latency_p95_s=.3)]
    assert m.select(grid,2.) is None
    assert m.select(grid,3.)['config']['high']==.7
    assert m.select(grid[::-1],3.)==m.select(grid,3.)

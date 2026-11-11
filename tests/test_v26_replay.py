from types import SimpleNamespace
import importlib
import numpy as np
import pytest
import torch


def api():
    return importlib.import_module('continuous_mmwave.v26.replay')


class Former(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    def initial_state(self, n): return torch.zeros(n, 2, 3)
    def build_prompt_cache(self, n): return None
    def stream_update(self, state, radar, maturity, mask, prompt):
        return SimpleNamespace(event_tokens=state + radar.mean((1,2))[:,None,None] + maturity)


def inputs(n=14):
    intervals=np.column_stack((np.arange(n)*.18,(np.arange(n)+1)*.18))
    return intervals,torch.arange(n*12).reshape(n,4,3).float()/100,torch.ones(n,4,dtype=torch.bool)


def test_complete_mixed_and_long_tail_match_legacy_oracle_positions():
    from continuous_mmwave.v26.features import event_prefix_features
    from continuous_mmwave.v26.targets import prediction_contract
    x,f,m=inputs(80); starts=[0.,.63,7.47]
    c=prediction_contract(x,np.ones(len(x),bool),starts)
    plan=api().replay_plan(x,c['observation_valid'],starts,'oracle')
    result=api().replay_events(Former(),f,m,plan,use_graph=False)
    old=event_prefix_features(Former(),f,m,c['proxy_episode_id'],c['event_assignment_valid'],32)
    assert len(result['indices']) == 80
    torch.testing.assert_close(result['events'][old['indices']],old['events'],rtol=0,atol=0)
    assert not plan['reset'][3] and plan['reset'][4]
    assert plan['age'][3] == 4 and plan['age'][4] == 1
    assert plan['age'].max() > 32


@pytest.mark.parametrize('view',['oracle','no_reset'])
def test_prefix_and_future_command_radar_suffix_invariance(view):
    x,f,m=inputs(); observed=np.ones(len(x),bool); cutoff=7
    plan=api().replay_plan(x,observed,[0.,.63,1.9],view)
    a=api().replay_events(Former(),f,m,plan,use_graph=False)
    # Only commands strictly after the cutoff may be moved/inserted/deleted.
    p=api().replay_plan(x[:cutoff],observed[:cutoff],[0.,.63,8.,9.],view)
    b=api().replay_events(Former(),f[:cutoff],m[:cutoff],p,use_graph=False)
    torch.testing.assert_close(a['events'][:cutoff],b['events'],rtol=0,atol=0)
    altered=f.clone();altered[cutoff:]=999
    d=api().replay_events(Former(),altered,m,plan,use_graph=False)
    assert torch.equal(a['events'][:cutoff],d['events'][:cutoff])


def test_no_reset_ignores_all_proxy_commands_and_gaps_restart_explicitly():
    x,f,m=inputs(); observed=np.ones(len(x),bool)
    a=api().replay_plan(x,observed,[0.,.63],'no_reset')
    b=api().replay_plan(x,observed,[0.,.1,.2,.9],'no_reset')
    for key in ('age','reset','segment_start'): np.testing.assert_array_equal(a[key],b[key])
    observed[4]=False;x[8:]+=.1
    p=api().replay_plan(x,observed,[0.],'no_reset')
    assert p['reset'][5] and p['reset'][8] and p['age'][4] == 0
    r=api().replay_events(Former(),f,m,p,use_graph=False)
    assert 4 not in r['indices'] and len(r['events']) == 13


def test_common_trend_and_causal_age_matched_shuffle():
    x,f,m=inputs(24);starts=[0.,.63,1.35,2.07,2.79,3.51]
    p=api().replay_plan(x,np.ones(len(x),bool),starts,'oracle')
    d=api().diagnostic_indices(x,p['valid'],starts,p)
    assert d['common_start'][0] == 1 and d['common_start'][-1] == 18
    t=d['common_start'];motion=torch.arange(24).float()[:,None]*2
    for h in (1,3,5):
        assert torch.equal(api().linear_trend(motion,t,h),motion[t+h])
    donor=d['shuffle_donor'];good=np.flatnonzero(donor>=0)
    assert len(good)>0 and np.all(donor[good]<good)
    assert np.array_equal(p['age'][donor[good]],p['age'][good])
    assert np.all(p['segment_start'][donor[good]] != p['segment_start'][good])
    no=api().replay_plan(x,p['valid'],starts,'no_reset')
    nd=api().diagnostic_indices(x,no['valid'],starts,no)
    assert (nd['shuffle_donor'] == -1).all()


def test_invalid_inputs_do_not_fabricate_state():
    x,f,m=inputs();p=api().replay_plan(x,np.ones(len(x),bool),[0.],'no_reset')
    f[1,0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):api().replay_events(Former(),f,m,p,use_graph=False)


def test_future_perturbation_stays_within_observed_support_and_keeps_prefix():
    import continuous_mmwave.v26.p1c as module
    _,f,_=inputs()
    assert hasattr(module,'perturb_future_radar'), 'bounded real-input perturbation missing'
    value=module.perturb_future_radar(f,7)
    assert torch.equal(value[:7],f[:7]) and not torch.equal(value[7:],f[7:])
    for row in value[7:]:assert any(torch.equal(row,old) for old in f[:7])

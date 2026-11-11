import pytest


def test_runner_budget_contract_rejects_drift():
    from continuous_mmwave.harrad.reset_runner import validate_config
    cfg={'seeds':[42,43,44],'arms':['control','full'],'head_steps':1000,'head_batch':64,
         'initialization_steps':300,'pilot_steps':50,'steps':1000,'chunk':32,'thresholds':[.25,.5,.75,.9]}
    validate_config(cfg)
    cfg['arms']=['full']
    with pytest.raises(ValueError): validate_config(cfg)


def test_calibration_uses_only_its_saved_closed_loop_operating_points():
    from continuous_mmwave.harrad.reset_runner import choose_threshold
    rows={'.25':{'boundary':{'fp_per_known_minute':15.,'recall':.9}},
          '.5':{'boundary':{'fp_per_known_minute':8.,'recall':.8}},
          '.75':{'boundary':{'fp_per_known_minute':4.,'recall':.5}}}
    assert choose_threshold(rows,10.)==(.5,True)
    assert choose_threshold(rows,1.)==(.75,False)


def test_saved_trajectory_replay_detects_any_changed_policy_output(tmp_path):
    import numpy as np
    from continuous_mmwave.harrad.reset_runner import verify_replay
    a=tmp_path/'first'; b=tmp_path/'second'; a.mkdir(); b.mkdir()
    values={'q':np.array([.2,.8]),'logits':np.ones((2,12)), 'mean':np.ones((2,3,64)),
            'episodes':np.array([[0,4,2]]),'cutoffs':np.array([4]),'effective_boundaries':np.array([2])}
    np.savez(a/'record-00.npz',**values); np.savez(b/'record-00.npz',**values)
    assert verify_replay(a,b)==0
    values['q'][0]=.3; np.savez(b/'record-00.npz',**values)
    with pytest.raises(RuntimeError,match='replay'): verify_replay(a,b)

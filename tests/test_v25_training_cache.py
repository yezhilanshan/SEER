import numpy as np
import pytest
from continuous_mmwave.v25 import data


def test_update_proxy_and_prefix_average_exclude_unknown_frames():
    values=np.arange(24,dtype=np.float32).reshape(6,4)/24
    mask=np.ones((6,4),dtype=bool); mask[0,2:]=False
    target,valid=data.update_evidence_targets(values,mask)
    np.testing.assert_allclose(target[0,:2], values[:2,:2].mean(0))
    np.testing.assert_allclose(target[0,2:], values[1,2:])
    assert valid.all()
    prefix,pmask=data.mean_evidence_target(values[:1],mask[:1])
    assert pmask.tolist()==[True,True,False,False]
    assert np.isfinite(prefix).all()


def test_clean_update_runs_do_not_bridge_mixed_or_missing_labels():
    labels=np.array([[1,1],[1,1],[1,2],[2,2],[-100,-100],[2,2]])
    assert data.clean_runs(labels)==[(0,2,1),(3,4,2),(5,6,2)]


def test_subject_sensor_sampling_mass_and_distinct_actions():
    from continuous_mmwave.v25.cache import BalancedRunSampler
    rows=[dict(uid=f'{s}-{sensor}-{label}',subject_id=s,sensor_id=sensor,label=label)
          for s in ('subject_02','subject_13') for sensor in data.SENSORS for label in (0,1)]
    sampler=BalancedRunSampler(rows,seed=42)
    selected=[sampler.next_group(4) for _ in range(20)]
    flat=[r for group in selected for r in group]
    assert all(len({r['label'] for r in group})>=2 for group in selected)
    assert all(sum(r['subject_id']==s for r in flat)==40 for s in ('subject_02','subject_13'))
    assert all(sum(r['sensor_id']==s for r in flat)==40 for s in data.SENSORS)
    with pytest.raises(ValueError,match='distinct'):
        BalancedRunSampler([{**r,'label':0} for r in rows],seed=42).next_group(2)

import torch
import pytest


def test_failure_counter_reads_adamw_not_callback_count(tmp_path):
    from continuous_mmwave.v25.training import save_failed_optimizer
    p=torch.nn.Parameter(torch.tensor([1.]))
    optimizer=torch.optim.AdamW([p]); p.sum().backward(); optimizer.step()
    report=save_failed_optimizer(optimizer,tmp_path,reason='callback-failed')
    assert report['max_adamw_parameter_step']==1
    saved=torch.load(tmp_path/'failure-optimizer.pt',weights_only=False)
    assert int(next(iter(saved['optimizer']['state'].values()))['step'])==1


def test_boundary_comparison_requires_same_recordings():
    from continuous_mmwave.v25.evaluate import compare_boundary
    def report(uid):
        return {'recordings':[dict(record_uid=uid,subject_id='s',metrics={'0.36':dict(f1=.6,precision=.6)})]}
    with pytest.raises(ValueError,match='same recording'):
        compare_boundary(report('a'),report('b'))

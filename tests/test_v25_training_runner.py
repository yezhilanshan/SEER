import pytest
from continuous_mmwave.v25 import training


def test_terminal_budget_lock_requires_both_planned_seeds_before_holdout():
    from continuous_mmwave.v25.runner import terminal_lock
    primary=dict(seed=42,status='dev-qualified',actual_optimizer_steps=3750,
                 bundle_sha256='a'*64,config_sha256='b'*64)
    with pytest.raises(ValueError,match='planned'):
        terminal_lock([primary],planned_seeds=(42,43))
    second={**primary,'seed':43,'bundle_sha256':'c'*64}
    result=terminal_lock([primary,second],planned_seeds=(42,43))
    assert result['holdout_unlock_allowed'] is True
    failed={**primary,'status':'failed'}
    with pytest.raises(ValueError,match='qualified'):
        terminal_lock([failed],planned_seeds=(42,))


def test_progress_cannot_count_micro_batches_as_steps(tmp_path):
    from continuous_mmwave.v25.runner import successful_updates
    (tmp_path/'A').mkdir()
    log=tmp_path/'A/training.jsonl'
    log.write_text('{"phase":"A","step":1,"actual_optimizer_steps":1}\n'
                   '{"phase":"A","step":2,"actual_optimizer_steps":2}\n')
    assert successful_updates(tmp_path)==2

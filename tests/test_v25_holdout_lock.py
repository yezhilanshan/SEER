import pytest


def test_holdout_unlock_rejects_missing_seed_and_changed_weight(tmp_path):
    from continuous_mmwave.v25.holdout import validate_unlock
    with pytest.raises(ValueError, match='both planned'):
        validate_unlock({'holdout_unlock_allowed':True, 'results':[]})
    results=[dict(seed=s,status='dev-qualified',actual_optimizer_steps=3750,
                  bundle_path=str(tmp_path/'missing.pt'),bundle_sha256='a'*64,
                  config_sha256='b'*64) for s in (42,43)]
    with pytest.raises(ValueError, match='weight'):
        validate_unlock({'holdout_unlock_allowed':True,'results':results})

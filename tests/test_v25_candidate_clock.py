import pytest
import torch


def test_oracle_elapsed_clock_matches_controller_initial_observation():
    from continuous_mmwave.v25.train_heads import build_oracle_pairs
    events=[dict(role='fit',uid=str(i),subject_id='s',sensor_id='sensor',label=i,
                 stream_key=torch.ones(128)*i,age_keys={j:torch.ones(128) for j in range(1,12)}) for i in (0,1)]
    pairs=build_oracle_pairs(events)
    assert pairs[0]['age_seconds']==0.
    assert pairs[2]['age_seconds']==pytest.approx(.36)


def test_closed_candidate_training_bucket_uses_observation_ordinal():
    from continuous_mmwave.v25.collect import candidate_training_bucket
    assert candidate_training_bucket(0.)==0
    assert candidate_training_bucket(.36)==0
    assert candidate_training_bucket(.54)==1
    assert candidate_training_bucket(1.80)==3
    assert candidate_training_bucket(1.98) is None

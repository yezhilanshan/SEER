import pytest
import torch

from continuous_mmwave.v25 import train_stage2


def test_prefix_targets_retain_history_and_do_not_read_future_frames():
    record=dict(frame_target=torch.arange(8).float()[:,None].expand(-1,4)/8,
                frame_target_mask=torch.ones(8,4,dtype=torch.bool))
    prefix=train_stage2.prefix_evidence_supervision(record,stream_range=(0,3),commit_range=(0,3))
    local=train_stage2.prefix_evidence_supervision(record,stream_range=(2,3),commit_range=(0,3))
    torch.testing.assert_close(prefix['stream_target'],prefix['commit_target'])
    assert not torch.equal(local['stream_target'],local['commit_target'])
    assert prefix['stream_visible_range'].tolist()==[0,3]
    assert prefix['stream_target_range'].tolist()==[0,6]
    record['frame_target'][6:]=1000
    unchanged=train_stage2.prefix_evidence_supervision(record,stream_range=(0,3),commit_range=(0,3))
    torch.testing.assert_close(prefix['stream_target'],unchanged['stream_target'])


def test_coverage_target_masks_are_computed_for_each_actual_interval():
    record=dict(frame_target=torch.ones(8,4),frame_target_mask=torch.zeros(8,4,dtype=torch.bool))
    record['frame_target_mask'][:2,0]=True
    result=train_stage2.prefix_evidence_supervision(record,stream_range=(2,3),commit_range=(0,3))
    assert not result['stream_target_mask'].any()
    assert result['commit_target_mask'][0]
    for bounds in ((0,5),(-1,1),(2,2)):
        with pytest.raises(ValueError):
            train_stage2.prefix_evidence_supervision(record,stream_range=bounds,commit_range=(0,3))

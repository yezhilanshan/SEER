import torch
from continuous_mmwave.v25.stage3 import V25Stage3


def test_physical_chunks_carry_prototype_and_include_boundary_between_chunks():
    stage=V25Stage3(radar_dim=8,event_dim=16,semantic_dim=8)
    stage.heads.local.eval()
    tokens=torch.randn(1,5,4,8); mask=torch.ones(1,5,4,dtype=torch.bool)
    full=stage.physical_stream_step(dict(tokens=tokens,mask=mask,target=torch.zeros(1,5),
                                        valid=torch.tensor([[False,True,True,True,True]])))
    first=stage.physical_stream_step(dict(tokens=tokens[:,:3],mask=mask[:,:3],target=torch.zeros(1,3),
                                         valid=torch.tensor([[False,True,True]])))
    second=stage.physical_stream_step(dict(tokens=tokens[:,3:],mask=mask[:,3:],target=torch.zeros(1,2),
        valid=torch.ones(1,2,dtype=torch.bool),initial_state=first['next_state'],
        previous_tokens=tokens[:,2],previous_mask=mask[:,2]))
    torch.testing.assert_close(full['next_state'].mean,second['next_state'].mean)
    torch.testing.assert_close(full['next_state'].m2,second['next_state'].m2)
    assert second['next_state'].count==5
    assert second['valid_transitions']==2

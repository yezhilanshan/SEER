import pytest
import torch
from continuous_mmwave.v25 import cache


def test_partial_stage2_uses_separately_loaded_stage1_but_requires_projection():
    front=torch.nn.Module()
    front.radar_encoder=torch.nn.Linear(2,2)
    front.event_former=torch.nn.Linear(2,2)
    front.semantic_projection=torch.nn.Linear(2,2)
    source={name:value for name,value in front.state_dict().items() if not name.startswith('radar_encoder.')}
    cache.validate_original_source(front,dict(trainable_only=True,model_state=source))
    source.pop('semantic_projection.weight')
    with pytest.raises(ValueError,match='semantic_projection.weight'):
        cache.validate_original_source(front,dict(trainable_only=True,model_state=source))

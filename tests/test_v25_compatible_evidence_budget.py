import json
import pytest
import torch

from continuous_mmwave.v25.checkpoint import module_digest
from continuous_mmwave.v25.compatible_evidence import PairedEvidenceBatches
from continuous_mmwave.v25.stage3 import V25Stage3
from continuous_mmwave.v25.train_heads import train_fixed_head
from test_v25_compatible_evidence import events_fixture


def test_evidence500_updates_only_head_and_readback_counts_every_adamw_parameter(tmp_path):
    torch.manual_seed(42)
    stage=V25Stage3(radar_dim=8,event_dim=4,semantic_dim=8)
    initial=module_digest(stage.heads.evidence)
    other={name:module_digest(getattr(stage.heads,name)) for name in ('local','radar','semantic')}
    batches=PairedEvidenceBatches(events_fixture(),seed=42,device='cpu')
    report=train_fixed_head(stage,batches,phase='evidence',steps=500,output=tmp_path/'training',source={'kind':'unit-fixture'})
    assert report['actual_optimizer_steps']==500
    assert stage.optimizer_steps=={'physical':0,'semantic':0,'evidence':500}
    assert module_digest(stage.heads.evidence)!=initial
    assert all(module_digest(getattr(stage.heads,name))==digest for name,digest in other.items())
    assert [row['step'] for row in report['checkpoints']]==[100,500]
    final=torch.load(report['checkpoints'][-1]['path'],weights_only=False)
    assert len(final['optimizer']['state'])==len(list(stage.heads.evidence.parameters()))
    assert all(int(state['step'])==500 for state in final['optimizer']['state'].values())
    assert all(row['draws']==400 for row in batches.coverage())


@pytest.mark.parametrize('failure', ['loss','builder'])
def test_failed_head_training_preserves_actual_partial_optimizer_and_weights(tmp_path,failure):
    stage=V25Stage3(radar_dim=8,event_dim=4,semantic_dim=8)
    batches=PairedEvidenceBatches(events_fixture(),seed=42,device='cpu')
    observed=[];calls=0
    def build(module):
        nonlocal calls
        calls+=1
        if calls==3:
            if failure=='builder':
                raise RuntimeError('fixture batch failure')
            return torch.tensor(float('nan'),requires_grad=True)
        return batches(module)
    with pytest.raises(RuntimeError):
        train_fixed_head(stage,build,phase='evidence',steps=500,output=tmp_path/'training',
                         source={'kind':'unit-fixture'},on_step=lambda row:observed.append(row['actual_optimizer_steps']))
    assert observed==[1,2]
    report=json.loads((tmp_path/'training/failure-readback.json').read_text())
    assert report['max_adamw_parameter_step']==2
    assert set(report['all_parameter_steps'])=={2}
    partial=torch.load(tmp_path/'training/failure-head.pt',weights_only=False)
    assert partial['optimizer_steps']['evidence']==2
    for key,value in stage.state_dict().items():
        assert torch.equal(partial['stage_state'][key],value)

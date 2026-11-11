import torch
import pytest
from continuous_mmwave.v25 import training


class Quadratic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(1.))
        self.frozen=torch.nn.Parameter(torch.tensor(3.),requires_grad=False)
    def token_step(self,row):
        return dict(loss=(self.weight-row['target']).square())


def test_accumulation_counts_adam_updates_and_preserves_frozen():
    module=Quadratic()
    optimizer=torch.optim.AdamW([module.weight],lr=.01,weight_decay=.01)
    rows=[dict(target=torch.tensor(float(i%2))) for i in range(8)]
    result=training.accumulated_updates(module,optimizer,rows,steps=2,accumulation=4)
    assert result['actual_optimizer_steps']==2
    assert int(optimizer.state[module.weight]['step'])==2
    assert module.frozen.item()==3
    assert module.weight.item()<1
    assert len(result['records'])==2


def test_nonfinite_stops_before_any_adam_update():
    module=Quadratic(); optimizer=torch.optim.AdamW([module.weight],lr=.01)
    with pytest.raises(RuntimeError,match='nonfinite'):
        training.accumulated_updates(module,optimizer,[dict(target=torch.tensor(float('nan')))],steps=1,accumulation=1)
    assert not optimizer.state


def test_saved_optimizer_resume_matches_uninterrupted_next_update(tmp_path):
    original=Quadratic(); optimizer=torch.optim.AdamW([original.weight],lr=.01,weight_decay=.01)
    training.accumulated_updates(original,optimizer,[dict(target=torch.tensor(0.))],steps=1,accumulation=1)
    path=tmp_path/'resume.pt'
    torch.save(dict(model=original.state_dict(),optimizer=optimizer.state_dict()),path)
    saved=torch.load(path,weights_only=True)
    resumed=Quadratic(); resumed.load_state_dict(saved['model'])
    second=torch.optim.AdamW([resumed.weight],lr=.01,weight_decay=.01); second.load_state_dict(saved['optimizer'])
    for module,opt in ((original,optimizer),(resumed,second)):
        training.accumulated_updates(module,opt,[dict(target=torch.tensor(.5))],steps=2,start_step=1,accumulation=1)
    torch.testing.assert_close(original.weight,resumed.weight,rtol=0,atol=0)
    assert int(second.state[resumed.weight]['step'])==2


def test_amp_overflow_is_zero_real_updates():
    class Overflow(Quadratic):
        def token_step(self,row):
            return dict(loss=self.weight*1e38)
    module=Overflow()
    with torch.no_grad(): module.weight.fill_(1e-38)
    optimizer=torch.optim.AdamW([module.weight],lr=.01)
    scaler=torch.amp.GradScaler('cpu',init_scale=256.)
    with pytest.raises(RuntimeError,match='AMP skip'):
        training.accumulated_updates(module,optimizer,[{}],steps=1,accumulation=1,scaler=scaler)
    assert not optimizer.state

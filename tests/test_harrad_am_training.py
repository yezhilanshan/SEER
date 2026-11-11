import json
import math
import numpy as np
import pytest
import torch
from test_harrad_joint_data import example
from test_harrad_am_model import make_am


def test_reference_bank_is_train_only_class_covered_and_deterministic():
    from continuous_mmwave.harrad.am_data import select_bank, diagnostic_panels
    from continuous_mmwave.harrad.joint_data import fixed_design
    records=[example(),example('validation','subject7')]
    a=select_bank(records,3,classes=(2,4)); b=select_bank(records,3,classes=(2,4))
    np.testing.assert_array_equal(a,b)
    assert a.shape==(6,3) and set(a[:,0])=={0} and np.bincount(a[:,2],minlength=5)[[2,4]].tolist()==[3,3]
    assert all(records[rid]['semantic'][t]==label for rid,t,label in a)
    with pytest.raises(ValueError,match='support'): select_bank(records,3,classes=(0,2,4))
    panels=diagnostic_panels(records,fixed_design(records,[42],3,4),train_size=8)
    assert len(panels['train'])==8 and len(panels['validation'])==4
    assert all(records[rid]['role']=='train' for rid in panels['train'][:,0])


def test_tiny_training_has_fixed_budget_curves_frozen_bank_and_replay(tmp_path):
    from continuous_mmwave.harrad.am_data import select_bank, diagnostic_panels, build_bank
    from continuous_mmwave.harrad.am_training import fit_alignment
    from continuous_mmwave.harrad.joint_data import fit_stats,fixed_design
    records=[example(),example('validation','subject7')]; stats=fit_stats(records)
    design=fixed_design(records,[42],3,4); panels=diagnostic_panels(records,design,train_size=8)
    model=make_am(); rows=select_bank(records,3,classes=(2,4)); bank=build_bank(model,records,stats,rows,'cpu',2)
    assert bank['vectors'].shape==(6,16) and not bank['vectors'].requires_grad
    cfg={'steps':3,'learning_rate':.0001,'weight_decay':.01,'amp':False,'max_batch_attempts':3,
         'checkpoints':[0,2,3],'evaluation_batch_size':2,'alignment_weight':.1,'temperature':.1}
    done=fit_alignment(model,records,stats,design['schedule_42'],panels,bank,cfg,tmp_path/'run','cpu')
    assert done['successful_steps']==done['attempts']==3 and done['replay_max_abs']==0
    assert done['frozen_unchanged'] and done['independent_verified']
    assert set(done['curves'])=={'0','2','3'}
    assert (tmp_path/'run/predictions-step-0000-train.npz').is_file()
    assert done['curves']['3']['validation']['n']==4
    assert done['modulation_off_validation']['n']==4
    with np.load(tmp_path/'run/predictions-step-0003-validation.npz') as arrays:
        similarities=arrays['query']@bank['vectors'].cpu().numpy().T
        allowed=arrays['labels'][:,None]!=bank['labels'].cpu().numpy()[None]
        expected=np.mean([s[m].mean() for s,m in zip(similarities,allowed)])
    assert done['curves']['3']['validation']['negative_cosine']==pytest.approx(expected,abs=1e-6)
    history=json.loads((tmp_path/'run/history.json').read_text())
    assert history[0]['semantic_ce_raw']==pytest.approx(history[0]['semantic']*math.log(12))

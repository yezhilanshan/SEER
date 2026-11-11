"""Semantic labels, pair/source shortcuts, calibration gates and masking."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest


def core():
    path=Path(__file__).parents[1]/'.audit-staging/stage3b_semantic_core.py'
    assert path.exists(), 'semantic pilot core absent'
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('semantic_pilot_core',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_labels_mask_mixed_and_near_boundary_and_reject_invalid_support():
    c=core();labels=np.repeat([0,1],20).reshape(20,2);t=np.arange(40).reshape(20,2)*.09;b=np.array([1.8])
    assert c.support_label(labels,t,b,2,6)==0
    assert c.support_label(labels,t,b,13,18)==1
    assert c.support_label(labels,t,b,8,13)==-1
    assert c.support_label(labels,t,b,7,10)==-1
    assert c.support_label(labels,t,b,-1,2)==-1
    assert c.replay_spans(30)==((12,24),(30,37))
    assert c.replay_spans(17) is None


def test_oracle_pairs_are_same_recording_disjoint_and_exact_age_class_balance():
    c=core();bank=[dict(record=i,label=y,start=12*j,end=12*j+12) for i in range(2) for j,y in enumerate([0,1,0,1])]
    pairs=c.make_pairs(bank,seed=42)
    assert pairs==c.make_pairs(bank,seed=42)
    for p in pairs:
        a,b=bank[p['reference']],bank[p['candidate']]
        assert a['record']==b['record'] and p['target']==int(a['label']!=b['label'])
        assert a['end']<=b['start'] or b['end']<=a['start']
        assert p['reference_action']==a['label'] and p['candidate_action']==b['label']
    for r in (0,1):
        for age in range(1,7):
            ys=[p['target'] for p in pairs if bank[p['reference']]['record']==r and p['k']==age]
            assert ys and ys.count(0)==ys.count(1)


def test_gate_only_uses_complete_calibration_and_cannot_be_rescued_by_dev():
    c=core();g=dict(auroc=.81,ap=.82,cells_with_both_classes=8)
    assert c.oracle_gate(g)
    assert not c.oracle_gate(dict(g,ap=.79))
    assert not c.oracle_gate(dict(g,cells_with_both_classes=7))
    with pytest.raises(ValueError):c.oracle_gate(dict(g,partition='dev'))


def test_balance_has_equal_record_age_class_mass_and_independent_loss_scale():
    import torch
    c=core();record=np.array([0,0,0,0,1,1]);age=np.ones(6,int);y=np.array([0,0,1,1,0,1])
    w=c.balanced_weights(record,age,y)
    for r in (0,1):
        for label in (0,1):assert w[(record==r)&(y==label)].sum()==pytest.approx(.25)
    loss=c.semantic_objective(torch.zeros(6),torch.tensor(y).float(),torch.tensor(w),
                              torch.zeros(2),torch.tensor([0.,1.]),torch.tensor([.5,.5]))
    assert loss.item()==pytest.approx(1.2*np.log(2))


def test_fifty_actual_updates_save_zero_and_fifty_with_no_input_gradients(tmp_path):
    import torch
    c=core();torch.manual_seed(42)
    from continuous_mmwave.v25.models import SemanticDifference
    model=SemanticDifference(4,8);a=torch.randn(12,4);b=torch.randn(12,4);y=torch.arange(12).remainder(2).float();w=torch.ones(12)/12
    before={k:v.clone() for k,v in model.state_dict().items()}
    info=c.train_head(model,a,b,y,w,tmp_path/'oracle',steps=50)
    assert info['actual_steps']==50 and set(info['parameter_steps'])=={50}
    zero=torch.load(tmp_path/'oracle-step0.pt',weights_only=False)
    final=torch.load(tmp_path/'oracle-step50.pt',weights_only=False)
    assert all(torch.equal(zero['model_state'][k],v) for k,v in before.items())
    assert any(not torch.equal(final['model_state'][k],v) for k,v in before.items())
    assert a.grad is None and b.grad is None


def test_cosine_dev_is_not_computed_before_lock_and_missing_inputs_stay_nan():
    c=core();a=np.array([[1.,0.],[np.nan,np.nan]],np.float32);b=np.array([[0.,1.],[np.nan,np.nan]],np.float32);valid=np.array([True,False])
    assert np.isnan(c.cosine_baseline(a,b,valid,enabled=False)).all()
    score=c.cosine_baseline(a,b,valid,enabled=True)
    assert score[0]==pytest.approx(.5) and np.isnan(score[1])


def test_existing_record_partition_is_preserved_and_conflicts_fail_closed():
    c=core();old=dict(uid='subject_02-set000-TI_xWR14xx',partition='train',events=[])
    assert c.pin_record_partition(old,'train')==old
    assert c.pin_record_partition(dict(uid='x'),'train')['partition']=='train'
    with pytest.raises(ValueError):c.pin_record_partition(old,'dev')


def test_batch_qualification_keeps_failed_cross_batch_tolerance_visible():
    c=core()
    with pytest.raises(AssertionError):c.qualify_key_replay(.00019,None)
    audit=dict(passed=True,original_batch_max_error=0.,mixed_batch_max_error=.00019)
    result=c.qualify_key_replay(.00019,audit)
    assert result['original_batch_passed'] and not result['cross_batch_1e_minus5_passed']
    with pytest.raises(AssertionError):c.qualify_key_replay(.00019,dict(audit,original_batch_max_error=1e-8))


import numpy as np
import pytest


def test_independent_saved_prediction_audit_detects_wrong_metrics(tmp_path):
    from continuous_mmwave.harrad.verification import verify_predictions
    logits=np.eye(12,dtype=np.float32)*10
    p=tmp_path/'validation.npz'
    np.savez(p,logits=logits,labels=np.arange(12),subjects=np.array(['s']*12))
    from continuous_mmwave.harrad.training import classification_metrics
    expected=classification_metrics(logits,np.arange(12),['s']*12)
    result=verify_predictions(p,expected)
    assert result['passed'] and result['macro_f1']==1
    wrong={**expected,'macro_f1':.9}
    with pytest.raises(AssertionError): verify_predictions(p,wrong)


def test_independent_audit_counts_out_of_group_mistakes(tmp_path):
    from continuous_mmwave.harrad.verification import verify_predictions
    from continuous_mmwave.harrad.training import classification_metrics
    labels=np.array([0,0,1,6,6,7]); prediction=np.array([0,6,0,6,0,6])
    logits=np.full((6,12),-2.,np.float32); logits[np.arange(6),prediction]=2
    p=tmp_path/'prediction.npz'; np.savez(p,logits=logits,labels=labels,subjects=np.array(['s']*6))
    expected=classification_metrics(logits,labels,['s']*6)
    result=verify_predictions(p,expected)
    assert result['accuracy']==pytest.approx(1/3)
    assert result['events_macro_f1']==pytest.approx(.5/6)
    assert result['gestures_macro_f1']==pytest.approx(.5/6)

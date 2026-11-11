from __future__ import annotations
import importlib
import numpy as np
import pytest


def mod(): return importlib.import_module('continuous_mmwave.split_birnn.diagnostics')


def logits(pred):
    x=np.zeros((len(pred),9),dtype=np.float32)
    for i,p in enumerate(pred):
        if p<0: x[i]=np.nan
        else: x[i,p]=10
    return x


def test_label_shifts_are_non_circular_and_do_not_mutate_primary_labels():
    y=np.array([0,0,1,1,2,2]); original=y.copy()
    assert mod().shift_labels(y,2).tolist()==[-100,-100,0,0,1,1]
    assert mod().shift_labels(y,-2).tolist()==[1,1,2,2,-100,-100]
    np.testing.assert_array_equal(y,original)
    assert mod().shift_labels(y,10).tolist()==[-100]*6


def test_collar_and_tail_views_use_prespecified_masks():
    d=mod();y=np.array([0,0,0,1,1,1,1,1])
    assert d.collar_mask(8,[3],1).tolist()==[True,True,False,False,False,True,True,True]
    report=d.reliability_diagnostics(logits([0,0,0,0,1,1,1,1]),y,[0,3],tail_start=6)
    assert report['tail']['valid_frames']==2
    assert report['non_tail']['valid_frames']==6
    assert report['primary_labels_unchanged'] is True


def test_matched_f1_and_full_stream_abstention_have_distinct_denominators():
    d=mod(); y=np.array([0,0,0])
    a=logits([0,0,0]); b=logits([0,0,-1])
    report=d.comparison_views(a,b,y)
    assert report['common_valid_frames']==2
    assert report['baseline_coverage']==2/3
    assert report['matched']['baseline']['weighted_f1']==1
    assert report['full']['baseline']['weighted_f1']==pytest.approx(.8)


def test_improve_harm_and_net_are_counts_over_valid_and_per_class():
    d=mod();y=np.array([0,0,1,1,-100])
    r=d.correction_gains(logits([1,0,0,1,0]),logits([0,1,1,1,0]),y)
    assert r['improved']==2 and r['harmed']==1
    assert r['improve_rate']==.5 and r['harm_rate']==.25 and r['net_gain']==.25
    assert r['per_class'][0]['improved']==1 and r['per_class'][0]['harmed']==1

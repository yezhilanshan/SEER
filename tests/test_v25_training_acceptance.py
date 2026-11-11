import pytest
from continuous_mmwave.v25 import training


def test_actual_seconds_excludes_outside_windows_and_counts_unique_candidates():
    p=[.9,.1,.8,.2,.1,.9]
    y=[1,0,1,0,1,0]
    age=[.54,.72,1.08,1.98,.18,.90]
    result=training.seconds_bias_probe(p,y,age,['p1','n1','p2','n2','p3','n3'],
        early_range=(.54,.72),mature_range=(1.08,1.98))
    assert result['early_auroc']==1
    assert result['mature_auroc']==1
    assert result['same_early_fpr']==0
    assert result['coverage']['early']=={'positive_candidates':1,'negative_candidates':1}
    duplicate=training.seconds_bias_probe([.9]*40+[.1]*40,[1]*40+[0]*40,[.54]*80,
        ['one-positive']*40+['one-negative']*40,early_range=(.54,.72),mature_range=(1.08,1.98))
    assert duplicate['coverage']['early']['positive_candidates']==1
    assert duplicate['mature_auroc'] is None


def test_constant_output_and_missing_coverage_do_not_pass():
    criteria=dict(min_early_auroc=.75,min_mature_auroc=.80,max_same_early_fpr=.10,max_abs_same_age_correlation=.30)
    metrics=training.seconds_bias_probe([.1,.1,.1,.1],[0,1,0,1],[.54,.54,1.08,1.08],['a','b','c','d'],
        early_range=(.54,.72),mature_range=(1.08,1.98))
    assert not training.seconds_probe_pass(metrics,criteria,min_candidates=20)
    assert metrics['same_age_correlation'] is None

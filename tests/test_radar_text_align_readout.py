import pytest


def test_p0_recipes_preserve_stage1_head_training_controls():
    from continuous_mmwave.radar_text.align_readout import p0_specs
    from continuous_mmwave.radar_text.head_ablation import arm_specs
    specs=p0_specs()
    assert specs==[s for s in arm_specs('original_optimizer') if s['arm'] in ('H1','H4')]


def test_paired_input_gate_rejects_changed_radar_or_reference():
    from continuous_mmwave.radar_text.align_readout import validate_window_inputs
    first=[dict(sample_id='a',actions=[1,2],input_sha256='abc',start=0,stop=8,full_window=True)]
    assert validate_window_inputs(first,first)==1
    for key,value in [('input_sha256','bad'),('actions',[2,1]),('stop',9)]:
        with pytest.raises(ValueError):validate_window_inputs(first,[dict(first[0],**{key:value})])
    with pytest.raises(ValueError):validate_window_inputs(first,first*2)


def test_count_mae_separates_under_over_and_signed_bias():
    from continuous_mmwave.radar_text.align_readout import count_metrics
    rows=[dict(reference_actions=[1,2,3],predicted_actions=[1]),
          dict(reference_actions=[1],predicted_actions=[1,2]),
          dict(reference_actions=[1,2],predicted_actions=[2,1])]
    metric=count_metrics(rows)
    assert metric['count_mae']==pytest.approx(1.)
    assert metric['count_bias_pred_minus_gt']==pytest.approx(-1/3)
    assert metric['mean_under_count']==pytest.approx(2/3)
    assert metric['mean_over_count']==pytest.approx(1/3)
    assert metric['under_windows']==metric['over_windows']==metric['equal_count_windows']==1


def test_readout_summary_covers_all14_classes_and_count_error():
    from continuous_mmwave.radar_text.align_readout import readout_summary
    row=dict(reference_actions=[1,2],predicted_actions=[1],sequence_exact=0.,action_precision=1.,
             action_recall=.5,order_lcs=.5,template_exact=0.,unmatched=0.)
    summary=readout_summary([row])
    assert len(summary['per_class'])==14
    assert summary['count_mae']==1
    assert summary['per_class'][2]['recall']==0.


def test_training_feature_extraction_allows_training_subjects_but_keeps_holdout_gate():
    from continuous_mmwave.radar_text.align_readout import validate_subjects
    checkpoint=dict(training_subjects=['s1'],selection_subjects=['s2'])
    records={'train':[dict(subject_id='s1')],'validation':[dict(subject_id='s2')],'test':[dict(subject_id='s3')]}
    validate_subjects(checkpoint,records)
    with pytest.raises(ValueError):validate_subjects(checkpoint,dict(records,test=records['train']))
    with pytest.raises(ValueError):validate_subjects(checkpoint,dict(records,test=records['validation']))
    with pytest.raises(ValueError):validate_subjects(checkpoint,dict(records,train=[dict(subject_id='s4')]))


def test_b3_encoder_identity_is_checked_across_different_pilot_paths():
    from continuous_mmwave.radar_text.align_readout import validate_encoder_identity
    previous={'/old/pilot/align/best.pt':'same','/old/pilot/stage1/best.pt':'other'}
    assert validate_encoder_identity('same',previous)=='same'
    with pytest.raises(ValueError):validate_encoder_identity('different',previous)
    with pytest.raises(ValueError):validate_encoder_identity('same',{})

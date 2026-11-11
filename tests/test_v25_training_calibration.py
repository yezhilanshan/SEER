from continuous_mmwave.v25.evidence import EvidenceConfig


def test_fixed_grid_has_exactly24_and_recomputes_label_at_decision_time():
    from continuous_mmwave.v25.calibrate import fixed_grid,score_episodes
    assert len(fixed_grid())==24
    episode=dict(episode_id='record:c1',proposal_time=1.08,reference_time=.72,end_time=1.98,
        gt_intervals=[(0.,1.,0),(1.,3.,1)],decision='confirm',
        observations=[dict(timestamp=1.08,age_s=.54,difference=.99,stability=1.,physical_difference=0.)])
    config=EvidenceConfig(warmup_s=.54,timeout_s=1.98,h_plus=.1,evidence_weight=0.)
    metric=score_episodes([episode],config)
    # At the simulated decision the new label is not yet established .54s.
    assert metric['positive_episodes']==0
    assert metric['unknown_confirms']==1
    assert metric['eligible'] is False

from continuous_mmwave.event_gating.mvp.metrics import evaluate_rollout
from test_saef_mvp_data_metrics import metric_record, result


def test_summary_handles_no_boundaries_and_preserves_unknown_burden():
    from continuous_mmwave.event_gating.mvp.reporting import summarize_arm, decision_report
    record=metric_record([dict(start=0,stop=100,label=1)])
    m=evaluate_rollout(result([30]),record,period=.1)
    arm=dict(records=[dict(file='subject7_a',metrics=m)])
    summary=summarize_arm(arm)
    assert summary['boundary']['0.5']['missed_ratio'] is None
    assert summary['boundary']['0.5']['fp']==1
    assert summary['boundary']['0.5']['valid_duration_s']==9.9
    assert summary['boundary']['0.5']['signed_delay_mean_s'] is None
    assert summary['oversegmentation']['extra_fires_inside_instances']==1
    d=decision_report({a:summary for a in ('B0','B1','B2','B3')},synthetic_only=False)
    assert not d['research_go']
    assert d['G1']['candidate_pass'] is None


def test_synthetic_gates_never_claim_research_success():
    from continuous_mmwave.event_gating.mvp.reporting import summarize_arm, decision_report
    record=metric_record([dict(start=10,stop=70,label=1)])
    m=evaluate_rollout(result([10,70]),record,period=.1)
    arm=dict(records=[dict(file='subject7_a',metrics=m)])
    summaries={a:summarize_arm(arm) for a in ('B0','B1','B2','B3')}
    d=decision_report(summaries,synthetic_only=True)
    assert d['status']=='synthetic_only' and not d['research_go']

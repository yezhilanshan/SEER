"""Do not promote macro recall gains that hide a subject's false positives."""
import copy


def test_nonconverged_required_semantic_probe_is_inconclusive():
    from continuous_mmwave.parrad_projection_evaluate import summarize_decision
    records, semantic = matrix()
    semantic[0].update(passed=False, complete=False)
    assert summarize_decision(records, semantic)['decision'] == 'inconclusive'


def matrix():
    records = []
    for group in ('natural-fixed_age-4', 'natural-fixed_age-5', 'natural-fixed_age-6',
                  'natural-fixed_age-7', 'natural-terminal-min3'):
        for training_seed in (1701, 1702, 1703):
            for seed in (3407, 7211, 12893):
                for subject in ('subject_02', 'subject_03', 'subject_04'):
                    records.append(dict(group=group, training_seed=training_seed, seed=seed,
                                        held_subject=subject, delta=dict(auroc=.03, average_precision=.02,
                                        **{'recall@0.05': .04, 'false_positive_rate@0.05': 0.,
                                           'recall@0.1': .02, 'false_positive_rate@0.1': 0.})))
    semantic = [dict(held_subject=s, training_seed=t, passed=True)
                for s in ('subject_02', 'subject_03', 'subject_04') for t in (1701, 1702, 1703)]
    return records, semantic


def test_successful_paired_matrix_only_allows_next_research_stage():
    from continuous_mmwave.parrad_projection_evaluate import summarize_decision
    records, semantic = matrix()
    summary = summarize_decision(records, semantic)
    assert summary['decision'] == 'go_research'
    assert summary['production_promotion_allowed'] is False
    assert summary['primary']['auroc'] == .03


def test_one_subject_seed_fpr_regression_fails_even_when_macro_is_good():
    from continuous_mmwave.parrad_projection_evaluate import summarize_decision
    records, semantic = matrix()
    entry = next(r for r in records if r['group'] == 'natural-fixed_age-6' and r['held_subject'] == 'subject_02')
    entry['delta']['false_positive_rate@0.05'] = .03
    assert summarize_decision(records, semantic)['decision'] == 'no_go'
    records, semantic = matrix()
    semantic[0]['passed'] = False
    assert summarize_decision(records, semantic)['decision'] == 'no_go'


def test_incomplete_fold_or_metric_is_inconclusive_never_imputed():
    from continuous_mmwave.parrad_projection_evaluate import summarize_decision
    records, semantic = matrix()
    assert summarize_decision(records[:-1], semantic)['decision'] == 'inconclusive'
    next(r for r in records if r['group'] == 'natural-fixed_age-6')['delta']['auroc'] = None
    assert summarize_decision(records, semantic)['decision'] == 'inconclusive'


def test_undefined_supporting_metric_is_reported_without_imputation_or_blocking_primary():
    from continuous_mmwave.parrad_projection_evaluate import summarize_decision
    records, semantic = matrix()
    records[0]['delta']['auroc'] = None
    records[0]['delta']['recall@0.05'] = None
    summary = summarize_decision(records, semantic)
    assert summary['decision'] == 'go_research'
    assert summary['undefined_supporting_metrics']
    records[0]['nonconverged'] = True
    assert summarize_decision(records, semantic)['decision'] == 'inconclusive'

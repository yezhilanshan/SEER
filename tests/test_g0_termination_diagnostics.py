import importlib.util
from pathlib import Path


def module():
    path=Path(__file__).resolve().parents[1]/'.audit-staging/g0_termination_diagnostics_v1.py'
    spec=importlib.util.spec_from_file_location('termination_diagnostics',path)
    loaded=importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
    return loaded


def row(seq,text,*,ids=None,ongoing=None):
    return dict(uid='example',reference=dict(sequence=seq,ongoing=ongoing or [False]*len(seq)),
                text=text,token_ids=ids or [10,151645],truncated=False)


def test_eos_before_complete_list_is_distinct_from_length_exhaustion():
    m=module()
    a=m.diagnose(row([0,1,0],'The person stood up, then sat down.'))
    assert a['termination_class']=='A_early_eos_exact_action_prefix'
    assert a['gt_instance_count']==3 and a['output_instance_count']==2
    assert a['first_action_error']['position']==3
    b=m.diagnose(row([0,1],'The person stood up, then sat down, then stood up, then sat down, then stood up, then sat down',ids=[10]*512))
    assert b['termination_class']=='B_length_after_exact_action_completion'
    assert b['output_instance_count']==6 and b['output_repeated_instances']==4
    assert b['periodic_suffix']['period']==2 and b['periodic_suffix']['repetitions']==3
    assert b['first_action_error']['position']==3 and not b['eos_generated']


def test_content_errors_before_overrun_do_not_count_as_exact_completion():
    m=module()
    r=m.diagnose(row([0,1],'The person sat down, then stood up, then sat down, then stood up, then sat down',ids=[10]*512))
    assert r['termination_class']=='length_after_ordered_coverage_with_prior_errors'
    assert not r['all_gt_actions_as_exact_prefix']
    assert r['ordered_coverage_completed_at_instance']==3
    assert r['first_action_error']['position']==1


def test_partial_last_phrase_is_not_counted_as_an_instance():
    m=module()
    r=m.diagnose(row([0,1],'The person stood up, then sat',ids=[10]*512))
    assert r['output_instance_count']==1 and r['parse_status']=='partial_final_action'
    assert r['termination_class']=='length_before_ordered_gt_coverage'
    assert r['first_action_error']['position']==2
    assert r['truncation_tail'].endswith('then sat')


def test_legitimate_gt_repetition_and_ongoing_error_are_separate():
    m=module()
    text='The person stood up, then stood up, then sat down.'
    r=m.diagnose(row([0,0,1],text,ongoing=[False,False,True]))
    assert r['termination_class']=='eos_exact_actions_with_state_error'
    assert r['gt_repeated_instances']==1 and r['excess_class_instances']==0
    assert r['first_action_error'] is None and r['first_state_error']['position']==3


def test_unstructured_negation_does_not_become_a_successful_action_prefix():
    m=module()
    r=m.diagnose(row([0,1],'The person did not stand up or sat down.'))
    assert r['termination_class']=='content_unresolved'
    assert r['output_instance_count'] is None
    assert r['literal_phrase_count']==1


def test_free_output_connectors_are_allowed_only_for_diagnostic_counting():
    m=module()
    r=m.diagnose(row([0,1,0],'The person stood up, sat down, and stood up.'))
    assert r['termination_class']=='eos_exact_actions_and_states'
    assert r['output_instance_count']==3


def test_truncated_connector_keeps_all_completed_instances():
    m=module()
    r=m.diagnose(row([0,0],'The person stood up, then stood up, then',ids=[10]*512))
    assert r['parse_reliable'] and r['parse_status']=='partial_connector'
    assert r['output_instance_count']==2
    assert r['termination_class']=='B_length_after_exact_action_completion'

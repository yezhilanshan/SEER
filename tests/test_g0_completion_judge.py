import importlib.util
import json
from pathlib import Path
from collections import Counter


def module():
    path=Path(__file__).resolve().parents[1]/'.audit-staging/g0_completion_judge_v1.py'
    spec=importlib.util.spec_from_file_location('completion_judge',path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded


def prior_fixture():
    path=Path(__file__).resolve().parents[1]/'.audit-staging/g0_stop_continue_pairs_v1.py'
    spec=importlib.util.spec_from_file_location('prior_pair_fixture',path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    return loaded.make_pairs()


def test_confirmation_prefixes_are_disjoint_and_wrong_order_controls_preserve_counts():
    m=module();old=prior_fixture();new=m.confirmation_pairs(old)
    assert len(new)==144 and len({tuple(p['prefix_sequence']) for p in new})==144
    assert not {tuple(p['prefix_sequence']) for p in old}&{tuple(p['prefix_sequence']) for p in new}
    conditions=m.conditions(new,'confirmation',{})
    assert len(conditions)==432
    for c in conditions:
        if c['condition']=='wrong_order':
            assert not c['expected_complete'] and c['evidence_sequence']!=c['description_sequence']
            assert Counter(c['evidence_sequence'])==Counter(c['description_sequence'])


def test_swapping_labels_does_not_change_evidence_or_description_and_does_not_leak_answer():
    m=module();c=dict(evidence_sequence=[0,0,1],description_sequence=[0,0])
    a=m.query(c,'A');b=m.query(c,'B')
    assert a.split('Which status applies?')[0]==b.split('Which status applies?')[0]
    assert 'A: COMPLETE\nB: NOT_COMPLETE' in a
    assert 'A: NOT_COMPLETE\nB: COMPLETE' in b
    assert 'expected_complete' not in a and 'remaining_instances' not in a


def test_one_correct_mapping_and_high_average_do_not_count_as_robust_success():
    m=module()
    stats=m.judge_metrics(True,4.,-1.)
    assert not stats['robust_correct'] and stats['symmetrized_correct'] and stats['mapping_disagreement']
    assert m.judge_metrics(False,-2.,-3.)['robust_correct']
    assert not m.judge_metrics(True,0.,2.)['robust_correct']


def test_prior_groups_cover_all_pairs_without_selecting_only_failures():
    m=module();pairs=prior_fixture()
    groups={p['uid']:i<101 for i,p in enumerate(pairs)}
    cases=m.conditions(pairs,'prior',groups)
    assert len(cases)==408
    assert len({c['pair_uid'] for c in cases if c['prior_pair_success']})==101
    assert len({c['pair_uid'] for c in cases if not c['prior_pair_success']})==43

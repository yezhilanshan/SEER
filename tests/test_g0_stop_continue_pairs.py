import importlib.util
from pathlib import Path
import torch


def module():
    path=Path(__file__).resolve().parents[1]/'.audit-staging/g0_stop_continue_pairs_v1.py'
    spec=importlib.util.spec_from_file_location('stop_continue_pairs',path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded


def test_panel_has_unique_prefixes_and_same_prefix_for_both_evidence_conditions():
    m=module();pairs=m.make_pairs()
    assert len(pairs)==144 and len({tuple(p['prefix_sequence']) for p in pairs})==144
    assert {p['family'] for p in pairs}=={'distinct','adjacent_repeat','alternating_repeat'}
    for p in pairs:
        seq=p['prefix_sequence'];assert p['complete_evidence']==seq
        assert p['remaining_evidence'][:len(seq)]==seq
        assert len(p['remaining_evidence'])-len(seq) in (1,2)
        assert p['prefix_text'].endswith(m.PHRASES[seq[-1]]) and not p['prefix_text'].endswith('.')
    repeats=[p for p in pairs if p['family']=='adjacent_repeat' and len(p['prefix_sequence'])==2]
    assert any(len(set(p['remaining_evidence']))==1 for p in repeats)


def test_directional_change_is_not_mistaken_for_correct_stop_continue_discrimination():
    m=module()
    r=m.pair_metrics({'margin':2.,'grammar_decision':'stop'},{'margin':1.,'grammar_decision':'stop'})
    assert r['directional_response'] and not r['both_binary_correct'] and not r['both_grammar_correct']
    r=m.pair_metrics({'margin':1.,'grammar_decision':'stop'},{'margin':-1.,'grammar_decision':'continue'})
    assert r['both_binary_correct'] and r['both_grammar_correct']
    r=m.pair_metrics({'margin':1.,'grammar_decision':'other_allowed'},{'margin':-1.,'grammar_decision':'continue'})
    assert r['both_binary_correct'] and not r['both_grammar_correct']


def test_repetition_penalty_changes_only_seen_tokens_and_biases_seen_comma_towards_stop():
    m=module();logits=torch.tensor([2.,3.,-3.,5.])
    changed=m.apply_repetition(logits,[1,1,2],1.1)
    torch.testing.assert_close(changed,torch.tensor([2.,3/1.1,-3*1.1,5.]))
    assert (changed[0]-changed[1])>(logits[0]-logits[1])
    torch.testing.assert_close(m.apply_repetition(logits,[3],1.1)[:3],logits[:3])
    torch.testing.assert_close(logits,torch.tensor([2.,3.,-3.,5.]))

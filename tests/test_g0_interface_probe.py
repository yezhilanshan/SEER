import importlib.util
from pathlib import Path
from collections import Counter
import torch


def module():
    path=Path(__file__).resolve().parents[1]/'.audit-staging/g0_interface_probe_v1.py'
    spec=importlib.util.spec_from_file_location('interface_probe',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Tokenizer:
    eos_token_id=99
    # A length-sensitive stand-in; actual model tokenizer is used for the experiment.
    def encode(self,text,add_special_tokens=False):return list(range(len(text.split())))


def test_candidates_use_equal_length_when_available_and_preserve_semantic_controls():
    m=module();ref=dict(sequence=[0,1,0,2],ongoing=[False,False,False,True])
    rows=m.candidates(dict(uid='case',reference=ref),Tokenizer())
    assert {r['kind'] for r in rows}=={'order','category','repeat_composition'}
    for r in rows:
        assert len(r['wrong']['sequence'])==len(ref['sequence'])
        assert r['wrong']['ongoing']==ref['ongoing'] and r['wrong']!=ref
        assert r['equal_token_length']==(r['correct_tokens']==r['wrong_tokens'])
        assert not r['equal_candidate_count'] or r['equal_token_length']
        if r['kind']=='order':
            assert Counter(r['wrong']['sequence'])==Counter(ref['sequence'])
            assert r['wrong']['sequence'][-1]==ref['sequence'][-1]
        if r['kind']=='repeat_composition':
            assert set(r['wrong']['sequence'])==set(ref['sequence'])
            assert Counter(r['wrong']['sequence'])!=Counter(ref['sequence'])
    assert m.candidates(dict(uid='empty',reference=dict(sequence=[],ongoing=[])),Tokenizer())==[]


def test_block_shuffle_preserves_content_within_units_and_does_not_move_partial_unit():
    m=module();cells=torch.arange(13*4*64).view(13,4,64).float();saved=cells.clone()
    result,permutation=m.shuffle_blocks(cells,'fixed')
    result2,p2=m.shuffle_blocks(cells,'fixed')
    assert permutation==p2 and torch.equal(result,result2) and torch.equal(cells,saved)
    assert sorted(permutation)==list(range(7)) and permutation[-1]==6 and permutation!=list(range(7))
    for target,source in enumerate(permutation[:-1]):assert torch.equal(result[2*target:2*target+2],cells[2*source:2*source+2])
    assert torch.equal(result[-1],cells[-1])


def test_panel_overlap_rejection_and_fit_dev_recording_separation():
    m=module()
    a=dict(uid='a',file='r',start=0,stop=60)
    assert m.overlap(a,dict(uid='b',file='r',start=30,stop=90))
    assert not m.overlap(a,dict(uid='c',file='r',start=60,stop=120))
    assert not m.overlap(a,dict(uid='d',file='s',start=0,stop=60))
    try:m.validate_split([dict(a,panel='old_fit'),dict(uid='b',file='r',start=120,stop=180,panel='dev')])
    except ValueError as e:assert 'recording' in str(e)
    else:raise AssertionError('fit/dev recording leakage accepted')


def test_record_macro_does_not_count_many_negatives_as_independent_records():
    m=module()
    rows=[dict(file='a',value=1.)]*9+[dict(file='b',value=-1.)]
    result=m.cluster_interval(rows,'value',seed=42,repeats=200)
    assert result['records']==2 and result['record_macro_mean']==0
    assert result['ci95'][0]<=0<=result['ci95'][1]
    assert result==m.cluster_interval(rows,'value',seed=42,repeats=200)

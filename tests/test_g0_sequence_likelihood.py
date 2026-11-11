import importlib.util
from pathlib import Path
from collections import Counter
import torch
import torch.nn.functional as F
from test_latent_memory_model import make_model


def module():
    path=Path(__file__).resolve().parents[1]/'.audit-staging/g0_sequence_likelihood_v1.py'
    spec=importlib.util.spec_from_file_location('sequence_likelihood',path)
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded


def test_mutations_preserve_declared_invariants_and_never_duplicate_correct_answer():
    m=module();case=dict(uid='fixed',reference=dict(sequence=[0,1,0,2],ongoing=[False,False,False,True]))
    negatives=m.negatives(case)
    assert {r['kind'] for r in negatives}=={'order','category','repeat_composition','state','delete','insert'}
    for r in negatives:assert r['reference']!=case['reference']
    by={r['kind']:r['reference'] for r in negatives};ref=case['reference']
    assert Counter(by['order']['sequence'])==Counter(ref['sequence'])
    assert by['order']['ongoing']==ref['ongoing']
    assert set(by['repeat_composition']['sequence'])==set(ref['sequence'])
    assert len(by['repeat_composition']['sequence'])==len(ref['sequence'])
    assert by['state']['sequence']==ref['sequence'] and by['state']['ongoing'][-1] is False
    assert len(by['delete']['sequence'])==3 and len(by['insert']['sequence'])==5
    assert by['insert']['ongoing'][-1] is True


def test_ineligible_repeated_or_order_examples_are_not_silently_relabelled():
    m=module()
    for seq,flags in [([],[]),([0],[True]),([0,0],[False,False])]:
        kinds={r['kind'] for r in m.negatives(dict(uid=str(seq),reference=dict(sequence=seq,ongoing=flags)))}
        assert 'order' not in kinds and 'repeat_composition' not in kinds
    assert len(m.negatives(dict(uid='empty',reference=dict(sequence=[],ongoing=[]))))==1


def test_teacher_forced_scores_match_native_shifted_loss_and_separate_eos():
    m=module();model=make_model();prompt=torch.tensor([[1,4,6]]);answer=[7,8,9,2]
    result=m.score_answer(model,torch.empty(1,0,32),prompt,answer)
    ids=torch.tensor([[1,4,6]+answer]);labels=ids.clone();labels[:,:3]=-100
    with torch.no_grad():native=model.backend.lm(input_ids=ids,labels=labels,return_dict=True).loss
    assert abs(result['mean_with_eos']+float(native))<1e-5
    assert result['tokens_with_eos']==4 and result['text_tokens']==3
    assert abs(result['total_with_eos']-sum(result['token_logprobs']))<1e-8
    assert abs(result['total_with_eos']-result['total_text']-result['eos_logprob'])<1e-8
    assert all(p.grad is None for p in model.backend.parameters())


def test_bidirectional_evidence_switch_is_distinct_from_preferring_one_candidate_everywhere():
    m=module()
    def score(x):return {'mean_with_eos':x}
    result=m.compare(score(-1),score(-2),score(-1),score(-2))
    assert result['mean_with_eos']['correct_wins']
    assert not result['mean_with_eos']['both_contexts_correct']
    assert result['mean_with_eos']['evidence_delta']==0


def test_pure_order_mutation_requires_two_finished_distinct_actions():
    m=module();case=dict(uid='ongoing',reference=dict(sequence=[0,1],ongoing=[False,True]))
    assert 'order' not in {r['kind'] for r in m.negatives(case)}


def test_radar_prefix_scores_match_native_inputs_embeds_loss():
    m=module();model=make_model();prompt=torch.tensor([[1,4,6]]);answer=[7,8,9,2]
    prefix=torch.randn(1,5,32)
    result=m.score_answer(model,prefix,prompt,answer)
    hidden=torch.cat((prefix,model.backend.embed(prompt),model.backend.embed(torch.tensor([answer]))),1)
    labels=torch.full(hidden.shape[:2],-100,dtype=torch.long);labels[:,-len(answer):]=torch.tensor([answer])
    with torch.no_grad():native=model.backend.lm(inputs_embeds=hidden,labels=labels,return_dict=True).loss
    assert abs(result['mean_with_eos']+float(native))<1e-5

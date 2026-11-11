"""Catches mismatched resumes, label leakage, and misleading sequence metrics."""
import json
import torch
import pytest
from test_latent_memory_model import make_model,make_memory


def test_resume_matches_uninterrupted_training_and_rejects_changed_source(tmp_path):
    from continuous_mmwave.latent_memory.training import train_steps,load_checkpoint
    sample=dict(memory=make_memory(),prompt_ids=torch.tensor([[1,3]]),answer_ids=torch.tensor([[8,2]]),uid='record:32')
    source={'manifest':'fixed-test-source'}
    a=make_model(); train_steps(a,[sample],tmp_path/'first',steps=1,source=source,seed=42)
    b=make_model(); train_steps(b,[sample],tmp_path/'resumed',steps=2,source=source,seed=42,
                               resume=tmp_path/'first'/'step-000001.pt')
    c=make_model(); train_steps(c,[sample],tmp_path/'continuous',steps=2,source=source,seed=42)
    for n,p in b.learned_state().items(): torch.testing.assert_close(p,c.learned_state()[n],rtol=0,atol=0)
    payload=torch.load(tmp_path/'resumed'/'step-000002.pt',weights_only=False)
    assert payload['step']==2 and all(int(v['step'])==2 for v in payload['optimizer']['state'].values())
    with pytest.raises(ValueError,match='source'):
        load_checkpoint(tmp_path/'first'/'step-000001.pt',make_model(),{'manifest':'changed'})
    with pytest.raises(FileExistsError): train_steps(a,[sample],tmp_path/'first',steps=2,source=source)


def test_config_rejects_invalid_budgets_and_implicit_training():
    from continuous_mmwave.latent_memory.config import LatentMemoryConfig
    assert LatentMemoryConfig().query_frames==30
    with pytest.raises(ValueError): LatentMemoryConfig(query_frames=0)
    with pytest.raises(ValueError): LatentMemoryConfig(capacities=(1,2,3))
    with pytest.raises(ValueError): LatentMemoryConfig(time_encoding='anchor')
    with pytest.raises(ValueError): LatentMemoryConfig.from_dict({'unknown_setting':1})


def test_sequence_metrics_keep_adjacent_repeats_and_report_parse_failures():
    from continuous_mmwave.latent_memory.evaluation import sequence_metrics,parse_description
    assert parse_description('The person stood up, then stood up, then sat down.')==[0,0,1]
    metrics=sequence_metrics([0,0,1],[0,1])
    assert metrics['edit_distance']==1 and metrics['exact']==False
    assert metrics['instance_count_mae']==pytest.approx(1/12)
    assert parse_description('This is not a valid description.') is None


def test_ongoing_action_state_is_scored_separately_from_action_order():
    from continuous_mmwave.latent_memory.evaluation import parse_states,state_metrics
    text='The person stood up, then sat down (ongoing).'
    labels,states=parse_states(text)
    assert labels==[0,1] and states==[False,True]
    assert state_metrics([0,1],[False,True],text)['state_sequence_exact']
    wrong=state_metrics([0,1],[False,True],'The person stood up, then sat down.')
    assert not wrong['state_sequence_exact'] and wrong['ongoing_accuracy_on_exact_sequence']==.5


def test_negative_or_uncertain_prose_cannot_receive_perfect_sequence_credit():
    from continuous_mmwave.latent_memory.evaluation import parse_description
    for text in ['The person never stood up.','The person possibly stood up.','The person stood up, but did not sat down.']:
        assert parse_description(text) is None


def test_failed_resume_and_keyboard_interrupt_leave_failed_status(tmp_path):
    from continuous_mmwave.latent_memory.training import train_steps
    example=dict(memory=make_memory(),prompt_ids=torch.tensor([[1,3]]),answer_ids=torch.tensor([[8,2]]),uid='a')
    model=make_model()
    train_steps(model,[example],tmp_path/'base',steps=1,source={'v':1})
    with pytest.raises(ValueError):
        train_steps(make_model(),[example],tmp_path/'bad',steps=2,source={'v':2},resume=tmp_path/'base'/'step-000001.pt')
    assert json.loads((tmp_path/'bad'/'status.json').read_text())['status']=='failed'
    class Interrupted:
        def __len__(self): return 1
        def __getitem__(self,index): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt): train_steps(make_model(),Interrupted(),tmp_path/'interrupt',steps=1,source={'v':1})
    state=json.loads((tmp_path/'interrupt'/'status.json').read_text())
    assert state['status']=='failed' and state['successful_steps']==0


def test_prefix_examples_rebuild_real_past_and_expose_no_annotations(tmp_path):
    import numpy as np
    from continuous_mmwave.latent_memory.data import PrefixExamples,runtime_record
    from continuous_mmwave.latent_memory.config import LatentMemoryConfig
    class Tokenizer:
        eos_token_id=2
        def encode(self,text,add_special_tokens=False): return [4,5]
    p=tmp_path/'cells.npy'; np.save(p,np.ones((7,4,64),np.float32))
    row=dict(file='subject1_01',frames=7,cells_path=str(p),role='train',subject='subject1',
             instances=[dict(start=1,stop=6,label=0)])
    examples=PrefixExamples([row],Tokenizer(),torch.tensor([[1,3]]),LatentMemoryConfig(query_frames=3))
    assert len(examples)==3
    assert set(runtime_record(row))=={'file','frames','cells_path'}
    assert examples[1]['memory'].observed_frames==6
    assert examples[1]['uid']=='subject1_01:6'
    assert examples[1]['memory'].candidates(2)[0].start_frame==0

import numpy as np
import pytest


def meta():
    return {'frames':48,'instances':[{'start':4,'stop':12,'label':1},
        {'start':16,'stop':26,'label':1},{'start':26,'stop':34,'label':2}]}


def test_partial_labels_include_stops_without_inventing_unknown_negatives():
    from continuous_mmwave.harrad.reset_data import reset_targets,split_policy_records
    out=reset_targets(meta())
    assert np.flatnonzero(out['boundary']).tolist()==[2,6,8,13,17]
    assert out['valid'][4] and not out['boundary'][4]
    assert not out['valid'][7] and out['labels'][7]==-100
    assert out['labels'][12]==1 and out['labels'][13]==2
    records=[{'role':'train','subject':f'subject{s}','file':f'subject{s}_{i}'} for s in (1,2) for i in (1,2)]
    fit,cal=split_policy_records(records)
    assert fit==[0,2] and cal==[1,3]
    with pytest.raises(ValueError): split_policy_records(records[:1])


def test_episode_matching_preserves_repeated_instances_and_unknown_burden():
    from continuous_mmwave.harrad.reset_data import reset_targets
    from continuous_mmwave.harrad.reset_evaluation import score_episodes
    m=meta(); known=reset_targets(m)['known_frames']
    good=score_episodes(np.array([[4,12,1],[12,16,5],[16,26,1],[26,34,2]]),m['instances'],known)
    assert (good['tp'],good['fp'],good['fn'],good['unscored_unknown'])==(3,0,0,1)
    merged=score_episodes(np.array([[4,26,1],[26,34,2]]),m['instances'],known)
    assert (merged['tp'],merged['fp'],merged['fn'])==(1,1,2)


@pytest.mark.parametrize('frames',[40,41])
def test_oracle_cannot_be_penalized_for_same_update_or_unavailable_terminal_anchors(frames):
    from continuous_mmwave.harrad.reset_data import reset_targets
    from continuous_mmwave.harrad.boundary import score_events
    m={'frames':frames,'instances':[{'start':3,'stop':8,'label':1},
        {'start':9,'stop':16,'label':1},{'start':17,'stop':frames,'label':2}]}
    out=reset_targets(m)
    cutoffs=2*np.flatnonzero(out['boundary'])[np.flatnonzero(out['boundary'])>0]+2
    score=score_events(cutoffs,out['truth'],out['known_frames'],4)
    assert score['fn']==0 and score['fp']==0 and score['f1']==1
    assert out['merged_anchor_count']==2 and out['excluded_recording_edge_anchors']==1


def test_instance_semantics_have_one_vote_per_instance_instead_of_duration():
    from continuous_mmwave.harrad.reset_data import reset_targets
    from continuous_mmwave.harrad.reset_evaluation import evaluate_record
    m={'frames':32,'instances':[{'start':2,'stop':6,'label':1},{'start':8,'stop':30,'label':2}]}
    r={'reset':reset_targets(m),'motion':np.zeros((16,64)),'normalized':np.zeros((16,64))}
    logits=np.zeros((16,12)); logits[1:3,3]=10; logits[4:15,2]=10
    arr={'q':np.zeros(16),'logits':logits,'mean':np.zeros((16,3,64)),
         'cutoffs':np.array([4,8,10,32]),'episodes':np.array([[2,6,3],[8,30,2]])}
    result=evaluate_record(arr,r,{'persistence_mse':np.ones(3)})
    assert result['semantic']['accuracy']==pytest.approx(11/13)
    assert result['instance_semantic']['accuracy']==.5
    assert result['instance_semantic']['n']==2 and result['instance_semantic']['coverage']==1

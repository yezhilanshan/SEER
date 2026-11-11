import torch
from continuous_mmwave.v25 import training


def test_retrieval_uses_fixed_queries_with_graded_ranking_and_missing_class():
    keys=torch.tensor([[1.,0.],[0.,1.],[.8,-.2]])
    queries=torch.eye(2)
    metric=training.fixed_query_retrieval(keys,torch.tensor([0,1,1]),queries)
    assert metric['macro_r_at_1']==1
    assert abs(metric['macro_map']-((1+(1+2/3)/2)/2))<1e-6
    missing=training.fixed_query_retrieval(keys,torch.tensor([0,0,0]),queries)
    assert missing['macro_map'] is None
    assert missing['missing_labels']==[1]


def test_top5_query_hit_and_best_positive_negative_margin_are_explicit():
    queries=torch.eye(2)
    keys=torch.tensor([[1.,0.],[0.,1.],[.8,-.2]])
    metric=training.fixed_query_retrieval(keys,torch.tensor([0,1,1]),queries)
    assert metric['macro_r_at_5']==1.
    assert metric['r_at_5_definition']=='at-least-one-relevant-event-in-top5'
    assert metric['per_label'][0]['margin']>0
    missing=training.fixed_query_retrieval(keys,torch.tensor([0,0,0]),queries)
    assert missing['macro_r_at_5'] is None
    assert missing['per_label'][1]['margin'] is None

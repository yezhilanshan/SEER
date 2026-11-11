import numpy as np
import torch


def test_tiny_training_saves_and_replays_prefix_and_final(tmp_path):
    from test_harrad_event import tiny_model
    from continuous_mmwave.harrad.event_training import fit_semantic
    rows=[]
    for role in ('train','validation'):
        for label in range(12):
            p=tmp_path/f'{role}{label}.npy'; np.save(p,np.random.default_rng(label).normal(size=(8,4,64)).astype('float32'))
            rows.append({'role':role,'subject':'s1' if role=='train' else 's2','uid':f'{role}{label}',
                         'start':0,'stop':8,'label':label,'instance_id':label,'cells_path':str(p)})
    cfg={'seed':42,'epochs':1,'batch_size':4,'evaluation_batch_size':4,'learning_rate':1e-4,
         'weight_decay':.01,'amp':False,'max_batch_attempts':3}
    result=fit_semantic(tiny_model(),rows,cfg,tmp_path/'run','cpu')
    assert result['epochs_completed']==1 and result['successful_steps']==3 and result['exposures']==12
    assert result['replay_max_abs']==0 and result['frozen_base_unchanged']
    assert (tmp_path/'run'/'independent-metrics.json').exists()

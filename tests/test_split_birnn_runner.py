from __future__ import annotations
import importlib
import json
import numpy as np
import torch


def test_training_validation_checkpoint_and_resume_run_end_to_end(tmp_path):
    runner=importlib.import_module('continuous_mmwave.split_birnn.runner')
    train,val=[],[]
    for i in range(4):
        p=tmp_path/f'r{i}.npz';rng=np.random.default_rng(i)
        np.savez(p,md=rng.normal(size=(24,128)).astype('float32'),labels=np.arange(24,dtype=np.int64)%3,timestamps=np.arange(24)*.09)
        row={'path':str(p),'recording_id':f'r{i}','subject_id':f's{i}','sensor_id':'x',
             'alignment':{'command_frame_indices':[0,8,16]}}
        (train if i<2 else val).append(row)
    config={'max_optimizer_steps':4,'validation_interval_steps':2,'batch_size':2,'min_length':5,'max_length':10,
            'minimum_steps_before_early_stopping':100,'early_stopping_patience_checks':15,'window':5,'hop':2,'seed':42}
    full=runner.run_training('split',train,val,tmp_path/'full',config,device='cpu',capture_step=2)
    resumed=runner.run_training('split',train,val,tmp_path/'resumed',config,device='cpu',resume=tmp_path/'full'/'step-2.pt')
    assert full['steps']==4 and resumed['steps']==4
    a=torch.load(tmp_path/'full'/'last.pt',weights_only=False)
    b=torch.load(tmp_path/'resumed'/'last.pt',weights_only=False)
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    assert a['sampler']==b['sampler']
    assert a['scheduler']==b['scheduler']
    assert (tmp_path/'full'/'best.pt').is_file()
    assert full['checkpoint_replay']['parameters_exact'] and full['checkpoint_replay']['predictions_exact']
    assert len((tmp_path/'full'/'train.jsonl').read_text().splitlines())==4
    validations=[json.loads(s) for s in (tmp_path/'full'/'validation.jsonl').read_text().splitlines()]
    assert [v['step'] for v in validations]==[2,4]

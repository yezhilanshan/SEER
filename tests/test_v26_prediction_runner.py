import importlib
import importlib.util
import pytest
import torch


def api():
    assert importlib.util.find_spec('continuous_mmwave.v26.prediction_runner') is not None, 'pilot runner missing'
    return importlib.import_module('continuous_mmwave.v26.prediction_runner')


def records():
    generator=torch.Generator().manual_seed(3)
    return [dict(uid=f's{i}-a-r',role='train',metadata=dict(subject_id=f's{i}',session_id='a',sensor_id='r'),
                 radar=torch.randn(7,4,8,generator=generator),motion=torch.randn(7,8,generator=generator),
                 selected=torch.tensor([1,2,3,4,5]),future=torch.tensor([2,3,4,5,6]),
                 prefix_start=torch.zeros(5,dtype=torch.long),event=torch.randn(5,8,16,generator=generator),
                 valid=torch.ones(7,dtype=torch.bool)) for i in range(2)]


def test_pilot_sampling_replay_and_subject_metrics():
    module=api(); rows=records(); stats=dict(mean=torch.zeros(8),std=torch.ones(8))
    samples=module.sample_schedule(rows,3,2,42)
    assert samples == module.sample_schedule(rows,3,2,42)
    batch=module.make_batch(rows,stats,samples[0],'cpu')
    assert batch['target'].shape == (2,8) and not batch['target'].requires_grad
    metrics=module.evaluate_predictions([dict(uid='a',subject_id='s1',nll=2.,mse=2.,variance=1.,samples=100),
                                         dict(uid='b',subject_id='s2',nll=10.,mse=10.,variance=1.,samples=1)],{})
    assert metrics['subject_macro']['mse'] == 6.


def test_pilot_exact_optimizer_steps_and_empty_batch_guard(tmp_path):
    module=api(); rows=records(); stats=dict(mean=torch.zeros(8),std=torch.ones(8))
    from continuous_mmwave.v26.predictors import make_predictor
    model=make_predictor('event_free',dict(motion_dim=8,radar_dim=8,event_dim=16,hidden=8,head_width=12))
    config=dict(steps=3,batch_size=2,seed=42,lr=.001,weight_decay=.01,clip_norm=1.)
    result=module.train_one(model,rows,stats,config,tmp_path/'new','cpu')
    assert result['actual_optimizer_steps'] == 3
    with pytest.raises(FileExistsError): module.train_one(model,rows,stats,config,tmp_path/'new','cpu')
    with pytest.raises(ValueError,match='sample'):
        module.sample_schedule([dict(rows[0],selected=torch.tensor([],dtype=torch.long))],3,2,42)


def test_cache_lock_rejects_normalization_tampering(tmp_path):
    module=api()
    from continuous_mmwave.v26.source import file_sha,write_json
    stats=tmp_path/'normalization.pt'; stats.write_bytes(b'original statistics')
    write_json(tmp_path/'cache-lock.json',{'normalization.pt':file_sha(stats)})
    assert hasattr(module,'check_cache_lock'), 'cache lock enforcement missing'
    module.check_cache_lock(tmp_path)
    stats.write_bytes(b'different statistics')
    with pytest.raises(ValueError,match='SHA'): module.check_cache_lock(tmp_path)


def test_final_checkpoint_replacement_and_incomplete_reports_are_rejected(tmp_path):
    module=api()
    from continuous_mmwave.v26.source import file_sha,write_json
    assert hasattr(module,'require_final_checkpoint'), 'final checkpoint identity guard missing'
    final=tmp_path/'train/event/final.pt'; final.parent.mkdir(parents=True)
    final.write_bytes(b'original final')
    write_json(tmp_path/'train/completed.json',{'event':{'checkpoint_sha256':file_sha(final)}})
    final.write_bytes(b'replacement')
    with pytest.raises(ValueError): module.require_final_checkpoint(tmp_path,'event')
    rows=records()
    assert hasattr(module,'validate_evaluation_report'), 'evaluation completeness guard missing'
    with pytest.raises(ValueError,match='arms'): module.validate_evaluation_report({},rows)
    report={kind:{'recordings':[dict(uid=r['uid'],subject_id=r['metadata']['subject_id'],samples=5)
                               for r in rows]} for kind in ('persistence','event_free','gru','tcn','event')}
    module.validate_evaluation_report(report,rows)
    report['event']['recordings'][0]['uid']='wrong-recording'
    with pytest.raises(ValueError,match='identity'): module.validate_evaluation_report(report,rows)


def test_replay_indices_must_match_cached_future_pairs():
    module=api(); row=records()[0]
    assert hasattr(module,'validate_saved_indices'), 'saved future-index guard missing'
    with pytest.raises(ValueError,match='indices'):
        module.validate_saved_indices({'selected':row['selected'].numpy(),'future':row['future'].numpy()+1},row)


def test_pilot_selects_first_complete_session_by_metadata(tmp_path):
    import json
    module=api()
    rows=[dict(subject_id='s9',session_id='set000',sensor_id='TI_xWR68xx',path='one')]
    rows += [dict(subject_id='s9',session_id='set001',sensor_id=sensor,path=sensor)
             for sensor in ('TI_xWR14xx','TI_xWR68xx')]
    path=tmp_path/'train.jsonl'; path.write_text('\n'.join(json.dumps(r) for r in rows))
    selected=module.select_pilot_rows(path)
    assert len(selected)==2 and all(r['session_id']=='set001' for r in selected)

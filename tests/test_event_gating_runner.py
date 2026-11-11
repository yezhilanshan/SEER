import json
from pathlib import Path
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def real_adapter_fixture(root):
    from continuous_mmwave.harrad.model import SegmentClassifier
    from continuous_mmwave.event_gating.data import sha256
    root.mkdir()
    torch.manual_seed(2)
    frontend = SegmentClassifier('rd').eval()
    ckpt = root/'radar.pt'
    torch.save(dict(arm='rd', model=frontend.state_dict(),normalizer={'rd_mean':np.zeros((40,64))}), ckpt)
    norms = root/'normalizers.npz'
    np.savez(norms, rd_mean=np.zeros((40,64)), rd_scale=np.ones((40,64)))
    records = []
    for subject, role in [('subject1','train'), ('subject7','validation')]:
        dest = root/subject
        dest.mkdir()
        vectors = np.random.default_rng(12).normal(size=(30,64)).astype(np.float32)
        with torch.no_grad():
            hidden = frontend.temporal(torch.from_numpy(vectors)[None]).frame_hidden[0].numpy()
        np.save(dest/'frame_features.npy', vectors)
        np.save(dest/'continuous_hidden.npy', hidden)
        records.append(dict(file=subject+'_01.hdf5',subject=subject,role=role,frames=30,
                            artifact_dir=str(dest),oracle_path='must-not-open',
                            instances=[dict(start=s,stop=s+6,label=y,instance_id=i)
                                       for i,(s,y) in enumerate([(1,0),(10,0),(20,1)])],
                            artifacts={n:dict(sha256=sha256(dest/n))
                                       for n in ['frame_features.npy','continuous_hidden.npy']}))
    manifest = root/'feature-manifest.json'
    manifest.write_text(json.dumps(records), encoding='utf-8')
    verification = root/'verification.json'
    verification.write_text(json.dumps(dict(passed=True,source_checkpoint_sha256=sha256(ckpt),
                                            source_normalizers_sha256=sha256(norms),test_signal_values_read=False)))
    return dict(locks={key:dict(path=str(p),sha256=sha256(p)) for key,p in
                       [('feature_manifest',manifest),('radar_checkpoint',ckpt),
                        ('radar_normalizers',norms),('feature_verification',verification)]},
                data=dict(sequence_stride=1,seed=7,max_events=2),
                gate=dict(hidden_dim=8,recent_tau_s=.2),
                training=dict(seed=17,cif_epochs=1,lif_distill_epochs=1,lif_epochs=1,
                              batch_size=2,lr=.001,threads=1))


def test_prepare_train_stage_transition_and_checkpoint_readback(tmp_path):
    from continuous_mmwave.event_gating.data import prepare, load_prepared, sha256
    from continuous_mmwave.event_gating.runner import train, load_model, evaluate
    settings = real_adapter_fixture(tmp_path/'source')
    source_before = {key:sha256(lock['path']) for key,lock in settings['locks'].items()}
    prepared = tmp_path/'prepared'
    payload = prepare(settings, prepared)
    assert len(payload['natural']) == 2
    assert any(row['targets']==[0,0] for row in payload['sequences'])
    load_prepared(prepared/'prepared.json', verify_natural=True)
    out = tmp_path/'cif'
    result = train(settings, prepared/'prepared.json', out, stage='cif')
    assert result['optimizer_steps'] > 0
    model, saved = load_model(out/'best.pt')
    initial = torch.load(out/'initial.pt', weights_only=True)
    assert any(not torch.equal(initial['model'][k],saved['model'][k]) for k in saved['model'])
    replay = evaluate(out/'best.pt', prepared/'prepared.json', tmp_path/'replay')
    assert replay['sequence']['sequence_exact'] == result['best_metrics']['sequence_exact']
    assert (tmp_path/'replay'/'natural-predictions.jsonl').exists()
    distill = train(settings, prepared/'prepared.json', tmp_path/'lif-a',stage='lif-distill',checkpoint=out/'best.pt')
    assert distill['optimizer_steps'] > 0
    final = train(settings, prepared/'prepared.json', tmp_path/'lif-b',stage='lif',checkpoint=tmp_path/'lif-a'/'best.pt')
    assert final['optimizer_steps'] > 0
    for key,lock in settings['locks'].items():
        assert sha256(lock['path']) == source_before[key]
    with pytest.raises(FileExistsError):
        train(settings, prepared/'prepared.json', out,stage='cif')


def test_train_rejects_changed_prepared_or_invalid_stage_provenance(tmp_path):
    from continuous_mmwave.event_gating.runner import train
    with pytest.raises(ValueError, match='checkpoint'):
        train({}, tmp_path/'does-not-exist',tmp_path/'wrong',stage='lif')


def test_cli_smoke_runs_training_and_saves_clear_synthetic_provenance(tmp_path):
    from continuous_mmwave.event_gating.runner import main
    out = tmp_path/'smoke'
    assert main(['smoke','--output-dir',str(out)]) == 0
    complete = json.loads((out/'complete.json').read_text())
    assert complete['synthetic_only'] is True and complete['real_harrad_training'] is False
    assert complete['stages'] == ['cif','lif-distill','lif','frame-baseline']
    assert (out/'lif'/'best.pt').exists()


def test_resume_keeps_earliest_best_when_later_epochs_tie(tmp_path):
    from continuous_mmwave.event_gating.data import prepare
    from continuous_mmwave.event_gating.runner import train
    settings = real_adapter_fixture(tmp_path/'source')
    settings['training']['lr'] = 1e-30  # finite gradients, parameter increments below FP32 resolution
    prepare(settings,tmp_path/'data')
    initial = train(settings,tmp_path/'data'/'prepared.json',tmp_path/'first')
    settings['training']['cif_epochs'] = 2
    continued = train(settings,tmp_path/'data'/'prepared.json',tmp_path/'second',
                      checkpoint=tmp_path/'first'/'best.pt',resume=True)
    assert continued['best_metrics'] == initial['best_metrics']
    assert continued['best_epoch'] == 1


def test_inference_rejects_empty_input_and_changed_frontend_provenance(tmp_path):
    from continuous_mmwave.event_gating.data import prepare,sha256
    from continuous_mmwave.event_gating.runner import train,infer
    settings = real_adapter_fixture(tmp_path/'source')
    prepare(settings,tmp_path/'data')
    train(settings,tmp_path/'data'/'prepared.json',tmp_path/'run')
    empty=tmp_path/'empty.npy'
    np.save(empty,np.empty((0,128),np.float32))
    with pytest.raises(ValueError,match='empty'):
        infer(tmp_path/'run'/'best.pt',empty,tmp_path/'empty-output')
    rd=tmp_path/'rd.npy'
    np.save(rd,np.zeros((2,40,64),np.float32))
    result=infer(tmp_path/'run'/'best.pt',rd,tmp_path/'matched-output',raw_rd=True,settings=settings)
    assert result['frames']==2 and result['events']>=1
    norms=tmp_path/'other-norms.npz'
    np.savez(norms,rd_mean=np.ones((40,64)),rd_scale=np.ones((40,64)))
    settings['locks']['radar_normalizers'] = dict(path=str(norms),sha256=sha256(norms))
    with pytest.raises(ValueError,match='provenance'):
        infer(tmp_path/'run'/'best.pt',rd,tmp_path/'wrong-output',raw_rd=True,settings=settings)


def test_train_rejects_config_that_misreports_prepared_frontend(tmp_path):
    from continuous_mmwave.event_gating.data import prepare,sha256
    from continuous_mmwave.event_gating.runner import train
    settings=real_adapter_fixture(tmp_path/'source')
    prepare(settings,tmp_path/'data')
    other=tmp_path/'other-normalizers.npz'
    np.savez(other,rd_mean=np.ones((40,64)),rd_scale=np.ones((40,64)))
    settings['locks']['radar_normalizers']=dict(path=str(other),sha256=sha256(other))
    with pytest.raises(ValueError,match='provenance'):
        train(settings,tmp_path/'data'/'prepared.json',tmp_path/'wrong-source')


def test_resume_compares_current_checkpoint_to_older_best_snapshot():
    from continuous_mmwave.event_gating.runner import select_best_snapshot
    old=dict(epoch=1,metrics=dict(sequence_exact=.2,count_mae=.4),decoder_minimum_frames=1,
             model={'weight':torch.tensor(1.)})
    current=dict(epoch=2,metrics=dict(sequence_exact=.5,count_mae=.1),decoder_minimum_frames=1,
                 model={'weight':torch.tensor(2.)},best_snapshot=old)
    chosen=select_best_snapshot(current)
    assert chosen['epoch']==2 and chosen['model']['weight'].item()==2
    current['metrics']=old['metrics']
    assert select_best_snapshot(current)['epoch']==1


def test_feature_inference_requires_verified_continuous_input_instead_of_oracle(tmp_path):
    from continuous_mmwave.event_gating.data import prepare
    from continuous_mmwave.event_gating.runner import train,infer
    settings=real_adapter_fixture(tmp_path/'source')
    payload=prepare(settings,tmp_path/'data')
    train(settings,tmp_path/'data'/'prepared.json',tmp_path/'run')
    hidden=payload['natural'][1]['path']
    with pytest.raises(ValueError,match='provenance'):
        infer(tmp_path/'run'/'best.pt',hidden,tmp_path/'no-proof')
    result=infer(tmp_path/'run'/'best.pt',hidden,tmp_path/'verified',prepared=tmp_path/'data'/'prepared.json')
    assert result['input_kind']=='continuous_hidden' and result['frames']==30
    oracle=tmp_path/'oracle_hidden.npy'
    np.save(oracle,np.zeros((30,128),np.float32))
    with pytest.raises(ValueError,match='provenance'):
        infer(tmp_path/'run'/'best.pt',oracle,tmp_path/'wrong-kind',prepared=tmp_path/'data'/'prepared.json')

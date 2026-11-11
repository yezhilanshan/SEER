import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from continuous_mmwave.models.event_former import EventStateGate
from continuous_mmwave.v25 import models
from continuous_mmwave.v25.checkpoint import save_bundle, load_bundle, module_digest
from continuous_mmwave.v25.controller import ControllerConfig
from continuous_mmwave.v25.evidence import EvidenceConfig
from continuous_mmwave.v25.stage3 import V25Stage3


def test_compatible_gate_preserves_fixed_teacher_and_does_not_read_age():
    torch.manual_seed(9)
    old = EventStateGate(4).float()
    with torch.no_grad():
        old.network[-1].weight.normal_(0, .4)
    snapshot = copy.deepcopy(old.state_dict())
    former = SimpleNamespace(state_gate=old)
    models.install_positive_coordinate_gate(former)
    previous, proposal = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    target, alpha = old(previous, proposal, 1.)
    new, actual = former.state_gate(previous, proposal, object())
    torch.testing.assert_close(new, target, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual, alpha, atol=1e-7, rtol=1e-5)
    assert (actual >= 0).all() and (actual <= 1).all()
    assert former.state_gate.network[0].in_features == 8
    assert former.state_gate.architecture == 'v25-age-free-positive-coordinate-v1'
    for name, value in old.state_dict().items():
        assert torch.equal(value, snapshot[name])
    with pytest.raises(ValueError):
        models.install_positive_coordinate_gate(former)


def test_bundle_cannot_load_positive_weights_into_signed_gate(tmp_path):
    from test_v25_pipeline import Frontend
    frontend = Frontend()
    frontend.event_former.state_gate = models.PositiveCoordinateGate(16)
    stage = V25Stage3(radar_dim=8, event_dim=16, semantic_dim=8)
    path = tmp_path/'positive.pt'
    sha = save_bundle(path, frontend, stage, ControllerConfig(), EvidenceConfig(),
                      provenance={'frontend_config_sha256':'a'*64, 'data_sha256':'b'*64})
    saved = torch.load(path, weights_only=True)
    assert saved['frontend_gate_architecture'] == frontend.event_former.state_gate.architecture
    assert saved['status'] == 'development-untrained' and not saved['formal_go']
    with pytest.raises(ValueError, match='development'):
        load_bundle(path, frontend, stage, expected_sha256=sha)
    with pytest.raises(ValueError, match='architecture'):
        load_bundle(path, Frontend(), stage, expected_sha256=sha, allow_development=True)
    load_bundle(path, frontend, stage, expected_sha256=sha, allow_development=True)


def events_fixture():
    events=[]
    for subject in ('subject_02','subject_03','subject_04','subject_08','subject_13'):
        for sensor in ('TI_xWR14xx','TI_xWR68xx'):
            for label in range(2):
                events.append(dict(uid=f'{subject}/{sensor}/{label}',role='fit',subject_id=subject,
                    sensor_id=sensor,label=label,stream_tokens=torch.ones(2,4,requires_grad=True),
                    commit_tokens=torch.ones(2,4,requires_grad=True)*2,
                    target=torch.full((4,),.5),target_mask=torch.ones(4,dtype=torch.bool)))
    return events


def test_paired_sampling_covers_both_paths_in_every_subject_sensor_cell_and_stops_gradient():
    from continuous_mmwave.v25.compatible_evidence import PairedEvidenceBatches
    batches=PairedEvidenceBatches(events_fixture(),seed=42,device='cpu')
    stage=V25Stage3(radar_dim=8,event_dim=4,semantic_dim=8)
    stage.set_phase('evidence')
    for _ in range(5):
        stage.zero_grad()
        loss=batches(stage)
        loss.backward()
    coverage=batches.coverage()
    assert len(coverage)==20
    assert all(row['draws']==4 for row in coverage)
    assert len(batches.draws)==40
    assert all(row['stream_tokens'].grad is None for row in batches.events)
    assert any(p.grad is not None for p in stage.heads.evidence.parameters())


def test_mae_groups_keep_missing_attributes_and_even_fit_median():
    from continuous_mmwave.v25.compatible_evidence import score_evidence_rows
    fit=[dict(sensor_id='s',target=[0.,0.,0.,0.],mask=[True,False,False,False]),
         dict(sensor_id='s',target=[1.,0.,0.,0.],mask=[True,False,False,False])]
    rows=[dict(uid='e',subject_id='p',sensor_id='s',path=p,prediction=[.3,0.,0.,0.],
               target=[.25,0.,0.,0.],mask=[True,False,False,False]) for p in ('stream','commit')]
    report=score_evidence_rows(fit,rows)
    assert report['passed']
    assert report['masked_mae']==pytest.approx(.05)
    assert report['fit_median_baseline_mae']==pytest.approx(.25)
    assert report['predictions'][0]['baseline'][0]==.5
    assert report['group_attribute_mae']['all'][1]['mae'] is None
    assert report['groups']['path/stream']['count']==1
    assert report['groups']['subject-sensor/p/s']['count']==2
    bad=copy.deepcopy(rows); bad[0]['mask'][1]=True
    with pytest.raises(ValueError,match='fit baseline'):
        score_evidence_rows(fit,bad)
    empty=copy.deepcopy(rows)
    for row in empty:
        row['mask']=[False]*4
    assert not score_evidence_rows(fit,empty)['passed']


def test_source_roles_block_holdouts_and_missing_visible_range():
    from continuous_mmwave.v25.compatible_evidence import validate_source_roles, validate_event_features
    bad=dict(records=[dict(role='subject_holdout')],runs=[])
    with pytest.raises(ValueError,match='fit/dev'):
        validate_source_roles(bad)
    row=events_fixture()[0]
    payload=dict(schema_version=25,architecture='v25-age-free-positive-coordinate-v1',frontend_digest='a'*64,
                 events=[row])
    with pytest.raises(ValueError,match='coverage'):
        validate_event_features(payload,'a'*64)
    for path in ('stream','commit'):
        row[path+'_visible_range']=[3,7];row[path+'_target_range']=[6,14]
    row.update(start=3,end=7)
    validate_event_features(payload,'a'*64)
    payload['architecture']='v25-age-free-tanh'
    with pytest.raises(ValueError,match='architecture'):
        validate_event_features(payload,'a'*64)


def test_cached_feature_tokens_have_explicit_full_run_coverage_and_no_graph(tmp_path):
    from test_v25_pipeline import Frontend
    from continuous_mmwave.v25.train_heads import build_event_features
    from continuous_mmwave.v25.checkpoint import file_sha
    from continuous_mmwave.v25.compatible_evidence import validate_event_features
    frontend=Frontend().eval()
    frontend.event_former.state_gate=models.PositiveCoordinateGate(16)
    source=tmp_path/'radar.pt'
    torch.save(dict(radar=torch.randn(4,4,8),radar_mask=torch.ones(4,4,dtype=torch.bool),
        frame_target=torch.arange(32,dtype=torch.float32).reshape(8,4)/32,
        frame_target_mask=torch.ones(8,4,dtype=torch.bool)),source)
    run=dict(uid='event',record_uid='record',role='fit',start=1,end=4,label=0,
             subject_id='subject_02',sensor_id='TI_xWR14xx',session_id='set000')
    index=dict(records=[dict(uid='record',role='fit',path=str(source),sha256=file_sha(source))],runs=[run])
    output=tmp_path/'features'
    events=build_event_features(frontend,index,output)
    row=events[0]
    assert row['stream_visible_range']==[1,4] and row['commit_target_range']==[2,8]
    torch.testing.assert_close(row['target'],torch.tensor([.5625,.59375,.6875,.71875]))
    assert not row['stream_tokens'].requires_grad
    validate_event_features(torch.load(output/'events.pt',weights_only=False),module_digest(frontend))


def test_paired_retrieval_rejects_wrong_diagnostic_keys_and_reports_both_paths(tmp_path):
    from continuous_mmwave.v25.compatible_evidence import paired_retrieval
    events=[dict(uid=f'e{i}',role='dev',label=i,subject_id='p',sensor_id='s',
                 stream_key=torch.eye(2)[i],commit_key=torch.eye(2)[i]) for i in range(2)]
    arrays=dict(uid=np.array(['e0','e1']),labels=np.array([0,1]),subject=np.array(['p','p']),
                sensor=np.array(['s','s']),role=np.array(['dev_regression']*2),queries=np.eye(2,dtype=np.float32))
    for key in ('stream_key','commit_key','teacher_stream_key','teacher_commit_key'):
        arrays[key]=np.eye(2,dtype=np.float32)
    np.savez(tmp_path/'ref.npz',**arrays)
    with np.load(tmp_path/'ref.npz') as reference:
        report=paired_retrieval(events,reference,tmp_path)
        assert report['passed']
        assert report['metrics']['p/s']['commit']['current']['macro_map']==1.
        events[0]['stream_key']=torch.tensor([0.,1.])
        with pytest.raises(ValueError,match='keys differ'):
            paired_retrieval(events,reference,tmp_path)


def test_event_proxy_delta_does_not_use_frame_before_visible_event(tmp_path):
    from test_v25_pipeline import Frontend
    from continuous_mmwave.v25.train_heads import build_event_features
    from continuous_mmwave.v25.checkpoint import file_sha
    frontend=Frontend().eval()
    frontend.event_former.state_gate=models.PositiveCoordinateGate(16)
    targets=torch.full((6,4),.2);targets[2,2:]=.9
    source=tmp_path/'radar.pt'
    torch.save(dict(radar=torch.randn(3,4,8),radar_mask=torch.ones(3,4,dtype=torch.bool),
                    frame_target=targets,frame_target_mask=torch.ones(6,4,dtype=torch.bool)),source)
    run=dict(uid='event',record_uid='record',role='fit',start=1,end=3,label=0,
             subject_id='subject_02',sensor_id='TI_xWR14xx',session_id='set000')
    index=dict(records=[dict(uid='record',role='fit',path=str(source),sha256=file_sha(source))],runs=[run])
    row=build_event_features(frontend,index,tmp_path/'features')[0]
    torch.testing.assert_close(row['target'],torch.full((4,),.2))


def test_development_metadata_cannot_be_relabelled_as_fit():
    from continuous_mmwave.v25.compatible_evidence import validate_source_roles
    metadata=dict(role='dev_regression',subject_id='subject_02',session_id='set002',sensor_id='TI_xWR14xx')
    record=dict(uid='r',role='fit',metadata=metadata)
    run=dict(uid='e',record_uid='r',role='fit',subject_id='subject_02',session_id='set002',sensor_id='TI_xWR14xx')
    with pytest.raises(ValueError,match='record role'):
        validate_source_roles(dict(records=[record],runs=[run]))


@pytest.mark.parametrize('difference',['signed-zero','dtype'])
def test_diagnostic_pair_requires_same_dtype_and_bytes(tmp_path,difference):
    from continuous_mmwave.v25.compatible_evidence import paired_retrieval
    events=[dict(uid=f'e{i}',role='dev',label=i,subject_id='p',sensor_id='s',
                 stream_key=torch.eye(2)[i],commit_key=torch.eye(2)[i]) for i in range(2)]
    arrays=dict(uid=np.array(['e0','e1']),labels=np.array([0,1]),subject=np.array(['p','p']),
                sensor=np.array(['s','s']),role=np.array(['dev_regression']*2),queries=np.eye(2,dtype=np.float32))
    for key in ('stream_key','commit_key','teacher_stream_key','teacher_commit_key'):
        arrays[key]=np.eye(2,dtype=np.float32)
    if difference=='signed-zero':
        events[0]['stream_key'][1]=-0.
    else:
        for row in events:
            row['stream_key']=row['stream_key'].double()
    np.savez(tmp_path/'ref.npz',**arrays)
    with np.load(tmp_path/'ref.npz') as reference:
        with pytest.raises(ValueError,match='keys differ'):
            paired_retrieval(events,reference,tmp_path)
